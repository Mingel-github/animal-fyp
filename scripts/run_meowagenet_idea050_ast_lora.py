"""Run IDEA-050 nested AST LoRA selection and matched head-only evaluation.

The select stage uses only inner-train and inner-validation calls.  It writes
one immutable selection lock for every repeat/fold before the evaluate stage is
allowed to request outer-test indices.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for local_root in (SCRIPTS_ROOT, SRC_ROOT):
    if str(local_root) not in sys.path:
        sys.path.insert(0, str(local_root))

import run_idea019_peft_placement as idea019  # noqa: E402
import run_meowagenet_ast_hpo_v1 as hpo  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT / "configs" / "protocol" / "meowagenet_idea050_ast_lora_v1.json"
)
LOCKED_MODEL_PROTOCOL_PATH = (
    REPO_ROOT / "configs" / "protocol" / "meowagenet_locked_v1.json"
)
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
PLAN_PATH = REPO_ROOT / "reports" / "19_IDEA-050_AST_LoRA_plan.md"
HPO_RUNNER_PATH = REPO_ROOT / "scripts" / "run_meowagenet_ast_hpo_v1.py"
RUNS_ROOT = REPO_ROOT / "runs"
PIPELINES = ("ast_head_only", "ast_lora")
LABELS = ("kitten", "adult", "senior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("audit", "smoke", "select", "evaluate"), required=True
    )
    parser.add_argument("--output-subdir", default="meowagenet_idea050_ast_lora_v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def candidate_by_id(protocol: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in protocol["lora"]["candidates"]:
        if candidate["candidate_id"] == candidate_id:
            return dict(candidate)
    raise KeyError(candidate_id)


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol["protocol_id"] != "meowagenet-idea050-ast-lora-v1":
        raise RuntimeError("Unexpected IDEA-050 protocol")
    if sha256(ROLES_PATH) != protocol["splits"]["roles_sha256"]:
        raise RuntimeError("IDEA-050 roles checksum mismatch")
    dependencies = protocol["dependencies"]
    expected = {
        LOCKED_MODEL_PROTOCOL_PATH: dependencies["locked_model_protocol_sha256"],
        Path(idea019.__file__).resolve(): dependencies["training_engine_sha256"],
        HPO_RUNNER_PATH: dependencies["hpo_reference_runner_sha256"],
        PLAN_PATH: dependencies["idea050_plan_sha256"],
        idea019.FEATURE_PATH: dependencies["fbank_sha256"],
        idea019.FROZEN_EMBEDDING_PATH: dependencies["frozen_embedding_sha256"],
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"IDEA-050 dependency checksum mismatch: {path}")
    candidates = protocol["lora"]["candidates"]
    identifiers = [candidate["candidate_id"] for candidate in candidates]
    if len(candidates) != 5 or len(identifiers) != len(set(identifiers)):
        raise RuntimeError("IDEA-050 requires five unique LoRA candidates")
    for candidate in candidates:
        if int(candidate["alpha"]) != 2 * int(candidate["rank"]):
            raise RuntimeError("Each IDEA-050 candidate must use alpha = 2 * rank")


def stage_manifest(
    stage: str,
    protocol: dict[str, Any],
    device: torch.device,
    selection_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "stage": stage,
        "outer_test_accessed": stage == "evaluate",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "roles_sha256": sha256(ROLES_PATH),
        "selection_path": repo_relative(selection_path) if selection_path else None,
        "selection_sha256": sha256(selection_path) if selection_path else None,
        "git_revision_at_start": git_revision(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
        },
    }


def model_protocol(
    locked_protocol: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    current = copy.deepcopy(locked_protocol)
    current["classifier_head"]["dropout"] = float(
        protocol["shared_training"]["dropout"]
    )
    return current


def training_args(protocol: dict[str, Any], maximum_epochs: int | None = None) -> SimpleNamespace:
    shared = protocol["shared_training"]
    return SimpleNamespace(
        adapter_width=32,
        adapter_learning_rate=0.001,
        head_learning_rate=float(shared["head_learning_rate"]),
        batch_size=int(shared["micro_batch_size"]),
        accumulation_steps=int(shared["gradient_accumulation_steps"]),
        max_epochs=(
            int(maximum_epochs)
            if maximum_epochs is not None
            else int(shared["maximum_epochs"])
        ),
        patience=int(shared["early_stopping_patience"]),
        gradient_clip=float(shared["gradient_clip"]),
        probe_cv_folds=3,
    )


def target_layers(candidate: dict[str, Any], total_layers: int) -> tuple[int, ...]:
    target = candidate["target_blocks"]
    if target == "last_4":
        return tuple(range(total_layers - 4, total_layers))
    if target == "all_12" and total_layers == 12:
        return tuple(range(total_layers))
    raise ValueError(f"Unsupported LoRA target blocks: {target}")


class LoRALinear(nn.Module):
    def __init__(
        self, base: nn.Linear, rank: int, alpha: int, dropout: float
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = float(alpha / rank)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.lora_B(self.lora_A(self.lora_dropout(inputs)))
        return self.base(inputs) + self.scaling * update


def make_head(
    current_protocol: dict[str, Any], train_indices: np.ndarray, store: Any
) -> nn.Module:
    embeddings = store.frozen_embeddings[train_indices]
    return idea019.ClassificationHead(
        mean=embeddings.mean(axis=0),
        scale=embeddings.std(axis=0),
        dropout=float(current_protocol["classifier_head"]["dropout"]),
    )


class OnlineASTClassifier(nn.Module):
    def __init__(self, current_protocol: dict[str, Any], head: nn.Module) -> None:
        super().__init__()
        ast_config = current_protocol["ast"]
        self.ast = idea019.ASTModel.from_pretrained(
            ast_config["checkpoint"],
            revision=ast_config["revision"],
            cache_dir=idea019.HF_CACHE,
            use_safetensors=True,
        )
        self.geometry_audit = idea019.adapt_standard_geometry(
            self.ast, current_protocol
        )
        for parameter in self.ast.parameters():
            parameter.requires_grad = False
        self.head = head

    def forward(
        self, instances: torch.Tensor, instance_to_call: torch.Tensor, call_count: int
    ) -> torch.Tensor:
        segment_embeddings = self.ast(input_values=instances).pooler_output
        call_embeddings = torch.zeros(
            (call_count, segment_embeddings.shape[-1]),
            dtype=segment_embeddings.dtype,
            device=segment_embeddings.device,
        )
        call_embeddings.index_add_(0, instance_to_call, segment_embeddings)
        counts = torch.bincount(instance_to_call, minlength=call_count).to(
            segment_embeddings.dtype
        )
        return self.head(call_embeddings / counts[:, None])


class LoRAASTClassifier(OnlineASTClassifier):
    def __init__(
        self,
        current_protocol: dict[str, Any],
        lora_protocol: dict[str, Any],
        candidate: dict[str, Any],
        head: nn.Module,
    ) -> None:
        super().__init__(current_protocol, head)
        layers = target_layers(candidate, len(self.ast.encoder.layer))
        projections = tuple(lora_protocol["target_projections"])
        for layer_index in layers:
            attention = self.ast.encoder.layer[layer_index].attention.attention
            for projection in projections:
                base = getattr(attention, projection)
                if not isinstance(base, nn.Linear):
                    raise TypeError(
                        f"Expected nn.Linear at layer {layer_index} {projection}"
                    )
                setattr(
                    attention,
                    projection,
                    LoRALinear(
                        base,
                        rank=int(candidate["rank"]),
                        alpha=int(candidate["alpha"]),
                        dropout=float(lora_protocol["dropout"]),
                    ),
                )
        self.target_layers = layers
        self.target_projections = projections
        self.candidate_id = str(candidate["candidate_id"])


def build_online_base(
    current_protocol: dict[str, Any], train_indices: np.ndarray, store: Any
) -> OnlineASTClassifier:
    return OnlineASTClassifier(
        current_protocol, make_head(current_protocol, train_indices, store)
    )


def build_lora_model(
    current_protocol: dict[str, Any],
    lora_protocol: dict[str, Any],
    candidate: dict[str, Any],
    train_indices: np.ndarray,
    store: Any,
) -> LoRAASTClassifier:
    return LoRAASTClassifier(
        current_protocol,
        lora_protocol,
        candidate,
        make_head(current_protocol, train_indices, store),
    )


def lora_parameter_records(model: nn.Module) -> list[dict[str, Any]]:
    return [
        {"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".lora_" in name
    ]


def make_lora_optimizer(
    model: LoRAASTClassifier,
    candidate: dict[str, Any],
    head_learning_rate: float,
) -> torch.optim.Optimizer:
    lora_parameters = [
        parameter
        for name, parameter in model.ast.named_parameters()
        if parameter.requires_grad and ".lora_" in name
    ]
    if not lora_parameters:
        raise RuntimeError("LoRA model has no trainable low-rank parameters")
    return torch.optim.Adamax(
        [
            {
                "params": lora_parameters,
                "lr": float(candidate["lora_learning_rate"]),
            },
            {"params": model.head.parameters(), "lr": head_learning_rate},
        ],
        eps=1.0e-7,
    )


def portable_trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    return {
        name: tensor.detach().cpu()
        for name, tensor in state.items()
        if name.startswith("head.") or ".lora_" in name
    }


def first_batch_logits(
    model: nn.Module, loader: Any, device: torch.device
) -> torch.Tensor:
    batch = idea019.move_batch(next(iter(loader)), device)
    model.eval()
    with torch.inference_mode():
        logits = model(
            batch["instances"], batch["instance_to_call"], len(batch["labels"])
        )
    return logits.float().cpu()


def fit_inner_lora(
    current_protocol: dict[str, Any],
    lora_protocol: dict[str, Any],
    candidate: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    args: SimpleNamespace,
    device: torch.device,
    seed: int,
    checkpoint_path: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    idea019.set_seed(seed)
    model = build_lora_model(
        current_protocol, lora_protocol, candidate, train_indices, store
    ).to(device)
    counts = idea019.trainable_counts(model)
    lora_records = lora_parameter_records(model)
    initial_trainable = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = make_lora_optimizer(model, candidate, args.head_learning_rate)
    weights = idea019.class_weights(store.labels[train_indices]).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    train_loader = idea019.build_loader(
        store, train_indices, "adapter", args.batch_size, True, seed
    )
    validation_loader = idea019.build_loader(
        store, validation_indices, "adapter", args.batch_size, False, seed
    )
    best_loss = float("inf")
    best_epoch = 1
    best_metrics: dict[str, Any] = {}
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.max_epochs + 1):
        train_loss = idea019.train_one_epoch(
            model,
            train_loader,
            optimizer,
            weights,
            device,
            args.accumulation_steps,
            args.gradient_clip,
            scaler,
        )
        validation_loss, validation_frame = idea019.predict_calls(
            model, validation_loader, store, device
        )
        validation_metrics, _ = idea019.evaluate_frame(validation_frame)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_balanced_accuracy": validation_metrics[
                    "balanced_accuracy"
                ],
                "validation_qwk": validation_metrics[
                    "quadratic_weighted_kappa"
                ],
            }
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_metrics = validation_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"lora {candidate['candidate_id']} epoch {epoch}: "
            f"train={train_loss:.4f}, val={validation_loss:.4f}, "
            f"val_F1={validation_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            break
    train_seconds = time.perf_counter() - started
    parameter_updates = {
        name: float(
            (parameter.detach().cpu() - initial_trainable[name]).abs().max().item()
        )
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    checkpoint_audit = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(portable_trainable_state(model), checkpoint_path)
        reference_logits = first_batch_logits(model, validation_loader, device)
        idea019.set_seed(seed)
        reloaded = build_lora_model(
            current_protocol, lora_protocol, candidate, train_indices, store
        ).to(device)
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = reloaded.state_dict()
        state.update(saved)
        reloaded.load_state_dict(state)
        reloaded_logits = first_batch_logits(reloaded, validation_loader, device)
        checkpoint_audit = {
            "path": repo_relative(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "saved_tensors": len(saved),
            "reload_max_abs_logit_difference": float(
                (reference_logits - reloaded_logits).abs().max().item()
            ),
        }
        del reloaded
    audit = {
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "best_validation_loss": best_loss,
        "best_validation_animal_metrics": best_metrics,
        "history": history,
        "train_seconds": train_seconds,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "parameters": counts,
        "trainable_lora_parameters": int(sum(row["numel"] for row in lora_records)),
        "lora_parameters": lora_records,
        "maximum_trainable_parameter_updates": parameter_updates,
        "all_trainable_parameters_updated": all(
            value > 0.0 for value in parameter_updates.values()
        ),
        "trainable_parameter_bytes_fp32": int(counts["trainable"] * 4),
        "checkpoint_audit": checkpoint_audit,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, audit


def fit_outer_lora(
    current_protocol: dict[str, Any],
    lora_protocol: dict[str, Any],
    candidate: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    epochs: int,
    args: SimpleNamespace,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    idea019.set_seed(seed)
    model = build_lora_model(
        current_protocol, lora_protocol, candidate, train_indices, store
    ).to(device)
    counts = idea019.trainable_counts(model)
    lora_records = lora_parameter_records(model)
    optimizer = make_lora_optimizer(model, candidate, args.head_learning_rate)
    weights = idea019.class_weights(store.labels[train_indices]).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    train_loader = idea019.build_loader(
        store, train_indices, "adapter", args.batch_size, True, seed
    )
    test_loader = idea019.build_loader(
        store, test_indices, "adapter", args.batch_size, False, seed
    )
    training_losses = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        train_loss = idea019.train_one_epoch(
            model,
            train_loader,
            optimizer,
            weights,
            device,
            args.accumulation_steps,
            args.gradient_clip,
            scaler,
        )
        training_losses.append(train_loss)
        print(
            f"lora {candidate['candidate_id']} outer retrain "
            f"{epoch}/{epochs}: train={train_loss:.4f}",
            flush=True,
        )
    test_loss, test_frame = idea019.predict_calls(model, test_loader, store, device)
    test_metrics, _ = idea019.evaluate_frame(test_frame)
    audit = {
        "epochs": epochs,
        "training_losses": training_losses,
        "test_loss": test_loss,
        "test_animal_metrics": test_metrics,
        "train_and_predict_seconds": time.perf_counter() - started,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "parameters": counts,
        "trainable_lora_parameters": int(sum(row["numel"] for row in lora_records)),
        "trainable_parameter_bytes_fp32": int(counts["trainable"] * 4),
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_frame, audit


def run_audit(
    run_root: Path,
    protocol: dict[str, Any],
    current_protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    output = run_root / "audit" / "audit_summary.json"
    if output.is_file():
        if not resume:
            raise FileExistsError(output)
        print(output.read_text(encoding="utf-8"), flush=True)
        return
    audit_protocol = protocol["audit"]
    repeat = int(audit_protocol["repeat"])
    outer_fold = int(audit_protocol["outer_fold"])
    seed = hpo.full_seed(int(audit_protocol["base_seed"]), repeat, outer_fold)
    indices = hpo.fold_indices(
        store, roles, repeat, outer_fold, include_test=False
    )
    candidate = candidate_by_id(protocol, audit_protocol["candidate"])
    loader = idea019.build_loader(
        store, indices["train"][:2], "adapter", 2, False, seed
    )
    idea019.set_seed(seed)
    base_model = build_online_base(current_protocol, indices["train"], store).to(device)
    base_logits = first_batch_logits(base_model, loader, device)
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    idea019.set_seed(seed)
    lora_model = build_lora_model(
        current_protocol, protocol["lora"], candidate, indices["train"], store
    ).to(device)
    lora_logits = first_batch_logits(lora_model, loader, device)
    records = lora_parameter_records(lora_model)
    trainable = [
        {"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
        for name, parameter in lora_model.named_parameters()
        if parameter.requires_grad
    ]
    forbidden = [
        row["name"]
        for row in trainable
        if not (row["name"].startswith("head.") or ".lora_" in row["name"])
    ]
    zero_b = all(
        torch.count_nonzero(parameter.detach()).item() == 0
        for name, parameter in lora_model.named_parameters()
        if ".lora_B." in name
    )
    counts = idea019.trainable_counts(lora_model)
    max_difference = float((base_logits - lora_logits).abs().max().item())
    tolerance = float(audit_protocol["initialization_logit_tolerance"])
    result = {
        "status": "passed" if max_difference <= tolerance and not forbidden else "failed",
        "stage": "implementation_audit",
        "outer_test_accessed": False,
        "candidate": candidate,
        "repeat": repeat,
        "outer_fold": outer_fold,
        "full_seed": seed,
        "target_layers_zero_based": list(lora_model.target_layers),
        "target_layers_one_based": [layer + 1 for layer in lora_model.target_layers],
        "target_projections": list(lora_model.target_projections),
        "injected_projection_count": len(records) // 2,
        "trainable_lora_parameters": int(sum(row["numel"] for row in records)),
        "trainable_total_parameters": counts["trainable"],
        "model_total_parameters": counts["total"],
        "trainable_parameters": trainable,
        "forbidden_trainable_parameters": forbidden,
        "all_lora_B_zero_initialized": zero_b,
        "initialization_equivalence": {
            "max_abs_logit_difference": max_difference,
            "tolerance": tolerance,
            "passed": max_difference <= tolerance,
        },
        "manifest": stage_manifest("audit", protocol, device),
    }
    del lora_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    write_json(output, result)
    if result["status"] != "passed":
        raise RuntimeError("IDEA-050 implementation audit failed")
    print(json.dumps(result, indent=2), flush=True)


def run_smoke(
    run_root: Path,
    protocol: dict[str, Any],
    current_protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    audit_path = run_root / "audit" / "audit_summary.json"
    if not audit_path.is_file() or read_json(audit_path)["status"] != "passed":
        raise RuntimeError("Run a passing IDEA-050 audit before smoke")
    output = run_root / "smoke" / "smoke_summary.json"
    if output.is_file():
        if not resume:
            raise FileExistsError(output)
        print(output.read_text(encoding="utf-8"), flush=True)
        return
    smoke = protocol["smoke"]
    repeat = int(smoke["repeat"])
    outer_fold = int(smoke["outer_fold"])
    seed = hpo.full_seed(int(smoke["base_seed"]), repeat, outer_fold)
    indices = hpo.fold_indices(
        store, roles, repeat, outer_fold, include_test=False
    )
    candidate = candidate_by_id(protocol, smoke["candidate"])
    checkpoint_path = run_root / "smoke" / "lora_trainable_state.pt"
    best_epoch, fit_audit = fit_inner_lora(
        current_protocol,
        protocol["lora"],
        candidate,
        store,
        indices["train"],
        indices["validation"],
        training_args(protocol, int(smoke["epochs"])),
        device,
        seed,
        checkpoint_path=checkpoint_path,
    )
    checkpoint_difference = fit_audit["checkpoint_audit"][
        "reload_max_abs_logit_difference"
    ]
    status = (
        "passed"
        if fit_audit["all_trainable_parameters_updated"]
        and checkpoint_difference <= 1.0e-6
        else "failed"
    )
    result = {
        "status": status,
        "stage": "inner_only_smoke",
        "outer_test_accessed": False,
        "candidate": candidate,
        "repeat": repeat,
        "outer_fold": outer_fold,
        "full_seed": seed,
        "inner_train_calls": int(len(indices["train"])),
        "inner_validation_calls": int(len(indices["validation"])),
        "selected_epoch": best_epoch,
        "fit_audit": fit_audit,
        "manifest": stage_manifest("smoke", protocol, device),
    }
    write_json(output, result)
    if status != "passed":
        raise RuntimeError("IDEA-050 smoke test failed")
    print(json.dumps(result, indent=2), flush=True)


def rank_fold_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranking = []
    for row in rows:
        metrics = row["audit"]["best_validation_animal_metrics"]
        ranking.append(
            {
                "candidate_id": row["candidate"]["candidate_id"],
                "parameters": row["candidate"],
                "selected_epoch": row["best_epoch"],
                "macro_f1": float(metrics["macro_f1"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "qwk": float(metrics["quadratic_weighted_kappa"]),
                "validation_loss": float(row["audit"]["best_validation_loss"]),
                "fit_summary_path": row["fit_summary_path"],
            }
        )
    ranking.sort(
        key=lambda row: (
            -row["macro_f1"],
            -row["balanced_accuracy"],
            -row["qwk"],
            row["validation_loss"],
            row["candidate_id"],
        )
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    return ranking


def run_selection(
    run_root: Path,
    protocol: dict[str, Any],
    current_protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    smoke_path = run_root / "smoke" / "smoke_summary.json"
    if not smoke_path.is_file() or read_json(smoke_path)["status"] != "passed":
        raise RuntimeError("Run a passing IDEA-050 smoke before selection")
    selection_root = run_root / "selection"
    index_path = selection_root / "selection_index.json"
    if index_path.is_file():
        if not resume:
            raise FileExistsError(index_path)
        print(index_path.read_text(encoding="utf-8"), flush=True)
        return
    write_json(
        selection_root / "run_manifest.json",
        stage_manifest("select", protocol, device),
    )
    base_seed = int(protocol["selection"]["base_seed"])
    locks = []
    completed = 0
    for repeat in protocol["splits"]["repeats"]:
        for outer_fold in protocol["splits"]["outer_folds"]:
            indices = hpo.fold_indices(
                store, roles, int(repeat), int(outer_fold), include_test=False
            )
            seed = hpo.full_seed(base_seed, int(repeat), int(outer_fold))
            rows = []
            for candidate in protocol["lora"]["candidates"]:
                output_dir = (
                    selection_root
                    / "candidates"
                    / f"repeat_{repeat}"
                    / f"fold_{outer_fold}"
                    / candidate["candidate_id"]
                )
                summary_path = output_dir / "fit_summary.json"
                if summary_path.is_file():
                    if not resume:
                        raise FileExistsError(summary_path)
                    summary = read_json(summary_path)
                else:
                    print(
                        f"SELECT repeat={repeat} fold={outer_fold} "
                        f"candidate={candidate['candidate_id']} seed={seed}",
                        flush=True,
                    )
                    best_epoch, fit_audit = fit_inner_lora(
                        current_protocol,
                        protocol["lora"],
                        dict(candidate),
                        store,
                        indices["train"],
                        indices["validation"],
                        training_args(protocol),
                        device,
                        seed,
                    )
                    summary = {
                        "status": "complete",
                        "stage": "nested_inner_only_candidate_selection",
                        "outer_test_accessed": False,
                        "candidate": candidate,
                        "repeat": int(repeat),
                        "outer_fold": int(outer_fold),
                        "base_seed": base_seed,
                        "full_seed": seed,
                        "inner_train_calls": int(len(indices["train"])),
                        "inner_validation_calls": int(len(indices["validation"])),
                        "best_epoch": best_epoch,
                        "audit": fit_audit,
                    }
                    write_json(summary_path, summary)
                summary["fit_summary_path"] = repo_relative(summary_path)
                rows.append(summary)
                completed += 1
            ranking = rank_fold_candidates(rows)
            lock_path = (
                selection_root
                / "locks"
                / f"repeat_{repeat}_fold_{outer_fold}.json"
            )
            lock = {
                "schema_version": "1.0",
                "status": "locked_before_outer_evaluation",
                "outer_test_accessed": False,
                "protocol_sha256": sha256(PROTOCOL_PATH),
                "runner_sha256": sha256(Path(__file__)),
                "repeat": int(repeat),
                "outer_fold": int(outer_fold),
                "base_seed": base_seed,
                "full_seed": seed,
                "ranking": ranking,
                "selected": ranking[0],
            }
            write_json(lock_path, lock)
            locks.append(
                {
                    "repeat": int(repeat),
                    "outer_fold": int(outer_fold),
                    "path": repo_relative(lock_path),
                    "sha256": sha256(lock_path),
                    "selected_candidate_id": ranking[0]["candidate_id"],
                }
            )
    frequencies = Counter(row["selected_candidate_id"] for row in locks)
    index = {
        "schema_version": "1.0",
        "status": "locked_before_initial_outer_evaluation",
        "outer_test_accessed": False,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "base_seed": base_seed,
        "completed_inner_fits": completed,
        "selection_locks": locks,
        "selection_frequency": {
            candidate["candidate_id"]: int(frequencies[candidate["candidate_id"]])
            for candidate in protocol["lora"]["candidates"]
        },
    }
    write_json(index_path, index)
    write_json(
        selection_root / "run_summary.json",
        {
            "status": "complete",
            "outer_test_accessed": False,
            "completed_inner_fits": completed,
            "selection_locks": len(locks),
            "selection_index_path": repo_relative(index_path),
            "selection_index_sha256": sha256(index_path),
        },
    )
    print(json.dumps(index, indent=2), flush=True)


def load_and_verify_selection(
    run_root: Path, protocol: dict[str, Any]
) -> tuple[Path, dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
    index_path = run_root / "selection" / "selection_index.json"
    if not index_path.is_file():
        raise FileNotFoundError("Run IDEA-050 selection before evaluation")
    index = read_json(index_path)
    if index["status"] != "locked_before_initial_outer_evaluation":
        raise RuntimeError("IDEA-050 selection index is not locked")
    if index["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise RuntimeError("IDEA-050 protocol changed after selection")
    if index["runner_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("IDEA-050 runner changed after selection")
    locks = {}
    for row in index["selection_locks"]:
        path = REPO_ROOT / row["path"]
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"IDEA-050 selection lock changed: {path}")
        lock = read_json(path)
        if lock["status"] != "locked_before_outer_evaluation":
            raise RuntimeError(f"IDEA-050 fold lock is not locked: {path}")
        locks[(int(row["repeat"]), int(row["outer_fold"]))] = lock
    if len(locks) != 12:
        raise RuntimeError("IDEA-050 requires 12 fold selection locks")
    return index_path, index, locks


def enriched_animal_metrics(
    metrics: dict[str, Any], animals: pd.DataFrame
) -> dict[str, Any]:
    result = dict(metrics)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        animals["true_label"],
        animals["predicted_label"],
        labels=[0, 1, 2],
        zero_division=0,
    )
    result["per_class_precision"] = {
        label: float(value) for label, value in zip(LABELS, precision)
    }
    result["per_class_recall"] = {
        label: float(value) for label, value in zip(LABELS, recall)
    }
    result["per_class_f1"] = {
        label: float(value) for label, value in zip(LABELS, class_f1)
    }
    result["per_class_support"] = {
        label: int(value) for label, value in zip(LABELS, support)
    }
    result["cross_two_level_errors"] = int(
        (
            (animals["true_label"] == 0) & (animals["predicted_label"] == 2)
        ).sum()
        + (
            (animals["true_label"] == 2) & (animals["predicted_label"] == 0)
        ).sum()
    )
    return result


def hierarchical_paired_bootstrap(
    animal_frames: dict[str, dict[int, pd.DataFrame]],
    paired_differences: list[float],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    repeat_ids = sorted(animal_frames["ast_head_only"])
    values = np.empty(repeats, dtype=np.float64)
    for bootstrap_index in range(repeats):
        sampled_repeats = rng.choice(repeat_ids, size=len(repeat_ids), replace=True)
        repeat_differences = []
        for repeat in sampled_repeats:
            head = animal_frames["ast_head_only"][int(repeat)].sort_values(
                "cat_id"
            ).reset_index(drop=True)
            lora = animal_frames["ast_lora"][int(repeat)].sort_values(
                "cat_id"
            ).reset_index(drop=True)
            if not head[["cat_id", "true_label"]].equals(
                lora[["cat_id", "true_label"]]
            ):
                raise RuntimeError("Paired bootstrap animal frames do not align")
            sampled_indices = []
            for label in range(3):
                positions = np.flatnonzero(head["true_label"].to_numpy() == label)
                sampled_indices.extend(
                    rng.choice(positions, size=len(positions), replace=True).tolist()
                )
            true = head.loc[sampled_indices, "true_label"].to_numpy()
            head_pred = head.loc[sampled_indices, "predicted_label"].to_numpy()
            lora_pred = lora.loc[sampled_indices, "predicted_label"].to_numpy()
            repeat_differences.append(
                f1_score(true, lora_pred, labels=[0, 1, 2], average="macro")
                - f1_score(true, head_pred, labels=[0, 1, 2], average="macro")
            )
        values[bootstrap_index] = float(np.mean(repeat_differences))
    return {
        "method": "paired repeat-and-class-stratified animal bootstrap",
        "bootstrap_repeats": repeats,
        "seed": seed,
        "observed_mean_difference": float(np.mean(paired_differences)),
        "bootstrap_mean": float(values.mean()),
        "ci_lower_2_5_percent": float(np.quantile(values, 0.025)),
        "ci_upper_97_5_percent": float(np.quantile(values, 0.975)),
        "probability_difference_positive": float(np.mean(values > 0.0)),
    }


def aggregate_evaluation(
    evaluation_root: Path,
    protocol: dict[str, Any],
    selection_index: dict[str, Any],
) -> dict[str, Any]:
    base_seed = int(protocol["initial_evaluation"]["base_seed"])
    metrics_by_pipeline: dict[str, list[dict[str, Any]]] = {}
    animal_frames: dict[str, dict[int, pd.DataFrame]] = {
        pipeline: {} for pipeline in PIPELINES
    }
    for pipeline in PIPELINES:
        rows = []
        for repeat in protocol["splits"]["repeats"]:
            parts = []
            for outer_fold in protocol["splits"]["outer_folds"]:
                path = (
                    evaluation_root
                    / "fits"
                    / pipeline
                    / f"repeat_{repeat}"
                    / f"fold_{outer_fold}"
                    / f"base_seed_{base_seed}"
                    / "outer_test_call_predictions.csv"
                )
                parts.append(pd.read_csv(path, dtype={"cat_id": str}))
            calls = pd.concat(parts, ignore_index=True)
            if len(calls) != 792 or calls["call_index"].nunique() != 792:
                raise RuntimeError("IDEA-050 complete OOF must cover 792 calls")
            metrics, animals = idea019.evaluate_frame(calls)
            if len(animals) != 111 or animals["cat_id"].nunique() != 111:
                raise RuntimeError("IDEA-050 complete OOF must cover 111 cats")
            metrics = enriched_animal_metrics(metrics, animals)
            animals.insert(0, "base_seed", base_seed)
            animals.insert(0, "repeat", int(repeat))
            animals.insert(0, "pipeline", pipeline)
            output = evaluation_root / "oof" / pipeline / f"repeat_{repeat}_animals.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            animals.to_csv(output, index=False)
            animal_frames[pipeline][int(repeat)] = animals
            rows.append({"repeat": int(repeat), **metrics, "n": 111})
        metrics_by_pipeline[pipeline] = rows
    aggregate = {}
    for pipeline, rows in metrics_by_pipeline.items():
        f1_values = [float(row["macro_f1"]) for row in rows]
        aggregate[pipeline] = {
            "macro_f1_mean": float(np.mean(f1_values)),
            "macro_f1_sample_sd": float(np.std(f1_values, ddof=1)),
            "balanced_accuracy_mean": float(
                np.mean([row["balanced_accuracy"] for row in rows])
            ),
            "qwk_mean": float(
                np.mean([row["quadratic_weighted_kappa"] for row in rows])
            ),
        }
    head = metrics_by_pipeline["ast_head_only"]
    lora = metrics_by_pipeline["ast_lora"]
    macro_differences = [
        float(candidate["macro_f1"] - reference["macro_f1"])
        for reference, candidate in zip(head, lora)
    ]
    ba_differences = [
        float(candidate["balanced_accuracy"] - reference["balanced_accuracy"])
        for reference, candidate in zip(head, lora)
    ]
    qwk_differences = [
        float(
            candidate["quadratic_weighted_kappa"]
            - reference["quadratic_weighted_kappa"]
        )
        for reference, candidate in zip(head, lora)
    ]
    positive = int(sum(value > 0 for value in macro_differences))
    secondary_consistent_decline = all(value < 0 for value in ba_differences) or all(
        value < 0 for value in qwk_differences
    )
    bootstrap_protocol = protocol["initial_evaluation"]
    paired = {
        "macro_f1_differences": macro_differences,
        "macro_f1_mean_difference": float(np.mean(macro_differences)),
        "macro_f1_positive_repeats": positive,
        "balanced_accuracy_differences": ba_differences,
        "balanced_accuracy_mean_difference": float(np.mean(ba_differences)),
        "qwk_differences": qwk_differences,
        "qwk_mean_difference": float(np.mean(qwk_differences)),
    }
    paired["hierarchical_bootstrap"] = hierarchical_paired_bootstrap(
        animal_frames,
        macro_differences,
        repeats=int(bootstrap_protocol["paired_bootstrap_repeats"]),
        seed=int(bootstrap_protocol["paired_bootstrap_seed"]),
    )
    paired["seed_expansion_signal"] = {
        "mean_macro_f1_positive": paired["macro_f1_mean_difference"] > 0.0,
        "minimum_two_positive_repeats": positive >= 2,
        "secondary_metrics_consistently_declined": secondary_consistent_decline,
        "supports_team_review_for_expansion": (
            paired["macro_f1_mean_difference"] > 0.0
            and positive >= 2
            and not secondary_consistent_decline
        ),
    }
    return {
        "status": "complete",
        "pipelines": metrics_by_pipeline,
        "aggregate": aggregate,
        "lora_minus_matched_head_only": paired,
        "selection_frequency": selection_index["selection_frequency"],
    }


def run_evaluation(
    run_root: Path,
    protocol: dict[str, Any],
    current_protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    selection_path, selection_index, locks = load_and_verify_selection(
        run_root, protocol
    )
    evaluation_root = run_root / "evaluation"
    summary_path = evaluation_root / "summary.json"
    if summary_path.is_file():
        if not resume:
            raise FileExistsError(summary_path)
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return
    write_json(
        evaluation_root / "run_manifest.json",
        stage_manifest("evaluate", protocol, device, selection_path),
    )
    base_seed = int(protocol["initial_evaluation"]["base_seed"])
    completed = []
    args = training_args(protocol)
    for repeat in protocol["splits"]["repeats"]:
        for outer_fold in protocol["splits"]["outer_folds"]:
            indices = hpo.fold_indices(
                store, roles, int(repeat), int(outer_fold), include_test=True
            )
            seed = hpo.full_seed(base_seed, int(repeat), int(outer_fold))
            outer_train = np.concatenate((indices["train"], indices["validation"]))
            lock = locks[(int(repeat), int(outer_fold))]
            for pipeline in PIPELINES:
                output_dir = (
                    evaluation_root
                    / "fits"
                    / pipeline
                    / f"repeat_{repeat}"
                    / f"fold_{outer_fold}"
                    / f"base_seed_{base_seed}"
                )
                fit_path = output_dir / "fit_summary.json"
                if fit_path.is_file():
                    if not resume:
                        raise FileExistsError(fit_path)
                    completed.append(read_json(fit_path))
                    continue
                print(
                    f"EVALUATE pipeline={pipeline} repeat={repeat} "
                    f"fold={outer_fold} seed={seed}",
                    flush=True,
                )
                if pipeline == "ast_head_only":
                    best_epoch, inner_audit = idea019.fit_inner(
                        "head_only",
                        None,
                        current_protocol,
                        store,
                        indices["train"],
                        indices["validation"],
                        args,
                        device,
                        seed,
                    )
                    test_frame, outer_audit = idea019.fit_outer_and_predict(
                        "head_only",
                        None,
                        current_protocol,
                        store,
                        outer_train,
                        indices["test"],
                        best_epoch,
                        args,
                        device,
                        seed,
                    )
                    candidate = None
                    selection_reference = None
                else:
                    candidate = dict(lock["selected"]["parameters"])
                    best_epoch = int(lock["selected"]["selected_epoch"])
                    inner_audit = {
                        "source": "locked_nested_inner_candidate_fit",
                        "fit_summary_path": lock["selected"]["fit_summary_path"],
                        "selected_validation_macro_f1": lock["selected"]["macro_f1"],
                        "selected_validation_balanced_accuracy": lock["selected"][
                            "balanced_accuracy"
                        ],
                        "selected_validation_qwk": lock["selected"]["qwk"],
                        "selected_validation_loss": lock["selected"][
                            "validation_loss"
                        ],
                    }
                    test_frame, outer_audit = fit_outer_lora(
                        current_protocol,
                        protocol["lora"],
                        candidate,
                        store,
                        outer_train,
                        indices["test"],
                        best_epoch,
                        args,
                        device,
                        seed,
                    )
                    selection_reference = {
                        "lock_path": next(
                            row["path"]
                            for row in selection_index["selection_locks"]
                            if int(row["repeat"]) == int(repeat)
                            and int(row["outer_fold"]) == int(outer_fold)
                        ),
                        "lock_sha256": next(
                            row["sha256"]
                            for row in selection_index["selection_locks"]
                            if int(row["repeat"]) == int(repeat)
                            and int(row["outer_fold"]) == int(outer_fold)
                        ),
                    }
                output_dir.mkdir(parents=True, exist_ok=True)
                prediction_path = output_dir / "outer_test_call_predictions.csv"
                test_frame.to_csv(prediction_path, index=False)
                fit = {
                    "status": "complete",
                    "stage": "initial_nested_paired_evaluation",
                    "outer_test_accessed": True,
                    "pipeline": pipeline,
                    "candidate": candidate,
                    "repeat": int(repeat),
                    "outer_fold": int(outer_fold),
                    "base_seed": base_seed,
                    "full_seed": seed,
                    "selected_epoch": best_epoch,
                    "inner": inner_audit,
                    "outer": outer_audit,
                    "selection_reference": selection_reference,
                    "prediction_path": repo_relative(prediction_path),
                }
                write_json(fit_path, fit)
                completed.append(fit)
    summary = aggregate_evaluation(evaluation_root, protocol, selection_index)
    summary["selection_index_path"] = repo_relative(selection_path)
    summary["selection_index_sha256"] = sha256(selection_path)
    summary["completed_outer_pipeline_fits"] = len(completed)
    write_json(summary_path, summary)
    write_json(
        evaluation_root / "run_summary.json",
        {
            "status": "complete",
            "completed_outer_pipeline_fits": len(completed),
            "expected_outer_pipeline_fits": int(
                protocol["initial_evaluation"]["outer_pipeline_fits"]
            ),
            "summary_path": repo_relative(summary_path),
            "fits": completed,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_PATH)
    locked_protocol = read_json(LOCKED_MODEL_PROTOCOL_PATH)
    verify_protocol(protocol)
    current_protocol = model_protocol(locked_protocol, protocol)
    run_root = (RUNS_ROOT / args.output_subdir).resolve()
    if RUNS_ROOT.resolve() not in run_root.parents:
        raise ValueError("--output-subdir must stay below runs")
    run_root.mkdir(parents=True, exist_ok=True)
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    store = idea019.load_feature_store()
    device = idea019.resolve_device(args.device)
    print(
        f"IDEA-050 stage={args.stage}; device={device}; "
        f"device_name={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}",
        flush=True,
    )
    if args.stage == "audit":
        run_audit(
            run_root,
            protocol,
            current_protocol,
            roles,
            store,
            device,
            args.resume,
        )
    elif args.stage == "smoke":
        run_smoke(
            run_root,
            protocol,
            current_protocol,
            roles,
            store,
            device,
            args.resume,
        )
    elif args.stage == "select":
        run_selection(
            run_root,
            protocol,
            current_protocol,
            roles,
            store,
            device,
            args.resume,
        )
    else:
        run_evaluation(
            run_root,
            protocol,
            current_protocol,
            roles,
            store,
            device,
            args.resume,
        )


if __name__ == "__main__":
    main()
