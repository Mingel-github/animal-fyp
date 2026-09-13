"""Run IDEA-054 constrained AST calibration with LayerNorm, SSF, and BitFit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_ast_finetuning as ast_base  # noqa: E402
import run_meowagenet_ast_cat_balance_global_weighting_v1 as weighting  # noqa: E402
import run_meowagenet_idea051_cat_set as evaluation  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea054_constrained_ast_calibration_v1.json"
)
PLAN_PATH = REPO_ROOT / "plan" / "IDEA-054_constrained_AST_calibration.md"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
LOCKED_MODEL_PROTOCOL_PATH = (
    REPO_ROOT / "configs" / "protocol" / "meowagenet_locked_v1.json"
)
RUN_ROOT = REPO_ROOT / "runs" / "meowagenet_idea054_constrained_ast_calibration_v1"
PIPELINES = (
    "A0_frozen_ast_reference",
    "P1_layernorm_tuning",
    "P2_block_output_ssf",
    "P3_bitfit",
)
CANDIDATES = PIPELINES[1:]
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
LABEL_NAMES = ("kitten", "adult", "senior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "evaluate"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def git_blob_sha256(revision: str, path: Path) -> str:
    content = subprocess.check_output(
        ["git", "show", f"{revision}:{repo_relative(path)}"], cwd=REPO_ROOT
    )
    return hashlib.sha256(content).hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol["protocol_id"] != "meowagenet-idea054-constrained-ast-calibration-v1":
        raise RuntimeError("Unexpected IDEA-054 protocol")
    if tuple(protocol["initial_evaluation"]["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-054 pipeline matrix changed")
    dependencies = protocol["dependencies"]
    checks = {
        PLAN_PATH: protocol["idea"]["sha256"],
        ROLES_PATH: protocol["splits"]["roles_sha256"],
        LOCKED_MODEL_PROTOCOL_PATH: protocol["ast"]["locked_model_protocol_sha256"],
        REPO_ROOT / dependencies["ast_base_runner"]: dependencies[
            "ast_base_runner_sha256"
        ],
        REPO_ROOT / dependencies["global_weighting_runner"]: dependencies[
            "global_weighting_runner_sha256"
        ],
        REPO_ROOT / dependencies["reference_runner"]: dependencies[
            "reference_runner_sha256"
        ],
        REPO_ROOT / dependencies["fbank_path"]: dependencies["fbank_sha256"],
        REPO_ROOT / dependencies["frozen_embedding_path"]: dependencies[
            "frozen_embedding_sha256"
        ],
    }
    for path, expected in checks.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"IDEA-054 dependency checksum mismatch: {path}")
    fixed = protocol["fixed_training"]
    if int(fixed["call_micro_batch_size"]) * int(
        fixed["gradient_accumulation_steps"]
    ) != int(fixed["accumulation_window_calls"]):
        raise RuntimeError("IDEA-054 accumulation window is inconsistent")
    settings = protocol["initial_evaluation"]
    expected_fits = (
        len(PIPELINES)
        * len(settings["base_seeds"])
        * len(settings["repeats"])
        * len(settings["outer_folds"])
    )
    if expected_fits != int(settings["total_outer_fits"]):
        raise RuntimeError("IDEA-054 fit budget is inconsistent")


def load_inputs() -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    store = ast_base.load_feature_store()
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    locked = read_json(LOCKED_MODEL_PROTOCOL_PATH)
    if len(store.call_ids) != 792 or len(np.unique(store.cat_ids)) != 111:
        raise RuntimeError("Unexpected IDEA-054 data scope")
    if not np.array_equal(store.call_ids, np.load(ast_base.FROZEN_EMBEDDING_PATH)["call_ids"].astype(str)):
        raise RuntimeError("Fbank and frozen embedding call order differs")
    return store, roles, locked


class SSFBlock(nn.Module):
    """Identity-initialized scale and shift after one frozen AST block."""

    def __init__(self, base: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.base = base
        self.ssf_scale = nn.Parameter(torch.ones(hidden_size))
        self.ssf_shift = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        outputs = self.base(*args, **kwargs)
        hidden = outputs[0] * self.ssf_scale + self.ssf_shift
        return (hidden, *outputs[1:])


def make_head(protocol: dict[str, Any], store: Any, train_indices: np.ndarray) -> nn.Module:
    values = store.frozen_embeddings[train_indices]
    return ast_base.ClassificationHead(
        mean=values.mean(axis=0),
        scale=values.std(axis=0),
        dropout=float(protocol["fixed_training"]["dropout"]),
    )


class OnlineCalibrationClassifier(nn.Module):
    def __init__(
        self,
        locked_protocol: dict[str, Any],
        pipeline: str,
        head: nn.Module,
    ) -> None:
        super().__init__()
        ast_config = locked_protocol["ast"]
        self.ast = ast_base.ASTModel.from_pretrained(
            ast_config["checkpoint"],
            revision=ast_config["revision"],
            cache_dir=ast_base.HF_CACHE,
            use_safetensors=True,
        )
        self.geometry_audit = ast_base.adapt_standard_geometry(
            self.ast, locked_protocol
        )
        for parameter in self.ast.parameters():
            parameter.requires_grad = False
        if pipeline == "P1_layernorm_tuning":
            for module in self.ast.modules():
                if isinstance(module, nn.LayerNorm):
                    for parameter in module.parameters(recurse=False):
                        parameter.requires_grad = True
        elif pipeline == "P2_block_output_ssf":
            hidden_size = int(self.ast.config.hidden_size)
            for index, layer in enumerate(list(self.ast.encoder.layer)):
                self.ast.encoder.layer[index] = SSFBlock(layer, hidden_size)
        elif pipeline == "P3_bitfit":
            for name, parameter in self.ast.named_parameters():
                if name.endswith("bias"):
                    parameter.requires_grad = True
        elif pipeline != "online_frozen":
            raise ValueError(f"Unknown online pipeline: {pipeline}")
        self.pipeline = pipeline
        self.head = head

    def call_embeddings(
        self,
        instances: torch.Tensor,
        instance_to_call: torch.Tensor,
        call_count: int,
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
        return call_embeddings / counts[:, None]

    def forward(
        self,
        instances: torch.Tensor,
        instance_to_call: torch.Tensor,
        call_count: int,
    ) -> torch.Tensor:
        return self.head(
            self.call_embeddings(instances, instance_to_call, call_count)
        )


def build_model(
    pipeline: str,
    protocol: dict[str, Any],
    locked_protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
) -> nn.Module:
    head = make_head(protocol, store, train_indices)
    if pipeline == PIPELINES[0]:
        return ast_base.FrozenClassifier(head)
    return OnlineCalibrationClassifier(locked_protocol, pipeline, head)


def adaptation_records(model: nn.Module) -> list[dict[str, Any]]:
    if not hasattr(model, "ast"):
        return []
    return [
        {"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
        for name, parameter in model.ast.named_parameters()
        if parameter.requires_grad
    ]


def parameter_audit(model: nn.Module, pipeline: str) -> dict[str, Any]:
    trainable = [
        {"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    adaptation = adaptation_records(model)
    expected = 0 if pipeline == PIPELINES[0] else None
    forbidden = []
    adaptation_names = {f"ast.{row['name']}" for row in adaptation}
    for row in trainable:
        if not row["name"].startswith("head.") and row["name"] not in adaptation_names:
            forbidden.append(row["name"])
    return {
        "model_total_parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_total_parameters": int(sum(row["numel"] for row in trainable)),
        "trainable_head_parameters": int(
            sum(row["numel"] for row in trainable if row["name"].startswith("head."))
        ),
        "trainable_adaptation_parameters": int(sum(row["numel"] for row in adaptation)),
        "trainable_adaptation_tensors": len(adaptation),
        "adaptation_parameters": adaptation,
        "forbidden_trainable_parameters": forbidden,
        "expected_reference_adaptation_parameters": expected,
    }


def build_loader(
    store: Any,
    indices: np.ndarray,
    pipeline: str,
    batch_size: int,
    training: bool,
    seed: int,
) -> DataLoader:
    mode = "frozen" if pipeline == PIPELINES[0] else "adapter"
    dataset = ast_base.CallDataset(store, indices, mode)
    if training:
        sampler = weighting.DeterministicNoSingletonBatchSampler(
            len(dataset), batch_size, seed
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=ast_base.collate_calls,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ast_base.collate_calls,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def training_values(protocol: dict[str, Any]) -> dict[str, Any]:
    fixed = protocol["fixed_training"]
    return {
        "head_lr": float(fixed["head_learning_rate"]),
        "adaptation_lr": float(fixed["adaptation_learning_rate"]),
        "epsilon": float(fixed["optimizer_epsilon"]),
        "gradient_clip": float(fixed["gradient_clip"]),
        "max_epochs": int(fixed["maximum_epochs"]),
        "patience": int(fixed["early_stopping_patience"]),
        "batch_size": int(fixed["call_micro_batch_size"]),
        "evaluation_batch_size": int(fixed["evaluation_batch_size"]),
        "accumulation_steps": int(fixed["gradient_accumulation_steps"]),
        "window_calls": int(fixed["accumulation_window_calls"]),
    }


def make_optimizer(model: nn.Module, pipeline: str, values: dict[str, Any]) -> torch.optim.Optimizer:
    if pipeline == PIPELINES[0]:
        return torch.optim.Adamax(
            model.parameters(), lr=values["head_lr"], eps=values["epsilon"]
        )
    adaptation = [parameter for parameter in model.ast.parameters() if parameter.requires_grad]
    if not adaptation:
        raise RuntimeError(f"{pipeline} has no trainable AST parameters")
    return torch.optim.Adamax(
        [
            {"params": adaptation, "lr": values["adaptation_lr"]},
            {"params": model.head.parameters(), "lr": values["head_lr"]},
        ],
        eps=values["epsilon"],
    )


def compact_train_audit(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        key: audit[key]
        for key in (
            "processed_calls",
            "unique_processed_calls",
            "batch_count",
            "batch_sizes",
            "optimizer_steps",
            "final_window_calls",
            "final_window_is_partial",
            "effective_coefficient_total",
            "effective_coefficient_by_class",
            "batch_order_sha256",
        )
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    weight_lookup: torch.Tensor,
    weight_lookup_numpy: np.ndarray,
    store: Any,
    device: torch.device,
    values: dict[str, Any],
    scaler: torch.cuda.amp.GradScaler,
) -> tuple[float, dict[str, Any]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    weighted_total = 0.0
    weight_total = 0.0
    processed_indices: list[int] = []
    batch_sizes: list[int] = []
    for step, cpu_batch in enumerate(loader):
        indices = cpu_batch["call_indices"].numpy().astype(np.int64).tolist()
        processed_indices.extend(indices)
        batch_sizes.append(len(indices))
        batch = move_batch(cpu_batch, device)
        labels = batch["labels"]
        weights = weight_lookup[batch["call_indices"]]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(batch["instances"], batch["instance_to_call"], len(labels))
            per_call_loss = torch.nn.functional.cross_entropy(
                logits, labels, reduction="none"
            )
            loss = weighting.global_weighted_micro_loss(
                per_call_loss, weights, values["window_calls"]
            )
        scaler.scale(loss).backward()
        should_step = (
            (step + 1) % values["accumulation_steps"] == 0
            or step + 1 == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), values["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        weighted_total += float((per_call_loss.detach() * weights).sum())
        weight_total += float(weights.sum())
    audit = weighting.effective_coefficient_audit(
        processed_indices,
        batch_sizes,
        weight_lookup_numpy,
        store,
        values["window_calls"],
        values["accumulation_steps"],
    )
    return weighted_total / weight_total, compact_train_audit(audit)


def predict_calls(
    model: nn.Module,
    loader: DataLoader,
    store: Any,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(
                    batch["instances"], batch["instance_to_call"], len(batch["labels"])
                )
                probabilities = torch.softmax(logits, dim=1).float().cpu().numpy()
            indices = cpu_batch["call_indices"].numpy().astype(np.int64)
            labels = cpu_batch["labels"].numpy().astype(np.int64)
            for local_index, call_index in enumerate(indices):
                rows.append(
                    {
                        "call_index": int(call_index),
                        "call_id": str(store.call_ids[call_index]),
                        "cat_id": str(store.cat_ids[call_index]),
                        "true_label": int(labels[local_index]),
                        **{
                            column: float(probabilities[local_index, class_index])
                            for class_index, column in enumerate(PROBABILITY_COLUMNS)
                        },
                        "predicted_label": int(probabilities[local_index].argmax()),
                    }
                )
    return pd.DataFrame(rows).sort_values("call_index").reset_index(drop=True)


def portable_trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: tensor.detach().cpu()
        for name, tensor in state.items()
        if name.startswith("head.") or name in names
    }


def fit_inner(
    pipeline: str,
    protocol: dict[str, Any],
    locked_protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
    max_epochs_override: int | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    weighting.historical.set_seed(seed)
    values = training_values(protocol)
    model = build_model(pipeline, protocol, locked_protocol, store, train_indices).to(device)
    parameters = parameter_audit(model, pipeline)
    initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = make_optimizer(model, pipeline, values)
    lookup_numpy = weighting.global_class_balanced_call_weights(
        store.labels, train_indices
    )
    lookup = torch.from_numpy(lookup_numpy).to(device)
    train_loader = build_loader(
        store, train_indices, pipeline, values["batch_size"], True, seed
    )
    validation_loader = build_loader(
        store,
        validation_indices,
        pipeline,
        values["evaluation_batch_size"],
        False,
        seed,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_loss = float("inf")
    best_epoch = 1
    best_metrics: dict[str, Any] = {}
    without_improvement = 0
    history = []
    maximum_epochs = max_epochs_override or values["max_epochs"]
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, maximum_epochs + 1):
        train_loss, train_audit = train_one_epoch(
            model,
            train_loader,
            optimizer,
            lookup,
            lookup_numpy,
            store,
            device,
            values,
            scaler,
        )
        validation_calls = predict_calls(model, validation_loader, store, device)
        validation_animals = evaluation.calls_to_animals(validation_calls)
        validation_loss = evaluation.animal_cross_entropy(validation_animals)
        metrics = evaluation.animal_metrics(validation_animals)
        history.append(
            {
                "epoch": epoch,
                "train_weighted_loss": train_loss,
                "validation_animal_cross_entropy": validation_loss,
                "validation_animal_macro_f1": metrics["macro_f1"],
                "validation_animal_balanced_accuracy": metrics["balanced_accuracy"],
                "validation_animal_qwk": metrics["quadratic_weighted_kappa"],
                "train_unit_audit": train_audit,
            }
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_metrics = metrics
            without_improvement = 0
        else:
            without_improvement += 1
        print(
            f"{pipeline} inner epoch={epoch} train={train_loss:.4f} "
            f"animal_val={validation_loss:.4f} val_F1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if max_epochs_override is None and without_improvement >= values["patience"]:
            break
    updates = {
        name: float((parameter.detach().cpu() - initial[name]).abs().max())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    checkpoint_audit = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        current_calls = predict_calls(model, validation_loader, store, device)
        saved = portable_trainable_state(model)
        torch.save(saved, checkpoint_path)
        weighting.historical.set_seed(seed)
        reloaded = build_model(
            pipeline, protocol, locked_protocol, store, train_indices
        ).to(device)
        state = reloaded.state_dict()
        state.update(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        reloaded.load_state_dict(state)
        reloaded_calls = predict_calls(reloaded, validation_loader, store, device)
        difference = float(
            np.abs(
                current_calls[list(PROBABILITY_COLUMNS)].to_numpy()
                - reloaded_calls[list(PROBABILITY_COLUMNS)].to_numpy()
            ).max()
        )
        checkpoint_audit = {
            "path": repo_relative(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "saved_tensors": len(saved),
            "reload_max_probability_difference": difference,
        }
        del reloaded
    adaptation_names = {f"ast.{row['name']}" for row in parameters["adaptation_parameters"]}
    audit = {
        "best_epoch": int(best_epoch),
        "stopped_epoch": len(history),
        "best_validation_animal_cross_entropy": float(best_loss),
        "best_validation_animal_metrics": best_metrics,
        "history": history,
        "train_seconds": time.perf_counter() - started,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "parameters": parameters,
        "maximum_trainable_parameter_updates": updates,
        "updated_head_parameter_tensors": int(
            sum(value > 0.0 for name, value in updates.items() if name.startswith("head."))
        ),
        "updated_adaptation_parameter_tensors": int(
            sum(value > 0.0 for name, value in updates.items() if name in adaptation_names)
        ),
        "target_weight_audit": weighting.target_weight_audit(
            lookup_numpy, store, train_indices
        ),
        "checkpoint_audit": checkpoint_audit,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, audit


def fit_outer(
    pipeline: str,
    protocol: dict[str, Any],
    locked_protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    epochs: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    weighting.historical.set_seed(seed)
    values = training_values(protocol)
    model = build_model(pipeline, protocol, locked_protocol, store, train_indices).to(device)
    parameters = parameter_audit(model, pipeline)
    optimizer = make_optimizer(model, pipeline, values)
    lookup_numpy = weighting.global_class_balanced_call_weights(
        store.labels, train_indices
    )
    lookup = torch.from_numpy(lookup_numpy).to(device)
    train_loader = build_loader(
        store, train_indices, pipeline, values["batch_size"], True, seed
    )
    test_loader = build_loader(
        store, test_indices, pipeline, values["evaluation_batch_size"], False, seed
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        loss, train_audit = train_one_epoch(
            model,
            train_loader,
            optimizer,
            lookup,
            lookup_numpy,
            store,
            device,
            values,
            scaler,
        )
        history.append(
            {"epoch": epoch, "train_weighted_loss": loss, "train_unit_audit": train_audit}
        )
        print(
            f"{pipeline} outer epoch={epoch}/{epochs} train={loss:.4f}", flush=True
        )
    calls = predict_calls(model, test_loader, store, device)
    animals = evaluation.calls_to_animals(calls)
    audit = {
        "epochs": int(epochs),
        "history": history,
        "test_animal_cross_entropy": evaluation.animal_cross_entropy(animals),
        "test_animal_metrics": evaluation.animal_metrics(animals),
        "train_and_predict_seconds": time.perf_counter() - started,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "parameters": parameters,
        "target_weight_audit": weighting.target_weight_audit(
            lookup_numpy, store, train_indices
        ),
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return calls, audit


def environment_lock(protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }


def first_batch_online_audit(
    protocol: dict[str, Any],
    locked_protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    loader = build_loader(store, validation_indices[:2], CANDIDATES[0], 2, False, seed)
    cpu_batch = next(iter(loader))
    batch = move_batch(cpu_batch, device)
    weighting.historical.set_seed(seed)
    base = OnlineCalibrationClassifier(
        locked_protocol,
        "online_frozen",
        make_head(protocol, store, train_indices),
    ).to(device)
    base.eval()
    with torch.inference_mode():
        embeddings = base.call_embeddings(
            batch["instances"], batch["instance_to_call"], len(batch["labels"])
        )
        base_logits = base.head(embeddings).float().cpu()
    cached = torch.from_numpy(
        store.frozen_embeddings[cpu_batch["call_indices"].numpy().astype(np.int64)]
    )
    result: dict[str, Any] = {
        "cached_vs_online_max_embedding_difference": float(
            (embeddings.float().cpu() - cached).abs().max()
        ),
        "candidate_initial_max_logit_difference": {},
        "parameter_audits": {},
    }
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for pipeline in CANDIDATES:
        weighting.historical.set_seed(seed)
        model = build_model(
            pipeline, protocol, locked_protocol, store, train_indices
        ).to(device)
        model.eval()
        with torch.inference_mode():
            logits = model(
                batch["instances"], batch["instance_to_call"], len(batch["labels"])
            ).float().cpu()
        result["candidate_initial_max_logit_difference"][pipeline] = float(
            (base_logits - logits).abs().max()
        )
        result["parameter_audits"][pipeline] = parameter_audit(model, pipeline)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def smoke(
    protocol: dict[str, Any],
    store: Any,
    roles: pd.DataFrame,
    locked_protocol: dict[str, Any],
    device: torch.device,
    resume: bool,
) -> None:
    output = RUN_ROOT / "smoke" / "summary.json"
    if output.is_file():
        if not resume:
            raise FileExistsError(output)
        print(output.read_text(encoding="utf-8"), flush=True)
        return
    settings = protocol["smoke"]
    seed = weighting.historical.full_seed(
        int(settings["base_seed"]), int(settings["repeat"]), int(settings["outer_fold"])
    )
    indices = weighting.historical.fold_indices(
        store,
        roles,
        int(settings["repeat"]),
        int(settings["outer_fold"]),
        include_test=False,
    )
    online = first_batch_online_audit(
        protocol,
        locked_protocol,
        store,
        indices["train"],
        indices["validation"],
        device,
        seed,
    )
    fits = {}
    for pipeline in CANDIDATES:
        print(f"IDEA054 smoke {pipeline}", flush=True)
        checkpoint = RUN_ROOT / "smoke" / f"{pipeline}_trainable_state.pt"
        best_epoch, audit = fit_inner(
            pipeline,
            protocol,
            locked_protocol,
            store,
            indices["train"],
            indices["validation"],
            device,
            seed,
            max_epochs_override=int(settings["epochs"]),
            checkpoint_path=checkpoint,
        )
        fits[pipeline] = {"selected_epoch": best_epoch, "audit": audit}
    expected = {
        pipeline: int(protocol["pipelines"][pipeline]["ast_trainable_parameters"])
        for pipeline in CANDIDATES
    }
    status = "passed"
    if online["cached_vs_online_max_embedding_difference"] > 1.0e-4:
        status = "failed"
    for pipeline in CANDIDATES:
        parameters = online["parameter_audits"][pipeline]
        audit = fits[pipeline]["audit"]
        if (
            online["candidate_initial_max_logit_difference"][pipeline] > 1.0e-5
            or parameters["trainable_adaptation_parameters"] != expected[pipeline]
            or parameters["trainable_head_parameters"] != 99075
            or parameters["forbidden_trainable_parameters"]
            or audit["updated_adaptation_parameter_tensors"] < 1
            or audit["updated_head_parameter_tensors"] < 1
            or audit["checkpoint_audit"]["reload_max_probability_difference"] > 1.0e-6
        ):
            status = "failed"
    result = {
        "status": status,
        "stage": "inner_only_smoke",
        "outer_test_accessed": False,
        "base_seed": int(settings["base_seed"]),
        "repeat": int(settings["repeat"]),
        "outer_fold": int(settings["outer_fold"]),
        "full_seed": int(seed),
        "calls": int(len(store.call_ids)),
        "cats": int(len(np.unique(store.cat_ids))),
        "inner_train_calls": int(len(indices["train"])),
        "inner_validation_calls": int(len(indices["validation"])),
        "online_initialization_audit": online,
        "fits": fits,
    }
    write_json(output, result)
    if status != "passed":
        raise RuntimeError("IDEA-054 smoke failed")
    environment_path = RUN_ROOT / "environment_lock.json"
    write_json(environment_path, environment_lock(protocol, device))
    revision = git_revision()
    if revision is None:
        raise RuntimeError("IDEA-054 requires a committed source revision")
    for path in (PLAN_PATH, PROTOCOL_PATH, Path(__file__)):
        if git_blob_sha256(revision, path) != sha256(path):
            raise RuntimeError(f"IDEA-054 source differs from commit {revision}: {path}")
    lock = {
        "schema_version": "1.0",
        "status": "locked_after_inner_only_smoke",
        "protocol_id": protocol["protocol_id"],
        "outer_test_accessed": False,
        "code_commit": revision,
        "plan_sha256": sha256(PLAN_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "roles_sha256": sha256(ROLES_PATH),
        "smoke_summary_sha256": sha256(output),
        "environment_lock_sha256": sha256(environment_path),
    }
    write_json(RUN_ROOT / "execution_lock.json", lock)
    print(json.dumps(result, indent=2), flush=True)


def verify_execution_lock(protocol: dict[str, Any]) -> dict[str, Any]:
    path = RUN_ROOT / "execution_lock.json"
    if not path.is_file():
        raise FileNotFoundError("Run IDEA-054 smoke before evaluation")
    lock = read_json(path)
    if lock["status"] != "locked_after_inner_only_smoke":
        raise RuntimeError("IDEA-054 execution lock is incomplete")
    checks = {
        PLAN_PATH: lock["plan_sha256"],
        PROTOCOL_PATH: lock["protocol_sha256"],
        Path(__file__): lock["runner_sha256"],
        ROLES_PATH: lock["roles_sha256"],
        RUN_ROOT / "smoke" / "summary.json": lock["smoke_summary_sha256"],
        RUN_ROOT / "environment_lock.json": lock["environment_lock_sha256"],
    }
    for source, expected in checks.items():
        if sha256(source) != expected:
            raise RuntimeError(f"IDEA-054 execution-lock mismatch: {source}")
    if lock["protocol_id"] != protocol["protocol_id"]:
        raise RuntimeError("IDEA-054 protocol ID changed after smoke")
    return lock


def paired_row(reference: pd.DataFrame, candidate: pd.DataFrame, repeat: int) -> dict[str, Any]:
    left = reference.sort_values("cat_id").reset_index(drop=True)
    right = candidate.sort_values("cat_id").reset_index(drop=True)
    if not np.array_equal(left["cat_id"].to_numpy(), right["cat_id"].to_numpy()):
        raise RuntimeError("IDEA-054 paired cat order differs")
    left_metrics = evaluation.animal_metrics(left)
    right_metrics = evaluation.animal_metrics(right)
    left_correct = left["predicted_label"].to_numpy() == left["true_label"].to_numpy()
    right_correct = right["predicted_label"].to_numpy() == right["true_label"].to_numpy()
    changed = left["predicted_label"].to_numpy() != right["predicted_label"].to_numpy()
    return {
        "repeat": int(repeat),
        "reference_macro_f1": left_metrics["macro_f1"],
        "candidate_macro_f1": right_metrics["macro_f1"],
        "macro_f1_difference": right_metrics["macro_f1"] - left_metrics["macro_f1"],
        "balanced_accuracy_difference": right_metrics["balanced_accuracy"] - left_metrics["balanced_accuracy"],
        "qwk_difference": right_metrics["quadratic_weighted_kappa"] - left_metrics["quadratic_weighted_kappa"],
        "plain_accuracy_difference": right_metrics["plain_accuracy"] - left_metrics["plain_accuracy"],
        "changed_animals": int(changed.sum()),
        "gained_correct_animals": int((~left_correct & right_correct).sum()),
        "lost_correct_animals": int((left_correct & ~right_correct).sum()),
    }


def aggregate_results(protocol: dict[str, Any]) -> dict[str, Any]:
    root = RUN_ROOT / "evaluation"
    settings = protocol["initial_evaluation"]
    metrics_by_pipeline = {pipeline: [] for pipeline in PIPELINES}
    animals_by_key: dict[tuple[str, int], pd.DataFrame] = {}
    parameter_rows = {pipeline: [] for pipeline in PIPELINES}
    for repeat in settings["repeats"]:
        for pipeline in PIPELINES:
            frames = []
            for fold in settings["outer_folds"]:
                fit_root = (
                    root
                    / "fits"
                    / pipeline
                    / f"base_seed_{settings['base_seeds'][0]}"
                    / f"repeat_{repeat}"
                    / f"fold_{fold}"
                )
                frames.append(
                    pd.read_csv(
                        fit_root / "outer_test_animal_predictions.csv",
                        dtype={"cat_id": str},
                    )
                )
                fit = read_json(fit_root / "fit_summary.json")
                parameter_rows[pipeline].append(fit["outer"]["parameters"])
            animals = pd.concat(frames, ignore_index=True).sort_values("cat_id")
            if len(animals) != 111 or animals["cat_id"].nunique() != 111:
                raise RuntimeError("IDEA-054 complete OOF must contain 111 cats")
            animals = animals.reset_index(drop=True)
            animals_by_key[(pipeline, int(repeat))] = animals
            metrics = evaluation.animal_metrics(animals)
            metrics["base_seed"] = int(settings["base_seeds"][0])
            metrics["repeat"] = int(repeat)
            metrics_by_pipeline[pipeline].append(metrics)
            output = root / "complete_oof" / f"{pipeline}_repeat_{repeat}_animals.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            animals.to_csv(output, index=False)
    aggregate = {
        pipeline: evaluation.aggregate_metrics(rows)
        for pipeline, rows in metrics_by_pipeline.items()
    }
    paired: dict[str, list[dict[str, Any]]] = {}
    paired_summary = {}
    bootstraps = {}
    gates = {}
    gate = protocol["seed_expansion_gate"]
    change_rows = []
    for candidate in CANDIDATES:
        name = f"{candidate}_minus_{PIPELINES[0]}"
        rows = [
            paired_row(
                animals_by_key[(PIPELINES[0], int(repeat))],
                animals_by_key[(candidate, int(repeat))],
                int(repeat),
            )
            for repeat in settings["repeats"]
        ]
        paired[name] = rows
        differences = [row["macro_f1_difference"] for row in rows]
        mean_difference = float(np.mean(differences))
        positive = int(sum(value > 0 for value in differences))
        supporting = []
        for metric, key in (
            ("balanced_accuracy", "balanced_accuracy_difference"),
            ("QWK", "qwk_difference"),
            ("plain_accuracy", "plain_accuracy_difference"),
        ):
            value = float(np.mean([row[key] for row in rows]))
            if value > 0:
                supporting.append({"metric": metric, "mean_difference": value})
        paired_summary[name] = {
            "macro_f1_differences": differences,
            "mean_macro_f1_difference": mean_difference,
            "positive_repeats": positive,
            "mean_balanced_accuracy_difference": float(np.mean([row["balanced_accuracy_difference"] for row in rows])),
            "mean_qwk_difference": float(np.mean([row["qwk_difference"] for row in rows])),
            "mean_plain_accuracy_difference": float(np.mean([row["plain_accuracy_difference"] for row in rows])),
        }
        bootstraps[name] = evaluation.paired_cat_bootstrap(
            animals_by_key, PIPELINES[0], candidate, protocol
        )
        passed = (
            mean_difference >= float(gate["minimum_mean_macro_f1_gain"])
            and positive >= int(gate["minimum_positive_repeats"])
            and bool(supporting)
        )
        gates[candidate] = {
            "passed": bool(passed),
            "strong_gain": bool(mean_difference >= float(gate["strong_gain"])),
            "supporting_signals": supporting,
        }
        for repeat in settings["repeats"]:
            left = animals_by_key[(PIPELINES[0], int(repeat))]
            right = animals_by_key[(candidate, int(repeat))]
            merged = left.merge(right, on=["cat_id", "true_label"], suffixes=("_reference", "_candidate"))
            merged.insert(0, "candidate", candidate)
            merged.insert(1, "repeat", int(repeat))
            change_rows.append(merged)
    changes = pd.concat(change_rows, ignore_index=True)
    changes.to_csv(root / "paired_prediction_changes.csv", index=False)
    parameter_summary = {}
    for pipeline, rows in parameter_rows.items():
        parameter_summary[pipeline] = {
            "trainable_total_parameters": sorted({int(row["trainable_total_parameters"]) for row in rows}),
            "trainable_head_parameters": sorted({int(row["trainable_head_parameters"]) for row in rows}),
            "trainable_adaptation_parameters": sorted({int(row["trainable_adaptation_parameters"]) for row in rows}),
        }
    return {
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "completed_outer_fits": int(settings["total_outer_fits"]),
        "complete_oof": metrics_by_pipeline,
        "aggregate": aggregate,
        "paired": paired,
        "paired_summary": paired_summary,
        "paired_cat_bootstrap": bootstraps,
        "parameter_summary": parameter_summary,
        "seed_expansion_gates": gates,
        "artifacts": {
            "paired_prediction_changes": repo_relative(root / "paired_prediction_changes.csv")
        },
    }


def evaluate(
    protocol: dict[str, Any],
    store: Any,
    roles: pd.DataFrame,
    locked_protocol: dict[str, Any],
    device: torch.device,
    resume: bool,
) -> None:
    lock = verify_execution_lock(protocol)
    root = RUN_ROOT / "evaluation"
    summary_path = root / "summary.json"
    if summary_path.is_file() and not resume:
        raise FileExistsError(summary_path)
    root.mkdir(parents=True, exist_ok=True)
    write_json(
        root / "run_manifest.json",
        {
            "status": "running",
            "stage": "idea054_initial_evaluation",
            "outer_test_accessed": True,
            "code_commit": lock["code_commit"],
            "protocol_sha256": lock["protocol_sha256"],
            "runner_sha256": lock["runner_sha256"],
            "device": str(device),
        },
    )
    settings = protocol["initial_evaluation"]
    completed = 0
    for pipeline in PIPELINES:
        for base_seed in settings["base_seeds"]:
            for repeat in settings["repeats"]:
                for fold in settings["outer_folds"]:
                    output = (
                        root
                        / "fits"
                        / pipeline
                        / f"base_seed_{base_seed}"
                        / f"repeat_{repeat}"
                        / f"fold_{fold}"
                    )
                    fit_path = output / "fit_summary.json"
                    if fit_path.is_file():
                        if not resume:
                            raise FileExistsError(fit_path)
                        completed += 1
                        continue
                    seed = weighting.historical.full_seed(
                        int(base_seed), int(repeat), int(fold)
                    )
                    indices = weighting.historical.fold_indices(
                        store, roles, int(repeat), int(fold), include_test=True
                    )
                    print(
                        f"IDEA054 {pipeline} repeat={repeat} fold={fold} seed={seed}",
                        flush=True,
                    )
                    best_epoch, inner = fit_inner(
                        pipeline,
                        protocol,
                        locked_protocol,
                        store,
                        indices["train"],
                        indices["validation"],
                        device,
                        seed,
                    )
                    outer_train = np.concatenate(
                        (indices["train"], indices["validation"])
                    )
                    calls, outer = fit_outer(
                        pipeline,
                        protocol,
                        locked_protocol,
                        store,
                        outer_train,
                        indices["test"],
                        best_epoch,
                        device,
                        seed,
                    )
                    animals = evaluation.calls_to_animals(calls)
                    output.mkdir(parents=True, exist_ok=True)
                    calls.to_csv(output / "outer_test_call_predictions.csv", index=False)
                    animals.to_csv(output / "outer_test_animal_predictions.csv", index=False)
                    fit = {
                        "status": "complete",
                        "stage": "idea054_initial_evaluation",
                        "outer_test_accessed": True,
                        "pipeline": pipeline,
                        "base_seed": int(base_seed),
                        "repeat": int(repeat),
                        "outer_fold": int(fold),
                        "full_seed": int(seed),
                        "inner_train_calls": int(len(indices["train"])),
                        "inner_validation_calls": int(len(indices["validation"])),
                        "outer_test_calls": int(len(indices["test"])),
                        "selected_epoch": int(best_epoch),
                        "inner": inner,
                        "outer": outer,
                    }
                    write_json(fit_path, fit)
                    completed += 1
                    print(
                        f"IDEA054 {pipeline} repeat={repeat} fold={fold} "
                        f"F1={outer['test_animal_metrics']['macro_f1']:.4f}",
                        flush=True,
                    )
    summary = aggregate_results(protocol)
    inventory = evaluation.raw_prediction_inventory(root)
    inventory_path = root / "raw_prediction_inventory.json"
    write_json(inventory_path, inventory)
    summary["raw_prediction_inventory"] = {
        "path": repo_relative(inventory_path),
        "sha256": sha256(inventory_path),
        "files": inventory["files"],
        "bytes": inventory["bytes"],
        "aggregate_sha256": inventory["aggregate_sha256"],
    }
    summary["code_commit"] = lock["code_commit"]
    summary["execution_lock_sha256"] = sha256(RUN_ROOT / "execution_lock.json")
    summary["environment_lock_sha256"] = sha256(RUN_ROOT / "environment_lock.json")
    summary["runner_sha256"] = lock["runner_sha256"]
    summary["protocol_sha256"] = lock["protocol_sha256"]
    write_json(summary_path, summary)
    write_json(
        root / "run_summary.json",
        {
            "status": "complete",
            "completed_outer_fits": completed,
            "expected_outer_fits": int(settings["total_outer_fits"]),
            "summary_path": repo_relative(summary_path),
            "summary_sha256": sha256(summary_path),
            "raw_prediction_inventory_sha256": sha256(inventory_path),
            "raw_prediction_aggregate_sha256": inventory["aggregate_sha256"],
        },
    )
    write_json(
        root / "run_manifest.json",
        {
            "status": "complete",
            "stage": "idea054_initial_evaluation",
            "outer_test_accessed": True,
            "code_commit": lock["code_commit"],
            "protocol_sha256": lock["protocol_sha256"],
            "runner_sha256": lock["runner_sha256"],
            "device": str(device),
            "completed_outer_fits": completed,
            "summary_path": repo_relative(summary_path),
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    device = resolve_device(args.device)
    store, roles, locked_protocol = load_inputs()
    print(f"IDEA-054 stage={args.stage}; device={device}", flush=True)
    if args.stage == "smoke":
        smoke(protocol, store, roles, locked_protocol, device, args.resume)
    else:
        evaluate(protocol, store, roles, locked_protocol, device, args.resume)


if __name__ == "__main__":
    main()
