"""Run IDEA-080 final-block Q/V LoRA x fixed C1 factorial evaluation."""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_idea078_ast_prelast_special_token_age_injection as idea078  # noqa: E402
import run_meowagenet_idea079_spatial_patch_adapter as idea079  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea080_ast_last_block_lora_c1_factorial_v1.json"
)
PIPELINES = (
    "A0_frozen_tail",
    "C1_bounded_age",
    "L1_lastblock_qv_lora",
    "CL1_C1_plus_lastblock_qv_lora",
)
BASE_SEEDS = (59, 7031, 1855)
HIDDEN_SIZE = 768
HEAD_WIDTH = 128
AGE_FEATURES = 20
AGE_WIDTH = 60
C1_CAP = 0.25
RMS_EPSILON = 1.0e-8
LORA_RANK = 6
LORA_ALPHA = 6
LORA_SCALING = 1.0
LORA_DROPOUT = 0.0
LORA_PARAMETERS = 18_432
AGE_PARAMETERS = 9_068
HEAD_PARAMETERS = 99_075
EXPECTED_PARAMETERS = {
    PIPELINES[0]: 99_075,
    PIPELINES[1]: 108_143,
    PIPELINES[2]: 117_507,
    PIPELINES[3]: 126_575,
}
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "run"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output-subdir",
        default="meowagenet_idea080_ast_last_block_lora_c1_factorial_v1",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--director-authorized",
        action="store_true",
        help="Required for the formal run after an explicit research-director GO.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


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


def set_seed(seed: int) -> None:
    idea078.set_seed(int(seed))


def configure_determinism() -> None:
    idea078.configure_determinism()


def full_seed(base_seed: int, repeat: int, fold: int) -> int:
    return int(base_seed + 10_000 * repeat + 100 * fold)


def trainable_parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def states_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[key], right[key]) for key in left
    )


class LoRALinear(nn.Module):
    """Frozen linear projection plus the literature-locked low-rank update."""

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        if base.in_features != HIDDEN_SIZE or base.out_features != HIDDEN_SIZE:
            raise RuntimeError("IDEA-080 LoRA projection geometry changed")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_dropout = nn.Identity()
        self.lora_A = nn.Linear(HIDDEN_SIZE, LORA_RANK, bias=False)
        self.lora_B = nn.Linear(LORA_RANK, HIDDEN_SIZE, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.rank = LORA_RANK
        self.alpha = LORA_ALPHA
        self.scaling = LORA_SCALING

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.lora_B(self.lora_A(self.lora_dropout(inputs)))
        return self.base(inputs) + self.scaling * update


class FactorialASTTail(nn.Module):
    """Cached block-11 tokens through frozen block 12, optionally with Q/V LoRA."""

    def __init__(self, use_lora: bool, initialization_seed: int) -> None:
        super().__init__()
        block12, final_layernorm = idea079._tail_template()
        self.block12 = copy.deepcopy(block12)
        self.final_layernorm = copy.deepcopy(final_layernorm)
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.use_lora = bool(use_lora)
        if use_lora:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(initialization_seed))
                attention = self.block12.attention.attention
                attention.query = LoRALinear(attention.query)
                attention.value = LoRALinear(attention.value)

    def forward(self, tokens_after_block11: torch.Tensor) -> torch.Tensor:
        sequence = self.block12(tokens_after_block11, None, False)[0]
        sequence = self.final_layernorm(sequence)
        return (sequence[:, 0] + sequence[:, 1]) / 2.0

    def lora_state(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.state_dict().items()
            if ".lora_A." in key or ".lora_B." in key
        }

    def audit(self) -> dict[str, Any]:
        trainable_names = [
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        ]
        expected_suffixes = {
            "block12.attention.attention.query.lora_A.weight",
            "block12.attention.attention.query.lora_B.weight",
            "block12.attention.attention.value.lora_A.weight",
            "block12.attention.attention.value.lora_B.weight",
        }
        return {
            "use_lora": self.use_lora,
            "target_block_one_based": 12 if self.use_lora else None,
            "target_projections": ["query", "value"] if self.use_lora else [],
            "rank": LORA_RANK if self.use_lora else 0,
            "alpha": LORA_ALPHA if self.use_lora else 0,
            "scaling": LORA_SCALING if self.use_lora else 0.0,
            "dropout": LORA_DROPOUT if self.use_lora else 0.0,
            "trainable_parameters": trainable_parameter_count(self),
            "trainable_parameter_names": trainable_names,
            "target_scope_exact": set(trainable_names) == expected_suffixes if self.use_lora else not trainable_names,
            "frozen_parameters": int(
                sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad)
            ),
        }


class FactorialHead(nn.Module):
    """Shared classifier head with the frozen IDEA-076 C1 branch when requested."""

    def __init__(
        self,
        ast_mean: np.ndarray,
        ast_scale: np.ndarray,
        age_train: np.ndarray,
        dropout: float,
        use_age: bool,
        head_seed: int,
        age_seed: int,
    ) -> None:
        super().__init__()
        safe_scale = np.where(ast_scale > 1.0e-12, ast_scale, 1.0).astype(np.float32)
        self.register_buffer("ast_mean", torch.from_numpy(ast_mean.astype(np.float32)))
        self.register_buffer("ast_scale", torch.from_numpy(safe_scale))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(head_seed))
            self.ast_linear = nn.Linear(HIDDEN_SIZE, HEAD_WIDTH)
            self.relu = nn.ReLU()
            self.batch_norm = nn.BatchNorm1d(HEAD_WIDTH, eps=1.0e-3, momentum=0.01)
            self.dropout = nn.Dropout(dropout)
            self.output = nn.Linear(HEAD_WIDTH, 3)
        self.use_age = bool(use_age)
        if use_age:
            with np.errstate(all="ignore"):
                median = np.nanmedian(age_train, axis=0)
            median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
            imputed = np.where(np.isfinite(age_train), age_train, median[None, :])
            age_mean = imputed.mean(axis=0).astype(np.float32)
            age_scale = imputed.std(axis=0).astype(np.float32)
            age_scale = np.where(age_scale > 1.0e-8, age_scale, 1.0).astype(np.float32)
            self.register_buffer("age_median", torch.from_numpy(median))
            self.register_buffer("age_mean", torch.from_numpy(age_mean))
            self.register_buffer("age_scale", torch.from_numpy(age_scale))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(age_seed))
                self.age_hidden = nn.Linear(AGE_FEATURES, AGE_WIDTH)
                self.age_output = nn.Linear(AGE_WIDTH, HEAD_WIDTH)
            nn.init.zeros_(self.age_output.weight)
            nn.init.zeros_(self.age_output.bias)
        else:
            self.age_hidden = None
            self.age_output = None
        self.reset_perturbation_audit()

    def reset_perturbation_audit(self) -> None:
        self._perturbation_ratios: list[float] = []

    def shared_head_state(self) -> dict[str, torch.Tensor]:
        prefixes = ("ast_linear.", "batch_norm.", "output.")
        return {
            key: value.detach().cpu().clone()
            for key, value in self.state_dict().items()
            if key.startswith(prefixes)
        }

    def age_state(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.state_dict().items()
            if key.startswith(("age_hidden.", "age_output."))
        }

    def forward(self, ast_embeddings: torch.Tensor, age_features: torch.Tensor) -> torch.Tensor:
        ast = (ast_embeddings - self.ast_mean) / self.ast_scale
        hidden = self.relu(self.ast_linear(ast))
        if self.use_age:
            if self.age_hidden is None or self.age_output is None:
                raise RuntimeError("IDEA-080 C1 branch is missing")
            imputed = torch.where(torch.isfinite(age_features), age_features, self.age_median)
            standardized = (imputed - self.age_mean) / self.age_scale
            context = torch.nn.functional.gelu(self.age_hidden(standardized))
            anchor = (
                torch.sqrt(hidden.float().square().mean(dim=1, keepdim=True) + RMS_EPSILON)
                .detach()
                .to(hidden.dtype)
            )
            residual = C1_CAP * anchor * torch.tanh(self.age_output(context))
            if not self.training:
                with torch.no_grad():
                    ratio = torch.linalg.vector_norm(residual.float(), dim=1) / torch.linalg.vector_norm(
                        hidden.float(), dim=1
                    ).clamp_min(1.0e-12)
                    self._perturbation_ratios.extend(ratio.cpu().tolist())
            hidden = hidden + residual
        hidden = self.batch_norm(hidden)
        hidden = self.dropout(hidden)
        return self.output(hidden)

    def perturbation_audit(self) -> dict[str, Any]:
        values = np.asarray(self._perturbation_ratios, dtype=np.float64)
        if len(values) == 0:
            return {
                "validation_relative_perturbation_calls": 0,
                "validation_relative_perturbation_mean": None,
                "validation_relative_perturbation_min": None,
                "validation_relative_perturbation_max": None,
            }
        return {
            "validation_relative_perturbation_calls": int(len(values)),
            "validation_relative_perturbation_mean": float(values.mean()),
            "validation_relative_perturbation_min": float(values.min()),
            "validation_relative_perturbation_max": float(values.max()),
        }


class FactorialClassifier(nn.Module):
    def __init__(self, pipeline: str, head: FactorialHead, tail: FactorialASTTail) -> None:
        super().__init__()
        if pipeline not in PIPELINES:
            raise ValueError(pipeline)
        self.pipeline = pipeline
        self.head = head
        self.tail = tail

    def train(self, mode: bool = True) -> "FactorialClassifier":
        super().train(mode)
        self.tail.eval()
        return self

    def reset_perturbation_audit(self) -> None:
        self.head.reset_perturbation_audit()

    def forward(
        self,
        tokens: torch.Tensor,
        segment_to_local_call: torch.Tensor,
        age_features: torch.Tensor,
    ) -> torch.Tensor:
        segment_embeddings = self.tail(tokens)
        call_count = len(age_features)
        calls = torch.zeros(
            (call_count, HIDDEN_SIZE),
            dtype=segment_embeddings.dtype,
            device=segment_embeddings.device,
        )
        calls.index_add_(0, segment_to_local_call, segment_embeddings)
        counts = torch.bincount(segment_to_local_call, minlength=call_count).to(
            segment_embeddings.dtype
        )
        if torch.any(counts == 0):
            raise RuntimeError("IDEA-080 batch contains a call with no segment")
        return self.head(calls / counts[:, None], age_features)

    def audit(self) -> dict[str, Any]:
        result = {
            "pipeline": self.pipeline,
            "trainable_parameters": trainable_parameter_count(self),
            "uses_C1": self.pipeline in (PIPELINES[1], PIPELINES[3]),
            "uses_LoRA": self.pipeline in (PIPELINES[2], PIPELINES[3]),
            "C1_parameters": AGE_PARAMETERS if self.pipeline in (PIPELINES[1], PIPELINES[3]) else 0,
            "LoRA_parameters": trainable_parameter_count(self.tail),
            "C1_cap": C1_CAP if self.pipeline in (PIPELINES[1], PIPELINES[3]) else None,
            "C1_width": AGE_WIDTH if self.pipeline in (PIPELINES[1], PIPELINES[3]) else None,
            "tail": self.tail.audit(),
        }
        result.update(self.head.perturbation_audit())
        return result


def build_model(
    pipeline: str,
    protocol: dict[str, Any],
    store: idea078.TokenStore,
    train_indices: np.ndarray,
    seed: int,
) -> FactorialClassifier:
    fixed = protocol["fixed_training"]
    use_age = pipeline in (PIPELINES[1], PIPELINES[3])
    use_lora = pipeline in (PIPELINES[2], PIPELINES[3])
    final_train = store.frozen_embeddings[train_indices]
    head = FactorialHead(
        final_train.mean(axis=0),
        final_train.std(axis=0),
        store.age_features[train_indices],
        float(fixed["dropout"]),
        use_age,
        seed + int(fixed["head_initialization_seed_offset"]),
        seed + int(fixed["age_initialization_seed_offset"]),
    )
    tail = FactorialASTTail(
        use_lora,
        seed + int(fixed["lora_initialization_seed_offset"]),
    )
    return FactorialClassifier(pipeline, head, tail)


def head_state(model: FactorialClassifier) -> dict[str, torch.Tensor]:
    return model.head.shared_head_state()


def age_state(model: FactorialClassifier) -> dict[str, torch.Tensor]:
    return model.head.age_state()


def lora_state(model: FactorialClassifier) -> dict[str, torch.Tensor]:
    return model.tail.lora_state()


def checkpoint_state(model: FactorialClassifier) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("tail.") or ".lora_A." in key or ".lora_B." in key
    }


def restore_checkpoint(model: FactorialClassifier, state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"IDEA-080 unexpected checkpoint state: {unexpected}")
    forbidden_missing = [
        key
        for key in missing
        if not key.startswith("tail.") or ".lora_A." in key or ".lora_B." in key
    ]
    if forbidden_missing:
        raise RuntimeError(f"IDEA-080 checkpoint omitted trainable state: {forbidden_missing}")


def predict(
    model: FactorialClassifier,
    store: idea078.TokenStore,
    indices: np.ndarray,
    cat_batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.reset_perturbation_audit()
    return idea078.predict(model, store, indices, cat_batch_size, device, seed)


def fit_inner(
    pipeline: str,
    protocol: dict[str, Any],
    store: idea078.TokenStore,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    model = build_model(pipeline, protocol, store, train_indices, seed).to(device)
    if model.audit()["trainable_parameters"] != EXPECTED_PARAMETERS[pipeline]:
        raise RuntimeError("IDEA-080 trainable parameter count changed")
    fixed = protocol["fixed_training"]
    training_seed = seed + int(fixed["post_build_seed_offset"])
    set_seed(training_seed)
    optimizer = torch.optim.Adamax(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(fixed["learning_rate"]),
        eps=float(fixed["optimizer_epsilon"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loader = idea078.build_loader(
        store,
        train_indices,
        int(fixed["cat_batch_size"]),
        True,
        seed,
    )
    weights = torch.from_numpy(idea078.class_weights(store.labels[train_indices])).to(device)
    best_loss = float("inf")
    best_epoch = 1
    best_state = checkpoint_state(model)
    best_animals: pd.DataFrame | None = None
    best_calls: pd.DataFrame | None = None
    history: list[dict[str, Any]] = []
    stale = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(fixed["maximum_epochs"]) + 1):
        train_loss, train_audit = idea078.train_one_epoch(
            model,
            loader,
            optimizer,
            scaler,
            store,
            weights,
            device,
            float(fixed["gradient_clip"]),
        )
        animals, calls = predict(
            model,
            store,
            validation_indices,
            int(fixed["cat_batch_size"]) * 2,
            device,
            seed,
        )
        validation_loss = idea078.idea077.animal_cross_entropy(animals)
        metrics = idea078.idea077.animal_metrics(animals)
        history.append(
            {
                "epoch": epoch,
                "train_call_loss": train_loss,
                "train_audit": train_audit,
                "validation_animal_cross_entropy": validation_loss,
                "validation_animal_brier": idea078.idea077.brier(animals),
                "validation_animal_metrics": metrics,
                "model": model.audit(),
            }
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = checkpoint_state(model)
            best_animals = animals.copy()
            best_calls = calls.copy()
            stale = 0
        else:
            stale += 1
        print(
            f"{pipeline} epoch={epoch} call_CE={train_loss:.4f} "
            f"val_CE={validation_loss:.4f} val_F1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if stale >= int(fixed["early_stopping_patience"]):
            break
    if best_animals is None or best_calls is None:
        raise RuntimeError("IDEA-080 selected no checkpoint")
    restore_checkpoint(model, best_state)
    reload_animals, _ = predict(
        model,
        store,
        validation_indices,
        int(fixed["cat_batch_size"]) * 2,
        device,
        seed,
    )
    reload_difference = float(
        np.abs(
            reload_animals[list(PROBABILITY_COLUMNS)].to_numpy()
            - best_animals[list(PROBABILITY_COLUMNS)].to_numpy()
        ).max()
    )
    audit = {
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "full_seed": seed,
        "training_seed": training_seed,
        "best_validation_animal_cross_entropy": best_loss,
        "best_validation_animal_brier": idea078.idea077.brier(best_animals),
        "best_validation_animal_metrics": idea078.idea077.animal_metrics(best_animals),
        "checkpoint_reload_max_probability_difference": reload_difference,
        "best_model": model.audit(),
        "train_seconds": float(time.perf_counter() - started),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "history": history,
        "outer_test_accessed": False,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return audit, best_animals, best_calls


def _probe_batch(
    store: idea078.TokenStore,
    probe_calls: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    segment_parts = [store.call_segment_indices[int(index)] for index in probe_calls]
    segments = np.concatenate(segment_parts).astype(np.int64)
    mapping = np.repeat(
        np.arange(len(probe_calls), dtype=np.int64),
        np.asarray([len(part) for part in segment_parts], dtype=np.int64),
    )
    return (
        torch.from_numpy(np.asarray(store.prelast_tokens[segments]).copy()).to(device),
        torch.from_numpy(mapping).to(device),
        torch.from_numpy(store.age_features[probe_calls]).to(device),
        torch.from_numpy(store.labels[probe_calls]).to(device),
    )


def initialization_audit(
    protocol: dict[str, Any],
    store: idea078.TokenStore,
    train_indices: np.ndarray,
    probe_indices: np.ndarray,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    tokens, mapping, ages, labels = _probe_batch(store, probe_indices, device)
    models: dict[str, FactorialClassifier] = {}
    logits: dict[str, torch.Tensor] = {}
    for pipeline in PIPELINES:
        model = build_model(pipeline, protocol, store, train_indices, seed).to(device).eval()
        models[pipeline] = model
        with torch.no_grad():
            logits[pipeline] = model(tokens, mapping, ages)
    parameters = {pipeline: models[pipeline].audit()["trainable_parameters"] for pipeline in PIPELINES}
    head_equal = all(
        states_equal(head_state(models[PIPELINES[0]]), head_state(models[pipeline]))
        for pipeline in PIPELINES[1:]
    )
    age_equal = states_equal(age_state(models[PIPELINES[1]]), age_state(models[PIPELINES[3]]))
    lora_equal = states_equal(lora_state(models[PIPELINES[2]]), lora_state(models[PIPELINES[3]]))
    differences = {
        pipeline: float(torch.max(torch.abs(logits[pipeline] - logits[PIPELINES[0]])).cpu())
        for pipeline in PIPELINES[1:]
    }
    losses = {
        pipeline: float(torch.nn.functional.cross_entropy(logits[pipeline], labels).cpu())
        for pipeline in PIPELINES
    }
    if (
        parameters != EXPECTED_PARAMETERS
        or not head_equal
        or not age_equal
        or not lora_equal
        or any(value != 0.0 for value in differences.values())
        or len(set(losses.values())) != 1
    ):
        raise RuntimeError("IDEA-080 paired initialization audit failed")
    return {
        "trainable_parameters": parameters,
        "shared_head_state_equal": head_equal,
        "C1_CL1_age_state_equal": age_equal,
        "L1_CL1_lora_state_equal": lora_equal,
        "max_initial_logit_difference_vs_A0": differences,
        "initial_loss": losses,
        "lora_scope": models[PIPELINES[2]].tail.audit(),
        "outer_test_accessed": False,
    }


def _max_gradient(parameter: torch.Tensor | None) -> float:
    if parameter is None:
        return 0.0
    return float(parameter.detach().abs().max().cpu())


def gradient_reachability_audit(
    protocol: dict[str, Any],
    store: idea078.TokenStore,
    train_indices: np.ndarray,
    probe_indices: np.ndarray,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    tokens, mapping, ages, labels = _probe_batch(store, probe_indices, device)
    result: dict[str, Any] = {}
    for pipeline in (PIPELINES[1], PIPELINES[2], PIPELINES[3]):
        model = build_model(pipeline, protocol, store, train_indices, seed).to(device).eval()
        loss = torch.nn.functional.cross_entropy(model(tokens, mapping, ages), labels)
        loss.backward()
        query = model.tail.block12.attention.attention.query
        value = model.tail.block12.attention.attention.value
        lora_B_zero = max(
            _max_gradient(query.lora_B.weight.grad) if isinstance(query, LoRALinear) else 0.0,
            _max_gradient(value.lora_B.weight.grad) if isinstance(value, LoRALinear) else 0.0,
        )
        lora_A_zero = max(
            _max_gradient(query.lora_A.weight.grad) if isinstance(query, LoRALinear) else 0.0,
            _max_gradient(value.lora_A.weight.grad) if isinstance(value, LoRALinear) else 0.0,
        )
        age_output_zero = _max_gradient(
            model.head.age_output.weight.grad
            if model.head.age_output is not None
            else None
        )
        age_hidden_zero = _max_gradient(
            model.head.age_hidden.weight.grad
            if model.head.age_hidden is not None
            else None
        )
        frozen_grads = sum(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad and name.startswith("tail.")
        )
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            if isinstance(query, LoRALinear):
                query.lora_B.weight.fill_(1.0e-4)
                value.lora_B.weight.fill_(1.0e-4)
            if model.head.age_output is not None:
                model.head.age_output.weight.fill_(1.0e-4)
        second_loss = torch.nn.functional.cross_entropy(model(tokens, mapping, ages), labels)
        second_loss.backward()
        lora_A_after = max(
            _max_gradient(query.lora_A.weight.grad) if isinstance(query, LoRALinear) else 0.0,
            _max_gradient(value.lora_A.weight.grad) if isinstance(value, LoRALinear) else 0.0,
        )
        age_hidden_after = _max_gradient(
            model.head.age_hidden.weight.grad
            if model.head.age_hidden is not None
            else None
        )
        uses_age = pipeline in (PIPELINES[1], PIPELINES[3])
        uses_lora = pipeline in (PIPELINES[2], PIPELINES[3])
        if (
            (uses_lora and (lora_B_zero <= 0.0 or lora_A_zero != 0.0 or lora_A_after <= 0.0))
            or (uses_age and (age_output_zero <= 0.0 or age_hidden_zero != 0.0 or age_hidden_after <= 0.0))
            or frozen_grads != 0
        ):
            raise RuntimeError(f"IDEA-080 gradient reachability failed for {pipeline}")
        result[pipeline] = {
            "lora_B_max_gradient_at_zero": lora_B_zero,
            "lora_A_max_gradient_at_zero": lora_A_zero,
            "lora_A_max_gradient_after_B_probe": lora_A_after,
            "age_output_max_gradient_at_zero": age_output_zero,
            "age_hidden_max_gradient_at_zero": age_hidden_zero,
            "age_hidden_max_gradient_after_output_probe": age_hidden_after,
            "frozen_tail_parameters_with_grad": frozen_grads,
        }
    return result


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea080-ast-last-block-lora-c1-factorial-v1":
        raise RuntimeError("Unexpected IDEA-080 protocol")
    if protocol.get("status") != "locked_before_formal_evaluation":
        raise RuntimeError("IDEA-080 protocol is not locked")
    model = protocol["model"]
    if tuple(model["pipelines"]) != PIPELINES or tuple(model["base_seeds"]) != BASE_SEEDS:
        raise RuntimeError("IDEA-080 pipelines or seeds changed")
    if model["trainable_parameters"] != EXPECTED_PARAMETERS:
        raise RuntimeError("IDEA-080 trainable parameter lock changed")
    if int(model["total_fits"]) != 144 or model["outer_test_predictions"] is not False:
        raise RuntimeError("IDEA-080 fit scope changed")
    lora = model["LoRA"]
    if (
        lora["target_block_one_based"] != 12
        or lora["target_projections"] != ["query", "value"]
        or lora["rank"] != LORA_RANK
        or lora["alpha"] != LORA_ALPHA
        or lora["scaling"] != LORA_SCALING
        or lora["dropout"] != LORA_DROPOUT
        or lora["parameters"] != LORA_PARAMETERS
    ):
        raise RuntimeError("IDEA-080 LoRA literature lock changed")
    c1 = model["C1"]
    if c1["hidden_width"] != AGE_WIDTH or c1["cap"] != C1_CAP or c1["parameters"] != AGE_PARAMETERS:
        raise RuntimeError("IDEA-080 C1 lock changed")
    for key in ("literature_report", "plan", "runner", "tests"):
        path = Path(protocol["dependencies"][f"{key}_path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        expected = protocol["dependencies"][f"{key}_sha256"]
        if expected == "PENDING" or not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"IDEA-080 dependency hash mismatch: {key}")
    for item in ("petl_ast_paper", "soft_mixture_ast_context", "ast_paper"):
        record = protocol["literature_lock"][item]
        path = Path(record["local_pdf_path"])
        if not path.is_file() or sha256(path) != record["local_pdf_sha256"]:
            raise RuntimeError(f"IDEA-080 literature PDF mismatch: {item}")


def resolve_run_root(output_subdir: str) -> Path:
    runs = (REPO_ROOT / "runs").resolve()
    root = (runs / output_subdir).resolve()
    if runs not in root.parents:
        raise ValueError("--output-subdir must stay below runs")
    return root


def cpu_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cpu":
        raise RuntimeError("IDEA-080 preflight is CPU-only")
    if torch.cuda.is_initialized():
        raise RuntimeError("IDEA-080 CPU preflight must not initialize CUDA")
    configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    source_audit = idea079.audit_source_fbank(protocol)
    store, cache_manifest = idea078.load_store(protocol)
    roles = pd.read_csv(REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    role_cells = idea078.validate_roles(protocol, store.call_ids, store.cat_ids, roles)
    indices = idea078.fold_indices(store, roles, 0, 0)
    probe_indices = indices["validation"][: min(8, len(indices["validation"]))]
    started = time.perf_counter()
    reconstruction = idea078.full_cache_reconstruction_audit(
        protocol,
        store,
        FactorialASTTail(False, BASE_SEEDS[0]).eval(),
    )
    initialization = initialization_audit(
        protocol,
        store,
        indices["train"],
        probe_indices,
        BASE_SEEDS[0],
        torch.device("cpu"),
    )
    gradients = gradient_reachability_audit(
        protocol,
        store,
        indices["train"],
        probe_indices,
        BASE_SEEDS[0],
        torch.device("cpu"),
    )
    result = {
        "status": "GO_FOR_FORMAL_GPU_RUN",
        "read_only": True,
        "device": "cpu",
        "gpu_used": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "literature_report_sha256": protocol["dependencies"]["literature_report_sha256"],
        "plan_sha256": protocol["dependencies"]["plan_sha256"],
        "source_audit": source_audit,
        "cache": {
            "read_only": True,
            "shape": list(store.prelast_tokens.shape),
            "dtype": str(store.prelast_tokens.dtype),
            "manifest_sha256": sha256(REPO_ROOT / protocol["cache"]["manifest_path"]),
            "token_sha256": cache_manifest["token_sha256"],
            "index_sha256": cache_manifest["index_sha256"],
        },
        "full_cache_A0_reconstruction": reconstruction,
        "role_cells": role_cells,
        "probe_calls": int(len(probe_indices)),
        "initialization": initialization,
        "gradient_reachability": gradients,
        "base_seeds": list(BASE_SEEDS),
        "unique_full_seeds": 36,
        "expected_fits": 144,
        "outer_test_accessed": False,
        "cpu_seconds": float(time.perf_counter() - started),
        "formal_GPU_run_authorized_by_preflight": True,
        "formal_GPU_run_still_requires_research_director_authorization": True,
    }
    output = resolve_run_root(args.output_subdir) / "cpu_preflight.json"
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    if output.exists():
        existing = read_json(output)
        stable = dict(result)
        prior = dict(existing)
        stable.pop("cpu_seconds", None)
        prior.pop("cpu_seconds", None)
        if stable != prior:
            raise RuntimeError("IDEA-080 resumed CPU preflight differs")
        result = existing
    else:
        write_json(output, result)
    return result


def metric_bundle(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "metrics": idea078.idea077.animal_metrics(frame),
        "cross_entropy": idea078.idea077.animal_cross_entropy(frame),
        "brier": idea078.idea077.brier(frame),
    }


def add_metrics(row: dict[str, Any], bundles: dict[str, Any]) -> None:
    for pipeline in PIPELINES:
        row[f"{pipeline}_macro_f1"] = bundles[pipeline]["metrics"]["macro_f1"]
        row[f"{pipeline}_balanced_accuracy"] = bundles[pipeline]["metrics"]["balanced_accuracy"]
        row[f"{pipeline}_cross_entropy"] = bundles[pipeline]["cross_entropy"]
        row[f"{pipeline}_brier"] = bundles[pipeline]["brier"]
        row[f"{pipeline}_senior_recall"] = bundles[pipeline]["metrics"]["per_class"]["senior"]["recall"]
    row["L1_minus_A0_macro_f1"] = row[f"{PIPELINES[2]}_macro_f1"] - row[f"{PIPELINES[0]}_macro_f1"]
    row["CL1_minus_C1_macro_f1"] = row[f"{PIPELINES[3]}_macro_f1"] - row[f"{PIPELINES[1]}_macro_f1"]
    row["C1_minus_A0_macro_f1"] = row[f"{PIPELINES[1]}_macro_f1"] - row[f"{PIPELINES[0]}_macro_f1"]
    row["CL1_minus_L1_macro_f1"] = row[f"{PIPELINES[3]}_macro_f1"] - row[f"{PIPELINES[2]}_macro_f1"]
    row["interaction_macro_f1"] = row["CL1_minus_C1_macro_f1"] - row["L1_minus_A0_macro_f1"]


def describe(values: pd.Series) -> dict[str, Any]:
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "median": float(values.median()),
        "positive": int((values > 0).sum()),
        "tied": int((values == 0).sum()),
        "negative": int((values < 0).sum()),
        "worst": float(values.min()),
        "best": float(values.max()),
    }


def _effect_gate(
    contrast: str,
    candidate: str,
    reference: str,
    seed_repeats: pd.DataFrame,
    split_cells: pd.DataFrame,
    per_seed: dict[str, dict[str, float]],
    per_seed_senior: dict[str, dict[str, float]],
    means: dict[str, dict[str, float]],
    gate: dict[str, Any],
) -> dict[str, bool]:
    delta = seed_repeats[contrast]
    split = split_cells[contrast]
    return {
        "mean_macro_f1_gain": float(delta.mean()) >= float(gate["minimum_mean_macro_f1"]),
        "positive_base_seed_means": sum(value > 0 for value in per_seed[contrast].values()) >= int(gate["minimum_positive_base_seeds"]),
        "positive_seed_repeats": int((delta > 0).sum()) >= int(gate["minimum_positive_seed_repeats"]),
        "nonnegative_split_cells": int((split >= 0).sum()) >= int(gate["minimum_nonnegative_split_cells"]),
        "worst_split_cell": float(split.min()) >= float(gate["minimum_worst_split_cell_delta"]),
        "cross_entropy_nonworse": means["cross_entropy"][candidate] <= means["cross_entropy"][reference],
        "brier_nonworse": means["brier"][candidate] <= means["brier"][reference],
        "per_base_seed_senior_recall": all(
            value >= float(gate["minimum_per_base_seed_senior_recall_delta"])
            for value in per_seed_senior[contrast].values()
        ),
    }


def aggregate(fits: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    by_key = {
        (fit["pipeline"], fit["base_seed"], fit["repeat"], fit["fold"]): fit
        for fit in fits
    }
    fold_rows: list[dict[str, Any]] = []
    seed_repeat_rows: list[dict[str, Any]] = []
    pooled_all: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
    for base_seed in BASE_SEEDS:
        for repeat in range(3):
            grouped: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
            for fold in range(4):
                frames = {
                    pipeline: pd.read_csv(
                        REPO_ROOT / by_key[(pipeline, base_seed, repeat, fold)]["validation_animal_predictions"],
                        dtype={"cat_id": str},
                    )
                    for pipeline in PIPELINES
                }
                row: dict[str, Any] = {"base_seed": base_seed, "repeat": repeat, "fold": fold}
                add_metrics(row, {pipeline: metric_bundle(frame) for pipeline, frame in frames.items()})
                fold_rows.append(row)
                for pipeline, frame in frames.items():
                    tagged = frame.copy()
                    tagged["base_seed"] = base_seed
                    tagged["repeat"] = repeat
                    tagged["fold"] = fold
                    grouped[pipeline].append(tagged)
                    pooled_all[pipeline].append(tagged)
            pooled = {pipeline: pd.concat(parts, ignore_index=True) for pipeline, parts in grouped.items()}
            row = {"base_seed": base_seed, "repeat": repeat}
            add_metrics(row, {pipeline: metric_bundle(frame) for pipeline, frame in pooled.items()})
            seed_repeat_rows.append(row)
    folds = pd.DataFrame(fold_rows)
    seed_repeats = pd.DataFrame(seed_repeat_rows)
    contrasts = (
        "L1_minus_A0_macro_f1",
        "CL1_minus_C1_macro_f1",
        "C1_minus_A0_macro_f1",
        "CL1_minus_L1_macro_f1",
        "interaction_macro_f1",
    )
    splits = folds.groupby(["repeat", "fold"], as_index=False)[list(contrasts)].mean()
    means = {
        metric: {
            pipeline: float(seed_repeats[f"{pipeline}_{metric}"].mean())
            for pipeline in PIPELINES
        }
        for metric in ("macro_f1", "balanced_accuracy", "cross_entropy", "brier")
    }
    per_seed = {
        contrast: {
            str(seed): float(seed_repeats[seed_repeats["base_seed"] == seed][contrast].mean())
            for seed in BASE_SEEDS
        }
        for contrast in contrasts
    }
    pooled_frames = {pipeline: pd.concat(parts, ignore_index=True) for pipeline, parts in pooled_all.items()}
    senior_pairs = {
        "L1_minus_A0_macro_f1": (PIPELINES[2], PIPELINES[0]),
        "CL1_minus_C1_macro_f1": (PIPELINES[3], PIPELINES[1]),
    }
    per_seed_senior: dict[str, dict[str, float]] = {key: {} for key in senior_pairs}
    for seed in BASE_SEEDS:
        senior = {
            pipeline: idea078.idea077.animal_metrics(frame[frame["base_seed"] == seed])["per_class"]["senior"]["recall"]
            for pipeline, frame in pooled_frames.items()
        }
        for contrast, (candidate, reference) in senior_pairs.items():
            per_seed_senior[contrast][str(seed)] = float(senior[candidate] - senior[reference])
    gate = protocol["gate"]
    main_conditions = _effect_gate(
        "L1_minus_A0_macro_f1", PIPELINES[2], PIPELINES[0], seed_repeats, splits,
        per_seed, per_seed_senior, means, gate["main_L1_minus_A0"],
    )
    combination_conditions = _effect_gate(
        "CL1_minus_C1_macro_f1", PIPELINES[3], PIPELINES[1], seed_repeats, splits,
        per_seed, per_seed_senior, means, gate["combination_CL1_minus_C1"],
    )
    interaction = seed_repeats["interaction_macro_f1"]
    interaction_split = splits["interaction_macro_f1"]
    interaction_gate = gate["positive_interaction"]
    interaction_conditions = {
        "mean_strictly_positive": float(interaction.mean()) > 0.0,
        "positive_base_seed_means": sum(value > 0 for value in per_seed["interaction_macro_f1"].values()) >= int(interaction_gate["minimum_positive_base_seeds"]),
        "positive_seed_repeats": int((interaction > 0).sum()) >= int(interaction_gate["minimum_positive_seed_repeats"]),
        "nonnegative_split_cells": int((interaction_split >= 0).sum()) >= int(interaction_gate["minimum_nonnegative_split_cells"]),
        "worst_split_cell": float(interaction_split.min()) >= float(interaction_gate["minimum_worst_split_cell_delta"]),
    }
    main_passed = bool(all(main_conditions.values()))
    combination_passed = bool(all(combination_conditions.values()))
    interaction_raw = bool(all(interaction_conditions.values()))
    return {
        "status": "complete",
        "outer_test_accessed": False,
        "fits": len(fits),
        "seed_repeat_estimates": len(seed_repeat_rows),
        "split_cell_estimates": int(len(splits)),
        "independence_note": "Repeated validation animals across seeds and repeats are descriptive occurrences, not independent samples.",
        "fold_results": fold_rows,
        "seed_repeat_results": seed_repeat_rows,
        "split_cell_results": splits.to_dict(orient="records"),
        "seed_repeat_equal_weight_means": means,
        "comparisons": {contrast: describe(seed_repeats[contrast]) for contrast in contrasts},
        "per_base_seed_mean_deltas": per_seed,
        "per_base_seed_senior_recall": per_seed_senior,
        "pooled_validation": {pipeline: metric_bundle(frame) for pipeline, frame in pooled_frames.items()},
        "main_gate_conditions": main_conditions,
        "main_gate_passed": main_passed,
        "combination_gate_conditions": combination_conditions,
        "combination_gate_passed": combination_passed,
        "interaction_gate_conditions": interaction_conditions,
        "interaction_gate_raw_passed": interaction_raw,
        "interaction_gate_interpretable": bool(main_passed and combination_passed),
        "interaction_gate_passed": bool(main_passed and combination_passed and interaction_raw),
        "factorial_gate_passed": bool(main_passed and combination_passed),
    }


def validate_completed_fit(
    fit: dict[str, Any],
    pipeline: str,
    base_seed: int,
    seed: int,
    repeat: int,
    fold: int,
) -> None:
    expected = {
        "status": "complete",
        "pipeline": pipeline,
        "base_seed": base_seed,
        "full_seed": seed,
        "repeat": repeat,
        "fold": fold,
        "outer_test_accessed": False,
    }
    if any(fit.get(key) != value for key, value in expected.items()):
        raise RuntimeError("IDEA-080 resume fit identity mismatch")
    for prefix in ("validation_animal", "validation_call"):
        path = REPO_ROOT / fit[f"{prefix}_predictions"]
        if not path.is_file() or sha256(path) != fit[f"{prefix}_sha256"]:
            raise RuntimeError(f"IDEA-080 resume prediction mismatch: {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.director_authorized:
        raise RuntimeError("IDEA-080 formal run requires explicit research-director authorization")
    configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    preflight_path = REPO_ROOT / protocol["outputs"]["cpu_preflight"]
    if not preflight_path.is_file() or read_json(preflight_path).get("status") != "GO_FOR_FORMAL_GPU_RUN":
        raise RuntimeError("IDEA-080 formal run requires the locked CPU preflight")
    store, cache_manifest = idea078.load_store(protocol)
    roles = pd.read_csv(REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    idea078.validate_roles(protocol, store.call_ids, store.cat_ids, roles)
    device = idea078.resolve_device(args.device)
    root = resolve_run_root(args.output_subdir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "git_revision": git_revision(),
        "outer_test_accessed": False,
        "model": protocol["model"],
        "cache_manifest_sha256": sha256(REPO_ROOT / protocol["cache"]["manifest_path"]),
        "cache_token_sha256": cache_manifest["token_sha256"],
        "cache_index_sha256": cache_manifest["index_sha256"],
        "cpu_preflight_sha256": sha256(preflight_path),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        if not args.resume or read_json(manifest_path) != manifest:
            raise RuntimeError("Existing IDEA-080 run manifest differs")
    else:
        write_json(manifest_path, manifest)
    completed: list[dict[str, Any]] = []
    for base_seed in BASE_SEEDS:
        for repeat in range(3):
            for fold in range(4):
                indices = idea078.fold_indices(store, roles, repeat, fold)
                seed = full_seed(base_seed, repeat, fold)
                init = initialization_audit(
                    protocol,
                    store,
                    indices["train"],
                    indices["validation"][: min(8, len(indices["validation"]))],
                    seed,
                    device,
                )
                quartet: list[dict[str, Any]] = []
                for pipeline in PIPELINES:
                    output = root / "fits" / pipeline / f"base_seed_{base_seed}" / f"repeat_{repeat}" / f"fold_{fold}"
                    summary_path = output / "fit_summary.json"
                    if summary_path.exists():
                        if not args.resume:
                            raise FileExistsError(summary_path)
                        fit = read_json(summary_path)
                        validate_completed_fit(fit, pipeline, base_seed, seed, repeat, fold)
                    else:
                        output.mkdir(parents=True, exist_ok=True)
                        print(
                            f"=== {pipeline} base_seed={base_seed} repeat={repeat} fold={fold} full_seed={seed} ===",
                            flush=True,
                        )
                        audit, animals, calls = fit_inner(
                            pipeline,
                            protocol,
                            store,
                            indices["train"],
                            indices["validation"],
                            device,
                            seed,
                        )
                        animal_path = output / "validation_animal_predictions.csv"
                        call_path = output / "validation_call_predictions.csv"
                        animals.to_csv(animal_path, index=False, lineterminator="\n")
                        calls.to_csv(call_path, index=False, lineterminator="\n")
                        fit = {
                            "status": "complete",
                            "pipeline": pipeline,
                            "base_seed": int(base_seed),
                            "full_seed": seed,
                            "repeat": repeat,
                            "fold": fold,
                            "outer_test_accessed": False,
                            "initialization_audit": init,
                            "audit": audit,
                            "validation_animal_predictions": repo_relative(animal_path),
                            "validation_animal_sha256": sha256(animal_path),
                            "validation_call_predictions": repo_relative(call_path),
                            "validation_call_sha256": sha256(call_path),
                        }
                        write_json(summary_path, fit)
                    completed.append(fit)
                    quartet.append(fit)
                common_epochs = min(len(item["audit"]["history"]) for item in quartet)
                for epoch in range(common_epochs):
                    reference = quartet[0]["audit"]["history"][epoch]["train_audit"]
                    for candidate in quartet[1:]:
                        current = candidate["audit"]["history"][epoch]["train_audit"]
                        for field in ("cat_order_sha256", "call_coverage_sha256", "segment_coverage_sha256"):
                            if current[field] != reference[field]:
                                raise RuntimeError("IDEA-080 paired batch coverage differs")
    summary = aggregate(completed, protocol)
    summary_path = root / "initial_evaluation_summary.json"
    expected_bytes = canonical_json_bytes(summary)
    if summary_path.exists():
        if not args.resume or read_json(summary_path) != summary or summary_path.read_bytes() != expected_bytes:
            raise RuntimeError("IDEA-080 resumed aggregate mismatch")
    else:
        write_json(summary_path, summary)
    compact = {
        "status": "complete",
        "completed_fits": len(completed),
        "expected_fits": 144,
        "outer_test_accessed": False,
        "main_gate_passed": summary["main_gate_passed"],
        "combination_gate_passed": summary["combination_gate_passed"],
        "interaction_gate_interpretable": summary["interaction_gate_interpretable"],
        "interaction_gate_passed": summary["interaction_gate_passed"],
        "factorial_gate_passed": summary["factorial_gate_passed"],
    }
    compact_path = root / "run_summary.json"
    if compact_path.exists():
        if read_json(compact_path) != compact or compact_path.read_bytes() != canonical_json_bytes(compact):
            raise RuntimeError("IDEA-080 resumed compact summary mismatch")
    else:
        write_json(compact_path, compact)
    return summary


def main() -> None:
    args = parse_args()
    result = cpu_preflight(args) if args.stage == "preflight" else run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
