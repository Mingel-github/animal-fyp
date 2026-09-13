"""Run IDEA-052 zero-initialized local residuals over frozen AST features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for local_root in (SCRIPTS_ROOT, SRC_ROOT):
    if str(local_root) not in sys.path:
        sys.path.insert(0, str(local_root))

import run_meowagenet_idea051_cat_set as previous  # noqa: E402


reference = previous.reference
PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea052_ast_local_residual_v1.json"
)
PLAN_PATH = REPO_ROOT / "plan" / "IDEA-052_AST_local_acoustic_residual.md"
DIAGNOSTIC_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_idea052_local_residual_diagnostics_v1.json"
)
DIAGNOSTIC_SCRIPT_PATH = SCRIPTS_ROOT / "diagnose_meowagenet_idea052_local_residual.py"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
REFERENCE_RUNNER_PATH = SCRIPTS_ROOT / "run_meowagenet_idea051_cat_set.py"
GLOBAL_EMBEDDING_PATH = (
    REPO_ROOT
    / "runs"
    / "ast_locked_v1"
    / "gpu_rerun_2026-08-26"
    / "ast_standard_call_embeddings.npz"
)
TEMPORAL_TOKEN_PATH = (
    REPO_ROOT / "runs" / "ast_temporal_tokens_v1" / "ast_standard_temporal_tokens.npz"
)
RUNS_ROOT = REPO_ROOT / "runs"
PIPELINES = (
    "R0_global_probability_mean",
    "R1_temporal_mean_residual",
    "R2_temporal_salience_residual",
)
RESIDUAL_PIPELINES = PIPELINES[1:]
LABEL_NAMES = previous.LABEL_NAMES
PROBABILITY_COLUMNS = previous.PROBABILITY_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "evaluate"), required=True)
    parser.add_argument(
        "--output-subdir", default="meowagenet_idea052_ast_local_residual_v1"
    )
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


def git_blob_object_id(revision: str, path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{revision}:{repo_relative(path)}"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def worktree_blob_object_id(path: Path) -> str:
    return subprocess.check_output(
        ["git", "hash-object", repo_relative(path)], cwd=REPO_ROOT, text=True
    ).strip()


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol["protocol_id"] != "meowagenet-idea052-ast-local-residual-v1":
        raise RuntimeError("Unexpected IDEA-052 protocol")
    if tuple(protocol["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-052 pipeline matrix changed")
    checks = {
        PLAN_PATH: protocol["idea"]["sha256"],
        DIAGNOSTIC_PATH: protocol["diagnostic"]["sha256"],
        DIAGNOSTIC_SCRIPT_PATH: protocol["diagnostic"]["script_sha256"],
        ROLES_PATH: protocol["splits"]["roles_sha256"],
        GLOBAL_EMBEDDING_PATH: protocol["dependencies"]["global_embedding_sha256"],
        TEMPORAL_TOKEN_PATH: protocol["dependencies"]["temporal_token_sha256"],
        REFERENCE_RUNNER_PATH: protocol["dependencies"]["reference_runner_sha256"],
    }
    for path, expected in checks.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"IDEA-052 dependency checksum mismatch: {path}")
    evaluation = protocol["initial_evaluation"]
    expected_fits = (
        len(PIPELINES)
        * len(evaluation["base_seeds"])
        * len(evaluation["repeats"])
        * len(evaluation["outer_folds"])
    )
    if expected_fits != int(evaluation["total_outer_fits"]):
        raise RuntimeError("IDEA-052 fit budget is inconsistent")
    fixed = protocol["fixed_training"]
    if int(fixed["call_micro_batch_size"]) * int(
        fixed["gradient_accumulation_steps"]
    ) != int(fixed["accumulation_window_calls"]):
        raise RuntimeError("IDEA-052 accumulation window is inconsistent")


@dataclass(frozen=True)
class ResidualStore:
    global_embeddings: np.ndarray
    temporal_tokens: np.ndarray
    call_token_indices: tuple[np.ndarray, ...]
    call_ids: np.ndarray
    cat_ids: np.ndarray
    labels: np.ndarray
    durations: np.ndarray


def load_store() -> ResidualStore:
    global_data = np.load(GLOBAL_EMBEDDING_PATH)
    token_data = np.load(TEMPORAL_TOKEN_PATH)
    for field in ("call_ids", "cat_ids", "labels"):
        if not np.array_equal(global_data[field].astype(str), token_data[field].astype(str)):
            raise RuntimeError(f"Global/token {field} order differs")
    token_call_indices = token_data["token_call_indices"].astype(np.int64)
    call_token_indices = tuple(
        np.flatnonzero(token_call_indices == call_index).astype(np.int64)
        for call_index in range(len(global_data["call_ids"]))
    )
    if any(len(indices) == 0 for indices in call_token_indices):
        raise RuntimeError("At least one call has no temporal token")
    return ResidualStore(
        global_embeddings=global_data["embeddings"].astype(np.float32),
        temporal_tokens=token_data["temporal_tokens"].astype(np.float32),
        call_token_indices=call_token_indices,
        call_ids=global_data["call_ids"].astype(str),
        cat_ids=global_data["cat_ids"].astype(str),
        labels=global_data["labels"].astype(np.int64),
        durations=global_data["durations"].astype(np.float32),
    )


class ResidualCallDataset(Dataset[tuple[np.ndarray, np.ndarray, int, int]]):
    def __init__(self, store: ResidualStore, call_indices: np.ndarray) -> None:
        self.store = store
        self.call_indices = np.asarray(call_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.call_indices)

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray, int, int]:
        call_index = int(self.call_indices[item])
        return (
            self.store.global_embeddings[call_index],
            self.store.temporal_tokens[self.store.call_token_indices[call_index]],
            int(self.store.labels[call_index]),
            call_index,
        )


def collate_residual_calls(
    rows: list[tuple[np.ndarray, np.ndarray, int, int]],
) -> dict[str, torch.Tensor]:
    global_embeddings = []
    temporal_tokens = []
    token_to_call = []
    labels = []
    call_indices = []
    token_counts = []
    for local_call, (global_embedding, call_tokens, label, call_index) in enumerate(rows):
        global_embeddings.append(torch.from_numpy(global_embedding))
        temporal_tokens.append(torch.from_numpy(call_tokens))
        token_to_call.extend([local_call] * len(call_tokens))
        labels.append(label)
        call_indices.append(call_index)
        token_counts.append(len(call_tokens))
    return {
        "global_embeddings": torch.stack(global_embeddings),
        "temporal_tokens": torch.cat(temporal_tokens, dim=0),
        "token_to_call": torch.tensor(token_to_call, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "call_indices": torch.tensor(call_indices, dtype=torch.long),
        "token_counts": torch.tensor(token_counts, dtype=torch.long),
    }


def build_loader(
    store: ResidualStore,
    indices: np.ndarray,
    batch_size: int,
    training: bool,
    seed: int,
) -> DataLoader:
    dataset = ResidualCallDataset(store, indices)
    if training:
        sampler = reference.DeterministicNoSingletonBatchSampler(
            len(dataset), batch_size, seed
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collate_residual_calls,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_residual_calls,
    )


def safe_scale(values: np.ndarray) -> np.ndarray:
    scale = values.std(axis=0)
    return np.where(scale > 1.0e-12, scale, 1.0).astype(np.float32)


class GlobalLocalResidualClassifier(nn.Module):
    def __init__(
        self,
        global_mean: np.ndarray,
        global_scale: np.ndarray,
        token_mean: np.ndarray,
        token_scale: np.ndarray,
        dropout: float,
        residual_mode: str,
    ) -> None:
        super().__init__()
        if residual_mode not in ("none", "mean", "salience"):
            raise ValueError(residual_mode)
        self.residual_mode = residual_mode
        self.register_buffer(
            "global_mean", torch.from_numpy(global_mean.astype(np.float32))
        )
        self.register_buffer(
            "global_scale", torch.from_numpy(global_scale.astype(np.float32))
        )
        self.register_buffer("token_mean", torch.from_numpy(token_mean.astype(np.float32)))
        self.register_buffer(
            "token_scale", torch.from_numpy(token_scale.astype(np.float32))
        )
        self.projection = nn.Linear(768, 128)
        self.batch_norm = nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, 3)
        if residual_mode != "none":
            self.residual_gate = nn.Parameter(torch.zeros(128))
        else:
            self.register_parameter("residual_gate", None)

    def hidden_and_residual(
        self,
        global_embeddings: torch.Tensor,
        temporal_tokens: torch.Tensor,
        token_to_call: torch.Tensor,
        call_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        standardized_global = (
            global_embeddings - self.global_mean
        ) / self.global_scale
        global_hidden = torch.relu(self.projection(standardized_global))
        if self.residual_mode == "none":
            return global_hidden, torch.zeros_like(global_hidden)
        standardized_tokens = (temporal_tokens - self.token_mean) / self.token_scale
        token_hidden = torch.relu(self.projection(standardized_tokens))
        token_sum = torch.zeros(
            call_count,
            token_hidden.shape[1],
            dtype=token_hidden.dtype,
            device=token_hidden.device,
        )
        token_sum.index_add_(0, token_to_call, token_hidden)
        counts = torch.bincount(token_to_call, minlength=call_count).clamp_min(1)
        token_mean_hidden = token_sum / counts[:, None]
        if self.residual_mode == "mean":
            residual = token_mean_hidden - global_hidden
        else:
            token_max_hidden = torch.stack(
                [token_hidden[token_to_call == index].max(dim=0).values for index in range(call_count)]
            )
            residual = token_max_hidden - token_mean_hidden
        gate = torch.tanh(self.residual_gate)[None, :]
        return global_hidden + gate * residual, residual

    def forward(
        self,
        global_embeddings: torch.Tensor,
        temporal_tokens: torch.Tensor,
        token_to_call: torch.Tensor,
        call_count: int,
    ) -> torch.Tensor:
        hidden, _ = self.hidden_and_residual(
            global_embeddings, temporal_tokens, token_to_call, call_count
        )
        return self.classifier(self.dropout(self.batch_norm(hidden)))

    def gate_summary(self) -> dict[str, float] | None:
        if self.residual_gate is None:
            return None
        values = torch.tanh(self.residual_gate.detach()).float().cpu().numpy()
        return {
            "mean_abs_gate": float(np.abs(values).mean()),
            "max_abs_gate": float(np.abs(values).max()),
            "rms_gate": float(np.sqrt(np.mean(values**2))),
            "positive_fraction": float(np.mean(values > 0)),
            "active_fraction_above_1e_3": float(np.mean(np.abs(values) > 1.0e-3)),
        }


def training_token_indices(store: ResidualStore, train_indices: np.ndarray) -> np.ndarray:
    return np.concatenate([store.call_token_indices[int(index)] for index in train_indices])


def build_model(
    pipeline: str,
    protocol: dict[str, Any],
    store: ResidualStore,
    train_indices: np.ndarray,
) -> GlobalLocalResidualClassifier:
    global_values = store.global_embeddings[train_indices]
    token_values = store.temporal_tokens[training_token_indices(store, train_indices)]
    modes = {
        PIPELINES[0]: "none",
        PIPELINES[1]: "mean",
        PIPELINES[2]: "salience",
    }
    return GlobalLocalResidualClassifier(
        global_values.mean(axis=0),
        safe_scale(global_values),
        token_values.mean(axis=0),
        safe_scale(token_values),
        float(protocol["fixed_training"]["dropout"]),
        modes[pipeline],
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def training_values(protocol: dict[str, Any]) -> dict[str, Any]:
    fixed = protocol["fixed_training"]
    return {
        "learning_rate": float(fixed["learning_rate"]),
        "optimizer_epsilon": float(fixed["optimizer_epsilon"]),
        "gradient_clip": float(fixed["gradient_clip"]),
        "max_epochs": int(fixed["maximum_epochs"]),
        "patience": int(fixed["early_stopping_patience"]),
        "batch_size": int(fixed["call_micro_batch_size"]),
        "accumulation_steps": int(fixed["gradient_accumulation_steps"]),
        "accumulation_window_calls": int(fixed["accumulation_window_calls"]),
    }


def train_one_epoch(
    model: GlobalLocalResidualClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    weight_lookup: torch.Tensor,
    weight_lookup_numpy: np.ndarray,
    store: ResidualStore,
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
    processed_tokens = 0
    for step, cpu_batch in enumerate(loader):
        call_indices = cpu_batch["call_indices"].numpy().astype(np.int64).tolist()
        processed_indices.extend(call_indices)
        batch_sizes.append(len(call_indices))
        processed_tokens += int(cpu_batch["token_counts"].sum())
        batch = move_batch(cpu_batch, device)
        labels = batch["labels"]
        weights = weight_lookup[batch["call_indices"]]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(
                batch["global_embeddings"],
                batch["temporal_tokens"],
                batch["token_to_call"],
                len(labels),
            )
            per_call_loss = torch.nn.functional.cross_entropy(
                logits, labels, reduction="none"
            )
            loss = reference.global_weighted_micro_loss(
                per_call_loss, weights, values["accumulation_window_calls"]
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
    audit = reference.effective_coefficient_audit(
        processed_indices,
        batch_sizes,
        weight_lookup_numpy,
        store,
        values["accumulation_window_calls"],
        values["accumulation_steps"],
    )
    audit["processed_temporal_tokens"] = int(processed_tokens)
    audit["expected_temporal_tokens"] = int(
        sum(len(store.call_token_indices[index]) for index in processed_indices)
    )
    audit["all_temporal_tokens_covered"] = (
        audit["processed_temporal_tokens"] == audit["expected_temporal_tokens"]
    )
    return weighted_total / weight_total, audit


def predict_calls(
    model: GlobalLocalResidualClassifier,
    loader: DataLoader,
    store: ResidualStore,
    device: torch.device,
) -> tuple[float, pd.DataFrame]:
    model.eval()
    rows = []
    total_loss = 0.0
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(
                    batch["global_embeddings"],
                    batch["temporal_tokens"],
                    batch["token_to_call"],
                    len(batch["labels"]),
                )
                total_loss += float(
                    torch.nn.functional.cross_entropy(
                        logits, batch["labels"], reduction="sum"
                    )
                )
                probabilities = torch.softmax(logits, dim=1).float().cpu().numpy()
            indices = cpu_batch["call_indices"].numpy().astype(np.int64)
            labels = cpu_batch["labels"].numpy().astype(np.int64)
            token_counts = cpu_batch["token_counts"].numpy().astype(np.int64)
            for local_index, call_index in enumerate(indices):
                rows.append(
                    {
                        "call_index": int(call_index),
                        "call_id": str(store.call_ids[call_index]),
                        "cat_id": str(store.cat_ids[call_index]),
                        "true_label": int(labels[local_index]),
                        "duration": float(store.durations[call_index]),
                        "temporal_token_count": int(token_counts[local_index]),
                        **{
                            column: float(probabilities[local_index, class_index])
                            for class_index, column in enumerate(PROBABILITY_COLUMNS)
                        },
                        "predicted_label": int(probabilities[local_index].argmax()),
                    }
                )
    calls = pd.DataFrame(rows).sort_values("call_index").reset_index(drop=True)
    return total_loss / len(calls), calls


def calls_to_animals(calls: pd.DataFrame) -> pd.DataFrame:
    animals = previous.calls_to_animals(calls)
    metadata = (
        calls.groupby("cat_id", sort=True)
        .agg(
            mean_call_duration=("duration", "mean"),
            mean_tokens_per_call=("temporal_token_count", "mean"),
            total_temporal_tokens=("temporal_token_count", "sum"),
        )
        .reset_index()
    )
    return animals.merge(metadata, on="cat_id", validate="one_to_one")


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def fit_inner(
    pipeline: str,
    protocol: dict[str, Any],
    store: ResidualStore,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
    max_epochs_override: int | None = None,
) -> tuple[int, dict[str, Any], dict[str, torch.Tensor], pd.DataFrame]:
    reference.historical.set_seed(seed)
    values = training_values(protocol)
    max_epochs = max_epochs_override or values["max_epochs"]
    model = build_model(pipeline, protocol, store, train_indices).to(device)
    parameters = reference.historical.idea019.trainable_counts(model)
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=values["learning_rate"], eps=values["optimizer_epsilon"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    lookup_numpy = reference.global_class_balanced_call_weights(
        store.labels, train_indices
    )
    lookup = torch.from_numpy(lookup_numpy).to(device)
    train_loader = build_loader(store, train_indices, values["batch_size"], True, seed)
    validation_loader = build_loader(
        store, validation_indices, values["batch_size"] * 2, False, seed
    )
    best_loss = float("inf")
    best_epoch = 1
    best_metrics: dict[str, Any] = {}
    best_state = cpu_state_dict(model)
    best_gate = model.gate_summary()
    best_calls = pd.DataFrame()
    history = []
    epochs_without_improvement = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, max_epochs + 1):
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
        _, validation_calls = predict_calls(model, validation_loader, store, device)
        validation_animals = calls_to_animals(validation_calls)
        validation_loss = previous.animal_cross_entropy(validation_animals)
        metrics = previous.animal_metrics(validation_animals)
        gate = model.gate_summary()
        history.append(
            {
                "epoch": epoch,
                "train_weighted_loss": train_loss,
                "validation_animal_cross_entropy": validation_loss,
                "validation_animal_macro_f1": metrics["macro_f1"],
                "validation_animal_balanced_accuracy": metrics["balanced_accuracy"],
                "validation_animal_qwk": metrics["quadratic_weighted_kappa"],
                "residual_gate": gate,
                "train_unit_audit": train_audit,
            }
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_metrics = metrics
            best_state = cpu_state_dict(model)
            best_gate = gate
            best_calls = validation_calls.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"{pipeline} inner epoch={epoch} train={train_loss:.4f} "
            f"animal_val={validation_loss:.4f} val_F1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if max_epochs_override is None and epochs_without_improvement >= values["patience"]:
            break
    model.load_state_dict(best_state)
    audit = {
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(len(history)),
        "best_validation_animal_cross_entropy": float(best_loss),
        "best_validation_animal_metrics": best_metrics,
        "best_residual_gate": best_gate,
        "history": history,
        "target_weight_audit": reference.target_weight_audit(
            lookup_numpy, store, train_indices
        ),
        "train_seconds": float(time.perf_counter() - started),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "parameters": parameters,
        "checkpoint_selection": "minimum_unweighted_animal_cross_entropy",
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, audit, best_state, best_calls


def fit_outer_and_predict(
    pipeline: str,
    protocol: dict[str, Any],
    store: ResidualStore,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    epochs: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    reference.historical.set_seed(seed)
    values = training_values(protocol)
    model = build_model(pipeline, protocol, store, train_indices).to(device)
    parameters = reference.historical.idea019.trainable_counts(model)
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=values["learning_rate"], eps=values["optimizer_epsilon"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    lookup_numpy = reference.global_class_balanced_call_weights(
        store.labels, train_indices
    )
    lookup = torch.from_numpy(lookup_numpy).to(device)
    train_loader = build_loader(store, train_indices, values["batch_size"], True, seed)
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
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
        history.append(
            {
                "epoch": epoch,
                "train_weighted_loss": train_loss,
                "residual_gate": model.gate_summary(),
                "train_unit_audit": train_audit,
            }
        )
        print(
            f"{pipeline} outer epoch={epoch}/{epochs} train={train_loss:.4f}",
            flush=True,
        )
    test_loader = build_loader(store, test_indices, values["batch_size"] * 2, False, seed)
    _, calls = predict_calls(model, test_loader, store, device)
    animals = calls_to_animals(calls)
    audit = {
        "epochs": int(epochs),
        "history": history,
        "test_animal_cross_entropy": previous.animal_cross_entropy(animals),
        "test_animal_metrics": previous.animal_metrics(animals),
        "residual_gate": model.gate_summary(),
        "target_weight_audit": reference.target_weight_audit(
            lookup_numpy, store, train_indices
        ),
        "train_and_predict_seconds": float(time.perf_counter() - started),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "parameters": parameters,
        "checkpoint_selection": "inner_minimum_unweighted_animal_cross_entropy",
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return animals, calls, audit


def initialization_audit(
    protocol: dict[str, Any],
    store: ResidualStore,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    models = []
    for pipeline in PIPELINES:
        reference.historical.set_seed(seed)
        models.append(build_model(pipeline, protocol, store, train_indices).to(device).eval())
    shared_keys = (
        "global_mean",
        "global_scale",
        "token_mean",
        "token_scale",
        "projection.weight",
        "projection.bias",
        "batch_norm.weight",
        "batch_norm.bias",
        "batch_norm.running_mean",
        "batch_norm.running_var",
        "classifier.weight",
        "classifier.bias",
    )
    states = [model.state_dict() for model in models]
    shared_equal = all(
        torch.equal(states[0][key], states[index][key])
        for key in shared_keys
        for index in (1, 2)
    )
    loader = build_loader(store, validation_indices[:8], 8, False, seed)
    batch = move_batch(next(iter(loader)), device)
    with torch.no_grad():
        logits = [
            model(
                batch["global_embeddings"],
                batch["temporal_tokens"],
                batch["token_to_call"],
                len(batch["labels"]),
            ).float()
            for model in models
        ]
    max_differences = [float((logits[index] - logits[0]).abs().max()) for index in (1, 2)]
    singleton_index = next(
        index for index, tokens in enumerate(store.call_token_indices) if len(tokens) == 1
    )
    singleton_batch = move_batch(
        collate_residual_calls([ResidualCallDataset(store, np.array([singleton_index]))[0]]),
        device,
    )
    with torch.no_grad():
        _, singleton_residual = models[2].hidden_and_residual(
            singleton_batch["global_embeddings"],
            singleton_batch["temporal_tokens"],
            singleton_batch["token_to_call"],
            1,
        )
    result = {
        "shared_initial_state_equal": bool(shared_equal),
        "R1_initial_max_logit_difference_from_R0": max_differences[0],
        "R2_initial_max_logit_difference_from_R0": max_differences[1],
        "zero_initialized_residual_gates": all(
            torch.count_nonzero(model.residual_gate) == 0 for model in models[1:]
        ),
        "single_token_salience_residual_max_abs": float(singleton_residual.abs().max()),
        "single_token_call_index": int(singleton_index),
    }
    del models
    if not result["shared_initial_state_equal"] or any(max_differences):
        raise RuntimeError("IDEA-052 residual models do not match R0 at initialization")
    if result["single_token_salience_residual_max_abs"] != 0.0:
        raise RuntimeError("Single-token salience residual must be zero")
    return result


def reload_probability_difference(
    pipeline: str,
    protocol: dict[str, Any],
    store: ResidualStore,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    checkpoint_path: Path,
    before: pd.DataFrame,
    device: torch.device,
    seed: int,
) -> float:
    model = build_model(pipeline, protocol, store, train_indices).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    loader = build_loader(
        store,
        validation_indices,
        training_values(protocol)["batch_size"] * 2,
        False,
        seed,
    )
    _, after = predict_calls(model, loader, store, device)
    left = before.sort_values("call_index")[list(PROBABILITY_COLUMNS)].to_numpy()
    right = after.sort_values("call_index")[list(PROBABILITY_COLUMNS)].to_numpy()
    difference = float(np.abs(left - right).max())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return difference


def environment_lock(protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "git_revision": git_revision(),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "idea_sha256": sha256(PLAN_PATH),
        "diagnostic_sha256": sha256(DIAGNOSTIC_PATH),
        "roles_sha256": sha256(ROLES_PATH),
        "global_embedding_sha256": sha256(GLOBAL_EMBEDDING_PATH),
        "temporal_token_sha256": sha256(TEMPORAL_TOKEN_PATH),
    }


def run_smoke(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: ResidualStore,
    device: torch.device,
    resume: bool,
) -> None:
    smoke_root = run_root / "smoke"
    summary_path = smoke_root / "summary.json"
    if summary_path.is_file() and not resume:
        raise FileExistsError(summary_path)
    settings = protocol["smoke"]
    indices = reference.historical.fold_indices(
        store,
        roles,
        int(settings["repeat"]),
        int(settings["outer_fold"]),
        include_test=False,
    )
    seed = reference.historical.full_seed(
        int(settings["base_seed"]),
        int(settings["repeat"]),
        int(settings["outer_fold"]),
    )
    initial = initialization_audit(
        protocol, store, indices["train"], indices["validation"], device, seed
    )
    fits = []
    for pipeline in PIPELINES:
        output_dir = smoke_root / "fits" / pipeline
        checkpoint_path = output_dir / "best_checkpoint.pt"
        best_epoch, inner, state, validation_calls = fit_inner(
            pipeline,
            protocol,
            store,
            indices["train"],
            indices["validation"],
            device,
            seed,
            max_epochs_override=int(settings["epochs"]),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": state, "best_epoch": best_epoch}, checkpoint_path)
        reload_difference = reload_probability_difference(
            pipeline,
            protocol,
            store,
            indices["train"],
            indices["validation"],
            checkpoint_path,
            validation_calls,
            device,
            seed,
        )
        fit = {
            "status": "complete",
            "stage": "inner_only_smoke",
            "outer_test_accessed": False,
            "pipeline": pipeline,
            "base_seed": int(settings["base_seed"]),
            "repeat": int(settings["repeat"]),
            "outer_fold": int(settings["outer_fold"]),
            "full_seed": int(seed),
            "selected_epoch": int(best_epoch),
            "inner": inner,
            "checkpoint_path": repo_relative(checkpoint_path),
            "checkpoint_reload_max_probability_difference": reload_difference,
        }
        write_json(output_dir / "fit_summary.json", fit)
        fits.append(fit)
    environment_path = run_root / "environment_lock.json"
    write_json(environment_path, environment_lock(protocol, device))
    code_commit = git_revision()
    if code_commit is None:
        raise RuntimeError("A Git commit is required before smoke lock")
    lock = {
        "schema_version": "1.0",
        "status": "locked_for_idea052_initial_evaluation",
        "outer_test_accessed": False,
        "protocol_id": protocol["protocol_id"],
        "code_commit": code_commit,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "idea_sha256": sha256(PLAN_PATH),
        "diagnostic_sha256": sha256(DIAGNOSTIC_PATH),
        "diagnostic_script_sha256": sha256(DIAGNOSTIC_SCRIPT_PATH),
        "roles_sha256": sha256(ROLES_PATH),
        "global_embedding_sha256": sha256(GLOBAL_EMBEDDING_PATH),
        "temporal_token_sha256": sha256(TEMPORAL_TOKEN_PATH),
        "environment_lock_sha256": sha256(environment_path),
        "initial_evaluation": protocol["initial_evaluation"],
        "fixed_training": protocol["fixed_training"],
        "initialization_audit": initial,
        "smoke_fit_sha256": {
            pipeline: sha256(smoke_root / "fits" / pipeline / "fit_summary.json")
            for pipeline in PIPELINES
        },
    }
    lock_path = run_root / "execution_lock.json"
    write_json(lock_path, lock)
    summary = {
        "status": "complete",
        "outer_test_accessed": False,
        "excluded_from_evaluation_summary": True,
        "completed_fits": len(fits),
        "calls": int(len(store.call_ids)),
        "cats": int(len(np.unique(store.cat_ids))),
        "temporal_tokens": int(len(store.temporal_tokens)),
        "token_count_range": [
            int(min(len(value) for value in store.call_token_indices)),
            int(max(len(value) for value in store.call_token_indices)),
        ],
        "initialization_audit": initial,
        "checkpoint_reload_passed": all(
            fit["checkpoint_reload_max_probability_difference"] == 0.0 for fit in fits
        ),
        "execution_lock": repo_relative(lock_path),
        "execution_lock_sha256": sha256(lock_path),
        "environment_lock": repo_relative(environment_path),
        "environment_lock_sha256": sha256(environment_path),
        "pipelines": {
            fit["pipeline"]: {
                "validation_animal_macro_f1": fit["inner"][
                    "best_validation_animal_metrics"
                ]["macro_f1"],
                "validation_animal_cross_entropy": fit["inner"][
                    "best_validation_animal_cross_entropy"
                ],
                "best_residual_gate": fit["inner"]["best_residual_gate"],
                "checkpoint_reload_max_probability_difference": fit[
                    "checkpoint_reload_max_probability_difference"
                ],
            }
            for fit in fits
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


def verify_execution_lock(run_root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    lock_path = run_root / "execution_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("Run smoke before IDEA-052 evaluation")
    lock = read_json(lock_path)
    if lock["status"] != "locked_for_idea052_initial_evaluation":
        raise RuntimeError("IDEA-052 execution lock status is invalid")
    checks = {
        PROTOCOL_PATH: lock["protocol_sha256"],
        Path(__file__).resolve(): lock["runner_sha256"],
        PLAN_PATH: lock["idea_sha256"],
        DIAGNOSTIC_PATH: lock["diagnostic_sha256"],
        DIAGNOSTIC_SCRIPT_PATH: lock["diagnostic_script_sha256"],
        ROLES_PATH: lock["roles_sha256"],
        GLOBAL_EMBEDDING_PATH: lock["global_embedding_sha256"],
        TEMPORAL_TOKEN_PATH: lock["temporal_token_sha256"],
    }
    for path, expected in checks.items():
        if sha256(path) != expected:
            raise RuntimeError(f"IDEA-052 execution-lock file changed: {path}")
    revision = lock["code_commit"]
    for path in (
        PROTOCOL_PATH,
        Path(__file__).resolve(),
        PLAN_PATH,
        DIAGNOSTIC_PATH,
        DIAGNOSTIC_SCRIPT_PATH,
    ):
        if git_blob_object_id(revision, path) != worktree_blob_object_id(path):
            raise RuntimeError(f"Locked commit blob differs from worktree: {path}")
    if lock["initial_evaluation"] != protocol["initial_evaluation"]:
        raise RuntimeError("IDEA-052 evaluation matrix differs from lock")
    return lock


def assert_batch_orders(fits: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = []
    for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
        left = fits[left_index]
        right = fits[right_index]
        comparison = {
            "left": left["pipeline"],
            "right": right["pipeline"],
        }
        for phase in ("inner", "outer"):
            left_history = left[phase]["history"]
            right_history = right[phase]["history"]
            common = min(len(left_history), len(right_history))
            for epoch in range(common):
                left_hash = left_history[epoch]["train_unit_audit"]["batch_order_sha256"]
                right_hash = right_history[epoch]["train_unit_audit"]["batch_order_sha256"]
                if left_hash != right_hash:
                    raise RuntimeError(
                        f"IDEA-052 batch order differs for {phase} epoch {epoch + 1}"
                    )
            comparison[f"{phase}_common_epochs"] = common
        comparisons.append(comparison)
    return {"all_common_epoch_hashes_match": True, "comparisons": comparisons}


def subgroup_metrics(frame: pd.DataFrame, column: str, boundaries: list[float]) -> dict[str, Any]:
    names = [
        f"<={boundaries[0]}",
        f"{boundaries[0]}-{boundaries[1]}",
        f">{boundaries[1]}",
    ]
    values = frame[column].to_numpy(dtype=float)
    masks = (
        values <= boundaries[0],
        (values > boundaries[0]) & (values <= boundaries[1]),
        values > boundaries[1],
    )
    result = {}
    for name, mask in zip(names, masks):
        group = frame.loc[mask]
        metrics = previous.animal_metrics(group)
        result[name] = {
            "cat_evaluations": int(len(group)),
            "unique_cats": int(group["cat_id"].nunique()),
            "class_support": {
                LABEL_NAMES[index]: int((group["true_label"] == index).sum())
                for index in range(3)
            },
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "plain_accuracy": metrics["plain_accuracy"],
        }
    return result


def aggregate_evaluation(
    evaluation_root: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    evaluation = protocol["initial_evaluation"]
    metrics_by_pipeline: dict[str, list[dict[str, Any]]] = {
        pipeline: [] for pipeline in PIPELINES
    }
    animals_by_key: dict[tuple[str, int], pd.DataFrame] = {}
    gates: dict[str, list[dict[str, Any]]] = {pipeline: [] for pipeline in RESIDUAL_PIPELINES}
    selected_epochs: dict[str, list[int]] = {pipeline: [] for pipeline in PIPELINES}
    for repeat in evaluation["repeats"]:
        for pipeline in PIPELINES:
            call_frames = []
            for outer_fold in evaluation["outer_folds"]:
                fit_root = (
                    evaluation_root
                    / "fits"
                    / pipeline
                    / f"base_seed_{evaluation['base_seeds'][0]}"
                    / f"repeat_{repeat}"
                    / f"fold_{outer_fold}"
                )
                call_frames.append(
                    pd.read_csv(
                        fit_root / "outer_test_call_predictions.csv",
                        dtype={"cat_id": str},
                    )
                )
                fit = read_json(fit_root / "fit_summary.json")
                selected_epochs[pipeline].append(int(fit["selected_epoch"]))
                if pipeline in RESIDUAL_PIPELINES:
                    gates[pipeline].append(
                        {
                            "repeat": int(repeat),
                            "outer_fold": int(outer_fold),
                            **fit["outer"]["residual_gate"],
                        }
                    )
            calls = (
                pd.concat(call_frames, ignore_index=True)
                .sort_values("call_index")
                .reset_index(drop=True)
            )
            if len(calls) != 792 or calls["call_index"].nunique() != 792:
                raise RuntimeError("Complete IDEA-052 OOF must contain 792 calls")
            animals = calls_to_animals(calls)
            if len(animals) != 111 or animals["cat_id"].nunique() != 111:
                raise RuntimeError("Complete IDEA-052 OOF must contain 111 cats")
            metrics = previous.animal_metrics(animals)
            metrics_by_pipeline[pipeline].append(
                {
                    "base_seed": int(evaluation["base_seeds"][0]),
                    "repeat": int(repeat),
                    **metrics,
                }
            )
            animals_by_key[(pipeline, int(repeat))] = animals
            output_root = evaluation_root / "oof" / pipeline
            output_root.mkdir(parents=True, exist_ok=True)
            calls.to_csv(output_root / f"repeat_{repeat}_calls.csv", index=False)
            animals.to_csv(output_root / f"repeat_{repeat}_animals.csv", index=False)

    contrasts = (
        (PIPELINES[0], PIPELINES[1]),
        (PIPELINES[0], PIPELINES[2]),
        (PIPELINES[1], PIPELINES[2]),
    )
    paired: dict[str, list[dict[str, Any]]] = {}
    change_frames = []
    for left_pipeline, right_pipeline in contrasts:
        name = f"{right_pipeline}_minus_{left_pipeline}"
        rows = []
        for repeat in evaluation["repeats"]:
            left_metrics = metrics_by_pipeline[left_pipeline][int(repeat)]
            right_metrics = metrics_by_pipeline[right_pipeline][int(repeat)]
            left = animals_by_key[(left_pipeline, int(repeat))]
            right = animals_by_key[(right_pipeline, int(repeat))]
            merged = left.merge(
                right,
                on=["cat_id", "true_label", "call_count"],
                suffixes=("_left", "_right"),
            )
            rows.append(
                {
                    "repeat": int(repeat),
                    "left_macro_f1": left_metrics["macro_f1"],
                    "right_macro_f1": right_metrics["macro_f1"],
                    "macro_f1_difference": right_metrics["macro_f1"]
                    - left_metrics["macro_f1"],
                    "balanced_accuracy_difference": right_metrics["balanced_accuracy"]
                    - left_metrics["balanced_accuracy"],
                    "qwk_difference": right_metrics["quadratic_weighted_kappa"]
                    - left_metrics["quadratic_weighted_kappa"],
                    "plain_accuracy_difference": right_metrics["plain_accuracy"]
                    - left_metrics["plain_accuracy"],
                    "changed_animals": int(
                        (
                            merged["predicted_label_left"]
                            != merged["predicted_label_right"]
                        ).sum()
                    ),
                    "gained_correct_animals": int(
                        (
                            (merged["predicted_label_right"] == merged["true_label"])
                            & (merged["predicted_label_left"] != merged["true_label"])
                        ).sum()
                    ),
                    "lost_correct_animals": int(
                        (
                            (merged["predicted_label_right"] != merged["true_label"])
                            & (merged["predicted_label_left"] == merged["true_label"])
                        ).sum()
                    ),
                }
            )
            changed = merged[
                merged["predicted_label_left"] != merged["predicted_label_right"]
            ].copy()
            changed.insert(0, "contrast", name)
            changed.insert(1, "repeat", int(repeat))
            change_frames.append(changed)
        paired[name] = rows
    paired_change_path = evaluation_root / "paired_prediction_changes.csv"
    pd.concat(change_frames, ignore_index=True).to_csv(paired_change_path, index=False)

    aggregate = {
        pipeline: previous.aggregate_metrics(metrics_by_pipeline[pipeline])
        for pipeline in PIPELINES
    }
    paired_summary = {}
    bootstraps = {}
    for left_pipeline, right_pipeline in contrasts:
        name = f"{right_pipeline}_minus_{left_pipeline}"
        rows = paired[name]
        paired_summary[name] = {
            "macro_f1_differences": [row["macro_f1_difference"] for row in rows],
            "mean_macro_f1_difference": float(
                np.mean([row["macro_f1_difference"] for row in rows])
            ),
            "positive_repeats": int(
                sum(row["macro_f1_difference"] > 0 for row in rows)
            ),
            "mean_balanced_accuracy_difference": float(
                np.mean([row["balanced_accuracy_difference"] for row in rows])
            ),
            "mean_qwk_difference": float(
                np.mean([row["qwk_difference"] for row in rows])
            ),
            "mean_plain_accuracy_difference": float(
                np.mean([row["plain_accuracy_difference"] for row in rows])
            ),
        }
        bootstraps[name] = previous.paired_cat_bootstrap(
            animals_by_key, left_pipeline, right_pipeline, protocol
        )

    all_animals = {
        pipeline: pd.concat(
            [animals_by_key[(pipeline, int(repeat))] for repeat in evaluation["repeats"]],
            ignore_index=True,
        )
        for pipeline in PIPELINES
    }
    subgroups = {
        pipeline: {
            "mean_call_duration_seconds": subgroup_metrics(
                frame, "mean_call_duration", [0.5, 1.0]
            ),
            "mean_temporal_tokens_per_call": subgroup_metrics(
                frame, "mean_tokens_per_call", [5.0, 9.0]
            ),
        }
        for pipeline, frame in all_animals.items()
    }
    gate_summary = {
        pipeline: {
            "fold_models": len(rows),
            "mean_abs_gate": float(np.mean([row["mean_abs_gate"] for row in rows])),
            "max_abs_gate": float(max(row["max_abs_gate"] for row in rows)),
            "mean_active_fraction_above_1e_3": float(
                np.mean([row["active_fraction_above_1e_3"] for row in rows])
            ),
            "details": rows,
        }
        for pipeline, rows in gates.items()
    }
    training_dynamics = {
        pipeline: {
            "selected_epochs": values,
            "mean_selected_epoch": float(np.mean(values)),
            "range": [int(min(values)), int(max(values))],
        }
        for pipeline, values in selected_epochs.items()
    }
    expansion_gate = {}
    for candidate in RESIDUAL_PIPELINES:
        name = f"{candidate}_minus_{PIPELINES[0]}"
        comparison = paired_summary[name]
        candidate_metrics = aggregate[candidate]
        reference_metrics = aggregate[PIPELINES[0]]
        supporting = []
        if comparison["mean_balanced_accuracy_difference"] >= 0:
            supporting.append("balanced_accuracy")
        if comparison["mean_qwk_difference"] >= 0:
            supporting.append("qwk")
        if candidate_metrics["macro_f1_range"][0] >= reference_metrics["macro_f1_range"][0]:
            supporting.append("worst_repeat_macro_f1")
        for label in LABEL_NAMES:
            if (
                candidate_metrics["mean_per_class"][label]["recall"]
                > reference_metrics["mean_per_class"][label]["recall"]
            ):
                supporting.append(f"{label}_recall")
        passed = (
            comparison["mean_macro_f1_difference"]
            >= float(protocol["seed_expansion_gate"]["minimum_mean_macro_f1_gain"])
            and comparison["positive_repeats"]
            >= int(protocol["seed_expansion_gate"]["minimum_positive_repeats"])
            and bool(supporting)
        )
        expansion_gate[candidate] = {
            "passed": bool(passed),
            "strong_gain": bool(
                comparison["mean_macro_f1_difference"]
                >= float(protocol["seed_expansion_gate"]["strong_gain"])
            ),
            "supporting_signals": supporting,
        }
    return {
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "completed_outer_fits": int(evaluation["total_outer_fits"]),
        "complete_oof": metrics_by_pipeline,
        "aggregate": aggregate,
        "paired": paired,
        "paired_summary": paired_summary,
        "paired_cat_bootstrap": bootstraps,
        "subgroups": subgroups,
        "residual_gates": gate_summary,
        "training_dynamics": training_dynamics,
        "seed_expansion_gate": expansion_gate,
        "artifacts": {"paired_prediction_changes": repo_relative(paired_change_path)},
    }


def run_evaluation(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: ResidualStore,
    device: torch.device,
    resume: bool,
) -> None:
    lock = verify_execution_lock(run_root, protocol)
    evaluation_root = run_root / "evaluation"
    summary_path = evaluation_root / "summary.json"
    if summary_path.is_file() and not resume:
        raise FileExistsError(summary_path)
    evaluation_root.mkdir(parents=True, exist_ok=True)
    write_json(
        evaluation_root / "run_manifest.json",
        {
            "status": "running",
            "stage": "idea052_initial_evaluation",
            "outer_test_accessed": True,
            "code_commit": lock["code_commit"],
            "protocol_sha256": lock["protocol_sha256"],
            "runner_sha256": lock["runner_sha256"],
            "execution_lock_sha256": sha256(run_root / "execution_lock.json"),
            "environment_lock_sha256": sha256(run_root / "environment_lock.json"),
            "device": str(device),
        },
    )
    completed = []
    order_audits = []
    evaluation = protocol["initial_evaluation"]
    for base_seed in evaluation["base_seeds"]:
        for repeat in evaluation["repeats"]:
            for outer_fold in evaluation["outer_folds"]:
                indices = reference.historical.fold_indices(
                    store, roles, int(repeat), int(outer_fold), include_test=True
                )
                outer_train = np.concatenate((indices["train"], indices["validation"]))
                seed = reference.historical.full_seed(
                    int(base_seed), int(repeat), int(outer_fold)
                )
                fit_group = []
                for pipeline in PIPELINES:
                    output_dir = (
                        evaluation_root
                        / "fits"
                        / pipeline
                        / f"base_seed_{base_seed}"
                        / f"repeat_{repeat}"
                        / f"fold_{outer_fold}"
                    )
                    fit_path = output_dir / "fit_summary.json"
                    if fit_path.is_file():
                        if not resume:
                            raise FileExistsError(fit_path)
                        fit = read_json(fit_path)
                        completed.append(fit)
                        fit_group.append(fit)
                        continue
                    print(
                        f"IDEA052 EVAL {pipeline} base_seed={base_seed} "
                        f"repeat={repeat} fold={outer_fold} seed={seed}",
                        flush=True,
                    )
                    best_epoch, inner, _, _ = fit_inner(
                        pipeline,
                        protocol,
                        store,
                        indices["train"],
                        indices["validation"],
                        device,
                        seed,
                    )
                    animals, calls, outer = fit_outer_and_predict(
                        pipeline,
                        protocol,
                        store,
                        outer_train,
                        indices["test"],
                        best_epoch,
                        device,
                        seed,
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    animal_path = output_dir / "outer_test_animal_predictions.csv"
                    call_path = output_dir / "outer_test_call_predictions.csv"
                    animals.to_csv(animal_path, index=False)
                    calls.to_csv(call_path, index=False)
                    fit = {
                        "status": "complete",
                        "stage": "idea052_initial_evaluation",
                        "outer_test_accessed": True,
                        "pipeline": pipeline,
                        "base_seed": int(base_seed),
                        "repeat": int(repeat),
                        "outer_fold": int(outer_fold),
                        "full_seed": int(seed),
                        "selected_epoch": int(best_epoch),
                        "inner": inner,
                        "outer": outer,
                        "animal_prediction_path": repo_relative(animal_path),
                        "call_prediction_path": repo_relative(call_path),
                    }
                    write_json(fit_path, fit)
                    completed.append(fit)
                    fit_group.append(fit)
                order_audits.append(
                    {
                        "base_seed": int(base_seed),
                        "repeat": int(repeat),
                        "outer_fold": int(outer_fold),
                        **assert_batch_orders(fit_group),
                    }
                )
    summary = aggregate_evaluation(evaluation_root, protocol)
    summary["batch_order_audit"] = {
        "fold_groups": len(order_audits),
        "all_common_epoch_hashes_match": True,
        "details": order_audits,
    }
    inventory = previous.raw_prediction_inventory(evaluation_root)
    inventory_path = evaluation_root / "raw_prediction_inventory.json"
    write_json(inventory_path, inventory)
    summary["raw_prediction_inventory"] = {
        "path": repo_relative(inventory_path),
        "inventory_sha256": sha256(inventory_path),
        "files": inventory["files"],
        "bytes": inventory["bytes"],
        "aggregate_sha256": inventory["aggregate_sha256"],
    }
    summary["code_commit"] = lock["code_commit"]
    summary["execution_lock_sha256"] = sha256(run_root / "execution_lock.json")
    summary["environment_lock_sha256"] = sha256(run_root / "environment_lock.json")
    summary["runner_sha256"] = lock["runner_sha256"]
    summary["protocol_sha256"] = lock["protocol_sha256"]
    write_json(summary_path, summary)
    write_json(
        evaluation_root / "run_summary.json",
        {
            "status": "complete",
            "completed_fits": len(completed),
            "expected_fits": int(evaluation["total_outer_fits"]),
            "summary_path": repo_relative(summary_path),
            "summary_sha256": sha256(summary_path),
            "raw_prediction_inventory_sha256": sha256(inventory_path),
            "raw_prediction_aggregate_sha256": inventory["aggregate_sha256"],
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = (RUNS_ROOT / args.output_subdir).resolve()
    if RUNS_ROOT.resolve() not in run_root.parents:
        raise ValueError("--output-subdir must stay below runs")
    run_root.mkdir(parents=True, exist_ok=True)
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    store = load_store()
    device = reference.historical.idea019.resolve_device(args.device)
    print(
        f"IDEA-052 stage={args.stage}; device={device}; "
        f"device_name={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}",
        flush=True,
    )
    if args.stage == "smoke":
        run_smoke(run_root, protocol, roles, store, device, args.resume)
    else:
        run_evaluation(run_root, protocol, roles, store, device, args.resume)


if __name__ == "__main__":
    main()
