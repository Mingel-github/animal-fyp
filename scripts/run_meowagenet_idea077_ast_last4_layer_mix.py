"""Run the preregistered IDEA-077 AST last-four global LayerMix screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea077_ast_last4_layer_mix_v1.json"
)
PIPELINES = ("A0_final", "M0_uniform_last4", "L1_global_layermix")
BASE_SEEDS = (6917, 1398, 5934)
LAST4_ZERO_BASED = (8, 9, 10, 11)
LABEL_NAMES = ("kitten", "adult", "senior")
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
EXPECTED_PARAMETERS = {
    "A0_final": 99_075,
    "M0_uniform_last4": 99_075,
    "L1_global_layermix": 99_080,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "run"), required=True)
    parser.add_argument(
        "--output-subdir", default="meowagenet_idea077_ast_last4_layer_mix_v1"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
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
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


@dataclass(frozen=True)
class RepresentationStore:
    frozen_embeddings: np.ndarray
    last4_embeddings: np.ndarray
    cached_last_layer: np.ndarray
    call_ids: np.ndarray
    cat_ids: np.ndarray
    labels: np.ndarray


def load_store(protocol: dict[str, Any]) -> tuple[RepresentationStore, dict[str, float]]:
    final_path = REPO_ROOT / protocol["data"]["frozen_embedding_path"]
    layer_path = REPO_ROOT / protocol["data"]["layer_embedding_path"]
    with np.load(final_path) as loaded_final, np.load(layer_path) as loaded_layers:
        final = loaded_final["embeddings"].astype(np.float32)
        layers = loaded_layers["embeddings"].astype(np.float32)
        call_ids = loaded_final["call_ids"].astype(str)
        cat_ids = loaded_final["cat_ids"].astype(str)
        labels = loaded_final["labels"].astype(np.int64)
        if not (
            np.array_equal(call_ids, loaded_layers["call_ids"].astype(str))
            and np.array_equal(cat_ids, loaded_layers["cat_ids"].astype(str))
            and np.array_equal(labels, loaded_layers["labels"].astype(np.int64))
        ):
            raise RuntimeError("IDEA-077 final/layer cache metadata differs")
    if final.shape != (792, 768) or layers.shape != (792, 12, 768):
        raise RuntimeError(
            f"IDEA-077 cache geometry changed: final={final.shape}, layers={layers.shape}"
        )
    if not np.isfinite(final).all() or not np.isfinite(layers).all():
        raise RuntimeError("IDEA-077 cache contains non-finite values")
    difference = np.abs(layers[:, -1] - final)
    audit = {
        "mean_absolute_difference": float(difference.mean()),
        "maximum_absolute_difference": float(difference.max()),
    }
    tolerance = protocol["representation"]["cached_last_layer_audit_tolerance"]
    if audit["mean_absolute_difference"] > float(tolerance["mean_absolute_maximum"]):
        raise RuntimeError("IDEA-077 cached final-layer mean difference is too large")
    if audit["maximum_absolute_difference"] > float(tolerance["absolute_maximum"]):
        raise RuntimeError("IDEA-077 cached final-layer maximum difference is too large")
    last4 = layers[:, list(LAST4_ZERO_BASED)].copy()
    last4[:, -1] = final
    return (
        RepresentationStore(final, last4, layers[:, -1].copy(), call_ids, cat_ids, labels),
        audit,
    )


def full_seed(base_seed: int, repeat: int, fold: int) -> int:
    return int(base_seed + 10_000 * repeat + 100 * fold)


def fold_indices(
    store: RepresentationStore,
    roles: pd.DataFrame,
    repeat: int,
    fold: int,
) -> dict[str, np.ndarray]:
    cell = roles[(roles["repeat"] == repeat) & (roles["outer_fold"] == fold)]
    if cell["cat_id"].duplicated().any():
        raise RuntimeError("IDEA-077 role cell assigns a cat more than once")
    mapping = dict(zip(cell["cat_id"].astype(str), cell["role"].astype(str)))
    if set(mapping) != set(store.cat_ids.astype(str)):
        raise RuntimeError("IDEA-077 role cell does not cover exactly the cached cats")
    call_roles = np.asarray([mapping[cat_id] for cat_id in store.cat_ids], dtype=str)
    return {
        "train": np.flatnonzero(call_roles == "train"),
        "validation": np.flatnonzero(call_roles == "validation"),
    }


class DeterministicNoSingletonBatchSampler(Sampler[list[int]]):
    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        if size < 2 or batch_size < 2:
            raise ValueError("IDEA-077 batching needs at least two cats")
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.generator = torch.Generator().manual_seed(int(seed))

    def __iter__(self) -> Iterator[list[int]]:
        order = torch.randperm(self.size, generator=self.generator).tolist()
        batches = [
            order[start : start + self.batch_size]
            for start in range(0, self.size, self.batch_size)
        ]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-1].insert(0, batches[-2].pop())
        if any(len(batch) == 1 for batch in batches):
            raise RuntimeError("IDEA-077 singleton cat batch remains")
        yield from batches

    def __len__(self) -> int:
        return math.ceil(self.size / self.batch_size)


class CatDataset(Dataset[tuple[np.ndarray, np.ndarray, np.ndarray, str]]):
    def __init__(self, store: RepresentationStore, call_indices: np.ndarray) -> None:
        self.store = store
        selected = np.asarray(call_indices, dtype=np.int64)
        self.cat_ids = np.asarray(sorted(np.unique(store.cat_ids[selected]).tolist()))
        self.call_indices: list[np.ndarray] = []
        for cat_id in self.cat_ids:
            calls = np.sort(selected[store.cat_ids[selected] == cat_id])
            labels = np.unique(store.labels[calls])
            if len(labels) != 1:
                raise RuntimeError(f"IDEA-077 cat {cat_id} has inconsistent labels")
            self.call_indices.append(calls)
        if sum(len(value) for value in self.call_indices) != len(selected):
            raise RuntimeError("IDEA-077 cat dataset lost calls")

    def __len__(self) -> int:
        return len(self.cat_ids)

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        calls = self.call_indices[item]
        return (
            self.store.frozen_embeddings[calls],
            self.store.last4_embeddings[calls],
            calls,
            str(self.cat_ids[item]),
        )


def collate_cats(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]) -> dict[str, Any]:
    final, last4, indices, cats = zip(*rows)
    return {
        "final": torch.from_numpy(np.concatenate(final).astype(np.float32, copy=False)),
        "last4": torch.from_numpy(np.concatenate(last4).astype(np.float32, copy=False)),
        "call_indices": torch.from_numpy(np.concatenate(indices).astype(np.int64, copy=False)),
        "cat_ids": list(cats),
    }


def build_loader(
    store: RepresentationStore,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = CatDataset(store, indices)
    if shuffle:
        return DataLoader(
            dataset,
            batch_sampler=DeterministicNoSingletonBatchSampler(
                len(dataset), batch_size, seed
            ),
            num_workers=0,
            collate_fn=collate_cats,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_cats,
    )


class LayerMixClassifier(nn.Module):
    def __init__(
        self,
        pipeline: str,
        final_mean: np.ndarray,
        final_scale: np.ndarray,
        dropout: float,
    ) -> None:
        super().__init__()
        if pipeline not in PIPELINES:
            raise ValueError(pipeline)
        self.pipeline = pipeline
        safe_scale = np.where(final_scale > 1.0e-12, final_scale, 1.0).astype(
            np.float32
        )
        self.register_buffer("feature_mean", torch.from_numpy(final_mean.astype(np.float32)))
        self.register_buffer("feature_scale", torch.from_numpy(safe_scale))
        self.ast_linear = nn.Linear(768, 128)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(128, 3)
        if pipeline == "L1_global_layermix":
            self.layer_logits = nn.Parameter(torch.zeros(4, dtype=torch.float32))
            self.residual_gate_raw = nn.Parameter(torch.zeros((), dtype=torch.float32))
        else:
            self.layer_logits = None
            self.residual_gate_raw = None

    def representation(
        self, final: torch.Tensor, last4: torch.Tensor
    ) -> torch.Tensor:
        if self.pipeline == "A0_final":
            return final
        uniform = last4.mean(dim=1)
        if self.pipeline == "M0_uniform_last4":
            return uniform
        if self.layer_logits is None or self.residual_gate_raw is None:
            raise RuntimeError("IDEA-077 L1 parameters are missing")
        weights = torch.softmax(self.layer_logits, dim=0)
        mixed = torch.sum(last4 * weights[None, :, None], dim=1)
        gate = torch.tanh(self.residual_gate_raw)
        return final + gate * (mixed - final)

    def forward(self, final: torch.Tensor, last4: torch.Tensor) -> torch.Tensor:
        representation = self.representation(final, last4)
        normalized = (representation - self.feature_mean) / self.feature_scale
        hidden = self.relu(self.ast_linear(normalized))
        hidden = self.dropout(self.batch_norm(hidden))
        return self.output(hidden)

    def audit(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "trainable_parameters": int(sum(p.numel() for p in self.parameters())),
            "representation": self.pipeline,
        }
        if self.layer_logits is not None and self.residual_gate_raw is not None:
            result["layer_weights_one_based_9_to_12"] = (
                torch.softmax(self.layer_logits.detach(), dim=0).cpu().tolist()
            )
            result["signed_residual_gate"] = float(
                torch.tanh(self.residual_gate_raw.detach()).cpu()
            )
            result["raw_gate"] = float(self.residual_gate_raw.detach().cpu())
        return result


def build_model(
    pipeline: str,
    protocol: dict[str, Any],
    store: RepresentationStore,
    train_indices: np.ndarray,
) -> LayerMixClassifier:
    final_train = store.frozen_embeddings[train_indices]
    return LayerMixClassifier(
        pipeline,
        final_train.mean(axis=0),
        final_train.std(axis=0),
        float(protocol["fixed_training"]["dropout"]),
    )


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("IDEA-077 training role is missing a class")
    return (len(labels) / (3.0 * counts)).astype(np.float32)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "final": batch["final"].to(device),
        "last4": batch["last4"].to(device),
        "call_indices": batch["call_indices"].to(device),
        "cat_ids": batch["cat_ids"],
    }


def train_one_epoch(
    model: LayerMixClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    store: RepresentationStore,
    weights: torch.Tensor,
    device: torch.device,
    gradient_clip: float,
) -> tuple[float, dict[str, Any]]:
    model.train()
    weighted_total = 0.0
    weight_total = 0.0
    processed_cats: list[str] = []
    processed_calls: list[int] = []
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        call_indices = batch["call_indices"]
        labels = torch.from_numpy(store.labels[call_indices.cpu().numpy()]).to(device)
        call_weights = weights[labels]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(batch["final"], batch["last4"])
            per_call = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            loss = (per_call * call_weights).sum() / call_weights.sum()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        weighted_total += float((per_call.detach() * call_weights).sum())
        weight_total += float(call_weights.sum())
        processed_cats.extend(str(value) for value in batch["cat_ids"])
        processed_calls.extend(int(value) for value in call_indices.cpu().tolist())
    if len(processed_cats) != len(set(processed_cats)):
        raise RuntimeError("IDEA-077 epoch repeated a cat")
    if len(processed_calls) != len(set(processed_calls)):
        raise RuntimeError("IDEA-077 epoch repeated a call")
    return weighted_total / weight_total, {
        "cats": len(processed_cats),
        "calls": len(processed_calls),
        "cat_order_sha256": hashlib.sha256(
            "\n".join(processed_cats).encode("utf-8")
        ).hexdigest(),
        "call_coverage_sha256": hashlib.sha256(
            np.sort(np.asarray(processed_calls, dtype="<i8")).tobytes()
        ).hexdigest(),
    }


def calls_to_animals(calls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat_id, group in calls.groupby("cat_id", sort=True):
        labels = group["true_label"].unique()
        if len(labels) != 1:
            raise RuntimeError(f"IDEA-077 cat {cat_id} has conflicting labels")
        probabilities = group[list(PROBABILITY_COLUMNS)].to_numpy(float).mean(axis=0)
        rows.append(
            {
                "cat_id": str(cat_id),
                "true_label": int(labels[0]),
                "call_count": int(len(group)),
                **{column: float(probabilities[i]) for i, column in enumerate(PROBABILITY_COLUMNS)},
                "predicted_label": int(probabilities.argmax()),
            }
        )
    return pd.DataFrame(rows)


def predict(
    model: LayerMixClassifier,
    store: RepresentationStore,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = build_loader(store, indices, batch_size, False, seed)
    model.eval()
    rows = []
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = model(batch["final"], batch["last4"])
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            call_indices = batch["call_indices"].cpu().numpy()
            for local_index, call_index in enumerate(call_indices):
                rows.append(
                    {
                        "call_index": int(call_index),
                        "call_id": str(store.call_ids[call_index]),
                        "cat_id": str(store.cat_ids[call_index]),
                        "true_label": int(store.labels[call_index]),
                        **{
                            column: float(probabilities[local_index, class_index])
                            for class_index, column in enumerate(PROBABILITY_COLUMNS)
                        },
                    }
                )
    calls = pd.DataFrame(rows).sort_values("call_index").reset_index(drop=True)
    return calls_to_animals(calls), calls


def animal_metrics(animals: pd.DataFrame) -> dict[str, Any]:
    labels = animals["true_label"].to_numpy(np.int64)
    predictions = animals["predicted_label"].to_numpy(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1, 2], zero_division=0
    )
    return {
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(labels, predictions, weights="quadratic")),
        "plain_accuracy": float(accuracy_score(labels, predictions)),
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            }
            for i in range(3)
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
        "n": int(len(animals)),
    }


def animal_cross_entropy(animals: pd.DataFrame) -> float:
    probabilities = animals[list(PROBABILITY_COLUMNS)].to_numpy(float)
    labels = animals["true_label"].to_numpy(np.int64)
    return float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def brier(animals: pd.DataFrame) -> float:
    probabilities = animals[list(PROBABILITY_COLUMNS)].to_numpy(float)
    labels = animals["true_label"].to_numpy(np.int64)
    return float(np.mean(np.sum((probabilities - np.eye(3)[labels]) ** 2, axis=1)))


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def fit_inner(
    pipeline: str,
    protocol: dict[str, Any],
    store: RepresentationStore,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    model = build_model(pipeline, protocol, store, train_indices).to(device)
    fixed = protocol["fixed_training"]
    training_seed = seed + int(fixed["post_build_seed_offset"])
    set_seed(training_seed)
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=float(fixed["learning_rate"]), eps=float(fixed["optimizer_epsilon"])
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    train_loader = build_loader(
        store, train_indices, int(fixed["cat_batch_size"]), True, seed
    )
    weights = torch.from_numpy(class_weights(store.labels[train_indices])).to(device)
    best_loss = float("inf")
    best_epoch = 1
    best_state = cpu_state_dict(model)
    best_animals: pd.DataFrame | None = None
    best_calls: pd.DataFrame | None = None
    history = []
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, int(fixed["maximum_epochs"]) + 1):
        train_loss, train_audit = train_one_epoch(
            model, train_loader, optimizer, scaler, store, weights, device,
            float(fixed["gradient_clip"]),
        )
        animals, calls = predict(
            model, store, validation_indices, int(fixed["cat_batch_size"]) * 2, device, seed
        )
        validation_loss = animal_cross_entropy(animals)
        metrics = animal_metrics(animals)
        history.append({
            "epoch": epoch,
            "train_call_loss": train_loss,
            "train_audit": train_audit,
            "validation_animal_cross_entropy": validation_loss,
            "validation_animal_brier": brier(animals),
            "validation_animal_metrics": metrics,
            "model": model.audit(),
        })
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = cpu_state_dict(model)
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
        raise RuntimeError("IDEA-077 selected no checkpoint")
    model.load_state_dict(best_state)
    reload_animals, _ = predict(
        model, store, validation_indices, int(fixed["cat_batch_size"]) * 2, device, seed
    )
    reload_difference = float(np.abs(
        reload_animals[list(PROBABILITY_COLUMNS)].to_numpy()
        - best_animals[list(PROBABILITY_COLUMNS)].to_numpy()
    ).max())
    audit = {
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "full_seed": seed,
        "training_seed": training_seed,
        "best_validation_animal_cross_entropy": best_loss,
        "best_validation_animal_brier": brier(best_animals),
        "best_validation_animal_metrics": animal_metrics(best_animals),
        "checkpoint_reload_max_probability_difference": reload_difference,
        "best_model": model.audit(),
        "train_seconds": float(time.perf_counter() - started),
        "history": history,
        "outer_test_accessed": False,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return audit, best_animals, best_calls


def shared_head_state(model: LayerMixClassifier) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in model.state_dict().items()
        if key not in {"layer_logits", "residual_gate_raw"}
    }


def initialization_audit(
    protocol: dict[str, Any],
    store: RepresentationStore,
    train_indices: np.ndarray,
    probe_indices: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    models: dict[str, LayerMixClassifier] = {}
    logits = {}
    losses = {}
    labels = torch.from_numpy(store.labels[probe_indices])
    final = torch.from_numpy(store.frozen_embeddings[probe_indices])
    last4 = torch.from_numpy(store.last4_embeddings[probe_indices])
    for pipeline in PIPELINES:
        set_seed(seed)
        model = build_model(pipeline, protocol, store, train_indices).eval()
        models[pipeline] = model
        with torch.no_grad():
            logits[pipeline] = model(final, last4)
            losses[pipeline] = float(torch.nn.functional.cross_entropy(logits[pipeline], labels))
    parameters = {
        pipeline: sum(parameter.numel() for parameter in model.parameters())
        for pipeline, model in models.items()
    }
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"IDEA-077 parameter mismatch: {parameters}")
    reference_state = shared_head_state(models[PIPELINES[0]])
    head_equal = {
        pipeline: reference_state.keys() == shared_head_state(models[pipeline]).keys()
        and all(
            torch.equal(reference_state[key], shared_head_state(models[pipeline])[key])
            for key in reference_state
        )
        for pipeline in PIPELINES[1:]
    }
    difference = float(torch.max(torch.abs(logits[PIPELINES[2]] - logits[PIPELINES[0]])))
    loss_difference = abs(losses[PIPELINES[2]] - losses[PIPELINES[0]])
    if difference != 0.0 or loss_difference != 0.0 or not all(head_equal.values()):
        raise RuntimeError("IDEA-077 paired initialization audit failed")
    return {
        "trainable_parameters": parameters,
        "shared_head_state_equal": head_equal,
        "L1_minus_A0_max_initial_logit_difference": difference,
        "L1_minus_A0_initial_loss_difference": loss_difference,
        "initial_L1": models[PIPELINES[2]].audit(),
    }


def gradient_reachability_audit(
    protocol: dict[str, Any], store: RepresentationStore, train_indices: np.ndarray, seed: int
) -> dict[str, Any]:
    selected = train_indices[: min(32, len(train_indices))]
    labels = torch.from_numpy(store.labels[selected])
    final = torch.from_numpy(store.frozen_embeddings[selected])
    last4 = torch.from_numpy(store.last4_embeddings[selected])
    set_seed(seed)
    model = build_model(PIPELINES[2], protocol, store, train_indices).eval()
    loss = torch.nn.functional.cross_entropy(model(final, last4), labels)
    loss.backward()
    assert model.layer_logits is not None and model.residual_gate_raw is not None
    gate_grad_at_zero = float(model.residual_gate_raw.grad)
    logits_grad_at_zero = float(model.layer_logits.grad.abs().max())
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.residual_gate_raw.fill_(0.1)
    loss_nonzero = torch.nn.functional.cross_entropy(model(final, last4), labels)
    loss_nonzero.backward()
    logits_grad_after_gate = float(model.layer_logits.grad.abs().max())
    if not np.isfinite(gate_grad_at_zero) or abs(gate_grad_at_zero) <= 0.0:
        raise RuntimeError("IDEA-077 zero-init gate is not gradient reachable")
    if logits_grad_at_zero != 0.0 or not np.isfinite(logits_grad_after_gate) or logits_grad_after_gate <= 0.0:
        raise RuntimeError("IDEA-077 LayerMix logits gradient schedule failed")
    return {
        "gate_gradient_at_zero": gate_grad_at_zero,
        "layer_logits_max_gradient_at_zero": logits_grad_at_zero,
        "layer_logits_max_gradient_after_gate_probe": logits_grad_after_gate,
    }


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea077-ast-last4-layer-mix-v1":
        raise RuntimeError("Unexpected IDEA-077 protocol")
    if protocol.get("status") != "locked_before_initial_evaluation":
        raise RuntimeError("IDEA-077 protocol is not locked")
    model = protocol["model"]
    if tuple(model["pipelines"]) != PIPELINES or tuple(model["base_seeds"]) != BASE_SEEDS:
        raise RuntimeError("IDEA-077 pipeline or seed bank changed")
    if model["repeats"] != [0, 1, 2] or model["folds"] != [0, 1, 2, 3]:
        raise RuntimeError("IDEA-077 split scope changed")
    if model["outer_test_predictions"] is not False:
        raise RuntimeError("IDEA-077 must not access outer test predictions")
    if model["trainable_parameters"] != EXPECTED_PARAMETERS:
        raise RuntimeError("IDEA-077 parameter lock changed")
    if int(model["total_fits"]) != 108 or int(model["primary_A0_L1_fits"]) != 72:
        raise RuntimeError("IDEA-077 fit budget changed")
    if tuple(protocol["representation"]["layers_one_based"]) != (9, 10, 11, 12):
        raise RuntimeError("IDEA-077 layer set changed")
    digest = hashlib.sha256(model["seed_derivation_text"].encode("utf-8")).hexdigest()
    if digest != model["seed_derivation_sha256"]:
        raise RuntimeError("IDEA-077 seed digest changed")
    derived = {
        full_seed(base, repeat, fold)
        for base in BASE_SEEDS
        for repeat in model["repeats"]
        for fold in model["folds"]
    }
    excluded = {
        int(base) + 10_000 * repeat + 100 * fold
        for base in model["excluded_meow_base_seeds_IDEA068_through_IDEA076"]
        for repeat in range(3)
        for fold in range(4)
    } | {
        int(base) + 10_000 * repeat + 100 * fold
        for base in model["excluded_dog_base_seeds_IDEA075"]
        for repeat in range(3)
        for fold in range(5)
    }
    if len(derived) != 36 or derived & excluded:
        raise RuntimeError("IDEA-077 full seeds collide or are not unique")
    checks = {
        REPO_ROOT / protocol["dependencies"]["idea_path"]: protocol["dependencies"]["idea_sha256"],
        REPO_ROOT / protocol["data"]["roles_path"]: protocol["data"]["roles_sha256"],
        REPO_ROOT / protocol["data"]["frozen_embedding_path"]: protocol["data"]["frozen_embedding_sha256"],
        REPO_ROOT / protocol["data"]["layer_embedding_path"]: protocol["data"]["layer_embedding_sha256"],
        REPO_ROOT / protocol["dependencies"]["layer_extractor_path"]: protocol["dependencies"]["layer_extractor_sha256"],
        Path(__file__).resolve(): protocol["dependencies"]["runner_sha256"],
    }
    for path, expected in checks.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"IDEA-077 dependency checksum mismatch: {path}")


def validate_roles(protocol: dict[str, Any], store: RepresentationStore, roles: pd.DataFrame) -> int:
    if set(roles["role"].astype(str)) != {"train", "validation", "test"}:
        raise RuntimeError("IDEA-077 role vocabulary changed")
    cells = 0
    for repeat in protocol["model"]["repeats"]:
        for fold in protocol["model"]["folds"]:
            cell = roles[(roles["repeat"] == repeat) & (roles["outer_fold"] == fold)]
            sets = {
                role: set(cell[cell["role"] == role]["cat_id"].astype(str))
                for role in ("train", "validation", "test")
            }
            if any(sets[a] & sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
                raise RuntimeError("IDEA-077 cat leakage across roles")
            indices = fold_indices(store, roles, repeat, fold)
            if np.intersect1d(indices["train"], indices["validation"]).size:
                raise RuntimeError("IDEA-077 call leakage across roles")
            cells += 1
    return cells


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    store, cache_audit = load_store(protocol)
    if len(np.unique(store.cat_ids)) != 111:
        raise RuntimeError("IDEA-077 expected 111 cats")
    roles = pd.read_csv(REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    cells = validate_roles(protocol, store, roles)
    indices = fold_indices(store, roles, 0, 0)
    probe = indices["validation"][: min(32, len(indices["validation"]))]
    initialization = initialization_audit(protocol, store, indices["train"], probe, BASE_SEEDS[0])
    gradients = gradient_reachability_audit(protocol, store, indices["train"], BASE_SEEDS[0])
    device = resolve_device(args.device)
    return {
        "status": "GO",
        "read_only": True,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "device": str(device),
        "gpu_used": False if device.type == "cpu" else True,
        "calls": len(store.call_ids),
        "cats": int(len(np.unique(store.cat_ids))),
        "role_cells": cells,
        "cache_shape": list(store.last4_embeddings.shape),
        "cache_last_layer_vs_locked_final": cache_audit,
        "new_hidden_state_cache_required": False,
        "base_seeds": list(BASE_SEEDS),
        "unique_full_seeds": 36,
        "expected_fits": 108,
        "primary_A0_L1_fits": 72,
        "outer_test_accessed": False,
        "initialization": initialization,
        "gradient_reachability": gradients,
    }


def metric_bundle(frame: pd.DataFrame) -> dict[str, Any]:
    return {"metrics": animal_metrics(frame), "cross_entropy": animal_cross_entropy(frame), "brier": brier(frame)}


def add_metrics(row: dict[str, Any], bundles: dict[str, Any]) -> None:
    for pipeline in PIPELINES:
        row[f"{pipeline}_macro_f1"] = bundles[pipeline]["metrics"]["macro_f1"]
        row[f"{pipeline}_balanced_accuracy"] = bundles[pipeline]["metrics"]["balanced_accuracy"]
        row[f"{pipeline}_cross_entropy"] = bundles[pipeline]["cross_entropy"]
        row[f"{pipeline}_brier"] = bundles[pipeline]["brier"]
        row[f"{pipeline}_senior_recall"] = bundles[pipeline]["metrics"]["per_class"]["senior"]["recall"]
    row["L1_minus_A0_macro_f1"] = row[f"{PIPELINES[2]}_macro_f1"] - row[f"{PIPELINES[0]}_macro_f1"]
    row["M0_minus_A0_macro_f1"] = row[f"{PIPELINES[1]}_macro_f1"] - row[f"{PIPELINES[0]}_macro_f1"]
    row["L1_minus_M0_macro_f1"] = row[f"{PIPELINES[2]}_macro_f1"] - row[f"{PIPELINES[1]}_macro_f1"]


def aggregate(fits: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    by_key = {(fit["pipeline"], fit["base_seed"], fit["repeat"], fit["fold"]): fit for fit in fits}
    fold_rows = []
    seed_repeat_rows = []
    pooled_all: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
    for base_seed in protocol["model"]["base_seeds"]:
        for repeat in protocol["model"]["repeats"]:
            grouped: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
            for fold in protocol["model"]["folds"]:
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
                    tagged["base_seed"], tagged["repeat"], tagged["fold"] = base_seed, repeat, fold
                    grouped[pipeline].append(tagged)
                    pooled_all[pipeline].append(tagged)
            pooled = {pipeline: pd.concat(parts, ignore_index=True) for pipeline, parts in grouped.items()}
            row = {"base_seed": base_seed, "repeat": repeat}
            add_metrics(row, {pipeline: metric_bundle(frame) for pipeline, frame in pooled.items()})
            seed_repeat_rows.append(row)
    folds = pd.DataFrame(fold_rows)
    seed_repeats = pd.DataFrame(seed_repeat_rows)
    comparison_columns = ("L1_minus_A0_macro_f1", "M0_minus_A0_macro_f1", "L1_minus_M0_macro_f1")
    split_cells = folds.groupby(["repeat", "fold"], as_index=False)[list(comparison_columns)].mean()
    mean_metrics = {
        metric: {pipeline: float(seed_repeats[f"{pipeline}_{metric}"].mean()) for pipeline in PIPELINES}
        for metric in ("macro_f1", "balanced_accuracy", "cross_entropy", "brier")
    }
    per_seed_delta = {
        str(seed): float(seed_repeats[seed_repeats["base_seed"] == seed]["L1_minus_A0_macro_f1"].mean())
        for seed in BASE_SEEDS
    }
    per_seed_senior = {}
    pooled_frames = {pipeline: pd.concat(parts, ignore_index=True) for pipeline, parts in pooled_all.items()}
    for seed in BASE_SEEDS:
        senior = {
            pipeline: animal_metrics(frame[frame["base_seed"] == seed])["per_class"]["senior"]["recall"]
            for pipeline, frame in pooled_frames.items()
        }
        per_seed_senior[str(seed)] = float(senior[PIPELINES[2]] - senior[PIPELINES[0]])
    delta = seed_repeats["L1_minus_A0_macro_f1"]
    split_delta = split_cells["L1_minus_A0_macro_f1"]
    gate = protocol["gate"]
    conditions = {
        "mean_macro_f1_gain": float(delta.mean()) >= float(gate["minimum_mean_seed_repeat_L1_minus_A0"]),
        "positive_base_seed_means": sum(value > 0 for value in per_seed_delta.values()) >= int(gate["minimum_positive_base_seeds"]),
        "positive_seed_repeats": int((delta > 0).sum()) >= int(gate["minimum_positive_seed_repeats"]),
        "nonnegative_split_cells": int((split_delta >= 0).sum()) >= int(gate["minimum_nonnegative_split_cells"]),
        "worst_split_cell": float(split_delta.min()) >= float(gate["minimum_worst_split_cell_delta"]),
        "cross_entropy_nonworse": mean_metrics["cross_entropy"][PIPELINES[2]] <= mean_metrics["cross_entropy"][PIPELINES[0]],
        "brier_nonworse": mean_metrics["brier"][PIPELINES[2]] <= mean_metrics["brier"][PIPELINES[0]],
        "per_base_seed_senior_recall": all(value >= float(gate["minimum_per_base_seed_senior_recall_delta"]) for value in per_seed_senior.values()),
    }
    comparisons = {}
    for column in comparison_columns:
        values = seed_repeats[column]
        comparisons[column] = {
            "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
            "median": float(values.median()), "positive": int((values > 0).sum()),
            "tied": int((values == 0).sum()), "negative": int((values < 0).sum()),
            "worst": float(values.min()), "best": float(values.max()),
        }
    layer_mix_audits = [fit["audit"]["best_model"] for fit in fits if fit["pipeline"] == PIPELINES[2]]
    return {
        "status": "complete", "outer_test_accessed": False, "fits": len(fits),
        "seed_repeat_estimates": len(seed_repeat_rows), "split_cell_estimates": int(len(split_cells)),
        "independence_note": "Repeated validation animals across folds, seeds, and repeats are descriptive occurrences, not independent samples.",
        "fold_results": fold_rows, "seed_repeat_results": seed_repeat_rows,
        "split_cell_results": split_cells.to_dict(orient="records"),
        "seed_repeat_equal_weight_means": mean_metrics, "comparisons": comparisons,
        "per_base_seed_L1_minus_A0_macro_f1": per_seed_delta,
        "per_base_seed_L1_minus_A0_senior_recall": per_seed_senior,
        "layer_mix_best_checkpoint_audits": layer_mix_audits,
        "gate_conditions": conditions, "gate_passed": bool(all(conditions.values())),
    }


def resolve_run_root(output_subdir: str) -> Path:
    runs = (REPO_ROOT / "runs").resolve()
    root = (runs / output_subdir).resolve()
    if runs not in root.parents:
        raise ValueError("--output-subdir must stay below runs")
    return root


def validate_completed_fit(fit: dict[str, Any], pipeline: str, base_seed: int, seed: int, repeat: int, fold: int) -> None:
    expected = {"status": "complete", "pipeline": pipeline, "base_seed": base_seed, "full_seed": seed, "repeat": repeat, "fold": fold, "outer_test_accessed": False}
    if any(fit.get(key) != value for key, value in expected.items()):
        raise RuntimeError("IDEA-077 resume fit identity mismatch")
    for prefix in ("validation_animal", "validation_call"):
        path = REPO_ROOT / fit[f"{prefix}_predictions"]
        if not path.is_file() or sha256(path) != fit[f"{prefix}_sha256"]:
            raise RuntimeError(f"IDEA-077 resume prediction mismatch: {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    store, cache_audit = load_store(protocol)
    roles = pd.read_csv(REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    validate_roles(protocol, store, roles)
    device = resolve_device(args.device)
    root = resolve_run_root(args.output_subdir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol_id": protocol["protocol_id"], "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()), "git_revision": git_revision(),
        "outer_test_accessed": False, "model": protocol["model"], "cache_audit": cache_audit,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        if not args.resume or read_json(manifest_path) != manifest:
            raise RuntimeError("Existing IDEA-077 run manifest differs")
    else:
        write_json(manifest_path, manifest)
    completed = []
    for base_seed in protocol["model"]["base_seeds"]:
        for repeat in protocol["model"]["repeats"]:
            for fold in protocol["model"]["folds"]:
                indices = fold_indices(store, roles, repeat, fold)
                seed = full_seed(base_seed, repeat, fold)
                init = initialization_audit(protocol, store, indices["train"], indices["validation"][:32], seed)
                for pipeline in PIPELINES:
                    output = root / "fits" / pipeline / f"base_seed_{base_seed}" / f"repeat_{repeat}" / f"fold_{fold}"
                    summary_path = output / "fit_summary.json"
                    if summary_path.exists():
                        if not args.resume:
                            raise FileExistsError(summary_path)
                        fit = read_json(summary_path)
                        validate_completed_fit(fit, pipeline, base_seed, seed, repeat, fold)
                        completed.append(fit)
                        continue
                    output.mkdir(parents=True, exist_ok=True)
                    print(f"=== {pipeline} base_seed={base_seed} repeat={repeat} fold={fold} full_seed={seed} ===", flush=True)
                    audit, animals, calls = fit_inner(pipeline, protocol, store, indices["train"], indices["validation"], device, seed)
                    animal_path = output / "validation_animal_predictions.csv"
                    call_path = output / "validation_call_predictions.csv"
                    animals.to_csv(animal_path, index=False, lineterminator="\n")
                    calls.to_csv(call_path, index=False, lineterminator="\n")
                    fit = {
                        "status": "complete", "pipeline": pipeline, "base_seed": int(base_seed),
                        "full_seed": seed, "repeat": repeat, "fold": fold, "outer_test_accessed": False,
                        "initialization_audit": init, "audit": audit,
                        "validation_animal_predictions": repo_relative(animal_path), "validation_animal_sha256": sha256(animal_path),
                        "validation_call_predictions": repo_relative(call_path), "validation_call_sha256": sha256(call_path),
                    }
                    write_json(summary_path, fit)
                    completed.append(fit)
    summary = aggregate(completed, protocol)
    write_json(root / "initial_evaluation_summary.json", summary)
    write_json(root / "run_summary.json", {"status": "complete", "completed_fits": len(completed), "expected_fits": 108, "outer_test_accessed": False, "gate_passed": summary["gate_passed"]})
    return summary


def main() -> None:
    args = parse_args()
    result = preflight(args) if args.stage == "preflight" else run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
