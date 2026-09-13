"""Run IDEA-051 cat-level set aggregation on frozen AST call embeddings."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.stats import spearmanr
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
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for local_root in (SCRIPTS_ROOT, SRC_ROOT):
    if str(local_root) not in sys.path:
        sys.path.insert(0, str(local_root))

import run_meowagenet_ast_cat_balance_global_weighting_v1 as reference  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT / "configs" / "protocol" / "meowagenet_idea051_cat_set_v1.json"
)
PLAN_PATH = REPO_ROOT / "plan" / "IDEA-051_cat_level_set_aggregation.md"
DIAGNOSTIC_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_idea051_cat_set_diagnostics_v1.json"
)
DIAGNOSTIC_SCRIPT_PATH = SCRIPTS_ROOT / "diagnose_meowagenet_idea051_cat_sets.py"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
REFERENCE_RUNNER_PATH = (
    SCRIPTS_ROOT / "run_meowagenet_ast_cat_balance_global_weighting_v1.py"
)
RUNS_ROOT = REPO_ROOT / "runs"
PIPELINES = (
    "S0_call_probability_mean",
    "S1_hidden_mean_set",
    "S2_attention_set",
)
SET_PIPELINES = PIPELINES[1:]
LABEL_NAMES = ("kitten", "adult", "senior")
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "evaluate"), required=True)
    parser.add_argument("--output-subdir", default="meowagenet_idea051_cat_set_v1")
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
    if protocol["protocol_id"] != "meowagenet-idea051-cat-set-v1":
        raise RuntimeError("Unexpected IDEA-051 protocol")
    if tuple(protocol["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-051 pipeline matrix changed")
    checks = {
        PLAN_PATH: protocol["idea"]["sha256"],
        DIAGNOSTIC_PATH: protocol["diagnostic"]["sha256"],
        DIAGNOSTIC_SCRIPT_PATH: protocol["diagnostic"]["script_sha256"],
        ROLES_PATH: protocol["splits"]["roles_sha256"],
        REFERENCE_RUNNER_PATH: protocol["dependencies"]["reference_runner_sha256"],
        reference.historical.idea019.FEATURE_PATH: protocol["dependencies"]["fbank_sha256"],
        reference.historical.idea019.FROZEN_EMBEDDING_PATH: protocol["dependencies"][
            "frozen_embedding_sha256"
        ],
    }
    for path, expected in checks.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"IDEA-051 dependency checksum mismatch: {path}")
    evaluation = protocol["initial_evaluation"]
    expected_fits = (
        len(PIPELINES)
        * len(evaluation["base_seeds"])
        * len(evaluation["repeats"])
        * len(evaluation["outer_folds"])
    )
    if expected_fits != int(evaluation["total_outer_fits"]):
        raise RuntimeError("IDEA-051 fit budget is inconsistent")
    fixed = protocol["fixed_training"]
    if int(fixed["reference_call_micro_batch_size"]) * int(
        fixed["reference_gradient_accumulation_steps"]
    ) != int(fixed["reference_accumulation_window_calls"]):
        raise RuntimeError("Reference accumulation window is inconsistent")
    if int(fixed["set_cat_batch_size"]) != int(fixed["set_loss_denominator_cats"]):
        raise RuntimeError("Set batch and fixed denominator must match")


class DeterministicNoSingletonBatchSampler(Sampler[list[int]]):
    """Deterministic shuffled batches with no final singleton."""

    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        if size < 2:
            raise ValueError("A training split must contain at least two units")
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
            raise RuntimeError("A singleton training batch remains")
        yield from batches

    def __len__(self) -> int:
        return math.ceil(self.size / self.batch_size)


class CatSetDataset(Dataset[tuple[np.ndarray, int, str, np.ndarray]]):
    def __init__(self, store: Any, call_indices: np.ndarray) -> None:
        self.store = store
        selected = np.asarray(call_indices, dtype=np.int64)
        self.cat_ids = np.asarray(sorted(np.unique(store.cat_ids[selected]).tolist()))
        self.call_indices: list[np.ndarray] = []
        self.labels = []
        for cat_id in self.cat_ids:
            calls = selected[store.cat_ids[selected] == cat_id]
            labels = np.unique(store.labels[calls])
            if len(labels) != 1:
                raise RuntimeError(f"Cat {cat_id} has inconsistent labels")
            self.call_indices.append(np.sort(calls))
            self.labels.append(int(labels[0]))
        self.labels = np.asarray(self.labels, dtype=np.int64)
        if sum(len(calls) for calls in self.call_indices) != len(selected):
            raise RuntimeError("Cat sets do not preserve every selected call")

    def __len__(self) -> int:
        return len(self.cat_ids)

    def __getitem__(self, item: int) -> tuple[np.ndarray, int, str, np.ndarray]:
        calls = self.call_indices[item]
        return (
            self.store.frozen_embeddings[calls],
            int(self.labels[item]),
            str(self.cat_ids[item]),
            calls,
        )


def collate_cat_sets(
    rows: list[tuple[np.ndarray, int, str, np.ndarray]],
) -> dict[str, Any]:
    embeddings = []
    instance_to_cat = []
    labels = []
    cat_ids = []
    call_indices = []
    for local_cat, (cat_embeddings, label, cat_id, cat_calls) in enumerate(rows):
        embeddings.append(torch.from_numpy(cat_embeddings.astype(np.float32, copy=False)))
        instance_to_cat.extend([local_cat] * len(cat_embeddings))
        labels.append(label)
        cat_ids.append(cat_id)
        call_indices.extend(int(value) for value in cat_calls)
    return {
        "embeddings": torch.cat(embeddings, dim=0),
        "instance_to_cat": torch.tensor(instance_to_cat, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "cat_ids": cat_ids,
        "call_indices": torch.tensor(call_indices, dtype=torch.long),
    }


def build_set_loader(
    dataset: CatSetDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    if shuffle:
        return DataLoader(
            dataset,
            batch_sampler=DeterministicNoSingletonBatchSampler(
                len(dataset), batch_size, seed
            ),
            num_workers=0,
            collate_fn=collate_cat_sets,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_cat_sets,
    )


class SetClassifier(nn.Module):
    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        dropout: float,
        pooling: str,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "attention"}:
            raise ValueError(f"Unknown set pooling: {pooling}")
        safe_scale = np.where(scale > 1.0e-12, scale, 1.0).astype(np.float32)
        self.register_buffer("feature_mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("feature_scale", torch.from_numpy(safe_scale))
        self.input_layer = nn.Linear(768, 128)
        self.activation = nn.ReLU()
        self.normalization = nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, 3)
        self.pooling = pooling
        if pooling == "attention":
            self.attention = nn.Linear(128, 1)
            nn.init.zeros_(self.attention.weight)
            nn.init.zeros_(self.attention.bias)
        else:
            self.attention = None

    def encode_calls(self, embeddings: torch.Tensor) -> torch.Tensor:
        normalized = (embeddings - self.feature_mean) / self.feature_scale
        hidden = self.activation(self.input_layer(normalized))
        return self.dropout(self.normalization(hidden))

    def pool_calls(
        self, hidden: torch.Tensor, instance_to_cat: torch.Tensor, cat_count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = torch.bincount(instance_to_cat, minlength=cat_count).tolist()
        chunks = torch.split(hidden, [int(value) for value in counts])
        pooled = []
        weight_parts = []
        for chunk in chunks:
            if self.attention is None:
                weights = torch.full(
                    (len(chunk),),
                    1.0 / float(len(chunk)),
                    dtype=chunk.dtype,
                    device=chunk.device,
                )
            else:
                weights = torch.softmax(self.attention(chunk).squeeze(-1), dim=0)
            pooled.append((chunk * weights[:, None]).sum(dim=0))
            weight_parts.append(weights)
        return torch.stack(pooled), torch.cat(weight_parts)

    def forward(
        self, embeddings: torch.Tensor, instance_to_cat: torch.Tensor, cat_count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_calls(embeddings)
        pooled, weights = self.pool_calls(hidden, instance_to_cat, cat_count)
        return self.classifier(pooled), weights


def set_class_weights(dataset: CatSetDataset) -> np.ndarray:
    counts = np.bincount(dataset.labels, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("A cat training split is missing an age class")
    return (len(dataset) / (3.0 * counts)).astype(np.float32)


def set_target_weight_audit(
    dataset: CatSetDataset, class_weights: np.ndarray
) -> dict[str, Any]:
    cat_weights = class_weights[dataset.labels]
    return {
        "cats": int(len(dataset)),
        "calls": int(sum(len(calls) for calls in dataset.call_indices)),
        "weight_total": float(cat_weights.sum()),
        "weight_mean": float(cat_weights.mean()),
        "weight_by_class": {
            LABEL_NAMES[class_index]: float(cat_weights[dataset.labels == class_index].sum())
            for class_index in range(3)
        },
        "weight_by_cat": {
            str(cat_id): float(weight)
            for cat_id, weight in zip(dataset.cat_ids, cat_weights)
        },
    }


def globally_weighted_set_micro_loss(
    per_cat_loss: torch.Tensor,
    cat_weights: torch.Tensor,
    fixed_cat_denominator: int,
) -> torch.Tensor:
    return (per_cat_loss * cat_weights).sum() / float(fixed_cat_denominator)


def fixed_training_values(protocol: dict[str, Any]) -> dict[str, Any]:
    fixed = protocol["fixed_training"]
    return {
        "learning_rate": float(fixed["learning_rate"]),
        "optimizer_epsilon": float(fixed["optimizer_epsilon"]),
        "gradient_clip": float(fixed["gradient_clip"]),
        "max_epochs": int(fixed["maximum_epochs"]),
        "patience": int(fixed["early_stopping_patience"]),
        "call_batch_size": int(fixed["reference_call_micro_batch_size"]),
        "call_accumulation_steps": int(
            fixed["reference_gradient_accumulation_steps"]
        ),
        "call_window": int(fixed["reference_accumulation_window_calls"]),
        "cat_batch_size": int(fixed["set_cat_batch_size"]),
        "cat_denominator": int(fixed["set_loss_denominator_cats"]),
    }


def build_model(
    pipeline: str, protocol: dict[str, Any], store: Any, train_indices: np.ndarray
) -> nn.Module:
    embeddings = store.frozen_embeddings[train_indices]
    mean = embeddings.mean(axis=0)
    scale = embeddings.std(axis=0)
    dropout = float(protocol["fixed_training"]["dropout"])
    if pipeline == PIPELINES[0]:
        head = reference.historical.idea019.ClassificationHead(mean, scale, dropout)
        return reference.historical.idea019.FrozenClassifier(head)
    pooling = "mean" if pipeline == PIPELINES[1] else "attention"
    return SetClassifier(mean, scale, dropout, pooling)


def move_set_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **batch,
        "embeddings": batch["embeddings"].to(device),
        "instance_to_cat": batch["instance_to_cat"].to(device),
        "labels": batch["labels"].to(device),
        "call_indices": batch["call_indices"].to(device),
    }


def train_one_set_epoch(
    model: SetClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    class_weights_numpy: np.ndarray,
    dataset: CatSetDataset,
    device: torch.device,
    values: dict[str, Any],
    scaler: torch.cuda.amp.GradScaler,
) -> tuple[float, dict[str, Any]]:
    model.train()
    weighted_total = 0.0
    weight_total = 0.0
    processed_cats: list[str] = []
    processed_calls: list[int] = []
    batch_sizes = []
    for cpu_batch in loader:
        batch = move_set_batch(cpu_batch, device)
        labels = batch["labels"]
        weights = class_weights[labels]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits, _ = model(
                batch["embeddings"], batch["instance_to_cat"], len(labels)
            )
            per_cat_loss = torch.nn.functional.cross_entropy(
                logits, labels, reduction="none"
            )
            loss = globally_weighted_set_micro_loss(
                per_cat_loss, weights, values["cat_denominator"]
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), values["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()
        weighted_total += float((per_cat_loss.detach() * weights).sum())
        weight_total += float(weights.sum())
        processed_cats.extend(str(value) for value in cpu_batch["cat_ids"])
        processed_calls.extend(
            int(value) for value in cpu_batch["call_indices"].numpy().tolist()
        )
        batch_sizes.append(int(len(labels)))
    if len(processed_cats) != len(set(processed_cats)):
        raise RuntimeError("A set epoch processed a cat more than once")
    if len(processed_calls) != len(set(processed_calls)):
        raise RuntimeError("A set epoch processed a call more than once")
    cat_to_label = {
        str(cat_id): int(label) for cat_id, label in zip(dataset.cat_ids, dataset.labels)
    }
    coefficients = {
        cat_id: float(class_weights_numpy[cat_to_label[cat_id]])
        / float(values["cat_denominator"])
        for cat_id in processed_cats
    }
    return weighted_total / weight_total, {
        "processed_cats": int(len(processed_cats)),
        "unique_processed_cats": int(len(set(processed_cats))),
        "processed_calls": int(len(processed_calls)),
        "unique_processed_calls": int(len(set(processed_calls))),
        "batch_count": int(len(batch_sizes)),
        "batch_sizes": batch_sizes,
        "optimizer_steps": int(len(batch_sizes)),
        "final_batch_cats": int(batch_sizes[-1]),
        "final_batch_is_partial": bool(batch_sizes[-1] != values["cat_denominator"]),
        "effective_coefficient_total": float(sum(coefficients.values())),
        "effective_coefficient_by_class": {
            LABEL_NAMES[class_index]: float(
                sum(
                    coefficient
                    for cat_id, coefficient in coefficients.items()
                    if cat_to_label[cat_id] == class_index
                )
            )
            for class_index in range(3)
        },
        "effective_coefficient_by_cat": coefficients,
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
            raise RuntimeError(f"Cat {cat_id} has inconsistent call labels")
        probabilities = group[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float).mean(axis=0)
        rows.append(
            {
                "cat_id": str(cat_id),
                "true_label": int(labels[0]),
                "call_count": int(len(group)),
                **{
                    column: float(probabilities[index])
                    for index, column in enumerate(PROBABILITY_COLUMNS)
                },
                "predicted_label": int(probabilities.argmax()),
            }
        )
    return pd.DataFrame(rows)


def animal_metrics(animals: pd.DataFrame) -> dict[str, Any]:
    labels = animals["true_label"].to_numpy(dtype=np.int64)
    predictions = animals["predicted_label"].to_numpy(dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1, 2],
        zero_division=0,
    )
    return {
        "macro_f1": float(
            f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(labels, predictions, weights="quadratic")
        ),
        "plain_accuracy": float(accuracy_score(labels, predictions)),
        "per_class": {
            LABEL_NAMES[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(3)
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
        "n": int(len(animals)),
    }


def animal_cross_entropy(animals: pd.DataFrame) -> float:
    probabilities = animals[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    labels = animals["true_label"].to_numpy(dtype=np.int64)
    return float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1.0e-12, 1.0)).mean())


def attention_summary(attention: pd.DataFrame | None) -> dict[str, Any] | None:
    if attention is None:
        return None
    rows = []
    for cat_id, group in attention.groupby("cat_id", sort=True):
        weights = group["attention_weight"].to_numpy(dtype=float)
        if not np.isclose(weights.sum(), 1.0, atol=1.0e-3):
            raise RuntimeError(f"Attention weights do not sum to one for {cat_id}")
        entropy = float(-(weights * np.log(np.clip(weights, 1.0e-12, 1.0))).sum())
        normalized_entropy = entropy / math.log(len(weights)) if len(weights) > 1 else 1.0
        rows.append(
            {
                "cat_id": str(cat_id),
                "call_count": int(len(weights)),
                "max_weight": float(weights.max()),
                "normalized_entropy": float(normalized_entropy),
                "effective_calls": float(math.exp(entropy)),
                "mean_abs_deviation_from_uniform": float(
                    np.abs(weights - 1.0 / len(weights)).mean()
                ),
            }
        )
    frame = pd.DataFrame(rows)
    association = spearmanr(frame["call_count"], frame["max_weight"])
    return {
        "cats": int(len(frame)),
        "mean_max_weight": float(frame["max_weight"].mean()),
        "mean_normalized_entropy": float(frame["normalized_entropy"].mean()),
        "mean_effective_calls": float(frame["effective_calls"].mean()),
        "mean_abs_deviation_from_uniform": float(
            frame["mean_abs_deviation_from_uniform"].mean()
        ),
        "call_count_vs_max_weight": {
            "rho": float(association.statistic),
            "pvalue": float(association.pvalue),
        },
    }


def predict_set_model(
    model: SetClassifier,
    dataset: CatSetDataset,
    store: Any,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    loader = build_set_loader(dataset, batch_size, False, seed)
    model.eval()
    animal_rows = []
    attention_rows = []
    loss_total = 0.0
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_set_batch(cpu_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits, weights = model(
                    batch["embeddings"], batch["instance_to_cat"], len(batch["labels"])
                )
                loss_total += float(
                    torch.nn.functional.cross_entropy(
                        logits, batch["labels"], reduction="sum"
                    )
                )
                probabilities = torch.softmax(logits, dim=1).float().cpu().numpy()
            weights_numpy = weights.float().cpu().numpy()
            call_indices = cpu_batch["call_indices"].numpy().astype(np.int64)
            instance_to_cat = cpu_batch["instance_to_cat"].numpy().astype(np.int64)
            labels = cpu_batch["labels"].numpy().astype(np.int64)
            for local_cat, cat_id in enumerate(cpu_batch["cat_ids"]):
                mask = instance_to_cat == local_cat
                cat_calls = call_indices[mask]
                animal_rows.append(
                    {
                        "cat_id": str(cat_id),
                        "true_label": int(labels[local_cat]),
                        "call_count": int(mask.sum()),
                        **{
                            column: float(probabilities[local_cat, index])
                            for index, column in enumerate(PROBABILITY_COLUMNS)
                        },
                        "predicted_label": int(probabilities[local_cat].argmax()),
                    }
                )
                for call_index, weight in zip(cat_calls, weights_numpy[mask]):
                    attention_rows.append(
                        {
                            "call_index": int(call_index),
                            "call_id": str(store.call_ids[call_index]),
                            "cat_id": str(cat_id),
                            "true_label": int(labels[local_cat]),
                            "call_count": int(mask.sum()),
                            "attention_weight": float(weight),
                        }
                    )
    animals = pd.DataFrame(animal_rows).sort_values("cat_id").reset_index(drop=True)
    attention = pd.DataFrame(attention_rows).sort_values("call_index").reset_index(drop=True)
    return loss_total / len(dataset), animals, attention


def predict_pipeline(
    pipeline: str,
    model: nn.Module,
    store: Any,
    indices: np.ndarray,
    device: torch.device,
    values: dict[str, Any],
    seed: int,
) -> tuple[float, pd.DataFrame, pd.DataFrame, str]:
    if pipeline == PIPELINES[0]:
        loader = reference.historical.idea019.build_loader(
            store, indices, "frozen", values["call_batch_size"], False, seed
        )
        _, calls = reference.historical.idea019.predict_calls(model, loader, store, device)
        animals = calls_to_animals(calls)
        return animal_cross_entropy(animals), animals, calls, "call_probabilities"
    dataset = CatSetDataset(store, indices)
    loss, animals, attention = predict_set_model(
        model,
        dataset,
        store,
        device,
        values["cat_batch_size"] * 2,
        seed,
    )
    return loss, animals, attention, "call_attention"


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def fit_inner(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
    max_epochs_override: int | None = None,
) -> tuple[int, dict[str, Any], dict[str, torch.Tensor], pd.DataFrame]:
    reference.historical.set_seed(seed)
    values = fixed_training_values(protocol)
    max_epochs = max_epochs_override or values["max_epochs"]
    model = build_model(pipeline, protocol, store, train_indices).to(device)
    parameters = reference.historical.idea019.trainable_counts(model)
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=values["learning_rate"], eps=values["optimizer_epsilon"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    if pipeline == PIPELINES[0]:
        call_lookup_numpy = reference.global_class_balanced_call_weights(
            store.labels, train_indices
        )
        call_lookup = torch.from_numpy(call_lookup_numpy).to(device)
        train_loader = reference.build_train_loader(
            store, train_indices, values["call_batch_size"], seed
        )
        target_audit = reference.target_weight_audit(
            call_lookup_numpy, store, train_indices
        )
        set_dataset = None
        set_weights_numpy = None
        set_weights = None
    else:
        set_dataset = CatSetDataset(store, train_indices)
        set_weights_numpy = set_class_weights(set_dataset)
        set_weights = torch.from_numpy(set_weights_numpy).to(device)
        train_loader = build_set_loader(
            set_dataset, values["cat_batch_size"], True, seed
        )
        target_audit = set_target_weight_audit(set_dataset, set_weights_numpy)
        call_lookup_numpy = None
        call_lookup = None

    best_loss = float("inf")
    best_epoch = 1
    best_metrics: dict[str, Any] = {}
    best_state = cpu_state_dict(model)
    best_attention: dict[str, Any] | None = None
    history = []
    epochs_without_improvement = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, max_epochs + 1):
        if pipeline == PIPELINES[0]:
            train_loss, train_audit = reference.train_one_epoch_global_weighted(
                model,
                train_loader,
                optimizer,
                call_lookup,
                call_lookup_numpy,
                store,
                device,
                {
                    "accumulation_window_calls": values["call_window"],
                    "accumulation_steps": values["call_accumulation_steps"],
                    "gradient_clip": values["gradient_clip"],
                },
                scaler,
            )
        else:
            train_loss, train_audit = train_one_set_epoch(
                model,
                train_loader,
                optimizer,
                set_weights,
                set_weights_numpy,
                set_dataset,
                device,
                values,
                scaler,
            )
        validation_loss, validation_animals, validation_detail, _ = predict_pipeline(
            pipeline,
            model,
            store,
            validation_indices,
            device,
            values,
            seed,
        )
        metrics = animal_metrics(validation_animals)
        epoch_attention = (
            attention_summary(validation_detail) if pipeline in SET_PIPELINES else None
        )
        history.append(
            {
                "epoch": epoch,
                "train_weighted_loss": train_loss,
                "validation_animal_cross_entropy": validation_loss,
                "validation_animal_macro_f1": metrics["macro_f1"],
                "validation_animal_balanced_accuracy": metrics["balanced_accuracy"],
                "validation_animal_qwk": metrics["quadratic_weighted_kappa"],
                "train_unit_audit": train_audit,
                "validation_attention": epoch_attention,
            }
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_metrics = metrics
            best_state = cpu_state_dict(model)
            best_attention = epoch_attention
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
    _, best_animals, _, _ = predict_pipeline(
        pipeline, model, store, validation_indices, device, values, seed
    )
    audit = {
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(len(history)),
        "best_validation_animal_cross_entropy": float(best_loss),
        "best_validation_animal_metrics": best_metrics,
        "best_validation_attention": best_attention,
        "history": history,
        "target_weight_audit": target_audit,
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
    return best_epoch, audit, best_state, best_animals


def fit_outer_and_predict(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    epochs: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, Any]]:
    reference.historical.set_seed(seed)
    values = fixed_training_values(protocol)
    model = build_model(pipeline, protocol, store, train_indices).to(device)
    parameters = reference.historical.idea019.trainable_counts(model)
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=values["learning_rate"], eps=values["optimizer_epsilon"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    if pipeline == PIPELINES[0]:
        call_lookup_numpy = reference.global_class_balanced_call_weights(
            store.labels, train_indices
        )
        call_lookup = torch.from_numpy(call_lookup_numpy).to(device)
        train_loader = reference.build_train_loader(
            store, train_indices, values["call_batch_size"], seed
        )
        target_audit = reference.target_weight_audit(
            call_lookup_numpy, store, train_indices
        )
        set_dataset = None
        set_weights_numpy = None
        set_weights = None
    else:
        set_dataset = CatSetDataset(store, train_indices)
        set_weights_numpy = set_class_weights(set_dataset)
        set_weights = torch.from_numpy(set_weights_numpy).to(device)
        train_loader = build_set_loader(
            set_dataset, values["cat_batch_size"], True, seed
        )
        target_audit = set_target_weight_audit(set_dataset, set_weights_numpy)
        call_lookup_numpy = None
        call_lookup = None
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        if pipeline == PIPELINES[0]:
            train_loss, train_audit = reference.train_one_epoch_global_weighted(
                model,
                train_loader,
                optimizer,
                call_lookup,
                call_lookup_numpy,
                store,
                device,
                {
                    "accumulation_window_calls": values["call_window"],
                    "accumulation_steps": values["call_accumulation_steps"],
                    "gradient_clip": values["gradient_clip"],
                },
                scaler,
            )
        else:
            train_loss, train_audit = train_one_set_epoch(
                model,
                train_loader,
                optimizer,
                set_weights,
                set_weights_numpy,
                set_dataset,
                device,
                values,
                scaler,
            )
        history.append(
            {
                "epoch": epoch,
                "train_weighted_loss": train_loss,
                "train_unit_audit": train_audit,
            }
        )
        print(
            f"{pipeline} outer epoch={epoch}/{epochs} train={train_loss:.4f}",
            flush=True,
        )
    test_loss, animals, detail, detail_kind = predict_pipeline(
        pipeline, model, store, test_indices, device, values, seed
    )
    metrics = animal_metrics(animals)
    audit = {
        "epochs": int(epochs),
        "history": history,
        "test_animal_cross_entropy": float(test_loss),
        "test_animal_metrics": metrics,
        "test_attention": attention_summary(detail)
        if pipeline in SET_PIPELINES
        else None,
        "target_weight_audit": target_audit,
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
    return animals, detail, detail_kind, audit


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
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "git_revision": git_revision(),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "idea_sha256": protocol["idea"]["sha256"],
        "diagnostic_sha256": protocol["diagnostic"]["sha256"],
        "roles_sha256": protocol["splits"]["roles_sha256"],
    }


def reload_probability_difference(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
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
    values = fixed_training_values(protocol)
    _, after, _, _ = predict_pipeline(
        pipeline, model, store, validation_indices, device, values, seed
    )
    left = before.sort_values("cat_id")[list(PROBABILITY_COLUMNS)].to_numpy()
    right = after.sort_values("cat_id")[list(PROBABILITY_COLUMNS)].to_numpy()
    difference = float(np.abs(left - right).max())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return difference


def run_smoke(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    smoke_root = run_root / "smoke"
    summary_path = smoke_root / "summary.json"
    if summary_path.is_file() and not resume:
        raise FileExistsError(summary_path)
    smoke = protocol["smoke"]
    indices = reference.historical.fold_indices(
        store,
        roles,
        int(smoke["repeat"]),
        int(smoke["outer_fold"]),
        include_test=False,
    )
    seed = reference.historical.full_seed(
        int(smoke["base_seed"]), int(smoke["repeat"]), int(smoke["outer_fold"])
    )
    fits = []
    for pipeline in PIPELINES:
        output_dir = smoke_root / "fits" / pipeline
        fit_path = output_dir / "fit_summary.json"
        checkpoint_path = output_dir / "best_checkpoint.pt"
        best_epoch, inner, state, validation_animals = fit_inner(
            pipeline,
            protocol,
            store,
            indices["train"],
            indices["validation"],
            device,
            seed,
            max_epochs_override=int(smoke["epochs"]),
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
            validation_animals,
            device,
            seed,
        )
        fit = {
            "status": "complete",
            "stage": "inner_only_smoke",
            "outer_test_accessed": False,
            "pipeline": pipeline,
            "base_seed": int(smoke["base_seed"]),
            "repeat": int(smoke["repeat"]),
            "outer_fold": int(smoke["outer_fold"]),
            "full_seed": int(seed),
            "selected_epoch": int(best_epoch),
            "inner": inner,
            "validation_animals": int(len(validation_animals)),
            "checkpoint_path": repo_relative(checkpoint_path),
            "checkpoint_reload_max_probability_difference": reload_difference,
        }
        write_json(fit_path, fit)
        fits.append(fit)
    set_pair_match = all(
        fits[1]["inner"]["history"][epoch]["train_unit_audit"]["cat_order_sha256"]
        == fits[2]["inner"]["history"][epoch]["train_unit_audit"]["cat_order_sha256"]
        for epoch in range(int(smoke["epochs"]))
    )
    if not set_pair_match:
        raise RuntimeError("Smoke S1/S2 cat batch order differs")
    full_dataset = CatSetDataset(store, np.arange(len(store.call_ids), dtype=np.int64))
    smoke_root.mkdir(parents=True, exist_ok=True)
    environment_path = run_root / "environment_lock.json"
    write_json(environment_path, environment_lock(protocol, device))
    code_commit = git_revision()
    if code_commit is None:
        raise RuntimeError("A Git commit is required before smoke lock")
    lock = {
        "schema_version": "1.0",
        "status": "locked_for_idea051_initial_evaluation",
        "outer_test_accessed": False,
        "protocol_id": protocol["protocol_id"],
        "code_commit": code_commit,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "idea_sha256": sha256(PLAN_PATH),
        "diagnostic_sha256": sha256(DIAGNOSTIC_PATH),
        "roles_sha256": sha256(ROLES_PATH),
        "environment_lock_sha256": sha256(environment_path),
        "pipelines": list(PIPELINES),
        "initial_evaluation": protocol["initial_evaluation"],
        "fixed_training": protocol["fixed_training"],
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
        "variable_set_audit": {
            "cats": int(len(full_dataset)),
            "calls": int(sum(len(calls) for calls in full_dataset.call_indices)),
            "single_call_cats": int(
                sum(len(calls) == 1 for calls in full_dataset.call_indices)
            ),
            "maximum_calls_per_cat": int(
                max(len(calls) for calls in full_dataset.call_indices)
            ),
        },
        "set_pair_batch_order_match": set_pair_match,
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
        raise FileNotFoundError("Run smoke before IDEA-051 evaluation")
    lock = read_json(lock_path)
    if lock["status"] != "locked_for_idea051_initial_evaluation":
        raise RuntimeError("IDEA-051 execution lock status is invalid")
    checks = {
        PROTOCOL_PATH: lock["protocol_sha256"],
        Path(__file__).resolve(): lock["runner_sha256"],
        PLAN_PATH: lock["idea_sha256"],
        DIAGNOSTIC_PATH: lock["diagnostic_sha256"],
        ROLES_PATH: lock["roles_sha256"],
    }
    for path, expected in checks.items():
        if sha256(path) != expected:
            raise RuntimeError(f"IDEA-051 execution-lock file changed: {path}")
    revision = lock["code_commit"]
    for path in (PROTOCOL_PATH, Path(__file__).resolve(), PLAN_PATH, DIAGNOSTIC_PATH):
        if git_blob_object_id(revision, path) != worktree_blob_object_id(path):
            raise RuntimeError(f"Locked commit blob differs from worktree: {path}")
    if lock["initial_evaluation"] != protocol["initial_evaluation"]:
        raise RuntimeError("IDEA-051 evaluation matrix differs from lock")
    return lock


def assert_set_pair_batch_orders(first: dict[str, Any], second: dict[str, Any]) -> dict[str, int]:
    result = {}
    for phase in ("inner", "outer"):
        left = first[phase]["history"]
        right = second[phase]["history"]
        common = min(len(left), len(right))
        for epoch in range(common):
            left_hash = left[epoch]["train_unit_audit"]["cat_order_sha256"]
            right_hash = right[epoch]["train_unit_audit"]["cat_order_sha256"]
            if left_hash != right_hash:
                raise RuntimeError(f"S1/S2 {phase} cat order differs at epoch {epoch + 1}")
        result[f"{phase}_common_epochs"] = common
    return result


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for metric in (
        "macro_f1",
        "balanced_accuracy",
        "quadratic_weighted_kappa",
        "plain_accuracy",
    ):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_sample_sd"] = float(values.std(ddof=1))
        result[f"{metric}_range"] = [float(values.min()), float(values.max())]
    result["mean_per_class"] = {
        label: {
            field: float(np.mean([row["per_class"][label][field] for row in rows]))
            for field in ("precision", "recall", "f1")
        }
        for label in LABEL_NAMES
    }
    return result


def bag_size_group(call_count: int) -> str:
    if call_count == 1:
        return "1"
    if call_count <= 4:
        return "2-4"
    if call_count <= 9:
        return "5-9"
    return "10+"


def bag_size_metrics(animals: pd.DataFrame) -> dict[str, Any]:
    result = {}
    for group_name in ("1", "2-4", "5-9", "10+"):
        group = animals[
            animals["call_count"].map(bag_size_group).astype(str) == group_name
        ]
        metrics = animal_metrics(group)
        result[group_name] = {
            "cat_evaluations": int(len(group)),
            "unique_cats": int(group["cat_id"].nunique()),
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "plain_accuracy": metrics["plain_accuracy"],
        }
    return result


def paired_cat_bootstrap(
    animals_by_key: dict[tuple[str, int], pd.DataFrame],
    left_pipeline: str,
    right_pipeline: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    repeats = [int(value) for value in protocol["initial_evaluation"]["repeats"]]
    reference_frame = animals_by_key[(left_pipeline, repeats[0])].sort_values("cat_id")
    labels = reference_frame["true_label"].to_numpy(dtype=np.int64)
    left_predictions = []
    right_predictions = []
    for repeat in repeats:
        left = animals_by_key[(left_pipeline, repeat)].sort_values("cat_id")
        right = animals_by_key[(right_pipeline, repeat)].sort_values("cat_id")
        if not np.array_equal(left["cat_id"].to_numpy(), right["cat_id"].to_numpy()):
            raise RuntimeError("Paired bootstrap cat order differs")
        left_predictions.append(left["predicted_label"].to_numpy(dtype=np.int64))
        right_predictions.append(right["predicted_label"].to_numpy(dtype=np.int64))
    settings = protocol["bootstrap"]
    generator = np.random.default_rng(int(settings["seed"]))
    differences = []
    for _ in range(int(settings["iterations"])):
        sampled = generator.integers(0, len(labels), size=len(labels))
        per_repeat = []
        for left, right in zip(left_predictions, right_predictions):
            left_score = f1_score(
                labels[sampled],
                left[sampled],
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
            right_score = f1_score(
                labels[sampled],
                right[sampled],
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
            per_repeat.append(float(right_score - left_score))
        differences.append(float(np.mean(per_repeat)))
    values = np.asarray(differences)
    return {
        "left": left_pipeline,
        "right": right_pipeline,
        "iterations": int(settings["iterations"]),
        "seed": int(settings["seed"]),
        "mean_difference": float(values.mean()),
        "interval": [
            float(np.quantile(values, float(settings["interval"][0]))),
            float(np.quantile(values, float(settings["interval"][1]))),
        ],
        "fraction_above_zero": float(np.mean(values > 0)),
    }


def raw_prediction_inventory(evaluation_root: Path) -> dict[str, Any]:
    paths = sorted(evaluation_root.rglob("*.csv"))
    entries = [
        {
            "path": repo_relative(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        }
        for path in paths
    ]
    canonical = "".join(f"{row['path']}:{row['sha256']}\n" for row in entries).encode()
    return {
        "schema_version": "1.0",
        "files": len(entries),
        "bytes": int(sum(row["bytes"] for row in entries)),
        "aggregate_sha256": hashlib.sha256(canonical).hexdigest(),
        "entries": entries,
    }


def aggregate_evaluation(
    evaluation_root: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    evaluation = protocol["initial_evaluation"]
    metrics_by_pipeline: dict[str, list[dict[str, Any]]] = {
        pipeline: [] for pipeline in PIPELINES
    }
    animals_by_key: dict[tuple[str, int], pd.DataFrame] = {}
    attention_by_key: dict[tuple[str, int], pd.DataFrame] = {}
    for repeat in evaluation["repeats"]:
        for pipeline in PIPELINES:
            frames = []
            detail_frames = []
            for outer_fold in evaluation["outer_folds"]:
                fit_root = (
                    evaluation_root
                    / "fits"
                    / pipeline
                    / f"base_seed_{evaluation['base_seeds'][0]}"
                    / f"repeat_{repeat}"
                    / f"fold_{outer_fold}"
                )
                frames.append(pd.read_csv(fit_root / "outer_test_animal_predictions.csv", dtype={"cat_id": str}))
                if pipeline in SET_PIPELINES:
                    detail_frames.append(
                        pd.read_csv(fit_root / "outer_test_call_attention.csv", dtype={"cat_id": str})
                    )
            animals = pd.concat(frames, ignore_index=True).sort_values("cat_id").reset_index(drop=True)
            if len(animals) != 111 or animals["cat_id"].nunique() != 111:
                raise RuntimeError("Complete OOF set evaluation must contain 111 cats")
            metrics = animal_metrics(animals)
            metrics_by_pipeline[pipeline].append(
                {"base_seed": int(evaluation["base_seeds"][0]), "repeat": int(repeat), **metrics}
            )
            animals_by_key[(pipeline, int(repeat))] = animals
            output = evaluation_root / "oof" / pipeline / f"repeat_{repeat}_animals.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            animals.to_csv(output, index=False)
            if detail_frames:
                attention_by_key[(pipeline, int(repeat))] = pd.concat(
                    detail_frames, ignore_index=True
                ).sort_values("call_index").reset_index(drop=True)

    paired: dict[str, list[dict[str, Any]]] = {}
    paired_changes = []
    contrasts = (
        (PIPELINES[0], PIPELINES[1]),
        (PIPELINES[0], PIPELINES[2]),
        (PIPELINES[1], PIPELINES[2]),
    )
    for left_pipeline, right_pipeline in contrasts:
        name = f"{right_pipeline}_minus_{left_pipeline}"
        rows = []
        for repeat in evaluation["repeats"]:
            left_metrics = metrics_by_pipeline[left_pipeline][int(repeat)]
            right_metrics = metrics_by_pipeline[right_pipeline][int(repeat)]
            left = animals_by_key[(left_pipeline, int(repeat))].sort_values("cat_id")
            right = animals_by_key[(right_pipeline, int(repeat))].sort_values("cat_id")
            merged = left.merge(right, on=["cat_id", "true_label", "call_count"], suffixes=("_left", "_right"))
            rows.append(
                {
                    "repeat": int(repeat),
                    "left_macro_f1": left_metrics["macro_f1"],
                    "right_macro_f1": right_metrics["macro_f1"],
                    "macro_f1_difference": right_metrics["macro_f1"] - left_metrics["macro_f1"],
                    "balanced_accuracy_difference": right_metrics["balanced_accuracy"] - left_metrics["balanced_accuracy"],
                    "qwk_difference": right_metrics["quadratic_weighted_kappa"] - left_metrics["quadratic_weighted_kappa"],
                    "plain_accuracy_difference": right_metrics["plain_accuracy"] - left_metrics["plain_accuracy"],
                    "changed_animals": int((merged["predicted_label_left"] != merged["predicted_label_right"]).sum()),
                    "gained_correct_animals": int(((merged["predicted_label_right"] == merged["true_label"]) & (merged["predicted_label_left"] != merged["true_label"])).sum()),
                    "lost_correct_animals": int(((merged["predicted_label_right"] != merged["true_label"]) & (merged["predicted_label_left"] == merged["true_label"])).sum()),
                }
            )
            changed = merged[merged["predicted_label_left"] != merged["predicted_label_right"]].copy()
            changed.insert(0, "contrast", name)
            changed.insert(1, "repeat", int(repeat))
            paired_changes.append(changed)
        paired[name] = rows
    paired_change_frame = pd.concat(paired_changes, ignore_index=True)
    paired_change_path = evaluation_root / "paired_prediction_changes.csv"
    paired_change_frame.to_csv(paired_change_path, index=False)

    aggregate = {
        pipeline: aggregate_metrics(metrics_by_pipeline[pipeline]) for pipeline in PIPELINES
    }
    paired_summary = {}
    bootstraps = {}
    for left_pipeline, right_pipeline in contrasts:
        name = f"{right_pipeline}_minus_{left_pipeline}"
        rows = paired[name]
        paired_summary[name] = {
            "macro_f1_differences": [row["macro_f1_difference"] for row in rows],
            "mean_macro_f1_difference": float(np.mean([row["macro_f1_difference"] for row in rows])),
            "positive_repeats": int(sum(row["macro_f1_difference"] > 0 for row in rows)),
            "mean_balanced_accuracy_difference": float(np.mean([row["balanced_accuracy_difference"] for row in rows])),
            "mean_qwk_difference": float(np.mean([row["qwk_difference"] for row in rows])),
            "mean_plain_accuracy_difference": float(np.mean([row["plain_accuracy_difference"] for row in rows])),
        }
        bootstraps[name] = paired_cat_bootstrap(
            animals_by_key, left_pipeline, right_pipeline, protocol
        )

    attention = {
        pipeline: {
            "per_repeat": [
                {"repeat": int(repeat), **attention_summary(attention_by_key[(pipeline, int(repeat))])}
                for repeat in evaluation["repeats"]
            ]
        }
        for pipeline in SET_PIPELINES
    }
    for pipeline in SET_PIPELINES:
        rows = attention[pipeline]["per_repeat"]
        attention[pipeline]["mean_normalized_entropy"] = float(
            np.mean([row["mean_normalized_entropy"] for row in rows])
        )
        attention[pipeline]["mean_max_weight"] = float(
            np.mean([row["mean_max_weight"] for row in rows])
        )
        attention[pipeline]["mean_abs_deviation_from_uniform"] = float(
            np.mean([row["mean_abs_deviation_from_uniform"] for row in rows])
        )

    all_animals = {
        pipeline: pd.concat(
            [animals_by_key[(pipeline, int(repeat))] for repeat in evaluation["repeats"]],
            ignore_index=True,
        )
        for pipeline in PIPELINES
    }
    subgroup = {pipeline: bag_size_metrics(frame) for pipeline, frame in all_animals.items()}
    gate = {}
    for candidate in SET_PIPELINES:
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
            if candidate_metrics["mean_per_class"][label]["recall"] > reference_metrics["mean_per_class"][label]["recall"]:
                supporting.append(f"{label}_recall")
        passed = (
            comparison["mean_macro_f1_difference"]
            >= float(protocol["seed_expansion_gate"]["minimum_mean_macro_f1_gain"])
            and comparison["positive_repeats"]
            >= int(protocol["seed_expansion_gate"]["minimum_positive_repeats"])
            and bool(supporting)
        )
        gate[candidate] = {
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
        "bag_size_subgroups": subgroup,
        "attention": attention,
        "seed_expansion_gate": gate,
        "artifacts": {"paired_prediction_changes": repo_relative(paired_change_path)},
    }


def run_evaluation(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
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
            "stage": "idea051_initial_evaluation",
            "outer_test_accessed": True,
            "code_commit": lock["code_commit"],
            "protocol_sha256": lock["protocol_sha256"],
            "runner_sha256": lock["runner_sha256"],
            "execution_lock_sha256": sha256(run_root / "execution_lock.json"),
            "environment_lock_sha256": sha256(run_root / "environment_lock.json"),
            "device": str(device),
        },
    )
    evaluation = protocol["initial_evaluation"]
    completed = []
    set_pair_audits = []
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
                fit_pair = []
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
                        if pipeline in SET_PIPELINES:
                            fit_pair.append(fit)
                        continue
                    print(
                        f"IDEA051 EVAL {pipeline} base_seed={base_seed} "
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
                    animals, detail, detail_kind, outer = fit_outer_and_predict(
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
                    animals.to_csv(animal_path, index=False)
                    if detail_kind == "call_probabilities":
                        detail_path = output_dir / "outer_test_call_predictions.csv"
                    else:
                        detail_path = output_dir / "outer_test_call_attention.csv"
                    detail.to_csv(detail_path, index=False)
                    fit = {
                        "status": "complete",
                        "stage": "idea051_initial_evaluation",
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
                        "detail_prediction_path": repo_relative(detail_path),
                        "detail_kind": detail_kind,
                    }
                    write_json(fit_path, fit)
                    completed.append(fit)
                    if pipeline in SET_PIPELINES:
                        fit_pair.append(fit)
                compared = assert_set_pair_batch_orders(fit_pair[0], fit_pair[1])
                set_pair_audits.append(
                    {
                        "base_seed": int(base_seed),
                        "repeat": int(repeat),
                        "outer_fold": int(outer_fold),
                        **compared,
                    }
                )
    summary = aggregate_evaluation(evaluation_root, protocol)
    summary["set_pair_batch_order_audit"] = {
        "pairs": len(set_pair_audits),
        "all_common_epoch_hashes_match": True,
        "details": set_pair_audits,
    }
    inventory = raw_prediction_inventory(evaluation_root)
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
    store = reference.historical.idea019.load_feature_store()
    device = reference.historical.idea019.resolve_device(args.device)
    print(
        f"IDEA-051 stage={args.stage}; device={device}; "
        f"device_name={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}",
        flush=True,
    )
    if args.stage == "smoke":
        run_smoke(run_root, protocol, roles, store, device, args.resume)
    else:
        run_evaluation(run_root, protocol, roles, store, device, args.resume)


if __name__ == "__main__":
    main()
