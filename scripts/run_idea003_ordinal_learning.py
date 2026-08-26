"""Run IDEA-003 nominal, ordinal, and cost-sensitive objectives on VGGish."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from animal_fyp.evaluation import animal_level_metrics, categorical_metrics  # noqa: E402


CONFIG_PATH = REPO_ROOT / "configs" / "protocol" / "meowagenet_locked_v1.json"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_nested_roles_v1.csv"
VGGISH_PATH = (
    REPO_ROOT
    / "data"
    / "meowagenet"
    / "official-3d02295bef15"
    / "embeddings"
    / "vggish_looped_embeddings.csv"
)
MODES = (
    "nominal_ce",
    "quadratic_cost",
    "coral_correct",
    "coral_reversed",
    "coral_shuffled",
)
CORAL_RANK_BY_LABEL = {
    "coral_correct": (0, 1, 2),
    "coral_reversed": (2, 1, 0),
    "coral_shuffled": (0, 2, 1),
}
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")


def parse_args() -> argparse.Namespace:
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    head = protocol["classifier_head"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--folds", default="0,1,2,3")
    parser.add_argument("--output-subdir", default="idea003_ordinal_learning_v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=int(head["batch_size"]))
    parser.add_argument("--max-epochs", type=int, default=int(head["max_epochs"]))
    parser.add_argument(
        "--patience", type=int, default=int(head["early_stopping_patience"])
    )
    parser.add_argument("--learning-rate", type=float, default=float(head["learning_rate"]))
    parser.add_argument("--quadratic-cost-strength", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class VGGishStore:
    features: np.ndarray
    unit_ids: np.ndarray
    cat_ids: np.ndarray
    labels: np.ndarray
    exact_ages: np.ndarray


def load_store(analysis_cat_ids: set[str]) -> VGGishStore:
    frame = pd.read_csv(VGGISH_PATH)
    frame["cat_id"] = frame["cat_id"].astype(str)
    frame = frame[frame["cat_id"].isin(analysis_cat_ids)].reset_index(drop=True)
    feature_columns = [str(index) for index in range(128)]
    exact_ages = frame["target"].to_numpy(dtype=np.float32)
    labels = np.where(exact_ages < 0.5, 0, np.where(exact_ages < 10.0, 1, 2)).astype(
        np.int64
    )
    cat_ids = frame["cat_id"].to_numpy()
    label_counts_per_cat = pd.DataFrame({"cat_id": cat_ids, "label": labels}).groupby(
        "cat_id"
    )["label"].nunique()
    if int(label_counts_per_cat.max()) != 1:
        raise RuntimeError("A cat_id has inconsistent ordinal labels")
    if len(frame) != 936 or len(np.unique(cat_ids)) != 111:
        raise RuntimeError(
            f"Unexpected VGGish analysis scope: {len(frame)} rows, {len(np.unique(cat_ids))} cats"
        )
    return VGGishStore(
        features=frame[feature_columns].to_numpy(dtype=np.float32),
        unit_ids=np.asarray([f"vggish-row-{index:04d}" for index in range(len(frame))]),
        cat_ids=cat_ids,
        labels=labels,
        exact_ages=exact_ages,
    )


class VGGishDataset(Dataset):
    def __init__(self, store: VGGishStore, indices: np.ndarray) -> None:
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[np.ndarray, int, int]:
        unit_index = int(self.indices[item])
        return self.store.features[unit_index], int(self.store.labels[unit_index]), unit_index


def collate_units(rows: list[tuple[np.ndarray, int, int]]) -> dict[str, torch.Tensor]:
    return {
        "features": torch.from_numpy(np.stack([row[0] for row in rows])),
        "labels": torch.tensor([row[1] for row in rows], dtype=torch.long),
        "unit_indices": torch.tensor([row[2] for row in rows], dtype=torch.long),
    }


class FixedScaler(nn.Module):
    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        super().__init__()
        safe_scale = np.where(scale > 1.0e-12, scale, 1.0).astype(np.float32)
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("scale", torch.from_numpy(safe_scale))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.scale


class SharedTrunk(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class NominalHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = nn.Linear(128, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(hidden)


class CoralHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.score = nn.Linear(128, 1, bias=False)
        nn.init.zeros_(self.score.weight)
        self.base_bias = nn.Parameter(torch.tensor(0.0))
        # softplus(log(3)) = log(4) gives initial P(rank > 0/1) = 2/3 and 1/3.
        self.raw_gap = nn.Parameter(torch.tensor(math.log(3.0)))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        score = self.score(hidden)
        gap = torch_functional.softplus(self.raw_gap)
        return torch.cat(
            (score + self.base_bias + gap / 2.0, score + self.base_bias - gap / 2.0),
            dim=1,
        )


class OrdinalObjectiveModel(nn.Module):
    def __init__(
        self,
        mode: str,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        dropout: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.scaler = FixedScaler(feature_mean, feature_scale)
        self.trunk = SharedTrunk(dropout)
        self._initialize_component(self.trunk, seed)
        if mode in ("nominal_ce", "quadratic_cost"):
            self.nominal_head: NominalHead | None = NominalHead()
            self.coral_head: CoralHead | None = None
            self.register_buffer("rank_by_label", torch.arange(3, dtype=torch.long))
        else:
            self.nominal_head = None
            self.coral_head = CoralHead()
            self.register_buffer(
                "rank_by_label",
                torch.tensor(CORAL_RANK_BY_LABEL[mode], dtype=torch.long),
            )

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @classmethod
    def _initialize_component(cls, module: nn.Module, seed: int) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            module.apply(cls._initialize)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(self.scaler(features))
        if self.nominal_head is not None:
            logits = self.nominal_head(hidden)
            return {"probabilities": torch.softmax(logits, dim=1), "nominal_logits": logits}
        if self.coral_head is None:
            raise RuntimeError("CORAL head is unavailable")
        threshold_logits = self.coral_head(hidden)
        exceedance = torch.sigmoid(threshold_logits)
        rank_probabilities = torch.stack(
            (
                1.0 - exceedance[:, 0],
                exceedance[:, 0] - exceedance[:, 1],
                exceedance[:, 1],
            ),
            dim=1,
        )
        probabilities = torch.empty_like(rank_probabilities)
        for original_label in range(3):
            probabilities[:, original_label] = rank_probabilities[
                :, self.rank_by_label[original_label]
            ]
        return {
            "probabilities": probabilities,
            "threshold_logits": threshold_logits,
            "rank_probabilities": rank_probabilities,
        }


def build_model(
    mode: str,
    protocol: dict[str, object],
    store: VGGishStore,
    train_indices: np.ndarray,
    seed: int,
) -> OrdinalObjectiveModel:
    training_features = store.features[train_indices]
    return OrdinalObjectiveModel(
        mode=mode,
        feature_mean=training_features.mean(axis=0),
        feature_scale=training_features.std(axis=0),
        dropout=float(protocol["classifier_head"]["dropout"]),
        seed=seed,
    )


def trainable_counts(model: nn.Module) -> dict[str, int]:
    return {
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total": sum(parameter.numel() for parameter in model.parameters()),
    }


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("A training split is missing an age class")
    return torch.tensor(len(labels) / (3.0 * counts), dtype=torch.float32)


def per_sample_objective(
    mode: str,
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    quadratic_cost_strength: float,
) -> torch.Tensor:
    if mode in ("nominal_ce", "quadratic_cost"):
        losses = torch_functional.cross_entropy(
            output["nominal_logits"], labels, reduction="none"
        )
        if mode == "quadratic_cost":
            classes = torch.arange(3, device=labels.device, dtype=torch.float32)
            normalized_squared_distance = (
                (classes[None, :] - labels[:, None].float()) / 2.0
            ).square()
            expected_cost = (
                output["probabilities"] * normalized_squared_distance
            ).sum(dim=1)
            losses = losses + quadratic_cost_strength * expected_cost
        return losses
    rank_targets = output["threshold_logits"].new_tensor((0, 1))[None, :]
    ranks = output["threshold_logits"].new_tensor(
        CORAL_RANK_BY_LABEL[mode], dtype=torch.long
    )[labels]
    cumulative_targets = (ranks[:, None] > rank_targets).to(torch.float32)
    return torch_functional.binary_cross_entropy_with_logits(
        output["threshold_logits"], cumulative_targets, reduction="none"
    ).mean(dim=1)


def weighted_mean(losses: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    sample_weights = weights[labels]
    return (losses * sample_weights).sum() / sample_weights.sum()


def build_loader(
    store: VGGishStore,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        VGGishDataset(store, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle and len(indices) % batch_size == 1,
        num_workers=0,
        collate_fn=collate_units,
        generator=generator,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(
    model: OrdinalObjectiveModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    weights: torch.Tensor,
    device: torch.device,
    quadratic_cost_strength: float,
    gradient_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_units = 0
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["features"])
        losses = per_sample_objective(
            model.mode, output, batch["labels"], quadratic_cost_strength
        )
        loss = weighted_mean(losses, batch["labels"], weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total_loss += float(loss.detach()) * len(batch["labels"])
        total_units += len(batch["labels"])
    return total_loss / total_units


def predict_units(
    model: OrdinalObjectiveModel,
    loader: DataLoader,
    store: VGGishStore,
    device: torch.device,
    quadratic_cost_strength: float,
) -> tuple[float, pd.DataFrame]:
    model.eval()
    rows = []
    total_loss = 0.0
    total_units = 0
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            output = model(batch["features"])
            losses = per_sample_objective(
                model.mode, output, batch["labels"], quadratic_cost_strength
            )
            probabilities = output["probabilities"].cpu().numpy()
            unit_indices = batch["unit_indices"].cpu().numpy()
            for local_index, unit_index in enumerate(unit_indices):
                unit_index = int(unit_index)
                rows.append(
                    {
                        "unit_index": unit_index,
                        "unit_id": store.unit_ids[unit_index],
                        "cat_id": store.cat_ids[unit_index],
                        "true_label": int(store.labels[unit_index]),
                        "exact_age_years": float(store.exact_ages[unit_index]),
                        "prob_kitten": float(probabilities[local_index, 0]),
                        "prob_adult": float(probabilities[local_index, 1]),
                        "prob_senior": float(probabilities[local_index, 2]),
                    }
                )
            total_loss += float(losses.mean()) * len(batch["labels"])
            total_units += len(batch["labels"])
    return total_loss / total_units, pd.DataFrame(rows).sort_values("unit_index")


def add_ordinal_metrics(
    metrics: dict[str, object], labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, object]:
    predicted = probabilities.argmax(axis=1)
    expected_rank = probabilities @ np.arange(3, dtype=np.float64)
    absolute_errors = np.abs(predicted - labels)
    entropy = -(
        probabilities * np.log(np.clip(probabilities, 1.0e-12, 1.0))
    ).sum(axis=1)
    metrics.update(
        {
            "accuracy": float((predicted == labels).mean()),
            "ordinal_mae": float(absolute_errors.mean()),
            "expected_rank_mae": float(np.abs(expected_rank - labels).mean()),
            "extreme_error_rate": float((absolute_errors == 2).mean()),
            "extreme_error_count": int((absolute_errors == 2).sum()),
            "adult_prediction_rate": float((predicted == 1).mean()),
            "mean_prediction_entropy": float(entropy.mean()),
        }
    )
    return metrics


def evaluate_frame(frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy()
    labels = frame["true_label"].to_numpy()
    metrics, animal_ids, animal_labels, animal_probabilities = animal_level_metrics(
        probabilities, frame["cat_id"].to_numpy(), labels
    )
    add_ordinal_metrics(metrics, animal_labels, animal_probabilities)
    animal_frame = pd.DataFrame(
        {
            "cat_id": animal_ids,
            "true_label": animal_labels,
            "prob_kitten": animal_probabilities[:, 0],
            "prob_adult": animal_probabilities[:, 1],
            "prob_senior": animal_probabilities[:, 2],
            "predicted_label": animal_probabilities.argmax(axis=1),
            "expected_rank": animal_probabilities @ np.arange(3, dtype=np.float64),
        }
    )
    return metrics, animal_frame


def call_indices_for_roles(
    store: VGGishStore, roles: pd.DataFrame, outer_fold: int
) -> dict[str, np.ndarray]:
    fold_roles = roles[roles["outer_fold"] == outer_fold]
    role_by_cat = dict(zip(fold_roles["cat_id"], fold_roles["role"], strict=True))
    unit_roles = np.asarray([role_by_cat[cat_id] for cat_id in store.cat_ids])
    indices = {
        role: np.flatnonzero(unit_roles == role)
        for role in ("train", "validation", "test")
    }
    if set(store.cat_ids[indices["train"]]) & set(store.cat_ids[indices["test"]]):
        raise RuntimeError(f"Cat-ID leakage in outer fold {outer_fold}")
    return indices


def fit_inner(
    mode: str,
    protocol: dict[str, object],
    store: VGGishStore,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[int, dict[str, object]]:
    set_seed(seed)
    model = build_model(mode, protocol, store, train_indices, seed).to(device)
    counts = trainable_counts(model)
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.learning_rate, eps=1.0e-7)
    weights = class_weights(store.labels[train_indices]).to(device)
    train_loader = build_loader(store, train_indices, args.batch_size, True, seed)
    validation_loader = build_loader(
        store, validation_indices, args.batch_size, False, seed
    )
    best_loss = float("inf")
    best_epoch = 1
    best_metrics: dict[str, object] = {}
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            weights,
            device,
            args.quadratic_cost_strength,
            args.gradient_clip,
        )
        validation_loss, validation_frame = predict_units(
            model,
            validation_loader,
            store,
            device,
            args.quadratic_cost_strength,
        )
        validation_metrics, _ = evaluate_frame(validation_frame)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_qwk": validation_metrics["quadratic_weighted_kappa"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_ordinal_mae": validation_metrics["ordinal_mae"],
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
            f"{mode} epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={validation_loss:.4f}, val_QWK={validation_metrics['quadratic_weighted_kappa']:.4f}, "
            f"val_F1={validation_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            break
    audit = {
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "best_validation_loss": best_loss,
        "best_validation_animal_metrics": best_metrics,
        "history": history,
        "train_seconds": time.perf_counter() - started,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "parameters": counts,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, audit


def fit_outer_and_predict(
    mode: str,
    protocol: dict[str, object],
    store: VGGishStore,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    set_seed(seed)
    model = build_model(mode, protocol, store, train_indices, seed).to(device)
    counts = trainable_counts(model)
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.learning_rate, eps=1.0e-7)
    weights = class_weights(store.labels[train_indices]).to(device)
    train_loader = build_loader(store, train_indices, args.batch_size, True, seed)
    test_loader = build_loader(store, test_indices, args.batch_size, False, seed)
    training_losses = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            weights,
            device,
            args.quadratic_cost_strength,
            args.gradient_clip,
        )
        training_losses.append(train_loss)
        print(
            f"{mode} outer retrain epoch {epoch}/{epochs}: train_loss={train_loss:.4f}",
            flush=True,
        )
    test_loss, test_frame = predict_units(
        model, test_loader, store, device, args.quadratic_cost_strength
    )
    test_metrics, _ = evaluate_frame(test_frame)
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
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_frame, audit


def bootstrap_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    categorical = categorical_metrics(labels, probabilities)
    predicted = probabilities.argmax(axis=1)
    absolute_errors = np.abs(predicted - labels)
    return {
        "qwk": float(categorical["quadratic_weighted_kappa"]),
        "macro_f1": float(categorical["macro_f1"]),
        "ordinal_mae": float(absolute_errors.mean()),
        "extreme_error_rate": float((absolute_errors == 2).mean()),
    }


def paired_stratified_bootstrap(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    repeats: int,
    seed: int = 20260826,
) -> dict[str, object]:
    merged = reference.merge(candidate, on=["cat_id", "true_label"], suffixes=("_ref", "_cand"))
    labels_all = merged["true_label"].to_numpy()
    reference_probabilities = merged[
        [f"{column}_ref" for column in PROBABILITY_COLUMNS]
    ].to_numpy()
    candidate_probabilities = merged[
        [f"{column}_cand" for column in PROBABILITY_COLUMNS]
    ].to_numpy()
    observed_reference = bootstrap_metrics(labels_all, reference_probabilities)
    observed_candidate = bootstrap_metrics(labels_all, candidate_probabilities)
    observed = {
        metric: observed_candidate[metric] - observed_reference[metric]
        for metric in observed_reference
    }
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels_all == label) for label in range(3)]
    differences = {metric: np.empty(repeats, dtype=np.float64) for metric in observed}
    for repeat in range(repeats):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        labels = labels_all[sampled]
        reference_metrics = bootstrap_metrics(labels, reference_probabilities[sampled])
        candidate_metrics = bootstrap_metrics(labels, candidate_probabilities[sampled])
        for metric in differences:
            differences[metric][repeat] = candidate_metrics[metric] - reference_metrics[metric]
    return {
        "repeats": repeats,
        "seed": seed,
        "observed_candidate_minus_reference": observed,
        "intervals": {
            metric: {
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "bootstrap_mean": float(values.mean()),
            }
            for metric, values in differences.items()
        },
    }


def main() -> None:
    args = parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown_modes = set(modes) - set(MODES)
    if unknown_modes:
        raise ValueError(f"Unknown modes: {sorted(unknown_modes)}")
    folds = [int(fold.strip()) for fold in args.folds.split(",") if fold.strip()]
    if not folds or any(fold not in range(4) for fold in folds):
        raise ValueError("Folds must be selected from 0,1,2,3")
    if args.batch_size < 2:
        raise ValueError("Batch size must be at least 2 because the trunk uses BatchNorm")
    output_dir = REPO_ROOT / "runs" / args.output_subdir
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    store = load_store(set(roles["cat_id"]))
    device = resolve_device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"IDEA-003 training device: {device} ({device_name})", flush=True)

    results: dict[str, object] = {}
    animal_frames: dict[str, pd.DataFrame] = {}
    for mode in modes:
        mode_fold_results = []
        mode_test_frames = []
        for outer_fold in folds:
            print(f"=== {mode} outer fold {outer_fold} ===", flush=True)
            indices = call_indices_for_roles(store, roles, outer_fold)
            seed = 42 + outer_fold
            best_epoch, inner_audit = fit_inner(
                mode,
                protocol,
                store,
                indices["train"],
                indices["validation"],
                args,
                device,
                seed,
            )
            outer_train_indices = np.concatenate((indices["train"], indices["validation"]))
            test_frame, outer_audit = fit_outer_and_predict(
                mode,
                protocol,
                store,
                outer_train_indices,
                indices["test"],
                best_epoch,
                args,
                device,
                seed,
            )
            test_frame.insert(0, "outer_fold", outer_fold)
            mode_test_frames.append(test_frame)
            mode_fold_results.append(
                {
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "inner_train_units": int(len(indices["train"])),
                    "inner_validation_units": int(len(indices["validation"])),
                    "outer_test_units": int(len(indices["test"])),
                    "inner": inner_audit,
                    "outer": outer_audit,
                }
            )
            print(
                f"{mode} fold {outer_fold}: best_epoch={best_epoch}, "
                f"outer_QWK={outer_audit['test_animal_metrics']['quadratic_weighted_kappa']:.4f}, "
                f"outer_F1={outer_audit['test_animal_metrics']['macro_f1']:.4f}",
                flush=True,
            )
        all_test_units = pd.concat(mode_test_frames, ignore_index=True).sort_values("unit_index")
        overall_metrics, animal_frame = evaluate_frame(all_test_units)
        all_test_units.to_csv(output_dir / f"{mode}_unit_predictions.csv", index=False)
        animal_frame.to_csv(output_dir / f"{mode}_animal_predictions.csv", index=False)
        results[mode] = {
            "folds": mode_fold_results,
            "overall_animal_metrics": overall_metrics,
        }
        animal_frames[mode] = animal_frame

    contrasts = {}
    for candidate, reference in (
        ("coral_correct", "nominal_ce"),
        ("quadratic_cost", "nominal_ce"),
        ("coral_correct", "quadratic_cost"),
        ("coral_correct", "coral_shuffled"),
        ("coral_reversed", "coral_correct"),
    ):
        if candidate not in results or reference not in results:
            continue
        contrasts[f"{candidate}_minus_{reference}"] = paired_stratified_bootstrap(
            animal_frames[reference],
            animal_frames[candidate],
            repeats=args.bootstrap_repeats,
        )

    summary = {
        "experiment": "idea003-vggish-ordinal-learning-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_config_sha256": sha256(CONFIG_PATH),
        "roles_sha256": sha256(ROLES_PATH),
        "vggish_csv_sha256": sha256(VGGISH_PATH),
        "execution": {
            "device": str(device),
            "device_name": device_name,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "data_scope": {
            "prediction_units": int(len(store.features)),
            "cats": int(len(np.unique(store.cat_ids))),
            "label_counts_by_unit": {
                str(label): int(count)
                for label, count in zip(*np.unique(store.labels, return_counts=True), strict=True)
            },
            "age_boundaries_years": [0.5, 10.0],
        },
        "training_config": {
            "modes": modes,
            "folds": folds,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "quadratic_cost_strength": args.quadratic_cost_strength,
            "quadratic_cost": "CE + strength * expected ((predicted_rank-true_rank)/2)^2",
            "gradient_clip": args.gradient_clip,
            "optimizer": "Adamax",
            "class_weight": "balanced by original class",
            "shared_trunk": "128-128 ReLU-BatchNorm-Dropout",
            "initialization_control": "identical trunk; all objectives begin with uniform original-class probabilities",
            "coral_orders": {
                mode: list(order) for mode, order in CORAL_RANK_BY_LABEL.items()
            },
            "early_stopping": "objective-specific inner validation loss",
        },
        "primary_metric": "animal-level quadratic weighted kappa",
        "results": results,
        "contrasts": contrasts,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
