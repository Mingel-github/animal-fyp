"""Run IDEA-013 frozen-AST temporal-pooling comparisons on MeowAgeNet."""

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
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from animal_fyp.evaluation import animal_level_metrics, categorical_metrics  # noqa: E402


CONFIG_PATH = REPO_ROOT / "configs" / "protocol" / "meowagenet_locked_v1.json"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_nested_roles_v1.csv"
TOKEN_PATH = (
    REPO_ROOT
    / "runs"
    / "ast_temporal_tokens_v1"
    / "ast_standard_temporal_tokens.npz"
)
EXTRACTION_SUMMARY_PATH = (
    REPO_ROOT / "runs" / "ast_temporal_tokens_v1" / "extraction_summary.json"
)
MODES = ("mean", "mean_capacity", "gated_attention")
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")


def parse_args() -> argparse.Namespace:
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    head = protocol["classifier_head"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--folds", default="0,1,2,3")
    parser.add_argument("--output-subdir", default="idea013_temporal_pooling_v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=int(head["batch_size"]))
    parser.add_argument("--max-epochs", type=int, default=int(head["max_epochs"]))
    parser.add_argument(
        "--patience", type=int, default=int(head["early_stopping_patience"])
    )
    parser.add_argument("--learning-rate", type=float, default=float(head["learning_rate"]))
    parser.add_argument("--attention-dimensions", type=int, default=64)
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
class TemporalStore:
    tokens: np.ndarray
    token_call_indices: np.ndarray
    call_token_indices: tuple[np.ndarray, ...]
    token_segment_indices: np.ndarray
    token_local_time_indices: np.ndarray
    segment_real_overlap: np.ndarray
    call_ids: np.ndarray
    cat_ids: np.ndarray
    labels: np.ndarray
    durations: np.ndarray
    segment_counts: np.ndarray
    call_token_counts: np.ndarray


def load_store() -> TemporalStore:
    loaded = np.load(TOKEN_PATH)
    tokens = loaded["temporal_tokens"].astype(np.float32)
    token_call_indices = loaded["token_call_indices"].astype(np.int64)
    call_ids = loaded["call_ids"].astype(str)
    call_token_indices = tuple(
        np.flatnonzero(token_call_indices == call_index).astype(np.int64)
        for call_index in range(len(call_ids))
    )
    if any(len(indices) == 0 for indices in call_token_indices):
        raise RuntimeError("At least one call has no temporal token")
    stored_counts = loaded["call_token_counts"].astype(np.int64)
    computed_counts = np.asarray([len(indices) for indices in call_token_indices])
    if not np.array_equal(stored_counts, computed_counts):
        raise RuntimeError("Stored and computed call-token counts differ")
    return TemporalStore(
        tokens=tokens,
        token_call_indices=token_call_indices,
        call_token_indices=call_token_indices,
        token_segment_indices=loaded["token_segment_indices"].astype(np.int64),
        token_local_time_indices=loaded["token_local_time_indices"].astype(np.int64),
        segment_real_overlap=loaded["segment_time_patch_real_overlap"].astype(np.int64),
        call_ids=call_ids,
        cat_ids=loaded["cat_ids"].astype(str),
        labels=loaded["labels"].astype(np.int64),
        durations=loaded["durations"].astype(np.float32),
        segment_counts=loaded["segment_counts"].astype(np.int64),
        call_token_counts=stored_counts,
    )


class TemporalCallDataset(Dataset):
    def __init__(self, store: TemporalStore, call_indices: np.ndarray) -> None:
        self.store = store
        self.call_indices = np.asarray(call_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.call_indices)

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray, int, int]:
        call_index = int(self.call_indices[item])
        token_indices = self.store.call_token_indices[call_index]
        return (
            self.store.tokens[token_indices],
            token_indices,
            int(self.store.labels[call_index]),
            call_index,
        )


def collate_temporal_calls(
    rows: list[tuple[np.ndarray, np.ndarray, int, int]],
) -> dict[str, torch.Tensor]:
    batch_size = len(rows)
    maximum_tokens = max(len(row[0]) for row in rows)
    dimensions = rows[0][0].shape[1]
    tokens = torch.zeros((batch_size, maximum_tokens, dimensions), dtype=torch.float32)
    mask = torch.zeros((batch_size, maximum_tokens), dtype=torch.bool)
    token_indices = torch.full((batch_size, maximum_tokens), -1, dtype=torch.long)
    labels = torch.empty(batch_size, dtype=torch.long)
    call_indices = torch.empty(batch_size, dtype=torch.long)
    for batch_index, (call_tokens, indices, label, call_index) in enumerate(rows):
        count = len(call_tokens)
        tokens[batch_index, :count] = torch.from_numpy(call_tokens)
        mask[batch_index, :count] = True
        token_indices[batch_index, :count] = torch.from_numpy(indices)
        labels[batch_index] = label
        call_indices[batch_index] = call_index
    return {
        "tokens": tokens,
        "mask": mask,
        "token_indices": token_indices,
        "labels": labels,
        "call_indices": call_indices,
    }


class FixedScaler(nn.Module):
    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        super().__init__()
        safe_scale = np.where(scale > 1.0e-12, scale, 1.0).astype(np.float32)
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("scale", torch.from_numpy(safe_scale))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.scale


class ClassificationHead(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(embeddings)


class ResidualCapacityAdapter(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.down = nn.Linear(768, dimensions)
        self.up = nn.Linear(dimensions, 768)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings + self.up(torch.relu(self.down(embeddings)))


class GatedAttention(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.value = nn.Linear(768, dimensions)
        self.gate = nn.Linear(768, dimensions)
        self.score = nn.Linear(dimensions, 1)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(self.value(tokens)) * torch.sigmoid(self.gate(tokens))
        scores = self.score(hidden).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = (tokens * weights[:, :, None]).sum(dim=1)
        return pooled, weights


class TemporalPoolingClassifier(nn.Module):
    def __init__(
        self,
        mode: str,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        dropout: float,
        attention_dimensions: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.scaler = FixedScaler(feature_mean, feature_scale)
        self.capacity_adapter = (
            ResidualCapacityAdapter(attention_dimensions)
            if mode == "mean_capacity"
            else None
        )
        self.attention = (
            GatedAttention(attention_dimensions) if mode == "gated_attention" else None
        )
        self.head = ClassificationHead(dropout)
        self._initialize_component(self.head, seed)
        if self.capacity_adapter is not None:
            self._initialize_component(self.capacity_adapter, seed + 10_000)
            nn.init.zeros_(self.capacity_adapter.up.weight)
            nn.init.zeros_(self.capacity_adapter.up.bias)
        if self.attention is not None:
            self._initialize_component(self.attention, seed + 10_000)
            # Uniform scores make the learned pooling start exactly at mean pooling.
            nn.init.zeros_(self.attention.score.weight)
            nn.init.zeros_(self.attention.score.bias)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @classmethod
    def _initialize_component(cls, module: nn.Module, seed: int) -> None:
        # Component-specific RNG streams keep the shared head identical across modes.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            module.apply(cls._initialize)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scaled_tokens = self.scaler(tokens)
        if self.attention is not None:
            pooled, weights = self.attention(scaled_tokens, mask)
        else:
            weights = mask.to(scaled_tokens.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True)
            pooled = (scaled_tokens * weights[:, :, None]).sum(dim=1)
            if self.capacity_adapter is not None:
                pooled = self.capacity_adapter(pooled)
        return self.head(pooled), weights


def mean_call_embeddings(store: TemporalStore, indices: np.ndarray) -> np.ndarray:
    return np.stack(
        [store.tokens[store.call_token_indices[int(index)]].mean(axis=0) for index in indices]
    ).astype(np.float32)


def build_model(
    mode: str,
    protocol: dict[str, object],
    store: TemporalStore,
    train_indices: np.ndarray,
    attention_dimensions: int,
    seed: int,
) -> TemporalPoolingClassifier:
    training_means = mean_call_embeddings(store, train_indices)
    return TemporalPoolingClassifier(
        mode=mode,
        feature_mean=training_means.mean(axis=0),
        feature_scale=training_means.std(axis=0),
        dropout=float(protocol["classifier_head"]["dropout"]),
        attention_dimensions=attention_dimensions,
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


def build_loader(
    store: TemporalStore,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TemporalCallDataset(store, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle and len(indices) % batch_size == 1,
        num_workers=0,
        collate_fn=collate_temporal_calls,
        generator=generator,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(
    model: TemporalPoolingClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    weights: torch.Tensor,
    device: torch.device,
    gradient_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_calls = 0
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch["tokens"], batch["mask"])
        loss = torch_functional.cross_entropy(logits, batch["labels"], weight=weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total_loss += float(loss.detach()) * len(batch["labels"])
        total_calls += len(batch["labels"])
    return total_loss / total_calls


def predict_calls(
    model: TemporalPoolingClassifier,
    loader: DataLoader,
    store: TemporalStore,
    device: torch.device,
    include_attention: bool,
) -> tuple[float, pd.DataFrame, pd.DataFrame | None]:
    model.eval()
    prediction_rows: list[dict[str, object]] = []
    attention_rows: list[dict[str, object]] = []
    total_loss = 0.0
    total_calls = 0
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            logits, attention = model(batch["tokens"], batch["mask"])
            loss = torch_functional.cross_entropy(logits, batch["labels"])
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            call_indices = batch["call_indices"].cpu().numpy()
            token_indices = batch["token_indices"].cpu().numpy()
            masks = batch["mask"].cpu().numpy()
            attention_array = attention.cpu().numpy()
            for local_index, call_index in enumerate(call_indices):
                call_index = int(call_index)
                prediction_rows.append(
                    {
                        "call_index": call_index,
                        "call_id": store.call_ids[call_index],
                        "cat_id": store.cat_ids[call_index],
                        "true_label": int(store.labels[call_index]),
                        "duration_seconds": float(store.durations[call_index]),
                        "segment_count": int(store.segment_counts[call_index]),
                        "temporal_token_count": int(store.call_token_counts[call_index]),
                        "prob_kitten": float(probabilities[local_index, 0]),
                        "prob_adult": float(probabilities[local_index, 1]),
                        "prob_senior": float(probabilities[local_index, 2]),
                    }
                )
                if include_attention:
                    valid_count = int(masks[local_index].sum())
                    valid_token_indices = token_indices[local_index, :valid_count]
                    valid_weights = attention_array[local_index, :valid_count]
                    for rank, (token_index, attention_weight) in enumerate(
                        zip(valid_token_indices, valid_weights, strict=True)
                    ):
                        token_index = int(token_index)
                        segment_index = int(store.token_segment_indices[token_index])
                        local_time = int(store.token_local_time_indices[token_index])
                        real_overlap = int(store.segment_real_overlap[segment_index, local_time])
                        attention_rows.append(
                            {
                                "token_index": token_index,
                                "call_index": call_index,
                                "call_id": store.call_ids[call_index],
                                "cat_id": store.cat_ids[call_index],
                                "true_label": int(store.labels[call_index]),
                                "duration_seconds": float(store.durations[call_index]),
                                "token_count": valid_count,
                                "sequence_rank": rank,
                                "relative_sequence_position": (
                                    rank / (valid_count - 1) if valid_count > 1 else 0.5
                                ),
                                "segment_index": segment_index,
                                "local_time_index": local_time,
                                "real_patch_overlap_frames": real_overlap,
                                "attention_weight": float(attention_weight),
                                "uniform_weight": 1.0 / valid_count,
                                "attention_lift": float(attention_weight * valid_count),
                            }
                        )
            total_loss += float(loss) * len(batch["labels"])
            total_calls += len(batch["labels"])
    predictions = pd.DataFrame(prediction_rows).sort_values("call_index")
    attention_frame = pd.DataFrame(attention_rows) if include_attention else None
    if attention_frame is not None:
        attention_frame = attention_frame.sort_values("token_index")
    return total_loss / total_calls, predictions, attention_frame


def evaluate_frame(frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy()
    labels = frame["true_label"].to_numpy()
    metrics, animal_ids, animal_labels, animal_probabilities = animal_level_metrics(
        probabilities, frame["cat_id"].to_numpy(), labels
    )
    animal_frame = pd.DataFrame(
        {
            "cat_id": animal_ids,
            "true_label": animal_labels,
            "prob_kitten": animal_probabilities[:, 0],
            "prob_adult": animal_probabilities[:, 1],
            "prob_senior": animal_probabilities[:, 2],
            "predicted_label": animal_probabilities.argmax(axis=1),
        }
    )
    return metrics, animal_frame


def fit_inner(
    mode: str,
    protocol: dict[str, object],
    store: TemporalStore,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[int, dict[str, object]]:
    set_seed(seed)
    model = build_model(
        mode, protocol, store, train_indices, args.attention_dimensions, seed
    ).to(device)
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
            model, train_loader, optimizer, weights, device, args.gradient_clip
        )
        validation_loss, validation_frame, _ = predict_calls(
            model, validation_loader, store, device, False
        )
        validation_metrics, _ = evaluate_frame(validation_frame)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_macro_f1": validation_metrics["macro_f1"],
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
            f"val_loss={validation_loss:.4f}, val_F1={validation_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            break
    elapsed = time.perf_counter() - started
    audit = {
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "best_validation_loss": best_loss,
        "best_validation_animal_metrics": best_metrics,
        "history": history,
        "train_seconds": elapsed,
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
    store: TemporalStore,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, object]]:
    set_seed(seed)
    model = build_model(
        mode, protocol, store, train_indices, args.attention_dimensions, seed
    ).to(device)
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
            model, train_loader, optimizer, weights, device, args.gradient_clip
        )
        training_losses.append(train_loss)
        print(
            f"{mode} outer retrain epoch {epoch}/{epochs}: train_loss={train_loss:.4f}",
            flush=True,
        )
    test_loss, predictions, attention = predict_calls(
        model, test_loader, store, device, mode == "gated_attention"
    )
    test_metrics, _ = evaluate_frame(predictions)
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
    return predictions, attention, audit


def call_indices_for_roles(
    store: TemporalStore, roles: pd.DataFrame, outer_fold: int
) -> dict[str, np.ndarray]:
    fold_roles = roles[roles["outer_fold"] == outer_fold]
    role_by_cat = dict(zip(fold_roles["cat_id"], fold_roles["role"], strict=True))
    call_roles = np.asarray([role_by_cat[cat_id] for cat_id in store.cat_ids])
    indices = {
        role: np.flatnonzero(call_roles == role)
        for role in ("train", "validation", "test")
    }
    if set(store.cat_ids[indices["train"]]) & set(store.cat_ids[indices["test"]]):
        raise RuntimeError(f"Cat-ID leakage in outer fold {outer_fold}")
    return indices


def stratified_paired_bootstrap(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    repeats: int,
    seed: int = 20260826,
) -> dict[str, object]:
    merged = reference.merge(candidate, on=["cat_id", "true_label"], suffixes=("_ref", "_cand"))
    rng = np.random.default_rng(seed)
    labels_all = merged["true_label"].to_numpy()
    class_indices = [np.flatnonzero(labels_all == label) for label in range(3)]
    reference_probabilities_all = merged[
        [f"{column}_ref" for column in PROBABILITY_COLUMNS]
    ].to_numpy()
    candidate_probabilities_all = merged[
        [f"{column}_cand" for column in PROBABILITY_COLUMNS]
    ].to_numpy()
    differences = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        labels = labels_all[sampled]
        differences[repeat] = (
            categorical_metrics(labels, candidate_probabilities_all[sampled])["macro_f1"]
            - categorical_metrics(labels, reference_probabilities_all[sampled])["macro_f1"]
        )
    return {
        "repeats": repeats,
        "seed": seed,
        "ci_lower": float(np.quantile(differences, 0.025)),
        "ci_upper": float(np.quantile(differences, 0.975)),
        "bootstrap_mean": float(differences.mean()),
    }


def length_features(store: TemporalStore) -> np.ndarray:
    return np.column_stack(
        (
            np.log1p(store.durations),
            np.log1p(store.segment_counts),
            np.log1p(store.call_token_counts),
        )
    ).astype(np.float64)


def run_length_only_probe(
    store: TemporalStore, roles: pd.DataFrame, folds: list[int]
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    features = length_features(store)
    fold_results = []
    test_frames = []
    for outer_fold in folds:
        indices = call_indices_for_roles(store, roles, outer_fold)
        train_indices = np.concatenate((indices["train"], indices["validation"]))
        test_indices = indices["test"]
        scaler = StandardScaler().fit(features[train_indices])
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42 + outer_fold,
        )
        classifier.fit(scaler.transform(features[train_indices]), store.labels[train_indices])
        probabilities = classifier.predict_proba(scaler.transform(features[test_indices]))
        frame = pd.DataFrame(
            {
                "outer_fold": outer_fold,
                "call_index": test_indices,
                "call_id": store.call_ids[test_indices],
                "cat_id": store.cat_ids[test_indices],
                "true_label": store.labels[test_indices],
                "duration_seconds": store.durations[test_indices],
                "segment_count": store.segment_counts[test_indices],
                "temporal_token_count": store.call_token_counts[test_indices],
                "prob_kitten": probabilities[:, 0],
                "prob_adult": probabilities[:, 1],
                "prob_senior": probabilities[:, 2],
            }
        )
        metrics, _ = evaluate_frame(frame)
        fold_results.append({"outer_fold": outer_fold, "test_animal_metrics": metrics})
        test_frames.append(frame)
    all_calls = pd.concat(test_frames, ignore_index=True).sort_values("call_index")
    overall_metrics, animal_frame = evaluate_frame(all_calls)
    return (
        {
            "features": ["log1p(duration)", "log1p(segment_count)", "log1p(token_count)"],
            "model": "balanced multinomial logistic regression",
            "folds": fold_results,
            "overall_animal_metrics": overall_metrics,
        },
        all_calls,
        animal_frame,
    )


def attention_diagnostics(frame: pd.DataFrame) -> dict[str, object]:
    call_rows = []
    for (call_index, duration), group in frame.groupby(["call_index", "duration_seconds"]):
        weights = group["attention_weight"].to_numpy(dtype=np.float64)
        count = len(weights)
        entropy = float(-(weights * np.log(np.clip(weights, 1.0e-12, None))).sum())
        normalized_entropy = entropy / math.log(count) if count > 1 else 1.0
        maximum_rank = int(group.iloc[int(weights.argmax())]["sequence_rank"])
        call_rows.append(
            {
                "call_index": int(call_index),
                "duration_seconds": float(duration),
                "token_count": count,
                "normalized_entropy": normalized_entropy,
                "concentration": 1.0 - normalized_entropy,
                "maximum_attention": float(weights.max()),
                "maximum_at_sequence_boundary": (
                    maximum_rank in (0, count - 1) if count > 1 else False
                ),
            }
        )
    calls = pd.DataFrame(call_rows)
    eligible = calls[calls["token_count"] > 1]
    duration_correlation = spearmanr(
        calls["duration_seconds"], calls["concentration"], nan_policy="omit"
    )
    overlap_correlation = spearmanr(
        frame["real_patch_overlap_frames"], frame["attention_lift"], nan_policy="omit"
    )
    return {
        "calls": int(len(calls)),
        "calls_with_multiple_tokens": int(len(eligible)),
        "mean_normalized_attention_entropy": float(calls["normalized_entropy"].mean()),
        "median_normalized_attention_entropy": float(calls["normalized_entropy"].median()),
        "mean_maximum_attention": float(calls["maximum_attention"].mean()),
        "maximum_attention_at_sequence_boundary_fraction": float(
            eligible["maximum_at_sequence_boundary"].mean()
        ),
        "spearman_duration_vs_attention_concentration": {
            "correlation": float(duration_correlation.statistic),
            "pvalue": float(duration_correlation.pvalue),
        },
        "spearman_real_overlap_vs_attention_lift": {
            "correlation": float(overlap_correlation.statistic),
            "pvalue": float(overlap_correlation.pvalue),
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
        raise ValueError("Batch size must be at least 2 because the head uses BatchNorm")
    output_dir = REPO_ROOT / "runs" / args.output_subdir
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    store = load_store()
    device = resolve_device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"IDEA-013 training device: {device} ({device_name})", flush=True)

    results: dict[str, object] = {}
    animal_frames: dict[str, pd.DataFrame] = {}
    attention_frames = []
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
            test_frame, attention_frame, outer_audit = fit_outer_and_predict(
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
            if attention_frame is not None:
                attention_frame.insert(0, "outer_fold", outer_fold)
                attention_frames.append(attention_frame)
            mode_fold_results.append(
                {
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "inner_train_calls": int(len(indices["train"])),
                    "inner_validation_calls": int(len(indices["validation"])),
                    "outer_test_calls": int(len(indices["test"])),
                    "inner": inner_audit,
                    "outer": outer_audit,
                }
            )
            print(
                f"{mode} fold {outer_fold}: best_epoch={best_epoch}, "
                f"outer_F1={outer_audit['test_animal_metrics']['macro_f1']:.4f}",
                flush=True,
            )
        all_test_calls = pd.concat(mode_test_frames, ignore_index=True).sort_values("call_index")
        overall_metrics, animal_frame = evaluate_frame(all_test_calls)
        all_test_calls.to_csv(output_dir / f"{mode}_call_predictions.csv", index=False)
        animal_frame.to_csv(output_dir / f"{mode}_animal_predictions.csv", index=False)
        results[mode] = {
            "folds": mode_fold_results,
            "overall_animal_metrics": overall_metrics,
        }
        animal_frames[mode] = animal_frame

    length_result, length_calls, length_animals = run_length_only_probe(store, roles, folds)
    length_calls.to_csv(output_dir / "length_only_call_predictions.csv", index=False)
    length_animals.to_csv(output_dir / "length_only_animal_predictions.csv", index=False)
    results["length_only"] = length_result
    animal_frames["length_only"] = length_animals

    attention_summary = None
    if attention_frames:
        all_attention = pd.concat(attention_frames, ignore_index=True).sort_values("token_index")
        all_attention.to_csv(output_dir / "gated_attention_token_weights.csv", index=False)
        attention_summary = attention_diagnostics(all_attention)

    contrasts = {}
    for candidate, reference in (
        ("mean_capacity", "mean"),
        ("gated_attention", "mean"),
        ("gated_attention", "mean_capacity"),
    ):
        if candidate not in results or reference not in results:
            continue
        observed = (
            results[candidate]["overall_animal_metrics"]["macro_f1"]
            - results[reference]["overall_animal_metrics"]["macro_f1"]
        )
        contrasts[f"{candidate}_minus_{reference}"] = {
            "observed_macro_f1_difference": float(observed),
            "paired_stratified_animal_bootstrap": stratified_paired_bootstrap(
                animal_frames[reference],
                animal_frames[candidate],
                repeats=args.bootstrap_repeats,
            ),
        }

    summary = {
        "experiment": "idea013-frozen-standard-ast-temporal-pooling-v2-headmatched",
        "protocol_id": protocol["protocol_id"],
        "protocol_config_sha256": sha256(CONFIG_PATH),
        "roles_sha256": sha256(ROLES_PATH),
        "temporal_token_sha256": sha256(TOKEN_PATH),
        "temporal_extraction_summary_sha256": sha256(EXTRACTION_SUMMARY_PATH),
        "execution": {
            "device": str(device),
            "device_name": device_name,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "training_config": {
            "modes": modes,
            "folds": folds,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "attention_dimensions": args.attention_dimensions,
            "gradient_clip": args.gradient_clip,
            "optimizer": "Adamax",
            "encoder": "frozen standard AST",
            "frequency_pooling": "mean over 12 frequency positions",
            "temporal_mask": "retain patches with >=50% real-frame overlap; minimum one per segment",
            "scaler": "fixed per-fold scaler fitted on mean-pooled outer-training call tokens",
            "mean_capacity": "64-dimensional residual bottleneck after mean pooling",
            "gated_attention": "64-dimensional tanh-sigmoid gated attention",
            "initialization_control": "identical head; capacity adapter starts as identity; gated attention starts as uniform mean",
        },
        "results": results,
        "contrasts": contrasts,
        "attention_diagnostics": attention_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
