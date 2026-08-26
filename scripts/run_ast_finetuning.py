"""Compare frozen, last-two-block, and full AST fine-tuning on MeowAgeNet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from torch.utils.data import DataLoader, Dataset
from transformers import ASTModel


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from animal_fyp.evaluation import animal_level_metrics, categorical_metrics  # noqa: E402


CONFIG_PATH = REPO_ROOT / "configs" / "protocol" / "meowagenet_locked_v1.json"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_nested_roles_v1.csv"
FEATURE_PATH = (
    REPO_ROOT
    / "runs"
    / "ast_locked_v1"
    / "gpu_rerun_2026-08-26"
    / "ast_fbank_128.npz"
)
FROZEN_EMBEDDING_PATH = (
    REPO_ROOT
    / "runs"
    / "ast_locked_v1"
    / "gpu_rerun_2026-08-26"
    / "ast_standard_call_embeddings.npz"
)
HF_CACHE = REPO_ROOT / "data" / "models" / "huggingface"
MODES = ("frozen", "last2", "full")
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--folds", default="0,1,2,3")
    parser.add_argument("--output-subdir", default="ast_finetuning_v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--encoder-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--head-learning-rate", type=float, default=3.109800273709165e-3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
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
        raise RuntimeError("CUDA requested but unavailable in the active PyTorch environment")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def adapt_standard_geometry(model: ASTModel, protocol: dict[str, object]) -> dict[str, object]:
    ast_config = protocol["ast"]
    variant = ast_config["variants"]["ast_standard"]
    max_length = int(ast_config["max_length_frames"])
    frequency_stride = int(variant["frequency_stride"])
    time_stride = int(variant["time_stride"])
    config = model.config
    embeddings = model.embeddings
    patch_size = int(config.patch_size)
    old_frequency = (config.num_mel_bins - patch_size) // config.frequency_stride + 1
    old_time = (config.max_length - patch_size) // config.time_stride + 1
    new_frequency = (config.num_mel_bins - patch_size) // frequency_stride + 1
    new_time = (max_length - patch_size) // time_stride + 1

    projection_weight = embeddings.patch_embeddings.projection.weight.detach().clone()
    projection_bias = embeddings.patch_embeddings.projection.bias.detach().clone()
    special_positions = embeddings.position_embeddings[:, :2]
    patch_positions = embeddings.position_embeddings[:, 2:]
    hidden_size = patch_positions.shape[-1]
    patch_positions = patch_positions.reshape(1, old_frequency, old_time, hidden_size)
    patch_positions = patch_positions.permute(0, 3, 1, 2)
    patch_positions = torch_functional.interpolate(
        patch_positions,
        size=(new_frequency, new_time),
        mode="bilinear",
        align_corners=False,
    )
    patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(
        1, new_frequency * new_time, hidden_size
    )
    embeddings.position_embeddings = nn.Parameter(
        torch.cat((special_positions, patch_positions), dim=1)
    )
    embeddings.patch_embeddings.projection.stride = (frequency_stride, time_stride)
    config.max_length = max_length
    config.frequency_stride = frequency_stride
    config.time_stride = time_stride
    embeddings.config = config

    projection_reused = torch.equal(
        projection_weight, embeddings.patch_embeddings.projection.weight.detach()
    ) and torch.equal(projection_bias, embeddings.patch_embeddings.projection.bias.detach())
    if not projection_reused:
        raise RuntimeError("Patch projection changed during standard-geometry adaptation")
    return {
        "source_grid": [old_frequency, old_time],
        "target_grid": [new_frequency, new_time],
        "patch_tokens": new_frequency * new_time,
        "position_interpolation": "bilinear",
        "patch_projection_weights_reused": True,
    }


@dataclass(frozen=True)
class FeatureStore:
    features: np.ndarray
    segment_call_indices: np.ndarray
    call_segment_indices: tuple[np.ndarray, ...]
    frozen_embeddings: np.ndarray
    call_ids: np.ndarray
    cat_ids: np.ndarray
    labels: np.ndarray


def load_feature_store() -> FeatureStore:
    loaded = np.load(FEATURE_PATH)
    frozen = np.load(FROZEN_EMBEDDING_PATH)
    call_ids = loaded["call_ids"].astype(str)
    if not np.array_equal(call_ids, frozen["call_ids"].astype(str)):
        raise RuntimeError("Fbank and frozen-embedding call order differs")
    segment_call_indices = loaded["segment_call_indices"].astype(np.int64)
    call_segment_indices = tuple(
        np.flatnonzero(segment_call_indices == call_index).astype(np.int64)
        for call_index in range(len(call_ids))
    )
    if any(len(indices) == 0 for indices in call_segment_indices):
        raise RuntimeError("At least one call has no fbank segment")
    return FeatureStore(
        features=loaded["features"].astype(np.float32),
        segment_call_indices=segment_call_indices,
        call_segment_indices=call_segment_indices,
        frozen_embeddings=frozen["embeddings"].astype(np.float32),
        call_ids=call_ids,
        cat_ids=loaded["cat_ids"].astype(str),
        labels=loaded["labels"].astype(np.int64),
    )


class CallDataset(Dataset):
    def __init__(self, store: FeatureStore, call_indices: np.ndarray, mode: str) -> None:
        self.store = store
        self.call_indices = np.asarray(call_indices, dtype=np.int64)
        self.mode = mode

    def __len__(self) -> int:
        return len(self.call_indices)

    def __getitem__(self, item: int) -> tuple[np.ndarray, int, int]:
        call_index = int(self.call_indices[item])
        if self.mode == "frozen":
            instances = self.store.frozen_embeddings[call_index][None, :]
        else:
            instances = self.store.features[self.store.call_segment_indices[call_index]]
        return instances, int(self.store.labels[call_index]), call_index


def collate_calls(
    rows: list[tuple[np.ndarray, int, int]],
) -> dict[str, torch.Tensor]:
    instances = []
    instance_to_call = []
    labels = []
    call_indices = []
    for local_call, (call_instances, label, call_index) in enumerate(rows):
        instances.append(torch.from_numpy(call_instances))
        instance_to_call.extend([local_call] * len(call_instances))
        labels.append(label)
        call_indices.append(call_index)
    return {
        "instances": torch.cat(instances, dim=0),
        "instance_to_call": torch.tensor(instance_to_call, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "call_indices": torch.tensor(call_indices, dtype=torch.long),
    }


class ClassificationHead(nn.Module):
    def __init__(self, mean: np.ndarray, scale: np.ndarray, dropout: float) -> None:
        super().__init__()
        safe_scale = np.where(scale > 1.0e-12, scale, 1.0).astype(np.float32)
        self.register_buffer("feature_mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("feature_scale", torch.from_numpy(safe_scale))
        self.network = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            # Match the Keras BatchNormalization defaults used by the locked MLP.
            nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = (embeddings - self.feature_mean) / self.feature_scale
        return self.network(embeddings)


class FrozenClassifier(nn.Module):
    def __init__(self, head: ClassificationHead) -> None:
        super().__init__()
        self.head = head

    def forward(
        self, instances: torch.Tensor, instance_to_call: torch.Tensor, call_count: int
    ) -> torch.Tensor:
        del instance_to_call, call_count
        return self.head(instances)


class FinetunedASTClassifier(nn.Module):
    def __init__(
        self,
        protocol: dict[str, object],
        mode: str,
        head: ClassificationHead,
    ) -> None:
        super().__init__()
        ast_config = protocol["ast"]
        self.ast = ASTModel.from_pretrained(
            ast_config["checkpoint"],
            revision=ast_config["revision"],
            cache_dir=HF_CACHE,
            use_safetensors=True,
        )
        self.geometry_audit = adapt_standard_geometry(self.ast, protocol)
        self.mode = mode
        for parameter in self.ast.parameters():
            parameter.requires_grad = mode == "full"
        if mode == "last2":
            for layer in self.ast.encoder.layer[-2:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            for parameter in self.ast.layernorm.parameters():
                parameter.requires_grad = True
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
        call_embeddings = call_embeddings / counts[:, None]
        return self.head(call_embeddings)


def trainable_counts(model: nn.Module) -> dict[str, int]:
    return {
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total": sum(parameter.numel() for parameter in model.parameters()),
    }


def build_model(
    mode: str,
    protocol: dict[str, object],
    train_indices: np.ndarray,
    store: FeatureStore,
) -> nn.Module:
    embeddings = store.frozen_embeddings[train_indices]
    head = ClassificationHead(
        mean=embeddings.mean(axis=0),
        scale=embeddings.std(axis=0),
        dropout=float(protocol["classifier_head"]["dropout"]),
    )
    if mode == "frozen":
        return FrozenClassifier(head)
    return FinetunedASTClassifier(protocol, mode, head)


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("A training split is missing an age class")
    weights = len(labels) / (3.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_loader(
    store: FeatureStore,
    indices: np.ndarray,
    mode: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        CallDataset(store, indices, mode),
        batch_size=batch_size,
        shuffle=shuffle,
        # BatchNorm cannot train on a singleton batch. Keep every call unless
        # the shuffled training split would otherwise end with exactly one.
        drop_last=shuffle and len(indices) % batch_size == 1,
        num_workers=0,
        collate_fn=collate_calls,
        generator=generator,
    )


def make_optimizer(
    model: nn.Module,
    mode: str,
    encoder_learning_rate: float,
    head_learning_rate: float,
) -> torch.optim.Optimizer:
    if mode == "frozen":
        return torch.optim.Adamax(model.parameters(), lr=head_learning_rate, eps=1.0e-7)
    encoder_parameters = [
        parameter for parameter in model.ast.parameters() if parameter.requires_grad
    ]
    return torch.optim.Adamax(
        [
            {"params": encoder_parameters, "lr": encoder_learning_rate},
            {"params": model.head.parameters(), "lr": head_learning_rate},
        ],
        eps=1.0e-7,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_weights: torch.Tensor,
    device: torch.device,
    accumulation_steps: int,
    gradient_clip: float,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_calls = 0
    for step, cpu_batch in enumerate(loader):
        batch = move_batch(cpu_batch, device)
        labels = batch["labels"]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(
                batch["instances"], batch["instance_to_call"], len(labels)
            )
            raw_loss = torch_functional.cross_entropy(logits, labels, weight=loss_weights)
            loss = raw_loss / accumulation_steps
        scaler.scale(loss).backward()
        should_step = (step + 1) % accumulation_steps == 0 or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(raw_loss.detach()) * len(labels)
        total_calls += len(labels)
    return total_loss / total_calls


def predict_calls(
    model: nn.Module,
    loader: DataLoader,
    store: FeatureStore,
    device: torch.device,
) -> tuple[float, pd.DataFrame]:
    model.eval()
    rows = []
    total_loss = 0.0
    total_calls = 0
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            labels = batch["labels"]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(
                    batch["instances"], batch["instance_to_call"], len(labels)
                )
                loss = torch_functional.cross_entropy(logits, labels)
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            call_indices = batch["call_indices"].cpu().numpy()
            for local_index, call_index in enumerate(call_indices):
                rows.append(
                    {
                        "call_index": int(call_index),
                        "call_id": store.call_ids[call_index],
                        "cat_id": store.cat_ids[call_index],
                        "true_label": int(store.labels[call_index]),
                        "prob_kitten": float(probabilities[local_index, 0]),
                        "prob_adult": float(probabilities[local_index, 1]),
                        "prob_senior": float(probabilities[local_index, 2]),
                    }
                )
            total_loss += float(loss) * len(labels)
            total_calls += len(labels)
    return total_loss / total_calls, pd.DataFrame(rows).sort_values("call_index")


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
    store: FeatureStore,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[int, dict[str, object], dict[str, object]]:
    set_seed(seed)
    model = build_model(mode, protocol, train_indices, store).to(device)
    counts = trainable_counts(model)
    optimizer = make_optimizer(
        model, mode, args.encoder_learning_rate, args.head_learning_rate
    )
    weights = class_weights(store.labels[train_indices]).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    train_loader = build_loader(
        store, train_indices, mode, args.batch_size, True, seed
    )
    validation_loader = build_loader(
        store, validation_indices, mode, args.batch_size, False, seed
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
            args.accumulation_steps,
            args.gradient_clip,
            scaler,
        )
        validation_loss, validation_frame = predict_calls(
            model, validation_loader, store, device
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
    peak_vram = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    audit = {
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "best_validation_loss": best_loss,
        "best_validation_animal_metrics": best_metrics,
        "history": history,
        "train_seconds": elapsed,
        "peak_vram_bytes": peak_vram,
        "parameters": counts,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, best_metrics, audit


def fit_outer_and_predict(
    mode: str,
    protocol: dict[str, object],
    store: FeatureStore,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    set_seed(seed)
    model = build_model(mode, protocol, train_indices, store).to(device)
    counts = trainable_counts(model)
    optimizer = make_optimizer(
        model, mode, args.encoder_learning_rate, args.head_learning_rate
    )
    weights = class_weights(store.labels[train_indices]).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    train_loader = build_loader(store, train_indices, mode, args.batch_size, True, seed)
    test_loader = build_loader(store, test_indices, mode, args.batch_size, False, seed)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_losses = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
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
            f"{mode} outer retrain epoch {epoch}/{epochs}: train_loss={train_loss:.4f}",
            flush=True,
        )
    test_loss, test_frame = predict_calls(model, test_loader, store, device)
    test_metrics, _ = evaluate_frame(test_frame)
    elapsed = time.perf_counter() - started
    peak_vram = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    audit = {
        "epochs": epochs,
        "training_losses": training_losses,
        "test_loss": test_loss,
        "test_animal_metrics": test_metrics,
        "train_and_predict_seconds": elapsed,
        "peak_vram_bytes": peak_vram,
        "parameters": counts,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_frame, audit


def call_indices_for_roles(
    store: FeatureStore, roles: pd.DataFrame, outer_fold: int
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
    repeats: int = 2000,
    seed: int = 20260826,
) -> dict[str, object]:
    merged = reference.merge(candidate, on=["cat_id", "true_label"], suffixes=("_ref", "_cand"))
    rng = np.random.default_rng(seed)
    labels_all = merged["true_label"].to_numpy()
    class_indices = [np.flatnonzero(labels_all == label) for label in range(3)]
    differences = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        labels = labels_all[sampled]
        reference_probabilities = merged[
            [f"{column}_ref" for column in PROBABILITY_COLUMNS]
        ].to_numpy()[sampled]
        candidate_probabilities = merged[
            [f"{column}_cand" for column in PROBABILITY_COLUMNS]
        ].to_numpy()[sampled]
        differences[repeat] = (
            categorical_metrics(labels, candidate_probabilities)["macro_f1"]
            - categorical_metrics(labels, reference_probabilities)["macro_f1"]
        )
    return {
        "repeats": repeats,
        "seed": seed,
        "ci_lower": float(np.quantile(differences, 0.025)),
        "ci_upper": float(np.quantile(differences, 0.975)),
        "bootstrap_mean": float(differences.mean()),
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
    store = load_feature_store()
    device = resolve_device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Fine-tuning device: {device} ({device_name})", flush=True)

    results: dict[str, object] = {}
    animal_frames: dict[str, pd.DataFrame] = {}
    for mode in modes:
        mode_fold_results = []
        mode_test_frames = []
        for outer_fold in folds:
            print(f"=== {mode} outer fold {outer_fold} ===", flush=True)
            indices = call_indices_for_roles(store, roles, outer_fold)
            seed = 42 + outer_fold
            best_epoch, _, inner_audit = fit_inner(
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
        all_test_calls = pd.concat(mode_test_frames, ignore_index=True)
        overall_metrics, animal_frame = evaluate_frame(all_test_calls)
        all_test_calls.to_csv(output_dir / f"{mode}_call_predictions.csv", index=False)
        animal_frame.to_csv(output_dir / f"{mode}_animal_predictions.csv", index=False)
        results[mode] = {
            "folds": mode_fold_results,
            "overall_animal_metrics": overall_metrics,
        }
        animal_frames[mode] = animal_frame

    contrasts = {}
    for candidate, reference in (("last2", "frozen"), ("full", "frozen"), ("full", "last2")):
        if candidate not in results or reference not in results:
            continue
        observed = (
            results[candidate]["overall_animal_metrics"]["macro_f1"]
            - results[reference]["overall_animal_metrics"]["macro_f1"]
        )
        contrasts[f"{candidate}_minus_{reference}"] = {
            "observed_macro_f1_difference": float(observed),
            "paired_stratified_animal_bootstrap": stratified_paired_bootstrap(
                animal_frames[reference], animal_frames[candidate]
            ),
        }

    summary = {
        "experiment": "ast-standard-finetuning-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_config_sha256": sha256(CONFIG_PATH),
        "roles_sha256": sha256(ROLES_PATH),
        "feature_sha256": sha256(FEATURE_PATH),
        "frozen_embedding_sha256": sha256(FROZEN_EMBEDDING_PATH),
        "execution": {
            "device": str(device),
            "device_name": device_name,
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
        },
        "training_config": {
            "modes": modes,
            "folds": folds,
            "batch_size": args.batch_size,
            "accumulation_steps": args.accumulation_steps,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "encoder_learning_rate": args.encoder_learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "gradient_clip": args.gradient_clip,
            "optimizer": "Adamax",
            "mixed_precision": device.type == "cuda",
            "segment_pooling": "mean AST pooler_output within each call before the MLP",
            "scaler": "fixed per-fold scaler fitted on pretrained frozen call embeddings",
        },
        "results": results,
        "contrasts": contrasts,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
