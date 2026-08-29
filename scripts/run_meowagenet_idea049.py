"""Run IDEA-049 frozen-backbone screening for MeowAgeNet.

The first supported candidate is the Hugging Face conversion of
SSAST-Base-Patch-400. The formal-v2.1 runner remains unchanged; this script
reuses only its audited split bank and animal-level evaluation definition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
import torchaudio
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, TensorDataset
from transformers import ASTModel, AutoFeatureExtractor


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from animal_fyp.evaluation import animal_level_metrics, categorical_metrics  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT / "configs" / "protocol" / "meowagenet_idea049_backbone_screening_v1.json"
)
RECIPE_PATH = (
    REPO_ROOT / "configs" / "experiment" / "idea049" / "ssast_base_patch400_frozen_v1.json"
)
DATA_MANIFEST_PATH = REPO_ROOT / "metadata" / "datasets" / "meowagenet" / "data_manifest.csv"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
FORMAL_SUMMARY_PATH = REPO_ROOT / "runs" / "meowagenet_formal_v2_1_core" / "formal_summary.json"
FORMAL_AST_OOF_ROOT = (
    REPO_ROOT / "runs" / "meowagenet_formal_v2_1_core" / "oof" / "ast_head_only"
)
HF_CACHE = REPO_ROOT / "data" / "models" / "huggingface"
RUNS_ROOT = REPO_ROOT / "runs"
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
LABEL_TO_INDEX = {"kitten": 0, "adult": 1, "senior": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "smoke", "initial"), required=True)
    parser.add_argument("--output-subdir", default="idea049_ssast_base_patch400_v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--extraction-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
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
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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


def full_model_seed(base_seed: int, repeat: int, outer_fold: int) -> int:
    return base_seed + 10_000 * repeat + 100 * outer_fold


def verify_inputs(protocol: dict[str, Any], recipe: dict[str, Any]) -> None:
    expected = {
        DATA_MANIFEST_PATH: protocol["dataset"]["manifest_sha256"],
        ROLES_PATH: protocol["splits"]["roles_sha256"],
    }
    for path, digest in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != digest:
            raise RuntimeError(f"Input checksum mismatch: {path}")
    if recipe["protocol_id"] != protocol["protocol_id"]:
        raise RuntimeError("Recipe and protocol IDs differ")
    if recipe["pipeline_id"] != "ssast_hf_base_patch400_frozen_mlp":
        raise RuntimeError("Unsupported IDEA-049 pipeline")


def load_analysis_manifest() -> pd.DataFrame:
    frame = pd.read_csv(
        DATA_MANIFEST_PATH,
        dtype={"analysis_cat_id": str, "published_cat_id": str},
    )
    include = frame["analysis_include"].astype(str).str.lower().eq("true")
    frame = frame.loc[include].copy().reset_index(drop=True)
    frame["label_index"] = frame["age_group_filename"].map(LABEL_TO_INDEX)
    if len(frame) != 792 or frame["analysis_cat_id"].nunique() != 111:
        raise RuntimeError("IDEA-049 requires the audited 792-call, 111-cat analysis view")
    if frame["label_index"].isna().any():
        raise RuntimeError("Unknown age-group label in the analysis manifest")
    local_paths = [REPO_ROOT / value for value in frame["local_relpath"]]
    missing = [path for path in local_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} analysis audio files; first: {missing[0]}")
    frame["absolute_path"] = [str(path) for path in local_paths]
    return frame


def checkpoint_path(recipe: dict[str, Any]) -> Path:
    model = recipe["model"]
    path = Path(
        hf_hub_download(
            model["runtime_model_id"],
            model["runtime_checkpoint_file"],
            revision=model["runtime_revision"],
            cache_dir=HF_CACHE,
        )
    )
    if path.stat().st_size != int(model["runtime_checkpoint_bytes"]):
        raise RuntimeError("SSAST checkpoint size mismatch")
    if sha256(path) != model["runtime_checkpoint_sha256"]:
        raise RuntimeError("SSAST checkpoint SHA-256 mismatch")
    return path


def feature_paths(run_root: Path) -> tuple[Path, Path]:
    feature_root = run_root / "features"
    return feature_root / "ssast_call_embeddings.npz", feature_root / "feature_manifest.json"


def prepare_embeddings(
    run_root: Path,
    protocol: dict[str, Any],
    recipe: dict[str, Any],
    device: torch.device,
    batch_size: int,
    resume: bool,
) -> Path:
    feature_path, feature_manifest_path = feature_paths(run_root)
    if feature_path.is_file():
        if not resume:
            raise FileExistsError(f"Embedding cache exists; pass --resume to verify it: {feature_path}")
        loaded = np.load(feature_path)
        if loaded["embeddings"].shape != (792, 768):
            raise RuntimeError("Existing SSAST embedding cache has the wrong shape")
        print(f"verified existing embedding cache: {feature_path}", flush=True)
        return feature_path

    checkpoint = checkpoint_path(recipe)
    rows = load_analysis_manifest()
    model_config = recipe["model"]
    model_id = model_config["runtime_model_id"]
    revision = model_config["runtime_revision"]
    extractor = AutoFeatureExtractor.from_pretrained(
        model_id, revision=revision, cache_dir=HF_CACHE
    )
    model = ASTModel.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=HF_CACHE,
        use_safetensors=True,
    ).to(device)
    model.eval()
    if int(model.config.hidden_size) != 768:
        raise RuntimeError(f"Unexpected SSAST hidden size: {model.config.hidden_size}")

    all_embeddings: list[np.ndarray] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, len(rows), batch_size):
        batch_rows = rows.iloc[start : start + batch_size]
        waveforms = []
        for path_value in batch_rows["absolute_path"]:
            waveform, sample_rate = torchaudio.load(path_value)
            waveform = waveform.mean(dim=0)
            if sample_rate != int(extractor.sampling_rate):
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, int(extractor.sampling_rate)
                )
            waveforms.append(waveform.numpy())
        features = extractor(
            waveforms,
            sampling_rate=int(extractor.sampling_rate),
            return_tensors="pt",
        )["input_values"].to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            embeddings = model(input_values=features).pooler_output
        all_embeddings.append(embeddings.float().cpu().numpy())
        print(f"SSAST embeddings {min(start + len(batch_rows), len(rows))}/792", flush=True)

    embedding_array = np.concatenate(all_embeddings).astype(np.float32)
    if embedding_array.shape != (792, 768) or not np.isfinite(embedding_array).all():
        raise RuntimeError("Invalid SSAST embedding output")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_path,
        embeddings=embedding_array,
        call_ids=rows["filename"].to_numpy(dtype=str),
        cat_ids=rows["analysis_cat_id"].to_numpy(dtype=str),
        labels=rows["label_index"].to_numpy(dtype=np.int64),
        durations=rows["duration_seconds"].to_numpy(dtype=np.float32),
        source_paths=rows["local_relpath"].to_numpy(dtype=str),
        model_revision=np.asarray([revision]),
    )
    elapsed = time.perf_counter() - started
    feature_manifest = {
        "status": "complete",
        "pipeline": recipe["pipeline_id"],
        "feature_path": repo_relative(feature_path),
        "feature_sha256": sha256(feature_path),
        "feature_bytes": feature_path.stat().st_size,
        "shape": list(embedding_array.shape),
        "calls": 792,
        "cats": 111,
        "finite": True,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "model_revision": revision,
        "pooling": model_config["pooling"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "extraction_batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "recipe_sha256": sha256(RECIPE_PATH),
    }
    write_json(feature_manifest_path, feature_manifest)
    print(json.dumps(feature_manifest, indent=2), flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return feature_path


@dataclass(frozen=True)
class FeatureStore:
    embeddings: np.ndarray
    call_ids: np.ndarray
    cat_ids: np.ndarray
    labels: np.ndarray


def load_feature_store(run_root: Path, recipe: dict[str, Any]) -> FeatureStore:
    feature_path, feature_manifest_path = feature_paths(run_root)
    if not feature_path.is_file() or not feature_manifest_path.is_file():
        raise FileNotFoundError("Run --stage prepare before smoke or initial screening")
    feature_manifest = read_json(feature_manifest_path)
    if sha256(feature_path) != feature_manifest["feature_sha256"]:
        raise RuntimeError("SSAST feature-cache checksum mismatch")
    loaded = np.load(feature_path)
    store = FeatureStore(
        embeddings=loaded["embeddings"].astype(np.float32),
        call_ids=loaded["call_ids"].astype(str),
        cat_ids=loaded["cat_ids"].astype(str),
        labels=loaded["labels"].astype(np.int64),
    )
    if store.embeddings.shape != (792, int(recipe["model"]["hidden_size"])):
        raise RuntimeError(f"Unexpected feature shape: {store.embeddings.shape}")
    if len(np.unique(store.cat_ids)) != 111 or not np.isfinite(store.embeddings).all():
        raise RuntimeError("Invalid SSAST feature store")
    return store


class ClassificationHead(nn.Module):
    def __init__(self, mean: np.ndarray, scale: np.ndarray, dropout: float) -> None:
        super().__init__()
        dimension = int(len(mean))
        safe_scale = np.where(scale > 1.0e-12, scale, 1.0).astype(np.float32)
        self.register_buffer("feature_mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("feature_scale", torch.from_numpy(safe_scale))
        self.network = nn.Sequential(
            nn.Linear(dimension, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        normalized = (embeddings - self.feature_mean) / self.feature_scale
        return self.network(normalized)


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("A training split is missing an age class")
    return torch.tensor(len(labels) / (3.0 * counts), dtype=torch.float32)


def make_loader(
    store: FeatureStore,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    indices = np.asarray(indices, dtype=np.int64)
    dataset = TensorDataset(
        torch.from_numpy(store.embeddings[indices]),
        torch.from_numpy(store.labels[indices]),
        torch.from_numpy(indices),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle and len(indices) % batch_size == 1,
        num_workers=0,
        generator=generator,
    )


def build_head(
    store: FeatureStore, train_indices: np.ndarray, protocol: dict[str, Any]
) -> ClassificationHead:
    train_embeddings = store.embeddings[train_indices]
    return ClassificationHead(
        train_embeddings.mean(axis=0),
        train_embeddings.std(axis=0),
        float(protocol["shared_head"]["dropout"]),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    weights: torch.Tensor,
    device: torch.device,
    accumulation_steps: int,
    gradient_clip: float,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_rows = 0
    for step, (features, labels, _) in enumerate(loader):
        features = features.to(device)
        labels = labels.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(features)
            raw_loss = torch_functional.cross_entropy(logits, labels, weight=weights)
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
        total_rows += len(labels)
    return total_loss / total_rows


def predict(
    model: nn.Module,
    loader: DataLoader,
    store: FeatureStore,
    device: torch.device,
) -> tuple[float, pd.DataFrame]:
    model.eval()
    rows = []
    total_loss = 0.0
    total_rows = 0
    with torch.inference_mode():
        for features, labels, indices in loader:
            features = features.to(device)
            labels = labels.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(features)
                loss = torch_functional.cross_entropy(logits, labels)
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            for local_index, call_index in enumerate(indices.numpy()):
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
            total_rows += len(labels)
    return total_loss / total_rows, pd.DataFrame(rows).sort_values("call_index")


def evaluate_frame(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics, animal_ids, labels, probabilities = animal_level_metrics(
        frame[list(PROBABILITY_COLUMNS)].to_numpy(),
        frame["cat_id"].to_numpy(dtype=str),
        frame["true_label"].to_numpy(dtype=np.int64),
    )
    animals = pd.DataFrame(
        {
            "cat_id": animal_ids,
            "true_label": labels,
            "prob_kitten": probabilities[:, 0],
            "prob_adult": probabilities[:, 1],
            "prob_senior": probabilities[:, 2],
            "predicted_label": probabilities.argmax(axis=1),
        }
    )
    return metrics, animals


def indices_for_roles(
    store: FeatureStore, roles: pd.DataFrame, repeat: int, outer_fold: int
) -> dict[str, np.ndarray]:
    selected = roles[(roles["repeat"] == repeat) & (roles["outer_fold"] == outer_fold)]
    role_by_cat = dict(zip(selected["cat_id"], selected["role"], strict=True))
    if set(role_by_cat) != set(store.cat_ids):
        raise RuntimeError("Split cats differ from the SSAST feature store")
    result = {
        role: np.flatnonzero(np.asarray([role_by_cat[cat] == role for cat in store.cat_ids]))
        for role in ("train", "validation", "test")
    }
    if any(len(indices) == 0 for indices in result.values()):
        raise RuntimeError("A split role contains no calls")
    return result


def fit_inner(
    store: FeatureStore,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    protocol: dict[str, Any],
    device: torch.device,
    seed: int,
    maximum_epochs: int,
) -> tuple[int, dict[str, Any]]:
    set_seed(seed)
    head_config = protocol["shared_head"]
    model = build_head(store, train_indices, protocol).to(device)
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=float(head_config["learning_rate"]), eps=1.0e-7
    )
    weights = class_weights(store.labels[train_indices]).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    train_loader = make_loader(
        store, train_indices, int(head_config["batch_size"]), True, seed
    )
    validation_loader = make_loader(
        store, validation_indices, int(head_config["batch_size"]), False, seed
    )
    best_loss = float("inf")
    best_epoch = 1
    best_metrics: dict[str, Any] = {}
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, maximum_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            weights,
            device,
            int(head_config["gradient_accumulation_steps"]),
            float(head_config["gradient_clip"]),
            scaler,
        )
        validation_loss, validation_frame = predict(
            model, validation_loader, store, device
        )
        validation_metrics, _ = evaluate_frame(validation_frame)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_qwk": validation_metrics["quadratic_weighted_kappa"],
            }
        )
        print(
            f"inner epoch={epoch} train={train_loss:.4f} val={validation_loss:.4f} "
            f"animal_F1={validation_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_metrics = validation_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(head_config["early_stopping_patience"]):
            break
    audit = {
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "best_validation_loss": best_loss,
        "best_validation_animal_metrics": best_metrics,
        "history": history,
        "train_seconds": time.perf_counter() - started,
        "parameters": {
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in model.parameters()),
        },
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, audit


def fit_outer(
    store: FeatureStore,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    protocol: dict[str, Any],
    device: torch.device,
    seed: int,
    epochs: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    set_seed(seed)
    head_config = protocol["shared_head"]
    model = build_head(store, train_indices, protocol).to(device)
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=float(head_config["learning_rate"]), eps=1.0e-7
    )
    weights = class_weights(store.labels[train_indices]).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    train_loader = make_loader(
        store, train_indices, int(head_config["batch_size"]), True, seed
    )
    test_loader = make_loader(
        store, test_indices, int(head_config["batch_size"]), False, seed
    )
    training_losses = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        training_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            weights,
            device,
            int(head_config["gradient_accumulation_steps"]),
            float(head_config["gradient_clip"]),
            scaler,
        )
        training_losses.append(training_loss)
        print(f"outer retrain {epoch}/{epochs} train={training_loss:.4f}", flush=True)
    test_loss, test_frame = predict(model, test_loader, store, device)
    test_metrics, _ = evaluate_frame(test_frame)
    audit = {
        "epochs": epochs,
        "training_losses": training_losses,
        "test_loss": test_loss,
        "test_animal_metrics": test_metrics,
        "train_and_predict_seconds": time.perf_counter() - started,
        "parameters": {
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in model.parameters()),
        },
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_frame, audit


def stage_manifest(
    stage: str,
    protocol: dict[str, Any],
    recipe: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "recipe_id": recipe["recipe_id"],
        "pipeline": recipe["pipeline_id"],
        "stage": stage,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "recipe_sha256": sha256(RECIPE_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "git_revision_at_start": git_revision(),
        "roles_sha256": sha256(ROLES_PATH),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def run_smoke(
    run_root: Path,
    store: FeatureStore,
    protocol: dict[str, Any],
    recipe: dict[str, Any],
    device: torch.device,
    resume: bool,
) -> None:
    smoke_root = run_root / "smoke"
    summary_path = smoke_root / "smoke_summary.json"
    if summary_path.exists():
        if not resume:
            raise FileExistsError(f"Smoke result exists; pass --resume to inspect it: {summary_path}")
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return
    smoke = protocol["stages"]["smoke"]
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    indices = indices_for_roles(store, roles, int(smoke["repeat"]), int(smoke["outer_fold"]))
    seed = full_model_seed(
        int(smoke["base_seed"]), int(smoke["repeat"]), int(smoke["outer_fold"])
    )
    best_epoch, audit = fit_inner(
        store,
        indices["train"],
        indices["validation"],
        protocol,
        device,
        seed,
        int(smoke["maximum_epochs"]),
    )
    result = {
        "status": "complete",
        "outer_test_accessed": False,
        "pipeline": recipe["pipeline_id"],
        "repeat": int(smoke["repeat"]),
        "outer_fold": int(smoke["outer_fold"]),
        "base_seed": int(smoke["base_seed"]),
        "full_seed": seed,
        "train_calls": int(len(indices["train"])),
        "validation_calls": int(len(indices["validation"])),
        "best_epoch": best_epoch,
        "audit": audit,
    }
    write_json(smoke_root / "run_manifest.json", stage_manifest("smoke", protocol, recipe, device))
    write_json(summary_path, result)
    print(json.dumps(result, indent=2), flush=True)


def fit_directory(initial_root: Path, repeat: int, outer_fold: int, base_seed: int) -> Path:
    return initial_root / "fits" / f"repeat_{repeat}" / f"fold_{outer_fold}" / f"base_seed_{base_seed}"


def paired_bootstrap(
    candidate_sets: dict[int, pd.DataFrame],
    reference_sets: dict[int, pd.DataFrame],
    bootstrap_repeats: int,
    seed: int = 20260829,
) -> dict[str, Any]:
    repeats = sorted(candidate_sets)
    canonical = candidate_sets[repeats[0]].sort_values("cat_id")
    cat_ids = canonical["cat_id"].to_numpy(dtype=str)
    labels = canonical["true_label"].to_numpy(dtype=np.int64)

    def cube(frames: dict[int, pd.DataFrame]) -> np.ndarray:
        values = np.empty((len(repeats), len(cat_ids), 3), dtype=np.float64)
        for repeat_index, repeat in enumerate(repeats):
            frame = frames[repeat].sort_values("cat_id")
            if not np.array_equal(frame["cat_id"].to_numpy(dtype=str), cat_ids):
                raise RuntimeError("Paired OOF cat order differs")
            if not np.array_equal(frame["true_label"].to_numpy(dtype=np.int64), labels):
                raise RuntimeError("Paired OOF labels differ")
            values[repeat_index] = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
        return values

    candidate = cube(candidate_sets)
    reference = cube(reference_sets)
    observed = []
    for index, repeat in enumerate(repeats):
        candidate_f1 = categorical_metrics(labels, candidate[index])["macro_f1"]
        reference_f1 = categorical_metrics(labels, reference[index])["macro_f1"]
        observed.append(
            {
                "repeat": repeat,
                "candidate_macro_f1": candidate_f1,
                "ast_head_only_macro_f1": reference_f1,
                "candidate_minus_ast_macro_f1": candidate_f1 - reference_f1,
            }
        )
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == label) for label in range(3)]
    bootstrap = np.empty(bootstrap_repeats, dtype=np.float64)
    for bootstrap_index in range(bootstrap_repeats):
        sampled_cats = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        sampled_repeats = rng.integers(0, len(repeats), size=len(repeats))
        differences = []
        sampled_labels = labels[sampled_cats]
        for repeat_index in sampled_repeats:
            differences.append(
                categorical_metrics(sampled_labels, candidate[repeat_index, sampled_cats])[
                    "macro_f1"
                ]
                - categorical_metrics(sampled_labels, reference[repeat_index, sampled_cats])[
                    "macro_f1"
                ]
            )
        bootstrap[bootstrap_index] = float(np.mean(differences))
    observed_values = np.asarray([row["candidate_minus_ast_macro_f1"] for row in observed])
    return {
        "complete_oof_comparisons": observed,
        "mean_candidate_minus_ast_macro_f1": float(observed_values.mean()),
        "positive_complete_oof_evaluations": int((observed_values > 0).sum()),
        "total_complete_oof_evaluations": int(len(observed_values)),
        "paired_bootstrap": {
            "resamples": bootstrap_repeats,
            "seed": seed,
            "bootstrap_mean": float(bootstrap.mean()),
            "ci_lower": float(np.quantile(bootstrap, 0.025)),
            "ci_upper": float(np.quantile(bootstrap, 0.975)),
        },
    }


def aggregate_initial(
    initial_root: Path,
    store: FeatureStore,
    protocol: dict[str, Any],
    bootstrap_repeats: int,
) -> dict[str, Any]:
    stage = protocol["stages"]["initial"]
    base_seed = int(stage["base_seeds"][0])
    candidate_sets: dict[int, pd.DataFrame] = {}
    oof_metrics: dict[str, Any] = {}
    for repeat in stage["repeats"]:
        parts = []
        for outer_fold in stage["outer_folds"]:
            path = fit_directory(initial_root, repeat, outer_fold, base_seed) / "outer_test_call_predictions.csv"
            if not path.is_file():
                raise RuntimeError(f"Missing initial-screening prediction file: {path}")
            parts.append(pd.read_csv(path, dtype={"cat_id": str}))
        calls = pd.concat(parts, ignore_index=True)
        metrics, animals = evaluate_frame(calls)
        if len(animals) != 111 or animals["cat_id"].nunique() != 111:
            raise RuntimeError("A complete IDEA-049 OOF set must contain 111 cats")
        animals.insert(0, "base_seed", base_seed)
        animals.insert(0, "repeat", repeat)
        animals.insert(0, "pipeline", "ssast_hf_base_patch400_frozen_mlp")
        output_path = initial_root / "oof" / f"repeat_{repeat}_base_seed_{base_seed}_animal_predictions.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        animals.to_csv(output_path, index=False)
        candidate_sets[int(repeat)] = animals
        oof_metrics[f"repeat_{repeat}_base_seed_{base_seed}"] = metrics

    values = [metrics["macro_f1"] for metrics in oof_metrics.values()]
    summary: dict[str, Any] = {
        "status": "complete_for_initial_screening",
        "pipeline": "ssast_hf_base_patch400_frozen_mlp",
        "base_seeds": [base_seed],
        "repeats": list(stage["repeats"]),
        "complete_oof_evaluations": len(values),
        "oof_metrics": oof_metrics,
        "aggregate": {
            "macro_f1_mean": float(np.mean(values)),
            "macro_f1_sample_sd": float(np.std(values, ddof=1)),
            "macro_f1_min": float(np.min(values)),
            "macro_f1_max": float(np.max(values)),
        },
    }
    reference_sets = {}
    for repeat in stage["repeats"]:
        reference_path = FORMAL_AST_OOF_ROOT / f"repeat_{repeat}_base_seed_{base_seed}_animal_predictions.csv"
        if reference_path.is_file():
            reference_sets[int(repeat)] = pd.read_csv(reference_path, dtype={"cat_id": str})
    if len(reference_sets) == len(candidate_sets):
        summary["paired_vs_ast_head_only"] = paired_bootstrap(
            candidate_sets, reference_sets, bootstrap_repeats
        )
    elif FORMAL_SUMMARY_PATH.is_file():
        formal = read_json(FORMAL_SUMMARY_PATH)
        comparisons = []
        for repeat in stage["repeats"]:
            key = f"repeat_{repeat}_base_seed_{base_seed}"
            candidate_f1 = oof_metrics[key]["macro_f1"]
            reference_f1 = formal["oof_metrics"]["ast_head_only"][key]["macro_f1"]
            comparisons.append(
                {
                    "repeat": repeat,
                    "candidate_macro_f1": candidate_f1,
                    "ast_head_only_macro_f1": reference_f1,
                    "candidate_minus_ast_macro_f1": candidate_f1 - reference_f1,
                }
            )
        summary["paired_vs_ast_head_only"] = {
            "complete_oof_comparisons": comparisons,
            "mean_candidate_minus_ast_macro_f1": float(
                np.mean([row["candidate_minus_ast_macro_f1"] for row in comparisons])
            ),
            "paired_bootstrap": None,
        }
    write_json(initial_root / "initial_summary.json", summary)
    return summary


def run_initial(
    run_root: Path,
    store: FeatureStore,
    protocol: dict[str, Any],
    recipe: dict[str, Any],
    device: torch.device,
    bootstrap_repeats: int,
    resume: bool,
) -> None:
    initial_root = run_root / "initial"
    initial_root.mkdir(parents=True, exist_ok=True)
    manifest_path = initial_root / "run_manifest.json"
    manifest = stage_manifest("initial", protocol, recipe, device)
    manifest.update(protocol["stages"]["initial"])
    if manifest_path.exists():
        existing = read_json(manifest_path)
        checked = ("protocol_sha256", "recipe_sha256", "runner_sha256", "pipeline")
        if any(existing.get(key) != manifest.get(key) for key in checked):
            raise RuntimeError("Initial-screening resume manifest differs")
        if not resume:
            raise FileExistsError("Initial run exists; pass --resume after review")
    else:
        write_json(manifest_path, manifest)

    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    stage = protocol["stages"]["initial"]
    completed = []
    for repeat in stage["repeats"]:
        for outer_fold in stage["outer_folds"]:
            for base_seed in stage["base_seeds"]:
                output_dir = fit_directory(initial_root, repeat, outer_fold, base_seed)
                summary_path = output_dir / "fit_summary.json"
                if summary_path.exists():
                    if not resume:
                        raise FileExistsError(summary_path)
                    fit_summary = read_json(summary_path)
                    completed.append(fit_summary)
                    print(f"resume: {summary_path}", flush=True)
                    continue
                output_dir.mkdir(parents=True, exist_ok=True)
                indices = indices_for_roles(store, roles, repeat, outer_fold)
                seed = full_model_seed(base_seed, repeat, outer_fold)
                print(
                    f"=== SSAST repeat={repeat} fold={outer_fold} base_seed={base_seed} "
                    f"full_seed={seed} ===",
                    flush=True,
                )
                best_epoch, inner_audit = fit_inner(
                    store,
                    indices["train"],
                    indices["validation"],
                    protocol,
                    device,
                    seed,
                    int(protocol["shared_head"]["maximum_epochs"]),
                )
                outer_train = np.concatenate((indices["train"], indices["validation"]))
                test_frame, outer_audit = fit_outer(
                    store,
                    outer_train,
                    indices["test"],
                    protocol,
                    device,
                    seed,
                    best_epoch,
                )
                test_frame.insert(0, "full_seed", seed)
                test_frame.insert(0, "base_seed", base_seed)
                test_frame.insert(0, "outer_fold", outer_fold)
                test_frame.insert(0, "repeat", repeat)
                test_frame.insert(0, "pipeline", recipe["pipeline_id"])
                prediction_path = output_dir / "outer_test_call_predictions.csv"
                test_frame.to_csv(prediction_path, index=False)
                fit_summary = {
                    "status": "complete",
                    "stage": "initial",
                    "pipeline": recipe["pipeline_id"],
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "base_seed": base_seed,
                    "full_seed": seed,
                    "train_calls": int(len(indices["train"])),
                    "validation_calls": int(len(indices["validation"])),
                    "test_calls": int(len(indices["test"])),
                    "selected_epoch": best_epoch,
                    "inner": inner_audit,
                    "outer": outer_audit,
                    "prediction_path": repo_relative(prediction_path),
                }
                write_json(summary_path, fit_summary)
                completed.append(fit_summary)

    initial_summary = aggregate_initial(
        initial_root, store, protocol, bootstrap_repeats
    )
    run_summary = {
        "status": "complete",
        "stage": "initial",
        "completed_fits": len(completed),
        "expected_fits": int(stage["fold_level_fits"]),
        "complete_oof_evaluations": initial_summary["complete_oof_evaluations"],
        "initial_summary": repo_relative(initial_root / "initial_summary.json"),
        "fits": completed,
    }
    write_json(initial_root / "run_summary.json", run_summary)
    print(json.dumps(initial_summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_PATH)
    recipe = read_json(RECIPE_PATH)
    verify_inputs(protocol, recipe)
    device = resolve_device(args.device)
    run_root = (RUNS_ROOT / args.output_subdir).resolve()
    if RUNS_ROOT.resolve() not in run_root.parents:
        raise ValueError("--output-subdir must stay below the repository runs directory")
    run_root.mkdir(parents=True, exist_ok=True)
    print(
        f"IDEA-049 stage={args.stage}; device={device}; "
        f"device_name={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}",
        flush=True,
    )
    if args.stage == "prepare":
        prepare_embeddings(
            run_root,
            protocol,
            recipe,
            device,
            args.extraction_batch_size,
            args.resume,
        )
        return
    store = load_feature_store(run_root, recipe)
    if args.stage == "smoke":
        run_smoke(run_root, store, protocol, recipe, device, args.resume)
        return
    run_initial(
        run_root,
        store,
        protocol,
        recipe,
        device,
        args.bootstrap_repeats,
        args.resume,
    )


if __name__ == "__main__":
    main()
