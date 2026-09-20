"""Run the IDEA-088 clean-111 projection of the original-style protocol.

The runner never regenerates splits.  It consumes the independently recovered
post-swap cat-role manifest and keeps test prediction behind a persisted
training-checkpoint lock.  VGGish uses the existing TensorFlow/Keras head;
A0/C1 use the existing PyTorch frozen-AST implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_formal_v2_1 as formal  # noqa: E402
import run_meowagenet_idea068_age_sensitive_ast as idea068  # noqa: E402
import run_meowagenet_idea071_bounded_dual_path_fusion as idea071  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea088_original_style_clean_v1.json"
)
PIPELINES = ("vggish_mlp", "A0_ast_only", "C1_bounded_wide_additive")
AST_PIPELINES = PIPELINES[1:]
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
CLASS_NAMES = ("kitten", "adult", "senior")
EXPECTED_AST_PARAMETERS = {
    "A0_ast_only": 99_075,
    "C1_bounded_wide_additive": 108_143,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "fit", "aggregate"), required=True)
    parser.add_argument(
        "--output-subdir", default="meowagenet_idea088_original_style_clean_v1"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--pipeline", choices=PIPELINES)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--fold", type=int, choices=range(4))
    parser.add_argument("--max-fits", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--director-authorized", action="store_true")
    parser.add_argument("--preflight-sha256")
    return parser.parse_args()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def relative_repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def resolve_run_root(output_subdir: str) -> Path:
    if Path(output_subdir).name != output_subdir or "088" not in output_subdir:
        raise RuntimeError("IDEA-088 output_subdir must be a single name containing 088")
    return REPO_ROOT / "runs" / output_subdir


def configure_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        formal.tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    try:
        formal.tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def set_torch_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol["protocol_id"] != "meowagenet-idea088-original-style-clean-v1":
        raise RuntimeError("Unexpected IDEA-088 protocol ID")
    if tuple(protocol["models"]["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-088 pipeline set changed")
    if protocol["data"]["calls"] != 792 or protocol["data"]["cats"] != 111:
        raise RuntimeError("IDEA-088 clean data scope changed")
    if protocol["data"]["vggish_rows"] != 936:
        raise RuntimeError("IDEA-088 clean VGGish row count changed")
    if protocol["data"]["excluded_aliases"] != ["049A"]:
        raise RuntimeError("IDEA-088 alias projection changed")
    if protocol["splits"]["seeds"] != [7270, 860, 5390, 5191, 5734]:
        raise RuntimeError("IDEA-088 original seed list changed")
    if protocol["splits"]["folds_per_seed"] != 4:
        raise RuntimeError("IDEA-088 fold count changed")
    if protocol["splits"]["forced_training_cat_ids"] != ["000A", "046A"]:
        raise RuntimeError("IDEA-088 forced-training cat set changed")
    fixed = protocol["training"]
    expected = {
        "optimizer": "Adamax",
        "learning_rate": 0.003109800273709165,
        "epsilon": 1.0e-7,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.0,
        "dropout": 0.44571035356880917,
        "batch_size": 128,
        "maximum_epochs": 1500,
        "early_stopping_monitor": "training_loss",
        "early_stopping_mode": "min",
        "early_stopping_min_delta": 0.001,
        "early_stopping_patience": 30,
        "restore_best_weights": True,
        "gradient_clip": None,
        "automatic_mixed_precision": False,
        "post_build_seed_offset": 1_000_000,
    }
    for key, value in expected.items():
        if fixed.get(key) != value:
            raise RuntimeError(f"IDEA-088 training field changed: {key}")
    if protocol["budget"] != {
        "pipelines": 3,
        "split_seeds": 5,
        "folds_per_seed": 4,
        "fits": 60,
    }:
        raise RuntimeError("IDEA-088 budget changed")
    if protocol["reporting"]["claims_complete_oof"]:
        raise RuntimeError("IDEA-088 must not claim complete OOF")
    for entry in protocol["dependencies"]:
        path = REPO_ROOT / entry["path"]
        if not path.is_file():
            raise RuntimeError(f"Missing IDEA-088 dependency: {path}")
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"IDEA-088 dependency hash changed: {path}")


def model_seed(split_seed: int, fold: int) -> int:
    return int(split_seed) + int(fold)


def training_seed(protocol: dict[str, Any], split_seed: int, fold: int) -> int:
    return model_seed(split_seed, fold) + int(
        protocol["training"]["post_build_seed_offset"]
    )


def epoch_seed(base_training_seed: int, epoch: int) -> int:
    if epoch < 1:
        raise ValueError("epoch is one-based")
    return int(base_training_seed) + int(epoch)


def epoch_order(unit_indices: np.ndarray, base_training_seed: int, epoch: int) -> np.ndarray:
    generator = np.random.default_rng(epoch_seed(base_training_seed, epoch))
    return unit_indices[generator.permutation(len(unit_indices))].astype(np.int64)


def batch_sizes_for_units(unit_count: int, batch_size: int) -> list[int]:
    if unit_count < 1 or batch_size < 1:
        raise ValueError("unit_count and batch_size must be positive")
    sizes = [int(batch_size)] * (int(unit_count) // int(batch_size))
    tail = int(unit_count) % int(batch_size)
    if tail:
        sizes.append(tail)
    return sizes


def class_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if len(labels) == 0 or np.any(counts == 0):
        raise RuntimeError("IDEA-088 training role is empty or missing a class")
    return (len(labels) / (3.0 * counts)).astype(np.float32)


def weighted_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    per_unit = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    return (per_unit * weights[labels]).sum() / int(labels.numel())


def early_stopping_update(
    current: float,
    best: float,
    stale: int,
    min_delta: float,
    patience: int,
) -> tuple[bool, float, int, bool]:
    improved = float(current) < float(best) - float(min_delta)
    if improved:
        return True, float(current), 0, False
    stale += 1
    return False, float(best), stale, stale >= int(patience)


def _sorted_unique_strings(values: Iterable[Any], label: str) -> list[str]:
    result = [str(value) for value in values]
    if result != sorted(result) or len(result) != len(set(result)):
        raise RuntimeError(f"IDEA-088 {label} must be unique and sorted")
    return result


def validate_roles_manifest(
    manifest: dict[str, Any], protocol: dict[str, Any], expected_cats: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if str(manifest.get("schema_version")) != "1.0":
        raise RuntimeError("IDEA-088 role manifest schema changed")
    manifest_protocol = manifest.get("protocol_id") or manifest.get("idea_id")
    if manifest_protocol not in {
        "meowagenet-idea088-original-style-clean-v1",
        "IDEA-088",
    }:
        raise RuntimeError("IDEA-088 role manifest identity changed")
    excluded = manifest.get("excluded_published_cat_ids")
    if excluded is None:
        excluded = manifest.get("projection", {}).get("excluded_cat_ids")
    if excluded != ["049A"]:
        raise RuntimeError("IDEA-088 role manifest alias projection changed")
    projection = manifest.get("projection", {})
    if projection.get("expected_calls") != 792 or projection.get("expected_cats") != 111:
        raise RuntimeError("IDEA-088 role manifest clean scope changed")
    if projection.get("expected_vggish_rows") != 936:
        raise RuntimeError("IDEA-088 role manifest VGGish row scope changed")
    if manifest.get("source_prediction_rows") != 937:
        raise RuntimeError("IDEA-088 role manifest source row scope changed")
    vggish_input = manifest.get("vggish_input", {})
    if vggish_input.get("feature_dimensions") != 129:
        raise RuntimeError("IDEA-088 role manifest VGGish input dimension changed")
    expected_vggish_columns = [str(index) for index in range(128)] + ["mean_freq"]
    if vggish_input.get("feature_columns") != expected_vggish_columns:
        raise RuntimeError("IDEA-088 role manifest VGGish input columns changed")
    label_audit = manifest.get("classifier_labels", {})
    if label_audit.get("clean_cat_counts") != protocol["data"]["expected_cat_class_counts"]:
        raise RuntimeError("IDEA-088 role manifest cat label counts changed")
    if label_audit.get("clean_call_counts") != protocol["data"]["expected_call_class_counts"]:
        raise RuntimeError("IDEA-088 role manifest call label counts changed")
    if manifest.get("split_seeds") != protocol["splits"]["seeds"]:
        raise RuntimeError("IDEA-088 role manifest seed order changed")
    if manifest.get("forced_training_cat_ids") != protocol["splits"]["forced_training_cat_ids"]:
        raise RuntimeError("IDEA-088 role manifest forced-training cats changed")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != 20:
        raise RuntimeError("IDEA-088 role manifest must contain exactly 20 cells")
    expected_identities = {
        (seed, fold)
        for seed in protocol["splits"]["seeds"]
        for fold in range(protocol["splits"]["folds_per_seed"])
    }
    identities: set[tuple[int, int]] = set()
    normalized: list[dict[str, Any]] = []
    per_seed_counts: dict[str, dict[str, int]] = {}
    for raw in cells:
        seed = int(raw["seed"] if "seed" in raw else raw["split_seed"])
        fold = int(raw["fold"])
        identity = (seed, fold)
        if identity in identities:
            raise RuntimeError("IDEA-088 role manifest contains a duplicate cell")
        identities.add(identity)
        clean = raw.get("clean", raw)
        train = _sorted_unique_strings(clean["train_cat_ids"], "train_cat_ids")
        test = _sorted_unique_strings(clean["test_cat_ids"], "test_cat_ids")
        train_set, test_set = set(train), set(test)
        if train_set & test_set:
            raise RuntimeError("IDEA-088 role manifest has train/test cat leakage")
        if train_set | test_set != expected_cats:
            raise RuntimeError("IDEA-088 role cell does not cover the clean 111 cats")
        if "049A" in train_set | test_set:
            raise RuntimeError("IDEA-088 role cell reintroduced excluded alias 049A")
        forced = set(protocol["splits"]["forced_training_cat_ids"])
        if not forced <= train_set or forced & test_set:
            raise RuntimeError("IDEA-088 forced-training cats changed role")
        normalized.append(
            {
                "cell_id": str(raw.get("cell_id", f"seed_{seed}_fold_{fold}")),
                "split_seed": seed,
                "fold": fold,
                "train_cat_ids": train,
                "test_cat_ids": test,
                "source_order": raw.get("source_order"),
                "source_reference": raw.get("source_reference"),
            }
        )
    if identities != expected_identities:
        raise RuntimeError("IDEA-088 role manifest cell identities are incomplete")
    for seed in protocol["splits"]["seeds"]:
        counts = Counter(
            cat
            for cell in normalized
            if cell["split_seed"] == seed
            for cat in cell["test_cat_ids"]
        )
        if any(counts.get(cat, 0) != 0 for cat in protocol["splits"]["forced_training_cat_ids"]):
            raise RuntimeError("IDEA-088 forced-training cat reached a test fold")
        per_seed_counts[str(seed)] = {
            cat: int(counts.get(cat, 0)) for cat in sorted(expected_cats)
        }
    normalized.sort(key=lambda row: (protocol["splits"]["seeds"].index(row["split_seed"]), row["fold"]))
    audit = {
        "cells": len(normalized),
        "expected_cats": len(expected_cats),
        "per_seed_test_counts": per_seed_counts,
        "forced_training_cats_never_test": True,
        "complete_oof_claim_allowed": False,
    }
    return normalized, audit


def load_feature_inputs(protocol: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    store = idea068.idea051.reference.historical.idea019.load_feature_store()
    if len(store.call_ids) != 792 or len(np.unique(store.cat_ids.astype(str))) != 111:
        raise RuntimeError("IDEA-088 expected 792 AST calls from 111 cats")
    feature_path = REPO_ROOT / protocol["data"]["acoustic_feature_path"]
    loaded = np.load(feature_path)
    if tuple(loaded["feature_names"].astype(str)) != idea068.FEATURE_NAMES:
        raise RuntimeError("IDEA-088 acoustic feature definition changed")
    if not np.array_equal(loaded["call_ids"].astype(str), store.call_ids.astype(str)):
        raise RuntimeError("IDEA-088 acoustic call order changed")
    age_features = loaded["features"].astype(np.float32)
    if age_features.shape != (792, 20):
        raise RuntimeError("IDEA-088 acoustic feature shape changed")

    frame = pd.read_csv(
        REPO_ROOT / protocol["data"]["vggish_csv_path"], dtype={"cat_id": str}
    )
    source_rows = len(frame)
    frame = frame[frame["cat_id"] != "049A"].copy()
    frame["source_row_index"] = frame.index.astype(np.int64)
    frame = frame.reset_index(drop=True)
    feature_columns = [str(index) for index in range(128)] + ["mean_freq"]
    if len(frame) != 936 or frame["cat_id"].nunique() != 111:
        raise RuntimeError("IDEA-088 expected 936 clean VGGish rows from 111 cats")
    if feature_columns[-1] != "mean_freq" or set(feature_columns[:-1]) != {
        str(index) for index in range(128)
    }:
        raise RuntimeError("IDEA-088 original-final 129-dimensional input changed")
    target = frame["target"].to_numpy(dtype=np.float64)
    vggish_labels = np.where(target < 0.5, 0, np.where(target < 10, 1, 2)).astype(np.int64)
    vggish = {
        "features": frame[feature_columns].to_numpy(dtype=np.float32),
        "labels": vggish_labels,
        "cat_ids": frame["cat_id"].to_numpy(dtype=str),
        "unit_ids": np.asarray(
            [f"vggish_source_row_{value}" for value in frame["source_row_index"]],
            dtype=str,
        ),
        "source_row_indices": frame["source_row_index"].to_numpy(dtype=np.int64),
    }
    ast_cat_labels: dict[str, int] = {}
    for cat in np.unique(store.cat_ids.astype(str)):
        labels = np.unique(store.labels[store.cat_ids.astype(str) == cat].astype(np.int64))
        if len(labels) != 1:
            raise RuntimeError(f"IDEA-088 AST cat has conflicting labels: {cat}")
        ast_cat_labels[cat] = int(labels[0])
    expected_cat_counts = [
        int(protocol["data"]["expected_cat_class_counts"][name]) for name in CLASS_NAMES
    ]
    actual_cat_counts = np.bincount(
        np.asarray(list(ast_cat_labels.values()), dtype=np.int64), minlength=3
    ).astype(int).tolist()
    if actual_cat_counts != expected_cat_counts:
        raise RuntimeError("IDEA-088 clean cat class counts changed")
    expected_call_counts = [
        int(protocol["data"]["expected_call_class_counts"][name]) for name in CLASS_NAMES
    ]
    actual_call_counts = np.bincount(store.labels.astype(np.int64), minlength=3).astype(int).tolist()
    if actual_call_counts != expected_call_counts:
        raise RuntimeError("IDEA-088 clean call class counts changed")
    for cat in np.unique(vggish["cat_ids"]):
        labels = np.unique(vggish_labels[vggish["cat_ids"] == cat])
        if len(labels) != 1 or int(labels[0]) != ast_cat_labels[cat]:
            raise RuntimeError(f"IDEA-088 VGGish/AST cat label mismatch: {cat}")
    audit = {
        "source_vggish_rows": int(source_rows),
        "clean_vggish_rows": int(len(frame)),
        "ast_calls": int(len(store.call_ids)),
        "cats": int(len(ast_cat_labels)),
        "labels": {name: int(index) for index, name in enumerate(CLASS_NAMES)},
        "cat_class_counts": actual_cat_counts,
        "call_class_counts": actual_call_counts,
        "vggish_feature_columns": feature_columns,
        "vggish_auxiliary_columns_excluded": ["gender", "target", "cat_id", "age_group"],
    }
    return vggish, age_features, {"store": store, "cat_labels": ast_cat_labels, "audit": audit}


def indices_for_cats(cat_ids: np.ndarray, cats: Iterable[str]) -> np.ndarray:
    cat_set = set(str(value) for value in cats)
    indices = np.flatnonzero(np.isin(cat_ids.astype(str), sorted(cat_set))).astype(np.int64)
    if set(cat_ids[indices].astype(str)) != cat_set:
        raise RuntimeError("IDEA-088 failed to resolve every role cat to prediction units")
    return indices


def safe_mean_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale = np.where(scale > 1.0e-12, scale, 1.0).astype(np.float32)
    return mean, scale


def build_ast_model(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    age_features: np.ndarray,
    train_indices: np.ndarray,
    seed: int,
) -> torch.nn.Module:
    set_torch_seed(seed)
    embeddings = store.frozen_embeddings[train_indices]
    common = {
        "ast_mean": embeddings.mean(axis=0),
        "ast_scale": embeddings.std(axis=0),
        "age_train": age_features[train_indices],
        "dropout": float(protocol["training"]["dropout"]),
    }
    if pipeline == "A0_ast_only":
        model = idea068.AgeResidualClassifier(
            pipeline="A0_ast_only", age_hidden_units=32, **common
        )
    elif pipeline == "C1_bounded_wide_additive":
        model = idea071.BoundedWideAdditiveClassifier(**common)
    else:
        raise ValueError(pipeline)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_AST_PARAMETERS[pipeline]:
        raise RuntimeError(f"IDEA-088 {pipeline} parameter count changed")
    return model


def torch_state_digest(model: torch.nn.Module, common_only: bool = False) -> str:
    digest = hashlib.sha256()
    common_prefixes = ("ast_mean", "ast_scale", "ast_linear.", "batch_norm.", "output.")
    for name, tensor in sorted(model.state_dict().items()):
        if common_only and not any(
            name == prefix or name.startswith(prefix) for prefix in common_prefixes
        ):
            continue
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def keras_state_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for index, array in enumerate(model.get_weights()):
        value = np.ascontiguousarray(array)
        digest.update(str(index).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def vggish_recipe(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "vggish_mlp": {
            "features": 129,
            "dropout": float(protocol["training"]["dropout"]),
            "learning_rate": float(protocol["training"]["learning_rate"]),
        }
    }


class LoggedVggishSequence(formal.tf.keras.utils.Sequence):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        source_indices: np.ndarray,
        batch_size: int,
        base_training_seed: int,
    ) -> None:
        self.features = features
        self.labels = labels
        self.source_indices = source_indices.astype(np.int64)
        self.batch_size = int(batch_size)
        self.base_training_seed = int(base_training_seed)
        self.epoch = 1
        self.orders: list[np.ndarray] = []
        self._set_order()

    def _set_order(self) -> None:
        source_order = epoch_order(self.source_indices, self.base_training_seed, self.epoch)
        position = {int(source): index for index, source in enumerate(self.source_indices)}
        self.local_order = np.asarray([position[int(value)] for value in source_order], dtype=np.int64)
        self.orders.append(source_order.copy())

    def __len__(self) -> int:
        return int(math.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        selected = self.local_order[index * self.batch_size : (index + 1) * self.batch_size]
        return self.features[selected], self.labels[selected]

    def on_epoch_end(self) -> None:
        self.epoch += 1
        self._set_order()


class FailOnNonFiniteTrainingLoss(formal.tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        value = None if logs is None else logs.get("loss")
        if value is None or not np.isfinite(float(value)):
            raise RuntimeError(
                f"IDEA-088 non-finite TensorFlow training loss at epoch {epoch + 1}"
            )


def restore_keras_recorded_best(model: Any, callback: Any) -> None:
    best_weights = getattr(callback, "best_weights", None)
    if not best_weights:
        raise RuntimeError("IDEA-088 TensorFlow early stopping recorded no best weights")
    model.set_weights(best_weights)


def metric_bundle(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    recalls = recall_score(
        labels, predictions, labels=[0, 1, 2], average=None, zero_division=0
    )
    clipped = np.clip(probabilities, 1.0e-12, 1.0)
    targets = np.eye(3, dtype=np.float64)[labels]
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(np.mean(recalls)),
        "class_recall": {
            name: float(value) for name, value in zip(CLASS_NAMES, recalls, strict=True)
        },
        "cross_entropy": float(-np.mean(np.log(clipped[np.arange(len(labels)), labels]))),
        "brier": float(np.mean(np.sum((probabilities - targets) ** 2, axis=1))),
    }


def units_to_cats(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cat_id, group in units.groupby("cat_id", sort=True):
        labels = group["true_label"].unique()
        if len(labels) != 1:
            raise RuntimeError(f"IDEA-088 conflicting test labels for cat {cat_id}")
        probabilities = group[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64).mean(axis=0)
        rows.append(
            {
                "cat_id": str(cat_id),
                "true_label": int(labels[0]),
                "unit_count": int(len(group)),
                **{
                    column: float(probabilities[index])
                    for index, column in enumerate(PROBABILITY_COLUMNS)
                },
                "predicted_label": int(probabilities.argmax()),
            }
        )
    return pd.DataFrame(rows).sort_values("cat_id").reset_index(drop=True)


def prediction_metrics(units: pd.DataFrame, cats: pd.DataFrame) -> dict[str, Any]:
    return {
        "public_cat_probability_mean": metric_bundle(
            cats["true_label"].to_numpy(dtype=np.int64),
            cats[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64),
        ),
        "native_prediction_unit": metric_bundle(
            units["true_label"].to_numpy(dtype=np.int64),
            units[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64),
        ),
    }


def fit_directory(run_root: Path, pipeline: str, split_seed: int, fold: int) -> Path:
    return run_root / "fits" / pipeline / f"seed_{split_seed}" / f"fold_{fold}"


def _save_epoch_orders(path: Path, orders: list[np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.stack([np.asarray(order, dtype=np.int32) for order in orders], axis=0)
    np.savez_compressed(path, unit_indices=matrix)
    return sha256(path)


def _save_preprocessing(path: Path, values: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **values)
    return sha256(path)


def _write_predictions(
    fit_root: Path, units: pd.DataFrame, cats: pd.DataFrame
) -> dict[str, Any]:
    unit_path = fit_root / "test_unit_predictions.csv"
    cat_path = fit_root / "test_cat_predictions.csv"
    units.to_csv(unit_path, index=False)
    cats.to_csv(cat_path, index=False)
    return {
        "unit_predictions": relative_repo_path(unit_path),
        "unit_predictions_sha256": sha256(unit_path),
        "cat_predictions": relative_repo_path(cat_path),
        "cat_predictions_sha256": sha256(cat_path),
    }


def _base_fit_identity(
    protocol: dict[str, Any], roles_path: Path, pipeline: str, split_seed: int, fold: int
) -> dict[str, Any]:
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "roles_sha256": sha256(roles_path),
        "pipeline": pipeline,
        "split_seed": int(split_seed),
        "fold": int(fold),
        "model_seed": model_seed(split_seed, fold),
        "training_seed": training_seed(protocol, split_seed, fold),
    }


def fit_vggish(
    protocol: dict[str, Any],
    roles_path: Path,
    vggish: dict[str, np.ndarray],
    cell: dict[str, Any],
    fit_root: Path,
) -> dict[str, Any]:
    split_seed = int(cell["split_seed"])
    fold = int(cell["fold"])
    train_indices = indices_for_cats(vggish["cat_ids"], cell["train_cat_ids"])
    test_indices = indices_for_cats(vggish["cat_ids"], cell["test_cat_ids"])
    mean, scale = safe_mean_scale(vggish["features"][train_indices])
    x_train = ((vggish["features"][train_indices] - mean) / scale).astype(np.float32)
    x_test = ((vggish["features"][test_indices] - mean) / scale).astype(np.float32)
    y_train = vggish["labels"][train_indices]
    weights = class_weights(y_train)
    seed = model_seed(split_seed, fold)
    base_training_seed = training_seed(protocol, split_seed, fold)
    model = formal.build_vggish_model(vggish_recipe(protocol), seed)
    initial_digest = keras_state_digest(model)
    sequence = LoggedVggishSequence(
        x_train,
        y_train,
        train_indices,
        int(protocol["training"]["batch_size"]),
        base_training_seed,
    )
    callback = formal.tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=float(protocol["training"]["early_stopping_min_delta"]),
        patience=int(protocol["training"]["early_stopping_patience"]),
        mode="min",
        restore_best_weights=True,
    )
    started = time.perf_counter()
    history_object = model.fit(
        sequence,
        epochs=int(protocol["training"]["maximum_epochs"]),
        class_weight={index: float(value) for index, value in enumerate(weights)},
        callbacks=[FailOnNonFiniteTrainingLoss(), callback],
        verbose=0,
        shuffle=False,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=1,
    )
    train_seconds = time.perf_counter() - started
    history_losses = [float(value) for value in history_object.history["loss"]]
    if not history_losses or not np.isfinite(np.asarray(history_losses)).all():
        raise RuntimeError("IDEA-088 TensorFlow training history contains non-finite loss")
    stopped_epoch = len(history_losses)
    best_epoch = int(getattr(callback, "best_epoch", int(np.argmin(history_losses))) + 1)
    best_loss = float(getattr(callback, "best", history_losses[best_epoch - 1]))
    restore_keras_recorded_best(model, callback)
    orders = sequence.orders[:stopped_epoch]
    order_path = fit_root / "epoch_unit_orders.npz"
    order_hash = _save_epoch_orders(order_path, orders)
    preprocessing_path = fit_root / "training_preprocessing.npz"
    preprocessing_hash = _save_preprocessing(
        preprocessing_path,
        {
            "feature_mean": mean,
            "feature_scale": scale,
            "class_weights": weights,
            "train_indices": train_indices.astype(np.int64),
            "test_indices": test_indices.astype(np.int64),
        },
    )
    weights_path = fit_root / "model.weights.h5"
    model.save_weights(weights_path)
    restored = formal.build_vggish_model(vggish_recipe(protocol), seed)
    restored.load_weights(weights_path)
    restored_digest = keras_state_digest(restored)
    if restored_digest != keras_state_digest(model):
        raise RuntimeError("IDEA-088 VGGish checkpoint reload changed weights")
    checkpoint_lock = {
        **_base_fit_identity(protocol, roles_path, "vggish_mlp", split_seed, fold),
        "status": "checkpoint_locked_before_test_prediction",
        "framework": "TensorFlow/Keras",
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_training_loss": best_loss,
        "model_weights": relative_repo_path(weights_path),
        "model_weights_sha256": sha256(weights_path),
        "restored_state_sha256": restored_digest,
        "test_accessed": False,
    }
    checkpoint_path = fit_root / "checkpoint_lock.json"
    write_json(checkpoint_path, checkpoint_lock)

    test_probabilities = restored.predict(
        x_test, batch_size=int(protocol["training"]["batch_size"]), verbose=0
    )
    units = pd.DataFrame(
        {
            "unit_index": test_indices,
            "unit_id": vggish["unit_ids"][test_indices],
            "cat_id": vggish["cat_ids"][test_indices],
            "true_label": vggish["labels"][test_indices],
            **{
                column: test_probabilities[:, index]
                for index, column in enumerate(PROBABILITY_COLUMNS)
            },
        }
    ).sort_values("unit_index").reset_index(drop=True)
    cats = units_to_cats(units)
    prediction_files = _write_predictions(fit_root, units, cats)
    return {
        **_base_fit_identity(protocol, roles_path, "vggish_mlp", split_seed, fold),
        "status": "complete",
        "framework": "TensorFlow/Keras 2.15",
        "framework_adaptation": "Existing formal Keras isomorphic upstream head; explicit 0=kitten,1=adult,2=senior labels. TensorFlow initialization is not bitwise matched to PyTorch A0/C1.",
        "input_unit": "VGGish embedding row",
        "train_cats": len(cell["train_cat_ids"]),
        "test_cats": len(cell["test_cat_ids"]),
        "train_units": int(len(train_indices)),
        "test_units": int(len(test_indices)),
        "class_weights": weights.astype(float).tolist(),
        "initial_state_sha256": initial_digest,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_training_loss": best_loss,
        "history": [
            {
                "epoch": index + 1,
                "training_loss": loss,
                "unit_order_sha256": array_sha256(orders[index].astype("<i8")),
            }
            for index, loss in enumerate(history_losses)
        ],
        "epoch_orders": relative_repo_path(order_path),
        "epoch_orders_sha256": order_hash,
        "preprocessing": relative_repo_path(preprocessing_path),
        "preprocessing_sha256": preprocessing_hash,
        "checkpoint_lock": relative_repo_path(checkpoint_path),
        "checkpoint_lock_sha256": sha256(checkpoint_path),
        "model_weights": relative_repo_path(weights_path),
        "model_weights_sha256": sha256(weights_path),
        "checkpoint_reload_state_match": True,
        "test_prediction_calls": 1,
        "metrics": prediction_metrics(units, cats),
        "train_seconds": float(train_seconds),
        **prediction_files,
    }


def predict_ast_units(
    model: torch.nn.Module,
    store: Any,
    age_features: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            embeddings = torch.from_numpy(store.frozen_embeddings[selected]).to(device)
            acoustic = torch.from_numpy(age_features[selected]).to(device)
            logits = model(embeddings, acoustic)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            for local, unit_index in enumerate(selected):
                rows.append(
                    {
                        "unit_index": int(unit_index),
                        "unit_id": str(store.call_ids[unit_index]),
                        "cat_id": str(store.cat_ids[unit_index]),
                        "true_label": int(store.labels[unit_index]),
                        **{
                            column: float(probabilities[local, class_index])
                            for class_index, column in enumerate(PROBABILITY_COLUMNS)
                        },
                    }
                )
    return pd.DataFrame(rows).sort_values("unit_index").reset_index(drop=True)


def fit_ast(
    protocol: dict[str, Any],
    roles_path: Path,
    pipeline: str,
    store: Any,
    age_features: np.ndarray,
    cell: dict[str, Any],
    fit_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    split_seed = int(cell["split_seed"])
    fold = int(cell["fold"])
    train_indices = indices_for_cats(store.cat_ids, cell["train_cat_ids"])
    test_indices = indices_for_cats(store.cat_ids, cell["test_cat_ids"])
    seed = model_seed(split_seed, fold)
    base_training_seed = training_seed(protocol, split_seed, fold)
    model = build_ast_model(
        pipeline, protocol, store, age_features, train_indices, seed
    ).to(device)
    initial_digest = torch_state_digest(model)
    initial_common_digest = torch_state_digest(model, common_only=True)
    weights = class_weights(store.labels[train_indices])
    weight_tensor = torch.from_numpy(weights).to(device)
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=float(protocol["training"]["learning_rate"]),
        betas=(
            float(protocol["training"]["beta1"]),
            float(protocol["training"]["beta2"]),
        ),
        eps=float(protocol["training"]["epsilon"]),
        weight_decay=0.0,
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = idea068.idea051.cpu_state_dict(model)
    stale = 0
    history: list[dict[str, Any]] = []
    orders: list[np.ndarray] = []
    batch_size = int(protocol["training"]["batch_size"])
    started = time.perf_counter()
    for epoch in range(1, int(protocol["training"]["maximum_epochs"]) + 1):
        order = epoch_order(train_indices, base_training_seed, epoch)
        if len(order) != len(train_indices) or len(np.unique(order)) != len(train_indices):
            raise RuntimeError("IDEA-088 epoch order is not a full unique training permutation")
        if not np.array_equal(np.sort(order), np.sort(train_indices)):
            raise RuntimeError("IDEA-088 epoch order changed training-call coverage")
        orders.append(order.copy())
        set_torch_seed(epoch_seed(base_training_seed, epoch))
        model.train()
        epoch_weighted_loss_sum = 0.0
        epoch_units = 0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            labels = torch.from_numpy(store.labels[selected].astype(np.int64)).to(device)
            embeddings = torch.from_numpy(store.frozen_embeddings[selected]).to(device)
            acoustic = torch.from_numpy(age_features[selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(embeddings, acoustic)
            loss = weighted_cross_entropy(logits, labels, weight_tensor)
            loss.backward()
            optimizer.step()
            epoch_weighted_loss_sum += float(loss.detach().cpu()) * len(selected)
            epoch_units += len(selected)
        if epoch_units != len(train_indices):
            raise RuntimeError("IDEA-088 epoch did not cover each training call once")
        epoch_loss = epoch_weighted_loss_sum / epoch_units
        if not math.isfinite(epoch_loss):
            raise RuntimeError(
                f"IDEA-088 non-finite PyTorch training loss at epoch {epoch}"
            )
        improved, best_loss, stale, should_stop = early_stopping_update(
            epoch_loss,
            best_loss,
            stale,
            float(protocol["training"]["early_stopping_min_delta"]),
            int(protocol["training"]["early_stopping_patience"]),
        )
        if improved:
            best_epoch = epoch
            best_state = idea068.idea051.cpu_state_dict(model)
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(epoch_loss),
                "unit_order_sha256": array_sha256(order.astype("<i8")),
            }
        )
        print(
            f"IDEA-088 {pipeline} seed={split_seed} fold={fold} "
            f"epoch={epoch} train_loss={epoch_loss:.6f}",
            flush=True,
        )
        if should_stop:
            break
    train_seconds = time.perf_counter() - started
    if best_epoch < 1:
        raise RuntimeError("IDEA-088 selected no training-loss checkpoint")
    model.load_state_dict(best_state)
    order_path = fit_root / "epoch_unit_orders.npz"
    order_hash = _save_epoch_orders(order_path, orders)
    preprocessing_values = {
        "ast_mean": model.ast_mean.detach().cpu().numpy(),
        "ast_scale": model.ast_scale.detach().cpu().numpy(),
        "class_weights": weights,
        "train_indices": train_indices.astype(np.int64),
        "test_indices": test_indices.astype(np.int64),
    }
    if pipeline == "C1_bounded_wide_additive":
        preprocessing_values.update(
            {
                "age_median": model.age_median.detach().cpu().numpy(),
                "age_mean": model.age_mean.detach().cpu().numpy(),
                "age_scale": model.age_scale.detach().cpu().numpy(),
            }
        )
    preprocessing_path = fit_root / "training_preprocessing.npz"
    preprocessing_hash = _save_preprocessing(preprocessing_path, preprocessing_values)
    weights_path = fit_root / "model_weights.pt"
    torch.save({"state_dict": best_state}, weights_path)
    restored = build_ast_model(
        pipeline, protocol, store, age_features, train_indices, seed
    ).to(device)
    loaded = torch.load(weights_path, map_location=device, weights_only=True)
    restored.load_state_dict(loaded["state_dict"])
    restored_digest = torch_state_digest(restored)
    if restored_digest != torch_state_digest(model):
        raise RuntimeError("IDEA-088 AST checkpoint reload changed weights")
    checkpoint_lock = {
        **_base_fit_identity(protocol, roles_path, pipeline, split_seed, fold),
        "status": "checkpoint_locked_before_test_prediction",
        "framework": "PyTorch",
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "best_training_loss": float(best_loss),
        "model_weights": relative_repo_path(weights_path),
        "model_weights_sha256": sha256(weights_path),
        "restored_state_sha256": restored_digest,
        "test_accessed": False,
    }
    checkpoint_path = fit_root / "checkpoint_lock.json"
    write_json(checkpoint_path, checkpoint_lock)

    units = predict_ast_units(
        restored, store, age_features, test_indices, batch_size, device
    )
    cats = units_to_cats(units)
    prediction_files = _write_predictions(fit_root, units, cats)
    return {
        **_base_fit_identity(protocol, roles_path, pipeline, split_seed, fold),
        "status": "complete",
        "framework": f"PyTorch {torch.__version__}",
        "framework_adaptation": "Existing IDEA-087 frozen-AST A0/C1 structures in FP32; model_seed=original split seed+fold and a fixed post-build offset control paired call order/dropout RNG.",
        "input_unit": "AST call",
        "train_cats": len(cell["train_cat_ids"]),
        "test_cats": len(cell["test_cat_ids"]),
        "train_units": int(len(train_indices)),
        "test_units": int(len(test_indices)),
        "class_weights": weights.astype(float).tolist(),
        "initial_state_sha256": initial_digest,
        "initial_common_state_sha256": initial_common_digest,
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "best_training_loss": float(best_loss),
        "history": history,
        "epoch_orders": relative_repo_path(order_path),
        "epoch_orders_sha256": order_hash,
        "preprocessing": relative_repo_path(preprocessing_path),
        "preprocessing_sha256": preprocessing_hash,
        "checkpoint_lock": relative_repo_path(checkpoint_path),
        "checkpoint_lock_sha256": sha256(checkpoint_path),
        "model_weights": relative_repo_path(weights_path),
        "model_weights_sha256": sha256(weights_path),
        "checkpoint_reload_state_match": True,
        "test_prediction_calls": 1,
        "metrics": prediction_metrics(units, cats),
        "train_seconds": float(train_seconds),
        **prediction_files,
    }


def load_protocol_and_roles() -> tuple[dict[str, Any], Path, dict[str, Any]]:
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    roles_path = REPO_ROOT / protocol["splits"]["roles_path"]
    if not roles_path.is_file():
        raise RuntimeError("IDEA-088 recovered role manifest is not available")
    if sha256(roles_path) != protocol["splits"]["roles_sha256"]:
        raise RuntimeError("IDEA-088 recovered role manifest hash changed")
    return protocol, roles_path, read_json(roles_path)


def preflight_path(run_root: Path) -> Path:
    return run_root / "preflight" / "cpu_preflight.json"


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cpu":
        raise RuntimeError("IDEA-088 preflight must run on CPU")
    configure_determinism()
    protocol, roles_path, roles_manifest = load_protocol_and_roles()
    vggish, age_features, ast = load_feature_inputs(protocol)
    store = ast["store"]
    expected_cats = set(store.cat_ids.astype(str))
    cells, role_audit = validate_roles_manifest(roles_manifest, protocol, expected_cats)
    first_cell = cells[0]
    train_indices = indices_for_cats(store.cat_ids, first_cell["train_cat_ids"])
    seed = model_seed(first_cell["split_seed"], first_cell["fold"])
    models = {
        pipeline: build_ast_model(
            pipeline, protocol, store, age_features, train_indices, seed
        )
        for pipeline in AST_PIPELINES
    }
    common = {
        pipeline: torch_state_digest(model, common_only=True)
        for pipeline, model in models.items()
    }
    if len(set(common.values())) != 1:
        raise RuntimeError("IDEA-088 A0/C1 common initial state is not paired")
    probe = train_indices[:16]
    logits = {}
    for pipeline, model in models.items():
        model.eval()
        with torch.no_grad():
            logits[pipeline] = model(
                torch.from_numpy(store.frozen_embeddings[probe]),
                torch.from_numpy(age_features[probe]),
            ).numpy()
    if not np.array_equal(logits[AST_PIPELINES[0]], logits[AST_PIPELINES[1]]):
        raise RuntimeError("IDEA-088 zero-init C1 does not match A0 initial logits")
    vggish_model = formal.build_vggish_model(vggish_recipe(protocol), seed)
    vggish_parameters = int(vggish_model.count_params())
    if vggish_parameters != 17_539:
        raise RuntimeError("IDEA-088 VGGish head parameter count changed")
    batch_size = int(protocol["training"]["batch_size"])
    ast_batch_audit = []
    for cell in cells:
        cell_train = indices_for_cats(store.cat_ids, cell["train_cat_ids"])
        tail = int(len(cell_train) % batch_size)
        batch_sizes = batch_sizes_for_units(len(cell_train), batch_size)
        ast_batch_audit.append(
            {
                "split_seed": int(cell["split_seed"]),
                "fold": int(cell["fold"]),
                "training_calls": int(len(cell_train)),
                "batch_sizes": batch_sizes,
                "tail_batch_size": tail,
            }
        )
        if tail == 1:
            raise RuntimeError(
                "IDEA-088 AST role cell creates a one-call tail batch; explicit policy required"
            )
    run_root = resolve_run_root(args.output_subdir)
    payload = {
        "status": "GO",
        "stage": "cpu_preflight",
        "outer_test_accessed": False,
        "training_started": False,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tests_sha256": sha256(REPO_ROOT / protocol["preflight"]["tests_path"]),
        "plan_sha256": sha256(REPO_ROOT / protocol["preflight"]["plan_path"]),
        "roles_path": relative_repo_path(roles_path),
        "roles_sha256": sha256(roles_path),
        "role_audit": role_audit,
        "data_audit": ast["audit"],
        "model_audit": {
            "vggish_parameters": vggish_parameters,
            "A0_parameters": EXPECTED_AST_PARAMETERS["A0_ast_only"],
            "C1_parameters": EXPECTED_AST_PARAMETERS["C1_bounded_wide_additive"],
            "A0_C1_common_initial_state_sha256": common["A0_ast_only"],
            "A0_C1_initial_logits_exact_match": True,
            "vggish_framework": "TensorFlow/Keras",
            "ast_framework": "PyTorch FP32",
        },
        "training_audit": {
            "fits_locked": 60,
            "batch_size_prediction_units": 128,
            "no_independent_validation": True,
            "early_stopping_monitor": "training_loss",
            "test_prediction_after_checkpoint_lock_only": True,
            "no_hpo": True,
            "no_gradient_clip": True,
            "no_amp": True,
            "ast_batch_sizes": ast_batch_audit,
            "one_call_tail_batches": 0,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "tensorflow": formal.tf.__version__,
        },
    }
    path = preflight_path(run_root)
    write_json(path, payload)
    return {**payload, "preflight_path": relative_repo_path(path), "preflight_sha256": sha256(path)}


def require_preflight(
    args: argparse.Namespace, protocol: dict[str, Any], roles_path: Path
) -> dict[str, Any]:
    if not args.preflight_sha256:
        raise RuntimeError("IDEA-088 fit requires exact preflight SHA-256")
    run_root = resolve_run_root(args.output_subdir)
    path = preflight_path(run_root)
    if not path.is_file() or sha256(path) != args.preflight_sha256:
        raise RuntimeError("IDEA-088 preflight identity mismatch")
    value = read_json(path)
    expected = {
        "status": "GO",
        "outer_test_accessed": False,
        "training_started": False,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tests_sha256": sha256(REPO_ROOT / protocol["preflight"]["tests_path"]),
        "plan_sha256": sha256(REPO_ROOT / protocol["preflight"]["plan_path"]),
        "roles_sha256": sha256(roles_path),
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"IDEA-088 preflight field changed: {key}")
    return value


def ensure_training_authorized(args: argparse.Namespace) -> None:
    if not args.director_authorized:
        raise RuntimeError("IDEA-088 fit requires explicit director authorization")
    if args.max_fits is not None and args.max_fits < 1:
        raise RuntimeError("IDEA-088 max-fits must be positive")


def validate_completed_fit(
    summary_path: Path,
    protocol: dict[str, Any],
    roles_path: Path,
    pipeline: str,
    split_seed: int,
    fold: int,
) -> dict[str, Any]:
    value = read_json(summary_path)
    expected = _base_fit_identity(protocol, roles_path, pipeline, split_seed, fold)
    if value.get("status") != "complete":
        raise RuntimeError(f"Incomplete IDEA-088 fit: {summary_path}")
    for key, item in expected.items():
        if value.get(key) != item:
            raise RuntimeError(f"IDEA-088 completed fit identity changed: {key}")
    for path_key, hash_key in (
        ("model_weights", "model_weights_sha256"),
        ("preprocessing", "preprocessing_sha256"),
        ("epoch_orders", "epoch_orders_sha256"),
        ("checkpoint_lock", "checkpoint_lock_sha256"),
        ("unit_predictions", "unit_predictions_sha256"),
        ("cat_predictions", "cat_predictions_sha256"),
    ):
        path = REPO_ROOT / value[path_key]
        if not path.is_file() or sha256(path) != value[hash_key]:
            raise RuntimeError(f"IDEA-088 completed fit artifact changed: {path_key}")
    return value


def run_fits(args: argparse.Namespace) -> dict[str, Any]:
    ensure_training_authorized(args)
    configure_determinism()
    protocol, roles_path, roles_manifest = load_protocol_and_roles()
    require_preflight(args, protocol, roles_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("IDEA-088 requested CUDA but CUDA is unavailable")
    device = torch.device(args.device)
    vggish, age_features, ast = load_feature_inputs(protocol)
    store = ast["store"]
    cells, _ = validate_roles_manifest(
        roles_manifest, protocol, set(store.cat_ids.astype(str))
    )
    selected_cells = [
        cell
        for cell in cells
        if (args.split_seed is None or cell["split_seed"] == args.split_seed)
        and (args.fold is None or cell["fold"] == args.fold)
    ]
    selected_pipelines = [args.pipeline] if args.pipeline else list(PIPELINES)
    run_root = resolve_run_root(args.output_subdir)
    completed = 0
    skipped = 0
    for cell in selected_cells:
        for pipeline in selected_pipelines:
            fit_root = fit_directory(
                run_root, pipeline, cell["split_seed"], cell["fold"]
            )
            summary_path = fit_root / "fit_summary.json"
            if summary_path.exists():
                if not args.resume:
                    raise RuntimeError(f"IDEA-088 fit already exists: {summary_path}")
                validate_completed_fit(
                    summary_path,
                    protocol,
                    roles_path,
                    pipeline,
                    cell["split_seed"],
                    cell["fold"],
                )
                skipped += 1
                continue
            if args.max_fits is not None and completed >= args.max_fits:
                break
            fit_root.mkdir(parents=True, exist_ok=True)
            if pipeline == "vggish_mlp":
                result = fit_vggish(protocol, roles_path, vggish, cell, fit_root)
            else:
                result = fit_ast(
                    protocol,
                    roles_path,
                    pipeline,
                    store,
                    age_features,
                    cell,
                    fit_root,
                    device,
                )
            write_json(summary_path, result)
            completed += 1
        if args.max_fits is not None and completed >= args.max_fits:
            break
    status = {
        "status": "fit_stage_returned",
        "completed_new_fits": completed,
        "validated_resume_fits": skipped,
        "filters": {
            "pipeline": args.pipeline,
            "split_seed": args.split_seed,
            "fold": args.fold,
            "max_fits": args.max_fits,
        },
    }
    write_json(run_root / "fit_stage_status.json", status)
    return status


def _sample_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def aggregate_results(args: argparse.Namespace) -> dict[str, Any]:
    protocol, roles_path, roles_manifest = load_protocol_and_roles()
    vggish, age_features, ast = load_feature_inputs(protocol)
    del vggish, age_features
    cells, role_audit = validate_roles_manifest(
        roles_manifest, protocol, set(ast["store"].cat_ids.astype(str))
    )
    run_root = resolve_run_root(args.output_subdir)
    fits: list[dict[str, Any]] = []
    for cell in cells:
        for pipeline in PIPELINES:
            path = (
                fit_directory(run_root, pipeline, cell["split_seed"], cell["fold"])
                / "fit_summary.json"
            )
            if not path.is_file():
                raise RuntimeError(f"IDEA-088 aggregate is missing fit: {path}")
            fits.append(
                validate_completed_fit(
                    path,
                    protocol,
                    roles_path,
                    pipeline,
                    cell["split_seed"],
                    cell["fold"],
                )
            )
    if len(fits) != 60:
        raise RuntimeError("IDEA-088 aggregate expected exactly 60 fits")
    metric_names = (
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "cross_entropy",
        "brier",
    )
    pipeline_summaries: dict[str, Any] = {}
    cell_lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for fit in fits:
        cell_lookup[(fit["pipeline"], fit["split_seed"], fit["fold"])] = fit
    for pipeline in PIPELINES:
        selected = [fit for fit in fits if fit["pipeline"] == pipeline]
        seed_rows = []
        for seed in protocol["splits"]["seeds"]:
            seed_fits = sorted(
                [fit for fit in selected if fit["split_seed"] == seed],
                key=lambda row: row["fold"],
            )
            if len(seed_fits) != 4:
                raise RuntimeError("IDEA-088 seed is missing one or more folds")
            seed_rows.append(
                {
                    "split_seed": seed,
                    "folds": 4,
                    "public_cat_probability_mean": {
                        metric: float(
                            np.mean(
                                [
                                    fit["metrics"]["public_cat_probability_mean"][metric]
                                    for fit in seed_fits
                                ]
                            )
                        )
                        for metric in metric_names
                    },
                    "native_prediction_unit": {
                        metric: float(
                            np.mean(
                                [
                                    fit["metrics"]["native_prediction_unit"][metric]
                                    for fit in seed_fits
                                ]
                            )
                        )
                        for metric in metric_names
                    },
                }
            )
        pipeline_summaries[pipeline] = {
            "input_unit": selected[0]["input_unit"],
            "folds": 20,
            "seed_four_fold_means": seed_rows,
            "public_cat_probability_mean": {
                metric: {
                    "mean_over_five_seed_means": float(
                        np.mean(
                            [row["public_cat_probability_mean"][metric] for row in seed_rows]
                        )
                    ),
                    "sample_sd_over_five_seed_means": _sample_sd(
                        [row["public_cat_probability_mean"][metric] for row in seed_rows]
                    ),
                }
                for metric in metric_names
            },
            "native_prediction_unit": {
                metric: {
                    "mean_over_five_seed_means": float(
                        np.mean([row["native_prediction_unit"][metric] for row in seed_rows])
                    ),
                    "sample_sd_over_five_seed_means": _sample_sd(
                        [row["native_prediction_unit"][metric] for row in seed_rows]
                    ),
                }
                for metric in metric_names
            },
        }
    paired_cells = []
    for seed in protocol["splits"]["seeds"]:
        for fold in range(4):
            a0 = cell_lookup[("A0_ast_only", seed, fold)]
            c1 = cell_lookup[("C1_bounded_wide_additive", seed, fold)]
            paired_cells.append(
                {
                    "split_seed": seed,
                    "fold": fold,
                    "C1_minus_A0_public_cat_macro_f1": float(
                        c1["metrics"]["public_cat_probability_mean"]["macro_f1"]
                        - a0["metrics"]["public_cat_probability_mean"]["macro_f1"]
                    ),
                    "C1_minus_A0_native_call_macro_f1": float(
                        c1["metrics"]["native_prediction_unit"]["macro_f1"]
                        - a0["metrics"]["native_prediction_unit"]["macro_f1"]
                    ),
                }
            )
    paired_seed_rows = []
    for seed in protocol["splits"]["seeds"]:
        rows = [row for row in paired_cells if row["split_seed"] == seed]
        paired_seed_rows.append(
            {
                "split_seed": seed,
                "C1_minus_A0_public_cat_macro_f1": float(
                    np.mean([row["C1_minus_A0_public_cat_macro_f1"] for row in rows])
                ),
                "C1_minus_A0_native_call_macro_f1": float(
                    np.mean([row["C1_minus_A0_native_call_macro_f1"] for row in rows])
                ),
            }
        )
    public_deltas = [row["C1_minus_A0_public_cat_macro_f1"] for row in paired_seed_rows]
    native_deltas = [row["C1_minus_A0_native_call_macro_f1"] for row in paired_seed_rows]
    payload = {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "status": "complete",
        "claim_boundary": "Clean-111 projection of recovered original post-swap groups plus the original final training recipe; not an exact upstream reproduction and not independent data.",
        "claims_complete_oof": False,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "roles_sha256": sha256(roles_path),
        "fits": 60,
        "role_audit": role_audit,
        "pipelines": pipeline_summaries,
        "paired_C1_minus_A0": {
            "cell_deltas": paired_cells,
            "seed_four_fold_mean_deltas": paired_seed_rows,
            "public_cat_macro_f1": {
                "mean": float(np.mean(public_deltas)),
                "sample_sd": _sample_sd(public_deltas),
                "positive_seeds": int(np.sum(np.asarray(public_deltas) > 0.0)),
                "tied_seeds": int(np.sum(np.asarray(public_deltas) == 0.0)),
                "negative_seeds": int(np.sum(np.asarray(public_deltas) < 0.0)),
            },
            "native_call_macro_f1": {
                "mean": float(np.mean(native_deltas)),
                "sample_sd": _sample_sd(native_deltas),
                "positive_seeds": int(np.sum(np.asarray(native_deltas) > 0.0)),
                "tied_seeds": int(np.sum(np.asarray(native_deltas) == 0.0)),
                "negative_seeds": int(np.sum(np.asarray(native_deltas) < 0.0)),
            },
        },
        "historical_context_only": {
            "old_vggish_reproduction_macro_f1": 0.700136,
            "directly_comparable_to_current_clean_projection": False,
        },
    }
    summary_path = run_root / "idea088_aggregate_results.json"
    metadata_path = (
        REPO_ROOT
        / "metadata"
        / "experiments"
        / "meowagenet_idea088_original_style_clean_v1_results.json"
    )
    write_json(summary_path, payload)
    write_json(metadata_path, {**payload, "run_summary": relative_repo_path(summary_path), "run_summary_sha256": sha256(summary_path)})
    return {**payload, "run_summary": relative_repo_path(summary_path), "metadata": relative_repo_path(metadata_path)}


def main() -> None:
    args = parse_args()
    if args.stage == "preflight":
        result = run_preflight(args)
    elif args.stage == "fit":
        result = run_fits(args)
    else:
        result = aggregate_results(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
