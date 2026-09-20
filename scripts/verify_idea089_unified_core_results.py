"""Independent verifier for IDEA-089 preflight and selection artifacts.

The verifier does not import the IDEA-089 runner.  It independently rebuilds
role identities, training-side preprocessing, deterministic unit order,
validation cat aggregation, metrics, early stopping, and artifact hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "configs" / "protocol" / "meowagenet_idea089_unified_core_v1.json"
)
RUN_ROOT = ROOT / "runs" / "meowagenet_idea089_unified_core_v1"
PIPELINES = (
    "VGG128_no_f0",
    "VGG129_with_f0",
    "A0_ast_only",
    "D0_direct_concat",
    "U1_wide_unbounded_additive",
    "C1_bounded_wide_additive",
)
VGG_PIPELINES = PIPELINES[:2]
AST_PIPELINES = PIPELINES[2:]
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
EXPECTED_TRAINABLE = {
    "VGG128_no_f0": 17155,
    "VGG129_with_f0": 17283,
    "A0_ast_only": 99075,
    "D0_direct_concat": 101635,
    "U1_wide_unbounded_additive": 108143,
    "C1_bounded_wide_additive": 108143,
}
EXPECTED_TOTAL = {
    **EXPECTED_TRAINABLE,
    "VGG128_no_f0": 17411,
    "VGG129_with_f0": 17539,
}
COMPARISONS = {
    "C1_minus_A0": ("C1_bounded_wide_additive", "A0_ast_only"),
    "C1_minus_U1": (
        "C1_bounded_wide_additive",
        "U1_wide_unbounded_additive",
    ),
    "C1_minus_D0": ("C1_bounded_wide_additive", "D0_direct_concat"),
    "U1_minus_A0": ("U1_wide_unbounded_additive", "A0_ast_only"),
    "D0_minus_A0": ("D0_direct_concat", "A0_ast_only"),
    "U1_minus_D0": ("U1_wide_unbounded_additive", "D0_direct_concat"),
    "VGG129_minus_VGG128": ("VGG129_with_f0", "VGG128_no_f0"),
    "A0_minus_VGG129": ("A0_ast_only", "VGG129_with_f0"),
    "C1_minus_VGG129": (
        "C1_bounded_wide_additive",
        "VGG129_with_f0",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def close(left: float, right: float, name: str, tolerance: float = 1.0e-12) -> None:
    if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{name}: {left} != {right}")


def finite_tree(value: Any, name: str = "root") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise AssertionError(f"Non-finite value at {name}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, f"{name}[{index}]")


def metric_bundle(animals: pd.DataFrame) -> dict[str, Any]:
    ordered = animals.sort_values("cat_id").reset_index(drop=True)
    labels = ordered["true_label"].to_numpy(dtype=np.int64)
    probabilities = ordered[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
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
            "kitten": float(recalls[0]),
            "adult": float(recalls[1]),
            "senior": float(recalls[2]),
        },
        "cross_entropy": float(
            -np.mean(np.log(clipped[np.arange(len(labels)), labels]))
        ),
        "brier": float(np.mean(np.sum((probabilities - targets) ** 2, axis=1))),
    }


def compare_metrics(observed: dict[str, Any], expected: dict[str, Any], name: str) -> None:
    if observed.keys() != expected.keys():
        raise AssertionError(f"{name}: metric keys differ")
    for key in observed:
        if isinstance(observed[key], dict):
            compare_metrics(observed[key], expected[key], f"{name}.{key}")
        elif key == "n":
            if int(observed[key]) != int(expected[key]):
                raise AssertionError(f"{name}.n differs")
        else:
            close(observed[key], expected[key], f"{name}.{key}")


def flatten_metrics(bundle: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy": float(bundle["accuracy"]),
        "macro_f1": float(bundle["macro_f1"]),
        "balanced_accuracy": float(bundle["balanced_accuracy"]),
        "recall_kitten": float(bundle["class_recall"]["kitten"]),
        "recall_adult": float(bundle["class_recall"]["adult"]),
        "recall_senior": float(bundle["class_recall"]["senior"]),
        "cross_entropy": float(bundle["cross_entropy"]),
        "brier": float(bundle["brier"]),
    }


def mean_and_sample_sd(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) < 2:
        raise AssertionError("Summary requires at least two values")
    return {"mean": float(array.mean()), "sample_sd": float(array.std(ddof=1))}


def stratified_bootstrap_indices(
    labels: np.ndarray, replicates: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    pieces = []
    for label in range(3):
        positions = np.flatnonzero(labels == label)
        if len(positions) == 0:
            raise AssertionError("Bootstrap label stratum is empty")
        pieces.append(
            rng.choice(positions, size=(int(replicates), len(positions)), replace=True)
        )
    return np.concatenate(pieces, axis=1).astype(np.int64)


def bootstrap_model_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_indices: np.ndarray,
    chunk_size: int = 250,
) -> dict[str, np.ndarray]:
    if probabilities.shape[1:] != (len(labels), 3):
        raise AssertionError("Bootstrap probability shape changed")
    names = (
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "recall_kitten",
        "recall_adult",
        "recall_senior",
        "cross_entropy",
        "brier",
    )
    result = {
        name: np.empty(len(sample_indices), dtype=np.float64) for name in names
    }
    targets = np.eye(3, dtype=np.float64)
    for start in range(0, len(sample_indices), chunk_size):
        stop = min(start + chunk_size, len(sample_indices))
        indices = sample_indices[start:stop]
        sampled_labels = labels[indices]
        accumulated = {
            name: np.zeros(stop - start, dtype=np.float64) for name in names
        }
        for oof_probabilities in probabilities:
            sampled_probabilities = oof_probabilities[indices]
            predictions = sampled_probabilities.argmax(axis=2)
            recalls = []
            f1_values = []
            for label in range(3):
                true = sampled_labels == label
                predicted = predictions == label
                tp = np.sum(true & predicted, axis=1).astype(np.float64)
                fn = np.sum(true & ~predicted, axis=1).astype(np.float64)
                fp = np.sum(~true & predicted, axis=1).astype(np.float64)
                recalls.append(tp / np.maximum(tp + fn, 1.0))
                denominator = 2.0 * tp + fp + fn
                f1_values.append(
                    np.divide(
                        2.0 * tp,
                        denominator,
                        out=np.zeros_like(tp),
                        where=denominator > 0.0,
                    )
                )
            true_probabilities = np.take_along_axis(
                sampled_probabilities, sampled_labels[..., None], axis=2
            )[..., 0]
            sampled_targets = targets[sampled_labels]
            accumulated["accuracy"] += np.mean(
                predictions == sampled_labels, axis=1
            )
            accumulated["macro_f1"] += np.mean(
                np.stack(f1_values, axis=1), axis=1
            )
            accumulated["balanced_accuracy"] += np.mean(
                np.stack(recalls, axis=1), axis=1
            )
            accumulated["recall_kitten"] += recalls[0]
            accumulated["recall_adult"] += recalls[1]
            accumulated["recall_senior"] += recalls[2]
            accumulated["cross_entropy"] += -np.mean(
                np.log(np.clip(true_probabilities, 1.0e-12, 1.0)), axis=1
            )
            accumulated["brier"] += np.mean(
                np.sum((sampled_probabilities - sampled_targets) ** 2, axis=2),
                axis=1,
            )
        for name in names:
            result[name][start:stop] = accumulated[name] / float(len(probabilities))
    return result


def units_to_animals(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat_id, group in units.groupby("cat_id", sort=True):
        labels = group["true_label"].to_numpy(dtype=np.int64)
        if len(np.unique(labels)) != 1:
            raise AssertionError(f"Non-unique unit labels for {cat_id}")
        probabilities = group[list(PROBABILITY_COLUMNS)].to_numpy(float).mean(axis=0)
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


def verify_prediction_pair(unit_path: Path, cat_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    units = pd.read_csv(unit_path, dtype={"unit_id": str, "cat_id": str})
    animals = pd.read_csv(cat_path, dtype={"cat_id": str})
    for frame, name in ((units, "units"), (animals, "animals")):
        probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
        if not np.isfinite(probabilities).all():
            raise AssertionError(f"Non-finite {name} probabilities")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise AssertionError(f"Out-of-range {name} probabilities")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0):
            raise AssertionError(f"Unnormalized {name} probabilities")
    rebuilt = units_to_animals(units)
    ordered = animals.sort_values("cat_id").reset_index(drop=True)
    for column in ("cat_id", "true_label", "unit_count", "predicted_label"):
        if not np.array_equal(rebuilt[column].to_numpy(), ordered[column].to_numpy()):
            raise AssertionError(f"Unit-to-cat mismatch for {column}")
    if not np.allclose(
        rebuilt[list(PROBABILITY_COLUMNS)].to_numpy(float),
        ordered[list(PROBABILITY_COLUMNS)].to_numpy(float),
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise AssertionError("Unit-to-cat probability mismatch")
    return units, ordered


def standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    return mean, np.where(scale > 1.0e-12, scale, 1.0).astype(np.float32)


def acoustic_standardizer(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        median = np.nanmedian(values, axis=0)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    imputed = np.where(np.isfinite(values), values, median[None, :])
    mean = imputed.mean(axis=0).astype(np.float32)
    scale = imputed.std(axis=0).astype(np.float32)
    scale = np.where(scale > 1.0e-8, scale, 1.0).astype(np.float32)
    return median, mean, scale


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int64), minlength=3).astype(np.float64)
    return (len(labels) / (3.0 * counts)).astype(np.float32)


def source_data(protocol: dict[str, Any]) -> dict[str, Any]:
    roles = pd.read_csv(ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    roles = roles[roles["repeat"].isin(protocol["data"]["repeats"])].copy()
    ast_loaded = np.load(ROOT / protocol["data"]["frozen_embedding_path"], allow_pickle=False)
    acoustic_loaded = np.load(ROOT / protocol["data"]["feature_path"], allow_pickle=False)
    vgg = pd.read_csv(ROOT / protocol["data"]["vggish_path"], dtype={"cat_id": str})
    analysis_cats = set(ast_loaded["cat_ids"].astype(str))
    vgg = vgg[vgg["cat_id"].isin(analysis_cats)].reset_index(drop=True)
    vgg_target = vgg["target"].to_numpy(dtype=float)
    return {
        "roles": roles,
        "ast_features": ast_loaded["embeddings"].astype(np.float32),
        "ast_call_ids": ast_loaded["call_ids"].astype(str),
        "ast_cat_ids": ast_loaded["cat_ids"].astype(str),
        "ast_labels": ast_loaded["labels"].astype(np.int64),
        "acoustic": acoustic_loaded["features"].astype(np.float32),
        "vgg128": vgg[[str(index) for index in range(128)]].to_numpy(np.float32),
        "vgg129": vgg[[str(index) for index in range(128)] + ["mean_freq"]].to_numpy(np.float32),
        "vgg_cat_ids": vgg["cat_id"].to_numpy(dtype=str),
        "vgg_labels": np.where(vgg_target < 0.5, 0, np.where(vgg_target < 10, 1, 2)).astype(np.int64),
    }


def indices_for(
    data: dict[str, Any], pipeline: str, repeat: int, fold: int
) -> dict[str, np.ndarray]:
    cell = data["roles"][(data["roles"]["repeat"] == repeat) & (data["roles"]["outer_fold"] == fold)]
    mapping = dict(zip(cell["cat_id"], cell["role"], strict=True))
    cat_ids = data["vgg_cat_ids"] if pipeline in VGG_PIPELINES else data["ast_cat_ids"]
    unit_roles = np.asarray([mapping[cat_id] for cat_id in cat_ids])
    return {
        role: np.flatnonzero(unit_roles == role).astype(np.int64)
        for role in ("train", "validation", "test")
    }


def training_identity(cat_ids: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    cats = sorted(set(cat_ids[indices].astype(str)))
    selected_labels = labels[indices].astype(np.int64)
    return {
        "cats": len(cats),
        "units": int(len(indices)),
        "cat_ids_sha256": text_sha256("\n".join(cats)),
        "unit_indices_sha256": hashlib.sha256(
            np.sort(indices.astype("<i8")).tobytes()
        ).hexdigest(),
        "unit_class_counts": np.bincount(selected_labels, minlength=3).astype(int).tolist(),
        "unit_class_weights": class_weights(selected_labels).astype(float).tolist(),
    }


def verify_training_identity(observed: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in ("cats", "units", "cat_ids_sha256", "unit_indices_sha256", "unit_class_counts"):
        if observed[key] != expected[key]:
            raise AssertionError(f"Training identity mismatch: {key}")
    if not np.allclose(
        observed["unit_class_weights"],
        expected["unit_class_weights"],
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise AssertionError("Training class weights differ")


def expected_preprocessing(
    data: dict[str, Any], pipeline: str, indices: np.ndarray
) -> dict[str, np.ndarray]:
    if pipeline in VGG_PIPELINES:
        features = data["vgg128"] if pipeline == VGG_PIPELINES[0] else data["vgg129"]
        mean, scale = standardizer(features[indices])
        return {"mean": mean, "scale": scale}
    ast_mean, ast_scale = standardizer(data["ast_features"][indices])
    result = {"ast_mean": ast_mean, "ast_scale": ast_scale}
    if pipeline != "A0_ast_only":
        median, mean, scale = acoustic_standardizer(data["acoustic"][indices])
        result.update({"age_median": median, "age_mean": mean, "age_scale": scale})
    return result


def verify_preprocessing(path: Path, expected: dict[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(expected):
            raise AssertionError(f"Preprocessing keys differ: {path}")
        for name, values in expected.items():
            if not np.array_equal(loaded[name], values):
                difference = float(np.max(np.abs(loaded[name] - values)))
                raise AssertionError(f"Preprocessing differs for {name}: {difference}")


def saved_model_state_digest(path: Path, pipeline: str) -> str:
    digest = hashlib.sha256()
    if pipeline in VGG_PIPELINES:
        with np.load(path, allow_pickle=False) as loaded:
            for key in sorted(loaded.files):
                array = np.ascontiguousarray(loaded[key])
                digest.update(key.encode("ascii"))
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
                digest.update(array.tobytes())
        return digest.hexdigest()
    state = torch.load(path, map_location="cpu", weights_only=False)
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def verify_history_order(
    history: list[dict[str, Any]],
    pipeline: str,
    train_indices: np.ndarray,
    cat_ids: np.ndarray,
    full_seed: int,
) -> None:
    if pipeline in VGG_PIPELINES:
        training_seed = full_seed + 1_000_000
        for epoch, row in enumerate(history, start=1):
            order = np.random.default_rng(training_seed + epoch).permutation(len(train_indices))
            expected = hashlib.sha256(train_indices[order].astype("<i8").tobytes()).hexdigest()
            if row["unit_order_sha256"] != expected:
                raise AssertionError(f"VGG unit order differs at epoch {epoch}")
        return
    cats = np.asarray(sorted(set(cat_ids[train_indices].astype(str))))
    generator = torch.Generator().manual_seed(int(full_seed))
    coverage = hashlib.sha256(np.sort(train_indices.astype("<i8")).tobytes()).hexdigest()
    for epoch, row in enumerate(history, start=1):
        order = torch.randperm(len(cats), generator=generator).numpy()
        expected_order = text_sha256("\n".join(cats[order].tolist()))
        audit = row["train_audit"]
        if audit["cat_order_sha256"] != expected_order:
            raise AssertionError(f"AST cat order differs at epoch {epoch}")
        if audit["call_coverage_sha256"] != coverage:
            raise AssertionError(f"AST call coverage differs at epoch {epoch}")
        if audit["cats"] != len(cats) or audit["calls"] != len(train_indices):
            raise AssertionError(f"AST epoch coverage count differs at epoch {epoch}")


def best_epoch_from_history(
    history: list[dict[str, Any]], minimum_delta: float
) -> tuple[int, float, int]:
    best = float("inf")
    best_epoch = 0
    stale = 0
    for row in history:
        value = float(row["validation_animal_metrics"]["cross_entropy"])
        if value < best - minimum_delta:
            best = value
            best_epoch = int(row["epoch"])
            stale = 0
        else:
            stale += 1
    return best_epoch, best, stale


def verify_protocol_dependencies(protocol: dict[str, Any]) -> dict[str, str]:
    dependencies = protocol["dependencies"]
    pairs = {
        dependencies["plan_path"]: dependencies["plan_sha256"],
        dependencies["runner_path"]: dependencies["runner_sha256"],
        dependencies["tests_path"]: dependencies["tests_sha256"],
        dependencies["idea068_runner_path"]: dependencies["idea068_runner_sha256"],
        dependencies["idea071_runner_path"]: dependencies["idea071_runner_sha256"],
        dependencies["idea076_runner_path"]: dependencies["idea076_runner_sha256"],
        dependencies["independent_design_audit_path"]: dependencies[
            "independent_design_audit_sha256"
        ],
        protocol["data"]["roles_path"]: protocol["data"]["roles_sha256"],
        protocol["data"]["dataset_manifest_path"]: protocol["data"][
            "dataset_manifest_sha256"
        ],
        protocol["data"]["vggish_path"]: protocol["data"]["vggish_sha256"],
        protocol["data"]["frozen_embedding_path"]: protocol["data"][
            "frozen_embedding_sha256"
        ],
        protocol["data"]["feature_path"]: protocol["data"]["feature_sha256"],
    }
    observed = {path: sha256(ROOT / path) for path in pairs}
    if observed != pairs:
        raise AssertionError("Protocol dependency hash mismatch")
    return observed


def verify_preflight(protocol: dict[str, Any]) -> dict[str, Any]:
    path = RUN_ROOT / protocol["outputs"]["cpu_preflight"]
    value = read_json(path)
    if value["status"] != "GO" or value["training_started"] is not False:
        raise AssertionError("CPU preflight is not no-training GO")
    if value["outer_test_accessed"] is not False:
        raise AssertionError("CPU preflight accessed outer test")
    if value["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise AssertionError("Preflight protocol hash mismatch")
    if value["runner_sha256"] != protocol["dependencies"]["runner_sha256"]:
        raise AssertionError("Preflight runner hash mismatch")
    if value["tests_sha256"] != protocol["dependencies"]["tests_sha256"]:
        raise AssertionError("Preflight tests hash mismatch")
    if value["trainable_parameters"] != EXPECTED_TRAINABLE:
        raise AssertionError("Preflight trainable parameter count mismatch")
    if value["total_state_parameters"] != EXPECTED_TOTAL:
        raise AssertionError("Preflight total parameter count mismatch")
    if value["selection_fits"] != 216 or value["outer_fits"] != 216 or value["physical_fits"] != 432:
        raise AssertionError("Preflight fit budget mismatch")
    if len(value["roles"]) != 12:
        raise AssertionError("Preflight role audit count mismatch")
    if not value["A0_U1_C1_common_head_state_exact"] or not value["U1_C1_complete_initial_state_exact"]:
        raise AssertionError("AST paired state preflight failed")
    if not value["D0_common_weights_exact"] or not value["D0_acoustic_columns_zero"]:
        raise AssertionError("D0 paired state preflight failed")
    if value["VGG129_F0_column_zero"] is not True:
        raise AssertionError("VGG F0 column is not zero initialized")
    gradients = [
        value["D0_acoustic_gradient_max_abs"],
        value["VGG129_F0_gradient_max_abs"],
        *value["U1_C1_zero_projection_gradient_max_abs"].values(),
    ]
    if not all(math.isfinite(float(item)) and float(item) > 0.0 for item in gradients):
        raise AssertionError("An added acoustic parameter has no finite positive gradient")
    if max(value["ast_initial_logit_max_abs_differences_vs_A0"].values()) > value[
        "ast_initial_logit_tolerance"
    ]:
        raise AssertionError("AST paired initial logits exceed tolerance")
    if value["VGG129_minus_VGG128_initial_logit_max_abs_difference"] > 1.0e-7:
        raise AssertionError("VGG paired initial logits exceed tolerance")
    finite_tree(value)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), **value}


def fit_directory(pipeline: str, repeat: int, fold: int, base_seed: int) -> Path:
    return (
        RUN_ROOT
        / "selection"
        / pipeline
        / f"repeat_{repeat}"
        / f"fold_{fold}"
        / f"seed_{base_seed}"
    )


def verify_selection_fit(
    protocol: dict[str, Any],
    data: dict[str, Any],
    pipeline: str,
    repeat: int,
    fold: int,
    base_seed: int,
) -> dict[str, Any]:
    directory = fit_directory(pipeline, repeat, fold, base_seed)
    summary_path = directory / "fit_summary.json"
    summary = read_json(summary_path)
    full_seed = base_seed + 10_000 * repeat + 100 * fold
    expected_identity = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": protocol["dependencies"]["runner_sha256"],
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "stage": "selection",
        "pipeline": pipeline,
        "repeat": repeat,
        "fold": fold,
        "base_seed": base_seed,
        "full_seed": full_seed,
    }
    if summary["status"] != "complete" or any(
        summary.get(key) != value for key, value in expected_identity.items()
    ):
        raise AssertionError(f"Selection fit identity mismatch: {pipeline}")
    if summary["outer_test_accessed"] is not False or summary["test_prediction_calls"] != 0:
        raise AssertionError(f"Selection touched outer test: {pipeline}")
    if summary["selection_metric"] != "unweighted_validation_cat_cross_entropy":
        raise AssertionError("Selection metric changed")
    if float(summary["minimum_delta"]) != 1.0e-6:
        raise AssertionError("Selection minimum delta changed")

    artifact_fields = (
        ("model_weights", "model_weights_sha256"),
        ("preprocessing", "preprocessing_sha256"),
        ("training_history", "training_history_sha256"),
        ("validation_unit_predictions", "validation_unit_predictions_sha256"),
        ("validation_cat_predictions", "validation_cat_predictions_sha256"),
    )
    for path_key, hash_key in artifact_fields:
        path = ROOT / summary[path_key]
        if not path.is_file() or sha256(path) != summary[hash_key]:
            raise AssertionError(f"Selection artifact mismatch: {pipeline}/{path_key}")

    indices = indices_for(data, pipeline, repeat, fold)
    if pipeline in VGG_PIPELINES:
        cat_ids, labels = data["vgg_cat_ids"], data["vgg_labels"]
    else:
        cat_ids, labels = data["ast_cat_ids"], data["ast_labels"]
    expected_identity = training_identity(cat_ids, labels, indices["train"])
    verify_training_identity(summary["audit"]["train_role"], expected_identity)
    verify_preprocessing(
        ROOT / summary["preprocessing"],
        expected_preprocessing(data, pipeline, indices["train"]),
    )

    history_wrapper = read_json(ROOT / summary["training_history"])
    history = history_wrapper["history"]
    if len(history) != int(summary["stopped_epoch"]):
        raise AssertionError(f"Stopped epoch/history mismatch: {pipeline}")
    if [int(row["epoch"]) for row in history] != list(range(1, len(history) + 1)):
        raise AssertionError(f"Non-contiguous history: {pipeline}")
    verify_history_order(history, pipeline, indices["train"], cat_ids, full_seed)
    best_epoch, best_ce, stale = best_epoch_from_history(history, 1.0e-6)
    if best_epoch != int(summary["best_epoch"]):
        raise AssertionError(f"Best epoch mismatch: {pipeline}")
    config = protocol["training"]["vgg" if pipeline in VGG_PIPELINES else "ast"]
    if len(history) > int(config["maximum_epochs"]):
        raise AssertionError(f"Maximum epoch exceeded: {pipeline}")
    if len(history) < int(config["maximum_epochs"]) and stale < int(
        config["early_stopping_patience"]
    ):
        raise AssertionError(f"Premature early stop: {pipeline}")

    units, animals = verify_prediction_pair(
        ROOT / summary["validation_unit_predictions"],
        ROOT / summary["validation_cat_predictions"],
    )
    expected_validation_cats = set(cat_ids[indices["validation"]].astype(str))
    if set(animals["cat_id"].astype(str)) != expected_validation_cats:
        raise AssertionError(f"Validation cat identities differ: {pipeline}")
    recomputed = metric_bundle(animals)
    compare_metrics(recomputed, summary["validation_metrics"], f"{pipeline}.summary")
    compare_metrics(
        recomputed,
        summary["audit"]["best_validation_animal_metrics"],
        f"{pipeline}.audit",
    )
    close(recomputed["cross_entropy"], best_ce, f"{pipeline}.best_ce")
    if not summary["checkpoint_reload_state_match"]:
        raise AssertionError(f"Checkpoint state reload failed: {pipeline}")
    if float(summary["checkpoint_reload_max_probability_difference"]) > 1.0e-6:
        raise AssertionError(f"Checkpoint prediction reload differs: {pipeline}")
    if int(summary["audit"]["trainable_parameters"]) != EXPECTED_TRAINABLE[pipeline]:
        raise AssertionError(f"Trainable parameter mismatch: {pipeline}")
    if int(summary["audit"]["total_state_parameters"]) != EXPECTED_TOTAL[pipeline]:
        raise AssertionError(f"Total parameter mismatch: {pipeline}")
    if int(summary["audit"]["seed"]) != full_seed:
        raise AssertionError(f"Fit seed mismatch: {pipeline}")
    if int(summary["audit"]["training_seed"]) != full_seed + 1_000_000:
        raise AssertionError(f"Post-build seed mismatch: {pipeline}")
    finite_tree(summary)
    return {
        "pipeline": pipeline,
        "summary_path": summary_path.relative_to(ROOT).as_posix(),
        "summary_sha256": sha256(summary_path),
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "validation_cats": int(len(animals)),
        "validation_units": int(len(units)),
        "checkpoint_reload_max_probability_difference": float(
            summary["checkpoint_reload_max_probability_difference"]
        ),
    }


def verify_first_six(protocol: dict[str, Any]) -> dict[str, Any]:
    data = source_data(protocol)
    fits = [verify_selection_fit(protocol, data, pipeline, 0, 0, 17) for pipeline in PIPELINES]
    authorization_path = RUN_ROOT / "authorizations" / "initial-six-selection.json"
    if not authorization_path.is_file():
        raise AssertionError("Initial-six authorization record is missing")
    authorization = read_json(authorization_path)
    if authorization["authorization_scope"] != "initial-six-selection":
        raise AssertionError("Initial-six authorization scope changed")
    if authorization["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise AssertionError("Initial-six authorization protocol mismatch")
    return {
        "status": "PASS_FIRST_SIX_SELECTION_ENGINEERING_AUDIT",
        "fits": fits,
        "authorization_path": authorization_path.relative_to(ROOT).as_posix(),
        "authorization_sha256": sha256(authorization_path),
        "outer_test_accessed": False,
        "decision_ignores_validation_scores": True,
    }


def all_selection_coordinates(protocol: dict[str, Any]) -> list[tuple[str, int, int, int]]:
    return [
        (pipeline, int(repeat), int(fold), int(base_seed))
        for repeat in protocol["data"]["repeats"]
        for fold in protocol["data"]["folds"]
        for base_seed in protocol["training"]["model_seeds"]
        for pipeline in PIPELINES
    ]


def verify_selection_lock(protocol: dict[str, Any]) -> dict[str, Any]:
    data = source_data(protocol)
    coordinates = all_selection_coordinates(protocol)
    fits = [
        verify_selection_fit(protocol, data, pipeline, repeat, fold, base_seed)
        for pipeline, repeat, fold, base_seed in coordinates
    ]
    if len(fits) != 216:
        raise AssertionError("Full selection audit did not find 216 fits")

    lock_path = RUN_ROOT / protocol["outputs"]["selection_lock"]
    lock = read_json(lock_path)
    if lock["status"] != "complete_locked_before_outer_evaluation":
        raise AssertionError("Selection lock status is incomplete")
    if lock["selection_fits"] != 216 or len(lock["entries"]) != 216:
        raise AssertionError("Selection lock fit count changed")
    if lock["outer_test_accessed"] is not False:
        raise AssertionError("Selection lock reports outer-test access")
    if lock["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise AssertionError("Selection lock protocol hash mismatch")
    if lock["runner_sha256"] != protocol["dependencies"]["runner_sha256"]:
        raise AssertionError("Selection lock runner hash mismatch")
    if lock["tests_sha256"] != protocol["dependencies"]["tests_sha256"]:
        raise AssertionError("Selection lock tests hash mismatch")

    indexed = {
        (
            str(row["pipeline"]),
            int(row["repeat"]),
            int(row["fold"]),
            int(row["base_seed"]),
        ): row
        for row in lock["entries"]
    }
    if set(indexed) != set(coordinates) or len(indexed) != 216:
        raise AssertionError("Selection lock coordinate bank changed")
    best_epochs: dict[str, list[int]] = {pipeline: [] for pipeline in PIPELINES}
    for pipeline, repeat, fold, base_seed in coordinates:
        row = indexed[(pipeline, repeat, fold, base_seed)]
        summary_path = fit_directory(pipeline, repeat, fold, base_seed) / "fit_summary.json"
        summary = read_json(summary_path)
        if row["selection_summary"] != summary_path.relative_to(ROOT).as_posix():
            raise AssertionError("Selection lock summary path mismatch")
        if row["selection_summary_sha256"] != sha256(summary_path):
            raise AssertionError("Selection lock summary hash mismatch")
        if int(row["best_epoch"]) != int(summary["best_epoch"]):
            raise AssertionError("Selection lock best epoch mismatch")
        if int(row["full_seed"]) != base_seed + 10_000 * repeat + 100 * fold:
            raise AssertionError("Selection lock full seed mismatch")
        for lock_key, summary_key in (
            ("selection_model_weights_sha256", "model_weights_sha256"),
            ("selection_preprocessing_sha256", "preprocessing_sha256"),
            ("selection_unit_predictions_sha256", "validation_unit_predictions_sha256"),
            ("selection_cat_predictions_sha256", "validation_cat_predictions_sha256"),
        ):
            if row[lock_key] != summary[summary_key]:
                raise AssertionError(f"Selection lock artifact hash differs: {lock_key}")
        best_epochs[pipeline].append(int(row["best_epoch"]))

    full_authorization_path = RUN_ROOT / "authorizations" / "full-selection.json"
    if not full_authorization_path.is_file():
        raise AssertionError("Full-selection authorization record is missing")
    authorization = read_json(full_authorization_path)
    if authorization["stage"] != "selection":
        raise AssertionError("Full-selection authorization stage changed")
    if authorization["authorization_scope"] != "full-selection":
        raise AssertionError("Full-selection authorization scope changed")
    if authorization["director_authorized"] is not True:
        raise AssertionError("Full-selection is not director-authorized")
    if authorization["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise AssertionError("Full-selection authorization protocol mismatch")
    if authorization["runner_sha256"] != protocol["dependencies"]["runner_sha256"]:
        raise AssertionError("Full-selection authorization runner mismatch")
    preflight_path = RUN_ROOT / protocol["outputs"]["cpu_preflight"]
    if authorization["preflight_sha256"] != sha256(preflight_path):
        raise AssertionError("Full-selection authorization preflight mismatch")
    if authorization["selection_lock_sha256"] is not None:
        raise AssertionError("Full-selection authorization unexpectedly binds a lock")
    if any(value is not None for value in authorization["filters"].values()):
        raise AssertionError("Full-selection authorization contains fit filters")
    return {
        "status": "PASS_FULL_SELECTION_AND_EPOCH_LOCK_AUDIT",
        "selection_fits": 216,
        "outer_test_accessed": False,
        "fit_summaries": fits,
        "selection_lock_path": lock_path.relative_to(ROOT).as_posix(),
        "selection_lock_sha256": sha256(lock_path),
        "full_selection_authorization_path": full_authorization_path.relative_to(
            ROOT
        ).as_posix(),
        "full_selection_authorization_sha256": sha256(full_authorization_path),
        "best_epoch_ranges": {
            pipeline: {"minimum": min(values), "maximum": max(values)}
            for pipeline, values in best_epochs.items()
        },
        "decision_ignores_validation_scores": True,
    }


def outer_fit_directory(pipeline: str, repeat: int, fold: int, base_seed: int) -> Path:
    return (
        RUN_ROOT
        / "outer"
        / pipeline
        / f"repeat_{repeat}"
        / f"fold_{fold}"
        / f"seed_{base_seed}"
    )


def verify_outer_fit(
    protocol: dict[str, Any],
    data: dict[str, Any],
    lock_sha256: str,
    lock_entry: dict[str, Any],
    pipeline: str,
    repeat: int,
    fold: int,
    base_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    directory = outer_fit_directory(pipeline, repeat, fold, base_seed)
    summary_path = directory / "fit_summary.json"
    summary = read_json(summary_path)
    full_seed = base_seed + 10_000 * repeat + 100 * fold
    expected_identity = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": protocol["dependencies"]["runner_sha256"],
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "stage": "outer",
        "pipeline": pipeline,
        "repeat": repeat,
        "fold": fold,
        "base_seed": base_seed,
        "full_seed": full_seed,
    }
    if summary["status"] != "complete" or any(
        summary.get(key) != value for key, value in expected_identity.items()
    ):
        raise AssertionError(f"Outer fit identity mismatch: {pipeline}/{repeat}/{fold}/{base_seed}")
    if summary["outer_test_accessed"] is not True or summary["test_prediction_calls"] != 1:
        raise AssertionError("Outer test access count changed")
    if summary["selection_lock_sha256"] != lock_sha256:
        raise AssertionError("Outer fit selection-lock hash mismatch")
    selection_path = ROOT / summary["selection_summary"]
    if sha256(selection_path) != summary["selection_summary_sha256"]:
        raise AssertionError("Outer fit selection-summary hash mismatch")
    if summary["selection_summary_sha256"] != lock_entry["selection_summary_sha256"]:
        raise AssertionError("Outer fit does not use its locked selection summary")
    if int(summary["refit_epoch"]) != int(lock_entry["best_epoch"]):
        raise AssertionError("Outer refit epoch differs from selection lock")

    artifact_fields = (
        ("model_weights", "model_weights_sha256"),
        ("preprocessing", "preprocessing_sha256"),
        ("training_history", "training_history_sha256"),
        ("checkpoint_lock", "checkpoint_lock_sha256"),
        ("test_unit_predictions", "test_unit_predictions_sha256"),
        ("test_cat_predictions", "test_cat_predictions_sha256"),
    )
    for path_key, hash_key in artifact_fields:
        path = ROOT / summary[path_key]
        if not path.is_file() or sha256(path) != summary[hash_key]:
            raise AssertionError(f"Outer artifact mismatch: {pipeline}/{path_key}")

    indices = indices_for(data, pipeline, repeat, fold)
    outer_train = np.sort(
        np.concatenate((indices["train"], indices["validation"]))
    ).astype(np.int64)
    if pipeline in VGG_PIPELINES:
        cat_ids, labels = data["vgg_cat_ids"], data["vgg_labels"]
    else:
        cat_ids, labels = data["ast_cat_ids"], data["ast_labels"]
    verify_training_identity(
        summary["outer_train_role"], training_identity(cat_ids, labels, outer_train)
    )
    verify_preprocessing(
        ROOT / summary["preprocessing"],
        expected_preprocessing(data, pipeline, outer_train),
    )

    history = read_json(ROOT / summary["training_history"])["history"]
    refit_epoch = int(summary["refit_epoch"])
    if len(history) != refit_epoch or int(summary["audit"]["stopped_epoch"]) != refit_epoch:
        raise AssertionError("Outer fixed-epoch history length changed")
    if summary["audit"]["fixed_epoch_training"] is not True:
        raise AssertionError("Outer fit is not marked fixed-epoch")
    if any("validation_animal_metrics" in row for row in history):
        raise AssertionError("Outer refit evaluated validation during fixed training")
    verify_history_order(history, pipeline, outer_train, cat_ids, full_seed)

    checkpoint = read_json(ROOT / summary["checkpoint_lock"])
    if checkpoint["locked_before_test_prediction"] is not True:
        raise AssertionError("Checkpoint was not locked before test prediction")
    if checkpoint["test_prediction_calls_before_lock"] != 0:
        raise AssertionError("Test was accessed before checkpoint lock")
    if checkpoint["selection_lock_sha256"] != lock_sha256:
        raise AssertionError("Checkpoint selection-lock hash mismatch")
    if int(checkpoint["refit_epoch"]) != refit_epoch:
        raise AssertionError("Checkpoint refit epoch mismatch")
    for checkpoint_key, summary_key in (
        ("model_weights_sha256", "model_weights_sha256"),
        ("preprocessing_sha256", "preprocessing_sha256"),
        ("training_history_sha256", "training_history_sha256"),
    ):
        if checkpoint[checkpoint_key] != summary[summary_key]:
            raise AssertionError(f"Checkpoint artifact differs: {checkpoint_key}")
    if checkpoint["model_state_sha256"] != saved_model_state_digest(
        ROOT / summary["model_weights"], pipeline
    ):
        raise AssertionError("Checkpoint model-state digest differs from saved weights")

    units, animals = verify_prediction_pair(
        ROOT / summary["test_unit_predictions"], ROOT / summary["test_cat_predictions"]
    )
    expected_test_cats = set(cat_ids[indices["test"]].astype(str))
    if set(animals["cat_id"].astype(str)) != expected_test_cats:
        raise AssertionError("Outer test cat identities differ from frozen roles")
    if int(summary["test_cats"]) != len(animals) or int(summary["test_units"]) != len(units):
        raise AssertionError("Outer test unit counts differ")
    recomputed = metric_bundle(animals)
    compare_metrics(recomputed, summary["metrics"], "outer.metrics")
    if summary["checkpoint_reload_state_match"] is not True:
        raise AssertionError("Outer checkpoint reload state mismatch")
    if int(summary["audit"]["trainable_parameters"]) != EXPECTED_TRAINABLE[pipeline]:
        raise AssertionError("Outer trainable parameter mismatch")
    if int(summary["audit"]["total_state_parameters"]) != EXPECTED_TOTAL[pipeline]:
        raise AssertionError("Outer total parameter mismatch")
    if int(summary["audit"]["seed"]) != full_seed:
        raise AssertionError("Outer fit seed mismatch")
    if int(summary["audit"]["training_seed"]) != full_seed + 1_000_000:
        raise AssertionError("Outer post-build seed mismatch")
    finite_tree(summary)
    audit = {
        "pipeline": pipeline,
        "repeat": repeat,
        "fold": fold,
        "base_seed": base_seed,
        "summary_path": summary_path.relative_to(ROOT).as_posix(),
        "summary_sha256": sha256(summary_path),
        "refit_epoch": refit_epoch,
        "test_cats": len(animals),
        "test_units": len(units),
    }
    return audit, animals


def verify_first_six_outer(protocol: dict[str, Any]) -> dict[str, Any]:
    """Verify the staged r0/f0/seed17 outer engineering checkpoint."""
    selection_audit = verify_selection_lock(protocol)
    lock_path = ROOT / selection_audit["selection_lock_path"]
    lock_sha = selection_audit["selection_lock_sha256"]
    lock = read_json(lock_path)
    lock_map = {
        (
            str(row["pipeline"]),
            int(row["repeat"]),
            int(row["fold"]),
            int(row["base_seed"]),
        ): row
        for row in lock["entries"]
    }
    data = source_data(protocol)
    fits = []
    for pipeline in PIPELINES:
        key = (pipeline, 0, 0, 17)
        audit, _ = verify_outer_fit(
            protocol,
            data,
            lock_sha,
            lock_map[key],
            pipeline,
            0,
            0,
            17,
        )
        fits.append(audit)

    authorization_path = RUN_ROOT / "authorizations" / "initial-six-outer.json"
    if not authorization_path.is_file():
        raise AssertionError("Initial-six outer authorization record is missing")
    authorization = read_json(authorization_path)
    if authorization["stage"] != "outer":
        raise AssertionError("Initial-six outer authorization stage changed")
    if authorization["authorization_scope"] != "initial-six-outer":
        raise AssertionError("Initial-six outer authorization scope changed")
    if authorization["director_authorized"] is not True:
        raise AssertionError("Initial-six outer is not director-authorized")
    if authorization["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise AssertionError("Initial-six outer authorization protocol mismatch")
    preflight_path = RUN_ROOT / protocol["outputs"]["cpu_preflight"]
    if authorization["preflight_sha256"] != sha256(preflight_path):
        raise AssertionError("Initial-six outer authorization preflight mismatch")
    if authorization["selection_lock_sha256"] != lock_sha:
        raise AssertionError("Initial-six outer authorization selection-lock mismatch")
    if any(value is not None for value in authorization["filters"].values()):
        raise AssertionError("Initial-six outer authorization contains fit filters")

    return {
        "status": "PASS_FIRST_SIX_OUTER_ENGINEERING_AUDIT",
        "selection_lock_sha256": lock_sha,
        "selection_fits_verified": 216,
        "outer_fits_verified": len(fits),
        "fits": fits,
        "authorization_path": authorization_path.relative_to(ROOT).as_posix(),
        "authorization_sha256": sha256(authorization_path),
        "continuation_decision_ignores_test_scores": True,
    }


def compare_flat_metric_dict(
    observed: dict[str, float], expected: dict[str, float], name: str
) -> None:
    if observed.keys() != expected.keys():
        raise AssertionError(f"{name}: flattened metric keys differ")
    for metric in observed:
        close(observed[metric], expected[metric], f"{name}.{metric}")


def verify_full_results(protocol: dict[str, Any]) -> dict[str, Any]:
    # Revalidate all selection artifacts and the epoch lock before reading outer data.
    selection_audit = verify_selection_lock(protocol)
    lock_path = ROOT / selection_audit["selection_lock_path"]
    lock_sha = selection_audit["selection_lock_sha256"]
    lock = read_json(lock_path)
    lock_map = {
        (
            str(row["pipeline"]),
            int(row["repeat"]),
            int(row["fold"]),
            int(row["base_seed"]),
        ): row
        for row in lock["entries"]
    }
    data = source_data(protocol)
    outer_audits = []
    outer_frames: dict[tuple[str, int, int, int], pd.DataFrame] = {}
    for pipeline, repeat, fold, base_seed in all_selection_coordinates(protocol):
        key = (pipeline, repeat, fold, base_seed)
        audit, animals = verify_outer_fit(
            protocol,
            data,
            lock_sha,
            lock_map[key],
            pipeline,
            repeat,
            fold,
            base_seed,
        )
        outer_audits.append(audit)
        outer_frames[key] = animals
    if len(outer_audits) != 216:
        raise AssertionError("Full outer audit did not find 216 fits")

    aggregate_path = RUN_ROOT / protocol["outputs"]["aggregate_results"]
    aggregate = read_json(aggregate_path)
    if aggregate["status"] != "complete" or aggregate["selection_lock_sha256"] != lock_sha:
        raise AssertionError("Aggregate result identity changed")
    expected_count_fields = {
        "physical_fits": 432,
        "selection_fits": 216,
        "outer_fits": 216,
        "unique_cats": 111,
        "prediction_occurrences_per_pipeline": 999,
        "independent_animal_count": 111,
    }
    if any(aggregate.get(key) != value for key, value in expected_count_fields.items()):
        raise AssertionError("Aggregate fit/animal counts changed")
    if tuple(aggregate["pipelines"]) != PIPELINES:
        raise AssertionError("Aggregate pipeline table changed")
    if tuple(aggregate["paired_comparisons"]) != tuple(COMPARISONS):
        raise AssertionError("Aggregate paired-comparison table changed")
    if tuple(protocol["evaluation"]["comparisons"]) != tuple(COMPARISONS):
        raise AssertionError("Protocol paired comparisons changed")
    if aggregate["claims_external_confirmation"] is not False:
        raise AssertionError("Aggregate claim boundary changed")

    oof_frames: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
    oof_metrics: dict[str, list[dict[str, float]]] = {pipeline: [] for pipeline in PIPELINES}
    for pipeline in PIPELINES:
        pipeline_result = aggregate["pipelines"][pipeline]
        if pipeline_result["complete_oof_sets"] != 9:
            raise AssertionError("Aggregate complete-OOF count changed")
        if pipeline_result["trainable_parameters"] != EXPECTED_TRAINABLE[pipeline]:
            raise AssertionError("Aggregate trainable parameter count changed")
        if pipeline_result["total_state_parameters"] != EXPECTED_TOTAL[pipeline]:
            raise AssertionError("Aggregate total parameter count changed")
        reported_rows = pipeline_result["oof_metrics"]
        reported_map = {
            (int(row["repeat"]), int(row["base_seed"])): row
            for row in reported_rows
        }
        expected_oof_keys = {
            (int(repeat), int(seed))
            for repeat in protocol["data"]["repeats"]
            for seed in protocol["training"]["model_seeds"]
        }
        if len(reported_map) != 9 or set(reported_map) != expected_oof_keys:
            raise AssertionError("Aggregate does not contain nine OOF metric sets")
        for repeat in protocol["data"]["repeats"]:
            for base_seed in protocol["training"]["model_seeds"]:
                pieces = [
                    outer_frames[(pipeline, int(repeat), int(fold), int(base_seed))]
                    for fold in protocol["data"]["folds"]
                ]
                frame = (
                    pd.concat(pieces, ignore_index=True)
                    .sort_values("cat_id")
                    .reset_index(drop=True)
                )
                if len(frame) != 111 or frame["cat_id"].nunique() != 111:
                    raise AssertionError("Recomputed OOF is not exactly 111 cats")
                metrics = metric_bundle(frame)
                reported = reported_map[(int(repeat), int(base_seed))]
                compare_metrics(metrics, reported["metrics"], "aggregate.oof")
                oof_path = ROOT / reported["oof_predictions"]
                if sha256(oof_path) != reported["oof_predictions_sha256"]:
                    raise AssertionError("OOF prediction hash mismatch")
                saved = pd.read_csv(oof_path, dtype={"cat_id": str})
                if not np.all(saved["repeat"].to_numpy(dtype=int) == int(repeat)):
                    raise AssertionError("OOF saved repeat identity differs")
                if not np.all(saved["base_seed"].to_numpy(dtype=int) == int(base_seed)):
                    raise AssertionError("OOF saved seed identity differs")
                saved_core = saved.drop(columns=["repeat", "base_seed"])
                rebuilt_core = frame[saved_core.columns]
                for column in ("cat_id", "true_label", "unit_count", "predicted_label"):
                    if not np.array_equal(saved_core[column], rebuilt_core[column]):
                        raise AssertionError(f"OOF saved identity differs: {column}")
                if not np.allclose(
                    saved_core[list(PROBABILITY_COLUMNS)],
                    rebuilt_core[list(PROBABILITY_COLUMNS)],
                    atol=1.0e-12,
                    rtol=0.0,
                ):
                    raise AssertionError("OOF saved probabilities differ")
                oof_frames[pipeline].append(frame)
                oof_metrics[pipeline].append(flatten_metrics(metrics))
        reported_summary = pipeline_result["summary_over_nine_complete_oof_sets"]
        for metric in oof_metrics[pipeline][0]:
            observed = mean_and_sample_sd(row[metric] for row in oof_metrics[pipeline])
            compare_flat_metric_dict(
                observed, reported_summary[metric], f"aggregate.summary.{pipeline}.{metric}"
            )

    for comparison, (left, right) in COMPARISONS.items():
        reported = aggregate["paired_comparisons"][comparison]
        if reported["left"] != left or reported["right"] != right:
            raise AssertionError(f"Paired comparison direction changed: {comparison}")
        deltas = []
        for left_metrics, right_metrics in zip(
            oof_metrics[left], oof_metrics[right], strict=True
        ):
            deltas.append(
                {
                    metric: left_metrics[metric] - right_metrics[metric]
                    for metric in left_metrics
                }
            )
        reported_deltas = reported["nine_complete_oof_deltas"]
        if len(reported_deltas) != 9:
            raise AssertionError(f"Paired delta count changed: {comparison}")
        for index, (observed_row, reported_row) in enumerate(
            zip(deltas, reported_deltas, strict=True)
        ):
            expected_repeat = protocol["data"]["repeats"][index // 3]
            expected_seed = protocol["training"]["model_seeds"][index % 3]
            if (
                int(reported_row["repeat"]) != int(expected_repeat)
                or int(reported_row["base_seed"]) != int(expected_seed)
            ):
                raise AssertionError(f"Paired delta identity changed: {comparison}")
            compare_flat_metric_dict(
                observed_row,
                {
                    metric: float(reported_row[metric])
                    for metric in observed_row
                },
                f"paired.{comparison}.row{index}",
            )
        for metric in deltas[0]:
            observed = mean_and_sample_sd(row[metric] for row in deltas)
            compare_flat_metric_dict(
                observed, reported["summary"][metric], f"paired.{comparison}.{metric}"
            )
        signs = {
            "positive": sum(row["macro_f1"] > 0.0 for row in deltas),
            "tied": sum(row["macro_f1"] == 0.0 for row in deltas),
            "negative": sum(row["macro_f1"] < 0.0 for row in deltas),
        }
        if signs != reported["macro_f1_signs"]:
            raise AssertionError(f"Paired sign counts differ: {comparison}")

    reference = oof_frames[PIPELINES[0]][0].sort_values("cat_id").reset_index(drop=True)
    labels = reference["true_label"].to_numpy(dtype=np.int64)
    cat_ids = reference["cat_id"].astype(str).to_numpy()
    probability_arrays = {}
    for pipeline in PIPELINES:
        arrays = []
        for frame in oof_frames[pipeline]:
            ordered = frame.sort_values("cat_id").reset_index(drop=True)
            if not np.array_equal(ordered["cat_id"].astype(str).to_numpy(), cat_ids):
                raise AssertionError("OOF cat order differs across pipelines")
            if not np.array_equal(ordered["true_label"].to_numpy(dtype=int), labels):
                raise AssertionError("OOF labels differ across pipelines")
            arrays.append(ordered[list(PROBABILITY_COLUMNS)].to_numpy(float))
        probability_arrays[pipeline] = np.stack(arrays)
    bootstrap = protocol["evaluation"]["bootstrap"]
    if (
        aggregate["bootstrap"]["replicates"] != int(bootstrap["replicates"])
        or aggregate["bootstrap"]["seed"] != int(bootstrap["seed"])
        or aggregate["bootstrap"][
            "shared_cat_draws_across_all_models_and_nine_oof_sets"
        ]
        is not True
    ):
        raise AssertionError("Aggregate bootstrap contract changed")
    sample_indices = stratified_bootstrap_indices(
        labels, int(bootstrap["replicates"]), int(bootstrap["seed"])
    )
    model_bootstrap = {
        pipeline: bootstrap_model_metrics(labels, probability_arrays[pipeline], sample_indices)
        for pipeline in PIPELINES
    }
    for comparison, (left, right) in COMPARISONS.items():
        intervals = aggregate["paired_comparisons"][comparison][
            "stratified_cat_bootstrap_interval"
        ]
        for metric in model_bootstrap[left]:
            difference = model_bootstrap[left][metric] - model_bootstrap[right][metric]
            close(
                np.quantile(difference, 0.025),
                intervals[metric]["lower_2_5_percentile"],
                f"bootstrap.{comparison}.{metric}.lower",
            )
            close(
                np.quantile(difference, 0.975),
                intervals[metric]["upper_97_5_percentile"],
                f"bootstrap.{comparison}.{metric}.upper",
            )

    metadata_path = ROOT / protocol["outputs"]["metadata"]
    metadata = read_json(metadata_path)
    if metadata["run_summary"] != aggregate_path.relative_to(ROOT).as_posix():
        raise AssertionError("Metadata aggregate path changed")
    if metadata["run_summary_sha256"] != sha256(aggregate_path):
        raise AssertionError("Metadata aggregate hash changed")
    metadata_core = dict(metadata)
    metadata_core.pop("run_summary")
    metadata_core.pop("run_summary_sha256")
    if metadata_core != aggregate:
        raise AssertionError("Metadata does not exactly embed aggregate results")
    outer_authorization_path = RUN_ROOT / "authorizations" / "full-outer.json"
    authorization = read_json(outer_authorization_path)
    if authorization["stage"] != "outer":
        raise AssertionError("Full-outer authorization stage changed")
    if authorization["authorization_scope"] != "full-outer":
        raise AssertionError("Full-outer authorization scope changed")
    if authorization["director_authorized"] is not True:
        raise AssertionError("Full-outer is not director-authorized")
    if authorization["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise AssertionError("Full-outer authorization protocol mismatch")
    if authorization["runner_sha256"] != protocol["dependencies"]["runner_sha256"]:
        raise AssertionError("Full-outer authorization runner mismatch")
    preflight_path = RUN_ROOT / protocol["outputs"]["cpu_preflight"]
    if authorization["preflight_sha256"] != sha256(preflight_path):
        raise AssertionError("Full-outer authorization preflight mismatch")
    if authorization["selection_lock_sha256"] != lock_sha:
        raise AssertionError("Full-outer authorization selection-lock mismatch")
    if any(value is not None for value in authorization["filters"].values()):
        raise AssertionError("Full-outer authorization contains fit filters")
    finite_tree(aggregate)
    return {
        "status": "PASS_FULL_OUTER_AND_AGGREGATE_AUDIT",
        "selection_lock_sha256": lock_sha,
        "selection_fits_verified": 216,
        "outer_fits_verified": 216,
        "complete_oof_sets_verified": 54,
        "prediction_occurrences_per_pipeline": 999,
        "independent_animals": 111,
        "bootstrap_replicates": int(bootstrap["replicates"]),
        "paired_comparisons_verified": len(COMPARISONS),
        "aggregate_path": aggregate_path.relative_to(ROOT).as_posix(),
        "aggregate_sha256": sha256(aggregate_path),
        "metadata_path": metadata_path.relative_to(ROOT).as_posix(),
        "metadata_sha256": sha256(metadata_path),
        "full_outer_authorization_path": outer_authorization_path.relative_to(
            ROOT
        ).as_posix(),
        "full_outer_authorization_sha256": sha256(outer_authorization_path),
        "outer_fit_summaries": outer_audits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "preflight",
            "first-six",
            "selection-lock",
            "first-six-outer",
            "full",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_PATH)
    dependencies = verify_protocol_dependencies(protocol)
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "audit_id": f"IDEA-089-independent-{args.mode}-audit",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "dependency_hashes_verified": dependencies,
        "preflight": verify_preflight(protocol),
    }
    if args.mode == "first-six":
        result["first_six"] = verify_first_six(protocol)
        result["status"] = "PASS_FIRST_SIX_SELECTION_ENGINEERING_AUDIT"
    elif args.mode == "selection-lock":
        result["selection_lock"] = verify_selection_lock(protocol)
        result["status"] = "PASS_FULL_SELECTION_AND_EPOCH_LOCK_AUDIT"
    elif args.mode == "first-six-outer":
        result["first_six_outer"] = verify_first_six_outer(protocol)
        result["status"] = "PASS_FIRST_SIX_OUTER_ENGINEERING_AUDIT"
    elif args.mode == "full":
        result["full"] = verify_full_results(protocol)
        result["status"] = "PASS_FULL_OUTER_AND_AGGREGATE_AUDIT"
    else:
        result["status"] = "PASS_INDEPENDENT_PREFLIGHT_AUDIT"
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        if "idea089" not in str(output).lower():
            raise ValueError("Refusing to write outside an IDEA-089-named path")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
