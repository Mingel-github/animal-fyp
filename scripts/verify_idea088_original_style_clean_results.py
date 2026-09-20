"""Independent verification for IDEA-088.

This module intentionally does not import the IDEA-088 training runner.  It
rebuilds role membership, preprocessing, epoch orders, early stopping,
unit-to-cat aggregation, every reported metric, and the final aggregation from
the frozen inputs and saved fit artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/protocol/meowagenet_idea088_original_style_clean_v1.json"
RUN_ROOT = ROOT / "runs/meowagenet_idea088_original_style_clean_v1"
AGGREGATE_PATH = RUN_ROOT / "idea088_aggregate_results.json"
METADATA_PATH = (
    ROOT
    / "metadata/experiments/meowagenet_idea088_original_style_clean_v1_results.json"
)
PIPELINES = ("vggish_mlp", "A0_ast_only", "C1_bounded_wide_additive")
AST_PIPELINES = PIPELINES[1:]
CLASS_NAMES = ("kitten", "adult", "senior")
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
METRIC_NAMES = ("accuracy", "macro_f1", "balanced_accuracy", "cross_entropy", "brier")
EXPECTED_PROTOCOL_SHA256 = "c12bd2a5a4abe6b4f0af425a153747a7d6c96b6c9214023684ffc33625d21847"
EXPECTED_RUNNER_SHA256 = "f1cd6c9993b505e3e3f5f6d19488ef1b755c55eec8c63ffb68b4918e8d0db5cc"
EXPECTED_ROLES_SHA256 = "a0ce4989985989ebdd982bbf565e33c080b4e314e232eb875f30fbdb4eac34b8"
EXPECTED_PREFLIGHT_SHA256 = "fb4fa4b55762bed639528387858b7f417f452083cde7e778792a40c4392eb449"
EXPECTED_HISTORICAL_FILES = 3965
EXPECTED_HISTORICAL_SHA256 = "77f950fbec6ed720e7f6cbc4ade6822019d225d3dc679495b9272f8db6e56807"
TOLERANCE = 1.0e-11
PREDICTION_TOLERANCE = 5.0e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("first-fit", "full", "protection"), required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def close(left: float, right: float, context: str, tolerance: float = TOLERANCE) -> float:
    difference = abs(float(left) - float(right))
    if not math.isfinite(difference) or difference > tolerance:
        raise RuntimeError(f"{context}: {left} != {right}")
    return difference


def compare_nested(
    rebuilt: Any, saved: Any, context: str, tolerance: float = TOLERANCE
) -> float:
    """Compare a rebuilt JSON-like subset, allowing only tiny float differences."""
    if isinstance(rebuilt, dict):
        if not isinstance(saved, dict):
            raise RuntimeError(f"{context}: saved value is not an object")
        missing = set(rebuilt) - set(saved)
        if missing:
            raise RuntimeError(f"{context}: saved object lacks {sorted(missing)}")
        return max(
            (
                compare_nested(rebuilt[key], saved[key], f"{context}.{key}", tolerance)
                for key in rebuilt
            ),
            default=0.0,
        )
    if isinstance(rebuilt, list):
        if not isinstance(saved, list) or len(rebuilt) != len(saved):
            raise RuntimeError(f"{context}: list shape differs")
        return max(
            (
                compare_nested(left, right, f"{context}[{index}]", tolerance)
                for index, (left, right) in enumerate(zip(rebuilt, saved))
            ),
            default=0.0,
        )
    if isinstance(rebuilt, bool) or isinstance(saved, bool):
        if rebuilt is not saved:
            raise RuntimeError(f"{context}: {rebuilt!r} != {saved!r}")
        return 0.0
    if isinstance(rebuilt, (int, float, np.integer, np.floating)) and isinstance(
        saved, (int, float, np.integer, np.floating)
    ):
        return close(float(rebuilt), float(saved), context, tolerance)
    if rebuilt != saved:
        raise RuntimeError(f"{context}: {rebuilt!r} != {saved!r}")
    return 0.0


def verify_locks(protocol: dict[str, Any]) -> dict[str, Any]:
    if sha256(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("IDEA-088 protocol hash changed")
    runner_path = ROOT / "scripts/run_meowagenet_idea088_original_style_clean.py"
    roles_path = ROOT / protocol["splits"]["roles_path"]
    preflight_path = RUN_ROOT / "preflight/cpu_preflight.json"
    expected = {
        runner_path: EXPECTED_RUNNER_SHA256,
        roles_path: EXPECTED_ROLES_SHA256,
        preflight_path: EXPECTED_PREFLIGHT_SHA256,
    }
    for path, expected_hash in expected.items():
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"IDEA-088 frozen artifact changed: {path}")
    dependency_hashes: dict[str, str] = {}
    for entry in protocol["dependencies"]:
        path = ROOT / entry["path"]
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise RuntimeError(f"IDEA-088 dependency changed: {entry['path']}")
        dependency_hashes[entry["path"]] = actual
    source = runner_path.read_text(encoding="utf-8")
    for function_name, prediction_marker in (
        ("def fit_vggish", "test_probabilities = restored.predict("),
        ("def fit_ast", "units = predict_ast_units("),
    ):
        start = source.index(function_name)
        lock_position = source.index("write_json(checkpoint_path, checkpoint_lock)", start)
        prediction_position = source.index(prediction_marker, start)
        if lock_position >= prediction_position:
            raise RuntimeError(f"{function_name} predicts before checkpoint lock")
    return {
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "roles_sha256": EXPECTED_ROLES_SHA256,
        "preflight_sha256": EXPECTED_PREFLIGHT_SHA256,
        "dependency_count": len(dependency_hashes),
        "dependency_mismatches": 0,
        "checkpoint_before_prediction_static": True,
    }


def load_inputs(protocol: dict[str, Any]) -> dict[str, Any]:
    ast_path = ROOT / protocol["data"]["frozen_ast_embedding_path"]
    ast = np.load(ast_path)
    ast_data = {key: ast[key] for key in ast.files}
    acoustic_loaded = np.load(ROOT / protocol["data"]["acoustic_feature_path"])
    acoustic = {key: acoustic_loaded[key] for key in acoustic_loaded.files}
    if not np.array_equal(acoustic["call_ids"].astype(str), ast_data["call_ids"].astype(str)):
        raise RuntimeError("IDEA-088 acoustic and AST call order differ")

    frame = pd.read_csv(
        ROOT / protocol["data"]["vggish_csv_path"], dtype={"cat_id": str}
    )
    if len(frame) != 937:
        raise RuntimeError("IDEA-088 VGGish source row count changed")
    clean = frame[frame["cat_id"] != "049A"].copy()
    clean["source_row_index"] = clean.index.astype(np.int64)
    clean = clean.reset_index(drop=True)
    feature_columns = [str(index) for index in range(128)] + ["mean_freq"]
    vgg_labels = np.where(
        clean["target"].to_numpy(float) < 0.5,
        0,
        np.where(clean["target"].to_numpy(float) < 10.0, 1, 2),
    ).astype(np.int64)
    vgg = {
        "features": clean[feature_columns].to_numpy(np.float32),
        "labels": vgg_labels,
        "cat_ids": clean["cat_id"].to_numpy(str),
        "unit_ids": np.asarray(
            [f"vggish_source_row_{index}" for index in clean["source_row_index"]],
            dtype=str,
        ),
    }
    if ast_data["embeddings"].shape != (792, 768):
        raise RuntimeError("IDEA-088 AST shape changed")
    if acoustic["features"].shape != (792, 20):
        raise RuntimeError("IDEA-088 acoustic shape changed")
    if vgg["features"].shape != (936, 129):
        raise RuntimeError("IDEA-088 VGGish clean shape changed")
    ast_cat_labels: dict[str, int] = {}
    for cat in np.unique(ast_data["cat_ids"].astype(str)):
        values = np.unique(
            ast_data["labels"][ast_data["cat_ids"].astype(str) == cat].astype(np.int64)
        )
        if len(values) != 1:
            raise RuntimeError(f"IDEA-088 AST cat label conflict: {cat}")
        ast_cat_labels[cat] = int(values[0])
    for cat in np.unique(vgg["cat_ids"]):
        values = np.unique(vgg_labels[vgg["cat_ids"] == cat])
        if len(values) != 1 or int(values[0]) != ast_cat_labels[cat]:
            raise RuntimeError(f"IDEA-088 VGGish/AST label mismatch: {cat}")
    return {"ast": ast_data, "acoustic": acoustic, "vggish": vgg}


def validate_roles(
    protocol: dict[str, Any], roles: dict[str, Any], expected_cats: set[str]
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    if sha256(ROOT / protocol["splits"]["roles_path"]) != protocol["splits"]["roles_sha256"]:
        raise RuntimeError("IDEA-088 role manifest hash differs from protocol")
    cells: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in roles["cells"]:
        key = (int(raw["split_seed"]), int(raw["fold"]))
        train = [str(value) for value in raw["train_cat_ids"]]
        test = [str(value) for value in raw["test_cat_ids"]]
        if train != sorted(set(train)) or test != sorted(set(test)):
            raise RuntimeError(f"IDEA-088 unsorted or duplicate roles: {key}")
        if set(train) & set(test) or set(train) | set(test) != expected_cats:
            raise RuntimeError(f"IDEA-088 role leakage or incomplete coverage: {key}")
        if not {"000A", "046A"} <= set(train) or {"000A", "046A"} & set(test):
            raise RuntimeError(f"IDEA-088 forced-training role changed: {key}")
        cells[key] = {"train": train, "test": test}
    expected_keys = {
        (int(seed), fold) for seed in protocol["splits"]["seeds"] for fold in range(4)
    }
    if set(cells) != expected_keys:
        raise RuntimeError("IDEA-088 role cell identities changed")
    multiplicity: dict[str, Any] = {}
    for seed in protocol["splits"]["seeds"]:
        counts = Counter(cat for fold in range(4) for cat in cells[(seed, fold)]["test"])
        duplicates = sorted(cat for cat, count in counts.items() if count == 2)
        omitted = sorted(cat for cat in expected_cats if counts.get(cat, 0) == 0)
        if sum(counts.values()) != 111 or len(counts) != 109:
            raise RuntimeError(f"IDEA-088 unexpected test multiplicity for seed {seed}")
        if omitted != ["000A", "046A"] or len(duplicates) != 2:
            raise RuntimeError(f"IDEA-088 forced/swap multiplicity changed for seed {seed}")
        multiplicity[str(seed)] = {
            "test_appearances": 111,
            "unique_test_cats": 109,
            "omitted_forced_training_cats": omitted,
            "duplicated_replacement_cats": duplicates,
        }
    return cells, multiplicity


def indices_for_cats(cat_ids: np.ndarray, cats: Iterable[str]) -> np.ndarray:
    cat_set = set(str(value) for value in cats)
    result = np.flatnonzero(np.isin(cat_ids.astype(str), sorted(cat_set))).astype(np.int64)
    if set(cat_ids[result].astype(str)) != cat_set:
        raise RuntimeError("IDEA-088 failed to resolve role cats")
    return result


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=3).astype(float)
    if np.any(counts == 0):
        raise RuntimeError("IDEA-088 training role is missing a class")
    return (len(labels) / (3.0 * counts)).astype(np.float32)


def epoch_order(indices: np.ndarray, training_seed: int, epoch: int) -> np.ndarray:
    generator = np.random.default_rng(int(training_seed) + int(epoch))
    return indices[generator.permutation(len(indices))].astype(np.int64)


def replay_early_stopping(history: list[dict[str, Any]]) -> dict[str, Any]:
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    stop_epoch: int | None = None
    for row in history:
        loss = float(row["training_loss"])
        if not math.isfinite(loss):
            raise RuntimeError("IDEA-088 history contains non-finite loss")
        if loss < best_loss - 0.001:
            best_loss = loss
            best_epoch = int(row["epoch"])
            stale = 0
        else:
            stale += 1
        if stale >= 30:
            stop_epoch = int(row["epoch"])
            break
    return {
        "best_epoch": best_epoch,
        "best_training_loss": best_loss,
        "stop_epoch": stop_epoch,
        "stale_epochs": stale,
    }


def metric_bundle(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    recalls = recall_score(
        labels, predictions, labels=[0, 1, 2], average=None, zero_division=0
    )
    targets = np.eye(3, dtype=np.float64)[labels]
    clipped = np.clip(probabilities, 1.0e-12, 1.0)
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
            raise RuntimeError(f"IDEA-088 prediction label conflict for cat {cat_id}")
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


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def fit_path(pipeline: str, seed: int, fold: int) -> Path:
    return RUN_ROOT / "fits" / pipeline / f"seed_{seed}" / f"fold_{fold}" / "fit_summary.json"


def verify_fit(
    protocol: dict[str, Any],
    inputs: dict[str, Any],
    cells: dict[tuple[int, int], dict[str, Any]],
    pipeline: str,
    seed: int,
    fold: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = fit_path(pipeline, seed, fold)
    if not path.is_file():
        raise RuntimeError(f"IDEA-088 missing fit: {path}")
    fit = read_json(path)
    expected_identity = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "roles_sha256": EXPECTED_ROLES_SHA256,
        "pipeline": pipeline,
        "split_seed": seed,
        "fold": fold,
        "model_seed": seed + fold,
        "training_seed": seed + fold + 1_000_000,
        "status": "complete",
        "test_prediction_calls": 1,
        "checkpoint_reload_state_match": True,
    }
    compare_nested(expected_identity, fit, f"fit.{pipeline}.{seed}.{fold}")
    for path_key, hash_key in (
        ("model_weights", "model_weights_sha256"),
        ("preprocessing", "preprocessing_sha256"),
        ("epoch_orders", "epoch_orders_sha256"),
        ("checkpoint_lock", "checkpoint_lock_sha256"),
        ("unit_predictions", "unit_predictions_sha256"),
        ("cat_predictions", "cat_predictions_sha256"),
    ):
        artifact = ROOT / fit[path_key]
        if not artifact.is_file() or sha256(artifact) != fit[hash_key]:
            raise RuntimeError(f"IDEA-088 fit artifact changed: {path_key} {path}")

    cell = cells[(seed, fold)]
    source = inputs["vggish"] if pipeline == "vggish_mlp" else inputs["ast"]
    source_features = (
        source["features"] if pipeline == "vggish_mlp" else source["embeddings"]
    )
    train_indices = indices_for_cats(source["cat_ids"], cell["train"])
    test_indices = indices_for_cats(source["cat_ids"], cell["test"])
    if len(set(train_indices) & set(test_indices)) or len(train_indices) + len(test_indices) != len(source["cat_ids"]):
        raise RuntimeError(f"IDEA-088 unit role leakage: {pipeline}/{seed}/{fold}")
    expected_counts = {
        "train_cats": len(cell["train"]),
        "test_cats": len(cell["test"]),
        "train_units": len(train_indices),
        "test_units": len(test_indices),
    }
    compare_nested(expected_counts, fit, f"fit_counts.{pipeline}.{seed}.{fold}")

    preprocessing = np.load(ROOT / fit["preprocessing"])
    np.testing.assert_array_equal(preprocessing["train_indices"], train_indices)
    np.testing.assert_array_equal(preprocessing["test_indices"], test_indices)
    expected_weights = class_weights(source["labels"][train_indices])
    np.testing.assert_array_equal(preprocessing["class_weights"], expected_weights)
    np.testing.assert_array_equal(np.asarray(fit["class_weights"], np.float32), expected_weights)
    if pipeline == "vggish_mlp":
        mean = source_features[train_indices].mean(axis=0).astype(np.float32)
        scale = source_features[train_indices].std(axis=0).astype(np.float32)
        scale = np.where(scale > 1.0e-12, scale, 1.0).astype(np.float32)
        np.testing.assert_array_equal(preprocessing["feature_mean"], mean)
        np.testing.assert_array_equal(preprocessing["feature_scale"], scale)
    else:
        mean = source_features[train_indices].mean(axis=0).astype(np.float32)
        scale = source_features[train_indices].std(axis=0).astype(np.float32)
        scale = np.where(scale > 1.0e-12, scale, 1.0).astype(np.float32)
        np.testing.assert_array_equal(preprocessing["ast_mean"], mean)
        np.testing.assert_array_equal(preprocessing["ast_scale"], scale)
        if pipeline == "C1_bounded_wide_additive":
            age_train = inputs["acoustic"]["features"][train_indices]
            with np.errstate(all="ignore"):
                median = np.nanmedian(age_train, axis=0)
            median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
            imputed = np.where(np.isfinite(age_train), age_train, median[None, :])
            age_mean = imputed.mean(axis=0).astype(np.float32)
            age_scale = imputed.std(axis=0).astype(np.float32)
            age_scale = np.where(age_scale > 1.0e-8, age_scale, 1.0).astype(np.float32)
            np.testing.assert_array_equal(preprocessing["age_median"], median)
            np.testing.assert_array_equal(preprocessing["age_mean"], age_mean)
            np.testing.assert_array_equal(preprocessing["age_scale"], age_scale)

    history = fit["history"]
    if len(history) != int(fit["stopped_epoch"]):
        raise RuntimeError(f"IDEA-088 history length differs: {path}")
    orders = np.load(ROOT / fit["epoch_orders"])["unit_indices"].astype(np.int64)
    if orders.shape != (len(history), len(train_indices)):
        raise RuntimeError(f"IDEA-088 epoch-order shape differs: {path}")
    for epoch_index, (row, order) in enumerate(zip(history, orders, strict=True), 1):
        if int(row["epoch"]) != epoch_index:
            raise RuntimeError(f"IDEA-088 non-sequential history: {path}")
        expected_order = epoch_order(train_indices, int(fit["training_seed"]), epoch_index)
        np.testing.assert_array_equal(order, expected_order)
        if array_sha256(order.astype("<i8")) != row["unit_order_sha256"]:
            raise RuntimeError(f"IDEA-088 epoch-order digest differs: {path}")
    replay = replay_early_stopping(history)
    compare_nested(
        {
            "best_epoch": replay["best_epoch"],
            "best_training_loss": replay["best_training_loss"],
        },
        fit,
        f"early_stop.{pipeline}.{seed}.{fold}",
    )
    if len(history) < int(protocol["training"]["maximum_epochs"]):
        if replay["stop_epoch"] != int(fit["stopped_epoch"]):
            raise RuntimeError(f"IDEA-088 early stopping differs: {path}")

    lock = read_json(ROOT / fit["checkpoint_lock"])
    compare_nested(
        {
            **{
                key: fit[key]
                for key in expected_identity
                if key
                not in {"status", "test_prediction_calls", "checkpoint_reload_state_match"}
            },
            "status": "checkpoint_locked_before_test_prediction",
            "best_epoch": fit["best_epoch"],
            "stopped_epoch": fit["stopped_epoch"],
            "best_training_loss": fit["best_training_loss"],
            "model_weights": fit["model_weights"],
            "model_weights_sha256": fit["model_weights_sha256"],
            "test_accessed": False,
        },
        lock,
        f"checkpoint.{pipeline}.{seed}.{fold}",
    )
    if pipeline in AST_PIPELINES:
        loaded = torch.load(ROOT / fit["model_weights"], map_location="cpu", weights_only=True)
        if state_digest(loaded["state_dict"]) != lock["restored_state_sha256"]:
            raise RuntimeError(f"IDEA-088 checkpoint state digest differs: {path}")

    units = pd.read_csv(ROOT / fit["unit_predictions"], dtype={"cat_id": str})
    cats = pd.read_csv(ROOT / fit["cat_predictions"], dtype={"cat_id": str})
    if len(units) != len(test_indices) or not units["unit_index"].is_unique:
        raise RuntimeError(f"IDEA-088 prediction-unit multiplicity differs: {path}")
    np.testing.assert_array_equal(units["unit_index"].to_numpy(np.int64), test_indices)
    np.testing.assert_array_equal(
        units["true_label"].to_numpy(np.int64), source["labels"][test_indices].astype(np.int64)
    )
    np.testing.assert_array_equal(
        units["cat_id"].to_numpy(str), source["cat_ids"][test_indices].astype(str)
    )
    expected_unit_ids = (
        source["unit_ids"][test_indices]
        if pipeline == "vggish_mlp"
        else source["call_ids"][test_indices].astype(str)
    )
    np.testing.assert_array_equal(units["unit_id"].to_numpy(str), expected_unit_ids)
    probabilities = units[list(PROBABILITY_COLUMNS)].to_numpy(float)
    if not np.isfinite(probabilities).all() or np.max(np.abs(probabilities.sum(axis=1) - 1.0)) > 1.0e-6:
        raise RuntimeError(f"IDEA-088 invalid probabilities: {path}")
    rebuilt_cats = units_to_cats(units)
    pd.testing.assert_frame_equal(
        rebuilt_cats,
        cats,
        check_exact=False,
        rtol=0.0,
        atol=PREDICTION_TOLERANCE,
    )
    rebuilt_metrics = {
        "public_cat_probability_mean": metric_bundle(
            rebuilt_cats["true_label"].to_numpy(np.int64),
            rebuilt_cats[list(PROBABILITY_COLUMNS)].to_numpy(float),
        ),
        "native_prediction_unit": metric_bundle(
            units["true_label"].to_numpy(np.int64), probabilities
        ),
    }
    metric_difference = compare_nested(
        rebuilt_metrics,
        fit["metrics"],
        f"metrics.{pipeline}.{seed}.{fold}",
        PREDICTION_TOLERANCE,
    )
    verified_fit = dict(fit)
    verified_fit["metrics"] = rebuilt_metrics
    return verified_fit, {
        "pipeline": pipeline,
        "split_seed": seed,
        "fold": fold,
        "train_units": len(train_indices),
        "test_units": len(test_indices),
        "epochs": len(history),
        "best_epoch": int(fit["best_epoch"]),
        "metric_max_abs_difference": metric_difference,
        "metrics": rebuilt_metrics,
        "initial_common_state_sha256": fit.get("initial_common_state_sha256"),
        "class_weights": expected_weights.astype(float).tolist(),
        "orders": orders,
        "test_cat_ids": rebuilt_cats["cat_id"].astype(str).tolist(),
    }


def sample_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def rebuild_aggregate(
    protocol: dict[str, Any], fits: dict[tuple[str, int, int], dict[str, Any]]
) -> dict[str, Any]:
    pipeline_summaries: dict[str, Any] = {}
    for pipeline in PIPELINES:
        seed_rows = []
        for seed in protocol["splits"]["seeds"]:
            seed_fits = [fits[(pipeline, seed, fold)] for fold in range(4)]
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
                        for metric in METRIC_NAMES
                    },
                    "native_prediction_unit": {
                        metric: float(
                            np.mean(
                                [fit["metrics"]["native_prediction_unit"][metric] for fit in seed_fits]
                            )
                        )
                        for metric in METRIC_NAMES
                    },
                }
            )
        pipeline_summaries[pipeline] = {
            "input_unit": fits[(pipeline, protocol["splits"]["seeds"][0], 0)]["input_unit"],
            "folds": 20,
            "seed_four_fold_means": seed_rows,
            "public_cat_probability_mean": {
                metric: {
                    "mean_over_five_seed_means": float(
                        np.mean([row["public_cat_probability_mean"][metric] for row in seed_rows])
                    ),
                    "sample_sd_over_five_seed_means": sample_sd(
                        [row["public_cat_probability_mean"][metric] for row in seed_rows]
                    ),
                }
                for metric in METRIC_NAMES
            },
            "native_prediction_unit": {
                metric: {
                    "mean_over_five_seed_means": float(
                        np.mean([row["native_prediction_unit"][metric] for row in seed_rows])
                    ),
                    "sample_sd_over_five_seed_means": sample_sd(
                        [row["native_prediction_unit"][metric] for row in seed_rows]
                    ),
                }
                for metric in METRIC_NAMES
            },
        }
    paired_cells = []
    for seed in protocol["splits"]["seeds"]:
        for fold in range(4):
            a0 = fits[("A0_ast_only", seed, fold)]
            c1 = fits[("C1_bounded_wide_additive", seed, fold)]
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
    seed_deltas = []
    for seed in protocol["splits"]["seeds"]:
        selected = [row for row in paired_cells if row["split_seed"] == seed]
        seed_deltas.append(
            {
                "split_seed": seed,
                "C1_minus_A0_public_cat_macro_f1": float(
                    np.mean([row["C1_minus_A0_public_cat_macro_f1"] for row in selected])
                ),
                "C1_minus_A0_native_call_macro_f1": float(
                    np.mean([row["C1_minus_A0_native_call_macro_f1"] for row in selected])
                ),
            }
        )
    public = [row["C1_minus_A0_public_cat_macro_f1"] for row in seed_deltas]
    native = [row["C1_minus_A0_native_call_macro_f1"] for row in seed_deltas]
    return {
        "pipelines": pipeline_summaries,
        "paired_C1_minus_A0": {
            "cell_deltas": paired_cells,
            "seed_four_fold_mean_deltas": seed_deltas,
            "public_cat_macro_f1": {
                "mean": float(np.mean(public)),
                "sample_sd": sample_sd(public),
                "positive_seeds": int(np.sum(np.asarray(public) > 0.0)),
                "tied_seeds": int(np.sum(np.asarray(public) == 0.0)),
                "negative_seeds": int(np.sum(np.asarray(public) < 0.0)),
            },
            "native_call_macro_f1": {
                "mean": float(np.mean(native)),
                "sample_sd": sample_sd(native),
                "positive_seeds": int(np.sum(np.asarray(native) > 0.0)),
                "tied_seeds": int(np.sum(np.asarray(native) == 0.0)),
                "negative_seeds": int(np.sum(np.asarray(native) < 0.0)),
            },
        },
    }


def rebuild_class_recall_summary(
    protocol: dict[str, Any], fits: dict[tuple[str, int, int], dict[str, Any]]
) -> dict[str, Any]:
    """Report the class recalls required by the protocol but not pooled by the runner."""
    result: dict[str, Any] = {}
    for pipeline in PIPELINES:
        result[pipeline] = {}
        for scope in ("public_cat_probability_mean", "native_prediction_unit"):
            result[pipeline][scope] = {}
            for class_name in CLASS_NAMES:
                seed_values = [
                    float(
                        np.mean(
                            [
                                fits[(pipeline, seed, fold)]["metrics"][scope][
                                    "class_recall"
                                ][class_name]
                                for fold in range(4)
                            ]
                        )
                    )
                    for seed in protocol["splits"]["seeds"]
                ]
                result[pipeline][scope][class_name] = {
                    "seed_four_fold_means": seed_values,
                    "mean_over_five_seed_means": float(np.mean(seed_values)),
                    "sample_sd_over_five_seed_means": sample_sd(seed_values),
                }
    return result


def protection_snapshot() -> dict[str, Any]:
    """Re-run the director's PowerShell-culture path ordering exactly."""
    escaped_root = str(ROOT).replace("'", "''")
    script = rf"""
$root=(Resolve-Path -LiteralPath '{escaped_root}').Path
$files=@()
foreach($d in @('configs/protocol','scripts','tests','reports','metadata/experiments','plan','splits')){{
  $files += Get-ChildItem -LiteralPath (Join-Path $root $d) -File | Where-Object {{ $_.Name -notmatch '088' }}
}}
foreach($d in @('meowagenet_idea076_C1_final_seed_confirmation_v1','meowagenet_idea083_C1_U1_equal_weight_fusion_v1','meowagenet_idea084_grouped_dual_branch_acoustic_residual_v1','meowagenet_idea085_acoustic_temporal_residual_v1','meowagenet_idea086_acoustic_set_residual_v1','meowagenet_idea087_nested_hpo_v1')){{
  $files += Get-ChildItem -LiteralPath (Join-Path $root ('runs/' + $d)) -File -Recurse
}}
$files=@($files | Sort-Object FullName -Unique)
$lines=@()
foreach($f in $files){{
  $rel=[System.IO.Path]::GetRelativePath($root,$f.FullName).Replace('\','/')
  $sha=(Get-FileHash -Algorithm SHA256 -LiteralPath $f.FullName).Hash.ToLowerInvariant()
  $lines += "${{rel}}:${{sha}}"
}}
$text=[string]::Join("`n",$lines)
$bytes=[System.Text.UTF8Encoding]::new($false).GetBytes($text)
$hash=[System.BitConverter]::ToString([System.Security.Cryptography.SHA256]::HashData($bytes)).Replace('-','').ToLowerInvariant()
@{{file_count=$files.Count; combined_sha256=$hash}} | ConvertTo-Json -Compress
"""
    completed = subprocess.run(
        ["pwsh", "-NoProfile", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    result = json.loads(completed.stdout.strip())
    result["matches_pre_execution_snapshot"] = (
        int(result["file_count"]) == EXPECTED_HISTORICAL_FILES
        and result["combined_sha256"] == EXPECTED_HISTORICAL_SHA256
    )
    if not result["matches_pre_execution_snapshot"]:
        raise RuntimeError(f"IDEA-088 historical protection snapshot changed: {result}")
    return result


def verify_first_fit() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    locks = verify_locks(protocol)
    inputs = load_inputs(protocol)
    roles = read_json(ROOT / protocol["splits"]["roles_path"])
    cells, multiplicity = validate_roles(
        protocol, roles, set(inputs["ast"]["cat_ids"].astype(str))
    )
    summaries = list((RUN_ROOT / "fits").rglob("fit_summary.json"))
    expected = fit_path("A0_ast_only", 7270, 0)
    if summaries != [expected]:
        raise RuntimeError("IDEA-088 first-fit gate expected exactly one A0/7270/0 summary")
    fit, audit = verify_fit(protocol, inputs, cells, "A0_ast_only", 7270, 0)
    return {
        "status": "PASS_FIRST_FIT",
        "locks": locks,
        "role_multiplicity": multiplicity,
        "fit_summary_sha256": sha256(expected),
        "fit": {key: value for key, value in audit.items() if key != "orders"},
        "remaining_fits_authorizable": 59,
        "score_based_decision": False,
        "saved_public_cat_macro_f1_context_only": fit["metrics"][
            "public_cat_probability_mean"
        ]["macro_f1"],
    }


def verify_full() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    locks = verify_locks(protocol)
    inputs = load_inputs(protocol)
    roles = read_json(ROOT / protocol["splits"]["roles_path"])
    cells, multiplicity = validate_roles(
        protocol, roles, set(inputs["ast"]["cat_ids"].astype(str))
    )
    expected_paths = {
        fit_path(pipeline, seed, fold)
        for pipeline in PIPELINES
        for seed in protocol["splits"]["seeds"]
        for fold in range(4)
    }
    actual_paths = set((RUN_ROOT / "fits").rglob("fit_summary.json"))
    if actual_paths != expected_paths:
        raise RuntimeError("IDEA-088 fit set is not the locked 60-cell matrix")
    fits: dict[tuple[str, int, int], dict[str, Any]] = {}
    audits: dict[tuple[str, int, int], dict[str, Any]] = {}
    maximum_metric_difference = 0.0
    for pipeline in PIPELINES:
        for seed in protocol["splits"]["seeds"]:
            for fold in range(4):
                fit, audit = verify_fit(protocol, inputs, cells, pipeline, seed, fold)
                fits[(pipeline, seed, fold)] = fit
                audits[(pipeline, seed, fold)] = audit
                maximum_metric_difference = max(
                    maximum_metric_difference, float(audit["metric_max_abs_difference"])
                )

    paired_initial_states = 0
    paired_order_prefixes = 0
    for seed in protocol["splits"]["seeds"]:
        for fold in range(4):
            a0 = audits[("A0_ast_only", seed, fold)]
            c1 = audits[("C1_bounded_wide_additive", seed, fold)]
            if a0["initial_common_state_sha256"] != c1["initial_common_state_sha256"]:
                raise RuntimeError(f"IDEA-088 A0/C1 common initialization differs: {seed}/{fold}")
            paired_initial_states += 1
            common_epochs = min(len(a0["orders"]), len(c1["orders"]))
            np.testing.assert_array_equal(a0["orders"][:common_epochs], c1["orders"][:common_epochs])
            np.testing.assert_array_equal(
                np.asarray(a0["class_weights"]), np.asarray(c1["class_weights"])
            )
            paired_order_prefixes += 1

    for pipeline in PIPELINES:
        for seed in protocol["splits"]["seeds"]:
            predicted_counts = Counter(
                cat
                for fold in range(4)
                for cat in audits[(pipeline, seed, fold)]["test_cat_ids"]
            )
            expected_counts = Counter(
                cat for fold in range(4) for cat in cells[(seed, fold)]["test"]
            )
            if predicted_counts != expected_counts:
                raise RuntimeError(
                    f"IDEA-088 saved prediction multiplicity differs: {pipeline}/{seed}"
                )

    rebuilt = rebuild_aggregate(protocol, fits)
    rebuilt_class_recalls = rebuild_class_recall_summary(protocol, fits)
    aggregate = read_json(AGGREGATE_PATH)
    aggregate_difference = compare_nested(
        rebuilt, aggregate, "aggregate", PREDICTION_TOLERANCE
    )
    if aggregate.get("fits") != 60 or aggregate.get("claims_complete_oof") is not False:
        raise RuntimeError("IDEA-088 aggregate claim or fit count changed")
    metadata = read_json(METADATA_PATH)
    metadata_difference = compare_nested(aggregate, metadata, "metadata")
    if metadata["run_summary_sha256"] != sha256(AGGREGATE_PATH):
        raise RuntimeError("IDEA-088 metadata run-summary hash differs")
    protection = protection_snapshot()
    return {
        "status": "PASS_FULL",
        "locks": locks,
        "fits_verified": len(fits),
        "fit_artifact_hashes_verified": len(fits) * 6,
        "paired_A0_C1_common_initial_states": paired_initial_states,
        "paired_A0_C1_order_prefixes": paired_order_prefixes,
        "test_prediction_calls_per_fit": 1,
        "role_multiplicity": multiplicity,
        "maximum_fit_metric_abs_difference": maximum_metric_difference,
        "maximum_aggregate_abs_difference": aggregate_difference,
        "maximum_metadata_abs_difference": metadata_difference,
        "aggregate_sha256": sha256(AGGREGATE_PATH),
        "metadata_sha256": sha256(METADATA_PATH),
        "historical_protection": protection,
        "rebuilt_results": rebuilt,
        "independent_class_recall_summary": rebuilt_class_recalls,
    }


def main() -> None:
    args = parse_args()
    if args.mode == "first-fit":
        result = verify_first_fit()
    elif args.mode == "full":
        result = verify_full()
    else:
        result = {"status": "PASS_PROTECTION", "historical_protection": protection_snapshot()}
    if args.output:
        output_path = (ROOT / args.output).resolve()
        if ROOT.resolve() not in output_path.parents or "088" not in output_path.as_posix():
            raise RuntimeError("IDEA-088 verifier output must stay in an 088-named workspace path")
        write_json(output_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
