#!/usr/bin/env python3
"""Independent, read-only verifier for the completed IDEA-075 benchmark.

The verifier deliberately does not import the training runner.  It reads the
locked inputs and completed artifacts, validates their structure and hashes,
recomputes metrics and gates from the per-fit prediction CSV files, and emits
one JSON document to stdout.  It never writes into the run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


PIPELINES = (
    "A0_ast_only",
    "U1_unbounded_age_residual",
    "C1_bounded_age_residual",
)
CLASS_NAMES = ("puppy", "juvenile", "adolescent", "adult", "senior")
BASE_SEEDS = (8075, 4270, 1872)
SPLIT_SEEDS = (17, 43, 101)
PROBABILITY_COLUMNS = tuple(f"prob_{name}" for name in CLASS_NAMES)
DUMMY_COLUMNS = tuple(f"dummy_prob_{name}" for name in CLASS_NAMES)
EXPECTED_PARAMETERS = {
    "A0_ast_only": 99_333,
    "U1_unbounded_age_residual": 108_401,
    "C1_bounded_age_residual": 108_401,
}
EXPECTED_FITS = 135
EXPECTED_OBSERVATIONS = 2290
EXPECTED_DOGS = 125
TOLERANCE = 1.0e-12


class AuditError(RuntimeError):
    """Raised when a locked IDEA-075 invariant does not hold."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def equal_array(left: Iterable[Any], right: Iterable[Any]) -> bool:
    return np.array_equal(np.asarray(list(left)), np.asarray(list(right)))


def sign_counts(values: Sequence[float], tolerance: float = TOLERANCE) -> dict[str, int]:
    array = np.asarray(values, dtype=float)
    return {
        "positive": int((array > tolerance).sum()),
        "tie": int((np.abs(array) <= tolerance).sum()),
        "negative": int((array < -tolerance).sum()),
    }


def class_recalls(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    recalls = []
    for label in range(len(CLASS_NAMES)):
        mask = labels == label
        recalls.append(float(np.mean(predictions[mask] == label)) if mask.any() else 0.0)
    return np.asarray(recalls, dtype=float)


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for label in range(len(CLASS_NAMES)):
        true_positive = int(np.sum((labels == label) & (predictions == label)))
        false_positive = int(np.sum((labels != label) & (predictions == label)))
        false_negative = int(np.sum((labels == label) & (predictions != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return float(np.mean(scores))


def metric_bundle(
    frame: pd.DataFrame, probability_columns: Sequence[str] = PROBABILITY_COLUMNS
) -> dict[str, Any]:
    probabilities = frame[list(probability_columns)].to_numpy(float)
    labels = frame["true_label"].to_numpy(np.int64)
    predictions = probabilities.argmax(axis=1)
    require(np.isfinite(probabilities).all(), "Non-finite prediction probability")
    require(bool(np.all(probabilities >= 0.0)), "Negative prediction probability")
    require(
        bool(np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-5, rtol=0.0)),
        "Prediction rows do not sum to one",
    )
    clipped_true = np.clip(
        probabilities[np.arange(len(labels)), labels], 1.0e-7, 1.0
    )
    targets = np.eye(len(CLASS_NAMES), dtype=float)[labels]
    recalls = class_recalls(labels, predictions)
    correct = (predictions == labels).astype(float)
    dog_accuracy = (
        pd.DataFrame({"dog_id": frame["dog_id"].astype(str), "correct": correct})
        .groupby("dog_id", sort=False)["correct"]
        .mean()
        .mean()
    )
    grouped_input = frame[["dog_id", "true_label"]].copy()
    grouped_input["_row"] = [row for row in probabilities]
    grouped = (
        grouped_input.groupby(["dog_id", "true_label"], sort=False)["_row"]
        .apply(lambda values: np.mean(np.stack(values), axis=0))
        .reset_index()
    )
    grouped_probabilities = np.stack(grouped["_row"])
    grouped_labels = grouped["true_label"].to_numpy(np.int64)
    grouped_predictions = grouped_probabilities.argmax(axis=1)
    return {
        "macro_f1": macro_f1(labels, predictions),
        "balanced_accuracy": float(recalls.mean()),
        "cross_entropy": float(np.mean(-np.log(clipped_true))),
        "multiclass_brier": float(
            np.mean(np.sum((probabilities - targets) ** 2, axis=1))
        ),
        "per_class_recall": {
            name: float(recalls[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "mean_per_dog_bark_unit_accuracy": float(dog_accuracy),
        "dog_age_group_macro_f1": macro_f1(grouped_labels, grouped_predictions),
        "dog_age_group_balanced_accuracy": float(
            class_recalls(grouped_labels, grouped_predictions).mean()
        ),
        "bark_units": int(len(frame)),
        "dogs": int(frame["dog_id"].nunique()),
        "dog_age_group_cells": int(len(grouped)),
    }


def flat_metrics(prefix: str, bundle: dict[str, Any]) -> dict[str, Any]:
    result = {
        f"{prefix}_macro_f1": bundle["macro_f1"],
        f"{prefix}_balanced_accuracy": bundle["balanced_accuracy"],
        f"{prefix}_cross_entropy": bundle["cross_entropy"],
        f"{prefix}_multiclass_brier": bundle["multiclass_brier"],
        f"{prefix}_mean_per_dog_bark_unit_accuracy": bundle[
            "mean_per_dog_bark_unit_accuracy"
        ],
        f"{prefix}_dog_age_group_macro_f1": bundle["dog_age_group_macro_f1"],
        f"{prefix}_dog_age_group_balanced_accuracy": bundle[
            "dog_age_group_balanced_accuracy"
        ],
    }
    result.update(
        {
            f"{prefix}_recall_{name}": value
            for name, value in bundle["per_class_recall"].items()
        }
    )
    return result


def compare_frames(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    keys: Sequence[str],
    label: str,
) -> None:
    require(set(expected.columns) == set(actual.columns), f"{label} columns differ")
    expected = expected[list(actual.columns)].sort_values(list(keys)).reset_index(drop=True)
    actual = actual.sort_values(list(keys)).reset_index(drop=True)
    require(len(expected) == len(actual), f"{label} row count differs")
    for column in actual.columns:
        if pd.api.types.is_numeric_dtype(actual[column]):
            require(
                bool(
                    np.allclose(
                        expected[column].to_numpy(float),
                        actual[column].to_numpy(float),
                        atol=1.0e-12,
                        rtol=1.0e-12,
                        equal_nan=True,
                    )
                ),
                f"{label} numeric column differs: {column}",
            )
        else:
            require(
                equal_array(
                    expected[column].astype(str), actual[column].astype(str)
                ),
                f"{label} text column differs: {column}",
            )


def compare_nested(expected: Any, actual: Any, path: str = "root") -> None:
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{path} type differs")
        require(set(expected) == set(actual), f"{path} keys differ")
        for key in expected:
            compare_nested(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list), f"{path} type differs")
        require(len(expected) == len(actual), f"{path} length differs")
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare_nested(left, right, f"{path}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(isinstance(actual, (int, float)), f"{path} numeric type differs")
        require(
            math.isclose(float(expected), float(actual), abs_tol=1.0e-12, rel_tol=1.0e-12),
            f"{path} value differs: {expected!r} != {actual!r}",
        )
    else:
        require(expected == actual, f"{path} differs: {expected!r} != {actual!r}")


def validate_prepared_assets(
    repo_root: Path,
    protocol: dict[str, Any],
    run_manifest: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = protocol["data"]
    for name, artifact in run_manifest["prepared_artifacts"].items():
        path = repo_root / artifact["path"]
        require(path.is_file(), f"Missing prepared artifact: {name}")
        require(sha256(path) == artifact["sha256"], f"Prepared hash mismatch: {name}")

    manifest = pd.read_csv(
        repo_root / data["manifest_path"],
        dtype={"recording_id": str, "animal_id": str},
    )
    require(len(manifest) == EXPECTED_OBSERVATIONS, "Manifest size changed")
    require(not manifest["recording_id"].duplicated().any(), "Duplicate manifest ID")

    with np.load(repo_root / data["frozen_embedding_path"], allow_pickle=False) as frozen:
        frozen_ids = frozen["recording_ids"].astype(str)
        frozen_dogs = frozen["animal_ids"].astype(str)
        frozen_paths = frozen["source_paths"].astype(str)
        frozen_labels = frozen["labels"].astype(np.int64)
        label_names = tuple(frozen["label_names"].astype(str))
        embedding_shape = tuple(frozen["embeddings"].shape)
    label_map = {name: index for index, name in enumerate(CLASS_NAMES)}
    manifest_labels = manifest["label_name"].map(label_map).to_numpy(np.int64)
    require(equal_array(manifest["recording_id"].astype(str), frozen_ids), "Manifest/AST ID order differs")
    require(equal_array(manifest["animal_id"].astype(str), frozen_dogs), "Manifest/AST dog order differs")
    require(equal_array(manifest["audio_path"].astype(str), frozen_paths), "Manifest/AST path order differs")
    require(equal_array(manifest_labels, frozen_labels), "Manifest/AST labels differ")
    require(label_names == CLASS_NAMES, "AST class order differs")
    require(embedding_shape == (EXPECTED_OBSERVATIONS, 768), "AST shape differs")

    with np.load(repo_root / data["age_feature_path"], allow_pickle=False) as loaded:
        features = loaded["features"].astype(np.float32)
        feature_names = tuple(loaded["feature_names"].astype(str))
        feature_ids = loaded["recording_ids"].astype(str)
        feature_dogs = loaded["dog_ids"].astype(str)
        analysis_frames = loaded["analysis_frame_counts"].astype(np.int64)
        voiced_frames = loaded["voiced_f0_frame_counts"].astype(np.int64)
    require(feature_names == tuple(protocol["acoustic_features"]["feature_names"]), "Feature names differ")
    require(features.shape == (EXPECTED_OBSERVATIONS, 20), "Feature shape differs")
    require(not np.isinf(features).any(), "Feature matrix contains infinity")
    require(equal_array(feature_ids, frozen_ids), "Feature/AST ID order differs")
    require(equal_array(feature_dogs, frozen_dogs), "Feature/AST dog order differs")

    roles = pd.read_csv(
        repo_root / data["roles_path"],
        dtype={"recording_id": str, "dog_id": str},
    )
    require(len(roles) == 15 * EXPECTED_OBSERVATIONS, "Role table size differs")
    role_groups = list(roles.groupby(["repeat", "outer_fold"], sort=True))
    require(len(role_groups) == 15, "Role table does not contain 15 cells")
    for (repeat, fold), frame in role_groups:
        require(len(frame) == EXPECTED_OBSERVATIONS, "Role cell size differs")
        require(equal_array(frame["recording_id"].astype(str), frozen_ids), "Role/AST IDs differ")
        require(equal_array(frame["dog_id"].astype(str), frozen_dogs), "Role/AST dogs differ")
        require(
            equal_array(frame["age_group"].map(label_map).to_numpy(np.int64), frozen_labels),
            "Role/AST labels differ",
        )
        require(int(frame["outer_split_seed"].iloc[0]) == SPLIT_SEEDS[int(repeat)], "Outer split seed differs")
        require(int(frame["inner_seed"].iloc[0]) == SPLIT_SEEDS[int(repeat)] + 1000 + int(fold), "Inner split seed differs")
        dog_sets = {
            role: set(frame.loc[frame["role"] == role, "dog_id"].astype(str))
            for role in ("fit", "validation", "test")
        }
        require(not (dog_sets["fit"] & dog_sets["validation"]), "Fit/validation dog leakage")
        require(not (dog_sets["fit"] & dog_sets["test"]), "Fit/test dog leakage")
        require(not (dog_sets["validation"] & dog_sets["test"]), "Validation/test dog leakage")
        for role in dog_sets:
            require(
                set(frame.loc[frame["role"] == role, "age_group"].astype(str))
                == set(CLASS_NAMES),
                f"Role {role} misses a class",
            )

    oof = pd.read_csv(
        repo_root / data["idea067_oof_path"],
        dtype={"recording_id": str, "animal_id": str},
    )
    require(
        set(map(tuple, oof[["repeat", "seed"]].drop_duplicates().to_numpy()))
        == {(index, seed) for index, seed in enumerate(SPLIT_SEEDS)},
        "IDEA-067 repeat/seed mapping differs",
    )
    for repeat in range(3):
        by_id = oof[oof["repeat"] == repeat].set_index("recording_id")
        require(by_id.index.is_unique, "Duplicate IDEA-067 OOF recording ID")
        require(set(by_id.index) == set(frozen_ids), "IDEA-067 OOF coverage differs")
        for fold in range(5):
            frame = roles[(roles["repeat"] == repeat) & (roles["outer_fold"] == fold)]
            expected_test = by_id.loc[frozen_ids, "fold"].to_numpy(np.int64) == fold
            require(
                equal_array(frame["role"].to_numpy(str) == "test", expected_test),
                "Locked outer-test roles differ from IDEA-067 OOF",
            )

    inventory = pd.read_csv(
        repo_root / data["source_audio_inventory_path"],
        dtype={"recording_id": str, "dog_id": str, "sha256": str},
    )
    require(len(inventory) == EXPECTED_OBSERVATIONS, "Audio inventory size differs")
    require(equal_array(inventory["recording_id"], manifest["recording_id"]), "Inventory ID order differs")
    require(equal_array(inventory["dog_id"], manifest["animal_id"]), "Inventory dog order differs")
    require(equal_array(inventory["audio_path"], manifest["audio_path"]), "Inventory path order differs")
    require(not inventory["sha256"].duplicated().any(), "Duplicate source-audio hash")
    canonical_digest = hashlib.sha256()
    for row in inventory.itertuples(index=False):
        audio_path = repo_root / str(row.audio_path)
        require(audio_path.is_file(), f"Missing source audio: {row.audio_path}")
        actual_hash = sha256(audio_path)
        require(actual_hash == str(row.sha256), f"Source audio hash differs: {row.recording_id}")
        with wave.open(str(audio_path), "rb") as handle:
            require(handle.getframerate() == 16000, f"Audio sample rate differs: {row.recording_id}")
            require(handle.getnchannels() == 1, f"Audio channels differ: {row.recording_id}")
            require(handle.getsampwidth() == 2, f"Audio sample width differs: {row.recording_id}")
            require(handle.getnframes() == int(row.frames), f"Audio frame count differs: {row.recording_id}")
        canonical_digest.update(str(row.recording_id).encode("utf-8"))
        canonical_digest.update(b"\0")
        canonical_digest.update(bytes.fromhex(actual_hash))
    canonical_hash = canonical_digest.hexdigest()
    require(canonical_hash == data["canonical_audio_content_sha256"], "Canonical audio hash differs")

    feature_summary = read_json(repo_root / data["feature_summary_path"])
    require(feature_summary["status"] == "complete", "Feature summary is incomplete")
    require(feature_summary["label_information_used_for_feature_extraction"] is False, "Feature extraction used labels")
    require(feature_summary["recordings"] == EXPECTED_OBSERVATIONS, "Feature summary recording count differs")
    require(feature_summary["dogs"] == EXPECTED_DOGS, "Feature summary dog count differs")
    require(feature_summary["feature_count"] == 20, "Feature summary dimension differs")
    missing = np.isnan(features).sum(axis=0)
    expected_missing = np.asarray(
        [feature_summary["missing_values_by_feature"][name] for name in feature_names]
    )
    require(equal_array(missing, expected_missing), "Feature missing counts differ")
    require(int(np.isfinite(features).all(axis=1).sum()) == feature_summary["fully_finite_recordings"], "Fully-finite feature count differs")
    require(
        [int(analysis_frames.min()), float(np.median(analysis_frames)), int(analysis_frames.max())]
        == feature_summary["analysis_frame_count_min_median_max"],
        "Analysis-frame summary differs",
    )
    require(
        [int(voiced_frames.min()), float(np.median(voiced_frames)), int(voiced_frames.max())]
        == feature_summary["voiced_f0_frame_count_min_median_max"],
        "Voiced-frame summary differs",
    )
    for name, key in (("roles", "roles_path"), ("inventory", "source_audio_inventory_path"), ("features", "age_feature_path")):
        require(
            sha256(repo_root / data[key]) == feature_summary["artifact_sha256"][name],
            f"Feature summary artifact hash differs: {name}",
        )

    return manifest, roles, {
        "recordings": EXPECTED_OBSERVATIONS,
        "dogs": EXPECTED_DOGS,
        "role_cells": 15,
        "source_audio_hashes_verified": EXPECTED_OBSERVATIONS,
        "unique_source_audio_hashes": int(inventory["sha256"].nunique()),
        "canonical_audio_content_sha256": canonical_hash,
        "ast_embedding_shape": list(embedding_shape),
        "age_feature_shape": list(features.shape),
        "fully_finite_age_feature_records": int(np.isfinite(features).all(axis=1).sum()),
        "records_with_age_feature_nan": int(np.isnan(features).any(axis=1).sum()),
    }


def validate_prediction_coverage(
    frame: pd.DataFrame,
    role_frame: pd.DataFrame,
    role: str,
    label_map: dict[str, int],
) -> pd.DataFrame:
    required = {"recording_index", "recording_id", "dog_id", "true_label", *PROBABILITY_COLUMNS}
    require(required.issubset(frame.columns), f"Prediction columns missing for role {role}")
    require(not frame["recording_id"].duplicated().any(), f"Duplicate prediction ID for role {role}")
    frame = frame.sort_values("recording_index").reset_index(drop=True)
    mask = role_frame["role"].to_numpy(str) == role
    expected_indices = np.flatnonzero(mask)
    expected = role_frame.iloc[expected_indices]
    require(equal_array(frame["recording_index"].astype(int), expected_indices), f"Prediction indices differ for role {role}")
    require(equal_array(frame["recording_id"].astype(str), expected["recording_id"].astype(str)), f"Prediction IDs differ for role {role}")
    require(equal_array(frame["dog_id"].astype(str), expected["dog_id"].astype(str)), f"Prediction dogs differ for role {role}")
    require(
        equal_array(
            frame["true_label"].astype(int),
            expected["age_group"].map(label_map).to_numpy(np.int64),
        ),
        f"Prediction labels differ for role {role}",
    )
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(float)
    require(np.isfinite(probabilities).all(), f"Non-finite probability for role {role}")
    require(bool(np.all(probabilities >= 0.0)), f"Negative probability for role {role}")
    require(
        bool(np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-5, rtol=0.0)),
        f"Probability sum differs for role {role}",
    )
    return frame


def validate_fits_and_load_predictions(
    repo_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    run_root: Path,
) -> tuple[
    dict[tuple[str, int, int, int], dict[str, Any]],
    dict[tuple[str, int, int, int], pd.DataFrame],
    dict[str, Any],
]:
    fit_index_path = run_root / "fit_index.csv"
    fit_index = pd.read_csv(fit_index_path)
    require(len(fit_index) == EXPECTED_FITS, "fit_index does not contain 135 rows")
    expected_keys = {
        (pipeline, seed, repeat, fold)
        for pipeline in PIPELINES
        for seed in BASE_SEEDS
        for repeat in range(3)
        for fold in range(5)
    }
    index_keys = set(
        map(
            tuple,
            fit_index[["pipeline", "base_seed", "repeat", "outer_fold"]].to_numpy(),
        )
    )
    require(index_keys == expected_keys, "fit_index identity set differs")
    label_map = {name: index for index, name in enumerate(CLASS_NAMES)}
    role_lookup = {
        (int(repeat), int(fold)): frame.reset_index(drop=True)
        for (repeat, fold), frame in roles.groupby(["repeat", "outer_fold"], sort=True)
    }
    summaries: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    predictions: dict[tuple[str, int, int, int], pd.DataFrame] = {}
    fit_summary_hashes = 0
    artifact_hashes = 0
    validation_rows = 0
    test_rows = 0
    common_epoch_count = 0
    full_seeds: set[int] = set()
    c1_ratio_means: list[float] = []
    c1_ratio_maxima: list[float] = []
    c1_ratio_counts: list[int] = []
    max_reload_difference = 0.0
    max_initial_logit_difference = 0.0

    artifact_names = {
        "outer_test_predictions": "outer_test_predictions.csv",
        "best_validation_predictions": "best_validation_predictions.csv",
        "checkpoint": "best_checkpoint.pt",
        "pre_test_summary": "pre_test_summary.json",
        "outer_test_access_marker": "outer_test_access_marker.json",
        "outer_test_audit": "outer_test_audit.json",
    }

    for row in fit_index.itertuples(index=False):
        pipeline = str(row.pipeline)
        seed = int(row.base_seed)
        repeat = int(row.repeat)
        fold = int(row.outer_fold)
        key = (pipeline, seed, repeat, fold)
        expected_dir = run_root / "fits" / pipeline / f"seed_{seed}" / f"repeat_{repeat}" / f"fold_{fold}"
        summary_path = repo_root / str(row.fit_summary_path)
        require(summary_path.resolve() == (expected_dir / "fit_summary.json").resolve(), f"fit_index path differs: {key}")
        require(summary_path.is_file(), f"Missing fit summary: {key}")
        require(sha256(summary_path) == str(row.fit_summary_sha256), f"fit summary hash differs: {key}")
        fit_summary_hashes += 1
        summary = read_json(summary_path)
        summaries[key] = summary
        expected_identity = {
            "pipeline": pipeline,
            "base_seed": seed,
            "repeat": repeat,
            "outer_fold": fold,
            "full_seed": seed + 10_000 * repeat + 100 * fold,
        }
        require(summary.get("status") == "complete", f"Fit is incomplete: {key}")
        require(int(summary.get("outer_test_access_count", -1)) == 1, f"Outer-test count differs: {key}")
        for name, value in expected_identity.items():
            require(summary.get(name) == value, f"Fit identity differs ({name}): {key}")
        require(
            int(summary["training_seed"])
            == expected_identity["full_seed"] + int(protocol["fixed_training"]["post_build_seed_offset"]),
            f"Training seed differs: {key}",
        )
        full_seeds.add(expected_identity["full_seed"])
        for stem, filename in artifact_names.items():
            artifact_path = repo_root / summary[f"{stem}_path"]
            require(artifact_path.resolve() == (expected_dir / filename).resolve(), f"Artifact path differs ({stem}): {key}")
            require(artifact_path.is_file(), f"Artifact missing ({stem}): {key}")
            require(sha256(artifact_path) == summary[f"{stem}_sha256"], f"Artifact hash differs ({stem}): {key}")
            artifact_hashes += 1
        require(not (expected_dir / "outer_test_predictions.csv.tmp").exists(), f"Stale test temp file: {key}")

        pretest = read_json(expected_dir / "pre_test_summary.json")
        marker = read_json(expected_dir / "outer_test_access_marker.json")
        test_audit = read_json(expected_dir / "outer_test_audit.json")
        require(pretest.get("status") == "ready_for_single_outer_test_access", f"Pre-test status differs: {key}")
        require(marker.get("status") == "started", f"Access marker status differs: {key}")
        for name, value in expected_identity.items():
            require(pretest.get(name) == value, f"Pre-test identity differs: {key}")
            require(marker.get(name) == value, f"Marker identity differs: {key}")
        for name, value in pretest.items():
            if name == "status":
                continue
            require(summary.get(name) == value, f"Final/pre-test summary differs ({name}): {key}")
        require(marker["pre_test_summary_sha256"] == sha256(expected_dir / "pre_test_summary.json"), f"Marker/pre-test hash link differs: {key}")
        require(test_audit["outer_test_predictions_sha256"] == sha256(expected_dir / "outer_test_predictions.csv"), f"Audit/test hash link differs: {key}")
        compare_nested(
            test_audit["model_audit_on_outer_test"],
            summary["model_audit_on_outer_test"],
            f"model_audit[{key}]",
        )

        initialization = summary["initialization_audit"]
        compare_nested(EXPECTED_PARAMETERS, initialization["trainable_parameters"], f"parameters[{key}]")
        require(initialization["shared_state_equal"] is True, f"Shared initialization differs: {key}")
        require(initialization["U1_C1_full_state_equal"] is True, f"U1/C1 initialization differs: {key}")
        initial_differences = [float(value) for value in initialization["max_logit_difference_vs_A0"].values()]
        max_initial_logit_difference = max(max_initial_logit_difference, *initial_differences)
        require(all(value == 0.0 for value in initial_differences), f"Initial logits differ: {key}")
        reload_difference = float(summary["checkpoint_reload_max_probability_difference"])
        max_reload_difference = max(max_reload_difference, reload_difference)
        require(reload_difference == 0.0, f"Checkpoint reload differs: {key}")
        model_audit = summary["model_audit_on_outer_test"]
        require(int(model_audit["trainable_parameters"]) == EXPECTED_PARAMETERS[pipeline], f"Model parameter audit differs: {key}")
        if pipeline == PIPELINES[2]:
            require(int(model_audit["maximum_dimension_budget_violations"]) == 0, f"C1 budget violation: {key}")
            ratio = model_audit["residual_to_hidden_norm_ratio"]
            c1_ratio_means.append(float(ratio["mean"]))
            c1_ratio_maxima.append(float(ratio["max"]))
            c1_ratio_counts.append(int(ratio["count"]))

        role_frame = role_lookup[(repeat, fold)]
        validation = pd.read_csv(
            expected_dir / "best_validation_predictions.csv",
            dtype={"recording_id": str, "dog_id": str},
        )
        validation = validate_prediction_coverage(validation, role_frame, "validation", label_map)
        validation_rows += len(validation)
        test = pd.read_csv(
            expected_dir / "outer_test_predictions.csv",
            dtype={"recording_id": str, "dog_id": str},
        )
        test = validate_prediction_coverage(test, role_frame, "test", label_map)
        require(set(DUMMY_COLUMNS).issubset(test.columns), f"Dummy columns missing: {key}")
        dummy = test[list(DUMMY_COLUMNS)].to_numpy(float)
        require(np.isfinite(dummy).all(), f"Dummy probabilities non-finite: {key}")
        require(bool(np.allclose(dummy.sum(axis=1), 1.0, atol=1.0e-12, rtol=0.0)), f"Dummy probabilities invalid: {key}")
        outer_train = role_frame[role_frame["role"] != "test"]
        counts = np.bincount(
            outer_train["age_group"].map(label_map).to_numpy(np.int64), minlength=5
        ).astype(float)
        expected_dummy = counts / counts.sum()
        require(bool(np.allclose(dummy, expected_dummy, atol=1.0e-15, rtol=0.0)), f"Dummy prior differs: {key}")
        require(bool(np.allclose(np.asarray(summary["dummy_probabilities"], dtype=float), expected_dummy, atol=1.0e-15, rtol=0.0)), f"Summary dummy prior differs: {key}")
        expected_role_counts = {
            "fit_bark_units": int((role_frame["role"] == "fit").sum()),
            "validation_bark_units": int((role_frame["role"] == "validation").sum()),
            "test_bark_units": int((role_frame["role"] == "test").sum()),
            "fit_dogs": int(role_frame.loc[role_frame["role"] == "fit", "dog_id"].nunique()),
            "validation_dogs": int(role_frame.loc[role_frame["role"] == "validation", "dog_id"].nunique()),
            "test_dogs": int(role_frame.loc[role_frame["role"] == "test", "dog_id"].nunique()),
        }
        for name, value in expected_role_counts.items():
            require(int(summary[name]) == value, f"Fit role count differs ({name}): {key}")
        test_rows += len(test)
        predictions[key] = test

    require(len(full_seeds) == 45, "Full training seeds are not unique")
    for seed in BASE_SEEDS:
        for repeat in range(3):
            for fold in range(5):
                cell = [summaries[(pipeline, seed, repeat, fold)] for pipeline in PIPELINES]
                common_epochs = min(len(summary["history"]) for summary in cell)
                common_epoch_count += common_epochs
                for epoch in range(common_epochs):
                    dog_hashes = {
                        summary["history"][epoch]["train_audit"]["dog_order_sha256"]
                        for summary in cell
                    }
                    coverage_hashes = {
                        summary["history"][epoch]["train_audit"]["recording_coverage_sha256"]
                        for summary in cell
                    }
                    require(len(dog_hashes) == 1, f"Paired dog order differs: {(seed, repeat, fold, epoch)}")
                    require(len(coverage_hashes) == 1, f"Paired recording coverage differs: {(seed, repeat, fold, epoch)}")

    tmp_files = list((run_root / "fits").rglob("*.tmp"))
    require(not tmp_files, "Temporary fit artifacts remain")
    return summaries, predictions, {
        "fits": len(summaries),
        "fit_summary_hashes_verified": fit_summary_hashes,
        "six_artifact_hashes_verified": artifact_hashes,
        "outer_test_access_count_sum": int(sum(summary["outer_test_access_count"] for summary in summaries.values())),
        "outer_test_prediction_rows": test_rows,
        "validation_prediction_rows": validation_rows,
        "temporary_files": len(tmp_files),
        "unique_full_training_seeds": len(full_seeds),
        "paired_cells": 45,
        "paired_common_epochs_verified": common_epoch_count,
        "maximum_initial_logit_difference": max_initial_logit_difference,
        "maximum_checkpoint_reload_probability_difference": max_reload_difference,
        "parameters": EXPECTED_PARAMETERS,
        "C1_budget_violations": int(
            max(
                summary["model_audit_on_outer_test"]["maximum_dimension_budget_violations"]
                for key, summary in summaries.items()
                if key[0] == PIPELINES[2]
            )
        ),
        "C1_residual_ratio": {
            "fit_mean_average": float(np.mean(c1_ratio_means)),
            "fit_mean_min": float(np.min(c1_ratio_means)),
            "fit_mean_max": float(np.max(c1_ratio_means)),
            "maximum_single_bark_unit": float(np.max(c1_ratio_maxima)),
            "bark_unit_occurrences": int(np.sum(c1_ratio_counts)),
        },
    }


def recompute_results(
    predictions: dict[tuple[str, int, int, int], pd.DataFrame],
    summaries: dict[tuple[str, int, int, int], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    fold_rows: list[dict[str, Any]] = []
    for key in sorted(predictions):
        pipeline, seed, repeat, fold = key
        fold_rows.append(
            {
                "pipeline": pipeline,
                "base_seed": seed,
                "repeat": repeat,
                "outer_fold": fold,
                **flat_metrics("unit", metric_bundle(predictions[key])),
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)

    pooled_frames: dict[tuple[str, int, int], pd.DataFrame] = {}
    seed_repeat_rows: list[dict[str, Any]] = []
    for pipeline in PIPELINES:
        for seed in BASE_SEEDS:
            for repeat in range(3):
                pooled = (
                    pd.concat(
                        [predictions[(pipeline, seed, repeat, fold)] for fold in range(5)],
                        ignore_index=True,
                    )
                    .sort_values("recording_index")
                    .reset_index(drop=True)
                )
                require(len(pooled) == EXPECTED_OBSERVATIONS, "Pooled OOF size differs")
                require(not pooled["recording_id"].duplicated().any(), "Pooled OOF IDs duplicate")
                pooled_frames[(pipeline, seed, repeat)] = pooled
                row = {
                    "pipeline": pipeline,
                    "base_seed": seed,
                    "repeat": repeat,
                    **flat_metrics("unit", metric_bundle(pooled)),
                }
                if pipeline == PIPELINES[2]:
                    row.update(flat_metrics("dummy", metric_bundle(pooled, DUMMY_COLUMNS)))
                seed_repeat_rows.append(row)
    seed_repeat = pd.DataFrame(seed_repeat_rows)

    comparisons = {
        "C1_minus_A0": (PIPELINES[2], PIPELINES[0]),
        "U1_minus_A0": (PIPELINES[1], PIPELINES[0]),
        "C1_minus_U1": (PIPELINES[2], PIPELINES[1]),
    }
    delta_rows: list[dict[str, Any]] = []
    dog_rows: list[dict[str, Any]] = []
    metrics_index = seed_repeat.set_index(["pipeline", "base_seed", "repeat"])
    for name, (candidate, reference) in comparisons.items():
        for seed in BASE_SEEDS:
            for repeat in range(3):
                left_row = metrics_index.loc[(candidate, seed, repeat)]
                right_row = metrics_index.loc[(reference, seed, repeat)]
                delta_rows.append(
                    {
                        "comparison": name,
                        "base_seed": seed,
                        "repeat": repeat,
                        "macro_f1_delta": float(left_row.unit_macro_f1 - right_row.unit_macro_f1),
                        "balanced_accuracy_delta": float(left_row.unit_balanced_accuracy - right_row.unit_balanced_accuracy),
                        "cross_entropy_delta": float(left_row.unit_cross_entropy - right_row.unit_cross_entropy),
                        "multiclass_brier_delta": float(left_row.unit_multiclass_brier - right_row.unit_multiclass_brier),
                        "senior_recall_delta": float(left_row.unit_recall_senior - right_row.unit_recall_senior),
                    }
                )
                left = pooled_frames[(candidate, seed, repeat)].set_index("recording_id")
                right = pooled_frames[(reference, seed, repeat)].set_index("recording_id")
                right = right.loc[left.index]
                for dog_id, ids in left.groupby("dog_id", sort=False).groups.items():
                    left_dog = left.loc[ids]
                    right_dog = right.loc[ids]
                    labels = left_dog["true_label"].to_numpy(np.int64)
                    left_probability = left_dog[list(PROBABILITY_COLUMNS)].to_numpy(float)
                    right_probability = right_dog[list(PROBABILITY_COLUMNS)].to_numpy(float)
                    left_ce = np.mean(
                        -np.log(
                            np.clip(
                                left_probability[np.arange(len(labels)), labels],
                                1.0e-7,
                                1.0,
                            )
                        )
                    )
                    right_ce = np.mean(
                        -np.log(
                            np.clip(
                                right_probability[np.arange(len(labels)), labels],
                                1.0e-7,
                                1.0,
                            )
                        )
                    )
                    dog_rows.append(
                        {
                            "comparison": name,
                            "base_seed": seed,
                            "repeat": repeat,
                            "dog_id": dog_id,
                            "cross_entropy_delta_candidate_minus_reference": float(left_ce - right_ce),
                            "accuracy_delta_candidate_minus_reference": float(
                                np.mean(left_probability.argmax(axis=1) == labels)
                                - np.mean(right_probability.argmax(axis=1) == labels)
                            ),
                        }
                    )
    deltas = pd.DataFrame(delta_rows)
    dog_deltas = pd.DataFrame(dog_rows)

    split_rows: list[dict[str, Any]] = []
    fold_index = fold_metrics.set_index(
        ["pipeline", "base_seed", "repeat", "outer_fold"]
    )
    for name, (candidate, reference) in comparisons.items():
        for repeat in range(3):
            for fold in range(5):
                values = [
                    float(
                        fold_index.loc[(candidate, seed, repeat, fold)].unit_macro_f1
                        - fold_index.loc[(reference, seed, repeat, fold)].unit_macro_f1
                    )
                    for seed in BASE_SEEDS
                ]
                split_rows.append(
                    {
                        "comparison": name,
                        "repeat": repeat,
                        "outer_fold": fold,
                        "macro_f1_delta_mean_across_base_seeds": float(np.mean(values)),
                    }
                )
    split_cells = pd.DataFrame(split_rows)

    comparison_summary: dict[str, Any] = {}
    base_seed_tables: dict[str, dict[str, float]] = {}
    for name in comparisons:
        current = deltas[deltas["comparison"] == name]
        base_means = current.groupby("base_seed")["macro_f1_delta"].mean()
        cells = split_cells[split_cells["comparison"] == name][
            "macro_f1_delta_mean_across_base_seeds"
        ]
        dog_means = dog_deltas[dog_deltas["comparison"] == name].groupby(
            "dog_id", sort=True
        )[
            [
                "cross_entropy_delta_candidate_minus_reference",
                "accuracy_delta_candidate_minus_reference",
            ]
        ].mean()
        comparison_summary[name] = {
            "mean_macro_f1_delta": float(current["macro_f1_delta"].mean()),
            "base_seed_macro_f1_deltas": {
                str(key): float(value) for key, value in base_means.items()
            },
            "seed_repeat_macro_f1_signs": sign_counts(current["macro_f1_delta"]),
            "split_cell_macro_f1_signs": sign_counts(cells),
            "worst_split_cell_macro_f1_delta": float(cells.min()),
            "mean_balanced_accuracy_delta": float(
                current["balanced_accuracy_delta"].mean()
            ),
            "mean_cross_entropy_delta": float(
                current["cross_entropy_delta"].mean()
            ),
            "mean_multiclass_brier_delta": float(
                current["multiclass_brier_delta"].mean()
            ),
            "per_dog_cross_entropy_delta_signs_candidate_minus_reference": sign_counts(
                dog_means["cross_entropy_delta_candidate_minus_reference"]
            ),
            "per_dog_accuracy_delta_signs_candidate_minus_reference": sign_counts(
                dog_means["accuracy_delta_candidate_minus_reference"]
            ),
            "per_dog_count": int(len(dog_means)),
        }
        base_seed_tables[name] = {
            str(key): float(value) for key, value in base_means.items()
        }

    all_metric_columns = [
        column for column in seed_repeat.columns if column.startswith("unit_")
    ]
    pipeline_mean_all_metrics = {
        pipeline: {
            column: float(
                seed_repeat.loc[seed_repeat["pipeline"] == pipeline, column].mean()
            )
            for column in all_metric_columns
        }
        for pipeline in PIPELINES
    }
    runner_metric_names = (
        "unit_macro_f1",
        "unit_balanced_accuracy",
        "unit_cross_entropy",
        "unit_multiclass_brier",
        "unit_recall_senior",
        "unit_mean_per_dog_bark_unit_accuracy",
        "unit_dog_age_group_macro_f1",
    )
    pipeline_mean_metrics = {
        pipeline: {
            name: pipeline_mean_all_metrics[pipeline][name]
            for name in runner_metric_names
        }
        for pipeline in PIPELINES
    }

    c1_a0 = deltas[deltas["comparison"] == "C1_minus_A0"]
    c1_u1 = deltas[deltas["comparison"] == "C1_minus_U1"]
    c1_a0_base = c1_a0.groupby("base_seed")["macro_f1_delta"].mean()
    c1_u1_base = c1_u1.groupby("base_seed")["macro_f1_delta"].mean()
    c1_a0_cells = split_cells[split_cells["comparison"] == "C1_minus_A0"][
        "macro_f1_delta_mean_across_base_seeds"
    ]
    c1_metrics = seed_repeat[seed_repeat["pipeline"] == PIPELINES[2]]
    a0_metrics = seed_repeat[seed_repeat["pipeline"] == PIPELINES[0]]
    u1_metrics = seed_repeat[seed_repeat["pipeline"] == PIPELINES[1]]
    senior_by_seed = {
        str(seed): float(
            c1_a0[c1_a0["base_seed"] == seed]["senior_recall_delta"].mean()
        )
        for seed in BASE_SEEDS
    }
    tier1_checks = {
        "mean_macro_f1_gain_at_least_0_005": float(c1_a0["macro_f1_delta"].mean())
        >= 0.005,
        "all_base_seed_means_strictly_positive": bool((c1_a0_base > 0).all()),
        "at_least_6_of_9_seed_repeats_positive": int(
            (c1_a0["macro_f1_delta"] > 0).sum()
        )
        >= 6,
        "at_least_10_of_15_split_cells_nonnegative": int((c1_a0_cells >= 0).sum())
        >= 10,
        "worst_split_cell_at_least_minus_0_03": float(c1_a0_cells.min()) >= -0.03,
        "mean_CE_not_worse": float(c1_metrics["unit_cross_entropy"].mean())
        <= float(a0_metrics["unit_cross_entropy"].mean()),
        "mean_Brier_not_worse": float(c1_metrics["unit_multiclass_brier"].mean())
        <= float(a0_metrics["unit_multiclass_brier"].mean()),
        "mean_balanced_accuracy_delta_nonnegative": float(
            c1_a0["balanced_accuracy_delta"].mean()
        )
        >= 0.0,
        "each_base_seed_senior_recall_delta_at_least_minus_0_02": all(
            value >= -0.02 for value in senior_by_seed.values()
        ),
    }
    tier2_checks = {
        "mean_C1_minus_U1_macro_f1_strictly_positive": float(
            c1_u1["macro_f1_delta"].mean()
        )
        > 0.0,
        "at_least_2_of_3_base_seed_means_strictly_positive": int(
            (c1_u1_base > 0).sum()
        )
        >= 2,
        "at_least_5_of_9_seed_repeats_strictly_positive": int(
            (c1_u1["macro_f1_delta"] > 0).sum()
        )
        >= 5,
        "mean_CE_not_worse_than_U1": float(
            c1_metrics["unit_cross_entropy"].mean()
        )
        <= float(u1_metrics["unit_cross_entropy"].mean()),
        "mean_Brier_not_worse_than_U1": float(
            c1_metrics["unit_multiclass_brier"].mean()
        )
        <= float(u1_metrics["unit_multiclass_brier"].mean()),
    }
    dummy_deltas = (
        c1_metrics["unit_macro_f1"].to_numpy(float)
        - c1_metrics["dummy_macro_f1"].to_numpy(float)
    )
    tier3_checks = {
        "C1_above_outer_train_prior_dummy_in_all_9_seed_repeats": bool(
            (dummy_deltas > 0).all()
        )
    }
    gate = {
        "tier1_C1_package_vs_A0": {
            "passed": all(tier1_checks.values()),
            "checks": tier1_checks,
            "senior_recall_delta_by_base_seed": senior_by_seed,
        },
        "tier2_C1_bound_vs_U1": {
            "interpretable": all(tier1_checks.values()),
            "passed": all(tier1_checks.values()) and all(tier2_checks.values()),
            "checks": tier2_checks,
        },
        "tier3_external_task_floor": {
            "passed": all(tier3_checks.values()),
            "checks": tier3_checks,
            "C1_minus_dummy_macro_f1": dummy_deltas.tolist(),
        },
    }
    c1_audits = [
        summary["model_audit_on_outer_test"]
        for key, summary in summaries.items()
        if key[0] == PIPELINES[2]
    ]
    mechanism_audit = {
        "fits": len(c1_audits),
        "maximum_dimension_budget_violations": max(
            int(audit["maximum_dimension_budget_violations"]) for audit in c1_audits
        ),
        "mean_residual_to_hidden_norm_ratio": float(
            np.mean(
                [audit["residual_to_hidden_norm_ratio"]["mean"] for audit in c1_audits]
            )
        ),
    }
    recomputed_summary = {
        "pipeline_mean_metrics": pipeline_mean_metrics,
        "comparisons": comparison_summary,
        "gate": gate,
        "C1_mechanism_audit": mechanism_audit,
    }
    tables = {
        "fold_metrics": fold_metrics,
        "seed_repeat_metrics": seed_repeat,
        "seed_repeat_deltas": deltas,
        "split_cell_deltas": split_cells,
        "per_dog_pair_deltas": dog_deltas,
    }
    report_data = {
        **recomputed_summary,
        "pipeline_mean_all_metrics": pipeline_mean_all_metrics,
        "base_seed_macro_f1_deltas": base_seed_tables,
        "seed_repeat_deltas": delta_rows,
        "split_cell_deltas": split_rows,
    }
    return report_data, tables


def audit_resume_bytes(summary_path: Path) -> dict[str, Any]:
    raw = summary_path.read_bytes()
    parsed = json.loads(raw.decode("utf-8"))
    canonical_lf = (
        json.dumps(parsed, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    normalized = raw.replace(b"\r\n", b"\n")
    return {
        "evaluation_summary_sha256": hashlib.sha256(raw).hexdigest(),
        "evaluation_summary_bytes": len(raw),
        "crlf_count": raw.count(b"\r\n"),
        "lone_lf_count": raw.count(b"\n") - raw.count(b"\r\n"),
        "raw_bytes_equal_canonical_lf": raw == canonical_lf,
        "crlf_normalized_bytes_equal_canonical_lf": normalized == canonical_lf,
        "diagnosis": "raw newline encoding mismatch only"
        if raw != canonical_lf and normalized == canonical_lf
        else "not isolated to CRLF/LF encoding",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the parent of scripts/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    protocol_path = repo_root / "configs/protocol/idea075_dog_C1_age_sensitive_AST_v1.json"
    runner_path = repo_root / "scripts/run_idea075_dog_C1_age_sensitive_AST.py"
    run_root = repo_root / "runs/idea075_dog_C1_age_sensitive_AST_v1"
    evaluation_path = run_root / "evaluation_summary.json"
    protocol = read_json(protocol_path)
    run_manifest = read_json(run_root / "run_manifest.json")
    evaluation = read_json(evaluation_path)

    require(protocol["protocol_id"] == "idea075-dog-C1-age-sensitive-AST-v1", "Protocol ID differs")
    require(tuple(protocol["pipelines"]) == PIPELINES, "Pipeline matrix differs")
    require(tuple(protocol["classes"]) == CLASS_NAMES, "Class order differs")
    require(tuple(protocol["base_seeds"]) == BASE_SEEDS, "Base seeds differ")
    require(tuple(protocol["outer_splits"]["split_seeds"]) == SPLIT_SEEDS, "Split seeds differ")
    require(protocol["status"] == "locked_before_initial_evaluation", "Protocol lock status differs")
    require(sha256(protocol_path) == run_manifest["protocol_sha256"], "Protocol/run manifest hash differs")
    require(sha256(runner_path) == run_manifest["runner_sha256"], "Runner/run manifest hash differs")
    require(run_manifest["expected_neural_fits"] == EXPECTED_FITS, "Run manifest fit budget differs")
    locked_settings = {
        "pipelines": list(PIPELINES),
        "classes": list(CLASS_NAMES),
        "base_seeds": list(BASE_SEEDS),
        "split_seeds": list(SPLIT_SEEDS),
        "model": protocol["model"],
        "acoustic_features": protocol["acoustic_features"],
        "fixed_training": protocol["fixed_training"],
        "gate": protocol["gate"],
    }
    require(canonical_json_sha256(locked_settings) == run_manifest["locked_settings_sha256"], "Locked settings hash differs")
    for stem in ("plan", "idea067_protocol", "idea067_runner", "idea068_protocol", "idea068_runner", "runner"):
        path_key = f"{stem}_path"
        hash_key = f"{stem}_sha256"
        if path_key in protocol.get("dependencies", {}):
            path = repo_root / protocol["dependencies"][path_key]
            require(path.is_file(), f"Missing dependency: {stem}")
            require(sha256(path) == protocol["dependencies"][hash_key], f"Dependency hash differs: {stem}")

    _, roles, prepared_audit = validate_prepared_assets(
        repo_root, protocol, run_manifest
    )
    summaries, predictions, fit_audit = validate_fits_and_load_predictions(
        repo_root, protocol, roles, run_root
    )
    recomputed, tables = recompute_results(predictions, summaries)

    stored_table_specs = {
        "fold_metrics": ("fold_metrics.csv", ("pipeline", "base_seed", "repeat", "outer_fold")),
        "seed_repeat_metrics": ("seed_repeat_metrics.csv", ("pipeline", "base_seed", "repeat")),
        "seed_repeat_deltas": ("seed_repeat_deltas.csv", ("comparison", "base_seed", "repeat")),
        "split_cell_deltas": ("split_cell_deltas.csv", ("comparison", "repeat", "outer_fold")),
        "per_dog_pair_deltas": ("per_dog_pair_deltas.csv", ("comparison", "base_seed", "repeat", "dog_id")),
    }
    for name, (filename, keys) in stored_table_specs.items():
        compare_frames(
            tables[name],
            pd.read_csv(run_root / filename, dtype={"dog_id": str}),
            keys,
            name,
        )

    for name in (
        "pipeline_mean_metrics",
        "comparisons",
        "gate",
        "C1_mechanism_audit",
    ):
        compare_nested(recomputed[name], evaluation[name], name)
    require(evaluation["status"] == "complete", "Evaluation summary is incomplete")
    require(evaluation["fits_completed"] == EXPECTED_FITS, "Evaluation fit count differs")
    require(evaluation["outer_test_predictions_per_fit"] == 1, "Per-fit outer-test count differs")
    require(evaluation["outer_test_accessed_during_selection"] is False, "Outer test accessed during selection")
    require(evaluation["outer_test_access_count"] == EXPECTED_FITS, "Evaluation outer-test count differs")

    linked_hashes = 0
    for name, artifact in evaluation["artifacts"].items():
        path = repo_root / artifact["path"]
        require(path.is_file() and sha256(path) == artifact["sha256"], f"Evaluation artifact hash differs: {name}")
        linked_hashes += 1
    for stem in ("pairing_audit", "run_manifest"):
        path = repo_root / evaluation[f"{stem}_path"]
        require(path.is_file() and sha256(path) == evaluation[f"{stem}_sha256"], f"Evaluation linked hash differs: {stem}")
        linked_hashes += 1

    pairing = pd.read_csv(run_root / "pairing_audit.csv")
    require(len(pairing) == 45, "Pairing audit row count differs")
    require(pairing["dog_order_hashes_equal"].astype(bool).all(), "Pairing audit dog order failed")
    require(pairing["recording_coverage_hashes_equal"].astype(bool).all(), "Pairing audit coverage failed")
    require(int(pairing["common_epochs"].sum()) == fit_audit["paired_common_epochs_verified"], "Pairing audit common epochs differ")

    result = {
        "schema_version": "1.0",
        "audit_id": "idea075-dog-C1-independent-read-only-verification-v1",
        "status": "passed",
        "read_only": True,
        "protocol_sha256": sha256(protocol_path),
        "runner_sha256": sha256(runner_path),
        "run_manifest_sha256": sha256(run_root / "run_manifest.json"),
        "prepared_assets": prepared_audit,
        "fit_artifacts": fit_audit,
        "independent_recompute": recomputed,
        "stored_aggregate_tables_verified": len(stored_table_specs),
        "evaluation_linked_hashes_verified": linked_hashes,
        "resume_newline_audit": audit_resume_bytes(evaluation_path),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
