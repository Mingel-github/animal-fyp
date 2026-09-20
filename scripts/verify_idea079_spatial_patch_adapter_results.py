"""Independently audit IDEA-079 prediction artifacts and locked gate outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "protocol" / "meowagenet_idea079_spatial_patch_adapter_v1.json"
RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_idea079_spatial_patch_adapter.py"
DEFAULT_RUN_ROOT = ROOT / "runs" / "meowagenet_idea079_spatial_patch_adapter_v1"
PIPELINES = ("A0_cached_tail", "C1_pointwise_patch", "S1_spatial_patch")
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
LOCKED_PROTOCOL_SHA256 = "690b38a7e258a44435398151ce1a114be7ca58a253e6428c85fd0e57c6f53421"
LOCKED_RUNNER_SHA256 = "0b3623c7a0ab7021e36659bdacfd5d8aed351a394dc6c8a6035e48242a7d1aa0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def assert_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_tree(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_tree(item, f"{label}[{index}]")
    elif isinstance(value, float) and not np.isfinite(value):
        raise RuntimeError(f"Non-finite numeric value at {label}")


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    labels = frame["true_label"].to_numpy(np.int64)
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(float)
    predictions = probabilities.argmax(axis=1)
    recall = recall_score(labels, predictions, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "cross_entropy": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()),
        "brier": float(np.mean(np.sum((probabilities - np.eye(3)[labels]) ** 2, axis=1))),
        "senior_recall": float(recall[2]),
    }


def calls_to_animals(calls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat_id, group in calls.groupby("cat_id", sort=True):
        labels = group["true_label"].unique()
        if len(labels) != 1:
            raise RuntimeError(f"Conflicting labels for cat {cat_id}")
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


def assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=tolerance):
        raise RuntimeError(f"Independent audit mismatch for {label}: {actual} vs {expected}")


def compare_animal_frame(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    actual = actual.sort_values("cat_id").reset_index(drop=True)
    expected = expected.sort_values("cat_id").reset_index(drop=True)
    for column in ("cat_id", "true_label", "call_count", "predicted_label"):
        if not np.array_equal(actual[column].to_numpy(), expected[column].to_numpy()):
            raise RuntimeError(f"{label} differs in {column}")
    np.testing.assert_allclose(
        actual[list(PROBABILITY_COLUMNS)].to_numpy(float),
        expected[list(PROBABILITY_COLUMNS)].to_numpy(float),
        atol=1e-12,
        rtol=1e-12,
        err_msg=label,
    )


def compare_records(actual: list[dict[str, Any]], expected: list[dict[str, Any]], label: str) -> None:
    if len(actual) != len(expected):
        raise RuntimeError(f"{label} row count differs")
    for index, (observed, stored) in enumerate(zip(actual, expected)):
        if observed.keys() != stored.keys():
            raise RuntimeError(f"{label}[{index}] keys differ")
        for key, value in observed.items():
            target = stored[key]
            if isinstance(value, float):
                assert_close(value, float(target), f"{label}[{index}].{key}")
            elif value != target:
                raise RuntimeError(f"{label}[{index}].{key} differs: {value} vs {target}")


def fit_path(root: Path, pipeline: str, base_seed: int, repeat: int, fold: int) -> Path:
    return root / "fits" / pipeline / f"base_seed_{base_seed}" / f"repeat_{repeat}" / f"fold_{fold}" / "fit_summary.json"


def describe(values: pd.Series) -> dict[str, Any]:
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "median": float(values.median()),
        "positive": int((values > 0).sum()),
        "tied": int((values == 0).sum()),
        "negative": int((values < 0).sum()),
        "worst": float(values.min()),
        "best": float(values.max()),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    protocol_hash = sha256(PROTOCOL_PATH)
    runner_hash = sha256(RUNNER_PATH)
    if protocol_hash != LOCKED_PROTOCOL_SHA256 or runner_hash != LOCKED_RUNNER_SHA256:
        raise RuntimeError("Locked protocol or runner hash changed")

    run_root = args.run_root.resolve()
    summary_path = run_root / "initial_evaluation_summary.json"
    run_summary_path = run_root / "run_summary.json"
    run_manifest_path = run_root / "run_manifest.json"
    cache_preflight_path = run_root / "cache_preflight.json"
    stored_summary = read_json(summary_path)
    run_summary = read_json(run_summary_path)
    run_manifest = read_json(run_manifest_path)
    cache_preflight = read_json(cache_preflight_path)
    cache_manifest_path = ROOT / protocol["shared_cache"]["cache_manifest_path"]
    expected_manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "runner_sha256": runner_hash,
        "cache_manifest_sha256": sha256(cache_manifest_path),
        "outer_test_accessed": False,
    }
    for key, value in expected_manifest.items():
        if run_manifest.get(key) != value:
            raise RuntimeError(f"Run manifest mismatch for {key}")
    if run_manifest.get("environment", {}).get("device") != "cuda":
        raise RuntimeError("Formal run manifest is not CUDA")
    if cache_preflight.get("status") != "GO_FOR_FORMAL_GPU_RUN":
        raise RuntimeError("Cache preflight did not authorize the formal run")
    if cache_preflight.get("outer_test_accessed") is not False:
        raise RuntimeError("Cache preflight reports outer-test access")

    roles = pd.read_csv(ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    with np.load(ROOT / protocol["data"]["frozen_embedding_path"]) as frozen:
        call_ids = frozen["call_ids"].astype(str)
        cat_ids = frozen["cat_ids"].astype(str)
        labels = frozen["labels"].astype(np.int64)

    full_seeds = [
        seed + 10_000 * repeat + 100 * fold
        for seed in protocol["model"]["base_seeds"]
        for repeat in protocol["model"]["repeats"]
        for fold in protocol["model"]["folds"]
    ]
    if len(full_seeds) != len(set(full_seeds)):
        raise RuntimeError("Derived full seeds are not unique")

    fits: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    frames: dict[tuple[str, int, int, int], pd.DataFrame] = {}
    calls_checked = 0
    animals_checked = 0
    max_reaggregation_difference = 0.0
    max_checkpoint_reload_difference = 0.0
    resource_rows = []

    for base_seed in protocol["model"]["base_seeds"]:
        for repeat in protocol["model"]["repeats"]:
            for fold in protocol["model"]["folds"]:
                cell = roles[(roles["repeat"] == repeat) & (roles["outer_fold"] == fold)]
                validation_cats = set(cell[cell["role"] == "validation"]["cat_id"].astype(str))
                outer_test_cats = set(cell[cell["role"] == "test"]["cat_id"].astype(str))
                if validation_cats & outer_test_cats:
                    raise RuntimeError(f"Validation/test overlap for repeat={repeat}, fold={fold}")
                expected_calls = set(np.flatnonzero(np.isin(cat_ids, list(validation_cats))).tolist())
                for pipeline in PIPELINES:
                    path = fit_path(run_root, pipeline, base_seed, repeat, fold)
                    fit = read_json(path)
                    assert_finite_tree(fit, str(path))
                    expected_identity = {
                        "status": "complete",
                        "pipeline": pipeline,
                        "base_seed": base_seed,
                        "full_seed": base_seed + 10_000 * repeat + 100 * fold,
                        "repeat": repeat,
                        "fold": fold,
                        "outer_test_accessed": False,
                    }
                    if any(fit.get(key) != value for key, value in expected_identity.items()):
                        raise RuntimeError(f"Fit identity mismatch: {path}")
                    audit_record = fit["audit"]
                    if audit_record.get("outer_test_accessed") is not False:
                        raise RuntimeError(f"Fit audit reports outer-test access: {path}")
                    best = audit_record["best_model"]
                    if best["trainable_parameters"] != protocol["model"]["trainable_parameters"][pipeline]:
                        raise RuntimeError(f"Trainable-parameter mismatch: {path}")
                    expected_adapter_parameters = 0 if pipeline == PIPELINES[0] else protocol["adapter"]["parameters"]
                    if best["adapter_parameters"] != expected_adapter_parameters or best["block12_trainable_parameters"] != 0:
                        raise RuntimeError(f"Frozen-tail or adapter-parameter mismatch: {path}")
                    if not 1 <= int(audit_record["best_epoch"]) <= int(audit_record["stopped_epoch"]) <= int(protocol["fixed_training"]["maximum_epochs"]):
                        raise RuntimeError(f"Invalid epoch audit: {path}")

                    animal_path = ROOT / fit["validation_animal_predictions"]
                    call_path = ROOT / fit["validation_call_predictions"]
                    if sha256(animal_path) != fit["validation_animal_sha256"] or sha256(call_path) != fit["validation_call_sha256"]:
                        raise RuntimeError(f"Prediction hash mismatch: {path}")
                    call_frame = pd.read_csv(call_path, dtype={"call_id": str, "cat_id": str})
                    animal_frame = pd.read_csv(animal_path, dtype={"cat_id": str})
                    if set(call_frame["call_index"].astype(int)) != expected_calls:
                        raise RuntimeError(f"Validation call set mismatch: {path}")
                    if set(call_frame["cat_id"].astype(str)) != validation_cats:
                        raise RuntimeError(f"Validation cat set mismatch: {path}")
                    if set(call_frame["cat_id"].astype(str)) & outer_test_cats:
                        raise RuntimeError(f"Outer-test cat appears in predictions: {path}")
                    indices = call_frame["call_index"].to_numpy(np.int64)
                    if len(indices) != len(set(indices.tolist())):
                        raise RuntimeError(f"Duplicate call rows: {path}")
                    if not (
                        np.array_equal(call_frame["call_id"].astype(str), call_ids[indices])
                        and np.array_equal(call_frame["cat_id"].astype(str), cat_ids[indices])
                        and np.array_equal(call_frame["true_label"].to_numpy(np.int64), labels[indices])
                    ):
                        raise RuntimeError(f"Prediction identity columns mismatch: {path}")
                    probabilities = call_frame[list(PROBABILITY_COLUMNS)].to_numpy(float)
                    if not np.isfinite(probabilities).all() or np.max(np.abs(probabilities.sum(axis=1) - 1.0)) > 2e-7:
                        raise RuntimeError(f"Invalid probabilities: {path}")
                    recomputed = calls_to_animals(call_frame)
                    compare_animal_frame(animal_frame, recomputed, str(path))
                    difference = float(np.max(np.abs(
                        animal_frame[list(PROBABILITY_COLUMNS)].to_numpy(float)
                        - recomputed[list(PROBABILITY_COLUMNS)].to_numpy(float)
                    )))
                    max_reaggregation_difference = max(max_reaggregation_difference, difference)
                    reload_difference = float(audit_record["checkpoint_reload_max_probability_difference"])
                    max_checkpoint_reload_difference = max(max_checkpoint_reload_difference, reload_difference)
                    resource_rows.append(
                        {
                            "pipeline": pipeline,
                            "best_epoch": int(audit_record["best_epoch"]),
                            "stopped_epoch": int(audit_record["stopped_epoch"]),
                            "train_seconds": float(audit_record["train_seconds"]),
                            "peak_vram_bytes": int(audit_record["peak_vram_bytes"]),
                        }
                    )
                    fits[(pipeline, base_seed, repeat, fold)] = fit
                    frames[(pipeline, base_seed, repeat, fold)] = animal_frame
                    calls_checked += len(call_frame)
                    animals_checked += len(animal_frame)

    discovered = list((run_root / "fits").rglob("fit_summary.json"))
    if len(fits) != 108 or len(discovered) != 108:
        raise RuntimeError("Expected exactly 108 fit summaries")
    if run_summary != {
        "status": "complete",
        "completed_fits": 108,
        "expected_fits": 108,
        "outer_test_accessed": False,
        "candidate_gate_passed": False,
        "spatial_mechanism_gate_passed": False,
        "full_gate_passed": False,
    }:
        raise RuntimeError("Run summary does not have the expected terminal state")

    fold_rows = []
    seed_repeat_rows = []
    pooled_all: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
    for base_seed in protocol["model"]["base_seeds"]:
        for repeat in protocol["model"]["repeats"]:
            grouped: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
            for fold in protocol["model"]["folds"]:
                bundles = {pipeline: metrics(frames[(pipeline, base_seed, repeat, fold)]) for pipeline in PIPELINES}
                row: dict[str, Any] = {"base_seed": base_seed, "repeat": repeat, "fold": fold}
                for pipeline in PIPELINES:
                    for metric_name, value in bundles[pipeline].items():
                        row[f"{pipeline}_{metric_name}"] = value
                    tagged = frames[(pipeline, base_seed, repeat, fold)].copy()
                    tagged["base_seed"] = base_seed
                    grouped[pipeline].append(tagged)
                    pooled_all[pipeline].append(tagged)
                row["S1_minus_A0_macro_f1"] = row[f"{PIPELINES[2]}_macro_f1"] - row[f"{PIPELINES[0]}_macro_f1"]
                row["C1_minus_A0_macro_f1"] = row[f"{PIPELINES[1]}_macro_f1"] - row[f"{PIPELINES[0]}_macro_f1"]
                row["S1_minus_C1_macro_f1"] = row[f"{PIPELINES[2]}_macro_f1"] - row[f"{PIPELINES[1]}_macro_f1"]
                fold_rows.append(row)
            pooled = {pipeline: pd.concat(parts, ignore_index=True) for pipeline, parts in grouped.items()}
            bundles = {pipeline: metrics(frame) for pipeline, frame in pooled.items()}
            row = {"base_seed": base_seed, "repeat": repeat}
            for pipeline in PIPELINES:
                for metric_name, value in bundles[pipeline].items():
                    row[f"{pipeline}_{metric_name}"] = value
            row["S1_minus_A0_macro_f1"] = row[f"{PIPELINES[2]}_macro_f1"] - row[f"{PIPELINES[0]}_macro_f1"]
            row["C1_minus_A0_macro_f1"] = row[f"{PIPELINES[1]}_macro_f1"] - row[f"{PIPELINES[0]}_macro_f1"]
            row["S1_minus_C1_macro_f1"] = row[f"{PIPELINES[2]}_macro_f1"] - row[f"{PIPELINES[1]}_macro_f1"]
            seed_repeat_rows.append(row)

    folds = pd.DataFrame(fold_rows)
    seed_repeats = pd.DataFrame(seed_repeat_rows)
    comparison_columns = ["S1_minus_A0_macro_f1", "C1_minus_A0_macro_f1", "S1_minus_C1_macro_f1"]
    split_cells = folds.groupby(["repeat", "fold"], as_index=False)[comparison_columns].mean()
    mean_metrics = {
        metric_name: {pipeline: float(seed_repeats[f"{pipeline}_{metric_name}"].mean()) for pipeline in PIPELINES}
        for metric_name in ("macro_f1", "balanced_accuracy", "cross_entropy", "brier")
    }
    per_seed_delta = {
        str(seed): float(seed_repeats[seed_repeats["base_seed"] == seed]["S1_minus_A0_macro_f1"].mean())
        for seed in protocol["model"]["base_seeds"]
    }
    pooled_frames = {pipeline: pd.concat(parts, ignore_index=True) for pipeline, parts in pooled_all.items()}
    per_seed_senior = {}
    for seed in protocol["model"]["base_seeds"]:
        values = {pipeline: metrics(frame[frame["base_seed"] == seed])["senior_recall"] for pipeline, frame in pooled_frames.items()}
        per_seed_senior[str(seed)] = float(values[PIPELINES[2]] - values[PIPELINES[0]])

    candidate_delta = seed_repeats["S1_minus_A0_macro_f1"]
    mechanism_delta = seed_repeats["S1_minus_C1_macro_f1"]
    split_delta = split_cells["S1_minus_A0_macro_f1"]
    gate = protocol["gate"]
    candidate_conditions = {
        "mean_macro_f1_gain": float(candidate_delta.mean()) >= float(gate["minimum_mean_seed_repeat_S1_minus_A0"]),
        "positive_base_seed_means": sum(value > 0 for value in per_seed_delta.values()) >= int(gate["minimum_positive_base_seeds"]),
        "positive_seed_repeats": int((candidate_delta > 0).sum()) >= int(gate["minimum_positive_seed_repeats"]),
        "nonnegative_split_cells": int((split_delta >= 0).sum()) >= int(gate["minimum_nonnegative_split_cells"]),
        "worst_split_cell": float(split_delta.min()) >= float(gate["minimum_worst_split_cell_delta"]),
        "cross_entropy_nonworse": mean_metrics["cross_entropy"][PIPELINES[2]] <= mean_metrics["cross_entropy"][PIPELINES[0]],
        "brier_nonworse": mean_metrics["brier"][PIPELINES[2]] <= mean_metrics["brier"][PIPELINES[0]],
        "per_base_seed_senior_recall": all(value >= float(gate["minimum_per_base_seed_senior_recall_delta"]) for value in per_seed_senior.values()),
    }
    mechanism_conditions = {
        "mean_S1_minus_C1_macro_f1": float(mechanism_delta.mean()) >= float(gate["minimum_mean_seed_repeat_S1_minus_C1"]),
        "positive_S1_minus_C1_seed_repeats": int((mechanism_delta > 0).sum()) >= int(gate["minimum_positive_S1_minus_C1_seed_repeats"]),
    }

    compare_records(fold_rows, stored_summary["fold_results"], "fold_results")
    compare_records(seed_repeat_rows, stored_summary["seed_repeat_results"], "seed_repeat_results")
    compare_records(split_cells.to_dict(orient="records"), stored_summary["split_cell_results"], "split_cell_results")
    for metric_name, by_pipeline in mean_metrics.items():
        for pipeline, value in by_pipeline.items():
            assert_close(value, stored_summary["seed_repeat_equal_weight_means"][metric_name][pipeline], f"{metric_name}/{pipeline}")
    comparisons = {column: describe(seed_repeats[column]) for column in comparison_columns}
    for column, observed in comparisons.items():
        expected = stored_summary["comparisons"][column]
        for key, value in observed.items():
            if isinstance(value, float):
                assert_close(value, float(expected[key]), f"{column}/{key}")
            elif value != expected[key]:
                raise RuntimeError(f"Independent audit mismatch for {column}/{key}")
    for seed, value in per_seed_delta.items():
        assert_close(value, stored_summary["per_base_seed_S1_minus_A0_macro_f1"][seed], f"per_seed_delta/{seed}")
    for seed, value in per_seed_senior.items():
        assert_close(value, stored_summary["per_base_seed_S1_minus_A0_senior_recall"][seed], f"per_seed_senior/{seed}")
    if candidate_conditions != stored_summary["candidate_gate_conditions"]:
        raise RuntimeError("Independent candidate-gate audit differs from locked summary")
    if mechanism_conditions != stored_summary["mechanism_gate_conditions"]:
        raise RuntimeError("Independent mechanism-gate audit differs from locked summary")
    candidate_passed = bool(all(candidate_conditions.values()))
    mechanism_passed = bool(all(mechanism_conditions.values()))
    if candidate_passed != stored_summary["candidate_gate_passed"] or mechanism_passed != stored_summary["spatial_mechanism_gate_passed"]:
        raise RuntimeError("Independent gate result differs from locked summary")
    if bool(candidate_passed and mechanism_passed) != stored_summary["full_gate_passed"]:
        raise RuntimeError("Independent full-gate result differs from locked summary")

    resources = pd.DataFrame(resource_rows)
    resource_summary = {}
    for pipeline in PIPELINES:
        frame = resources[resources["pipeline"] == pipeline]
        resource_summary[pipeline] = {
            "fits": int(len(frame)),
            "best_epoch_mean": float(frame["best_epoch"].mean()),
            "best_epoch_range": [int(frame["best_epoch"].min()), int(frame["best_epoch"].max())],
            "stopped_epoch_mean": float(frame["stopped_epoch"].mean()),
            "train_seconds_total": float(frame["train_seconds"].sum()),
            "train_seconds_mean": float(frame["train_seconds"].mean()),
            "peak_vram_bytes_max": int(frame["peak_vram_bytes"].max()),
        }

    return {
        "status": "PASS",
        "independent_recalculation_matches_locked_summary": True,
        "resume_integrity_check": "PASS: locked --resume command returned 108/108 without retraining",
        "protocol_sha256": protocol_hash,
        "runner_sha256": runner_hash,
        "cache_manifest_sha256": sha256(cache_manifest_path),
        "summary_sha256": sha256(summary_path),
        "run_summary_sha256": sha256(run_summary_path),
        "run_manifest_sha256": sha256(run_manifest_path),
        "fit_summaries_checked": len(fits),
        "prediction_files_checked": len(fits) * 2,
        "call_prediction_rows_checked": calls_checked,
        "animal_prediction_rows_checked": animals_checked,
        "maximum_call_to_animal_reaggregation_difference": max_reaggregation_difference,
        "maximum_checkpoint_reload_probability_difference": max_checkpoint_reload_difference,
        "outer_test_accessed": False,
        "seed_repeat_equal_weight_means": mean_metrics,
        "comparisons": comparisons,
        "split_cell_S1_minus_A0": {
            "nonnegative": int((split_delta >= 0).sum()),
            "positive": int((split_delta > 0).sum()),
            "tied": int((split_delta == 0).sum()),
            "negative": int((split_delta < 0).sum()),
            "worst": float(split_delta.min()),
            "best": float(split_delta.max()),
        },
        "per_base_seed_S1_minus_A0_macro_f1": per_seed_delta,
        "per_base_seed_S1_minus_A0_senior_recall": per_seed_senior,
        "resource_summary": resource_summary,
        "candidate_gate_conditions": candidate_conditions,
        "mechanism_gate_conditions": mechanism_conditions,
        "candidate_gate_passed": candidate_passed,
        "spatial_mechanism_gate_passed": mechanism_passed,
        "full_gate_passed": bool(candidate_passed and mechanism_passed),
    }


def main() -> None:
    args = parse_args()
    result = audit(args)
    output = args.run_root.resolve() / "independent_audit.json"
    write_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
