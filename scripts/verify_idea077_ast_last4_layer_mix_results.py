"""Independently audit IDEA-077 prediction artifacts and locked gate outcomes."""

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
PROTOCOL_PATH = ROOT / "configs" / "protocol" / "meowagenet_idea077_ast_last4_layer_mix_v1.json"
DEFAULT_RUN_ROOT = ROOT / "runs" / "meowagenet_idea077_ast_last4_layer_mix_v1"
PIPELINES = ("A0_final", "M0_uniform_last4", "L1_global_layermix")
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")


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


def compare_frame(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
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


def fit_path(root: Path, pipeline: str, base_seed: int, repeat: int, fold: int) -> Path:
    return root / "fits" / pipeline / f"base_seed_{base_seed}" / f"repeat_{repeat}" / f"fold_{fold}" / "fit_summary.json"


def audit(args: argparse.Namespace) -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    run_root = args.run_root.resolve()
    summary_path = run_root / "initial_evaluation_summary.json"
    run_summary_path = run_root / "run_summary.json"
    stored_summary = read_json(summary_path)
    run_summary = read_json(run_summary_path)
    roles = pd.read_csv(ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    with np.load(ROOT / protocol["data"]["frozen_embedding_path"]) as frozen:
        call_ids = frozen["call_ids"].astype(str)
        cat_ids = frozen["cat_ids"].astype(str)
        labels = frozen["labels"].astype(np.int64)

    fits: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    frames: dict[tuple[str, int, int, int], pd.DataFrame] = {}
    calls_checked = 0
    animals_checked = 0
    max_reaggregation_difference = 0.0
    max_checkpoint_reload_difference = 0.0
    l1_gates = []
    l1_weights = []

    for base_seed in protocol["model"]["base_seeds"]:
        for repeat in protocol["model"]["repeats"]:
            for fold in protocol["model"]["folds"]:
                cell = roles[(roles["repeat"] == repeat) & (roles["outer_fold"] == fold)]
                validation_cats = set(cell[cell["role"] == "validation"]["cat_id"].astype(str))
                expected_calls = set(np.flatnonzero(np.isin(cat_ids, list(validation_cats))).tolist())
                for pipeline in PIPELINES:
                    path = fit_path(run_root, pipeline, base_seed, repeat, fold)
                    fit = read_json(path)
                    expected_identity = {
                        "status": "complete", "pipeline": pipeline, "base_seed": base_seed,
                        "full_seed": base_seed + 10_000 * repeat + 100 * fold,
                        "repeat": repeat, "fold": fold, "outer_test_accessed": False,
                    }
                    if any(fit.get(key) != value for key, value in expected_identity.items()):
                        raise RuntimeError(f"Fit identity mismatch: {path}")
                    initial = fit["initialization_audit"]
                    if initial["L1_minus_A0_max_initial_logit_difference"] != 0.0 or initial["L1_minus_A0_initial_loss_difference"] != 0.0:
                        raise RuntimeError(f"Initialization equality failed: {path}")
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
                    indices = call_frame["call_index"].to_numpy(np.int64)
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
                    compare_frame(animal_frame, recomputed, str(path))
                    difference = float(np.max(np.abs(
                        animal_frame[list(PROBABILITY_COLUMNS)].to_numpy(float)
                        - recomputed[list(PROBABILITY_COLUMNS)].to_numpy(float)
                    )))
                    max_reaggregation_difference = max(max_reaggregation_difference, difference)
                    max_checkpoint_reload_difference = max(
                        max_checkpoint_reload_difference,
                        float(fit["audit"]["checkpoint_reload_max_probability_difference"]),
                    )
                    if pipeline == PIPELINES[2]:
                        best = fit["audit"]["best_model"]
                        weights = np.asarray(best["layer_weights_one_based_9_to_12"], dtype=float)
                        if not np.isclose(weights.sum(), 1.0, atol=1e-7) or np.any(weights < 0):
                            raise RuntimeError(f"Invalid learned weights: {path}")
                        gate = float(best["signed_residual_gate"])
                        if not -1.0 <= gate <= 1.0:
                            raise RuntimeError(f"Invalid learned gate: {path}")
                        l1_gates.append(gate)
                        l1_weights.append(weights)
                    fits[(pipeline, base_seed, repeat, fold)] = fit
                    frames[(pipeline, base_seed, repeat, fold)] = animal_frame
                    calls_checked += len(call_frame)
                    animals_checked += len(animal_frame)

    discovered = list((run_root / "fits").rglob("fit_summary.json"))
    if len(fits) != 108 or len(discovered) != 108:
        raise RuntimeError("Expected exactly 108 fit summaries")
    if run_summary != {
        "status": "complete", "completed_fits": 108, "expected_fits": 108,
        "outer_test_accessed": False, "gate_passed": False,
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
                row["L1_minus_A0_macro_f1"] = row[f"{PIPELINES[2]}_macro_f1"] - row[f"{PIPELINES[0]}_macro_f1"]
                row["M0_minus_A0_macro_f1"] = row[f"{PIPELINES[1]}_macro_f1"] - row[f"{PIPELINES[0]}_macro_f1"]
                row["L1_minus_M0_macro_f1"] = row[f"{PIPELINES[2]}_macro_f1"] - row[f"{PIPELINES[1]}_macro_f1"]
                fold_rows.append(row)
            pooled = {pipeline: pd.concat(parts, ignore_index=True) for pipeline, parts in grouped.items()}
            bundles = {pipeline: metrics(frame) for pipeline, frame in pooled.items()}
            row = {"base_seed": base_seed, "repeat": repeat}
            for pipeline in PIPELINES:
                for metric_name, value in bundles[pipeline].items():
                    row[f"{pipeline}_{metric_name}"] = value
            row["L1_minus_A0_macro_f1"] = row[f"{PIPELINES[2]}_macro_f1"] - row[f"{PIPELINES[0]}_macro_f1"]
            row["M0_minus_A0_macro_f1"] = row[f"{PIPELINES[1]}_macro_f1"] - row[f"{PIPELINES[0]}_macro_f1"]
            row["L1_minus_M0_macro_f1"] = row[f"{PIPELINES[2]}_macro_f1"] - row[f"{PIPELINES[1]}_macro_f1"]
            seed_repeat_rows.append(row)

    folds = pd.DataFrame(fold_rows)
    seed_repeats = pd.DataFrame(seed_repeat_rows)
    comparison_columns = ["L1_minus_A0_macro_f1", "M0_minus_A0_macro_f1", "L1_minus_M0_macro_f1"]
    split_cells = folds.groupby(["repeat", "fold"], as_index=False)[comparison_columns].mean()
    mean_metrics = {
        metric_name: {
            pipeline: float(seed_repeats[f"{pipeline}_{metric_name}"].mean())
            for pipeline in PIPELINES
        }
        for metric_name in ("macro_f1", "balanced_accuracy", "cross_entropy", "brier")
    }
    per_seed_delta = {
        str(seed): float(seed_repeats[seed_repeats["base_seed"] == seed]["L1_minus_A0_macro_f1"].mean())
        for seed in protocol["model"]["base_seeds"]
    }
    pooled_frames = {pipeline: pd.concat(parts, ignore_index=True) for pipeline, parts in pooled_all.items()}
    per_seed_senior = {}
    for seed in protocol["model"]["base_seeds"]:
        values = {
            pipeline: metrics(frame[frame["base_seed"] == seed])["senior_recall"]
            for pipeline, frame in pooled_frames.items()
        }
        per_seed_senior[str(seed)] = float(values[PIPELINES[2]] - values[PIPELINES[0]])
    delta = seed_repeats["L1_minus_A0_macro_f1"]
    split_delta = split_cells["L1_minus_A0_macro_f1"]
    gate = protocol["gate"]
    conditions = {
        "mean_macro_f1_gain": float(delta.mean()) >= float(gate["minimum_mean_seed_repeat_L1_minus_A0"]),
        "positive_base_seed_means": sum(value > 0 for value in per_seed_delta.values()) >= int(gate["minimum_positive_base_seeds"]),
        "positive_seed_repeats": int((delta > 0).sum()) >= int(gate["minimum_positive_seed_repeats"]),
        "nonnegative_split_cells": int((split_delta >= 0).sum()) >= int(gate["minimum_nonnegative_split_cells"]),
        "worst_split_cell": float(split_delta.min()) >= float(gate["minimum_worst_split_cell_delta"]),
        "cross_entropy_nonworse": mean_metrics["cross_entropy"][PIPELINES[2]] <= mean_metrics["cross_entropy"][PIPELINES[0]],
        "brier_nonworse": mean_metrics["brier"][PIPELINES[2]] <= mean_metrics["brier"][PIPELINES[0]],
        "per_base_seed_senior_recall": all(value >= float(gate["minimum_per_base_seed_senior_recall_delta"]) for value in per_seed_senior.values()),
    }

    for metric_name, by_pipeline in mean_metrics.items():
        for pipeline, value in by_pipeline.items():
            assert_close(value, stored_summary["seed_repeat_equal_weight_means"][metric_name][pipeline], f"{metric_name}/{pipeline}")
    for column in comparison_columns:
        values = seed_repeats[column]
        expected = stored_summary["comparisons"][column]
        observed = {
            "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
            "median": float(values.median()), "positive": int((values > 0).sum()),
            "tied": int((values == 0).sum()), "negative": int((values < 0).sum()),
            "worst": float(values.min()), "best": float(values.max()),
        }
        for key, value in observed.items():
            if isinstance(value, float):
                assert_close(value, float(expected[key]), f"{column}/{key}")
            elif value != expected[key]:
                raise RuntimeError(f"Independent audit mismatch for {column}/{key}")
    if conditions != stored_summary["gate_conditions"] or bool(all(conditions.values())) != stored_summary["gate_passed"]:
        raise RuntimeError("Independent gate audit differs from locked summary")

    gates = np.asarray(l1_gates, dtype=float)
    weights = np.stack(l1_weights)
    return {
        "status": "PASS",
        "independent_recalculation_matches_locked_summary": True,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": protocol["dependencies"]["runner_sha256"],
        "summary_sha256": sha256(summary_path),
        "run_summary_sha256": sha256(run_summary_path),
        "fit_summaries_checked": len(fits),
        "prediction_files_checked": len(fits) * 2,
        "call_prediction_rows_checked": calls_checked,
        "animal_prediction_rows_checked": animals_checked,
        "maximum_call_to_animal_reaggregation_difference": max_reaggregation_difference,
        "maximum_checkpoint_reload_probability_difference": max_checkpoint_reload_difference,
        "outer_test_accessed": False,
        "seed_repeat_equal_weight_means": mean_metrics,
        "comparisons": {
            column: {
                "mean": float(seed_repeats[column].mean()),
                "positive": int((seed_repeats[column] > 0).sum()),
                "tied": int((seed_repeats[column] == 0).sum()),
                "negative": int((seed_repeats[column] < 0).sum()),
                "worst": float(seed_repeats[column].min()),
                "best": float(seed_repeats[column].max()),
            }
            for column in comparison_columns
        },
        "split_cell_L1_minus_A0": {
            "nonnegative": int((split_delta >= 0).sum()),
            "positive": int((split_delta > 0).sum()),
            "negative": int((split_delta < 0).sum()),
            "worst": float(split_delta.min()),
            "best": float(split_delta.max()),
        },
        "per_base_seed_L1_minus_A0_macro_f1": per_seed_delta,
        "per_base_seed_L1_minus_A0_senior_recall": per_seed_senior,
        "layer_mix_best_checkpoint_summary": {
            "signed_gate_mean": float(gates.mean()),
            "signed_gate_sd": float(gates.std(ddof=1)),
            "signed_gate_min": float(gates.min()),
            "signed_gate_max": float(gates.max()),
            "mean_weights_layers_9_to_12": weights.mean(axis=0).tolist(),
            "sd_weights_layers_9_to_12": weights.std(axis=0, ddof=1).tolist(),
        },
        "gate_conditions": conditions,
        "gate_passed": bool(all(conditions.values())),
    }


def main() -> None:
    args = parse_args()
    result = audit(args)
    output = args.run_root.resolve() / "independent_audit.json"
    write_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
