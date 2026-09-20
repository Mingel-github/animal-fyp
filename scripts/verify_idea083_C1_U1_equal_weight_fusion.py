from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
SHORTS = ("A0", "C1", "U1", "E")
TOLERANCE = 1.0e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.write_bytes(
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    targets = np.eye(3, dtype=float)[labels]
    recalls = recall_score(labels, predictions, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "macro_f1": float(
            f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "cross_entropy": float(
            -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
        ),
        "brier": float(np.mean(np.sum((probabilities - targets) ** 2, axis=1))),
        "senior_recall": float(recalls[2]),
    }


def close(left: float, right: float, message: str) -> None:
    if not np.isclose(left, right, atol=TOLERANCE, rtol=0.0):
        raise RuntimeError(f"{message}: {left} != {right}")


def recompute_dataset(
    dataset_id: str,
    animal_frame: pd.DataFrame,
    call_frame: pd.DataFrame,
    fold_frame: pd.DataFrame,
    saved: dict[str, Any],
) -> dict[str, Any]:
    animals = animal_frame[animal_frame["dataset_id"] == dataset_id].copy()
    calls = call_frame[call_frame["dataset_id"] == dataset_id].copy()
    folds = fold_frame[fold_frame["dataset_id"] == dataset_id].copy()
    animals = animals.sort_values(
        ["base_seed", "repeat", "fold", "cat_id"]
    ).reset_index(drop=True)
    calls = calls.sort_values(
        ["base_seed", "repeat", "fold", "call_index", "call_id"]
    ).reset_index(drop=True)
    if animals.duplicated(["base_seed", "repeat", "fold", "cat_id"]).any():
        raise RuntimeError(f"Duplicate animal keys in {dataset_id}")
    if calls.duplicated(["base_seed", "repeat", "fold", "call_index", "call_id"]).any():
        raise RuntimeError(f"Duplicate call keys in {dataset_id}")
    if int(len(animals)) != int(saved["animal_occurrences"]):
        raise RuntimeError(f"Animal occurrence count mismatch in {dataset_id}")
    if int(animals["cat_id"].nunique()) != int(saved["unique_cats"]):
        raise RuntimeError(f"Unique cat count mismatch in {dataset_id}")

    c1_call = calls[[f"C1_{column}" for column in PROBABILITY_COLUMNS]].to_numpy(dtype=float)
    u1_call = calls[[f"U1_{column}" for column in PROBABILITY_COLUMNS]].to_numpy(dtype=float)
    e_call = calls[[f"E_{column}" for column in PROBABILITY_COLUMNS]].to_numpy(dtype=float)
    call_formula_max = float(np.abs(e_call - 0.5 * (c1_call + u1_call)).max(initial=0.0))
    if call_formula_max > TOLERANCE:
        raise RuntimeError(f"Call fusion formula mismatch in {dataset_id}")

    c1_animal = animals[[f"C1_{column}" for column in PROBABILITY_COLUMNS]].to_numpy(dtype=float)
    u1_animal = animals[[f"U1_{column}" for column in PROBABILITY_COLUMNS]].to_numpy(dtype=float)
    e_animal = animals[[f"E_{column}" for column in PROBABILITY_COLUMNS]].to_numpy(dtype=float)
    animal_formula_max = float(np.abs(e_animal - 0.5 * (c1_animal + u1_animal)).max(initial=0.0))
    if animal_formula_max > TOLERANCE:
        raise RuntimeError(f"Animal fusion formula mismatch in {dataset_id}")

    call_grouped = (
        calls.groupby(["base_seed", "repeat", "fold", "cat_id"], sort=True)[
            [f"E_{column}" for column in PROBABILITY_COLUMNS]
        ]
        .mean()
        .reset_index()
        .sort_values(["base_seed", "repeat", "fold", "cat_id"])
        .reset_index(drop=True)
    )
    animal_sorted = animals.sort_values(
        ["base_seed", "repeat", "fold", "cat_id"]
    ).reset_index(drop=True)
    if not call_grouped[["base_seed", "repeat", "fold", "cat_id"]].equals(
        animal_sorted[["base_seed", "repeat", "fold", "cat_id"]]
    ):
        raise RuntimeError(f"Call/animal key mismatch in {dataset_id}")
    call_animal_max = float(
        np.abs(
            call_grouped[[f"E_{column}" for column in PROBABILITY_COLUMNS]].to_numpy(dtype=float)
            - e_animal
        ).max(initial=0.0)
    )
    if call_animal_max > TOLERANCE:
        raise RuntimeError(f"Call mean does not reproduce animal ensemble in {dataset_id}")

    seed_rows: list[dict[str, Any]] = []
    for (base_seed, repeat), group in animals.groupby(["base_seed", "repeat"], sort=True):
        labels = group["true_label"].to_numpy(dtype=np.int64)
        row: dict[str, Any] = {"base_seed": int(base_seed), "repeat": int(repeat)}
        for short in SHORTS:
            probabilities = group[
                [f"{short}_{column}" for column in PROBABILITY_COLUMNS]
            ].to_numpy(dtype=float)
            for name, value in metrics(labels, probabilities).items():
                row[f"{short}_{name}"] = value
        for reference in ("A0", "C1", "U1"):
            for metric in ("macro_f1", "balanced_accuracy", "cross_entropy", "brier", "senior_recall"):
                row[f"E_minus_{reference}_{metric}"] = row[f"E_{metric}"] - row[f"{reference}_{metric}"]
        seed_rows.append(row)
    seed_frame = pd.DataFrame(seed_rows)
    if len(seed_frame) != int(saved["seed_repeat_units"]):
        raise RuntimeError(f"Seed-repeat count mismatch in {dataset_id}")

    max_saved_metric_difference = 0.0
    for short in SHORTS:
        for metric in ("macro_f1", "balanced_accuracy", "cross_entropy", "brier", "senior_recall"):
            recomputed = float(seed_frame[f"{short}_{metric}"].mean())
            stored = float(saved["pipeline_seed_repeat_equal_weight_means"][short][metric])
            max_saved_metric_difference = max(max_saved_metric_difference, abs(recomputed - stored))
            close(recomputed, stored, f"Pipeline mean mismatch {dataset_id}/{short}/{metric}")

    for reference in ("A0", "C1", "U1"):
        values = seed_frame[f"E_minus_{reference}_macro_f1"].to_numpy(dtype=float)
        stored = saved["comparisons"][f"E_minus_{reference}"]["macro_f1"]
        checks = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "worst": float(values.min()),
            "best": float(values.max()),
        }
        for name, value in checks.items():
            close(value, float(stored[name]), f"Comparison mismatch {dataset_id}/{reference}/{name}")
        counts = {
            "positive": int((values > TOLERANCE).sum()),
            "tied": int((np.abs(values) <= TOLERANCE).sum()),
            "negative": int((values < -TOLERANCE).sum()),
        }
        for name, value in counts.items():
            if value != int(stored[name]):
                raise RuntimeError(f"Sign count mismatch {dataset_id}/{reference}/{name}")

    # Independently recompute fold metrics from occurrence rows, then common split cells.
    fold_rebuilt: list[dict[str, Any]] = []
    for (base_seed, repeat, fold), group in animals.groupby(
        ["base_seed", "repeat", "fold"], sort=True
    ):
        labels = group["true_label"].to_numpy(dtype=np.int64)
        row: dict[str, Any] = {
            "base_seed": int(base_seed),
            "repeat": int(repeat),
            "fold": int(fold),
        }
        for short in SHORTS:
            probabilities = group[
                [f"{short}_{column}" for column in PROBABILITY_COLUMNS]
            ].to_numpy(dtype=float)
            row[f"{short}_macro_f1"] = metrics(labels, probabilities)["macro_f1"]
        for reference in ("A0", "C1", "U1"):
            row[f"E_minus_{reference}_macro_f1"] = row["E_macro_f1"] - row[f"{reference}_macro_f1"]
        fold_rebuilt.append(row)
    rebuilt_folds = pd.DataFrame(fold_rebuilt).sort_values(
        ["base_seed", "repeat", "fold"]
    ).reset_index(drop=True)
    stored_folds = folds.sort_values(["base_seed", "repeat", "fold"]).reset_index(drop=True)
    for reference in ("A0", "C1", "U1"):
        column = f"E_minus_{reference}_macro_f1"
        difference = np.abs(
            rebuilt_folds[column].to_numpy(dtype=float)
            - stored_folds[column].to_numpy(dtype=float)
        )
        if float(difference.max(initial=0.0)) > TOLERANCE:
            raise RuntimeError(f"Fold delta mismatch in {dataset_id}/{reference}")
        split_values = rebuilt_folds.groupby(["repeat", "fold"])[column].mean().to_numpy(dtype=float)
        split_saved = saved["comparisons"][f"E_minus_{reference}"]["split_cells"]
        close(float(split_values.min()), float(split_saved["worst"]), f"Worst split mismatch {dataset_id}/{reference}")
        if int((split_values >= -TOLERANCE).sum()) != int(split_saved["nonnegative"]):
            raise RuntimeError(f"Nonnegative split count mismatch in {dataset_id}/{reference}")

    absolute_splits = (
        rebuilt_folds.groupby(["repeat", "fold"], as_index=False)[
            [f"{short}_macro_f1" for short in SHORTS]
        ]
        .mean()
        .sort_values(["repeat", "fold"])
        .reset_index(drop=True)
    )
    worst_absolute: dict[str, Any] = {}
    for short in SHORTS:
        row = absolute_splits.loc[absolute_splits[f"{short}_macro_f1"].idxmin()]
        worst_absolute[short] = {
            "repeat": int(row["repeat"]),
            "fold": int(row["fold"]),
            "macro_f1": float(row[f"{short}_macro_f1"]),
        }
    u1_worst_row = absolute_splits.loc[
        absolute_splits["U1_macro_f1"].idxmin()
    ]
    u1_worst_split_detail = {
        "repeat": int(u1_worst_row["repeat"]),
        "fold": int(u1_worst_row["fold"]),
        **{
            f"{short}_macro_f1": float(u1_worst_row[f"{short}_macro_f1"])
            for short in SHORTS
        },
        "E_minus_U1_macro_f1": float(
            u1_worst_row["E_macro_f1"] - u1_worst_row["U1_macro_f1"]
        ),
        "E_minus_C1_macro_f1": float(
            u1_worst_row["E_macro_f1"] - u1_worst_row["C1_macro_f1"]
        ),
    }

    c1_correct = animals["C1_correct"].to_numpy(dtype=bool)
    u1_correct = animals["U1_correct"].to_numpy(dtype=bool)
    e_correct = animals["E_correct"].to_numpy(dtype=bool)
    comp_saved = saved["complementarity_animal_occurrences"]
    complementarity = {
        "both_correct": int((c1_correct & u1_correct).sum()),
        "C1_wrong_U1_correct": int(((~c1_correct) & u1_correct).sum()),
        "U1_wrong_C1_correct": int((c1_correct & (~u1_correct)).sum()),
        "both_wrong": int(((~c1_correct) & (~u1_correct)).sum()),
        "different_predictions": int(
            (animals["C1_predicted_label"].to_numpy() != animals["U1_predicted_label"].to_numpy()).sum()
        ),
        "E_vs_C1_corrected": int(((~c1_correct) & e_correct).sum()),
        "E_vs_C1_damaged": int((c1_correct & (~e_correct)).sum()),
        "E_vs_U1_corrected": int(((~u1_correct) & e_correct).sum()),
        "E_vs_U1_damaged": int((u1_correct & (~e_correct)).sum()),
        "oracle_upper_bound_correct_occurrences": int((c1_correct | u1_correct).sum()),
    }
    for name, value in complementarity.items():
        if value != int(comp_saved[name]):
            raise RuntimeError(f"Complementarity mismatch in {dataset_id}/{name}")

    ce_margins = seed_frame["E_cross_entropy"] - 0.5 * (
        seed_frame["C1_cross_entropy"] + seed_frame["U1_cross_entropy"]
    )
    brier_margins = seed_frame["E_brier"] - 0.5 * (
        seed_frame["C1_brier"] + seed_frame["U1_brier"]
    )
    if float(ce_margins.max()) > TOLERANCE or float(brier_margins.max()) > TOLERANCE:
        raise RuntimeError(f"Convexity sanity failed in independent verification: {dataset_id}")

    return {
        "dataset_id": dataset_id,
        "animal_occurrences": int(len(animals)),
        "unique_cats": int(animals["cat_id"].nunique()),
        "seed_repeat_units": int(len(seed_frame)),
        "call_formula_max_abs_difference": call_formula_max,
        "animal_formula_max_abs_difference": animal_formula_max,
        "call_mean_vs_animal_max_abs_difference": call_animal_max,
        "maximum_saved_metric_difference": max_saved_metric_difference,
        "CE_max_convexity_margin": float(ce_margins.max()),
        "Brier_max_convexity_margin": float(brier_margins.max()),
        "absolute_split_analysis": {
            "definition": "For each repeat-by-fold cell, average fold Macro-F1 equally across base seeds.",
            "worst_absolute_macro_f1_by_pipeline": worst_absolute,
            "U1_worst_split_detail": u1_worst_split_detail,
        },
        "core_counts": complementarity,
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-subdir",
        default="meowagenet_idea083_C1_U1_equal_weight_fusion_v1",
    )
    args = parser.parse_args()
    run_root = (RUNS_ROOT / args.run_subdir).resolve()
    if RUNS_ROOT.resolve() not in run_root.parents:
        raise ValueError("--run-subdir must stay below runs")

    summary_path = run_root / "analysis_summary.json"
    manifest_path = run_root / "run_manifest.json"
    input_hash_path = run_root / "input_file_hashes.json"
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    input_hashes = read_json(input_hash_path)
    if summary["global_audit"]["outer_test_accessed"] is not False:
        raise RuntimeError("Summary claims outer-test access")
    if summary["global_audit"]["training_performed"] is not False:
        raise RuntimeError("Summary claims training")
    if summary["global_audit"]["gpu_used"] is not False:
        raise RuntimeError("Summary claims GPU use")
    if summary["fusion"] != {
        "C1_weight": 0.5,
        "U1_weight": 0.5,
        "formula": "pE = 0.5*pC1 + 0.5*pU1",
        "oracle_routing_used": False,
        "weight_selected_from_results": False,
    }:
        raise RuntimeError("Fusion lock changed")

    for name, info in manifest["outputs"].items():
        path = run_root / name
        if not path.is_file() or sha256_file(path) != info["sha256"]:
            raise RuntimeError(f"Output hash mismatch: {path}")
    input_failures = []
    for relative, expected in input_hashes.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            input_failures.append(relative)
    if input_failures:
        raise RuntimeError(f"Input hash failures: {input_failures[:3]}")

    animal_frame = pd.read_csv(run_root / "animal_occurrences.csv", dtype={"cat_id": str})
    call_frame = pd.read_csv(
        run_root / "ensemble_call_predictions.csv",
        dtype={"call_id": str, "cat_id": str},
    )
    fold_frame = pd.read_csv(run_root / "fold_metrics.csv")
    dataset_audits = {
        dataset_id: recompute_dataset(
            dataset_id,
            animal_frame,
            call_frame,
            fold_frame,
            saved,
        )
        for dataset_id, saved in summary["datasets"].items()
    }
    audit = {
        "analysis_id": summary["analysis_id"],
        "status": "PASS",
        "date": "2026-09-19",
        "analysis_summary_path": str(summary_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "analysis_summary_sha256": sha256_file(summary_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "input_hashes_checked": int(len(input_hashes)),
        "input_hash_failures": 0,
        "output_hashes_checked": int(len(manifest["outputs"])),
        "output_hash_failures": 0,
        "outer_test_accessed": False,
        "training_performed": False,
        "gpu_used": False,
        "dataset_audits": dataset_audits,
        "checks": {
            "fixed_equal_weight_formula": True,
            "call_level_formula": True,
            "animal_level_formula": True,
            "call_mean_equals_animal_mean": True,
            "seed_repeat_metrics_recomputed": True,
            "fold_and_split_metrics_recomputed": True,
            "complementarity_counts_recomputed": True,
            "CE_and_Brier_convexity": True,
            "oracle_not_treated_as_model_score": True,
        },
    }
    output_path = run_root / "independent_verification.json"
    write_json(output_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
