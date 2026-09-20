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
CALL_COLUMNS = (
    "call_index",
    "call_id",
    "cat_id",
    "true_label",
    *PROBABILITY_COLUMNS,
)
ANIMAL_COLUMNS = (
    "cat_id",
    "true_label",
    "call_count",
    *PROBABILITY_COLUMNS,
    "predicted_label",
)
PIPELINES = {
    "A0": "A0_ast_only",
    "U1": "U1_wide_unbounded_additive",
    "C1": "C1_bounded_wide_additive",
}
RUN_SPECS = (
    {
        "dataset_id": "IDEA076_primary",
        "evidence_role": "primary_posthoc_exploratory",
        "run_subdir": "meowagenet_idea076_C1_final_seed_confirmation_v1",
    },
    {
        "dataset_id": "IDEA072_historical_sensitivity",
        "evidence_role": "historical_sensitivity_only",
        "run_subdir": "meowagenet_idea072_C1_seed_confirmation_v1",
    },
    {
        "dataset_id": "IDEA073_historical_sensitivity",
        "evidence_role": "historical_sensitivity_only",
        "run_subdir": "meowagenet_idea073_soft_radial_budget_residual_v1",
    },
)
TOLERANCE = 1.0e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_validation_path(path: Path) -> None:
    lowered = str(path).lower().replace("\\", "/")
    if "outer_test" in lowered or "outer-test" in lowered:
        raise RuntimeError(f"Outer-test path is forbidden: {path}")
    if "validation_" not in path.name:
        raise RuntimeError(f"Only saved validation predictions are allowed: {path}")


def validate_probabilities(frame: pd.DataFrame, path: Path) -> None:
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"Non-finite probability in {path}")
    if (probabilities < -1.0e-8).any() or (probabilities > 1.0 + 1.0e-8).any():
        raise RuntimeError(f"Probability outside [0,1] in {path}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0):
        raise RuntimeError(f"Probabilities do not sum to one in {path}")


def calls_to_animals(calls: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cat_id, group in calls.groupby("cat_id", sort=True):
        labels = group["true_label"].unique()
        if len(labels) != 1:
            raise RuntimeError(f"Inconsistent labels for cat {cat_id}")
        probabilities = group[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float).mean(axis=0)
        rows.append(
            {
                "cat_id": str(cat_id),
                "true_label": int(labels[0]),
                "call_count": int(len(group)),
                **{
                    column: float(probabilities[index])
                    for index, column in enumerate(PROBABILITY_COLUMNS)
                },
                "predicted_label": int(probabilities.argmax()),
            }
        )
    return pd.DataFrame(rows, columns=ANIMAL_COLUMNS)


def metric_bundle(frame: pd.DataFrame) -> dict[str, float]:
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    labels = frame["true_label"].to_numpy(dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    targets = np.eye(3, dtype=float)[labels]
    recalls = recall_score(labels, predictions, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "macro_f1": float(
            f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "cross_entropy": float(
            -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1.0e-12, 1.0)).mean()
        ),
        "brier": float(np.mean(np.sum((probabilities - targets) ** 2, axis=1))),
        "senior_recall": float(recalls[2]),
    }


def sign_summary(values: pd.Series) -> dict[str, Any]:
    array = values.to_numpy(dtype=float)
    positive = int((array > TOLERANCE).sum())
    tied = int((np.abs(array) <= TOLERANCE).sum())
    negative = int((array < -TOLERANCE).sum())
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "positive": positive,
        "tied": tied,
        "negative": negative,
        "worst": float(array.min()),
        "best": float(array.max()),
    }


def complementarity_counts(frame: pd.DataFrame) -> dict[str, Any]:
    c1_correct = frame["C1_correct"].to_numpy(dtype=bool)
    u1_correct = frame["U1_correct"].to_numpy(dtype=bool)
    e_correct = frame["E_correct"].to_numpy(dtype=bool)
    c1_pred = frame["C1_predicted_label"].to_numpy(dtype=int)
    u1_pred = frame["U1_predicted_label"].to_numpy(dtype=int)
    both_wrong = (~c1_correct) & (~u1_correct)
    either_wrong = (~c1_correct) | (~u1_correct)
    either_correct = c1_correct | u1_correct
    union_errors = int(either_wrong.sum())
    both_wrong_count = int(both_wrong.sum())
    return {
        "animal_occurrences": int(len(frame)),
        "unique_cats": int(frame["cat_id"].nunique()),
        "both_correct": int((c1_correct & u1_correct).sum()),
        "C1_wrong_U1_correct": int(((~c1_correct) & u1_correct).sum()),
        "U1_wrong_C1_correct": int((c1_correct & (~u1_correct)).sum()),
        "both_wrong": both_wrong_count,
        "different_predictions": int((c1_pred != u1_pred).sum()),
        "same_predictions": int((c1_pred == u1_pred).sum()),
        "both_wrong_same_prediction": int((both_wrong & (c1_pred == u1_pred)).sum()),
        "both_wrong_different_prediction": int((both_wrong & (c1_pred != u1_pred)).sum()),
        "C1_error_occurrences": int((~c1_correct).sum()),
        "U1_error_occurrences": int((~u1_correct).sum()),
        "error_intersection": both_wrong_count,
        "error_union": union_errors,
        "error_overlap_jaccard": (
            float(both_wrong_count / union_errors) if union_errors else 1.0
        ),
        "E_correct_occurrences": int(e_correct.sum()),
        "E_wrong_occurrences": int((~e_correct).sum()),
        "E_vs_C1_corrected": int(((~c1_correct) & e_correct).sum()),
        "E_vs_C1_damaged": int((c1_correct & (~e_correct)).sum()),
        "E_vs_C1_net_correct": int(((~c1_correct) & e_correct).sum() - (c1_correct & (~e_correct)).sum()),
        "E_vs_U1_corrected": int(((~u1_correct) & e_correct).sum()),
        "E_vs_U1_damaged": int((u1_correct & (~e_correct)).sum()),
        "E_vs_U1_net_correct": int(((~u1_correct) & e_correct).sum() - (u1_correct & (~e_correct)).sum()),
        "ensemble_correct_when_at_least_one_parent_correct": int((e_correct & either_correct).sum()),
        "ensemble_wrong_when_at_least_one_parent_correct": int(((~e_correct) & either_correct).sum()),
        "oracle_upper_bound_correct_occurrences": int(either_correct.sum()),
        "oracle_upper_bound_fraction": float(either_correct.mean()),
        "oracle_is_model_score": False,
    }


def prefixed_animal_frame(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    result = frame[["cat_id", "true_label", "call_count"]].copy()
    for column in PROBABILITY_COLUMNS:
        result[f"{prefix}_{column}"] = frame[column].to_numpy(dtype=float)
    result[f"{prefix}_predicted_label"] = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float).argmax(axis=1)
    result[f"{prefix}_correct"] = result[f"{prefix}_predicted_label"] == result["true_label"]
    return result


def validate_saved_animals(calls: pd.DataFrame, animals: pd.DataFrame, path: Path) -> float:
    rebuilt = calls_to_animals(calls).sort_values("cat_id").reset_index(drop=True)
    saved = animals.sort_values("cat_id").reset_index(drop=True)
    for column in ("cat_id", "true_label", "call_count", "predicted_label"):
        if rebuilt[column].tolist() != saved[column].tolist():
            raise RuntimeError(f"Saved animal identity/label mismatch for {path}: {column}")
    difference = np.abs(
        rebuilt[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        - saved[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    )
    maximum = float(difference.max(initial=0.0))
    if maximum > TOLERANCE:
        raise RuntimeError(f"Saved animal probabilities do not rebuild from calls: {path} ({maximum})")
    return maximum


def read_fit_bundle(
    run_root: Path,
    pipeline: str,
    base_seed: int,
    repeat: int,
    fold: int,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    fit_root = run_root / "fits" / pipeline / f"base_seed_{base_seed}" / f"repeat_{repeat}" / f"fold_{fold}"
    fit_path = fit_root / "fit_summary.json"
    fit = read_json(fit_path)
    input_hashes[str(fit_path.relative_to(REPO_ROOT)).replace("\\", "/")] = sha256_file(fit_path)
    expected_identity = {
        "status": "complete",
        "pipeline": pipeline,
        "base_seed": base_seed,
        "full_seed": base_seed + 10_000 * repeat + 100 * fold,
        "repeat": repeat,
        "fold": fold,
        "outer_test_accessed": False,
    }
    for key, expected in expected_identity.items():
        if fit.get(key) != expected:
            raise RuntimeError(f"Fit identity mismatch for {fit_path}: {key}")
    call_path = REPO_ROOT / fit["validation_call_predictions"]
    animal_path = REPO_ROOT / fit["validation_animal_predictions"]
    for path in (call_path, animal_path):
        ensure_validation_path(path)
    if sha256_file(call_path) != fit["validation_call_sha256"]:
        raise RuntimeError(f"Call prediction hash mismatch: {call_path}")
    if sha256_file(animal_path) != fit["validation_animal_sha256"]:
        raise RuntimeError(f"Animal prediction hash mismatch: {animal_path}")
    input_hashes[str(call_path.relative_to(REPO_ROOT)).replace("\\", "/")] = fit["validation_call_sha256"]
    input_hashes[str(animal_path.relative_to(REPO_ROOT)).replace("\\", "/")] = fit["validation_animal_sha256"]
    calls = pd.read_csv(call_path, dtype={"call_id": str, "cat_id": str})
    animals = pd.read_csv(animal_path, dtype={"cat_id": str})
    if tuple(calls.columns) != CALL_COLUMNS:
        raise RuntimeError(f"Unexpected call probability column order: {call_path}")
    if tuple(animals.columns) != ANIMAL_COLUMNS:
        raise RuntimeError(f"Unexpected animal probability column order: {animal_path}")
    validate_probabilities(calls, call_path)
    validate_probabilities(animals, animal_path)
    reaggregation_max = validate_saved_animals(calls, animals, animal_path)
    return {
        "fit": fit,
        "calls": calls,
        "animals": animals,
        "reaggregation_max": reaggregation_max,
    }


def analyze_dataset(spec: dict[str, str], input_hashes: dict[str, str]) -> dict[str, Any]:
    run_root = RUNS_ROOT / spec["run_subdir"]
    manifest_path = run_root / "run_manifest.json"
    manifest = read_json(manifest_path)
    input_hashes[str(manifest_path.relative_to(REPO_ROOT)).replace("\\", "/")] = sha256_file(manifest_path)
    if manifest.get("outer_test_accessed") is not False:
        raise RuntimeError(f"Historical run is not validation-only: {manifest_path}")
    if manifest["model"].get("outer_test_predictions") is not False:
        raise RuntimeError(f"Historical run permits outer test: {manifest_path}")
    for pipeline in PIPELINES.values():
        if pipeline not in manifest["pipelines"]:
            raise RuntimeError(f"Required pipeline missing from {manifest_path}: {pipeline}")

    fold_rows: list[dict[str, Any]] = []
    animal_rows: list[pd.DataFrame] = []
    call_rows: list[pd.DataFrame] = []
    max_reaggregation_difference = 0.0
    max_call_vs_animal_fusion_difference = 0.0
    role_signatures: dict[tuple[int, int], tuple[str, ...]] = {}

    for base_seed in manifest["model"]["base_seeds"]:
        for repeat in manifest["model"]["repeats"]:
            for fold in manifest["model"]["folds"]:
                bundles = {
                    short: read_fit_bundle(
                        run_root, pipeline, int(base_seed), int(repeat), int(fold), input_hashes
                    )
                    for short, pipeline in PIPELINES.items()
                }
                max_reaggregation_difference = max(
                    max_reaggregation_difference,
                    *(bundle["reaggregation_max"] for bundle in bundles.values()),
                )
                reference_calls = bundles["A0"]["calls"]
                reference_identity = reference_calls[["call_index", "call_id", "cat_id", "true_label"]]
                for short in ("C1", "U1"):
                    candidate_identity = bundles[short]["calls"][["call_index", "call_id", "cat_id", "true_label"]]
                    if not reference_identity.equals(candidate_identity):
                        raise RuntimeError(
                            f"Call identity/order mismatch for {spec['dataset_id']} seed={base_seed} repeat={repeat} fold={fold}"
                        )

                cats = tuple(sorted(reference_calls["cat_id"].unique().tolist()))
                signature_key = (int(repeat), int(fold))
                if signature_key in role_signatures and role_signatures[signature_key] != cats:
                    raise RuntimeError(
                        f"Validation role mismatch across base seeds for {spec['dataset_id']} repeat={repeat} fold={fold}"
                    )
                role_signatures[signature_key] = cats

                c1_calls = bundles["C1"]["calls"]
                u1_calls = bundles["U1"]["calls"]
                ensemble_calls = reference_identity.copy()
                ensemble_calls[list(PROBABILITY_COLUMNS)] = 0.5 * (
                    c1_calls[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
                    + u1_calls[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
                )
                validate_probabilities(ensemble_calls, Path("in_memory_ensemble_calls"))
                ensemble_from_calls = calls_to_animals(ensemble_calls).sort_values("cat_id").reset_index(drop=True)

                sorted_animals = {
                    short: bundle["animals"].sort_values("cat_id").reset_index(drop=True)
                    for short, bundle in bundles.items()
                }
                base_identity = sorted_animals["A0"][["cat_id", "true_label", "call_count"]]
                for short in ("C1", "U1"):
                    if not base_identity.equals(
                        sorted_animals[short][["cat_id", "true_label", "call_count"]]
                    ):
                        raise RuntimeError(
                            f"Animal identity/label/role mismatch for {spec['dataset_id']} seed={base_seed} repeat={repeat} fold={fold}"
                        )

                ensemble_direct = base_identity.copy()
                ensemble_direct[list(PROBABILITY_COLUMNS)] = 0.5 * (
                    sorted_animals["C1"][list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
                    + sorted_animals["U1"][list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
                )
                ensemble_direct["predicted_label"] = ensemble_direct[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float).argmax(axis=1)
                direct_difference = np.abs(
                    ensemble_from_calls[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
                    - ensemble_direct[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
                )
                cell_max = float(direct_difference.max(initial=0.0))
                max_call_vs_animal_fusion_difference = max(
                    max_call_vs_animal_fusion_difference, cell_max
                )
                if cell_max > TOLERANCE:
                    raise RuntimeError(
                        f"Call-first and animal-first fusion differ: {spec['dataset_id']} seed={base_seed} repeat={repeat} fold={fold} ({cell_max})"
                    )

                metric_frames = {
                    "A0": sorted_animals["A0"],
                    "C1": sorted_animals["C1"],
                    "U1": sorted_animals["U1"],
                    "E": ensemble_direct,
                }
                fold_row: dict[str, Any] = {
                    "dataset_id": spec["dataset_id"],
                    "base_seed": int(base_seed),
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "animal_occurrences": int(len(ensemble_direct)),
                    "unique_cats": int(ensemble_direct["cat_id"].nunique()),
                }
                for short, frame in metric_frames.items():
                    for metric, value in metric_bundle(frame).items():
                        fold_row[f"{short}_{metric}"] = value
                for reference in ("A0", "C1", "U1"):
                    fold_row[f"E_minus_{reference}_macro_f1"] = (
                        fold_row["E_macro_f1"] - fold_row[f"{reference}_macro_f1"]
                    )
                fold_rows.append(fold_row)

                merged = prefixed_animal_frame(sorted_animals["A0"], "A0")
                for short in ("C1", "U1"):
                    prefixed = prefixed_animal_frame(sorted_animals[short], short)
                    merged = merged.merge(
                        prefixed.drop(columns=["true_label", "call_count"]),
                        on="cat_id",
                        validate="one_to_one",
                    )
                prefixed_e = prefixed_animal_frame(ensemble_direct, "E")
                merged = merged.merge(
                    prefixed_e.drop(columns=["true_label", "call_count"]),
                    on="cat_id",
                    validate="one_to_one",
                )
                merged.insert(0, "fold", int(fold))
                merged.insert(0, "repeat", int(repeat))
                merged.insert(0, "base_seed", int(base_seed))
                merged.insert(0, "evidence_role", spec["evidence_role"])
                merged.insert(0, "dataset_id", spec["dataset_id"])
                animal_rows.append(merged)

                call_output = reference_identity.copy()
                for short, frame in (("C1", c1_calls), ("U1", u1_calls), ("E", ensemble_calls)):
                    for column in PROBABILITY_COLUMNS:
                        call_output[f"{short}_{column}"] = frame[column].to_numpy(dtype=float)
                call_output.insert(0, "fold", int(fold))
                call_output.insert(0, "repeat", int(repeat))
                call_output.insert(0, "base_seed", int(base_seed))
                call_output.insert(0, "evidence_role", spec["evidence_role"])
                call_output.insert(0, "dataset_id", spec["dataset_id"])
                call_rows.append(call_output)

    fold_frame = pd.DataFrame(fold_rows).sort_values(
        ["base_seed", "repeat", "fold"]
    ).reset_index(drop=True)
    animal_frame = pd.concat(animal_rows, ignore_index=True)
    call_frame = pd.concat(call_rows, ignore_index=True)
    key_columns = ["base_seed", "repeat", "fold", "cat_id"]
    if animal_frame.duplicated(key_columns).any():
        raise RuntimeError(f"Duplicate animal occurrence key in {spec['dataset_id']}")
    if call_frame.duplicated(["base_seed", "repeat", "fold", "call_index", "call_id"]).any():
        raise RuntimeError(f"Duplicate call occurrence key in {spec['dataset_id']}")

    seed_repeat_rows: list[dict[str, Any]] = []
    complementarity_rows: list[dict[str, Any]] = []
    for (base_seed, repeat), group in animal_frame.groupby(["base_seed", "repeat"], sort=True):
        row: dict[str, Any] = {
            "dataset_id": spec["dataset_id"],
            "base_seed": int(base_seed),
            "repeat": int(repeat),
            "animal_occurrences": int(len(group)),
            "unique_cats": int(group["cat_id"].nunique()),
        }
        for short in ("A0", "C1", "U1", "E"):
            metric_frame = pd.DataFrame(
                {
                    "true_label": group["true_label"].to_numpy(dtype=int),
                    **{
                        column: group[f"{short}_{column}"].to_numpy(dtype=float)
                        for column in PROBABILITY_COLUMNS
                    },
                }
            )
            for metric, value in metric_bundle(metric_frame).items():
                row[f"{short}_{metric}"] = value
        for reference in ("A0", "C1", "U1"):
            for metric in (
                "macro_f1",
                "balanced_accuracy",
                "cross_entropy",
                "brier",
                "senior_recall",
            ):
                row[f"E_minus_{reference}_{metric}"] = (
                    row[f"E_{metric}"] - row[f"{reference}_{metric}"]
                )
        ce_bound = 0.5 * (row["C1_cross_entropy"] + row["U1_cross_entropy"])
        brier_bound = 0.5 * (row["C1_brier"] + row["U1_brier"])
        row["CE_convexity_margin_E_minus_parent_average"] = row["E_cross_entropy"] - ce_bound
        row["Brier_convexity_margin_E_minus_parent_average"] = row["E_brier"] - brier_bound
        if row["CE_convexity_margin_E_minus_parent_average"] > TOLERANCE:
            raise RuntimeError(f"CE convexity sanity failed for {spec['dataset_id']} {base_seed}/{repeat}")
        if row["Brier_convexity_margin_E_minus_parent_average"] > TOLERANCE:
            raise RuntimeError(f"Brier convexity sanity failed for {spec['dataset_id']} {base_seed}/{repeat}")
        seed_repeat_rows.append(row)
        comp = complementarity_counts(group)
        complementarity_rows.append(
            {
                "dataset_id": spec["dataset_id"],
                "base_seed": int(base_seed),
                "repeat": int(repeat),
                **comp,
            }
        )

    seed_repeat_frame = pd.DataFrame(seed_repeat_rows).sort_values(
        ["base_seed", "repeat"]
    ).reset_index(drop=True)
    complementarity_frame = pd.DataFrame(complementarity_rows).sort_values(
        ["base_seed", "repeat"]
    ).reset_index(drop=True)

    split_rows: list[dict[str, Any]] = []
    for (repeat, fold), group in fold_frame.groupby(["repeat", "fold"], sort=True):
        row = {
            "dataset_id": spec["dataset_id"],
            "repeat": int(repeat),
            "fold": int(fold),
        }
        for reference in ("A0", "C1", "U1"):
            column = f"E_minus_{reference}_macro_f1"
            row[column] = float(group[column].mean())
        split_rows.append(row)
    split_frame = pd.DataFrame(split_rows)

    pipeline_means: dict[str, dict[str, float]] = {}
    for short in ("A0", "C1", "U1", "E"):
        pipeline_means[short] = {
            metric: float(seed_repeat_frame[f"{short}_{metric}"].mean())
            for metric in (
                "macro_f1",
                "balanced_accuracy",
                "cross_entropy",
                "brier",
                "senior_recall",
            )
        }
    comparisons: dict[str, Any] = {}
    for reference in ("A0", "C1", "U1"):
        macro_column = f"E_minus_{reference}_macro_f1"
        comparison = {
            "macro_f1": sign_summary(seed_repeat_frame[macro_column]),
            "mean_metric_deltas_E_minus_reference": {
                metric: float(seed_repeat_frame[f"E_minus_{reference}_{metric}"].mean())
                for metric in (
                    "balanced_accuracy",
                    "cross_entropy",
                    "brier",
                    "senior_recall",
                )
            },
        }
        split_values = split_frame[macro_column]
        comparison["split_cells"] = {
            **sign_summary(split_values),
            "nonnegative": int((split_values >= -TOLERANCE).sum()),
            "cells": int(len(split_values)),
        }
        comparisons[f"E_minus_{reference}"] = comparison

    overall_complementarity = complementarity_counts(animal_frame)
    fold_overlap = (
        animal_frame.groupby(["base_seed", "repeat", "cat_id"])["fold"]
        .nunique()
        .reset_index(name="validation_fold_occurrences")
    )
    summary = {
        "dataset_id": spec["dataset_id"],
        "evidence_role": spec["evidence_role"],
        "source_run": str(run_root.relative_to(REPO_ROOT)).replace("\\", "/"),
        "fusion": "pE = 0.5*pC1 + 0.5*pU1",
        "base_seeds": [int(value) for value in manifest["model"]["base_seeds"]],
        "repeats": [int(value) for value in manifest["model"]["repeats"]],
        "folds": [int(value) for value in manifest["model"]["folds"]],
        "paired_folds": int(len(fold_frame)),
        "seed_repeat_units": int(len(seed_repeat_frame)),
        "split_cells": int(len(split_frame)),
        "animal_occurrences": int(len(animal_frame)),
        "unique_cats": int(animal_frame["cat_id"].nunique()),
        "cat_fold_reuse": {
            "base_seed_repeat_cat_rows": int(len(fold_overlap)),
            "rows_seen_in_multiple_validation_folds": int(
                (fold_overlap["validation_fold_occurrences"] > 1).sum()
            ),
            "maximum_validation_folds_for_one_cat_within_seed_repeat": int(
                fold_overlap["validation_fold_occurrences"].max()
            ),
        },
        "pipeline_seed_repeat_equal_weight_means": pipeline_means,
        "comparisons": comparisons,
        "complementarity_animal_occurrences": overall_complementarity,
        "convexity_sanity": {
            "CE_all_seed_repeats_pass": bool(
                (seed_repeat_frame["CE_convexity_margin_E_minus_parent_average"] <= TOLERANCE).all()
            ),
            "CE_max_margin": float(
                seed_repeat_frame["CE_convexity_margin_E_minus_parent_average"].max()
            ),
            "Brier_all_seed_repeats_pass": bool(
                (seed_repeat_frame["Brier_convexity_margin_E_minus_parent_average"] <= TOLERANCE).all()
            ),
            "Brier_max_margin": float(
                seed_repeat_frame["Brier_convexity_margin_E_minus_parent_average"].max()
            ),
        },
        "audits": {
            "outer_test_accessed": False,
            "training_performed": False,
            "gpu_used": False,
            "prediction_hashes_checked": int(len(manifest["model"]["base_seeds"]) * 3 * 4 * 3 * 2),
            "fit_summaries_checked": int(len(manifest["model"]["base_seeds"]) * 3 * 4 * 3),
            "probability_column_order_checked": True,
            "paired_call_identity_checked": True,
            "paired_animal_identity_label_role_checked": True,
            "role_signatures_match_across_base_seeds": True,
            "maximum_saved_animal_reaggregation_difference": max_reaggregation_difference,
            "maximum_call_first_vs_animal_first_fusion_difference": max_call_vs_animal_fusion_difference,
        },
        "independence_warning": (
            "Animal rows are validation occurrences. The same cats recur across seeds, repeats, "
            "and sometimes folds; occurrence counts and seed-repeat estimates are not independent cats."
        ),
        "oracle_warning": (
            "Oracle fields only count whether at least one parent was correct and are a potential upper bound, "
            "not model predictions or model performance."
        ),
    }
    return {
        "summary": summary,
        "fold_frame": fold_frame,
        "seed_repeat_frame": seed_repeat_frame,
        "split_frame": split_frame,
        "complementarity_frame": complementarity_frame,
        "animal_frame": animal_frame,
        "call_frame": call_frame,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-subdir",
        default="meowagenet_idea083_C1_U1_equal_weight_fusion_v1",
    )
    args = parser.parse_args()
    output_root = (RUNS_ROOT / args.output_subdir).resolve()
    if RUNS_ROOT.resolve() not in output_root.parents:
        raise ValueError("--output-subdir must stay below runs")
    output_root.mkdir(parents=True, exist_ok=True)

    input_hashes: dict[str, str] = {}
    results = [analyze_dataset(spec, input_hashes) for spec in RUN_SPECS]
    summary = {
        "analysis_id": "meowagenet-idea083-C1-U1-equal-weight-fusion-v1",
        "date": "2026-09-19",
        "status": "complete_pending_independent_verification",
        "posthoc_exploratory": True,
        "fusion": {
            "formula": "pE = 0.5*pC1 + 0.5*pU1",
            "C1_weight": 0.5,
            "U1_weight": 0.5,
            "weight_selected_from_results": False,
            "oracle_routing_used": False,
        },
        "primary_dataset_id": "IDEA076_primary",
        "sensitivity_dataset_ids": [
            "IDEA072_historical_sensitivity",
            "IDEA073_historical_sensitivity",
        ],
        "datasets": {result["summary"]["dataset_id"]: result["summary"] for result in results},
        "global_audit": {
            "source_runs_read_only": True,
            "outer_test_accessed": False,
            "training_performed": False,
            "gpu_used": False,
            "input_files_hashed": int(len(input_hashes)),
            "input_hash_ledger_sha256": hashlib.sha256(
                "\n".join(f"{path}:{digest}" for path, digest in sorted(input_hashes.items())).encode("utf-8")
            ).hexdigest(),
        },
        "decision_boundary": (
            "IDEA-083 is a post-hoc exploratory analysis of fixed historical validation predictions. "
            "It does not alter IDEA-076 gates, choose a fusion weight, or authorize further same-data tuning."
        ),
    }

    pd.concat([result["fold_frame"] for result in results], ignore_index=True).to_csv(
        output_root / "fold_metrics.csv", index=False
    )
    pd.concat([result["seed_repeat_frame"] for result in results], ignore_index=True).to_csv(
        output_root / "seed_repeat_metrics.csv", index=False
    )
    pd.concat([result["split_frame"] for result in results], ignore_index=True).to_csv(
        output_root / "split_cell_metrics.csv", index=False
    )
    pd.concat([result["complementarity_frame"] for result in results], ignore_index=True).to_csv(
        output_root / "seed_repeat_complementarity.csv", index=False
    )
    pd.concat([result["animal_frame"] for result in results], ignore_index=True).to_csv(
        output_root / "animal_occurrences.csv", index=False
    )
    pd.concat([result["call_frame"] for result in results], ignore_index=True).to_csv(
        output_root / "ensemble_call_predictions.csv", index=False
    )
    write_json(output_root / "input_file_hashes.json", input_hashes)
    write_json(output_root / "analysis_summary.json", summary)

    output_files = {}
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            output_files[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    run_manifest = {
        "analysis_id": summary["analysis_id"],
        "status": summary["status"],
        "outer_test_accessed": False,
        "training_performed": False,
        "gpu_used": False,
        "source_run_manifests": {
            spec["dataset_id"]: {
                "path": f"runs/{spec['run_subdir']}/run_manifest.json",
                "sha256": input_hashes[f"runs/{spec['run_subdir']}/run_manifest.json"],
            }
            for spec in RUN_SPECS
        },
        "outputs": output_files,
    }
    write_json(output_root / "run_manifest.json", run_manifest)
    print(json.dumps({
        "status": summary["status"],
        "output_root": str(output_root),
        "primary": summary["datasets"]["IDEA076_primary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
