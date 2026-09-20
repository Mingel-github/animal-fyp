"""Independently recompute IDEA-084 results from saved validation predictions.

This verifier deliberately does not import or call the IDEA-084 runner/aggregate.
It reads the locked protocol, fit ledgers, and prediction CSVs only.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "configs/protocol/meowagenet_idea084_grouped_dual_branch_acoustic_residual_v1.json"
)
RUNNER_PATH = (
    REPO_ROOT / "scripts/run_meowagenet_idea084_grouped_dual_branch_acoustic_residual.py"
)
RUN_ROOT = REPO_ROOT / "runs/meowagenet_idea084_grouped_dual_branch_acoustic_residual_v1"
MANIFEST_PATH = RUN_ROOT / "run_manifest.json"
SUMMARY_PATH = RUN_ROOT / "initial_evaluation_summary.json"
AUDIT_PATH = RUN_ROOT / "independent_results_audit.json"

LOCKED_PROTOCOL_SHA256 = (
    "0c4b50f2a6903bb5e55157591e5a93400c56602161fa2b1f8c8fd0d112bb7eef"
)
LOCKED_RUNNER_SHA256 = (
    "fa3913fd7b2a7a66f3a4e61deb150ef7358cc32ee5b65eafdfe9d2c7e2abe12f"
)
PIPELINES = (
    "A0_ast_only",
    "C1_bounded_wide_additive",
    "P1_grouped_dual_branch",
    "R1_hash_random_dual_branch",
)
BASE_SEEDS = (3583, 5080, 9355)
REPEATS = (0, 1, 2)
FOLDS = (0, 1, 2, 3)
COMPARISONS = {
    "P1_minus_C1": ("P1_grouped_dual_branch", "C1_bounded_wide_additive"),
    "P1_minus_A0": ("P1_grouped_dual_branch", "A0_ast_only"),
    "P1_minus_R1": (
        "P1_grouped_dual_branch",
        "R1_hash_random_dual_branch",
    ),
}
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
TOLERANCE = 1.0e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def confusion(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        matrix[int(label), int(prediction)] += 1
    return matrix


def metric_bundle(frame: pd.DataFrame) -> dict[str, float]:
    labels = frame["true_label"].to_numpy(dtype=np.int64)
    predictions = frame["predicted_label"].to_numpy(dtype=np.int64)
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    matrix = confusion(labels, predictions)
    if np.any(matrix.sum(axis=1) == 0):
        raise RuntimeError("IDEA-084 verification encountered a missing true class")
    recall = np.diag(matrix) / matrix.sum(axis=1)
    precision = np.divide(
        np.diag(matrix),
        matrix.sum(axis=0),
        out=np.zeros(3, dtype=np.float64),
        where=matrix.sum(axis=0) != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(3, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    targets = np.eye(3, dtype=np.float64)[labels]
    return {
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "cross_entropy": float(
            -np.log(
                np.clip(
                    probabilities[np.arange(len(labels)), labels], 1.0e-12, 1.0
                )
            ).mean()
        ),
        "brier": float(np.mean(np.sum((probabilities - targets) ** 2, axis=1))),
        "senior_recall": float(recall[2]),
    }


def validate_probabilities(frame: pd.DataFrame, identity: str) -> None:
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError(f"{identity}: non-finite probability")
    if np.any(probabilities < -1.0e-7) or np.any(probabilities > 1.0 + 1.0e-7):
        raise RuntimeError(f"{identity}: probability outside [0,1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0):
        raise RuntimeError(f"{identity}: probabilities do not sum to one")
    labels = frame["true_label"].to_numpy(dtype=np.int64)
    if np.any(labels < 0) or np.any(labels > 2):
        raise RuntimeError(f"{identity}: unexpected label")
    if "predicted_label" in frame.columns:
        predictions = frame["predicted_label"].to_numpy(dtype=np.int64)
        if not np.array_equal(predictions, probabilities.argmax(axis=1)):
            raise RuntimeError(f"{identity}: saved prediction is not probability argmax")


def reconstruct_animals(calls: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cat_id, group in calls.groupby("cat_id", sort=True):
        labels = group["true_label"].to_numpy(dtype=np.int64)
        if len(np.unique(labels)) != 1:
            raise RuntimeError(f"cat {cat_id}: inconsistent call labels")
        probabilities = group[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64).mean(
            axis=0
        )
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
    return pd.DataFrame(rows).sort_values("cat_id").reset_index(drop=True)


def compare_reconstruction(
    saved: pd.DataFrame, reconstructed: pd.DataFrame, identity: str
) -> None:
    saved = saved.sort_values("cat_id").reset_index(drop=True)
    columns_exact = ("cat_id", "true_label", "call_count", "predicted_label")
    for column in columns_exact:
        if not np.array_equal(saved[column].to_numpy(), reconstructed[column].to_numpy()):
            raise RuntimeError(f"{identity}: animal reconstruction differs in {column}")
    if not np.allclose(
        saved[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64),
        reconstructed[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64),
        atol=TOLERANCE,
        rtol=0.0,
    ):
        raise RuntimeError(f"{identity}: animal probability reconstruction differs")


def contrast_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "median": float(np.median(array)),
        "positive": int((array > 0).sum()),
        "tied": int((array == 0).sum()),
        "negative": int((array < 0).sum()),
        "worst": float(array.min()),
        "best": float(array.max()),
    }


def paired_transitions(candidate: pd.DataFrame, comparator: pd.DataFrame) -> dict[str, int]:
    keys = ["base_seed", "repeat", "fold", "cat_id"]
    left = candidate[keys + ["true_label", "predicted_label"]].rename(
        columns={"true_label": "candidate_true", "predicted_label": "candidate_prediction"}
    )
    right = comparator[keys + ["true_label", "predicted_label"]].rename(
        columns={
            "true_label": "comparator_true",
            "predicted_label": "comparator_prediction",
        }
    )
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError("paired transition rows are incomplete")
    if not np.array_equal(merged["candidate_true"], merged["comparator_true"]):
        raise RuntimeError("paired transition labels differ")
    candidate_correct = merged["candidate_prediction"] == merged["candidate_true"]
    comparator_correct = merged["comparator_prediction"] == merged["comparator_true"]
    corrected = int((candidate_correct & ~comparator_correct).sum())
    introduced = int((~candidate_correct & comparator_correct).sum())
    return {
        "paired_occurrences": int(len(merged)),
        "corrected_errors": corrected,
        "introduced_errors": introduced,
        "net_corrections": corrected - introduced,
        "unchanged_correct": int((candidate_correct & comparator_correct).sum()),
        "unchanged_wrong": int((~candidate_correct & ~comparator_correct).sum()),
    }


def add_metrics(row: dict[str, Any], bundles: dict[str, dict[str, float]]) -> None:
    for pipeline, values in bundles.items():
        for metric, value in values.items():
            row[f"{pipeline}_{metric}"] = value
    for name, (candidate, comparator) in COMPARISONS.items():
        for metric in ("macro_f1", "balanced_accuracy", "senior_recall"):
            row[f"{name}_{metric}"] = (
                row[f"{candidate}_{metric}"] - row[f"{comparator}_{metric}"]
            )
        for metric in ("cross_entropy", "brier"):
            row[f"{name}_{metric}_gain"] = (
                row[f"{comparator}_{metric}"] - row[f"{candidate}_{metric}"]
            )


def nested_compare(
    official: Any,
    independent: Any,
    path: str,
    differences: list[dict[str, Any]],
) -> None:
    if isinstance(independent, dict):
        if not isinstance(official, dict):
            differences.append({"path": path, "official": official, "independent": independent})
            return
        for key, value in independent.items():
            if key not in official:
                differences.append(
                    {"path": f"{path}.{key}", "official": "<missing>", "independent": value}
                )
            else:
                nested_compare(official[key], value, f"{path}.{key}", differences)
        return
    if isinstance(independent, list):
        if not isinstance(official, list) or len(official) != len(independent):
            differences.append({"path": path, "official": official, "independent": independent})
            return
        for index, value in enumerate(independent):
            nested_compare(official[index], value, f"{path}[{index}]", differences)
        return
    if isinstance(independent, (float, np.floating)):
        if not isinstance(official, (int, float)) or not math.isclose(
            float(official), float(independent), rel_tol=0.0, abs_tol=TOLERANCE
        ):
            differences.append(
                {"path": path, "official": official, "independent": float(independent)}
            )
        return
    if official != independent:
        differences.append({"path": path, "official": official, "independent": independent})


def main() -> None:
    required = (PROTOCOL_PATH, RUNNER_PATH, MANIFEST_PATH, SUMMARY_PATH)
    missing = [path.relative_to(REPO_ROOT).as_posix() for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"IDEA-084 full result is not ready: {missing}")
    if sha256(PROTOCOL_PATH) != LOCKED_PROTOCOL_SHA256:
        raise RuntimeError("IDEA-084 locked protocol hash changed")
    if sha256(RUNNER_PATH) != LOCKED_RUNNER_SHA256:
        raise RuntimeError("IDEA-084 locked runner hash changed")

    protocol = read_json(PROTOCOL_PATH)
    manifest = read_json(MANIFEST_PATH)
    official = read_json(SUMMARY_PATH)
    if manifest.get("protocol_sha256") != LOCKED_PROTOCOL_SHA256:
        raise RuntimeError("IDEA-084 manifest protocol hash differs")
    if manifest.get("runner_sha256") != LOCKED_RUNNER_SHA256:
        raise RuntimeError("IDEA-084 manifest runner hash differs")
    if manifest.get("outer_test_accessed") is not False:
        raise RuntimeError("IDEA-084 manifest says outer test was accessed")
    if official.get("status") != "complete" or official.get("fits") != 144:
        raise RuntimeError("IDEA-084 official summary is not a complete 144-fit result")
    if official.get("outer_test_accessed") is not False:
        raise RuntimeError("IDEA-084 official summary says outer test was accessed")

    roles = pd.read_csv(
        REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str}
    )
    frames: dict[tuple[int, int, int, str], pd.DataFrame] = {}
    fit_hashes: dict[str, str] = {}
    prediction_hashes: dict[str, str] = {}
    identities: set[tuple[int, int, int, str]] = set()

    for summary_path in sorted((RUN_ROOT / "fits").rglob("fit_summary.json")):
        fit = read_json(summary_path)
        identity = (
            int(fit["base_seed"]),
            int(fit["repeat"]),
            int(fit["fold"]),
            str(fit["pipeline"]),
        )
        if identity in identities:
            raise RuntimeError(f"duplicate fit identity: {identity}")
        identities.add(identity)
        if fit.get("status") != "complete" or fit.get("outer_test_accessed") is not False:
            raise RuntimeError(f"incomplete or outer-test fit: {identity}")
        base_seed, repeat, fold, pipeline = identity
        expected_full_seed = base_seed + 10_000 * repeat + 100 * fold
        if fit.get("full_seed") != expected_full_seed:
            raise RuntimeError(f"full seed mismatch: {identity}")
        fit_relative = summary_path.relative_to(REPO_ROOT).as_posix()
        fit_hashes[fit_relative] = sha256(summary_path)

        animal_path = REPO_ROOT / fit["validation_animal_predictions"]
        call_path = REPO_ROOT / fit["validation_call_predictions"]
        for path, expected_hash in (
            (animal_path, fit["validation_animal_sha256"]),
            (call_path, fit["validation_call_sha256"]),
        ):
            actual_hash = sha256(path)
            if actual_hash != expected_hash:
                raise RuntimeError(f"prediction hash mismatch: {path}")
            prediction_hashes[path.relative_to(REPO_ROOT).as_posix()] = actual_hash

        animals = pd.read_csv(animal_path, dtype={"cat_id": str})
        calls = pd.read_csv(call_path, dtype={"cat_id": str, "call_id": str})
        validate_probabilities(animals, f"{identity}/animals")
        validate_probabilities(calls, f"{identity}/calls")
        if animals["cat_id"].duplicated().any() or calls["call_id"].duplicated().any():
            raise RuntimeError(f"duplicate animal or call prediction: {identity}")
        reconstructed = reconstruct_animals(calls)
        compare_reconstruction(animals, reconstructed, str(identity))

        expected_cats = set(
            roles[
                (roles["repeat"] == repeat)
                & (roles["outer_fold"] == fold)
                & (roles["role"] == "validation")
            ]["cat_id"].astype(str)
        )
        if set(animals["cat_id"].astype(str)) != expected_cats:
            raise RuntimeError(f"validation animal role mismatch: {identity}")
        if set(calls["cat_id"].astype(str)) != expected_cats:
            raise RuntimeError(f"validation call role mismatch: {identity}")
        animals = animals.sort_values("cat_id").reset_index(drop=True)
        animals["base_seed"] = base_seed
        animals["repeat"] = repeat
        animals["fold"] = fold
        frames[identity] = animals

    expected_identities = {
        (seed, repeat, fold, pipeline)
        for seed in BASE_SEEDS
        for repeat in REPEATS
        for fold in FOLDS
        for pipeline in PIPELINES
    }
    if identities != expected_identities:
        raise RuntimeError(
            f"fit matrix differs: missing={len(expected_identities-identities)} "
            f"extra={len(identities-expected_identities)}"
        )

    for seed in BASE_SEEDS:
        for repeat in REPEATS:
            for fold in FOLDS:
                reference_animals = frames[(seed, repeat, fold, PIPELINES[0])]
                reference_ids = reference_animals["cat_id"].to_numpy()
                reference_labels = reference_animals["true_label"].to_numpy(dtype=np.int64)
                for pipeline in PIPELINES[1:]:
                    candidate = frames[(seed, repeat, fold, pipeline)]
                    if not np.array_equal(reference_ids, candidate["cat_id"].to_numpy()):
                        raise RuntimeError("paired animal IDs differ across pipelines")
                    if not np.array_equal(
                        reference_labels, candidate["true_label"].to_numpy(dtype=np.int64)
                    ):
                        raise RuntimeError("paired labels differ across pipelines")

    fold_rows: list[dict[str, Any]] = []
    seed_repeat_rows: list[dict[str, Any]] = []
    pooled_all: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
    for seed in BASE_SEEDS:
        for repeat in REPEATS:
            pooled: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
            for fold in FOLDS:
                bundles: dict[str, dict[str, float]] = {}
                for pipeline in PIPELINES:
                    frame = frames[(seed, repeat, fold, pipeline)]
                    bundles[pipeline] = metric_bundle(frame)
                    pooled[pipeline].append(frame)
                    pooled_all[pipeline].append(frame)
                row: dict[str, Any] = {
                    "base_seed": seed,
                    "repeat": repeat,
                    "fold": fold,
                }
                add_metrics(row, bundles)
                fold_rows.append(row)
            pooled_frames = {
                pipeline: pd.concat(parts, ignore_index=True)
                for pipeline, parts in pooled.items()
            }
            bundles = {
                pipeline: metric_bundle(frame) for pipeline, frame in pooled_frames.items()
            }
            row = {"base_seed": seed, "repeat": repeat}
            add_metrics(row, bundles)
            for name, (candidate, comparator) in COMPARISONS.items():
                transition = paired_transitions(
                    pooled_frames[candidate], pooled_frames[comparator]
                )
                for key, value in transition.items():
                    row[f"{name}_{key}"] = value
            seed_repeat_rows.append(row)

    fold_frame = pd.DataFrame(fold_rows)
    seed_repeat_frame = pd.DataFrame(seed_repeat_rows)
    pooled_frames = {
        pipeline: pd.concat(parts, ignore_index=True)
        for pipeline, parts in pooled_all.items()
    }
    metric_names = (
        "macro_f1",
        "balanced_accuracy",
        "cross_entropy",
        "brier",
    )
    pipeline_means = {
        metric: {
            pipeline: float(seed_repeat_frame[f"{pipeline}_{metric}"].mean())
            for pipeline in PIPELINES
        }
        for metric in metric_names
    }
    split_rows: list[dict[str, Any]] = []
    for repeat in REPEATS:
        for fold in FOLDS:
            selected = fold_frame[
                (fold_frame["repeat"] == repeat) & (fold_frame["fold"] == fold)
            ]
            split_row: dict[str, Any] = {"repeat": repeat, "fold": fold}
            for name in COMPARISONS:
                split_row[f"{name}_macro_f1"] = float(
                    selected[f"{name}_macro_f1"].mean()
                )
            split_rows.append(split_row)

    gate = protocol["gate"]
    comparison_results: dict[str, Any] = {}
    for name, (candidate, comparator) in COMPARISONS.items():
        f1_values = seed_repeat_frame[f"{name}_macro_f1"].tolist()
        split_values = np.asarray(
            [row[f"{name}_macro_f1"] for row in split_rows], dtype=np.float64
        )
        per_seed = {
            str(seed): float(
                seed_repeat_frame[seed_repeat_frame["base_seed"] == seed][
                    f"{name}_macro_f1"
                ].mean()
            )
            for seed in BASE_SEEDS
        }
        per_seed_senior: dict[str, float] = {}
        for seed in BASE_SEEDS:
            candidate_frame = pooled_frames[candidate]
            comparator_frame = pooled_frames[comparator]
            candidate_selected = candidate_frame[candidate_frame["base_seed"] == seed]
            comparator_selected = comparator_frame[comparator_frame["base_seed"] == seed]
            per_seed_senior[str(seed)] = (
                metric_bundle(candidate_selected)["senior_recall"]
                - metric_bundle(comparator_selected)["senior_recall"]
            )
        conditions = {
            "mean_macro_f1_delta": float(np.mean(f1_values))
            >= float(gate["minimum_mean_seed_repeat_macro_f1_delta"]),
            "positive_base_seed_means": sum(value > 0.0 for value in per_seed.values())
            >= int(gate["minimum_positive_base_seed_means"]),
            "positive_seed_repeats": int((np.asarray(f1_values) > 0).sum())
            >= int(gate["minimum_positive_seed_repeats"]),
            "nonnegative_split_cells": int((split_values >= 0).sum())
            >= int(gate["minimum_nonnegative_split_cells"]),
            "worst_split_cell": float(split_values.min())
            >= float(gate["minimum_worst_split_cell_delta"]),
            "mean_cross_entropy_nonworse": pipeline_means["cross_entropy"][candidate]
            <= pipeline_means["cross_entropy"][comparator],
            "mean_brier_nonworse": pipeline_means["brier"][candidate]
            <= pipeline_means["brier"][comparator],
            "mean_balanced_accuracy_nonworse": pipeline_means["balanced_accuracy"][
                candidate
            ]
            >= pipeline_means["balanced_accuracy"][comparator],
            "per_base_seed_senior_recall_safety": all(
                value >= float(gate["minimum_per_base_seed_senior_recall_delta"])
                for value in per_seed_senior.values()
            ),
        }
        correction_keys = (
            "paired_occurrences",
            "corrected_errors",
            "introduced_errors",
            "net_corrections",
            "unchanged_correct",
            "unchanged_wrong",
        )
        correction_profile = {
            key: {
                "mean_per_seed_repeat": float(
                    seed_repeat_frame[f"{name}_{key}"].mean()
                ),
                "total_descriptive_repeated_occurrences": int(
                    seed_repeat_frame[f"{name}_{key}"].sum()
                ),
            }
            for key in correction_keys
        }
        net = seed_repeat_frame[f"{name}_net_corrections"].to_numpy()
        correction_profile["net_correction_positive_tied_negative"] = {
            "positive": int((net > 0).sum()),
            "tied": int((net == 0).sum()),
            "negative": int((net < 0).sum()),
        }
        comparison_results[name] = {
            "candidate": candidate,
            "comparator": comparator,
            "macro_f1": contrast_summary(f1_values),
            "per_base_seed_mean_delta": per_seed,
            "split_cell_nonnegative": int((split_values >= 0).sum()),
            "split_cell_worst": float(split_values.min()),
            "mean_balanced_accuracy_delta": pipeline_means["balanced_accuracy"][candidate]
            - pipeline_means["balanced_accuracy"][comparator],
            "mean_cross_entropy_gain": pipeline_means["cross_entropy"][comparator]
            - pipeline_means["cross_entropy"][candidate],
            "mean_brier_gain": pipeline_means["brier"][comparator]
            - pipeline_means["brier"][candidate],
            "per_base_seed_senior_recall_delta": per_seed_senior,
            "error_correction_profile": correction_profile,
            "conditions": conditions,
            "gate_passed": bool(all(conditions.values())),
        }

    independent = {
        "pipeline_seed_repeat_means": pipeline_means,
        "comparison_results": comparison_results,
        "fold_results": fold_rows,
        "seed_repeat_results": seed_repeat_rows,
        "split_cell_results": split_rows,
    }
    official_selected = {
        "pipeline_seed_repeat_means": official["pipeline_seed_repeat_means"],
        "comparison_results": official["comparison_results"],
        "fold_results": official["fold_results"],
        "seed_repeat_results": official["seed_repeat_results"],
        "split_cell_results": official["split_cell_results"],
    }
    differences: list[dict[str, Any]] = []
    nested_compare(official_selected, independent, "results", differences)

    pooled_core = {
        pipeline: {
            "animal_occurrences": int(len(frame)),
            **metric_bundle(frame),
        }
        for pipeline, frame in pooled_frames.items()
    }
    for pipeline, values in pooled_core.items():
        official_values = official["pooled_validation"][pipeline]
        selected = {
            "animal_occurrences": official_values["animal_occurrences"],
            "macro_f1": official_values["metrics"]["macro_f1"],
            "balanced_accuracy": official_values["metrics"]["balanced_accuracy"],
            "cross_entropy": official_values["cross_entropy"],
            "brier": official_values["brier"],
            "senior_recall": official_values["metrics"]["per_class"]["senior"][
                "recall"
            ],
        }
        nested_compare(selected, values, f"pooled_validation.{pipeline}", differences)

    audit = {
        "schema_version": "1.0",
        "audit_id": "meowagenet-idea084-independent-results-audit-v1",
        "status": "PASS" if not differences else "FAIL",
        "method": "Independent reconstruction from saved call/animal validation CSVs; no import or call of the IDEA-084 runner or aggregate.",
        "locked_inputs": {
            "protocol": {
                "path": PROTOCOL_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256(PROTOCOL_PATH),
            },
            "runner": {
                "path": RUNNER_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256(RUNNER_PATH),
            },
            "run_manifest": {
                "path": MANIFEST_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256(MANIFEST_PATH),
            },
            "official_summary": {
                "path": SUMMARY_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256(SUMMARY_PATH),
            },
        },
        "integrity": {
            "fit_summaries_checked": len(fit_hashes),
            "prediction_files_checked": len(prediction_hashes),
            "animal_predictions_reconstructed_from_calls": len(frames),
            "complete_expected_fit_matrix": identities == expected_identities,
            "outer_test_accessed": False,
            "fit_summary_sha256": fit_hashes,
            "prediction_sha256": prediction_hashes,
        },
        "recomputed": independent,
        "pooled_validation_core": pooled_core,
        "comparison_to_official": {
            "tolerance": TOLERANCE,
            "difference_count": len(differences),
            "differences": differences,
        },
    }
    write_json(AUDIT_PATH, audit)
    print(json.dumps({
        "status": audit["status"],
        "fit_summaries_checked": len(fit_hashes),
        "prediction_files_checked": len(prediction_hashes),
        "difference_count": len(differences),
        "audit_path": AUDIT_PATH.relative_to(REPO_ROOT).as_posix(),
        "audit_sha256": sha256(AUDIT_PATH),
    }, indent=2, ensure_ascii=False))
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
