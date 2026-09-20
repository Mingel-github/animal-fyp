"""Independent IDEA-085 first-cell and complete-results verifier.

This module deliberately does not import the IDEA-085 runner or call its
aggregation function.  It reconstructs animal predictions from saved call
predictions and recomputes every reported result from the saved CSV files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/protocol/meowagenet_idea085_acoustic_temporal_residual_v1.json"
RUNNER_PATH = ROOT / "scripts/run_meowagenet_idea085_acoustic_temporal_residual.py"
RUN_ROOT = ROOT / "runs/meowagenet_idea085_acoustic_temporal_residual_v1"
MANIFEST_PATH = RUN_ROOT / "run_manifest.json"
CPU_PREFLIGHT_PATH = RUN_ROOT / "cpu_preflight.json"
PARTIAL_PATH = RUN_ROOT / "partial_run_summary.json"
SUMMARY_PATH = RUN_ROOT / "initial_evaluation_summary.json"
FIRST_CELL_AUDIT_PATH = RUN_ROOT / "independent_first_cell_audit.json"
FULL_AUDIT_PATH = RUN_ROOT / "independent_results_audit.json"

LOCKED_PROTOCOL_SHA256 = "592b8fccf721aeeab8f2bf20839534b703c9e5caa582c93dc220854d77f66bca"
LOCKED_RUNNER_SHA256 = "471bb5bd0e007be696decd86dec9317b3ef8c3f732490718a4eed99f28e821b4"
LOCKED_CPU_PREFLIGHT_SHA256 = "c76fcb1177ca4a14b432993efaf8b8ab2b4df99c148c3ca80405ad3456b67387"
PIPELINES = (
    "A0_ast_only",
    "C1_bounded_wide_additive",
    "T1_acoustic_temporal_residual",
    "J1_frame_shuffled_temporal_residual",
)
EXPECTED_PARAMETERS = {
    PIPELINES[0]: 99_075,
    PIPELINES[1]: 108_143,
    PIPELINES[2]: 108_355,
    PIPELINES[3]: 108_355,
}
BASE_SEEDS = (2713, 5395, 5226)
REPEATS = (0, 1, 2)
FOLDS = (0, 1, 2, 3)
COMPARISONS = {
    "T1_minus_C1": (PIPELINES[2], PIPELINES[1]),
    "T1_minus_A0": (PIPELINES[2], PIPELINES[0]),
    "T1_minus_J1": (PIPELINES[2], PIPELINES[3]),
}
DESCRIPTIVE_COMPARISONS = {
    "C1_minus_A0_descriptive": (PIPELINES[1], PIPELINES[0]),
}
CLASS_NAMES = ("kitten", "adult", "senior")
PROBABILITY_COLUMNS = tuple(f"prob_{name}" for name in CLASS_NAMES)
TOLERANCE = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("first-cell", "full"), required=True)
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
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise RuntimeError(f"non-finite numeric value: {path}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")


def confusion(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        matrix[int(label), int(prediction)] += 1
    return matrix


def metric_bundle(frame: pd.DataFrame) -> dict[str, Any]:
    labels = frame["true_label"].to_numpy(dtype=np.int64)
    predictions = frame["predicted_label"].to_numpy(dtype=np.int64)
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    matrix = confusion(labels, predictions)
    support = matrix.sum(axis=1)
    if np.any(support == 0):
        raise RuntimeError("metric bundle encountered a missing true class")
    recall = np.diag(matrix) / support
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
        "n": int(len(frame)),
        "plain_accuracy": float((labels == predictions).mean()),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "class_recall": {
            name: float(recall[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "cross_entropy": float(
            -np.log(
                np.clip(probabilities[np.arange(len(labels)), labels], 1.0e-12, 1.0)
            ).mean()
        ),
        "brier": float(np.mean(np.sum((probabilities - targets) ** 2, axis=1))),
        "confusion_matrix": matrix.tolist(),
    }


def validate_probabilities(frame: pd.DataFrame, identity: str) -> None:
    required = {"true_label", *PROBABILITY_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{identity}: missing columns {sorted(missing)}")
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
            raise RuntimeError(f"{identity}: saved label is not probability argmax")


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
    for column in ("cat_id", "true_label", "call_count", "predicted_label"):
        if not np.array_equal(saved[column].to_numpy(), reconstructed[column].to_numpy()):
            raise RuntimeError(f"{identity}: animal reconstruction differs in {column}")
    if not np.allclose(
        saved[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64),
        reconstructed[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64),
        atol=TOLERANCE,
        rtol=0.0,
    ):
        raise RuntimeError(f"{identity}: animal probability reconstruction differs")


def paired_transitions(candidate: pd.DataFrame, comparator: pd.DataFrame) -> dict[str, int]:
    keys = ["base_seed", "repeat", "fold", "cat_id"]
    left = candidate[keys + ["true_label", "predicted_label"]].rename(
        columns={"true_label": "candidate_true", "predicted_label": "candidate_prediction"}
    )
    right = comparator[keys + ["true_label", "predicted_label"]].rename(
        columns={"true_label": "comparator_true", "predicted_label": "comparator_prediction"}
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


def contrast_summary(values: list[float] | pd.Series) -> dict[str, Any]:
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


def verify_locked_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    required = (PROTOCOL_PATH, RUNNER_PATH, CPU_PREFLIGHT_PATH, MANIFEST_PATH)
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"IDEA-085 run is not ready: {missing}")
    if sha256(PROTOCOL_PATH) != LOCKED_PROTOCOL_SHA256:
        raise RuntimeError("IDEA-085 locked protocol hash changed")
    if sha256(RUNNER_PATH) != LOCKED_RUNNER_SHA256:
        raise RuntimeError("IDEA-085 locked runner hash changed")
    if sha256(CPU_PREFLIGHT_PATH) != LOCKED_CPU_PREFLIGHT_SHA256:
        raise RuntimeError("IDEA-085 locked CPU preflight hash changed")
    protocol = read_json(PROTOCOL_PATH)
    manifest = read_json(MANIFEST_PATH)
    expected_manifest = {
        "protocol_sha256": LOCKED_PROTOCOL_SHA256,
        "runner_sha256": LOCKED_RUNNER_SHA256,
        "cpu_preflight_sha256": LOCKED_CPU_PREFLIGHT_SHA256,
        "cpu_preflight_status": "GO",
        "director_authorized": True,
        "outer_test_accessed": False,
        "pipelines": list(PIPELINES),
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"IDEA-085 manifest mismatch: {key}")
    environment = manifest.get("environment", {})
    if environment.get("device") != "cuda" or not environment.get("gpu"):
        raise RuntimeError("IDEA-085 formal results were not produced on CUDA")
    return protocol, manifest


def validation_population(
    protocol: dict[str, Any], repeat: int, fold: int
) -> tuple[set[str], set[str], set[str], int]:
    roles = pd.read_csv(ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    cell = roles[(roles["repeat"] == repeat) & (roles["outer_fold"] == fold)]
    validation_cats = set(cell[cell["role"] == "validation"]["cat_id"].astype(str))
    test_cats = set(cell[cell["role"] == "test"]["cat_id"].astype(str))
    train_cats = set(cell[cell["role"] == "train"]["cat_id"].astype(str))
    if validation_cats & test_cats or validation_cats & train_cats or test_cats & train_cats:
        raise RuntimeError("role cell cat sets overlap")
    with np.load(ROOT / protocol["data"]["frozen_embedding_path"], allow_pickle=False) as data:
        call_ids = data["call_ids"].astype(str)
        cat_ids = data["cat_ids"].astype(str)
    validation_calls = set(call_ids[np.isin(cat_ids, list(validation_cats))])
    train_call_count = int(np.isin(cat_ids, list(train_cats)).sum())
    return validation_cats, validation_calls, test_cats, train_call_count


def fit_summary_path(identity: tuple[int, int, int, str]) -> Path:
    base_seed, repeat, fold, pipeline = identity
    return (
        RUN_ROOT
        / "fits"
        / pipeline
        / f"base_seed_{base_seed}"
        / f"repeat_{repeat}"
        / f"fold_{fold}"
        / "fit_summary.json"
    )


def load_and_validate_fit(
    protocol: dict[str, Any], identity: tuple[int, int, int, str]
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base_seed, repeat, fold, pipeline = identity
    path = fit_summary_path(identity)
    if not path.is_file():
        raise RuntimeError(f"missing fit: {identity}")
    fit = read_json(path)
    validation_cats, validation_calls, test_cats, train_call_count = validation_population(
        protocol, repeat, fold
    )
    expected = {
        "status": "complete",
        "pipeline": pipeline,
        "base_seed": base_seed,
        "full_seed": base_seed + 10_000 * repeat + 100 * fold,
        "repeat": repeat,
        "fold": fold,
        "outer_test_accessed": False,
        "train_calls": train_call_count,
        "validation_calls": len(validation_calls),
        "validation_cats": len(validation_cats),
    }
    for key, value in expected.items():
        if fit.get(key) != value:
            raise RuntimeError(f"{identity}: fit mismatch in {key}")
    animal_path = ROOT / fit["validation_animal_predictions"]
    call_path = ROOT / fit["validation_call_predictions"]
    if sha256(animal_path) != fit["validation_animal_sha256"]:
        raise RuntimeError(f"{identity}: animal prediction hash mismatch")
    if sha256(call_path) != fit["validation_call_sha256"]:
        raise RuntimeError(f"{identity}: call prediction hash mismatch")
    animals = pd.read_csv(animal_path, dtype={"cat_id": str})
    calls = pd.read_csv(call_path, dtype={"cat_id": str, "call_id": str})
    validate_probabilities(animals, f"{identity}/animals")
    validate_probabilities(calls, f"{identity}/calls")
    if animals["cat_id"].duplicated().any() or calls["call_id"].duplicated().any():
        raise RuntimeError(f"{identity}: duplicate animal or call prediction")
    if set(animals["cat_id"].astype(str)) != validation_cats:
        raise RuntimeError(f"{identity}: validation animal role mismatch")
    if set(calls["call_id"].astype(str)) != validation_calls:
        raise RuntimeError(f"{identity}: validation call role mismatch")
    if set(animals["cat_id"].astype(str)) & test_cats:
        raise RuntimeError(f"{identity}: outer-test animal in predictions")
    reconstructed = reconstruct_animals(calls)
    compare_reconstruction(animals, reconstructed, str(identity))
    animals = animals.sort_values("cat_id").reset_index(drop=True)
    bundle = metric_bundle(animals)
    audit = fit["audit"]
    assert_finite(audit, f"{identity}.audit")
    if audit.get("outer_test_accessed") is not False:
        raise RuntimeError(f"{identity}: fit audit outer-test flag changed")
    if audit["model"]["trainable_parameters"] != EXPECTED_PARAMETERS[pipeline]:
        raise RuntimeError(f"{identity}: parameter count mismatch")
    if audit["checkpoint_reload_max_probability_difference"] > 1.0e-6:
        raise RuntimeError(f"{identity}: checkpoint reload mismatch")
    if not (1 <= audit["best_epoch"] <= audit["stopped_epoch"] <= 50):
        raise RuntimeError(f"{identity}: epoch bounds mismatch")
    if len(audit["history"]) != audit["stopped_epoch"]:
        raise RuntimeError(f"{identity}: history length mismatch")
    saved_metrics = audit["best_validation_animal_metrics"]
    checks = {
        "plain_accuracy": bundle["plain_accuracy"],
        "macro_f1": bundle["macro_f1"],
        "balanced_accuracy": bundle["balanced_accuracy"],
    }
    for key, value in checks.items():
        if not math.isclose(saved_metrics[key], value, rel_tol=0.0, abs_tol=TOLERANCE):
            raise RuntimeError(f"{identity}: independently recomputed {key} differs")
    for class_name in CLASS_NAMES:
        saved = saved_metrics["per_class"][class_name]["recall"]
        if not math.isclose(
            saved, bundle["class_recall"][class_name], rel_tol=0.0, abs_tol=TOLERANCE
        ):
            raise RuntimeError(f"{identity}: independently recomputed {class_name} recall differs")
    if not math.isclose(
        audit["best_validation_animal_cross_entropy"],
        bundle["cross_entropy"], rel_tol=0.0, abs_tol=TOLERANCE,
    ):
        raise RuntimeError(f"{identity}: independently recomputed CE differs")
    if not math.isclose(
        audit["best_validation_animal_brier"],
        bundle["brier"], rel_tol=0.0, abs_tol=TOLERANCE,
    ):
        raise RuntimeError(f"{identity}: independently recomputed Brier differs")
    animals["base_seed"] = base_seed
    animals["repeat"] = repeat
    animals["fold"] = fold
    evidence = {
        "fit_summary_sha256": sha256(path),
        "validation_animal_sha256": sha256(animal_path),
        "validation_call_sha256": sha256(call_path),
        "animals": int(len(animals)),
        "calls": int(len(calls)),
        "best_epoch": int(audit["best_epoch"]),
        "stopped_epoch": int(audit["stopped_epoch"]),
        "metrics": bundle,
    }
    return fit, animals, calls, evidence


def check_paired_frames(frames: dict[str, pd.DataFrame]) -> None:
    reference = frames[PIPELINES[0]]
    for pipeline in PIPELINES[1:]:
        current = frames[pipeline]
        for column in ("cat_id", "true_label"):
            if not np.array_equal(reference[column].to_numpy(), current[column].to_numpy()):
                raise RuntimeError(f"paired {column} differs across pipelines: {pipeline}")


def check_initialization(initialization: dict[str, Any]) -> None:
    if initialization["parameters"] != EXPECTED_PARAMETERS:
        raise RuntimeError("initialization parameter counts differ")
    if any(initialization["max_logit_difference_vs_A0"].values()):
        raise RuntimeError("initial logits differ from A0")
    if not all(initialization["common_AST_state_equal_to_A0"].values()):
        raise RuntimeError("common AST initialization differs")
    if initialization.get("T1_J1_trainable_state_equal") is not True:
        raise RuntimeError("T1/J1 initial trainable state differs")
    if not all(initialization["zero_initialized_residual_outputs"].values()):
        raise RuntimeError("residual output was not zero initialized")


def add_pipeline_metrics(row: dict[str, Any], bundles: dict[str, Any]) -> None:
    for pipeline in PIPELINES:
        metrics = bundles[pipeline]
        for key in (
            "macro_f1", "plain_accuracy", "balanced_accuracy", "cross_entropy", "brier"
        ):
            row[f"{pipeline}_{key}"] = metrics[key]
        for class_name in CLASS_NAMES:
            row[f"{pipeline}_{class_name}_recall"] = metrics["class_recall"][class_name]
    for name, (candidate, comparator) in {**COMPARISONS, **DESCRIPTIVE_COMPARISONS}.items():
        for key in ("macro_f1", "plain_accuracy", "balanced_accuracy"):
            row[f"{name}_{key}"] = row[f"{candidate}_{key}"] - row[f"{comparator}_{key}"]
        row[f"{name}_cross_entropy_gain"] = (
            row[f"{comparator}_cross_entropy"] - row[f"{candidate}_cross_entropy"]
        )
        row[f"{name}_brier_gain"] = row[f"{comparator}_brier"] - row[f"{candidate}_brier"]
        for class_name in CLASS_NAMES:
            row[f"{name}_{class_name}_recall"] = (
                row[f"{candidate}_{class_name}_recall"]
                - row[f"{comparator}_{class_name}_recall"]
            )


def first_cell_audit() -> dict[str, Any]:
    protocol, manifest = verify_locked_inputs()
    if not PARTIAL_PATH.is_file():
        raise RuntimeError("IDEA-085 first-cell partial summary is not ready")
    partial = read_json(PARTIAL_PATH)
    expected_partial = {
        "status": "partial_first_cell_audit_required",
        "completed_cells_this_invocation": 1,
        "completed_fits_visible_this_invocation": 4,
        "expected_total_fits": 144,
        "aggregation_generated": False,
        "resume_required": True,
        "outer_test_accessed": False,
        "protocol_sha256": LOCKED_PROTOCOL_SHA256,
        "runner_sha256": LOCKED_RUNNER_SHA256,
        "cpu_preflight_sha256": LOCKED_CPU_PREFLIGHT_SHA256,
    }
    if partial != expected_partial:
        raise RuntimeError("IDEA-085 partial summary differs from locked first-cell stop")
    for forbidden in (SUMMARY_PATH, RUN_ROOT / "run_summary.json"):
        if forbidden.exists():
            raise RuntimeError(f"aggregate exists before first-cell audit: {forbidden.name}")
    # The director may authorize --resume immediately after the runner's locked
    # one-cell pause.  The immutable partial summary proves that the initial
    # invocation stopped at four fits; later fit summaries must not invalidate
    # a read-only audit of that first cell.
    all_summaries = sorted((RUN_ROOT / "fits").rglob("fit_summary.json"))
    if len(all_summaries) < 4:
        raise RuntimeError(f"expected at least four first-cell fits, found {len(all_summaries)}")

    base_seed, repeat, fold = BASE_SEEDS[0], REPEATS[0], FOLDS[0]
    frames: dict[str, pd.DataFrame] = {}
    fits: dict[str, dict[str, Any]] = {}
    evidence: dict[str, Any] = {}
    batch_histories: dict[str, list[dict[str, Any]]] = {}
    common_initialization: dict[str, Any] | None = None
    for pipeline in PIPELINES:
        identity = (base_seed, repeat, fold, pipeline)
        fit, animals, _calls, fit_evidence = load_and_validate_fit(protocol, identity)
        initialization = fit["initialization"]
        check_initialization(initialization)
        if common_initialization is None:
            common_initialization = initialization
        elif initialization != common_initialization:
            raise RuntimeError("per-fit common initialization audit differs")
        model_audit = fit["audit"]["model"]
        if pipeline in PIPELINES[2:]:
            if model_audit.get("call_lookup_used_only_for_ragged_retrieval") is not True:
                raise RuntimeError(f"{pipeline}: lookup entered model feature")
            if model_audit.get("time_or_identity_feature_entered_encoder") is not False:
                raise RuntimeError(f"{pipeline}: time/identity entered encoder")
            if model_audit["validation_relative_perturbation_max"] > 0.25 + 1.0e-5:
                raise RuntimeError(f"{pipeline}: runtime residual cap exceeded")
        frames[pipeline] = animals
        fits[pipeline] = fit
        evidence[pipeline] = fit_evidence
        batch_histories[pipeline] = [
            item["train_audit"] for item in fit["audit"]["history"]
        ]
    check_paired_frames(frames)
    common_epochs = min(len(history) for history in batch_histories.values())
    for epoch in range(common_epochs):
        reference = batch_histories[PIPELINES[0]][epoch]
        for pipeline in PIPELINES[1:]:
            current = batch_histories[pipeline][epoch]
            if (
                current["cat_order_sha256"] != reference["cat_order_sha256"]
                or current["call_coverage_sha256"] != reference["call_coverage_sha256"]
            ):
                raise RuntimeError(f"paired batch coverage differs: {pipeline}/epoch={epoch+1}")
    result = {
        "schema_version": "1.0",
        "audit_id": "meowagenet-idea085-independent-first-cell-audit-v1",
        "status": "PASS_FIRST_CELL_ENGINEERING_AUDIT",
        "method": (
            "Independent reconstruction from call/animal validation CSVs; no import "
            "or call of the IDEA-085 runner or aggregate."
        ),
        "first_complete_cell": {
            "base_seed": base_seed, "repeat": repeat, "fold": fold
        },
        "complete_fits_audited": 4,
        "locked_inputs": {
            "protocol_sha256": LOCKED_PROTOCOL_SHA256,
            "runner_sha256": LOCKED_RUNNER_SHA256,
            "cpu_preflight_sha256": LOCKED_CPU_PREFLIGHT_SHA256,
            "manifest_sha256": sha256(MANIFEST_PATH),
            "partial_summary_sha256": sha256(PARTIAL_PATH),
        },
        "gpu_device": manifest["environment"]["gpu"],
        "outer_test_accessed": False,
        "prediction_population_equal_all_four": True,
        "animal_predictions_reconstructed_from_calls": 4,
        "prediction_hashes_valid": True,
        "probabilities_finite_normalized_and_argmax_consistent": True,
        "checkpoint_reload_valid": True,
        "initial_logits_and_common_state_valid": True,
        "T1_J1_trainable_initial_state_valid": True,
        "lookup_is_retrieval_only": True,
        "paired_batch_coverage_valid_for_common_epochs": True,
        "common_epochs_audited": common_epochs,
        "aggregate_not_generated": True,
        "partial_recorded_exactly_one_cell_before_resume": True,
        "fit_summaries_visible_at_audit_time": len(all_summaries),
        "resume_may_be_in_progress": len(all_summaries) > 4,
        "metrics_are_descriptive_not_a_resume_or_gate_decision": True,
        "resume_authorization_is_external_to_this_audit": True,
        "pipelines": evidence,
    }
    write_json(FIRST_CELL_AUDIT_PATH, result)
    return result


def occurrence_accounting(frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame.groupby("cat_id").size().to_numpy(dtype=np.int64)
    return {
        "repeated_animal_occurrences": int(len(frame)),
        "unique_cats": int(frame["cat_id"].nunique()),
        "occurrences_per_cat_min": int(counts.min()),
        "occurrences_per_cat_median": float(np.median(counts)),
        "occurrences_per_cat_max": int(counts.max()),
        "note": (
            "Occurrences reuse animals across seeds/repeats and are descriptive; "
            "unique cats are counted separately and are not treated as new samples."
        ),
    }


def nested_compare(
    official: Any, independent: Any, path: str, differences: list[dict[str, Any]]
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


def full_results_audit() -> dict[str, Any]:
    protocol, _manifest = verify_locked_inputs()
    if not SUMMARY_PATH.is_file():
        raise RuntimeError("IDEA-085 complete 144-fit summary is not ready")
    official = read_json(SUMMARY_PATH)
    if official.get("status") != "complete" or official.get("fits") != 144:
        raise RuntimeError("IDEA-085 official summary is not a complete 144-fit result")
    if official.get("outer_test_accessed") is not False:
        raise RuntimeError("IDEA-085 official summary says outer test was accessed")

    frames: dict[tuple[int, int, int, str], pd.DataFrame] = {}
    evidence: dict[str, Any] = {}
    for base_seed in BASE_SEEDS:
        for repeat in REPEATS:
            for fold in FOLDS:
                paired: dict[str, pd.DataFrame] = {}
                for pipeline in PIPELINES:
                    identity = (base_seed, repeat, fold, pipeline)
                    _fit, animals, _calls, fit_evidence = load_and_validate_fit(
                        protocol, identity
                    )
                    frames[identity] = animals
                    paired[pipeline] = animals
                    evidence[fit_summary_path(identity).relative_to(ROOT).as_posix()] = fit_evidence
                check_paired_frames(paired)

    expected_summaries = 144
    actual_summaries = len(list((RUN_ROOT / "fits").rglob("fit_summary.json")))
    if actual_summaries != expected_summaries:
        raise RuntimeError(f"fit summary count differs: {actual_summaries}")

    fold_rows: list[dict[str, Any]] = []
    seed_repeat_rows: list[dict[str, Any]] = []
    pooled_all: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
    for base_seed in BASE_SEEDS:
        for repeat in REPEATS:
            pooled: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
            for fold in FOLDS:
                bundles: dict[str, Any] = {}
                for pipeline in PIPELINES:
                    frame = frames[(base_seed, repeat, fold, pipeline)]
                    bundles[pipeline] = metric_bundle(frame)
                    pooled[pipeline].append(frame)
                    pooled_all[pipeline].append(frame)
                row: dict[str, Any] = {
                    "base_seed": base_seed, "repeat": repeat, "fold": fold
                }
                add_pipeline_metrics(row, bundles)
                fold_rows.append(row)
            pooled_frames = {
                pipeline: pd.concat(parts, ignore_index=True)
                for pipeline, parts in pooled.items()
            }
            row = {"base_seed": base_seed, "repeat": repeat}
            add_pipeline_metrics(
                row,
                {pipeline: metric_bundle(frame) for pipeline, frame in pooled_frames.items()},
            )
            for name, (candidate, comparator) in COMPARISONS.items():
                for key, value in paired_transitions(
                    pooled_frames[candidate], pooled_frames[comparator]
                ).items():
                    row[f"{name}_{key}"] = value
            seed_repeat_rows.append(row)

    folds = pd.DataFrame(fold_rows)
    seed_repeats = pd.DataFrame(seed_repeat_rows)
    split_rows: list[dict[str, Any]] = []
    for repeat in REPEATS:
        for fold in FOLDS:
            selected = folds[(folds["repeat"] == repeat) & (folds["fold"] == fold)]
            row: dict[str, Any] = {"repeat": repeat, "fold": fold}
            for name in COMPARISONS:
                row[f"{name}_macro_f1"] = float(selected[f"{name}_macro_f1"].mean())
            split_rows.append(row)
    split_cells = pd.DataFrame(split_rows)
    pooled_frames = {
        pipeline: pd.concat(parts, ignore_index=True)
        for pipeline, parts in pooled_all.items()
    }

    metric_names = (
        "macro_f1", "plain_accuracy", "balanced_accuracy", "cross_entropy", "brier",
        "kitten_recall", "adult_recall", "senior_recall",
    )
    pipeline_means = {
        metric: {
            pipeline: float(seed_repeats[f"{pipeline}_{metric}"].mean())
            for pipeline in PIPELINES
        }
        for metric in metric_names
    }
    gate = protocol["classification_gate"]
    comparison_results: dict[str, Any] = {}
    for name, (candidate, comparator) in COMPARISONS.items():
        values = seed_repeats[f"{name}_macro_f1"]
        split_values = split_cells[f"{name}_macro_f1"]
        per_seed = {
            str(seed): float(
                seed_repeats[seed_repeats["base_seed"] == seed][f"{name}_macro_f1"].mean()
            )
            for seed in BASE_SEEDS
        }
        conditions = {
            "mean_macro_f1_delta": float(values.mean())
            >= float(gate["minimum_mean_seed_repeat_macro_f1_delta"]),
            "positive_base_seed_means": sum(value > 0 for value in per_seed.values())
            >= int(gate["minimum_positive_base_seed_means"]),
            "positive_seed_repeats": int((values > 0).sum())
            >= int(gate["minimum_positive_seed_repeats"]),
            "nonnegative_split_cells": int((split_values >= 0).sum())
            >= int(gate["minimum_nonnegative_split_cells"]),
            "worst_split_cell": float(split_values.min())
            >= float(gate["minimum_worst_split_cell_delta"]),
        }
        correction_keys = (
            "paired_occurrences", "corrected_errors", "introduced_errors",
            "net_corrections", "unchanged_correct", "unchanged_wrong",
        )
        correction_profile = {
            key: {
                "mean_per_seed_repeat": float(seed_repeats[f"{name}_{key}"].mean()),
                "total_descriptive_repeated_occurrences": int(
                    seed_repeats[f"{name}_{key}"].sum()
                ),
            }
            for key in correction_keys
        }
        correction_profile["net_correction_positive_tied_negative"] = {
            "positive": int((seed_repeats[f"{name}_net_corrections"] > 0).sum()),
            "tied": int((seed_repeats[f"{name}_net_corrections"] == 0).sum()),
            "negative": int((seed_repeats[f"{name}_net_corrections"] < 0).sum()),
        }
        comparison_results[name] = {
            "candidate": candidate,
            "comparator": comparator,
            "macro_f1": contrast_summary(values),
            "per_base_seed_mean_delta": per_seed,
            "split_cell_nonnegative": int((split_values >= 0).sum()),
            "split_cell_worst": float(split_values.min()),
            "classification_conditions": conditions,
            "classification_gate_passed": bool(all(conditions.values())),
            "gate_passed": bool(all(conditions.values())),
            "auxiliary_profile": {
                "mean_plain_accuracy_delta": pipeline_means["plain_accuracy"][candidate]
                - pipeline_means["plain_accuracy"][comparator],
                "mean_balanced_accuracy_delta": pipeline_means["balanced_accuracy"][candidate]
                - pipeline_means["balanced_accuracy"][comparator],
                "mean_cross_entropy_gain": pipeline_means["cross_entropy"][comparator]
                - pipeline_means["cross_entropy"][candidate],
                "mean_brier_gain": pipeline_means["brier"][comparator]
                - pipeline_means["brier"][candidate],
                "mean_class_recall_delta": {
                    class_name: pipeline_means[f"{class_name}_recall"][candidate]
                    - pipeline_means[f"{class_name}_recall"][comparator]
                    for class_name in CLASS_NAMES
                },
                "auxiliary_axes_are_not_classification_gate_conditions": True,
            },
            "error_correction_profile": correction_profile,
            "interpretation_boundary": (
                "Real order versus one fixed joint frame permutation; not complete causality and not pure-F0 isolation."
                if name == "T1_minus_J1"
                else "Same 111 cats; not independent external confirmation."
            ),
        }

    descriptive_results: dict[str, Any] = {}
    for name, (candidate, comparator) in DESCRIPTIVE_COMPARISONS.items():
        descriptive_results[name] = {
            "candidate": candidate,
            "comparator": comparator,
            "macro_f1": contrast_summary(seed_repeats[f"{name}_macro_f1"]),
            "mean_plain_accuracy_delta": pipeline_means["plain_accuracy"][candidate]
            - pipeline_means["plain_accuracy"][comparator],
            "mean_balanced_accuracy_delta": pipeline_means["balanced_accuracy"][candidate]
            - pipeline_means["balanced_accuracy"][comparator],
            "mean_cross_entropy_gain": pipeline_means["cross_entropy"][comparator]
            - pipeline_means["cross_entropy"][candidate],
            "mean_brier_gain": pipeline_means["brier"][comparator]
            - pipeline_means["brier"][candidate],
            "status": "contemporaneous_context_only",
        }

    independent = {
        "pipeline_seed_repeat_means": pipeline_means,
        "comparison_results": comparison_results,
        "descriptive_results": descriptive_results,
        "fold_results": fold_rows,
        "seed_repeat_results": seed_repeat_rows,
        "split_cell_results": split_rows,
    }
    official_selected = {key: official[key] for key in independent}
    differences: list[dict[str, Any]] = []
    nested_compare(official_selected, independent, "results", differences)

    pooled_validation: dict[str, Any] = {}
    occurrence_counts: dict[str, Any] = {}
    for pipeline, frame in pooled_frames.items():
        bundle = metric_bundle(frame)
        occurrence_counts[pipeline] = occurrence_accounting(frame)
        official_values = official["pooled_validation"][pipeline]
        selected = {
            "animal_occurrences": official_values["animal_occurrences"],
            "plain_accuracy": official_values["metrics"]["plain_accuracy"],
            "macro_f1": official_values["metrics"]["macro_f1"],
            "balanced_accuracy": official_values["metrics"]["balanced_accuracy"],
            "class_recall": {
                name: official_values["metrics"]["per_class"][name]["recall"]
                for name in CLASS_NAMES
            },
            "cross_entropy": official_values["cross_entropy"],
            "brier": official_values["brier"],
        }
        recomputed = {
            "animal_occurrences": int(len(frame)),
            "plain_accuracy": bundle["plain_accuracy"],
            "macro_f1": bundle["macro_f1"],
            "balanced_accuracy": bundle["balanced_accuracy"],
            "class_recall": bundle["class_recall"],
            "cross_entropy": bundle["cross_entropy"],
            "brier": bundle["brier"],
        }
        nested_compare(selected, recomputed, f"pooled_validation.{pipeline}", differences)
        pooled_validation[pipeline] = recomputed

    audit = {
        "schema_version": "1.0",
        "audit_id": "meowagenet-idea085-independent-results-audit-v1",
        "status": "PASS" if not differences else "FAIL",
        "method": (
            "Independent reconstruction from saved call/animal validation CSVs; no "
            "import or call of the IDEA-085 runner or aggregate."
        ),
        "locked_inputs": {
            "protocol_sha256": LOCKED_PROTOCOL_SHA256,
            "runner_sha256": LOCKED_RUNNER_SHA256,
            "cpu_preflight_sha256": LOCKED_CPU_PREFLIGHT_SHA256,
            "manifest_sha256": sha256(MANIFEST_PATH),
            "official_summary_sha256": sha256(SUMMARY_PATH),
        },
        "integrity": {
            "fit_summaries_checked": len(evidence),
            "prediction_files_checked": 2 * len(evidence),
            "animal_predictions_reconstructed_from_calls": len(evidence),
            "complete_expected_fit_matrix": len(evidence) == 144,
            "outer_test_accessed": False,
            "fit_evidence": evidence,
        },
        "recomputed": independent,
        "pooled_validation_core": pooled_validation,
        "animal_occurrence_accounting": occurrence_counts,
        "classification_gate_scope": {
            "three_comparisons_are_independent": True,
            "single_global_gate_exists": False,
            "CE_Brier_accuracy_BA_and_recalls_are_auxiliary": True,
        },
        "interpretation_boundaries": {
            "T1_minus_C1": "Temporal residual versus summary C1; cannot be attributed purely to order.",
            "T1_minus_J1": (
                "Real order versus one fixed joint permutation; includes missingness and "
                "voicing local-order organization, not pure-F0 isolation."
            ),
        },
        "comparison_to_official": {
            "tolerance": TOLERANCE,
            "difference_count": len(differences),
            "differences": differences,
        },
    }
    write_json(FULL_AUDIT_PATH, audit)
    if differences:
        raise RuntimeError(f"independent result differs at {len(differences)} fields")
    return audit


def main() -> None:
    args = parse_args()
    result = first_cell_audit() if args.mode == "first-cell" else full_results_audit()
    output_path = FIRST_CELL_AUDIT_PATH if args.mode == "first-cell" else FULL_AUDIT_PATH
    print(
        json.dumps(
            {
                "status": result["status"],
                "audit_path": output_path.relative_to(ROOT).as_posix(),
                "audit_sha256": sha256(output_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
