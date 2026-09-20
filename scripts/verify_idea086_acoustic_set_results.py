"""Independent IDEA-086 first-fit and full-results verification.

The verifier never imports the IDEA-086 runner and never calls its aggregate.
It rebuilds cat predictions and all reported metrics from prediction CSVs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts/verify_idea085_acoustic_temporal_results.py"
SPEC = importlib.util.spec_from_file_location("independent_idea085_helpers", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load independent IDEA-085 verification helpers")
helpers = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = helpers
SPEC.loader.exec_module(helpers)

PROTOCOL_PATH = ROOT / "configs/protocol/meowagenet_idea086_acoustic_set_residual_v1.json"
RUNNER_PATH = ROOT / "scripts/run_meowagenet_idea086_acoustic_set_residual.py"
TESTS_PATH = ROOT / "tests/test_idea086_acoustic_set_residual.py"
RUN_ROOT = ROOT / "runs/meowagenet_idea086_acoustic_set_residual_v1"
REFERENCE_ROOT = ROOT / "runs/meowagenet_idea085_acoustic_temporal_residual_v1"
CPU_PREFLIGHT_PATH = RUN_ROOT / "cpu_preflight.json"
REUSE_MANIFEST_PATH = RUN_ROOT / "reuse_manifest.json"
MANIFEST_PATH = RUN_ROOT / "run_manifest.json"
PARTIAL_PATH = RUN_ROOT / "partial_run_summary.json"
SUMMARY_PATH = RUN_ROOT / "initial_evaluation_summary.json"
FIRST_FIT_AUDIT_PATH = RUN_ROOT / "independent_first_fit_audit.json"
FULL_AUDIT_PATH = RUN_ROOT / "independent_results_audit.json"

SET_PIPELINE = "SET1_acoustic_set_residual"
REFERENCE_PIPELINES = (
    "A0_ast_only",
    "C1_bounded_wide_additive",
    "T1_acoustic_temporal_residual",
    "J1_frame_shuffled_temporal_residual",
)
PIPELINES = (SET_PIPELINE, *REFERENCE_PIPELINES)
EXPECTED_PARAMETERS = {
    SET_PIPELINE: 108_323,
    REFERENCE_PIPELINES[0]: 99_075,
    REFERENCE_PIPELINES[1]: 108_143,
    REFERENCE_PIPELINES[2]: 108_355,
    REFERENCE_PIPELINES[3]: 108_355,
}
BASE_SEEDS = (2713, 5395, 5226)
REPEATS = (0, 1, 2)
FOLDS = (0, 1, 2, 3)
COMPARISONS = {
    "SET1_minus_A0": (SET_PIPELINE, REFERENCE_PIPELINES[0]),
    "SET1_minus_C1": (SET_PIPELINE, REFERENCE_PIPELINES[1]),
    "SET1_minus_T1": (SET_PIPELINE, REFERENCE_PIPELINES[2]),
    "SET1_minus_J1": (SET_PIPELINE, REFERENCE_PIPELINES[3]),
}
GATED_COMPARISONS = ("SET1_minus_A0", "SET1_minus_C1")
CLASS_NAMES = helpers.CLASS_NAMES
PROBABILITY_COLUMNS = helpers.PROBABILITY_COLUMNS
TOLERANCE = helpers.TOLERANCE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("first-fit", "full"), required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return helpers.sha256(path)


def read_json(path: Path) -> dict[str, Any]:
    return helpers.read_json(path)


def write_json(path: Path, value: Any) -> None:
    helpers.write_json(path, value)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def fit_path(identity: tuple[int, int, int, str]) -> Path:
    base_seed, repeat, fold, pipeline = identity
    root = RUN_ROOT if pipeline == SET_PIPELINE else REFERENCE_ROOT
    return (
        root / "fits" / pipeline / f"base_seed_{base_seed}"
        / f"repeat_{repeat}" / f"fold_{fold}" / "fit_summary.json"
    )


def validate_reuse_manifest(protocol: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(REUSE_MANIFEST_PATH)
    if manifest.get("fit_summaries") != 144 or manifest.get("prediction_files") != 288:
        raise RuntimeError("IDEA-086 reuse manifest matrix differs")
    entries = manifest.get("entries", [])
    if len(entries) != 432:
        raise RuntimeError("IDEA-086 reuse manifest must contain 432 artifacts")
    paths: set[str] = set()
    for entry in entries:
        relative = str(entry["path"])
        if relative in paths:
            raise RuntimeError(f"duplicate reuse manifest path: {relative}")
        paths.add(relative)
        path = ROOT / relative
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"read-only IDEA-085 artifact changed: {relative}")
    core = {
        "source_experiment": manifest["source_experiment"],
        "fit_summaries": manifest["fit_summaries"],
        "prediction_files": manifest["prediction_files"],
        "entries": entries,
    }
    content_sha = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
    if manifest.get("content_sha256") != content_sha:
        raise RuntimeError("IDEA-086 reuse manifest content hash differs")
    if manifest.get("outer_test_accessed") is not False:
        raise RuntimeError("IDEA-086 reuse manifest says outer test was accessed")
    expected_snapshot = protocol["reuse"]["previous_second_resume_snapshot_sha256"]
    if manifest.get("previous_second_resume_snapshot_sha256") != expected_snapshot:
        raise RuntimeError("IDEA-086 recorded IDEA-085 snapshot differs")
    protected_entries = manifest.get("director_protected_entries", [])
    expected_count = int(protocol["reuse"]["director_readonly_baseline_files"])
    expected_digest = protocol["reuse"]["director_readonly_baseline_sha256"]
    if len(protected_entries) != expected_count:
        raise RuntimeError("IDEA-086 director-protected baseline count differs")
    protected_paths: set[str] = set()
    for entry in protected_entries:
        relative = str(entry["path"])
        if relative in protected_paths:
            raise RuntimeError(f"duplicate protected baseline path: {relative}")
        protected_paths.add(relative)
        path = ROOT / relative
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"director-protected artifact changed: {relative}")
    protected_lines = "\n".join(
        f"{entry['path']}:{entry['sha256']}" for entry in protected_entries
    )
    protected_digest = hashlib.sha256(protected_lines.encode("utf-8")).hexdigest()
    if (
        protected_digest != expected_digest
        or manifest.get("director_protected_baseline_sha256") != expected_digest
        or manifest.get("director_protected_baseline_files") != expected_count
    ):
        raise RuntimeError("IDEA-086 director-protected 450-file baseline differs")
    return manifest


def verify_run_locks(require_gpu_manifest: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    required = (
        PROTOCOL_PATH, RUNNER_PATH, TESTS_PATH, CPU_PREFLIGHT_PATH, REUSE_MANIFEST_PATH
    )
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"IDEA-086 locked inputs are not ready: {missing}")
    protocol = read_json(PROTOCOL_PATH)
    dependencies = protocol["dependencies"]
    if dependencies.get("idea086_runner_sha256") != sha256(RUNNER_PATH):
        raise RuntimeError("IDEA-086 protocol runner lock differs")
    if dependencies.get("idea086_tests_sha256") != sha256(TESTS_PATH):
        raise RuntimeError("IDEA-086 protocol tests lock differs")
    preflight = read_json(CPU_PREFLIGHT_PATH)
    if preflight.get("status") != "GO" or preflight.get("device") != "cpu":
        raise RuntimeError("IDEA-086 CPU preflight is not GO")
    expected = {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(RUNNER_PATH),
        "reuse_manifest_sha256": sha256(REUSE_MANIFEST_PATH),
        "new_candidate_fits": 36,
        "reused_reference_fits": 144,
        "outer_test_predictions_or_metrics_accessed": False,
        "cuda_initialized": False,
    }
    for key, value in expected.items():
        if preflight.get(key) != value:
            raise RuntimeError(f"IDEA-086 CPU preflight mismatch: {key}")
    if "tests_sha256" in preflight and preflight["tests_sha256"] != sha256(TESTS_PATH):
        raise RuntimeError("IDEA-086 tests changed after CPU preflight")
    reuse_manifest = validate_reuse_manifest(protocol)
    if require_gpu_manifest:
        if not MANIFEST_PATH.is_file():
            raise RuntimeError("IDEA-086 GPU manifest is missing")
        manifest = read_json(MANIFEST_PATH)
        manifest_expected = {
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "runner_sha256": sha256(RUNNER_PATH),
            "cpu_preflight_sha256": sha256(CPU_PREFLIGHT_PATH),
            "reuse_manifest_sha256": sha256(REUSE_MANIFEST_PATH),
            "reuse_manifest_content_sha256": reuse_manifest["content_sha256"],
            "director_authorized": True,
            "outer_test_accessed": False,
            "new_pipeline": SET_PIPELINE,
            "reference_pipelines": list(REFERENCE_PIPELINES),
            "new_candidate_fits": 36,
            "reused_reference_fits": 144,
        }
        for key, value in manifest_expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"IDEA-086 run manifest mismatch: {key}")
        if manifest.get("environment", {}).get("device") != "cuda":
            raise RuntimeError("IDEA-086 formal result was not produced on CUDA")
    return protocol, preflight


def load_and_validate_fit(
    protocol: dict[str, Any], identity: tuple[int, int, int, str]
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base_seed, repeat, fold, pipeline = identity
    path = fit_path(identity)
    if not path.is_file():
        raise RuntimeError(f"missing IDEA-086 paired fit: {identity}")
    fit = read_json(path)
    validation_cats, validation_calls, test_cats, train_call_count = (
        helpers.validation_population(protocol, repeat, fold)
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
    helpers.validate_probabilities(animals, f"{identity}/animals")
    helpers.validate_probabilities(calls, f"{identity}/calls")
    if animals["cat_id"].duplicated().any() or calls["call_id"].duplicated().any():
        raise RuntimeError(f"{identity}: duplicate animal or call prediction")
    if set(animals["cat_id"].astype(str)) != validation_cats:
        raise RuntimeError(f"{identity}: validation animal role mismatch")
    if set(calls["call_id"].astype(str)) != validation_calls:
        raise RuntimeError(f"{identity}: validation call role mismatch")
    if set(animals["cat_id"].astype(str)) & test_cats:
        raise RuntimeError(f"{identity}: outer-test cat appeared")
    reconstructed = helpers.reconstruct_animals(calls)
    helpers.compare_reconstruction(animals, reconstructed, str(identity))
    animals = animals.sort_values("cat_id").reset_index(drop=True)
    bundle = helpers.metric_bundle(animals)
    audit = fit["audit"]
    helpers.assert_finite(audit, f"{identity}.audit")
    if audit.get("outer_test_accessed") is not False:
        raise RuntimeError(f"{identity}: audit outer-test flag changed")
    if audit["model"]["trainable_parameters"] != EXPECTED_PARAMETERS[pipeline]:
        raise RuntimeError(f"{identity}: trainable parameter count changed")
    if audit["checkpoint_reload_max_probability_difference"] > 1.0e-6:
        raise RuntimeError(f"{identity}: checkpoint reload mismatch")
    saved = audit["best_validation_animal_metrics"]
    for key in ("plain_accuracy", "macro_f1", "balanced_accuracy"):
        if not math.isclose(saved[key], bundle[key], rel_tol=0.0, abs_tol=TOLERANCE):
            raise RuntimeError(f"{identity}: independently recomputed {key} differs")
    for name in CLASS_NAMES:
        if not math.isclose(
            saved["per_class"][name]["recall"], bundle["class_recall"][name],
            rel_tol=0.0, abs_tol=TOLERANCE,
        ):
            raise RuntimeError(f"{identity}: independently recomputed {name} recall differs")
    if not math.isclose(
        audit["best_validation_animal_cross_entropy"], bundle["cross_entropy"],
        rel_tol=0.0, abs_tol=TOLERANCE,
    ):
        raise RuntimeError(f"{identity}: independently recomputed CE differs")
    if not math.isclose(
        audit["best_validation_animal_brier"], bundle["brier"],
        rel_tol=0.0, abs_tol=TOLERANCE,
    ):
        raise RuntimeError(f"{identity}: independently recomputed Brier differs")
    if pipeline == SET_PIPELINE:
        model_audit = audit["model"]
        trained_probe = model_audit.get("trained_checkpoint_permutation_invariance")
        if not trained_probe or trained_probe.get("passed") is not True:
            raise RuntimeError(f"{identity}: trained permutation audit missing or failed")
        if model_audit.get("projection_weight_norm", 0.0) <= 0.0:
            raise RuntimeError(f"{identity}: trained projection stayed zero")
        if model_audit.get("call_lookup_used_only_for_ragged_retrieval") is not True:
            raise RuntimeError(f"{identity}: lookup entered SET1 encoder")
        initialization = fit["initialization"]
        required_initialization = {
            "parameters": 108_323,
            "branch_parameters": 9_248,
            "common_AST_state_equal_to_A0": True,
            "common_AST_state_equal_to_T1": True,
            "zero_init_max_logit_difference_vs_A0": 0.0,
            "projection_zero_initialized": True,
            "train_only_preprocessing_equal_to_IDEA085_T1": True,
            "post_build_rng_reset_probe_equal": True,
            "C1_branch_stacked": False,
        }
        for key, value in required_initialization.items():
            if initialization.get(key) != value:
                raise RuntimeError(f"{identity}: initialization mismatch in {key}")
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


def check_paired_population(frames: dict[str, pd.DataFrame]) -> None:
    reference = frames[SET_PIPELINE]
    for pipeline in REFERENCE_PIPELINES:
        current = frames[pipeline]
        for column in ("cat_id", "true_label"):
            if not np.array_equal(reference[column].to_numpy(), current[column].to_numpy()):
                raise RuntimeError(f"paired {column} differs: {pipeline}")


def check_paired_batch_history(fits: dict[str, dict[str, Any]]) -> int:
    histories = {pipeline: fits[pipeline]["audit"]["history"] for pipeline in PIPELINES}
    common_epochs = min(len(history) for history in histories.values())
    for epoch in range(common_epochs):
        reference = histories[SET_PIPELINE][epoch]["train_audit"]
        for pipeline in REFERENCE_PIPELINES:
            current = histories[pipeline][epoch]["train_audit"]
            if (
                current["cat_order_sha256"] != reference["cat_order_sha256"]
                or current["call_coverage_sha256"] != reference["call_coverage_sha256"]
            ):
                raise RuntimeError(f"paired batch history differs: {pipeline}/epoch={epoch+1}")
    return common_epochs


def add_metrics(row: dict[str, Any], bundles: dict[str, Any]) -> None:
    for pipeline in PIPELINES:
        values = bundles[pipeline]
        for metric in (
            "plain_accuracy", "macro_f1", "balanced_accuracy", "cross_entropy", "brier"
        ):
            row[f"{pipeline}_{metric}"] = values[metric]
        for name in CLASS_NAMES:
            row[f"{pipeline}_{name}_recall"] = values["class_recall"][name]
    for name, (candidate, comparator) in COMPARISONS.items():
        for metric in ("plain_accuracy", "macro_f1", "balanced_accuracy"):
            row[f"{name}_{metric}"] = row[f"{candidate}_{metric}"] - row[f"{comparator}_{metric}"]
        row[f"{name}_cross_entropy_gain"] = (
            row[f"{comparator}_cross_entropy"] - row[f"{candidate}_cross_entropy"]
        )
        row[f"{name}_brier_gain"] = row[f"{comparator}_brier"] - row[f"{candidate}_brier"]
        for class_name in CLASS_NAMES:
            row[f"{name}_{class_name}_recall"] = (
                row[f"{candidate}_{class_name}_recall"]
                - row[f"{comparator}_{class_name}_recall"]
            )


def first_fit_audit() -> dict[str, Any]:
    protocol, preflight = verify_run_locks(require_gpu_manifest=True)
    if not PARTIAL_PATH.is_file():
        raise RuntimeError("IDEA-086 first-fit partial summary is not ready")
    partial = read_json(PARTIAL_PATH)
    expected = {
        "status": "partial_first_fit_audit_required",
        "completed_new_fits_visible": 1,
        "expected_new_fits": 36,
        "reused_reference_fits": 144,
        "aggregation_generated": False,
        "resume_required": True,
        "outer_test_accessed": False,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(RUNNER_PATH),
        "cpu_preflight_sha256": sha256(CPU_PREFLIGHT_PATH),
    }
    for key, value in expected.items():
        if partial.get(key) != value:
            raise RuntimeError(f"IDEA-086 first-fit partial mismatch: {key}")
    if SUMMARY_PATH.exists() or (RUN_ROOT / "run_summary.json").exists():
        raise RuntimeError("IDEA-086 aggregate exists before first-fit audit")
    identity_prefix = (BASE_SEEDS[0], REPEATS[0], FOLDS[0])
    frames: dict[str, pd.DataFrame] = {}
    evidence: dict[str, Any] = {}
    fits: dict[str, dict[str, Any]] = {}
    for pipeline in PIPELINES:
        identity = (*identity_prefix, pipeline)
        fit, animals, _calls, item = load_and_validate_fit(protocol, identity)
        frames[pipeline] = animals
        evidence[pipeline] = item
        fits[pipeline] = fit
    check_paired_population(frames)
    set_fit = fits[SET_PIPELINE]
    initialization = set_fit["initialization"]
    required_initialization = {
        "parameters": 108_323,
        "branch_parameters": 9_248,
        "common_AST_state_equal_to_A0": True,
        "common_AST_state_equal_to_T1": True,
        "zero_init_max_logit_difference_vs_A0": 0.0,
        "projection_zero_initialized": True,
        "train_only_preprocessing_equal_to_IDEA085_T1": True,
        "post_build_rng_reset_probe_equal": True,
        "C1_branch_stacked": False,
    }
    for key, value in required_initialization.items():
        if initialization.get(key) != value:
            raise RuntimeError(f"IDEA-086 first-fit initialization mismatch: {key}")
    common_epochs = check_paired_batch_history(fits)
    trained = set_fit["audit"]["model"]["trained_checkpoint_permutation_invariance"]
    if partial.get("first_fit_trained_permutation_invariance") != trained:
        raise RuntimeError("IDEA-086 partial trained permutation evidence differs")
    result = {
        "schema_version": "1.0",
        "audit_id": "meowagenet-idea086-independent-first-fit-audit-v1",
        "status": "PASS_FIRST_FIT_ENGINEERING_AUDIT",
        "method": "Independent CSV/hash/metric reconstruction; IDEA-086 runner not imported.",
        "first_fit": {"base_seed": BASE_SEEDS[0], "repeat": 0, "fold": 0},
        "new_fits_audited": 1,
        "reused_paired_fits_audited": 4,
        "locked_inputs": {
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "runner_sha256": sha256(RUNNER_PATH),
            "tests_sha256": sha256(TESTS_PATH),
            "cpu_preflight_sha256": sha256(CPU_PREFLIGHT_PATH),
            "reuse_manifest_sha256": sha256(REUSE_MANIFEST_PATH),
            "run_manifest_sha256": sha256(MANIFEST_PATH),
            "partial_summary_sha256": sha256(PARTIAL_PATH),
        },
        "outer_test_accessed": False,
        "prediction_populations_equal_all_five": True,
        "animal_predictions_reconstructed_from_calls": 5,
        "trained_SET1_permutation_invariance": trained,
        "paired_batch_coverage_valid_for_common_epochs": True,
        "common_epochs_audited": common_epochs,
        "metrics_are_descriptive_not_a_resume_or_gate_decision": True,
        "pipelines": evidence,
        "cpu_preflight_sha256_seen": sha256(CPU_PREFLIGHT_PATH),
        "cpu_preflight_status_seen": preflight["status"],
    }
    write_json(FIRST_FIT_AUDIT_PATH, result)
    return result


def aggregate_independently(
    protocol: dict[str, Any], frames: dict[tuple[int, int, int, str], pd.DataFrame]
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
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
                    bundles[pipeline] = helpers.metric_bundle(frame)
                    pooled[pipeline].append(frame)
                    pooled_all[pipeline].append(frame)
                row: dict[str, Any] = {
                    "base_seed": base_seed, "repeat": repeat, "fold": fold
                }
                add_metrics(row, bundles)
                fold_rows.append(row)
            pooled_frames = {
                pipeline: pd.concat(parts, ignore_index=True)
                for pipeline, parts in pooled.items()
            }
            row = {"base_seed": base_seed, "repeat": repeat}
            add_metrics(
                row,
                {pipeline: helpers.metric_bundle(frame) for pipeline, frame in pooled_frames.items()},
            )
            for name, (candidate, comparator) in COMPARISONS.items():
                for key, value in helpers.paired_transitions(
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
        "plain_accuracy", "macro_f1", "balanced_accuracy", "cross_entropy", "brier",
        "kitten_recall", "adult_recall", "senior_recall",
    )
    means = {
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
        splits = split_cells[f"{name}_macro_f1"]
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
            "nonnegative_split_cells": int((splits >= 0).sum())
            >= int(gate["minimum_nonnegative_split_cells"]),
            "worst_split_cell": float(splits.min())
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
        net = seed_repeats[f"{name}_net_corrections"]
        correction_profile["net_correction_positive_tied_negative"] = {
            "positive": int((net > 0).sum()),
            "tied": int((net == 0).sum()),
            "negative": int((net < 0).sum()),
        }
        gated = name in GATED_COMPARISONS
        comparison_results[name] = {
            "candidate": candidate,
            "comparator": comparator,
            "macro_f1": helpers.contrast_summary(values),
            "per_base_seed_mean_delta": per_seed,
            "positive_base_seed_means": sum(value > 0 for value in per_seed.values()),
            "split_cell_nonnegative": int((splits >= 0).sum()),
            "split_cell_worst": float(splits.min()),
            "classification_gate_applicable": gated,
            "classification_conditions": conditions if gated else None,
            "classification_gate_passed": bool(all(conditions.values())) if gated else None,
            "auxiliary_profile": {
                "mean_plain_accuracy_delta": means["plain_accuracy"][candidate]
                - means["plain_accuracy"][comparator],
                "mean_balanced_accuracy_delta": means["balanced_accuracy"][candidate]
                - means["balanced_accuracy"][comparator],
                "mean_cross_entropy_gain": means["cross_entropy"][comparator]
                - means["cross_entropy"][candidate],
                "mean_brier_gain": means["brier"][comparator]
                - means["brier"][candidate],
                "mean_class_recall_delta": {
                    class_name: means[f"{class_name}_recall"][candidate]
                    - means[f"{class_name}_recall"][comparator]
                    for class_name in CLASS_NAMES
                },
                "not_classification_gate_conditions": True,
            },
            "error_correction_profile": correction_profile,
        }
    return (
        {
            "pipeline_seed_repeat_means": means,
            "comparison_results": comparison_results,
            "fold_results": fold_rows,
            "seed_repeat_results": seed_repeat_rows,
            "split_cell_results": split_rows,
        },
        pooled_frames,
    )


def full_results_audit() -> dict[str, Any]:
    protocol, _preflight = verify_run_locks(require_gpu_manifest=True)
    if not SUMMARY_PATH.is_file():
        raise RuntimeError("IDEA-086 complete summary is not ready")
    official = read_json(SUMMARY_PATH)
    if (
        official.get("status") != "complete"
        or official.get("new_candidate_fits") != 36
        or official.get("reused_reference_fits") != 144
        or official.get("outer_test_accessed") is not False
    ):
        raise RuntimeError("IDEA-086 official summary is incomplete")
    frames: dict[tuple[int, int, int, str], pd.DataFrame] = {}
    evidence: dict[str, Any] = {}
    for base_seed in BASE_SEEDS:
        for repeat in REPEATS:
            for fold in FOLDS:
                paired: dict[str, pd.DataFrame] = {}
                paired_fits: dict[str, dict[str, Any]] = {}
                for pipeline in PIPELINES:
                    identity = (base_seed, repeat, fold, pipeline)
                    fit, animals, _calls, item = load_and_validate_fit(protocol, identity)
                    frames[identity] = animals
                    paired[pipeline] = animals
                    paired_fits[pipeline] = fit
                    evidence[fit_path(identity).relative_to(ROOT).as_posix()] = item
                check_paired_population(paired)
                check_paired_batch_history(paired_fits)
    if len(list((RUN_ROOT / "fits" / SET_PIPELINE).rglob("fit_summary.json"))) != 36:
        raise RuntimeError("IDEA-086 new fit matrix is not exactly 36")
    independent, pooled_frames = aggregate_independently(protocol, frames)
    differences: list[dict[str, Any]] = []
    official_core = {
        "pipeline_seed_repeat_means": official["pipeline_seed_repeat_means"],
        "comparison_results": official["comparison_results"],
        "fold_results": official["fold_results"],
        "seed_repeat_results": official["seed_repeat_results"],
        "split_cell_results": official["split_cell_results"],
    }
    helpers.nested_compare(official_core, independent, "results", differences)
    pooled_core: dict[str, Any] = {}
    occurrence_accounting: dict[str, Any] = {}
    for pipeline, frame in pooled_frames.items():
        bundle = helpers.metric_bundle(frame)
        recomputed = {
            "animal_occurrences": int(len(frame)),
            "plain_accuracy": bundle["plain_accuracy"],
            "macro_f1": bundle["macro_f1"],
            "balanced_accuracy": bundle["balanced_accuracy"],
            "class_recall": bundle["class_recall"],
            "cross_entropy": bundle["cross_entropy"],
            "brier": bundle["brier"],
        }
        saved = official["pooled_validation"][pipeline]
        official_values = {
            "animal_occurrences": saved["animal_occurrences"],
            "plain_accuracy": saved["metrics"]["plain_accuracy"],
            "macro_f1": saved["metrics"]["macro_f1"],
            "balanced_accuracy": saved["metrics"]["balanced_accuracy"],
            "class_recall": {
                name: saved["metrics"]["per_class"][name]["recall"]
                for name in CLASS_NAMES
            },
            "cross_entropy": saved["cross_entropy"],
            "brier": saved["brier"],
        }
        helpers.nested_compare(
            official_values, recomputed, f"pooled_validation.{pipeline}", differences
        )
        pooled_core[pipeline] = recomputed
        occurrence_accounting[pipeline] = helpers.occurrence_accounting(frame)
    audit = {
        "schema_version": "1.0",
        "audit_id": "meowagenet-idea086-independent-results-audit-v1",
        "status": "PASS" if not differences else "FAIL",
        "method": (
            "Independent call-to-cat reconstruction and metric aggregation; IDEA-086 "
            "runner not imported and its aggregate not called."
        ),
        "locked_inputs": {
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "runner_sha256": sha256(RUNNER_PATH),
            "tests_sha256": sha256(TESTS_PATH),
            "cpu_preflight_sha256": sha256(CPU_PREFLIGHT_PATH),
            "reuse_manifest_sha256": sha256(REUSE_MANIFEST_PATH),
            "run_manifest_sha256": sha256(MANIFEST_PATH),
            "official_summary_sha256": sha256(SUMMARY_PATH),
        },
        "integrity": {
            "new_fit_summaries_checked": 36,
            "reused_fit_summaries_checked": 144,
            "prediction_files_checked": 360,
            "animal_predictions_reconstructed_from_calls": 180,
            "complete_paired_matrix": len(evidence) == 180,
            "outer_test_accessed": False,
            "fit_evidence": evidence,
        },
        "recomputed": independent,
        "pooled_validation_core": pooled_core,
        "animal_occurrence_accounting": occurrence_accounting,
        "gate_scope": {
            "gated_comparisons": list(GATED_COMPARISONS),
            "auxiliary_no_gate": ["SET1_minus_T1", "SET1_minus_J1"],
            "single_global_gate_exists": False,
            "CE_Brier_accuracy_BA_and_recalls_are_auxiliary": True,
        },
        "interpretation_boundaries": {
            "SET1_minus_T1_J1": (
                "Approximately parameter-matched method comparison, not a pure removal-of-time "
                "causal ablation; superiority would not prove time information harmful."
            ),
            "permutation_scope": (
                "Only the auxiliary acoustic frame-set branch is permutation invariant for a "
                "fixed frozen AST call embedding; the AST path may still encode time."
            ),
            "distribution_scope": (
                "Mean nonlinear frame embeddings summarize an empirical frame distribution but "
                "do not preserve every distributional property or reconstruct the full contour."
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
        raise RuntimeError(f"IDEA-086 independent result differs at {len(differences)} fields")
    return audit


def main() -> None:
    args = parse_args()
    result = first_fit_audit() if args.mode == "first-fit" else full_results_audit()
    output = FIRST_FIT_AUDIT_PATH if args.mode == "first-fit" else FULL_AUDIT_PATH
    print(
        json.dumps(
            {
                "status": result["status"],
                "audit_path": output.relative_to(ROOT).as_posix(),
                "audit_sha256": sha256(output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
