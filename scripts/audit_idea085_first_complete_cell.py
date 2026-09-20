"""Audit IDEA-085's first complete four-pipeline GPU cell without result selection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "meowagenet_idea085_acoustic_temporal_residual_v1"
RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_idea085_acoustic_temporal_residual.py"
OUTPUT = RUN_ROOT / "first_complete_cell_audit.json"
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
BASE_SEED = 2713


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_runner():
    spec = importlib.util.spec_from_file_location("idea085_first_cell_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load locked IDEA-085 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assert_finite(value, path: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise RuntimeError(f"Non-finite numeric audit value: {path}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")


def main() -> None:
    runner = load_runner()
    manifest = read_json(RUN_ROOT / "run_manifest.json")
    preflight = read_json(RUN_ROOT / "cpu_preflight.json")
    partial = read_json(RUN_ROOT / "partial_run_summary.json")
    protocol = read_json(runner.PROTOCOL_PATH)
    expected_hashes = {
        "protocol_sha256": sha256(runner.PROTOCOL_PATH),
        "runner_sha256": sha256(RUNNER_PATH),
        "cpu_preflight_sha256": sha256(RUN_ROOT / "cpu_preflight.json"),
    }
    for key, expected in expected_hashes.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"Manifest hash mismatch: {key}")
    if manifest.get("source_feature_sha256") != protocol["data"]["trajectory_sha256"]:
        raise RuntimeError("Manifest trajectory hash mismatch")
    if preflight.get("status") != "GO" or manifest.get("cpu_preflight_status") != "GO":
        raise RuntimeError("IDEA-085 first cell lacks a matching CPU GO")
    if manifest.get("director_authorized") is not True:
        raise RuntimeError("Director authorization was not recorded")
    environment = manifest["environment"]
    if environment.get("device") != "cuda" or not environment.get("gpu"):
        raise RuntimeError("IDEA-085 first cell was not executed on CUDA")
    expected_partial = {
        "status": "partial_first_cell_audit_required",
        "completed_cells_this_invocation": 1,
        "completed_fits_visible_this_invocation": 4,
        "expected_total_fits": 144,
        "aggregation_generated": False,
        "resume_required": True,
        "outer_test_accessed": False,
        **expected_hashes,
    }
    if partial != expected_partial:
        raise RuntimeError("IDEA-085 partial summary differs from first-cell stop")
    for forbidden in ("initial_evaluation_summary.json", "run_summary.json"):
        if (RUN_ROOT / forbidden).exists():
            raise RuntimeError(f"Full aggregate exists before audit: {forbidden}")
    fit_summaries = list((RUN_ROOT / "fits").rglob("fit_summary.json"))
    if len(fit_summaries) != 4:
        raise RuntimeError("IDEA-085 first-cell pause does not expose exactly four fits")

    store = runner.idea068.idea051.reference.historical.idea019.load_feature_store()
    roles = pd.read_csv(ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    indices = runner.idea084.role_cell_indices(store, roles, repeat=0, fold=0)
    expected_call_ids = set(store.call_ids[indices["validation"]].astype(str))
    expected_cat_ids = set(store.cat_ids[indices["validation"]].astype(str))
    cell = roles[(roles["repeat"] == 0) & (roles["outer_fold"] == 0)]
    test_cat_ids = set(cell[cell["role"] == "test"]["cat_id"].astype(str))
    if expected_cat_ids & test_cat_ids:
        raise RuntimeError("Expected validation population overlaps outer test")

    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    rows: dict[str, dict] = {}
    initialization = None
    batch_histories: dict[str, list[dict]] = {}
    probability_columns = list(runner.idea068.PROBABILITY_COLUMNS)
    for pipeline in PIPELINES:
        folder = (
            RUN_ROOT / "fits" / pipeline / f"base_seed_{BASE_SEED}"
            / "repeat_0" / "fold_0"
        )
        summary_path = folder / "fit_summary.json"
        fit = read_json(summary_path)
        identity = {
            "status": "complete",
            "pipeline": pipeline,
            "base_seed": BASE_SEED,
            "full_seed": BASE_SEED,
            "repeat": 0,
            "fold": 0,
            "outer_test_accessed": False,
            "train_calls": int(len(indices["train"])),
            "validation_calls": int(len(indices["validation"])),
            "validation_cats": int(len(expected_cat_ids)),
        }
        if any(fit.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"First-cell identity/role count mismatch: {pipeline}")
        animal_path = ROOT / fit["validation_animal_predictions"]
        call_path = ROOT / fit["validation_call_predictions"]
        if sha256(animal_path) != fit["validation_animal_sha256"]:
            raise RuntimeError(f"Animal prediction hash mismatch: {pipeline}")
        if sha256(call_path) != fit["validation_call_sha256"]:
            raise RuntimeError(f"Call prediction hash mismatch: {pipeline}")
        animals = pd.read_csv(animal_path, dtype={"cat_id": str})
        calls = pd.read_csv(call_path, dtype={"cat_id": str, "call_id": str})
        if set(animals["cat_id"].astype(str)) != expected_cat_ids:
            raise RuntimeError(f"Validation animal role mismatch: {pipeline}")
        if set(calls["call_id"].astype(str)) != expected_call_ids:
            raise RuntimeError(f"Validation call role mismatch: {pipeline}")
        if set(animals["cat_id"].astype(str)) & test_cat_ids:
            raise RuntimeError(f"Outer-test animal appeared in predictions: {pipeline}")
        for name, frame, id_column in (
            ("animal", animals, "cat_id"), ("call", calls, "call_id")
        ):
            probabilities = frame[probability_columns].to_numpy(float)
            if not np.isfinite(probabilities).all():
                raise RuntimeError(f"Non-finite {name} probability: {pipeline}")
            if float(np.max(np.abs(probabilities.sum(axis=1) - 1.0))) > 1.0e-5:
                raise RuntimeError(f"Unnormalized {name} probability: {pipeline}")
            if frame[id_column].duplicated().any():
                raise RuntimeError(f"Duplicate {name} ID: {pipeline}")
            if "predicted_label" in frame.columns and not np.array_equal(
                probabilities.argmax(axis=1), frame["predicted_label"].to_numpy(int)
            ):
                raise RuntimeError(f"Predicted label/probability mismatch: {pipeline}")

        audit = fit["audit"]
        assert_finite(audit, pipeline)
        if audit.get("outer_test_accessed") is not False:
            raise RuntimeError(f"Outer-test audit flag changed: {pipeline}")
        if audit["model"]["trainable_parameters"] != EXPECTED_PARAMETERS[pipeline]:
            raise RuntimeError(f"Parameter mismatch: {pipeline}")
        if audit["checkpoint_reload_max_probability_difference"] > 1.0e-6:
            raise RuntimeError(f"Best-state checkpoint reload mismatch: {pipeline}")
        if not (1 <= audit["best_epoch"] <= audit["stopped_epoch"] <= 50):
            raise RuntimeError(f"Epoch audit mismatch: {pipeline}")
        if len(audit["history"]) != audit["stopped_epoch"]:
            raise RuntimeError(f"History length mismatch: {pipeline}")
        current_initialization = fit["initialization"]
        if initialization is None:
            initialization = current_initialization
        elif current_initialization != initialization:
            raise RuntimeError("Per-fit common initialization audit differs")
        if any(initialization["max_logit_difference_vs_A0"].values()):
            raise RuntimeError("Initial logits differ from A0")
        if not all(initialization["common_AST_state_equal_to_A0"].values()):
            raise RuntimeError("Common AST initialization differs")
        if initialization["T1_J1_trainable_state_equal"] is not True:
            raise RuntimeError("T1/J1 trainable initial state differs")
        if not all(initialization["zero_initialized_residual_outputs"].values()):
            raise RuntimeError("Residual output was not zero initialized")
        if pipeline != PIPELINES[0]:
            relative_max = audit["model"]["validation_relative_perturbation_max"]
            if relative_max > runner.idea071.CAP + 1.0e-5:
                raise RuntimeError(f"Runtime perturbation cap exceeded: {pipeline}")
        if pipeline in PIPELINES[2:]:
            if audit["model"]["raw_channels"] != 6:
                raise RuntimeError(f"Temporal raw-channel count changed: {pipeline}")
            if audit["model"]["finite_indicators"] != 6:
                raise RuntimeError(f"Temporal finite indicators changed: {pipeline}")
            if audit["model"]["time_or_identity_feature_entered_encoder"] is not False:
                raise RuntimeError(f"Identity/time entered temporal encoder: {pipeline}")

        metrics = runner.metric_bundle(animals)
        locked_metrics = audit["best_validation_animal_metrics"]
        if metrics["metrics"] != locked_metrics:
            raise RuntimeError(f"Animal metric recomputation mismatch: {pipeline}")
        if abs(metrics["cross_entropy"] - audit["best_validation_animal_cross_entropy"]) > 1e-12:
            raise RuntimeError(f"CE recomputation mismatch: {pipeline}")
        if abs(metrics["brier"] - audit["best_validation_animal_brier"]) > 1e-12:
            raise RuntimeError(f"Brier recomputation mismatch: {pipeline}")
        batch_histories[pipeline] = [item["train_audit"] for item in audit["history"]]
        frames[pipeline] = (animals, calls)
        per_class = metrics["metrics"]["per_class"]
        rows[pipeline] = {
            "summary_sha256": sha256(summary_path),
            "animal_predictions_sha256": fit["validation_animal_sha256"],
            "call_predictions_sha256": fit["validation_call_sha256"],
            "animals": int(len(animals)),
            "calls": int(len(calls)),
            "best_epoch": int(audit["best_epoch"]),
            "stopped_epoch": int(audit["stopped_epoch"]),
            "checkpoint_reload_max_probability_difference": float(
                audit["checkpoint_reload_max_probability_difference"]
            ),
            "trainable_parameters": int(audit["model"]["trainable_parameters"]),
            "runtime_relative_perturbation_max": (
                None if pipeline == PIPELINES[0]
                else float(audit["model"]["validation_relative_perturbation_max"])
            ),
            "descriptive_only": {
                "plain_accuracy": float(metrics["metrics"]["plain_accuracy"]),
                "macro_f1": float(metrics["metrics"]["macro_f1"]),
                "balanced_accuracy": float(metrics["metrics"]["balanced_accuracy"]),
                "kitten_recall": float(per_class["kitten"]["recall"]),
                "adult_recall": float(per_class["adult"]["recall"]),
                "senior_recall": float(per_class["senior"]["recall"]),
                "cross_entropy": float(metrics["cross_entropy"]),
                "brier": float(metrics["brier"]),
            },
        }

    reference_animals, reference_calls = frames[PIPELINES[0]]
    for pipeline in PIPELINES[1:]:
        animals, calls = frames[pipeline]
        for reference, current, columns in (
            (reference_animals, animals, ["cat_id", "true_label"]),
            (reference_calls, calls, ["call_id", "cat_id", "true_label"]),
        ):
            if not reference[columns].reset_index(drop=True).equals(
                current[columns].reset_index(drop=True)
            ):
                raise RuntimeError(f"Prediction population mismatch: {pipeline}")
    common_epochs = min(len(history) for history in batch_histories.values())
    for epoch in range(common_epochs):
        reference = batch_histories[PIPELINES[0]][epoch]
        for pipeline in PIPELINES[1:]:
            current = batch_histories[pipeline][epoch]
            if (
                reference["cat_order_sha256"] != current["cat_order_sha256"]
                or reference["call_coverage_sha256"]
                != current["call_coverage_sha256"]
            ):
                raise RuntimeError(f"Paired batch coverage mismatch: {pipeline}")
    next_summary = (
        RUN_ROOT / "fits" / PIPELINES[0] / f"base_seed_{BASE_SEED}"
        / "repeat_0" / "fold_1" / "fit_summary.json"
    )
    if next_summary.exists():
        raise RuntimeError("First-cell pause occurred after the next cell started")

    result = {
        "status": "PASS_PAUSED_AWAITING_DIRECTOR_RESUME_AUTHORIZATION",
        "first_complete_cell": {"base_seed": BASE_SEED, "repeat": 0, "fold": 0},
        "complete_fits_audited": 4,
        **expected_hashes,
        "outer_test_accessed": False,
        "gpu_device": environment["gpu"],
        "prediction_population_equal_all_four": True,
        "prediction_hashes_valid": True,
        "probabilities_finite_normalized_and_argmax_consistent": True,
        "best_state_checkpoint_reload_valid": True,
        "initial_logits_and_common_state_valid": True,
        "T1_J1_trainable_initial_state_valid": True,
        "paired_batch_coverage_valid_for_common_epochs": True,
        "common_epochs_audited": int(common_epochs),
        "parameters_valid": True,
        "runtime_total_budget_valid": True,
        "aggregate_not_generated": True,
        "next_cell_not_started": True,
        "metrics_are_descriptive_and_not_a_resume_decision": True,
        "pipelines": rows,
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
