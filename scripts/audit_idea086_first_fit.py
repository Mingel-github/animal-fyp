"""Audit the single director-authorized IDEA-086 GPU fit and then stop."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_idea086_acoustic_set_residual.py"
SPEC = importlib.util.spec_from_file_location("idea086_locked_runner_for_audit", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

RUN_ROOT = ROOT / "runs" / "meowagenet_idea086_acoustic_set_residual_v1"
FIT_PATH = (
    RUN_ROOT / "fits" / runner.PIPELINE / "base_seed_2713" / "repeat_0"
    / "fold_0" / "fit_summary.json"
)
OUTPUT_PATH = RUN_ROOT / "first_fit_audit.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    protocol = read_json(runner.PROTOCOL_PATH)
    runner.verify_protocol(protocol)
    fit_paths = sorted((RUN_ROOT / "fits").rglob("fit_summary.json"))
    if fit_paths != [FIT_PATH]:
        raise RuntimeError(f"IDEA-086 first-fit audit expected exactly one fit, found {len(fit_paths)}")
    if (RUN_ROOT / "initial_evaluation_summary.json").exists() or (RUN_ROOT / "run_summary.json").exists():
        raise RuntimeError("IDEA-086 aggregate exists before first-fit authorization")
    fit = read_json(FIT_PATH)
    runner.validate_new_fit(fit, 2713, 2713, 0, 0)
    manifest = read_json(RUN_ROOT / "run_manifest.json")
    partial = read_json(RUN_ROOT / "partial_run_summary.json")
    preflight = read_json(RUN_ROOT / "cpu_preflight.json")
    reuse_manifest = read_json(RUN_ROOT / "reuse_manifest.json")
    expected_manifest = {
        "protocol_sha256": runner.sha256(runner.PROTOCOL_PATH),
        "runner_sha256": runner.sha256(RUNNER_PATH),
        "cpu_preflight_sha256": runner.sha256(RUN_ROOT / "cpu_preflight.json"),
        "reuse_manifest_sha256": runner.sha256(RUN_ROOT / "reuse_manifest.json"),
        "director_authorized": True,
        "outer_test_accessed": False,
        "new_candidate_fits": 36,
        "reused_reference_fits": 144,
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"IDEA-086 first-fit manifest mismatch: {key}")
    if manifest["environment"]["device"] != "cuda" or not manifest["environment"]["gpu"]:
        raise RuntimeError("IDEA-086 first fit was not a CUDA fit")
    if partial.get("status") != "partial_first_fit_audit_required" or partial.get("completed_new_fits_visible") != 1 or partial.get("aggregation_generated") is not False:
        raise RuntimeError("IDEA-086 partial stop contract failed")
    if preflight.get("status") != "GO" or preflight.get("outer_test_predictions_or_metrics_accessed") is not False:
        raise RuntimeError("IDEA-086 CPU preflight boundary changed")
    if reuse_manifest.get("director_protected_baseline_sha256") != protocol["reuse"]["director_readonly_baseline_sha256"]:
        raise RuntimeError("IDEA-086 protected baseline differs")

    store = runner.idea068.idea051.reference.historical.idea019.load_feature_store()
    roles = pd.read_csv(ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    indices = runner.idea084.role_cell_indices(store, roles, 0, 0)
    calls = pd.read_csv(ROOT / fit["validation_call_predictions"], dtype={"call_id": str, "cat_id": str})
    animals = pd.read_csv(ROOT / fit["validation_animal_predictions"], dtype={"cat_id": str})
    runner.validate_prediction_probabilities(calls, "SET1 first fit calls")
    runner.validate_prediction_probabilities(animals, "SET1 first fit animals")
    reconstructed = runner.reconstruct_animals_from_calls(calls)
    runner.compare_animal_reconstruction(animals, reconstructed, "SET1 first fit")
    expected_calls = set(store.call_ids[indices["validation"]].astype(str))
    expected_cats = set(store.cat_ids[indices["validation"]].astype(str))
    if set(calls["call_id"].astype(str)) != expected_calls or set(animals["cat_id"].astype(str)) != expected_cats:
        raise RuntimeError("IDEA-086 first fit validation identity differs")

    reference_checks: dict[str, Any] = {}
    set_history = fit["audit"]["history"]
    for pipeline in runner.REFERENCE_PIPELINES:
        reference = read_json(runner.old_fit_path(pipeline, 2713, 0, 0))
        runner.validate_reused_fit(reference, pipeline, 2713, 0, 0)
        ref_calls = pd.read_csv(ROOT / reference["validation_call_predictions"], dtype={"call_id": str, "cat_id": str})
        ref_animals = pd.read_csv(ROOT / reference["validation_animal_predictions"], dtype={"cat_id": str})
        if not np.array_equal(
            calls.sort_values("call_id")[["call_id", "cat_id", "true_label"]].to_numpy(),
            ref_calls.sort_values("call_id")[["call_id", "cat_id", "true_label"]].to_numpy(),
        ):
            raise RuntimeError(f"IDEA-086 first fit call identity/label differs from {pipeline}")
        if not np.array_equal(
            animals.sort_values("cat_id")[["cat_id", "true_label"]].to_numpy(),
            ref_animals.sort_values("cat_id")[["cat_id", "true_label"]].to_numpy(),
        ):
            raise RuntimeError(f"IDEA-086 first fit animal identity/label differs from {pipeline}")
        ref_history = reference["audit"]["history"]
        common = min(len(set_history), len(ref_history))
        for epoch in range(common):
            left = set_history[epoch]["train_audit"]
            right = ref_history[epoch]["train_audit"]
            if left["cat_order_sha256"] != right["cat_order_sha256"] or left["call_coverage_sha256"] != right["call_coverage_sha256"]:
                raise RuntimeError(f"IDEA-086 first fit batch order differs from {pipeline}")
        reference_checks[pipeline] = {
            "identity_and_labels_equal": True,
            "common_epochs_batch_order_equal": common,
            "reference_fit_summary_sha256": runner.sha256(runner.old_fit_path(pipeline, 2713, 0, 0)),
        }

    model_audit = fit["audit"]["model"]
    permutation = model_audit.get("trained_checkpoint_permutation_invariance", {})
    if not permutation.get("passed") or model_audit.get("projection_weight_norm", 0.0) <= 0.0:
        raise RuntimeError("IDEA-086 trained nonzero permutation check failed")
    if fit["audit"]["checkpoint_reload_max_probability_difference"] > 1.0e-6:
        raise RuntimeError("IDEA-086 checkpoint reload mismatch")
    protected = runner.director_protected_baseline(protocol)
    result = {
        "status": "GO_FOR_DIRECTOR_REVIEW_OF_REMAINING_35_NOT_AN_AUTHORIZATION",
        "fit_summaries_present": 1,
        "aggregation_generated": False,
        "outer_test_accessed": False,
        "device": manifest["environment"]["device"],
        "gpu": manifest["environment"]["gpu"],
        "fit_identity": {key: fit[key] for key in ("pipeline", "base_seed", "full_seed", "repeat", "fold")},
        "train_calls": fit["train_calls"],
        "validation_calls": fit["validation_calls"],
        "validation_cats": fit["validation_cats"],
        "best_epoch": fit["audit"]["best_epoch"],
        "stopped_epoch": fit["audit"]["stopped_epoch"],
        "checkpoint_reload_max_probability_difference": fit["audit"]["checkpoint_reload_max_probability_difference"],
        "trainable_parameters": model_audit["trainable_parameters"],
        "projection_weight_norm": model_audit["projection_weight_norm"],
        "trained_checkpoint_permutation_invariance": permutation,
        "probabilities_finite_normalized_argmax_valid": True,
        "call_to_animal_reconstruction_equal": True,
        "same_cell_reference_checks": reference_checks,
        "validation_animal_sha256": fit["validation_animal_sha256"],
        "validation_call_sha256": fit["validation_call_sha256"],
        "fit_summary_sha256": runner.sha256(FIT_PATH),
        "run_manifest_sha256": runner.sha256(RUN_ROOT / "run_manifest.json"),
        "partial_run_summary_sha256": runner.sha256(RUN_ROOT / "partial_run_summary.json"),
        "director_protected_baseline_files": protected["count"],
        "director_protected_baseline_sha256": protected["sha256"],
    }
    runner.write_json(OUTPUT_PATH, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
