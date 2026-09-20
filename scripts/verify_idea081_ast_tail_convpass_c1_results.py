"""Independent integrity and paired-result audit for completed IDEA-081."""

from __future__ import annotations

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
RUN_ROOT = ROOT / "runs" / "meowagenet_idea081_ast_tail_convpass_c1_factorial_v1"
RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_idea081_ast_tail_convpass_c1_factorial.py"
PROTOCOL_PATH = ROOT / "configs" / "protocol" / "meowagenet_idea081_ast_tail_convpass_c1_factorial_v1.json"
OUTPUT_PATH = RUN_ROOT / "independent_audit.json"
PIPELINES = (
    "A0_frozen_tail",
    "C1_bounded_age",
    "V1_tail_full_convpass",
    "CV1_C1_plus_tail_full_convpass",
)
BASE_SEEDS = (9217, 7339, 4211)
EXPECTED_PARAMETERS = {
    PIPELINES[0]: 99_075,
    PIPELINES[1]: 108_143,
    PIPELINES[2]: 126_371,
    PIPELINES[3]: 135_439,
}
LOCKED_PROTOCOL_SHA256 = "50d30df1ece31dacfa33bc33307f8c7992b2d2f0c5d9863b08214821c0dbccd4"
LOCKED_RUNNER_SHA256 = "def87e10852506c3ac26d1133a232594f92768d9d81912d47de1d47227f97903"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def load_runner():
    spec = importlib.util.spec_from_file_location("idea081_audit_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import locked IDEA-081 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = load_runner()
    protocol = read_json(PROTOCOL_PATH)
    if sha256(PROTOCOL_PATH) != LOCKED_PROTOCOL_SHA256:
        raise RuntimeError("Locked IDEA-081 protocol hash changed")
    if sha256(RUNNER_PATH) != LOCKED_RUNNER_SHA256:
        raise RuntimeError("Locked IDEA-081 runner hash changed")
    runner.verify_protocol(protocol)
    authorization = read_json(RUN_ROOT / "gpu_authorization.json")
    startup = read_json(RUN_ROOT / "gpu_startup_audit.json")
    first_fold = read_json(RUN_ROOT / "first_complete_fold_audit.json")
    preflight = read_json(RUN_ROOT / "cpu_preflight.json")
    manifest = read_json(RUN_ROOT / "run_manifest.json")
    run_summary = read_json(RUN_ROOT / "run_summary.json")
    saved_summary_path = RUN_ROOT / "initial_evaluation_summary.json"
    saved_summary = read_json(saved_summary_path)
    for artifact, expected_status in (
        (authorization, "AUTHORIZED_FOR_FORMAL_GPU_RUN"),
        (startup, "PASS"),
        (first_fold, "PASS_RESUME_AUTHORIZED"),
        (preflight, "GO_FOR_FORMAL_GPU_RUN"),
        (run_summary, "complete"),
        (saved_summary, "complete"),
    ):
        if artifact.get("status") != expected_status:
            raise RuntimeError(f"Prerequisite status mismatch: {expected_status}")
    for artifact in (authorization, startup, first_fold, preflight, manifest):
        if artifact.get("protocol_sha256") != LOCKED_PROTOCOL_SHA256:
            raise RuntimeError("Protocol provenance mismatch")
        if artifact.get("runner_sha256") != LOCKED_RUNNER_SHA256:
            raise RuntimeError("Runner provenance mismatch")
    if any(
        artifact.get("outer_test_accessed") is not False
        for artifact in (first_fold, preflight, manifest, run_summary, saved_summary)
    ):
        raise RuntimeError("Outer-test access invariant failed")
    if authorization.get("outer_test_predictions") is not False:
        raise RuntimeError("Authorization outer-test invariant failed")

    store, _ = runner.load_token_store(protocol)
    roles = pd.read_csv(ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    runner.base.validate_roles(protocol, store, roles)
    probability_columns = list(runner.base.PROBABILITY_COLUMNS)
    fits: list[dict[str, Any]] = []
    keys = set()
    populations: dict[tuple[int, int, int], dict[str, tuple[pd.DataFrame, pd.DataFrame]]] = {}
    peak_vram = {pipeline: 0 for pipeline in PIPELINES}
    epoch_counts = {pipeline: [] for pipeline in PIPELINES}
    reload_max = 0.0
    for base_seed in BASE_SEEDS:
        for repeat in range(3):
            for fold in range(4):
                indices = runner.base.fold_indices(store, roles, repeat, fold)
                expected_call_indices = np.sort(indices["validation"].astype(np.int64))
                expected_call_ids = set(store.call_ids[expected_call_indices].astype(str))
                expected_cat_ids = set(store.cat_ids[expected_call_indices].astype(str))
                expected_train_calls = np.sort(indices["train"].astype(np.int64))
                train_call_hash = hashlib.sha256(
                    expected_train_calls.astype("<i8").tobytes()
                ).hexdigest()
                train_cats = len(set(store.cat_ids[indices["train"]].astype(str)))
                cell = (base_seed, repeat, fold)
                populations[cell] = {}
                for pipeline in PIPELINES:
                    folder = (
                        RUN_ROOT
                        / "fits"
                        / pipeline
                        / f"base_seed_{base_seed}"
                        / f"repeat_{repeat}"
                        / f"fold_{fold}"
                    )
                    summary_path = folder / "fit_summary.json"
                    fit = read_json(summary_path)
                    key = (pipeline, base_seed, repeat, fold)
                    if key in keys:
                        raise RuntimeError(f"Duplicate fit identity: {key}")
                    keys.add(key)
                    expected_identity = {
                        "status": "complete",
                        "pipeline": pipeline,
                        "base_seed": base_seed,
                        "full_seed": runner.full_seed(base_seed, repeat, fold),
                        "repeat": repeat,
                        "fold": fold,
                        "outer_test_accessed": False,
                    }
                    if any(fit.get(name) != value for name, value in expected_identity.items()):
                        raise RuntimeError(f"Fit identity mismatch: {key}")
                    animal_path = ROOT / fit["validation_animal_predictions"]
                    call_path = ROOT / fit["validation_call_predictions"]
                    if sha256(animal_path) != fit["validation_animal_sha256"]:
                        raise RuntimeError(f"Animal prediction hash mismatch: {key}")
                    if sha256(call_path) != fit["validation_call_sha256"]:
                        raise RuntimeError(f"Call prediction hash mismatch: {key}")
                    animals = pd.read_csv(animal_path, dtype={"cat_id": str})
                    calls = pd.read_csv(
                        call_path, dtype={"call_id": str, "cat_id": str}
                    )
                    if set(calls["call_id"]) != expected_call_ids:
                        raise RuntimeError(f"Validation call coverage mismatch: {key}")
                    if set(animals["cat_id"]) != expected_cat_ids:
                        raise RuntimeError(f"Validation cat coverage mismatch: {key}")
                    if not np.array_equal(
                        np.sort(calls["call_index"].to_numpy(np.int64)), expected_call_indices
                    ):
                        raise RuntimeError(f"Validation call indices mismatch: {key}")
                    if calls["call_id"].duplicated().any() or animals["cat_id"].duplicated().any():
                        raise RuntimeError(f"Duplicate prediction identity: {key}")
                    for frame in (animals, calls):
                        probabilities = frame[probability_columns].to_numpy(float)
                        if not np.isfinite(probabilities).all():
                            raise RuntimeError(f"Non-finite probability: {key}")
                        if float(np.max(np.abs(probabilities.sum(axis=1) - 1.0))) > 1e-5:
                            raise RuntimeError(f"Probability normalization failed: {key}")
                    rebuilt_animals = runner.base.calls_to_animals(calls)
                    compare_columns = [
                        "cat_id",
                        "true_label",
                        "call_count",
                        *probability_columns,
                        "predicted_label",
                    ]
                    left = animals[compare_columns].sort_values("cat_id").reset_index(drop=True)
                    right = rebuilt_animals[compare_columns].sort_values("cat_id").reset_index(drop=True)
                    if not left[["cat_id", "true_label", "call_count", "predicted_label"]].equals(
                        right[["cat_id", "true_label", "call_count", "predicted_label"]]
                    ):
                        raise RuntimeError(f"Call-to-animal identity mismatch: {key}")
                    if float(
                        np.max(
                            np.abs(
                                left[probability_columns].to_numpy(float)
                                - right[probability_columns].to_numpy(float)
                            )
                        )
                    ) > 1e-12:
                        raise RuntimeError(f"Call-to-animal probability mismatch: {key}")
                    audit = fit["audit"]
                    if audit["model"]["trainable_parameters"] != EXPECTED_PARAMETERS[pipeline]:
                        raise RuntimeError(f"Trainable parameter mismatch: {key}")
                    if not (1 <= audit["best_epoch"] <= audit["stopped_epoch"] <= 50):
                        raise RuntimeError(f"Epoch bounds mismatch: {key}")
                    if len(audit["history"]) != audit["stopped_epoch"]:
                        raise RuntimeError(f"History length mismatch: {key}")
                    if audit["checkpoint_reload_max_probability_difference"] > 1e-6:
                        raise RuntimeError(f"Checkpoint reload mismatch: {key}")
                    reload_max = max(
                        reload_max, audit["checkpoint_reload_max_probability_difference"]
                    )
                    peak_vram[pipeline] = max(peak_vram[pipeline], audit["peak_vram_bytes"])
                    epoch_counts[pipeline].append(audit["stopped_epoch"])
                    for epoch in audit["history"]:
                        train_audit = epoch["train_audit"]
                        if train_audit["calls"] != len(expected_train_calls):
                            raise RuntimeError(f"Training call coverage count mismatch: {key}")
                        if train_audit["cats"] != train_cats:
                            raise RuntimeError(f"Training cat coverage count mismatch: {key}")
                        if train_audit["call_coverage_sha256"] != train_call_hash:
                            raise RuntimeError(f"Training call coverage hash mismatch: {key}")
                    metrics = runner.base.animal_metrics(animals)
                    if abs(
                        metrics["macro_f1"]
                        - audit["best_validation_animal_metrics"]["macro_f1"]
                    ) > 1e-12:
                        raise RuntimeError(f"Macro-F1 mismatch: {key}")
                    if abs(
                        runner.base.animal_cross_entropy(animals)
                        - audit["best_validation_animal_cross_entropy"]
                    ) > 1e-12:
                        raise RuntimeError(f"Cross-entropy mismatch: {key}")
                    if abs(
                        runner.base.brier(animals) - audit["best_validation_animal_brier"]
                    ) > 1e-12:
                        raise RuntimeError(f"Brier mismatch: {key}")
                    populations[cell][pipeline] = (animals, calls)
                    fits.append(fit)
                reference_animals, reference_calls = populations[cell][PIPELINES[0]]
                for pipeline in PIPELINES[1:]:
                    animals, calls = populations[cell][pipeline]
                    if not reference_animals[["cat_id", "true_label"]].equals(
                        animals[["cat_id", "true_label"]]
                    ):
                        raise RuntimeError(f"Paired animal rows mismatch: {cell}, {pipeline}")
                    if not reference_calls[["call_index", "call_id", "cat_id", "true_label"]].equals(
                        calls[["call_index", "call_id", "cat_id", "true_label"]]
                    ):
                        raise RuntimeError(f"Paired call rows mismatch: {cell}, {pipeline}")

    if len(keys) != 144 or len(fits) != 144:
        raise RuntimeError("IDEA-081 fit matrix is incomplete")
    recomputed = runner.aggregate(fits, protocol)
    if canonical(recomputed) != saved_summary_path.read_bytes():
        raise RuntimeError("Aggregate canonical bytes differ on independent replay")
    if run_summary != {
        "status": "complete",
        "completed_fits": 144,
        "expected_fits": 144,
        "outer_test_accessed": False,
        "V1_minus_A0_gate_passed": recomputed["V1_minus_A0_gate_passed"],
        "CV1_minus_C1_gate_passed": recomputed["CV1_minus_C1_gate_passed"],
        "interaction_gate_passed": recomputed["interaction_gate_passed"],
        "full_factorial_gate_passed": recomputed["full_factorial_gate_passed"],
    }:
        raise RuntimeError("Run summary semantic mismatch")

    split = pd.DataFrame(recomputed["split_cell_results"])
    contrast_summary = {}
    for name in ("V1_minus_A0", "CV1_minus_C1", "interaction"):
        values = np.asarray(
            [row[f"{name}_macro_f1"] for row in recomputed["seed_repeat_results"]],
            dtype=float,
        )
        split_values = split[f"{name}_macro_f1"].to_numpy(float)
        contrast_summary[name] = {
            "mean_macro_f1": float(values.mean()),
            "positive_seed_repeats": int((values > 0).sum()),
            "tied_seed_repeats": int((values == 0).sum()),
            "negative_seed_repeats": int((values < 0).sum()),
            "nonnegative_split_cells": int((split_values >= 0).sum()),
            "worst_split_cell": float(split_values.min()),
            "gate_passed": bool(
                recomputed[
                    {
                        "V1_minus_A0": "V1_minus_A0_gate_passed",
                        "CV1_minus_C1": "CV1_minus_C1_gate_passed",
                        "interaction": "interaction_gate_passed",
                    }[name]
                ]
            ),
        }
    result = {
        "status": "PASS",
        "protocol_sha256": LOCKED_PROTOCOL_SHA256,
        "runner_sha256": LOCKED_RUNNER_SHA256,
        "fit_summaries": 144,
        "unique_fit_identities": 144,
        "expected_factorial_cells": 36,
        "pipelines_per_cell": 4,
        "prediction_hashes_valid": True,
        "validation_role_coverage_valid": True,
        "paired_prediction_populations_equal": True,
        "probabilities_finite_and_normalized": True,
        "call_to_animal_aggregation_valid": True,
        "training_role_coverage_valid_every_epoch": True,
        "trainable_parameter_counts_valid": True,
        "checkpoint_reload_max_probability_difference": float(reload_max),
        "maximum_peak_vram_bytes_by_pipeline": peak_vram,
        "stopped_epoch_range_by_pipeline": {
            pipeline: [int(min(values)), int(max(values))]
            for pipeline, values in epoch_counts.items()
        },
        "aggregate_semantic_match": True,
        "aggregate_canonical_byte_match": True,
        "aggregate_sha256": sha256(saved_summary_path),
        "contrast_summary": contrast_summary,
        "gate_results": {
            "V1_minus_A0": recomputed["V1_minus_A0_gate_passed"],
            "CV1_minus_C1": recomputed["CV1_minus_C1_gate_passed"],
            "interaction": recomputed["interaction_gate_passed"],
            "full_factorial": recomputed["full_factorial_gate_passed"],
        },
        "outer_test_accessed": False,
    }
    OUTPUT_PATH.write_bytes(canonical(result))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
