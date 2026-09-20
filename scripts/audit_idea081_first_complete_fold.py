"""Audit the mandatory first complete IDEA-081 fold before resuming all fits."""

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
RUN_ROOT = ROOT / "runs" / "meowagenet_idea081_ast_tail_convpass_c1_factorial_v1"
RUNNER = ROOT / "scripts" / "run_meowagenet_idea081_ast_tail_convpass_c1_factorial.py"
OUTPUT = RUN_ROOT / "first_complete_fold_audit.json"
PIPELINES = (
    "A0_frozen_tail",
    "C1_bounded_age",
    "V1_tail_full_convpass",
    "CV1_C1_plus_tail_full_convpass",
)
EXPECTED_PARAMETERS = {
    PIPELINES[0]: 99_075,
    PIPELINES[1]: 108_143,
    PIPELINES[2]: 126_371,
    PIPELINES[3]: 135_439,
}


def load_runner():
    spec = importlib.util.spec_from_file_location("idea081_first_fold_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load IDEA-081 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    runner = load_runner()
    manifest = read_json(RUN_ROOT / "run_manifest.json")
    authorization = read_json(RUN_ROOT / "gpu_authorization.json")
    preflight = read_json(RUN_ROOT / "cpu_preflight.json")
    if manifest["protocol_sha256"] != authorization["protocol_sha256"]:
        raise RuntimeError("Manifest/protocol authorization mismatch")
    if manifest["runner_sha256"] != authorization["runner_sha256"]:
        raise RuntimeError("Manifest/runner authorization mismatch")
    if preflight["status"] != "GO_FOR_FORMAL_GPU_RUN":
        raise RuntimeError("CPU preflight is not GO")
    if any(value.get("outer_test_accessed") is not False for value in (manifest, preflight)):
        raise RuntimeError("Outer-test lock violated before first-fold audit")

    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    rows = {}
    for pipeline in PIPELINES:
        folder = (
            RUN_ROOT
            / "fits"
            / pipeline
            / "base_seed_9217"
            / "repeat_0"
            / "fold_0"
        )
        summary_path = folder / "fit_summary.json"
        fit = read_json(summary_path)
        identity = {
            "status": "complete",
            "pipeline": pipeline,
            "base_seed": 9217,
            "full_seed": 9217,
            "repeat": 0,
            "fold": 0,
            "outer_test_accessed": False,
        }
        if any(fit.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"First-fold identity mismatch: {pipeline}")
        animal_path = ROOT / fit["validation_animal_predictions"]
        call_path = ROOT / fit["validation_call_predictions"]
        if sha256(animal_path) != fit["validation_animal_sha256"]:
            raise RuntimeError(f"Animal prediction hash mismatch: {pipeline}")
        if sha256(call_path) != fit["validation_call_sha256"]:
            raise RuntimeError(f"Call prediction hash mismatch: {pipeline}")
        animals = pd.read_csv(animal_path, dtype={"cat_id": str})
        calls = pd.read_csv(call_path, dtype={"cat_id": str, "call_id": str})
        probability_columns = list(runner.base.PROBABILITY_COLUMNS)
        for name, frame, id_column in (
            ("animal", animals, "cat_id"),
            ("call", calls, "call_id"),
        ):
            probabilities = frame[probability_columns].to_numpy(float)
            if not np.isfinite(probabilities).all():
                raise RuntimeError(f"Non-finite {name} probability: {pipeline}")
            if float(np.max(np.abs(probabilities.sum(axis=1) - 1.0))) > 1.0e-5:
                raise RuntimeError(f"Unnormalized {name} probability: {pipeline}")
            if frame[id_column].duplicated().any():
                raise RuntimeError(f"Duplicate {name} ID: {pipeline}")
        audit = fit["audit"]
        model = audit["model"]
        if model["trainable_parameters"] != EXPECTED_PARAMETERS[pipeline]:
            raise RuntimeError(f"Parameter mismatch: {pipeline}")
        if not math.isfinite(audit["best_validation_animal_cross_entropy"]):
            raise RuntimeError(f"Non-finite validation CE: {pipeline}")
        if audit["checkpoint_reload_max_probability_difference"] > 1.0e-6:
            raise RuntimeError(f"Checkpoint reload mismatch: {pipeline}")
        if not (1 <= audit["best_epoch"] <= audit["stopped_epoch"] <= 50):
            raise RuntimeError(f"Epoch audit mismatch: {pipeline}")
        if audit["peak_vram_bytes"] >= 8_188 * 1024**2:
            raise RuntimeError(f"Peak VRAM exceeds device capacity: {pipeline}")
        if len(audit["history"]) != audit["stopped_epoch"]:
            raise RuntimeError(f"History length mismatch: {pipeline}")
        for epoch in audit["history"]:
            if epoch["train_audit"]["cats"] <= 0 or epoch["train_audit"]["calls"] <= 0:
                raise RuntimeError(f"Empty training epoch: {pipeline}")
        metrics = runner.base.animal_metrics(animals)
        ce = runner.base.animal_cross_entropy(animals)
        brier = runner.base.brier(animals)
        if abs(metrics["macro_f1"] - audit["best_validation_animal_metrics"]["macro_f1"]) > 1e-12:
            raise RuntimeError(f"Macro-F1 recomputation mismatch: {pipeline}")
        if abs(ce - audit["best_validation_animal_cross_entropy"]) > 1e-12:
            raise RuntimeError(f"CE recomputation mismatch: {pipeline}")
        if abs(brier - audit["best_validation_animal_brier"]) > 1e-12:
            raise RuntimeError(f"Brier recomputation mismatch: {pipeline}")
        frames[pipeline] = (animals, calls)
        rows[pipeline] = {
            "summary_sha256": sha256(summary_path),
            "animals": int(len(animals)),
            "calls": int(len(calls)),
            "best_epoch": int(audit["best_epoch"]),
            "stopped_epoch": int(audit["stopped_epoch"]),
            "checkpoint_reload_max_probability_difference": float(
                audit["checkpoint_reload_max_probability_difference"]
            ),
            "trainable_parameters": int(model["trainable_parameters"]),
            "peak_vram_bytes": int(audit["peak_vram_bytes"]),
            "macro_f1": float(metrics["macro_f1"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "cross_entropy": float(ce),
            "brier": float(brier),
            "senior_recall": float(metrics["per_class"]["senior"]["recall"]),
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

    partial = (
        RUN_ROOT
        / "fits"
        / PIPELINES[0]
        / "base_seed_9217"
        / "repeat_0"
        / "fold_1"
    )
    partial_summary = partial / "fit_summary.json"
    if partial_summary.exists():
        raise RuntimeError("Forced pause occurred after a fifth completed fit")
    result = {
        "status": "PASS_RESUME_AUTHORIZED",
        "first_complete_cell": {"base_seed": 9217, "repeat": 0, "fold": 0},
        "complete_fits_audited": 4,
        "protocol_sha256": manifest["protocol_sha256"],
        "runner_sha256": manifest["runner_sha256"],
        "outer_test_accessed": False,
        "prediction_population_equal_all_four": True,
        "prediction_hashes_valid": True,
        "probabilities_finite_and_normalized": True,
        "checkpoint_reload_valid": True,
        "parameter_counts_valid": True,
        "partial_next_fit_without_summary": partial.exists(),
        "pipelines": rows,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
