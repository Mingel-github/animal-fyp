"""Mandatory audit of IDEA-082's first complete nine-pipeline cell."""

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
RUN_ROOT = ROOT / "runs" / "meowagenet_idea082_age_acoustic_group_ablation_v1"
RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_idea082_age_acoustic_group_ablation.py"
OUTPUT = RUN_ROOT / "first_complete_cell_audit.json"
PIPELINES = (
    "A0_ast_only",
    "G1_f0_real",
    "G1_f0_shuffled",
    "G2_stability_real",
    "G2_stability_shuffled",
    "G3_spectral_energy_real",
    "G3_spectral_energy_shuffled",
    "G12_f0_stability_real",
    "G12_f0_stability_shuffled",
)
EXPECTED_PARAMETERS = {
    "A0_ast_only": 99_075,
    "G1_f0_real": 108_181,
    "G1_f0_shuffled": 108_181,
    "G2_stability_real": 108_099,
    "G2_stability_shuffled": 108_099,
    "G3_spectral_energy_real": 108_181,
    "G3_spectral_energy_shuffled": 108_181,
    "G12_f0_stability_real": 108_131,
    "G12_f0_stability_shuffled": 108_131,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_runner():
    spec = importlib.util.spec_from_file_location("idea082_first_cell_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load locked IDEA-082 runner")
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
    authorization = read_json(RUN_ROOT / "gpu_authorization.json")
    preflight = read_json(RUN_ROOT / "cpu_preflight.json")
    pause = read_json(RUN_ROOT / "first_complete_cell_pause.json")
    if manifest["protocol_sha256"] != authorization["protocol_sha256"]:
        raise RuntimeError("Manifest/protocol authorization mismatch")
    if manifest["runner_sha256"] != authorization["runner_sha256"]:
        raise RuntimeError("Manifest/runner authorization mismatch")
    if preflight["status"] != "GO" or pause["status"] != "PAUSED_FOR_MANDATORY_FIRST_CELL_AUDIT":
        raise RuntimeError("IDEA-082 was not paused after a clean CPU preflight")
    if any(
        value.get(key) is not False
        for value, key in (
            (manifest, "outer_test_accessed"),
            (pause, "outer_test_accessed"),
        )
    ):
        raise RuntimeError("Outer-test lock violated before first-cell audit")
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    rows: dict[str, dict] = {}
    initialization = None
    batch_histories = {}
    shuffle_hashes = {"train": set(), "validation": set()}
    for pipeline in PIPELINES:
        folder = (
            RUN_ROOT
            / "fits"
            / pipeline
            / "base_seed_8694"
            / "repeat_0"
            / "fold_0"
        )
        summary_path = folder / "fit_summary.json"
        fit = read_json(summary_path)
        identity = {
            "status": "complete",
            "pipeline": pipeline,
            "base_seed": 8694,
            "full_seed": 8694,
            "repeat": 0,
            "fold": 0,
            "outer_test_accessed": False,
        }
        if any(fit.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"First-cell identity mismatch: {pipeline}")
        animal_path = ROOT / fit["validation_animal_predictions"]
        call_path = ROOT / fit["validation_call_predictions"]
        if sha256(animal_path) != fit["validation_animal_sha256"]:
            raise RuntimeError(f"Animal prediction hash mismatch: {pipeline}")
        if sha256(call_path) != fit["validation_call_sha256"]:
            raise RuntimeError(f"Call prediction hash mismatch: {pipeline}")
        animals = pd.read_csv(animal_path, dtype={"cat_id": str})
        calls = pd.read_csv(call_path, dtype={"cat_id": str, "call_id": str})
        for name, frame, id_column in (
            ("animal", animals, "cat_id"),
            ("call", calls, "call_id"),
        ):
            probabilities = frame[list(runner.idea068.PROBABILITY_COLUMNS)].to_numpy(float)
            if not np.isfinite(probabilities).all():
                raise RuntimeError(f"Non-finite {name} probability: {pipeline}")
            if float(np.max(np.abs(probabilities.sum(axis=1) - 1.0))) > 1.0e-5:
                raise RuntimeError(f"Unnormalized {name} probability: {pipeline}")
            if frame[id_column].duplicated().any():
                raise RuntimeError(f"Duplicate {name} ID: {pipeline}")
        audit = fit["audit"]
        assert_finite(audit, pipeline)
        if audit.get("outer_test_accessed") is not False:
            raise RuntimeError(f"Outer-test audit flag changed: {pipeline}")
        if audit["model"]["trainable_parameters"] != EXPECTED_PARAMETERS[pipeline]:
            raise RuntimeError(f"Parameter mismatch: {pipeline}")
        if audit["checkpoint_reload_max_probability_difference"] > 1.0e-6:
            raise RuntimeError(f"Checkpoint reload mismatch: {pipeline}")
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
        if not all(initialization["real_shuffled_full_state_equal"].values()):
            raise RuntimeError("Real/shuffled state initialization differs")
        shuffle = fit.get("shuffle_audit")
        if pipeline.endswith("_shuffled"):
            if shuffle is None or shuffle["test_rows_materialized"] is not False:
                raise RuntimeError(f"Shuffle audit missing: {pipeline}")
            for role in ("train", "validation"):
                role_audit = shuffle["roles"][role]
                if role_audit["fixed_points"] != 0 or role_audit["multiset_equal"] is not True:
                    raise RuntimeError(f"Invalid role-local shuffle: {pipeline}/{role}")
                shuffle_hashes[role].add(role_audit["mapping_sha256"])
            if shuffle["training_statistics_equal"] is not True:
                raise RuntimeError(f"Shuffled training statistics changed: {pipeline}")
        elif shuffle is not None:
            raise RuntimeError(f"Unexpected shuffle audit: {pipeline}")
        metrics = runner.idea068.idea051.animal_metrics(animals)
        ce = runner.idea068.idea051.animal_cross_entropy(animals)
        brier = runner.idea068.brier(animals)
        if abs(metrics["macro_f1"] - audit["best_validation_animal_metrics"]["macro_f1"]) > 1.0e-12:
            raise RuntimeError(f"Macro-F1 recomputation mismatch: {pipeline}")
        if abs(ce - audit["best_validation_animal_cross_entropy"]) > 1.0e-12:
            raise RuntimeError(f"CE recomputation mismatch: {pipeline}")
        if abs(brier - audit["best_validation_animal_brier"]) > 1.0e-12:
            raise RuntimeError(f"Brier recomputation mismatch: {pipeline}")
        batch_histories[pipeline] = [item["train_audit"] for item in audit["history"]]
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
            "trainable_parameters": int(audit["model"]["trainable_parameters"]),
            "macro_f1": float(metrics["macro_f1"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "cross_entropy": float(ce),
            "brier": float(brier),
            "senior_recall": float(metrics["per_class"]["senior"]["recall"]),
        }
    if any(len(values) != 1 for values in shuffle_hashes.values()):
        raise RuntimeError("Groups used different role-local derangements")
    preflight_cell = preflight["shuffle_cells"][0]
    for role in ("train", "validation"):
        if next(iter(shuffle_hashes[role])) != preflight_cell["mapping_sha256"][role]:
            raise RuntimeError(f"Runtime/preflight shuffle hash mismatch: {role}")
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
        RUN_ROOT
        / "fits"
        / PIPELINES[0]
        / "base_seed_8694"
        / "repeat_0"
        / "fold_1"
        / "fit_summary.json"
    )
    if next_summary.exists():
        raise RuntimeError("Forced pause occurred after the next cell started")
    result = {
        "status": "PASS_RESUME_AUTHORIZED",
        "first_complete_cell": {"base_seed": 8694, "repeat": 0, "fold": 0},
        "complete_fits_audited": 9,
        "protocol_sha256": manifest["protocol_sha256"],
        "runner_sha256": manifest["runner_sha256"],
        "outer_test_accessed": False,
        "prediction_population_equal_all_nine": True,
        "prediction_hashes_valid": True,
        "probabilities_finite_and_normalized": True,
        "checkpoint_reload_valid": True,
        "initial_logits_and_common_state_valid": True,
        "real_shuffled_state_pairs_valid": True,
        "role_local_derangement_hashes_match_preflight": True,
        "paired_batch_coverage_valid": True,
        "parameters_valid": True,
        "next_cell_not_started": True,
        "shuffle_mapping_sha256": {
            role: next(iter(values)) for role, values in shuffle_hashes.items()
        },
        "pipelines": rows,
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
