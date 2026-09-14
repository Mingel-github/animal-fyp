"""Complete IDEA-057 aggregation from immutable locked-run predictions.

The locked runner completed all 36 fits before its reused IDEA-052 aggregator
requested a temporal-token summary column that IDEA-057 had not derived at the
animal level.  Every IDEA-057 call has exactly nine grid cells and the raw call
prediction files already contain ``temporal_token_count=9``.  This postprocessor
reconstructs that derived column and adapts the legacy gate-key names.  It never
trains a model, selects a recipe, or changes a prediction.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_idea057_structured_local_patch as runner  # noqa: E402


def revision_file_sha256(revision: str, path: Path) -> str:
    content = subprocess.check_output(
        ["git", "show", f"{revision}:{runner.repo_relative(path)}"],
        cwd=REPO_ROOT,
    )
    return hashlib.sha256(content).hexdigest()


def compatible_calls_to_animals(calls: pd.DataFrame) -> pd.DataFrame:
    animals = runner.calls_to_animals(calls)
    token_metadata = (
        calls.groupby("cat_id", sort=True)
        .agg(
            mean_tokens_per_call=("temporal_token_count", "mean"),
            total_temporal_tokens=("temporal_token_count", "sum"),
        )
        .reset_index()
    )
    return animals.merge(token_metadata, on="cat_id", validate="one_to_one")


def verify_completed_predictions(
    run_root: Path, protocol: dict[str, Any], lock: dict[str, Any]
) -> list[Path]:
    if lock["status"] != "locked_for_idea057_initial_evaluation":
        raise RuntimeError("Unexpected IDEA-057 lock status")
    if runner.sha256(runner.PROTOCOL_PATH) != lock["protocol_sha256"]:
        raise RuntimeError("IDEA-057 protocol differs from execution lock")
    if revision_file_sha256(lock["code_commit"], runner.Path(runner.__file__)) != lock[
        "runner_sha256"
    ]:
        raise RuntimeError("Locked IDEA-057 runner cannot be recovered from Git")
    evaluation = protocol["initial_evaluation"]
    fit_paths = []
    for base_seed in evaluation["base_seeds"]:
        for repeat in evaluation["repeats"]:
            for outer_fold in evaluation["outer_folds"]:
                for pipeline in runner.PIPELINES:
                    fit_root = (
                        run_root
                        / "evaluation"
                        / "fits"
                        / pipeline
                        / f"base_seed_{base_seed}"
                        / f"repeat_{repeat}"
                        / f"fold_{outer_fold}"
                    )
                    fit_path = fit_root / "fit_summary.json"
                    call_path = fit_root / "outer_test_call_predictions.csv"
                    animal_path = fit_root / "outer_test_animal_predictions.csv"
                    if not all(path.is_file() for path in (fit_path, call_path, animal_path)):
                        raise FileNotFoundError(f"Incomplete IDEA-057 fit: {fit_root}")
                    fit = runner.read_json(fit_path)
                    if fit["status"] != "complete" or fit["outer_test_accessed"] is not True:
                        raise RuntimeError(f"Invalid IDEA-057 fit summary: {fit_path}")
                    calls = pd.read_csv(call_path)
                    if set(calls["temporal_token_count"].unique()) != {9}:
                        raise RuntimeError(f"Unexpected IDEA-057 grid-cell count: {call_path}")
                    fit_paths.append(fit_path)
    if len(fit_paths) != int(evaluation["total_outer_fits"]):
        raise RuntimeError("IDEA-057 completed-fit count is inconsistent")
    return fit_paths


def main() -> None:
    protocol = runner.read_json(runner.PROTOCOL_PATH)
    runner.verify_protocol(protocol)
    run_root = runner.RUNS_ROOT / runner.DEFAULT_OUTPUT_SUBDIR
    lock = runner.read_json(run_root / "execution_lock.json")
    fit_paths = verify_completed_predictions(run_root, protocol, lock)
    runner.configure_base_runtime()
    runner.base.calls_to_animals = compatible_calls_to_animals
    compatible_protocol = copy.deepcopy(protocol)
    compatible_protocol["seed_expansion_gate"]["minimum_mean_macro_f1_gain"] = protocol[
        "seed_expansion_gate"
    ]["minimum_mean_macro_f1_gain_over_R0"]
    compatible_protocol["seed_expansion_gate"]["minimum_positive_repeats"] = protocol[
        "seed_expansion_gate"
    ]["minimum_positive_repeats_over_R0"]
    compatible_protocol["seed_expansion_gate"]["strong_gain"] = 0.01
    evaluation_root = run_root / "evaluation"
    summary = runner.base.aggregate_evaluation(evaluation_root, compatible_protocol)
    runner.add_idea057_gate(summary, evaluation_root, protocol)
    inventory = runner.base.previous.raw_prediction_inventory(evaluation_root)
    inventory_path = evaluation_root / "raw_prediction_inventory.json"
    runner.write_json(inventory_path, inventory)
    summary["raw_prediction_inventory"] = {
        "path": runner.repo_relative(inventory_path),
        "inventory_sha256": runner.sha256(inventory_path),
        "files": inventory["files"],
        "bytes": inventory["bytes"],
        "aggregate_sha256": inventory["aggregate_sha256"],
    }
    summary["selected_recipe_id"] = lock["selected_recipe_id"]
    summary["selection_record_sha256"] = lock["selection_record_sha256"]
    summary["code_commit"] = lock["code_commit"]
    summary["execution_lock_sha256"] = runner.sha256(run_root / "execution_lock.json")
    summary["environment_lock_sha256"] = runner.sha256(
        run_root / "environment_lock.json"
    )
    summary["runner_sha256"] = lock["runner_sha256"]
    summary["protocol_sha256"] = lock["protocol_sha256"]
    summary["postprocessing_amendment"] = {
        "reason": "derive the constant nine-grid-cell animal summary column expected by the reused IDEA-052 aggregator and map legacy gate-key names",
        "model_training_changed": False,
        "recipe_selection_changed": False,
        "raw_predictions_changed": False,
        "completed_fit_summaries_verified": len(fit_paths),
        "postprocessor_path": runner.repo_relative(Path(__file__)),
        "postprocessor_sha256": runner.sha256(Path(__file__)),
        "locked_runner_recoverable_from_commit": True,
    }
    summary_path = evaluation_root / "summary.json"
    runner.write_json(summary_path, summary)
    runner.write_json(
        evaluation_root / "run_summary.json",
        {
            "status": "complete",
            "completed_fits": len(fit_paths),
            "expected_fits": int(protocol["initial_evaluation"]["total_outer_fits"]),
            "summary_path": runner.repo_relative(summary_path),
            "summary_sha256": runner.sha256(summary_path),
            "raw_prediction_inventory_sha256": runner.sha256(inventory_path),
            "raw_prediction_aggregate_sha256": inventory["aggregate_sha256"],
            "postprocessed_without_prediction_changes": True,
        },
    )
    result_path = REPO_ROOT / protocol["outputs"]["result_metadata"]
    runner.write_json(result_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
