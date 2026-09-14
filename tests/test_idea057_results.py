from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_idea057_structured_local_patch_v1_results.json"
)
RUN_ROOT = ROOT / "runs" / "meowagenet_idea057_structured_local_patch_v1"


def result():
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_idea057_completed_all_locked_outer_fits() -> None:
    data = result()
    assert data["status"] == "complete"
    assert data["completed_outer_fits"] == 36
    assert data["selected_recipe_id"] == "middle_frequency_strip"
    for pipeline, repeats in data["complete_oof"].items():
        assert pipeline in data["aggregate"]
        assert len(repeats) == 3
        assert all(row["n"] == 111 for row in repeats)


def test_r0_leads_m1_and_position_removed_control() -> None:
    data = result()
    aggregate = data["aggregate"]
    r0 = aggregate["R0_tuned_frozen_ast"]["macro_f1_mean"]
    m1 = aggregate["M1_position_aware_local_patch"]["macro_f1_mean"]
    c1 = aggregate["C1_position_removed_control"]["macro_f1_mean"]
    assert abs(r0 - 0.7570206188850608) < 1.0e-12
    assert abs(m1 - 0.7265022198626832) < 1.0e-12
    assert abs(c1 - 0.7348329998866446) < 1.0e-12
    assert r0 > c1 > m1


def test_primary_paired_result_and_gate_are_recorded() -> None:
    data = result()
    comparison = data["paired_summary"][
        "M1_position_aware_local_patch_minus_R0_tuned_frozen_ast"
    ]
    assert comparison["macro_f1_differences"] == [
        0.0,
        -0.04964480031802643,
        -0.04191039674910646,
    ]
    assert comparison["positive_repeats"] == 0
    assert data["idea057_seed_expansion_gate"]["passed"] is False


def test_locked_predictions_are_unchanged_by_postprocessing() -> None:
    data = result()
    amendment = data["postprocessing_amendment"]
    assert amendment["completed_fit_summaries_verified"] == 36
    assert amendment["model_training_changed"] is False
    assert amendment["recipe_selection_changed"] is False
    assert amendment["raw_predictions_changed"] is False
    assert amendment["locked_runner_recoverable_from_commit"] is True
    inventory = data["raw_prediction_inventory"]
    assert inventory["aggregate_sha256"] == (
        "f1986b1c971bf2ad6f67d81870035278b9c5c0248395e59d63f16d02937c490a"
    )
    assert (RUN_ROOT / "execution_lock.json").is_file()
