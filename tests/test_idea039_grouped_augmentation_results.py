from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_idea039_grouped_augmentation_v1_results.json"
)
REPORT_PATH = ROOT / "reports" / "34_IDEA-039_grouped_augmentation_results.md"
STAGE_UPDATE_PATH = ROOT / "plan" / "POST_IDEA039_stage_update.md"
SELECTION_PATH = (
    ROOT
    / "runs"
    / "meowagenet_idea039_grouped_augmentation_v1"
    / "selection"
    / "per_fold_policy_locks.json"
)


def result() -> dict:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_idea039_result_is_complete_and_covers_all_complete_oof_sets() -> None:
    data = result()
    assert data["status"] == "complete"
    assert data["stage"] == "idea039_grouped_augmentation_evaluation"
    assert data["outer_test_accessed"] is True
    assert data["protocol_id"] == "meowagenet-idea039-grouped-augmentation-v1"
    assert data["primary_unit"] == "animal"
    assert data["primary_metric"] == "macro_f1"
    assert data["complete_oof"] == {
        "calls_per_pipeline_repeat": 792,
        "animals_per_pipeline_repeat": 111,
        "evaluations": 12,
    }
    assert set(data["metrics_by_repeat"]) == {
        "R0_cached_frozen_ast",
        "C0_online_identity",
        "A1_fixed_specaugment",
        "A2_nested_policy",
    }
    assert all(len(rows) == 3 for rows in data["metrics_by_repeat"].values())
    assert all(
        row["n"] == 111
        for rows in data["metrics_by_repeat"].values()
        for row in rows
    )


def test_idea039_inner_only_policy_locks_are_complete_and_stable() -> None:
    data = result()
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    locks = data["policy_selection"]["per_fold_policy_locks"]
    assert selection["status"] == "complete"
    assert selection["outer_test_accessed"] is False
    assert selection["logical_candidate_fits_completed"] == 144
    assert selection["fold_locks"] == 12
    assert len(locks) == 12
    assert len({(row["repeat"], row["outer_fold"]) for row in locks}) == 12
    assert data["policy_selection"]["selected_policy_counts"] == {
        "P0_identity": 3,
        "P1_specaugment_light": 3,
        "P2_gain_noise_light": 2,
        "P3_shift_combo_light": 4,
    }
    assert data["policy_selection"]["stream_stable_fold_locks"] == 11


def test_idea039_primary_results_and_gates_are_frozen() -> None:
    data = result()
    aggregate = data["aggregate"]
    expected_macro_f1 = {
        "R0_cached_frozen_ast": 0.7570206188850608,
        "C0_online_identity": 0.7556784419652294,
        "A1_fixed_specaugment": 0.7578464294003341,
        "A2_nested_policy": 0.7556108909330113,
    }
    for pipeline, expected in expected_macro_f1.items():
        assert abs(aggregate[pipeline]["macro_f1_mean"] - expected) < 1.0e-12

    a1 = data["paired_summary"][
        "A1_fixed_specaugment_minus_R0_cached_frozen_ast"
    ]
    a2 = data["paired_summary"][
        "A2_nested_policy_minus_R0_cached_frozen_ast"
    ]
    assert a1["positive_repeats"] == 2
    assert a2["positive_repeats"] == 2
    assert abs(a1["mean_macro_f1_difference"] - 0.0008258105152732694) < 1e-12
    assert abs(a2["mean_macro_f1_difference"] - (-0.001409727952049522)) < 1e-12
    assert data["H039_A_fixed_augmentation_gate"]["passed"] is False
    assert data["H039_B_nested_selector_gate"]["passed"] is False


def test_idea039_path_gate_inventory_and_reports_are_recorded() -> None:
    data = result()
    path_gate = data["path_equivalence_gate"]
    assert path_gate["passed"] is False
    assert abs(path_gate["C0_minus_R0_mean_macro_f1"] - (-0.001342176919831406)) < 1e-12
    assert abs(path_gate["mean_prediction_agreement"] - 0.9579579579579579) < 1e-12

    inventory = data["raw_prediction_inventory"]
    assert inventory["files"] == 120
    assert inventory["bytes"] == 1_898_700
    assert inventory["aggregate_sha256"] == (
        "256813341d557f1fac2b3e04f7d0272c207a0b9c31303d371b57973660415673"
    )

    report = REPORT_PATH.read_text(encoding="utf-8")
    assert "0.7578 ± 0.0405" in report
    assert "95.80%" in report
    assert "固定增强弱正副指标信号、nested selector 未形成增益" in report

    stage_update = STAGE_UPDATE_PATH.read_text(encoding="utf-8")
    assert "Stage C：IDEA-039 grouped augmentation | 完成" in stage_update
    assert "Stage D：Nuisance-variable robustness | 下一活动" in stage_update
