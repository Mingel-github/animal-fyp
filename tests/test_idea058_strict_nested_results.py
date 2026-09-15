from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_idea058_strict_nested_v1_results.json"
)
REPORT_PATH = ROOT / "reports" / "33_IDEA-058_strict_nested_confirmation_results.md"
LOCK_PATH = (
    ROOT
    / "runs"
    / "meowagenet_idea058_strict_nested_v1"
    / "selection"
    / "per_fold_recipe_locks.json"
)


def result():
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_strict_nested_result_is_complete_and_covers_complete_oof() -> None:
    data = result()
    assert data["status"] == "complete"
    assert data["stage"] == "idea058_strict_nested_evaluation"
    assert data["protocol_id"] == "meowagenet-idea058-strict-nested-v1"
    assert data["selection_boundary"] == "per_repeat_outer_fold"
    assert data["complete_oof"] == {
        "calls_per_pipeline_repeat": 792,
        "animals_per_pipeline_repeat": 111,
        "evaluations": 9,
    }
    assert all(len(rows) == 3 for rows in data["metrics_by_repeat"].values())
    assert all(
        row["n"] == 111
        for rows in data["metrics_by_repeat"].values()
        for row in rows
    )


def test_per_fold_recipe_locks_are_complete_and_inner_only() -> None:
    data = result()
    selection = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    locks = data["per_fold_recipe_locks"]
    assert selection["status"] == "complete"
    assert selection["outer_test_accessed"] is False
    assert selection["candidate_fits_completed"] == 48
    assert selection["fold_locks"] == 12
    assert len(locks) == 12
    assert len({(row["repeat"], row["outer_fold"]) for row in locks}) == 12
    assert data["selected_recipe_counts"] == {
        "top1_lr1e-5": 4,
        "top1_lr3e-6": 3,
        "top2_lr1e-5": 2,
        "top2_lr3e-6": 3,
    }


def test_strict_primary_results_comparison_and_gate_are_frozen() -> None:
    data = result()
    aggregate = data["aggregate"]
    r0 = aggregate["R0_tuned_frozen_ast"]["macro_f1_mean"]
    m1 = aggregate["M1_top_block_adaptation"]["macro_f1_mean"]
    c1 = aggregate["C1_bottom_block_control"]["macro_f1_mean"]
    assert abs(r0 - 0.7570206188850608) < 1.0e-12
    assert abs(m1 - 0.7493261116879242) < 1.0e-12
    assert abs(c1 - 0.7337217185848804) < 1.0e-12
    assert r0 > m1 > c1

    paired = data["paired_summary"][
        "M1_top_block_adaptation_minus_R0_tuned_frozen_ast"
    ]
    assert paired["macro_f1_differences_by_repeat"] == [
        -0.014899313234780553,
        0.013071845404703075,
        -0.021256053761332105,
    ]
    assert paired["positive_repeats"] == 1
    assert data["idea058_strict_seed_expansion_gate"]["passed"] is False

    comparison = data["comparison_with_exploratory_idea058"]["pipelines"]
    assert comparison["R0_tuned_frozen_ast"]["strict_minus_exploratory"] == 0.0
    assert abs(
        comparison["M1_top_block_adaptation"]["strict_minus_exploratory"]
        - (-0.011351440468898022)
    ) < 1.0e-12


def test_strict_audits_inventory_and_chinese_report_are_recorded() -> None:
    data = result()
    assert data["batch_order_audit"]["fold_groups"] == 12
    assert data["batch_order_audit"]["all_common_epoch_hashes_match"] is True
    inventory = data["raw_prediction_inventory"]
    assert inventory["files"] == 91
    assert inventory["aggregate_sha256"] == (
        "48f60d63bb06e459fbd69c4d432f0051833bb5301be21f6fccace4b55426be8f"
    )
    report = REPORT_PATH.read_text(encoding="utf-8")
    assert "0.7493 ± 0.0211" in report
    assert "探索性弱正、严格复核未确认" in report
    assert "IDEA-039 grouped augmentation policy" in report
