from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_idea058_constrained_top_block_v1_results.json"
)
REPORT_PATH = ROOT / "reports" / "32_IDEA-058_constrained_top_block_results.md"


def result():
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_idea058_completed_locked_complete_oof_evaluation() -> None:
    data = result()
    assert data["status"] == "complete"
    assert data["protocol_id"] == "meowagenet-idea058-constrained-top-block-v1"
    assert data["selected_recipe_id"] == "top1_lr1e-5"
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


def test_idea058_primary_results_and_gate_are_frozen() -> None:
    data = result()
    aggregate = data["aggregate"]
    r0 = aggregate["R0_tuned_frozen_ast"]["macro_f1_mean"]
    m1 = aggregate["M1_top_block_adaptation"]["macro_f1_mean"]
    c1 = aggregate["C1_bottom_block_control"]["macro_f1_mean"]
    assert abs(r0 - 0.7570206188850608) < 1.0e-12
    assert abs(m1 - 0.7606775521568222) < 1.0e-12
    assert abs(c1 - 0.7384972960637125) < 1.0e-12
    assert m1 > r0 > c1

    paired = data["paired_summary"][
        "M1_top_block_adaptation_minus_R0_tuned_frozen_ast"
    ]
    assert paired["macro_f1_differences_by_repeat"] == [
        0.006732870340321551,
        0.033899317200866474,
        -0.029661387725903765,
    ]
    assert paired["positive_repeats"] == 2
    assert data["idea058_seed_expansion_gate"]["passed"] is False
    assert data["idea058_seed_expansion_gate"]["checks"] == {
        "mean_macro_f1_gain_over_R0": False,
        "positive_repeats_over_R0": True,
        "mean_macro_f1_gain_over_C1": True,
        "positive_repeats_over_C1": True,
        "balanced_accuracy_cost": True,
        "qwk_cost": True,
        "animal_ce_cost": True,
        "class_recall_cost": True,
    }


def test_idea058_parameter_location_and_batch_order_audits_are_recorded() -> None:
    data = result()
    training = data["training_and_parameters"]
    r0 = training["R0_tuned_frozen_ast"]["parameters"]
    m1 = training["M1_top_block_adaptation"]["parameters"]
    c1 = training["C1_bottom_block_control"]["parameters"]
    assert r0["trainable"] == 99_075
    assert m1["trainable"] == c1["trainable"] == 7_188_483
    assert m1["trainable_encoder"] == c1["trainable_encoder"] == 7_089_408
    assert m1["block_indices_zero_based"] == [11]
    assert c1["block_indices_zero_based"] == [0]
    assert data["batch_order_audit"]["fold_groups"] == 12
    assert data["batch_order_audit"]["all_common_epoch_hashes_match"] is True
    inventory = data["raw_prediction_inventory"]
    assert inventory["files"] == 91
    assert inventory["aggregate_sha256"] == (
        "98261cea320ca4275a12ae760403d80d4ea9b35e16b3e48cd4bd77c72b9dc203"
    )


def test_idea058_chinese_report_records_the_pre_acceptance_order_fix() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")
    assert "0.7607 ± 0.0371" in report
    assert "superseded_sorted_outer_order_2026-09-15" in report
    assert "517169b" in report
    assert "重跑全部 36 个 fit" in report
