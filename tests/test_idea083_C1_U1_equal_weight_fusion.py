from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_idea083_C1_U1_equal_weight_fusion.py"
SPEC = importlib.util.spec_from_file_location("idea083", SCRIPT)
assert SPEC and SPEC.loader
idea083 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(idea083)


def call_frame(probabilities: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "call_index": [0, 1, 2],
            "call_id": ["a", "b", "c"],
            "cat_id": ["cat1", "cat1", "cat2"],
            "true_label": [0, 0, 2],
            "prob_kitten": [row[0] for row in probabilities],
            "prob_adult": [row[1] for row in probabilities],
            "prob_senior": [row[2] for row in probabilities],
        }
    )


def test_fixed_fusion_commutes_with_cat_mean() -> None:
    c1 = call_frame([[0.8, 0.1, 0.1], [0.4, 0.5, 0.1], [0.1, 0.2, 0.7]])
    u1 = call_frame([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.2, 0.1, 0.7]])
    ensemble = c1.copy()
    ensemble[list(idea083.PROBABILITY_COLUMNS)] = 0.5 * (
        c1[list(idea083.PROBABILITY_COLUMNS)].to_numpy()
        + u1[list(idea083.PROBABILITY_COLUMNS)].to_numpy()
    )
    from_calls = idea083.calls_to_animals(ensemble).sort_values("cat_id")
    c1_animals = idea083.calls_to_animals(c1).sort_values("cat_id")
    u1_animals = idea083.calls_to_animals(u1).sort_values("cat_id")
    direct = 0.5 * (
        c1_animals[list(idea083.PROBABILITY_COLUMNS)].to_numpy()
        + u1_animals[list(idea083.PROBABILITY_COLUMNS)].to_numpy()
    )
    np.testing.assert_allclose(
        from_calls[list(idea083.PROBABILITY_COLUMNS)].to_numpy(), direct, atol=1e-15, rtol=0
    )


def test_complementarity_and_ensemble_actions() -> None:
    frame = pd.DataFrame(
        {
            "cat_id": ["a", "b", "c", "d"],
            "C1_correct": [True, False, True, False],
            "U1_correct": [True, True, False, False],
            "E_correct": [True, True, False, False],
            "C1_predicted_label": [0, 1, 2, 1],
            "U1_predicted_label": [0, 0, 1, 2],
        }
    )
    result = idea083.complementarity_counts(frame)
    assert result["both_correct"] == 1
    assert result["C1_wrong_U1_correct"] == 1
    assert result["U1_wrong_C1_correct"] == 1
    assert result["both_wrong"] == 1
    assert result["E_vs_C1_corrected"] == 1
    assert result["E_vs_C1_damaged"] == 1
    assert result["E_vs_C1_net_correct"] == 0
    assert result["oracle_upper_bound_correct_occurrences"] == 3
    assert result["oracle_is_model_score"] is False


def test_metric_bundle_and_probability_convexity() -> None:
    c1 = pd.DataFrame(
        {
            "true_label": [0, 1, 2],
            "prob_kitten": [0.8, 0.2, 0.1],
            "prob_adult": [0.1, 0.7, 0.2],
            "prob_senior": [0.1, 0.1, 0.7],
        }
    )
    u1 = pd.DataFrame(
        {
            "true_label": [0, 1, 2],
            "prob_kitten": [0.6, 0.1, 0.2],
            "prob_adult": [0.3, 0.8, 0.1],
            "prob_senior": [0.1, 0.1, 0.7],
        }
    )
    ensemble = c1.copy()
    ensemble[list(idea083.PROBABILITY_COLUMNS)] = 0.5 * (
        c1[list(idea083.PROBABILITY_COLUMNS)].to_numpy()
        + u1[list(idea083.PROBABILITY_COLUMNS)].to_numpy()
    )
    c1_metrics = idea083.metric_bundle(c1)
    u1_metrics = idea083.metric_bundle(u1)
    e_metrics = idea083.metric_bundle(ensemble)
    assert e_metrics["macro_f1"] == 1.0
    assert e_metrics["cross_entropy"] <= 0.5 * (
        c1_metrics["cross_entropy"] + u1_metrics["cross_entropy"]
    ) + 1e-15
    assert e_metrics["brier"] <= 0.5 * (c1_metrics["brier"] + u1_metrics["brier"]) + 1e-15


def test_sign_summary_uses_numeric_tolerance() -> None:
    result = idea083.sign_summary(pd.Series([0.1, 1e-14, -0.2]))
    assert result["positive"] == 1
    assert result["tied"] == 1
    assert result["negative"] == 1
    assert result["worst"] == -0.2


def test_outer_test_paths_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="Outer-test"):
        idea083.ensure_validation_path(Path("runs/example/outer_test_predictions.csv"))
    with pytest.raises(RuntimeError, match="validation"):
        idea083.ensure_validation_path(Path("runs/example/predictions.csv"))


def test_analysis_lock_is_equal_weight_and_primary_is_idea076() -> None:
    assert idea083.RUN_SPECS[0]["dataset_id"] == "IDEA076_primary"
    assert idea083.RUN_SPECS[0]["evidence_role"] == "primary_posthoc_exploratory"
    assert [spec["dataset_id"] for spec in idea083.RUN_SPECS[1:]] == [
        "IDEA072_historical_sensitivity",
        "IDEA073_historical_sensitivity",
    ]
