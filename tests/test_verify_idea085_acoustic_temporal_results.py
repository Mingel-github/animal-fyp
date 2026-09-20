"""Unit tests for the independent IDEA-085 result verifier."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/verify_idea085_acoustic_temporal_results.py"
SPEC = importlib.util.spec_from_file_location("verify_idea085_results", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


def probability_frame() -> pd.DataFrame:
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.6, 0.2, 0.2],
            [0.1, 0.8, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.1, 0.8],
            [0.2, 0.2, 0.6],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    return pd.DataFrame(
        {
            "cat_id": [f"cat_{index}" for index in range(6)],
            "true_label": labels,
            "prob_kitten": probabilities[:, 0],
            "prob_adult": probabilities[:, 1],
            "prob_senior": probabilities[:, 2],
            "predicted_label": probabilities.argmax(axis=1),
        }
    )


def test_metric_bundle_recomputes_all_requested_axes() -> None:
    frame = probability_frame()
    result = verifier.metric_bundle(frame)
    assert result["n"] == 6
    assert result["plain_accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["class_recall"] == {"kitten": 1.0, "adult": 1.0, "senior": 1.0}
    expected_ce = -np.log([0.8, 0.6, 0.8, 0.6, 0.8, 0.6]).mean()
    assert result["cross_entropy"] == pytest.approx(expected_ce)
    targets = np.eye(3)[frame["true_label"].to_numpy()]
    probabilities = frame[list(verifier.PROBABILITY_COLUMNS)].to_numpy()
    expected_brier = np.mean(np.sum((probabilities - targets) ** 2, axis=1))
    assert result["brier"] == pytest.approx(expected_brier)


def test_reconstruct_animals_uses_mean_call_probabilities() -> None:
    calls = pd.DataFrame(
        {
            "call_id": ["a1", "a2", "b1"],
            "cat_id": ["A", "A", "B"],
            "true_label": [0, 0, 1],
            "prob_kitten": [0.8, 0.4, 0.1],
            "prob_adult": [0.1, 0.3, 0.8],
            "prob_senior": [0.1, 0.3, 0.1],
        }
    )
    animals = verifier.reconstruct_animals(calls)
    assert animals["cat_id"].tolist() == ["A", "B"]
    assert animals["call_count"].tolist() == [2, 1]
    assert animals["predicted_label"].tolist() == [0, 1]
    assert animals.loc[0, "prob_kitten"] == pytest.approx(0.6)


def test_validate_probabilities_allows_call_rows_without_saved_argmax() -> None:
    calls = probability_frame().drop(columns="predicted_label")
    verifier.validate_probabilities(calls, "calls")


def test_validate_probabilities_rejects_wrong_saved_argmax() -> None:
    animals = probability_frame()
    animals.loc[0, "predicted_label"] = 2
    with pytest.raises(RuntimeError, match="not probability argmax"):
        verifier.validate_probabilities(animals, "animals")


def test_paired_transitions_counts_all_outcomes() -> None:
    common = {
        "base_seed": [1, 1, 1, 1],
        "repeat": [0, 0, 0, 0],
        "fold": [0, 0, 0, 0],
        "cat_id": ["A", "B", "C", "D"],
        "true_label": [0, 1, 2, 0],
    }
    candidate = pd.DataFrame({**common, "predicted_label": [0, 1, 0, 2]})
    comparator = pd.DataFrame({**common, "predicted_label": [1, 1, 2, 2]})
    result = verifier.paired_transitions(candidate, comparator)
    assert result == {
        "paired_occurrences": 4,
        "corrected_errors": 1,
        "introduced_errors": 1,
        "net_corrections": 0,
        "unchanged_correct": 1,
        "unchanged_wrong": 1,
    }


def test_occurrence_accounting_separates_repeats_from_unique_cats() -> None:
    frame = pd.DataFrame({"cat_id": ["A", "A", "A", "B", "B", "C"]})
    result = verifier.occurrence_accounting(frame)
    assert result["repeated_animal_occurrences"] == 6
    assert result["unique_cats"] == 3
    assert result["occurrences_per_cat_min"] == 1
    assert result["occurrences_per_cat_median"] == 2.0
    assert result["occurrences_per_cat_max"] == 3


def test_three_gated_comparisons_exclude_auxiliary_metrics() -> None:
    assert set(verifier.COMPARISONS) == {
        "T1_minus_C1", "T1_minus_A0", "T1_minus_J1"
    }
    assert set(verifier.DESCRIPTIVE_COMPARISONS) == {"C1_minus_A0_descriptive"}
    source = MODULE_PATH.read_text(encoding="utf-8")
    condition_block = source[source.index('conditions = {'):source.index('correction_keys = (')]
    assert "cross_entropy" not in condition_block
    assert "brier" not in condition_block
    assert "balanced_accuracy" not in condition_block
    assert "plain_accuracy" not in condition_block
