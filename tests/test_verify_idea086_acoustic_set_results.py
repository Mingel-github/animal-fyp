"""Tests for the independent IDEA-086 result verifier."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/verify_idea086_acoustic_set_results.py"
SPEC = importlib.util.spec_from_file_location("verify_idea086_results", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


def perfect_animals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cat_id": ["K", "A", "S"],
            "true_label": [0, 1, 2],
            "predicted_label": [0, 1, 2],
            "prob_kitten": [0.8, 0.1, 0.1],
            "prob_adult": [0.1, 0.8, 0.1],
            "prob_senior": [0.1, 0.1, 0.8],
            "base_seed": [0, 0, 0],
            "repeat": [0, 0, 0],
            "fold": [0, 0, 0],
        }
    )


def test_pipeline_and_gate_scope_is_locked() -> None:
    assert verifier.PIPELINES == (
        "SET1_acoustic_set_residual",
        "A0_ast_only",
        "C1_bounded_wide_additive",
        "T1_acoustic_temporal_residual",
        "J1_frame_shuffled_temporal_residual",
    )
    assert verifier.GATED_COMPARISONS == ("SET1_minus_A0", "SET1_minus_C1")
    assert set(verifier.COMPARISONS) == {
        "SET1_minus_A0", "SET1_minus_C1", "SET1_minus_T1", "SET1_minus_J1"
    }


def test_add_metrics_covers_accuracy_f1_ba_recalls_ce_and_brier() -> None:
    bundle = verifier.helpers.metric_bundle(perfect_animals())
    row: dict = {}
    verifier.add_metrics(row, {pipeline: bundle for pipeline in verifier.PIPELINES})
    for pipeline in verifier.PIPELINES:
        for metric in (
            "plain_accuracy", "macro_f1", "balanced_accuracy",
            "cross_entropy", "brier", "kitten_recall", "adult_recall", "senior_recall",
        ):
            assert f"{pipeline}_{metric}" in row
    for comparison in verifier.COMPARISONS:
        assert row[f"{comparison}_macro_f1"] == 0.0
        assert row[f"{comparison}_cross_entropy_gain"] == 0.0


def test_independent_source_does_not_import_idea086_runner() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "import run_meowagenet_idea086" not in source
    assert ".aggregate(" not in source


def test_locked_cpu_reuse_and_450_file_baselines() -> None:
    protocol, preflight = verifier.verify_run_locks(require_gpu_manifest=True)
    assert preflight["status"] == "GO"
    manifest = verifier.validate_reuse_manifest(protocol)
    assert len(manifest["entries"]) == 432
    assert len(manifest["director_protected_entries"]) == 450
    assert manifest["director_protected_baseline_sha256"] == (
        "161e2357e544eaffed62b458f8d25eedc4b360a60000f02ca36c09fa7db89af3"
    )


def test_current_full_results_recompute_without_differences() -> None:
    audit = verifier.full_results_audit()
    assert audit["status"] == "PASS"
    assert audit["integrity"]["new_fit_summaries_checked"] == 36
    assert audit["integrity"]["reused_fit_summaries_checked"] == 144
    assert audit["integrity"]["prediction_files_checked"] == 360
    assert audit["integrity"]["animal_predictions_reconstructed_from_calls"] == 180
    assert audit["comparison_to_official"]["difference_count"] == 0
    for values in audit["animal_occurrence_accounting"].values():
        assert values["repeated_animal_occurrences"] == 612
        assert values["unique_cats"] == 97
