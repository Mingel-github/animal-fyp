from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_formal_v2_1_core_results.json"
)
TRACKED_RUN_ROOT = REPO_ROOT / "runs" / "meowagenet_formal_v2_1_core"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_record() -> dict[str, object]:
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


def test_formal_result_record_has_complete_locked_scope() -> None:
    record = load_record()
    execution = record["execution"]
    audit = record["integrity_audit"]
    assert record["status"] == "complete_for_selected_scope"
    assert record["formal_outcomes_accessed"] is True
    assert execution["repeat_indices"] == [0, 1, 2]
    assert execution["outer_folds"] == [0, 1, 2, 3]
    assert execution["model_seeds"] == [17, 43, 101]
    assert execution["optional_modules"] == []
    assert audit["completed_fold_level_fits"] == 108
    assert audit["failed_fold_level_fits"] == 0
    assert audit["complete_oof_files"] == 27
    assert audit["unique_cats_per_oof"] == 111
    assert audit["recomputed_metrics_match_formal_summary"] is True


def test_formal_result_record_matches_versioned_input_hashes() -> None:
    record = load_record()
    for entry in record["versioned_inputs"].values():
        path = REPO_ROOT / entry["path"]
        assert path.is_file()
        assert sha256(path) == entry["sha256"]


def test_formal_primary_contrasts_record_expected_pattern() -> None:
    record = load_record()
    contrasts = record["primary_contrasts"]
    h048 = contrasts["H048_adapter_minus_vggish"]
    h019 = contrasts["H019_adapter_minus_ast_head_only"]
    assert h048["mean_macro_f1_difference"] == pytest.approx(0.07647581521213324)
    assert h048["positive_complete_oof_evaluations"] == 9
    assert all(delta > 0 for delta in h048["paired_differences"])
    assert h019["mean_macro_f1_difference"] == pytest.approx(0.0051945822766756855)
    assert h019["positive_complete_oof_evaluations"] == 5
    assert any(delta > 0 for delta in h019["paired_differences"])
    assert any(delta < 0 for delta in h019["paired_differences"])


def test_versioned_formal_artifact_hashes() -> None:
    record = load_record()
    artifacts = record["local_run_artifacts"]
    run_dir = REPO_ROOT / artifacts["directory"]
    assert run_dir == TRACKED_RUN_ROOT
    assert artifacts["versioned_json_files"] == 123
    assert artifacts["versioned_json_bytes"] == 1_958_340
    for name, entry in record["local_run_artifacts"]["files"].items():
        path = run_dir / name
        assert path.is_file()
        assert path.stat().st_size == entry["bytes"]
        assert sha256(path) == entry["sha256"]


def test_versioned_formal_json_audit_is_complete() -> None:
    json_files = list(TRACKED_RUN_ROOT.rglob("*.json"))
    fit_summaries = list((TRACKED_RUN_ROOT / "fits").rglob("fit_summary.json"))
    probe_summaries = list((TRACKED_RUN_ROOT / "probes").glob("*.json"))
    assert len(json_files) == 123
    assert sum(path.stat().st_size for path in json_files) == 1_958_340
    assert len(fit_summaries) == 108
    assert len(probe_summaries) == 12

    run_summary = json.loads(
        (TRACKED_RUN_ROOT / "run_summary.json").read_text(encoding="utf-8")
    )
    assert run_summary["status"] == "complete"
    assert run_summary["scope"] == "formal"
    assert run_summary["completed_fits"] == 108
    assert run_summary["outer_test_predictions_produced"] is True
    assert run_summary["formal_aggregate_written"] is True
    assert len(run_summary["fits"]) == 108

    formal_summary = json.loads(
        (TRACKED_RUN_ROOT / "formal_summary.json").read_text(encoding="utf-8")
    )
    expected_means = {
        "vggish_mlp": 0.6524995850585984,
        "ast_head_only": 0.7237808179940559,
        "ast_probe_guided_adapter": 0.7289754002707316,
    }
    for pipeline, expected_mean in expected_means.items():
        evaluations = formal_summary["oof_metrics"][pipeline]
        assert len(evaluations) == 9
        assert all(metrics["n"] == 111 for metrics in evaluations.values())
        observed_mean = sum(
            metrics["macro_f1"] for metrics in evaluations.values()
        ) / len(evaluations)
        assert observed_mean == pytest.approx(expected_mean)
