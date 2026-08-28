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


def test_local_formal_artifact_hashes_when_run_directory_is_available() -> None:
    record = load_record()
    run_dir = REPO_ROOT / record["local_run_artifacts"]["directory"]
    if not run_dir.is_dir():
        pytest.skip("Local ignored formal run directory is not present")
    for name, entry in record["local_run_artifacts"]["files"].items():
        path = run_dir / name
        assert path.is_file()
        assert path.stat().st_size == entry["bytes"]
        assert sha256(path) == entry["sha256"]
