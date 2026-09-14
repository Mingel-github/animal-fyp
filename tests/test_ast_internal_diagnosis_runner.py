from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_ast_internal_diagnosis as runner  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_ast_internal_diagnosis_v1.json"
)
RUN_ROOT = REPO_ROOT / "runs" / "meowagenet_ast_internal_diagnosis_v1"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_keeps_all_diagnostics_inside_inner_roles() -> None:
    protocol = load_json(PROTOCOL_PATH)
    assert protocol["stage_boundary"]["outer_test_accessed"] is False
    assert protocol["splits"]["roles_used"] == ["train", "validation"]
    assert protocol["splits"]["excluded_role"] == "test"
    assert protocol["splits"]["diagnostic_split_count"] == 12
    assert protocol["diagnosis"]["layer_probe_fits"] == 144
    assert protocol["diagnosis"]["regional_probe_fits"] == 72
    assert protocol["diagnosis"]["masked_evaluations"] == 72
    assert protocol["diagnosis"]["domain_training_trajectories"] == 24
    runner.verify_protocol(protocol)


def test_relative_groups_cover_short_and_full_sequences() -> None:
    singleton = runner.relative_groups(np.asarray([4]))
    assert [group.tolist() for group in singleton] == [[4], [4], [4]]
    pairs = runner.relative_groups(np.asarray([2, 3]))
    assert all(len(group) >= 1 for group in pairs)
    full = runner.relative_groups(np.arange(12))
    assert [len(group) for group in full] == [4, 4, 4]
    assert np.array_equal(np.concatenate(full), np.arange(12))


def test_mean_replacement_preserves_shape_and_complement() -> None:
    segment = np.arange(8 * 6, dtype=np.float32).reshape(8, 6)
    for region_index in range(6):
        replaced = runner.mean_replace_region(segment, 6, region_index)
        assert replaced.shape == segment.shape
        assert np.isfinite(replaced).all()
        assert np.array_equal(replaced[6:], segment[6:])
        assert not np.array_equal(replaced[:6], segment[:6])


def test_probe_weights_give_equal_total_to_each_class() -> None:
    labels = np.asarray([0, 0, 1, 1, 1, 2], dtype=np.int64)
    weights = runner.training_sample_weights(labels)
    totals = [weights[labels == class_index].sum() for class_index in range(3)]
    assert weights.mean() == pytest.approx(1.0)
    assert totals[0] == pytest.approx(totals[1])
    assert totals[1] == pytest.approx(totals[2])


def test_runner_has_separate_prepare_smoke_and_diagnosis_stages() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {
        "prepare_features",
        "run_smoke",
        "run_split_diagnosis",
        "aggregate_diagnosis",
        "run_diagnosis",
        "main",
    } <= names
    assert 'choices=("prepare", "smoke", "diagnose")' in source
    assert "include_test=False" in source
    assert "include_test=True" not in source
    assert "outer_test_accessed\": False" in source


def test_completed_smoke_preserves_outer_boundary() -> None:
    path = RUN_ROOT / "smoke" / "summary.json"
    if not path.is_file():
        pytest.skip("AST internal-diagnosis smoke has not run yet")
    smoke = load_json(path)
    assert smoke["status"] == "passed"
    assert smoke["outer_test_accessed"] is False
    assert smoke["layer_probe_count"] == 2
    assert smoke["regional_probe_count"] == 2
    assert smoke["masked_evaluation_count"] == 2
    assert smoke["domain_parameters"]["frozen"]["trainable"] == 99075
    assert smoke["domain_parameters"]["last2"]["trainable"] == 14276355


def test_completed_diagnosis_has_all_three_axes() -> None:
    path = RUN_ROOT / "diagnosis" / "summary.json"
    if not path.is_file():
        pytest.skip("AST internal diagnosis has not completed")
    summary = load_json(path)
    assert summary["status"] == "complete"
    assert summary["outer_test_accessed"] is False
    assert summary["diagnostic_splits"] == 12
    assert set(summary["axes"]) == {
        "intermediate_layer",
        "local_patch",
        "pretraining_domain",
    }
    assert summary["decision_record"]["status"] == "awaiting_team_selection"
