from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_idea056_checkpoint_selection as runner  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea056_checkpoint_selection_confirmation_v1.json"
)
RUN_ROOT = (
    REPO_ROOT / "runs" / "meowagenet_idea056_checkpoint_selection_confirmation_v1"
)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_locks_six_new_paired_complete_oof_comparisons() -> None:
    protocol = load_json(PROTOCOL_PATH)
    settings = protocol["confirmation_evaluation"]
    assert tuple(settings["pipelines"]) == runner.PIPELINES
    assert settings["base_seeds"] == [43, 101]
    assert settings["shared_training_trajectories"] == 24
    assert settings["pipeline_fold_predictions"] == 48
    assert settings["complete_oof_evaluations"] == 12
    assert settings["paired_complete_oof_comparisons"] == 6
    assert protocol["historical_seed17"]["decision_role"].startswith("context_only")
    runner.verify_protocol(protocol)


def test_call_cross_entropy_is_unweighted_over_calls() -> None:
    calls = pd.DataFrame(
        {
            "true_label": [0, 1, 2],
            "prob_kitten": [0.8, 0.1, 0.1],
            "prob_adult": [0.1, 0.6, 0.2],
            "prob_senior": [0.1, 0.3, 0.7],
        }
    )
    expected = -np.log([0.8, 0.6, 0.7]).mean()
    assert runner.unweighted_call_cross_entropy(calls) == pytest.approx(expected)


def test_epoch_selection_uses_each_metric_and_earliest_tie() -> None:
    history = [
        {"epoch": 1, "call": 0.5, "animal": 0.7},
        {"epoch": 2, "call": 0.4, "animal": 0.6},
        {"epoch": 3, "call": 0.4, "animal": 0.65},
    ]
    assert runner.earliest_minimum_epoch(history, "call") == 2
    assert runner.earliest_minimum_epoch(history, "animal") == 2


def test_dual_patience_requires_both_metrics_to_expire() -> None:
    assert runner.dual_patience_exhausted(8, 8, 8)
    assert not runner.dual_patience_exhausted(8, 7, 8)
    assert not runner.dual_patience_exhausted(7, 8, 8)


def test_runner_separates_inner_smoke_and_outer_evaluation() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {
        "fit_inner_trajectory",
        "fit_outer_trajectory",
        "smoke",
        "verify_execution_lock",
        "aggregate_results",
        "evaluate",
        "main",
    } <= names
    assert 'choices=("smoke", "evaluate")' in source
    assert "include_test=False" in source
    assert "include_test=True" in source
    assert "locked_after_inner_only_smoke" in source


def test_completed_smoke_preserves_outer_boundary() -> None:
    path = RUN_ROOT / "smoke" / "summary.json"
    if not path.is_file():
        pytest.skip("IDEA-056 smoke has not run yet")
    smoke = load_json(path)
    assert smoke["status"] == "passed"
    assert smoke["outer_test_accessed"] is False
    assert smoke["earliest_tie_epoch_test"] == 1
    assert smoke["dual_patience_test"]["both_eight"] is True


def test_completed_evaluation_has_six_new_pairs() -> None:
    path = RUN_ROOT / "evaluation" / "summary.json"
    if not path.is_file():
        pytest.skip("IDEA-056 evaluation has not run yet")
    summary = load_json(path)
    assert summary["status"] == "complete"
    assert summary["completed_shared_training_trajectories"] == 24
    assert summary["pipeline_fold_predictions"] == 48
    assert len(summary["paired"]) == 6
    assert set(summary["aggregate"]) == set(runner.PIPELINES)
    assert all(len(rows) == 6 for rows in summary["complete_oof"].values())
    assert all(
        row["n"] == 111
        for rows in summary["complete_oof"].values()
        for row in rows
    )
