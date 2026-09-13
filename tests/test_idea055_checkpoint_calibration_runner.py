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

import run_meowagenet_idea055_checkpoint_calibration as runner  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea055_checkpoint_calibration_v1.json"
)
RUN_ROOT = REPO_ROOT / "runs" / "meowagenet_idea055_checkpoint_calibration_v1"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def animal_frame(probabilities: list[list[float]], labels: list[int]) -> pd.DataFrame:
    values = np.asarray(probabilities, dtype=float)
    return pd.DataFrame(
        {
            "cat_id": [f"cat-{index}" for index in range(len(values))],
            "true_label": labels,
            "call_count": [1] * len(values),
            "prob_kitten": values[:, 0],
            "prob_adult": values[:, 1],
            "prob_senior": values[:, 2],
            "predicted_label": values.argmax(axis=1),
        }
    )


def test_protocol_locks_shared_2x2_evaluation() -> None:
    protocol = load_json(PROTOCOL_PATH)
    assert protocol["protocol_id"] == "meowagenet-idea055-checkpoint-calibration-v1"
    assert tuple(protocol["initial_evaluation"]["pipelines"]) == runner.PIPELINES
    assert protocol["initial_evaluation"]["shared_training_trajectories"] == 12
    assert protocol["initial_evaluation"]["pipeline_fold_predictions"] == 48
    assert protocol["initial_evaluation"]["complete_oof_evaluations"] == 12
    assert protocol["class_bias_calibration"]["grid_points"] == 25
    runner.verify_protocol(protocol)


def test_tail_epochs_are_consecutive_and_end_at_selected_epoch() -> None:
    assert runner.tail_epochs(1, 3) == [1]
    assert runner.tail_epochs(2, 3) == [1, 2]
    assert runner.tail_epochs(9, 3) == [7, 8, 9]


def test_checkpoint_probability_average_is_exact_and_normalized() -> None:
    first = animal_frame([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], [0, 1])
    second = animal_frame([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]], [0, 1])
    averaged = runner.average_probability_frames([first, second], "cat_id")
    expected = (
        first[list(runner.PROBABILITY_COLUMNS)].to_numpy()
        + second[list(runner.PROBABILITY_COLUMNS)].to_numpy()
    ) / 2.0
    assert averaged[list(runner.PROBABILITY_COLUMNS)].to_numpy() == pytest.approx(
        expected
    )
    assert averaged[list(runner.PROBABILITY_COLUMNS)].sum(axis=1).to_numpy() == pytest.approx(
        1.0
    )


def test_zero_bias_is_identity_and_positive_kitten_bias_changes_odds() -> None:
    animals = animal_frame([[0.4, 0.5, 0.1], [0.1, 0.7, 0.2]], [0, 1])
    identity = runner.apply_class_bias(animals, [0.0, 0.0, 0.0])
    assert identity.equals(animals)
    calibrated = runner.apply_class_bias(animals, [0.4, 0.0, 0.0])
    original_odds = animals["prob_kitten"] / animals["prob_adult"]
    calibrated_odds = calibrated["prob_kitten"] / calibrated["prob_adult"]
    assert calibrated_odds.to_numpy() == pytest.approx(
        original_odds.to_numpy() * np.exp(0.4)
    )
    assert runner.probability_sum_error(calibrated) <= 1e-12


def test_bias_selection_uses_all_25_points_and_prefers_zero_on_exact_tie() -> None:
    protocol = load_json(PROTOCOL_PATH)
    animals = animal_frame(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0, 1, 2],
    )
    bias, curve = runner.select_class_bias(animals, protocol)
    assert len(curve) == 25
    assert bias == [0.0, 0.0, 0.0]


def test_runner_separates_inner_only_smoke_and_outer_evaluation() -> None:
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
        pytest.skip("IDEA-055 smoke has not run yet")
    smoke = load_json(path)
    assert smoke["status"] == "passed"
    assert smoke["outer_test_accessed"] is False
    assert smoke["single_bias_grid_points"] == 25
    assert smoke["ensemble_bias_grid_points"] == 25
    assert smoke["zero_bias_max_probability_difference"] == 0.0


def test_completed_evaluation_has_twelve_complete_oof_results() -> None:
    path = RUN_ROOT / "evaluation" / "summary.json"
    if not path.is_file():
        pytest.skip("IDEA-055 evaluation has not run yet")
    summary = load_json(path)
    assert summary["status"] == "complete"
    assert summary["completed_training_trajectories"] == 12
    assert summary["pipeline_fold_predictions"] == 48
    assert set(summary["aggregate"]) == set(runner.PIPELINES)
    assert all(len(rows) == 3 for rows in summary["complete_oof"].values())
    assert all(
        row["n"] == 111
        for rows in summary["complete_oof"].values()
        for row in rows
    )
