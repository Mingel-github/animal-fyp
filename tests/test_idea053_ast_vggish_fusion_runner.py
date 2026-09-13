from __future__ import annotations

import inspect
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

import run_meowagenet_idea053_ast_vggish_fusion as runner  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea053_ast_vggish_probability_fusion_v1.json"
)
DIAGNOSTIC_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_idea053_ast_vggish_complementarity_v1.json"
)


def animal_frame(probabilities: list[list[float]], labels: list[int]) -> pd.DataFrame:
    values = np.asarray(probabilities, dtype=float)
    return pd.DataFrame(
        {
            "cat_id": [f"cat-{index}" for index in range(len(values))],
            "true_label": labels,
            "prob_kitten": values[:, 0],
            "prob_adult": values[:, 1],
            "prob_senior": values[:, 2],
            "predicted_label": values.argmax(axis=1),
        }
    )


def test_protocol_locks_24_model_fits_and_three_oof_pipelines() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == "meowagenet-idea053-ast-vggish-probability-fusion-v1"
    assert tuple(protocol["initial_evaluation"]["reported_pipelines"]) == runner.PIPELINES
    assert protocol["initial_evaluation"]["total_model_fits"] == 24
    assert protocol["initial_evaluation"]["complete_oof_evaluations"] == 9
    assert protocol["pipelines"][runner.PIPELINES[2]]["alpha_grid"] == pytest.approx(
        np.linspace(0.0, 1.0, 11)
    )
    runner.verify_protocol(protocol)


def test_diagnostic_finds_exclusive_vggish_corrections_and_blend_signal() -> None:
    diagnostic = json.loads(DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
    joint = diagnostic["aggregate_joint_correctness"]
    decision = diagnostic["diagnostic_decision"]
    assert joint == {
        "both_correct": 209,
        "ast_only_correct": 43,
        "vggish_only_correct": 21,
        "both_wrong": 60,
        "prediction_disagreement": 66,
        "oracle_either_correct_plain_accuracy": pytest.approx(0.8198198198198198),
    }
    assert decision["best_descriptive_interior_blend"]["ast_weight"] == pytest.approx(0.7)
    assert decision["mean_macro_f1_gain"] == pytest.approx(0.005982785258863976)
    assert decision["probability_fusion_signal_present"] is True


def test_input_stores_align_on_111_cats() -> None:
    ast_store, vggish_store, roles, _ = runner.load_inputs()
    assert ast_store.global_embeddings.shape == (792, 768)
    assert vggish_store["features"].shape == (936, 128)
    assert len(np.unique(ast_store.cat_ids)) == 111
    assert len(np.unique(vggish_store["cat_ids"])) == 111
    assert roles["cat_id"].nunique() == 111


def test_fused_probabilities_are_convex_and_normalized() -> None:
    ast = animal_frame([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], [0, 1])
    vggish = animal_frame([[0.2, 0.7, 0.1], [0.1, 0.2, 0.7]], [0, 1])
    fused = runner.fused_animals(ast, vggish, 0.7)
    expected = 0.7 * ast[list(runner.PROBABILITY_COLUMNS)].to_numpy() + 0.3 * vggish[
        list(runner.PROBABILITY_COLUMNS)
    ].to_numpy()
    assert fused[list(runner.PROBABILITY_COLUMNS)].to_numpy() == pytest.approx(expected)
    assert fused[list(runner.PROBABILITY_COLUMNS)].sum(axis=1).to_numpy() == pytest.approx(1.0)


def test_alpha_selection_uses_animal_cross_entropy_and_ast_tie_break() -> None:
    ast = animal_frame([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]], [0, 1])
    vggish = animal_frame([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3]], [0, 1])
    alpha, curve = runner.select_alpha(ast, vggish, [0.0, 0.5, 1.0])
    assert alpha == 1.0
    assert len(curve) == 3
    identical_alpha, _ = runner.select_alpha(ast, ast.copy(), [0.0, 0.5, 1.0])
    assert identical_alpha == 1.0


def test_runner_keeps_smoke_and_outer_evaluation_separate() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert 'choices=("smoke", "evaluate")' in source
    assert "include_outer=False" in inspect.getsource(runner.smoke)
    assert "verify_execution_lock" in inspect.getsource(runner.evaluate)
    assert "inner_alpha_curve" in source
    assert "locked_for_idea053_initial_evaluation" in source
