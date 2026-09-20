from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_idea087_nested_hpo_results.py"
PROTOCOL_PATH = ROOT / "configs/protocol/meowagenet_idea087_nested_hpo_v1.json"


def load_verifier():
    spec = importlib.util.spec_from_file_location("idea087_independent_verifier", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


verifier = load_verifier()


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_verifier_does_not_import_main_runner() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import run_meowagenet_idea087_nested_hpo" not in source
    assert "from run_meowagenet_idea087_nested_hpo" not in source


def test_locked_inputs_and_independent_roles_match_preflight() -> None:
    value = protocol()
    verifier.verify_locks(value)
    roles, calls, cats = verifier.load_locked_inputs(value)
    rebuilt = verifier.rebuild_inner_roles(value, roles)
    saved_path = verifier.RUN_ROOT / value["outputs"]["inner_roles"]
    saved = pd.read_csv(saved_path, dtype={"cat_id": str})
    pd.testing.assert_frame_equal(rebuilt, saved, check_dtype=False)
    assert len(calls) == 792
    assert len(cats) == 111
    assert len(rebuilt) == 999


def test_training_identity_rebuilds_first_preflight_cell() -> None:
    value = protocol()
    roles, calls, _ = verifier.load_locked_inputs(value)
    inner = verifier.rebuild_inner_roles(value, roles)
    cell = inner[(inner["outer_fold"] == 0) & (inner["inner_fold"] == 0)]
    cats = set(cell[cell["role"] == "train"]["cat_id"].astype(str))
    rebuilt = verifier.training_identity(calls, cats)
    preflight = verifier.read_json(
        verifier.RUN_ROOT / value["outputs"]["cpu_preflight"]
    )
    saved = next(
        row["train"]
        for row in preflight["inner_cells"]
        if row["outer_fold"] == 0 and row["inner_fold"] == 0
    )
    assert verifier.compare_nested(rebuilt, saved, "train", 1.0e-10) <= 1.0e-10


def test_metrics_refit_epoch_and_selection_are_independent() -> None:
    animals = pd.DataFrame(
        {
            "cat_id": ["a", "b", "c"],
            "true_label": [0, 1, 2],
            "prob_kitten": [0.8, 0.6, 0.1],
            "prob_adult": [0.1, 0.3, 0.2],
            "prob_senior": [0.1, 0.1, 0.7],
        }
    )
    metrics = verifier.metric_bundle(animals)
    assert metrics["plain_accuracy"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx(2 / 3)
    assert metrics["brier"] == pytest.approx(np.mean([0.06, 0.86, 0.14]))
    assert verifier.refit_epoch([1, 2, 2, 3, 9, 10]) == 3

    scores = [
        {
            "config_id": f"q{index:02d}",
            "mean_macro_f1": 0.70,
            "mean_brier": 0.40,
            "macro_f1_sample_sd": 0.02,
            "mean_plain_accuracy": 0.70,
        }
        for index in range(8)
    ]
    scores[0]["mean_macro_f1"] = 0.702
    scores[1]["mean_macro_f1"] = 0.70 - 5.0e-13
    scores[1]["mean_brier"] = 0.39
    selected, pool = verifier.select_configuration(scores, 0.002)
    assert "q01" in pool
    assert selected["config_id"] == "q01"


def test_nested_comparison_and_direction_tolerance() -> None:
    assert verifier.compare_nested(
        {"x": [1.0, {"y": True}]},
        {"x": [1.0 + 5.0e-13, {"y": True}], "extra": "allowed"},
        "root",
    ) == pytest.approx(5.0e-13)
    with pytest.raises(RuntimeError):
        verifier.compare_nested({"x": 1.0}, {"x": 1.1}, "root")

    rows = []
    for delta in (0.0, 5.0e-13, 0.1):
        row = {}
        for metric in (
            "plain_accuracy",
            "macro_f1",
            "balanced_accuracy",
            "kitten_recall",
            "adult_recall",
            "senior_recall",
            "cross_entropy",
            "brier",
        ):
            row[f"candidate_{metric}"] = 1.0 + delta
            row[f"control_{metric}"] = 1.0
        rows.append(row)
    result = verifier.comparison_summary(rows, "candidate", "control")
    assert result["direction_tolerance"] == 1.0e-12
    assert result["macro_f1_delta"]["positive"] == 1
    assert result["macro_f1_delta"]["tied"] == 2


def test_authorized_first_fit_rebuilds_from_raw_calls() -> None:
    value = protocol()
    roles, calls, _ = verifier.load_locked_inputs(value)
    inner = verifier.rebuild_inner_roles(value, roles)
    cell = inner[(inner["outer_fold"] == 0) & (inner["inner_fold"] == 0)]
    train_cats = set(cell[cell["role"] == "train"]["cat_id"].astype(str))
    validation_cats = set(
        cell[cell["role"] == "validation"]["cat_id"].astype(str)
    )
    path = verifier.inner_fit_path(
        "A0_ast_only", "q00", 0, 0, value["search"]["search_base_seeds"][0]
    )
    fit = verifier.read_json(path)
    expected = {
        "status": "complete",
        "stage": "inner",
        "pipeline": "A0_ast_only",
        "config_id": "q00",
        "config": verifier.configuration_map(value)["q00"],
        "outer_fold": 0,
        "inner_fold": 0,
        "search_base_seed": value["search"]["search_base_seeds"][0],
        "full_seed": value["search"]["search_base_seeds"][0],
        "validation_cat_ids_sha256": verifier.text_sha256(
            "\n".join(sorted(validation_cats))
        ),
        "outer_test_accessed": False,
    }
    verifier.validate_common_fit_fields(
        fit,
        value,
        expected,
        verifier.training_identity(calls, train_cats),
        "first_fit",
    )
    animals, reconstruction = verifier.validate_and_rebuild_predictions(
        fit, validation_cats, calls, "first_fit"
    )
    assert reconstruction <= verifier.TOLERANCE
    difference = verifier.verify_inner_history(
        fit, value, verifier.metric_bundle(animals), "first_fit"
    )
    assert difference <= verifier.TOLERANCE
    assert fit["audit"]["best_epoch"] == 14
    assert fit["audit"]["stopped_epoch"] == 22
