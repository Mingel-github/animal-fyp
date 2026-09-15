from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_idea058_strict_nested.py"
PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea058_strict_nested_v1.json"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("idea058_strict_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_freezes_per_outer_fold_selection_boundary() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == "meowagenet-idea058-strict-nested-v1"
    assert protocol["splits"]["selection_boundary"] == "per_repeat_outer_fold"
    assert protocol["splits"]["excluded_selection_role"] == (
        "current_outer_fold_test"
    )
    assert protocol["inner_selection"]["candidate_fits"] == 48
    assert protocol["initial_evaluation"]["total_outer_fits"] == 36
    assert protocol["initial_evaluation"]["recipe_assignment"] == (
        "per_repeat_outer_fold_lock"
    )


def test_protocol_dependencies_and_fit_budgets_verify() -> None:
    runner = load_runner()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    runner.verify_protocol(protocol)


def test_only_strict_evaluation_requests_current_outer_test_indices() -> None:
    runner = load_runner()
    assert "include_test=False" in inspect.getsource(runner.run_selection)
    assert "include_test=False" in inspect.getsource(runner.run_smoke)
    assert "include_test=True" in inspect.getsource(runner.run_evaluation)


def test_evaluation_loads_recipe_inside_each_repeat_fold_boundary() -> None:
    runner = load_runner()
    source = inspect.getsource(runner.run_evaluation)
    recipe_position = source.index("recipe_lock = fold_recipe")
    fit_position = source.index("for pipeline in PIPELINES")
    assert recipe_position < fit_position
    assert 'protocol["_active_recipe_id"] = recipe_lock["selected_recipe_id"]' in source
    assert "outer_train = np.concatenate" in source
    assert "outer_train = np.sort" not in source
    assert 'pop("parameters")' in source
    assert '"parameters_vary_by_fold"' in source


def test_fold_candidate_ranking_is_local_and_uses_declared_tiebreaks() -> None:
    runner = load_runner()

    def row(recipe_id, f1, ce, parameters):
        return {
            "recipe_id": recipe_id,
            "best_validation_animal_metrics": {"macro_f1": f1},
            "best_validation_animal_cross_entropy": ce,
            "parameters": {"trainable": parameters},
        }

    rows = [
        row("larger", 0.75, 0.70, 14_000_000),
        row("higher_f1", 0.76, 0.90, 7_000_000),
        row("lower_ce", 0.75, 0.60, 14_000_000),
        row("smaller", 0.75, 0.60, 7_000_000),
    ]
    assert [entry["recipe_id"] for entry in runner.rank_fold_candidates(rows)] == [
        "higher_f1",
        "smaller",
        "lower_ce",
        "larger",
    ]


def test_fold_recipe_requires_exactly_one_matching_lock() -> None:
    runner = load_runner()
    selection = {
        "per_fold_recipe_locks": [
            {"repeat": 0, "outer_fold": 0, "selected_recipe_id": "recipe"}
        ]
    }
    assert runner.fold_recipe(selection, 0, 0)["selected_recipe_id"] == "recipe"
