from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_ast_cat_balance_seed_expansion_v1 as runner  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_ast_cat_balance_seed_expansion_v1.json"
)
RUNNER_PATH = (
    REPO_ROOT
    / "scripts"
    / "run_meowagenet_ast_cat_balance_seed_expansion_v1.py"
)
RUN_ROOT = REPO_ROOT / "runs" / "meowagenet_ast_cat_balance_seed_expansion_v1"
RESULT_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_ast_cat_balance_seed_expansion_v1_results.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_freezes_the_matched_pair_and_seed_expansion_budget() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    evaluation = protocol["evaluation"]
    assert protocol["protocol_id"] == "meowagenet-ast-cat-balance-seed-expansion-v1"
    assert tuple(protocol["pipelines"]) == runner.PIPELINES
    assert evaluation["base_seeds"] == [43, 101]
    assert evaluation["repeats"] == [0, 1, 2]
    assert evaluation["outer_folds"] == [0, 1, 2, 3]
    assert evaluation["total_outer_fits"] == 48
    assert evaluation["new_complete_oof_evaluations"] == 12
    assert protocol["combined_decision"]["comparison_count_with_seed17"] == 9


def test_runner_separates_inner_only_smoke_lock_and_outer_evaluation() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert {
        "run_smoke",
        "verify_execution_lock",
        "run_evaluation",
        "aggregate_evaluation",
        "main",
    } <= function_names
    assert 'choices=("smoke", "evaluate")' in source
    assert "include_test=False" in source
    assert "include_test=True" in source
    assert "locked_for_seed43_101_evaluation" in source


def test_expansion_reuses_the_locked_seed17_training_implementation() -> None:
    protocol = runner.read_json(PROTOCOL_PATH)
    runner.verify_protocol(protocol)
    assert runner.PIPELINES == (
        "A0_final_class_balanced",
        "A1_final_cat_balanced",
    )
    assert runner.core.is_cat_balanced(runner.PIPELINES[0]) is False
    assert runner.core.is_cat_balanced(runner.PIPELINES[1]) is True


def test_combined_decision_is_positive_mean_and_six_of_nine() -> None:
    protocol = runner.read_json(PROTOCOL_PATH)
    rule = protocol["combined_decision"]["confirmation_rule"]
    assert "mean macro F1 above zero" in rule
    assert "six of nine" in rule


def test_seed_expansion_completed_48_fits_and_nine_combined_oof_pairs() -> None:
    result = runner.read_json(RESULT_PATH)
    summary = runner.read_json(RUN_ROOT / "evaluation" / "summary.json")
    assert result["status"] == "seed_expansion_complete_boundary_confirmation"
    assert summary["status"] == "complete"
    assert summary["new_completed_outer_fits"] == 48
    assert len(list((RUN_ROOT / "evaluation" / "fits").rglob("fit_summary.json"))) == 48
    assert len(list(RUN_ROOT.rglob("*.json"))) == 56
    for pipeline in runner.PIPELINES:
        assert len(summary["new_complete_oof"][pipeline]) == 6
        assert all(row["n"] == 111 for row in summary["new_complete_oof"][pipeline])
    combined = summary["combined_seed17_43_101"]
    assert combined["complete_oof_count_per_pipeline"] == 9
    assert len(combined["paired_macro_f1_differences"]) == 9


def test_cat_balancing_meets_the_locked_boundary_confirmation_rule() -> None:
    result = runner.read_json(RESULT_PATH)
    summary = runner.read_json(RUN_ROOT / "evaluation" / "summary.json")
    combined = summary["combined_seed17_43_101"]
    assert combined["mean_A1_minus_A0_macro_f1"] == pytest.approx(
        0.003142076449712847
    )
    assert combined["positive_comparisons"] == 6
    assert summary["stage_decision"]["confirmation_met"] is True
    assert result["combined_aggregate"]["A1_final_cat_balanced"][
        "macro_f1_mean"
    ] == pytest.approx(0.7416314286190961)
    assert result["combined_aggregate"]["A0_final_class_balanced"][
        "macro_f1_mean"
    ] == pytest.approx(0.7384893521693832)


def test_result_provenance_matches_locked_expansion_artifacts() -> None:
    result = runner.read_json(RESULT_PATH)
    paths = {
        "protocol": PROTOCOL_PATH,
        "runner": RUNNER_PATH,
        "roles": runner.ROLES_PATH,
        "execution_lock": RUN_ROOT / "execution_lock.json",
        "smoke_summary": RUN_ROOT / "smoke" / "summary.json",
        "evaluation_summary": RUN_ROOT / "evaluation" / "summary.json",
        "evaluation_run_summary": RUN_ROOT / "evaluation" / "run_summary.json",
    }
    for name, path in paths.items():
        assert path.is_file(), name
        assert sha256(path) == result["provenance_sha256"][name]
    lock = runner.read_json(RUN_ROOT / "execution_lock.json")
    assert lock["runner_sha256"] == sha256(RUNNER_PATH)
    assert lock["protocol_sha256"] == sha256(PROTOCOL_PATH)
