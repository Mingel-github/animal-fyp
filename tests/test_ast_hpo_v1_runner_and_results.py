from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "protocol" / "meowagenet_ast_hpo_v1.json"
RUNNER_PATH = REPO_ROOT / "scripts" / "run_meowagenet_ast_hpo_v1.py"
ENGINE_PATH = REPO_ROOT / "scripts" / "run_idea019_peft_placement.py"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
RUN_ROOT = REPO_ROOT / "runs" / "meowagenet_ast_hpo_v1"
RESULT_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_ast_hpo_v1_results.json"
)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ast_hpo_runner_has_separate_search_and_locked_evaluation_stages() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert {
        "run_search",
        "run_evaluation",
        "rank_trials",
        "aggregate_evaluation",
        "load_or_compute_probe",
        "main",
    } <= function_names
    assert 'choices=("search", "evaluate")' in source
    assert 'include_test=False' in source
    assert 'include_test=True' in source
    assert 'locked_before_exploratory_outer_evaluation' in source
    assert 'selection["runner_sha256"] != sha256(Path(__file__))' in source
    assert "import run_idea019_peft_placement as idea019" in source


def test_ast_hpo_protocol_declares_balanced_inner_only_search() -> None:
    protocol = load_json(PROTOCOL_PATH)
    assert protocol["protocol_id"] == "meowagenet-ast-hpo-v1"
    assert protocol["pipelines"] == [
        "ast_head_only",
        "ast_probe_guided_adapter",
    ]
    search = protocol["search"]
    assert search["development_repeat"] == 0
    assert search["outer_folds_used_as_inner_development_splits"] == [0, 1, 2, 3]
    assert search["trials_per_pipeline"] == 8
    assert search["total_inner_fits"] == 64
    assert len(search["head_only_trials"]) == 8
    assert len(search["adapter_trials"]) == 8
    evaluation = protocol["exploratory_evaluation"]
    assert evaluation["requires_locked_selection"] is True
    assert evaluation["total_fits"] == 24
    assert evaluation["primary_unit"] == "animal"


def test_ast_hpo_selection_was_locked_before_outer_evaluation() -> None:
    selection = load_json(RUN_ROOT / "search" / "selection.json")
    assert selection["status"] == "locked_before_exploratory_outer_evaluation"
    assert selection["outer_test_accessed"] is False
    assert selection["completed_inner_fits"] == 64
    assert selection["selected"]["ast_head_only"]["trial_id"] == "h_d04457_lr006"
    assert selection["selected"]["ast_head_only"]["parameters"][
        "head_learning_rate"
    ] == pytest.approx(0.006)
    assert selection["selected"]["ast_probe_guided_adapter"]["trial_id"] == (
        "a_baseline"
    )
    assert selection["selected_minus_baseline_inner_macro_f1"][
        "ast_head_only"
    ] == pytest.approx(0.091992755266749)
    assert selection["selected_minus_baseline_inner_macro_f1"][
        "ast_probe_guided_adapter"
    ] == pytest.approx(0.0)


def test_ast_hpo_result_matches_six_complete_oof_evaluations() -> None:
    result = load_json(RESULT_PATH)
    summary = load_json(RUN_ROOT / "evaluation" / "summary.json")
    assert result["status"] == "complete_for_exploratory_hpo"
    assert summary["status"] == "complete"
    assert summary["completed_fits"] == 24
    for pipeline in ("ast_head_only", "ast_probe_guided_adapter"):
        assert len(result["complete_oof"][pipeline]) == 3
        assert all(row["n_animals"] == 111 for row in result["complete_oof"][pipeline])
    assert result["aggregate"]["ast_head_only"]["macro_f1_mean"] == pytest.approx(
        0.7487952289623951
    )
    assert result["aggregate"]["ast_probe_guided_adapter"][
        "macro_f1_mean"
    ] == pytest.approx(0.7366843627983052)
    paired = result["paired_current_comparison"][
        "adapter_minus_head_only_macro_f1"
    ]
    assert paired["mean_difference"] == pytest.approx(-0.012110866164089814)
    assert paired["head_only_positive_repeats"] == 2


def test_ast_hpo_historical_comparison_and_provenance() -> None:
    result = load_json(RESULT_PATH)
    historical = result["historical_seed17_comparison"]
    assert historical["ast_head_only"]["mean_macro_f1_difference"] == pytest.approx(
        0.0151117575854253
    )
    adapter = historical["ast_probe_guided_adapter"]
    assert adapter["same_hyperparameters"] is True
    assert adapter["same_probe_layers_for_all_12_fits"] is True
    assert adapter["fits_with_changed_selected_epoch"] == 5
    assert adapter["mean_macro_f1_difference"] == pytest.approx(
        0.00185951454342725
    )

    paths = {
        "protocol": PROTOCOL_PATH,
        "runner": RUNNER_PATH,
        "training_engine": ENGINE_PATH,
        "roles": ROLES_PATH,
        "search_selection": RUN_ROOT / "search" / "selection.json",
        "search_run_summary": RUN_ROOT / "search" / "run_summary.json",
        "search_run_manifest": RUN_ROOT / "search" / "run_manifest.json",
        "evaluation_summary": RUN_ROOT / "evaluation" / "summary.json",
        "evaluation_run_summary": RUN_ROOT / "evaluation" / "run_summary.json",
        "evaluation_run_manifest": RUN_ROOT / "evaluation" / "run_manifest.json",
    }
    for name, path in paths.items():
        assert path.is_file(), name
        assert sha256(path) == result["provenance_sha256"][name]


def test_ast_hpo_compact_json_audit_is_complete() -> None:
    result = load_json(RESULT_PATH)
    json_files = list(RUN_ROOT.rglob("*.json"))
    fit_summaries = list(RUN_ROOT.rglob("fit_summary.json"))
    probe_summaries = list((RUN_ROOT / "probes").glob("*.json"))
    audit = result["versioned_json_audit"]
    assert len(json_files) == audit["files"] == 106
    assert sum(path.stat().st_size for path in json_files) == audit["bytes"] == 576_320
    assert len(fit_summaries) == audit["fit_summaries"] == 88
    assert len(probe_summaries) == audit["probe_summaries"] == 12

    evaluation_fits = list((RUN_ROOT / "evaluation" / "fits").rglob("fit_summary.json"))
    assert len(evaluation_fits) == 24
    assert {
        (load_json(path)["pipeline"], load_json(path)["repeat"], load_json(path)["outer_fold"])
        for path in evaluation_fits
    } == {
        (pipeline, repeat, fold)
        for pipeline in ("ast_head_only", "ast_probe_guided_adapter")
        for repeat in range(3)
        for fold in range(4)
    }
