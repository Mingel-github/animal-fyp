from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT / "configs" / "protocol" / "meowagenet_idea050_ast_lora_v1.json"
)
RUNNER_PATH = REPO_ROOT / "scripts" / "run_meowagenet_idea050_ast_lora.py"
RUN_ROOT = REPO_ROOT / "runs" / "meowagenet_idea050_ast_lora_v1"
RESULT_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_idea050_ast_lora_initial_v1_results.json"
)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_idea050_runner_separates_inner_selection_and_outer_evaluation() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert {
        "run_audit",
        "run_smoke",
        "run_selection",
        "run_evaluation",
        "load_and_verify_selection",
        "hierarchical_paired_bootstrap",
        "main",
    } <= function_names
    assert 'choices=("audit", "smoke", "select", "evaluate")' in source
    assert "include_test=False" in source
    assert "include_test=True" in source
    assert "locked_before_initial_outer_evaluation" in source
    assert 'index["runner_sha256"] != sha256(Path(__file__))' in source


def test_idea050_protocol_has_five_bounded_candidates_and_matched_control() -> None:
    protocol = load_json(PROTOCOL_PATH)
    assert protocol["protocol_id"] == "meowagenet-idea050-ast-lora-v1"
    assert protocol["pipelines"] == ["ast_head_only", "ast_lora"]
    assert protocol["shared_training"]["head_learning_rate"] == 0.006
    assert protocol["shared_training"]["dropout"] == 0.44571035356880917
    candidates = protocol["lora"]["candidates"]
    assert [candidate["candidate_id"] for candidate in candidates] == [
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
    ]
    assert all(candidate["alpha"] == 2 * candidate["rank"] for candidate in candidates)
    assert protocol["selection"]["candidate_inner_fits"] == 60
    assert protocol["initial_evaluation"]["outer_pipeline_fits"] == 24


def test_lora_linear_zero_initialization_preserves_base_output() -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_meowagenet_idea050_ast_lora as runner

    torch.manual_seed(17)
    base = torch.nn.Linear(7, 5)
    wrapper = runner.LoRALinear(base, rank=2, alpha=4, dropout=0.05)
    wrapper.eval()
    inputs = torch.randn(3, 7)
    assert torch.equal(wrapper(inputs), base(inputs))
    assert sum(parameter.numel() for parameter in wrapper.parameters() if parameter.requires_grad) == 24
    assert all(not parameter.requires_grad for parameter in wrapper.base.parameters())


def test_idea050_audit_and_smoke_records_pass_without_outer_access() -> None:
    audit = load_json(RUN_ROOT / "audit" / "audit_summary.json")
    smoke = load_json(RUN_ROOT / "smoke" / "smoke_summary.json")
    assert audit["status"] == "passed"
    assert audit["outer_test_accessed"] is False
    assert audit["target_layers_one_based"] == [9, 10, 11, 12]
    assert audit["injected_projection_count"] == 8
    assert audit["trainable_lora_parameters"] == 49_152
    assert audit["forbidden_trainable_parameters"] == []
    assert audit["initialization_equivalence"]["max_abs_logit_difference"] == 0.0
    assert smoke["status"] == "passed"
    assert smoke["outer_test_accessed"] is False
    assert smoke["fit_audit"]["all_trainable_parameters_updated"] is True
    assert smoke["fit_audit"]["checkpoint_audit"][
        "reload_max_abs_logit_difference"
    ] == 0.0


def test_idea050_selection_locked_all_twelve_folds_before_outer_access() -> None:
    index = load_json(RUN_ROOT / "selection" / "selection_index.json")
    assert index["status"] == "locked_before_initial_outer_evaluation"
    assert index["outer_test_accessed"] is False
    assert index["completed_inner_fits"] == 60
    assert index["selection_frequency"] == {
        "L1": 2,
        "L2": 3,
        "L3": 2,
        "L4": 3,
        "L5": 2,
    }
    assert len(index["selection_locks"]) == 12

    observed = set()
    for item in index["selection_locks"]:
        path = REPO_ROOT / item["path"]
        assert sha256(path) == item["sha256"]
        lock = load_json(path)
        assert lock["status"] == "locked_before_outer_evaluation"
        assert lock["outer_test_accessed"] is False
        assert lock["runner_sha256"] == sha256(RUNNER_PATH)
        assert lock["protocol_sha256"] == sha256(PROTOCOL_PATH)
        observed.add((lock["repeat"], lock["outer_fold"]))
    assert observed == {(repeat, fold) for repeat in range(3) for fold in range(4)}


def test_idea050_result_matches_completed_nested_paired_evaluation() -> None:
    result = load_json(RESULT_PATH)
    assert result["status"] == "complete_for_initial_nested_paired_evaluation"
    assert result["candidate_status"] == "screened_below_matched_head_only"
    assert result["scope"]["candidate_inner_fits"] == 60
    assert result["scope"]["outer_pipeline_fits"] == 24
    assert result["scope"]["complete_oof_evaluations"] == 6
    assert result["aggregate"]["ast_head_only"]["macro_f1_mean"] == pytest.approx(
        0.7487952289623951
    )
    assert result["aggregate"]["ast_lora"]["macro_f1_mean"] == pytest.approx(
        0.7174459939652099
    )

    contrast = result["lora_minus_matched_head_only"]
    assert contrast["macro_f1_mean_difference"] == pytest.approx(
        -0.03134923499718504
    )
    assert contrast["macro_f1_positive_repeats"] == 0
    assert all(delta < 0 for delta in contrast["macro_f1_differences"])
    assert contrast["hierarchical_bootstrap"][
        "probability_difference_positive"
    ] == pytest.approx(0.0358)
    assert result["decision"]["supports_seed_43_101_expansion"] is False
    assert result["decision"]["seeds_43_101_executed"] is False


def test_idea050_compact_audit_and_provenance_are_complete() -> None:
    result = load_json(RESULT_PATH)
    json_files = list(RUN_ROOT.rglob("*.json"))
    candidate_fits = list(
        (RUN_ROOT / "selection" / "candidates").rglob("fit_summary.json")
    )
    outer_fits = list((RUN_ROOT / "evaluation" / "fits").rglob("fit_summary.json"))
    audit = result["integrity_audit"]
    assert len(json_files) == audit["versioned_json_files"] == 104
    assert sum(path.stat().st_size for path in json_files) == (
        audit["versioned_json_bytes"]
    )
    assert len(candidate_fits) == audit["candidate_fit_summary_files"] == 60
    assert len(outer_fits) == audit["outer_fit_summary_files"] == 24
    assert all(load_json(path)["status"] == "complete" for path in candidate_fits)
    assert all(load_json(path)["status"] == "complete" for path in outer_fits)

    paths = {
        "protocol": PROTOCOL_PATH,
        "runner": RUNNER_PATH,
        "audit_summary": RUN_ROOT / "audit" / "audit_summary.json",
        "smoke_summary": RUN_ROOT / "smoke" / "smoke_summary.json",
        "selection_index": RUN_ROOT / "selection" / "selection_index.json",
        "evaluation_summary": RUN_ROOT / "evaluation" / "summary.json",
        "evaluation_run_summary": RUN_ROOT / "evaluation" / "run_summary.json",
        "evaluation_run_manifest": RUN_ROOT / "evaluation" / "run_manifest.json",
    }
    for name, path in paths.items():
        assert sha256(path) == result["provenance_sha256"][name]
