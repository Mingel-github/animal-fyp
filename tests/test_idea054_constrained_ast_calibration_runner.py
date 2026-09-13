from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea054_constrained_ast_calibration_v1.json"
)
RUNNER_PATH = (
    REPO_ROOT / "scripts" / "run_meowagenet_idea054_constrained_ast_calibration.py"
)
RUN_ROOT = REPO_ROOT / "runs" / "meowagenet_idea054_constrained_ast_calibration_v1"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_has_matched_four_pipeline_screen() -> None:
    protocol = load_json(PROTOCOL_PATH)
    assert protocol["protocol_id"] == "meowagenet-idea054-constrained-ast-calibration-v1"
    assert protocol["initial_evaluation"]["pipelines"] == [
        "A0_frozen_ast_reference",
        "P1_layernorm_tuning",
        "P2_block_output_ssf",
        "P3_bitfit",
    ]
    assert protocol["initial_evaluation"]["total_outer_fits"] == 48
    assert protocol["fixed_training"]["head_learning_rate"] == 0.006
    assert protocol["fixed_training"]["adaptation_learning_rate"] == 0.0003
    assert protocol["fixed_training"]["accumulation_window_calls"] == 32


def test_protocol_parameter_budgets_are_small_and_distinct() -> None:
    protocol = load_json(PROTOCOL_PATH)
    pipelines = protocol["pipelines"]
    assert pipelines["P1_layernorm_tuning"]["ast_trainable_parameters"] == 38_400
    assert pipelines["P2_block_output_ssf"]["ast_trainable_parameters"] == 18_432
    assert pipelines["P3_bitfit"]["ast_trainable_parameters"] == 102_912
    assert protocol["fixed_training"]["head_trainable_parameters"] == 99_075


def test_runner_separates_inner_only_smoke_and_outer_evaluation() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {
        "smoke",
        "verify_execution_lock",
        "fit_inner",
        "fit_outer",
        "aggregate_results",
        "evaluate",
        "main",
    } <= names
    assert 'choices=("smoke", "evaluate")' in source
    assert "include_test=False" in source
    assert "include_test=True" in source
    assert "global_weighted_micro_loss" in source
    assert "locked_after_inner_only_smoke" in source


def test_ssf_block_is_identity_at_initialization() -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_meowagenet_idea054_constrained_ast_calibration as runner

    class Dummy(torch.nn.Module):
        def forward(self, values: torch.Tensor) -> tuple[torch.Tensor]:
            return (values + 2.0,)

    block = runner.SSFBlock(Dummy(), hidden_size=5)
    values = torch.randn(3, 4, 5)
    assert torch.equal(block(values)[0], values + 2.0)
    assert sum(parameter.numel() for parameter in block.parameters()) == 10


def test_completed_smoke_has_no_outer_access_and_exact_parameter_counts() -> None:
    path = RUN_ROOT / "smoke" / "summary.json"
    if not path.is_file():
        pytest.skip("IDEA-054 smoke has not run yet")
    smoke = load_json(path)
    assert smoke["status"] == "passed"
    assert smoke["outer_test_accessed"] is False
    expected = {
        "P1_layernorm_tuning": 38_400,
        "P2_block_output_ssf": 18_432,
        "P3_bitfit": 102_912,
    }
    for pipeline, count in expected.items():
        audit = smoke["online_initialization_audit"]["parameter_audits"][pipeline]
        assert audit["trainable_adaptation_parameters"] == count
        assert audit["trainable_head_parameters"] == 99_075
        assert audit["forbidden_trainable_parameters"] == []
        fit = smoke["fits"][pipeline]["audit"]
        assert fit["updated_adaptation_parameter_tensors"] >= 1
        assert fit["updated_head_parameter_tensors"] >= 1
        assert fit["checkpoint_audit"]["reload_max_probability_difference"] <= 1e-6


def test_completed_evaluation_has_twelve_complete_oof_results() -> None:
    path = RUN_ROOT / "evaluation" / "summary.json"
    if not path.is_file():
        pytest.skip("IDEA-054 evaluation has not run yet")
    summary = load_json(path)
    assert summary["status"] == "complete"
    assert summary["completed_outer_fits"] == 48
    assert set(summary["aggregate"]) == {
        "A0_frozen_ast_reference",
        "P1_layernorm_tuning",
        "P2_block_output_ssf",
        "P3_bitfit",
    }
    assert all(len(rows) == 3 for rows in summary["complete_oof"].values())
    assert all(
        row["n"] == 111
        for rows in summary["complete_oof"].values()
        for row in rows
    )

