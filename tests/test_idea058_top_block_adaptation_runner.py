from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_idea058_top_block_adaptation.py"
PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea058_constrained_top_block_v1.json"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("idea058_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleList([nn.Linear(4, 4) for _ in range(12)])


class FakeAST(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = FakeEncoder()
        self.layernorm = nn.LayerNorm(4)


class FakeASTFactory:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        del args, kwargs
        return FakeAST()


def test_protocol_freezes_bounded_selection_and_matched_evaluation() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == "meowagenet-idea058-constrained-top-block-v1"
    assert protocol["splits"]["excluded_selection_role"] == "test"
    assert set(protocol["inner_selection"]["candidate_recipes"]) == {
        "top1_lr3e-6",
        "top1_lr1e-5",
        "top2_lr3e-6",
        "top2_lr1e-5",
    }
    assert protocol["inner_selection"]["candidate_fits"] == 48
    assert protocol["initial_evaluation"]["total_outer_fits"] == 36
    assert protocol["smoke"]["outer_test_accessed"] is False


def test_only_evaluation_requests_outer_test_indices() -> None:
    runner = load_runner()
    assert "include_test=False" in inspect.getsource(runner.run_selection)
    assert "include_test=False" in inspect.getsource(runner.run_smoke)
    assert "include_test=True" in inspect.getsource(runner.run_evaluation)


def test_outer_training_preserves_locked_train_then_validation_order() -> None:
    runner = load_runner()
    source = inspect.getsource(runner.run_evaluation)
    assert 'outer_train = np.concatenate(' in source
    assert 'outer_train = np.sort(' not in source


def test_top_and_bottom_controls_use_expected_blocks_and_equal_parameters(monkeypatch) -> None:
    runner = load_runner()
    monkeypatch.setattr(runner.ast_base, "ASTModel", FakeASTFactory)
    monkeypatch.setattr(
        runner.ast_base, "adapt_standard_geometry", lambda model, protocol: {"ok": True}
    )
    locked = {"ast": {"checkpoint": "fake", "revision": "fake"}}
    recipe = {"block_count": 2, "encoder_learning_rate": 1.0e-5}
    torch.manual_seed(17)
    top = runner.LocatedBlockASTClassifier(
        locked, runner.PIPELINES[1], recipe, True, nn.Linear(4, 3)
    )
    torch.manual_seed(17)
    bottom = runner.LocatedBlockASTClassifier(
        locked, runner.PIPELINES[2], recipe, True, nn.Linear(4, 3)
    )
    top_audit = runner.parameter_audit(top)
    bottom_audit = runner.parameter_audit(bottom)
    assert top.block_indices == [10, 11]
    assert bottom.block_indices == [0, 1]
    assert top_audit["trainable_encoder"] == bottom_audit["trainable_encoder"]
    assert top_audit["trainable"] == bottom_audit["trainable"]
    assert all(
        parameter.requires_grad == (index in top.block_indices)
        for index, block in enumerate(top.ast.encoder.layer)
        for parameter in block.parameters()
    )
    assert all(
        parameter.requires_grad == (index in bottom.block_indices)
        for index, block in enumerate(bottom.ast.encoder.layer)
        for parameter in block.parameters()
    )


def test_portable_checkpoint_restores_trainable_tensors_and_head_buffers() -> None:
    runner = load_runner()
    model = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 3))
    state = runner.portable_state_dict(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
        model[0].running_mean.add_(1.0)
    runner.load_portable_state_dict(model, state)
    restored = model.state_dict()
    for name, value in state.items():
        assert torch.equal(restored[name], value)


def test_seed_gate_requires_reference_gain_and_top_location_gain() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    gate = protocol["seed_expansion_gate"]
    assert gate["minimum_mean_macro_f1_gain_over_R0"] == 0.005
    assert gate["minimum_positive_repeats_over_R0"] == 2
    assert gate["minimum_mean_macro_f1_gain_over_C1"] == 0.0
    assert gate["minimum_positive_repeats_over_C1"] == 2
