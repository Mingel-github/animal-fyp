from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_idea071_bounded_dual_path_fusion.py"
SPEC = importlib.util.spec_from_file_location("idea071_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
idea071 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = idea071
SPEC.loader.exec_module(idea071)


def load_inputs():
    protocol = idea071.read_json(idea071.PROTOCOL_PATH)
    store = idea071.idea068.idea051.reference.historical.idea019.load_feature_store()
    features, _ = idea071.idea069.load_features(protocol, store.call_ids)
    train_indices = np.arange(0, 500, dtype=np.int64)
    probe_indices = np.arange(500, 532, dtype=np.int64)
    return protocol, store, features, train_indices, probe_indices


def test_protocol_and_hash_locks() -> None:
    idea071.verify_protocol(idea071.read_json(idea071.PROTOCOL_PATH))


def test_all_models_are_exactly_identical_at_initialization() -> None:
    protocol, store, features, train_indices, probe_indices = load_inputs()
    differences = idea071.initial_logit_differences(
        protocol,
        store,
        features,
        train_indices,
        probe_indices,
        seed=4412,
    )
    assert differences == {
        "A1_age_residual": 0.0,
        "C1_bounded_wide_additive": 0.0,
        "D1_bounded_dual_path": 0.0,
    }


def test_parameter_counts_match_preregistered_capacity_control() -> None:
    protocol, store, features, train_indices, _ = load_inputs()
    observed = {}
    for pipeline in idea071.PIPELINES:
        model = idea071.build_model(
            pipeline, protocol, store, features, train_indices
        )
        observed[pipeline] = sum(parameter.numel() for parameter in model.parameters())
    assert observed == idea071.EXPECTED_PARAMETERS
    assert observed["D1_bounded_dual_path"] - observed[
        "C1_bounded_wide_additive"
    ] == 52


def test_bounded_relative_perturbations_hold_after_large_outputs() -> None:
    protocol, store, features, train_indices, probe_indices = load_inputs()
    ast = torch.from_numpy(store.frozen_embeddings[probe_indices])
    age = torch.from_numpy(features[probe_indices])
    for pipeline, maximum_ratio in (
        ("C1_bounded_wide_additive", 0.25),
        ("D1_bounded_dual_path", 0.50),
    ):
        model = idea071.build_model(
            pipeline, protocol, store, features, train_indices
        ).eval()
        assert model.age_output is not None
        torch.nn.init.constant_(model.age_output.weight, 100.0)
        torch.nn.init.constant_(model.age_output.bias, 100.0)
        if pipeline == "D1_bounded_dual_path":
            torch.nn.init.constant_(model.shift_output.weight, 100.0)
            torch.nn.init.constant_(model.shift_output.bias, 100.0)
        model.reset_perturbation_audit()
        with torch.no_grad():
            model(ast, age)
        audit = model.perturbation_audit()
        assert audit["validation_relative_perturbation_calls"] == len(probe_indices)
        assert audit["validation_relative_perturbation_max"] <= maximum_ratio + 1e-5


def test_rms_anchor_is_detached() -> None:
    hidden = torch.randn(4, 128, requires_grad=True)
    rms = idea071.hidden_rms(hidden)
    assert rms.requires_grad is False
    np.testing.assert_allclose(
        rms.numpy().reshape(-1),
        np.sqrt(
            (hidden.detach().numpy().astype(np.float64) ** 2).mean(axis=1)
            + idea071.RMS_EPSILON
        ),
        rtol=2e-6,
        atol=2e-6,
    )
