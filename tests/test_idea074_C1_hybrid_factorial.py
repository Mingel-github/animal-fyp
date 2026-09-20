from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_idea074_C1_hybrid_factorial.py"
SPEC = importlib.util.spec_from_file_location("idea074_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
idea074 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = idea074
SPEC.loader.exec_module(idea074)


def load_inputs():
    protocol = idea074.read_json(idea074.PROTOCOL_PATH)
    store = idea074.idea068.idea051.reference.historical.idea019.load_feature_store()
    features, _ = idea074.idea069.load_features(protocol, store.call_ids)
    train_indices = np.arange(0, 500, dtype=np.int64)
    probe_indices = np.arange(500, 532, dtype=np.int64)
    return protocol, store, features, train_indices, probe_indices


def test_protocol_and_hash_locks() -> None:
    idea074.verify_protocol(idea074.read_json(idea074.PROTOCOL_PATH))


def test_initial_logits_and_loss_paired_states_are_identical() -> None:
    protocol, store, features, train_indices, probe_indices = load_inputs()
    audit = idea074.initialization_audit(
        protocol,
        store,
        features,
        train_indices,
        probe_indices,
        seed=6346,
    )
    assert audit["trainable_parameters"] == idea074.EXPECTED_PARAMETERS
    assert audit["max_logit_difference_vs_A0_C"] == {
        "A0_H_hybrid": 0.0,
        "C1_C_call_only": 0.0,
        "C1_H_hybrid": 0.0,
    }
    assert audit["paired_state_dict_equal"] == {
        "A0_call_vs_hybrid": True,
        "C1_call_vs_hybrid": True,
    }
    assert audit["shared_AST_state_equal"] is True


def test_parameter_counts_match_factorial_capacity_lock() -> None:
    protocol, store, features, train_indices, _ = load_inputs()
    observed = {}
    for pipeline in idea074.PIPELINES:
        model = idea074.build_model(
            pipeline, protocol, store, features, train_indices
        )
        observed[pipeline] = sum(parameter.numel() for parameter in model.parameters())
    assert observed == idea074.EXPECTED_PARAMETERS
    assert observed[idea074.PIPELINES[0]] == observed[idea074.PIPELINES[1]]
    assert observed[idea074.PIPELINES[2]] == observed[idea074.PIPELINES[3]]


def test_loss_axis_is_fixed_and_does_not_change_architecture() -> None:
    assert [idea074.is_hybrid(pipeline) for pipeline in idea074.PIPELINES] == [
        False,
        True,
        False,
        True,
    ]
    assert [idea074.is_c1(pipeline) for pipeline in idea074.PIPELINES] == [
        False,
        False,
        True,
        True,
    ]
    assert idea074.HYBRID_CALL_WEIGHT == 0.5
    assert idea074.HYBRID_CAT_WEIGHT == 0.5


def test_C1_formula_remains_bounded_after_large_outputs() -> None:
    protocol, store, features, train_indices, probe_indices = load_inputs()
    model = idea074.build_model(
        "C1_H_hybrid", protocol, store, features, train_indices
    ).eval()
    assert isinstance(model, idea074.idea071.BoundedWideAdditiveClassifier)
    assert model.age_output is not None
    torch.nn.init.constant_(model.age_output.weight, 100.0)
    torch.nn.init.constant_(model.age_output.bias, 100.0)
    model.reset_perturbation_audit()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(store.frozen_embeddings[probe_indices]),
            torch.from_numpy(features[probe_indices]),
        )
    audit = model.perturbation_audit()
    assert torch.isfinite(logits).all()
    assert audit["validation_relative_perturbation_calls"] == len(probe_indices)
    assert audit["validation_relative_perturbation_max"] <= 0.25 + 1.0e-5


def test_cat_probability_pooling_is_an_arithmetic_mean() -> None:
    probabilities = torch.tensor(
        [[0.8, 0.1, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]],
        dtype=torch.float32,
    )
    mapping = torch.tensor([0, 0, 1], dtype=torch.long)
    pooled = idea074.idea066.cat_probabilities(probabilities, mapping, 2)
    torch.testing.assert_close(
        pooled,
        torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]]),
    )
