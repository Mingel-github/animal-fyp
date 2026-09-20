from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_idea072_C1_seed_confirmation.py"
SPEC = importlib.util.spec_from_file_location("idea072_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
idea072 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = idea072
SPEC.loader.exec_module(idea072)


def load_inputs():
    protocol = idea072.read_json(idea072.PROTOCOL_PATH)
    store = idea072.idea068.idea051.reference.historical.idea019.load_feature_store()
    features, _ = idea072.idea069.load_features(protocol, store.call_ids)
    train_indices = np.arange(0, 500, dtype=np.int64)
    probe_indices = np.arange(500, 532, dtype=np.int64)
    return protocol, store, features, train_indices, probe_indices


def test_protocol_and_hash_locks() -> None:
    idea072.verify_protocol(idea072.read_json(idea072.PROTOCOL_PATH))


def test_all_models_are_exactly_identical_at_initialization() -> None:
    protocol, store, features, train_indices, probe_indices = load_inputs()
    differences = idea072.initial_logit_differences(
        protocol,
        store,
        features,
        train_indices,
        probe_indices,
        seed=6530,
    )
    assert differences == {
        "A1_age_residual": 0.0,
        "U1_wide_unbounded_additive": 0.0,
        "C1_bounded_wide_additive": 0.0,
    }


def test_U1_and_C1_have_exactly_matched_capacity() -> None:
    protocol, store, features, train_indices, _ = load_inputs()
    observed = {}
    for pipeline in idea072.PIPELINES:
        model = idea072.build_model(
            pipeline, protocol, store, features, train_indices
        )
        observed[pipeline] = sum(parameter.numel() for parameter in model.parameters())
    assert observed == idea072.EXPECTED_PARAMETERS
    assert observed["U1_wide_unbounded_additive"] == observed[
        "C1_bounded_wide_additive"
    ]


def test_U1_and_C1_start_from_identical_parameters() -> None:
    protocol, store, features, train_indices, _ = load_inputs()
    states = {}
    for pipeline in ("U1_wide_unbounded_additive", "C1_bounded_wide_additive"):
        idea072.idea068.idea051.reference.historical.set_seed(6530)
        model = idea072.build_model(
            pipeline, protocol, store, features, train_indices
        )
        states[pipeline] = model.state_dict()
    left = states["U1_wide_unbounded_additive"]
    right = states["C1_bounded_wide_additive"]
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)


def test_C1_relative_residual_norm_is_bounded() -> None:
    protocol, store, features, train_indices, probe_indices = load_inputs()
    model = idea072.build_model(
        "C1_bounded_wide_additive", protocol, store, features, train_indices
    ).eval()
    assert model.age_output is not None
    torch.nn.init.constant_(model.age_output.weight, 100.0)
    torch.nn.init.constant_(model.age_output.bias, 100.0)
    model.reset_perturbation_audit()
    with torch.no_grad():
        model(
            torch.from_numpy(store.frozen_embeddings[probe_indices]),
            torch.from_numpy(features[probe_indices]),
        )
    audit = model.perturbation_audit()
    assert audit["validation_relative_perturbation_calls"] == len(probe_indices)
    assert audit["validation_relative_perturbation_max"] <= idea072.idea071.CAP + 1e-5
