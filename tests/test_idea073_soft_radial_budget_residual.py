from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT / "scripts" / "run_meowagenet_idea073_soft_radial_budget_residual.py"
)
SPEC = importlib.util.spec_from_file_location("idea073_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
idea073 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = idea073
SPEC.loader.exec_module(idea073)


def load_inputs():
    protocol = idea073.read_json(idea073.PROTOCOL_PATH)
    store = idea073.idea068.idea051.reference.historical.idea019.load_feature_store()
    features, _ = idea073.idea069.load_features(protocol, store.call_ids)
    train_indices = np.arange(0, 500, dtype=np.int64)
    probe_indices = np.arange(500, 532, dtype=np.int64)
    return protocol, store, features, train_indices, probe_indices


def test_protocol_and_hash_locks() -> None:
    idea073.verify_protocol(idea073.read_json(idea073.PROTOCOL_PATH))


def test_all_models_are_exactly_identical_at_initialization() -> None:
    protocol, store, features, train_indices, probe_indices = load_inputs()
    differences = idea073.initial_logit_differences(
        protocol,
        store,
        features,
        train_indices,
        probe_indices,
        seed=4025,
    )
    assert differences == {
        "U1_wide_unbounded_additive": 0.0,
        "C1_bounded_wide_additive": 0.0,
        "S1_soft_radial_budget": 0.0,
    }


def test_parameter_counts_are_locked_and_wide_branches_match() -> None:
    protocol, store, features, train_indices, _ = load_inputs()
    observed = {}
    for pipeline in idea073.PIPELINES:
        model = idea073.build_model(
            pipeline, protocol, store, features, train_indices
        )
        observed[pipeline] = sum(parameter.numel() for parameter in model.parameters())
    assert observed == idea073.EXPECTED_PARAMETERS
    assert len(set(observed[pipeline] for pipeline in idea073.PIPELINES[1:])) == 1


def test_U1_C1_and_S1_start_from_identical_state_dicts() -> None:
    protocol, store, features, train_indices, _ = load_inputs()
    states = {}
    for pipeline in idea073.PIPELINES[1:]:
        idea073.idea068.idea051.reference.historical.set_seed(4025)
        model = idea073.build_model(
            pipeline, protocol, store, features, train_indices
        )
        states[pipeline] = model.state_dict()
    reference = states[idea073.PIPELINES[1]]
    for pipeline in idea073.PIPELINES[2:]:
        candidate = states[pipeline]
        assert reference.keys() == candidate.keys()
        assert all(torch.equal(reference[key], candidate[key]) for key in reference)


def test_S1_preserves_direction_and_respects_radial_budget() -> None:
    protocol, store, features, train_indices, probe_indices = load_inputs()
    model = idea073.build_model(
        "S1_soft_radial_budget", protocol, store, features, train_indices
    ).eval()
    assert model.age_output is not None
    torch.nn.init.constant_(model.age_output.weight, 100.0)
    torch.nn.init.constant_(model.age_output.bias, 100.0)
    model.reset_perturbation_audit()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(store.frozen_embeddings[probe_indices]),
            torch.from_numpy(features[probe_indices]),
        )
    audit = model.radial_audit()
    assert torch.isfinite(logits).all()
    assert audit["validation_raw_to_budget_calls"] == len(probe_indices)
    assert audit["validation_residual_to_budget_calls"] == len(probe_indices)
    assert audit["validation_residual_to_budget_max"] < 1.0
    assert audit["validation_budget_violation_calls"] == 0
    assert audit["validation_max_budget_violation"] == 0.0
    assert audit["validation_direction_cosine_min"] > 0.99999


def test_S1_FP32_budget_path_stays_finite_under_cuda_autocast() -> None:
    if not torch.cuda.is_available():
        return
    protocol, store, features, train_indices, probe_indices = load_inputs()
    model = idea073.build_model(
        "S1_soft_radial_budget", protocol, store, features, train_indices
    ).cuda().eval()
    assert model.age_output is not None
    torch.nn.init.constant_(model.age_output.weight, 100.0)
    torch.nn.init.constant_(model.age_output.bias, 100.0)
    model.reset_perturbation_audit()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(
            torch.from_numpy(store.frozen_embeddings[probe_indices]).cuda(),
            torch.from_numpy(features[probe_indices]).cuda(),
        )
    audit = model.radial_audit()
    assert torch.isfinite(logits).all()
    assert audit["validation_budget_violation_calls"] == 0
    assert audit["validation_residual_to_budget_max"] <= 1.0 + 1.0e-5
