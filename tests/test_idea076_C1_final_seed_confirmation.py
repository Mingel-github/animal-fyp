from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_idea076_C1_final_seed_confirmation.py"
SPEC = importlib.util.spec_from_file_location("idea076_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
idea076 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = idea076
SPEC.loader.exec_module(idea076)


def load_inputs():
    protocol = idea076.read_json(idea076.PROTOCOL_PATH)
    store = idea076.idea068.idea051.reference.historical.idea019.load_feature_store()
    features, _ = idea076.idea069.load_features(protocol, store.call_ids)
    train_indices = np.arange(0, 500, dtype=np.int64)
    probe_indices = np.arange(500, 532, dtype=np.int64)
    return protocol, store, features, train_indices, probe_indices


def test_protocol_and_all_hash_locks() -> None:
    idea076.verify_protocol(idea076.read_json(idea076.PROTOCOL_PATH))


def test_seed_derivation_and_collision_exclusion() -> None:
    protocol = idea076.read_json(idea076.PROTOCOL_PATH)
    model = protocol["model"]
    digest = hashlib.sha256(model["seed_derivation_text"].encode("utf-8")).digest()
    candidates = [
        int.from_bytes(digest[offset : offset + 4], "big") % 10000
        for offset in range(0, len(digest), 4)
    ]
    assert candidates[:6] == list(idea076.BASE_SEEDS)
    new_full = {
        base + 10_000 * repeat + 100 * fold
        for base in idea076.BASE_SEEDS
        for repeat in range(3)
        for fold in range(4)
    }
    excluded = {
        base + 10_000 * repeat + 100 * fold
        for base in model["excluded_meow_base_seeds_IDEA068_through_IDEA074"]
        for repeat in range(3)
        for fold in range(4)
    } | {
        base + 10_000 * repeat + 100 * fold
        for base in model["excluded_dog_base_seeds_IDEA075"]
        for repeat in range(3)
        for fold in range(5)
    }
    assert len(new_full) == 72
    assert not (new_full & excluded)


def test_all_models_are_identical_at_initialization() -> None:
    protocol, store, features, train_indices, probe_indices = load_inputs()
    differences = idea076.initial_logit_differences(
        protocol, store, features, train_indices, probe_indices, seed=8807
    )
    assert differences == {
        "U1_wide_unbounded_additive": 0.0,
        "C1_bounded_wide_additive": 0.0,
    }


def test_parameters_and_U1_C1_initial_state_are_exactly_matched() -> None:
    protocol, store, features, train_indices, _ = load_inputs()
    observed = {}
    states = {}
    for pipeline in idea076.PIPELINES:
        idea076.idea068.idea051.reference.historical.set_seed(8807)
        model = idea076.build_model(
            pipeline, protocol, store, features, train_indices
        )
        observed[pipeline] = sum(parameter.numel() for parameter in model.parameters())
        if pipeline != "A0_ast_only":
            states[pipeline] = model.state_dict()
    assert observed == idea076.EXPECTED_PARAMETERS
    left = states["U1_wide_unbounded_additive"]
    right = states["C1_bounded_wide_additive"]
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)


def test_C1_relative_residual_norm_is_bounded() -> None:
    protocol, store, features, train_indices, probe_indices = load_inputs()
    model = idea076.build_model(
        "C1_bounded_wide_additive", protocol, store, features, train_indices
    ).eval()
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
    assert audit["validation_relative_perturbation_max"] <= idea076.idea071.CAP + 1e-5


def test_gate_thresholds_and_mechanism_dependency_are_locked() -> None:
    gate = idea076.read_json(idea076.PROTOCOL_PATH)["gate"]
    assert gate["minimum_mean_seed_repeat_C1_minus_A0"] == 0.005
    assert gate["minimum_positive_base_seeds_C1_minus_A0"] == 4
    assert gate["minimum_positive_seed_repeats_C1_minus_A0"] == 12
    assert gate["minimum_nonnegative_split_cells_C1_minus_A0"] == 8
    assert gate["minimum_worst_split_cell_delta"] == -0.03
    assert gate["minimum_per_base_seed_senior_recall_delta_vs_A0"] == -0.02
    assert gate["minimum_positive_base_seeds_C1_minus_U1"] == 4
    assert gate["minimum_positive_seed_repeats_C1_minus_U1"] == 10
    assert gate["mechanism_interpretable_only_when_main_gate_passes"] is True


def test_canonical_json_is_LF_only_and_semantically_stable() -> None:
    value = {"status": "complete", "unicode": "猫", "nested": {"x": 1}}
    payload = idea076.canonical_json_bytes(value)
    assert b"\r\n" not in payload
    assert payload.endswith(b"\n")
    assert json.loads(payload.decode("utf-8")) == value
