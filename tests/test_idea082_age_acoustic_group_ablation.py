from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_idea082_age_acoustic_group_ablation.py"
SPEC = importlib.util.spec_from_file_location("idea082_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
idea082 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = idea082
SPEC.loader.exec_module(idea082)


def load_cell():
    protocol = idea082.read_json(idea082.PROTOCOL_PATH)
    store = idea082.idea068.idea051.reference.historical.idea019.load_feature_store()
    features, _ = idea082.load_features(protocol, store.call_ids)
    roles = pd.read_csv(
        ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str}
    )
    indices = idea082.role_cell_indices(store, roles, repeat=0, fold=0)
    matrices = {idea082.PIPELINES[0]: features}
    for pipeline in idea082.PIPELINES[1:]:
        matrix, _ = idea082.prepare_pipeline_features(
            pipeline,
            features,
            indices["train"],
            indices["validation"],
            repeat=0,
            fold=0,
        )
        matrices[pipeline] = matrix
    return protocol, store, features, indices, matrices


def test_protocol_hash_locks_and_frozen_group_partition() -> None:
    protocol = idea082.read_json(idea082.PROTOCOL_PATH)
    idea082.verify_protocol(protocol)
    primary = [set(idea082.GROUP_INDICES[group]) for group in idea082.GROUPS[:3]]
    assert not (primary[0] & primary[1])
    assert not (primary[0] & primary[2])
    assert not (primary[1] & primary[2])
    assert set().union(*primary) == set(range(20))
    assert set(idea082.GROUP_INDICES["G12_f0_stability"]) == primary[0] | primary[1]


def test_parameter_budget_formula_and_pipeline_counts() -> None:
    protocol = idea082.read_json(idea082.PROTOCOL_PATH)
    assert len(idea082.PIPELINES) == 9
    assert protocol["model"]["total_fits"] == 324
    for group, indices in idea082.GROUP_INDICES.items():
        width = idea082.GROUP_WIDTHS[group]
        age_parameters = width * (len(indices) + 129) + 128
        assert age_parameters == protocol["feature_groups"][group][
            "age_branch_parameters"
        ]
        assert abs(age_parameters - 9068) <= 44


def test_seed_derivation_and_prior_collision_exclusion() -> None:
    model = idea082.read_json(idea082.PROTOCOL_PATH)["model"]
    digest = hashlib.sha256(model["seed_derivation_text"].encode("utf-8")).digest()
    candidates = [
        int.from_bytes(digest[offset : offset + 4], "big") % 10000
        for offset in range(0, len(digest), 4)
    ]
    assert candidates == model["candidate_sequence"]
    assert candidates[1:4] == list(idea082.BASE_SEEDS)
    current = {
        seed + 10_000 * repeat + 100 * fold
        for seed in idea082.BASE_SEEDS
        for repeat in range(3)
        for fold in range(4)
    }
    prior = {
        seed + 10_000 * repeat + 100 * fold
        for seed in model["known_prior_base_seeds"]
        for repeat in range(3)
        for fold in range(5)
    }
    assert len(current) == 36
    assert not (current & prior)
    assert 4712 in prior


def test_role_derangement_is_reproducible_and_role_local() -> None:
    _, _, _, indices, _ = load_cell()
    mappings = {}
    for role in ("train", "validation"):
        first = idea082.role_derangement(indices[role], 0, 0, role)
        second = idea082.role_derangement(indices[role], 0, 0, role)
        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])
        assert first[2] == second[2]
        assert first[2]["fixed_points"] == 0
        assert set(first[0]) == set(indices[role])
        assert set(first[1]) == set(indices[role])
        mappings[role] = first[2]["mapping_sha256"]
    assert mappings["train"] != mappings["validation"]


def test_shuffled_controls_preserve_multisets_stats_and_leave_test_unmaterialized() -> None:
    protocol, store, features, indices, _ = load_cell()
    roles = pd.read_csv(
        ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str}
    )
    cell = roles[(roles["repeat"] == 0) & (roles["outer_fold"] == 0)]
    test_cats = set(cell[cell["role"] == "test"]["cat_id"].astype(str))
    test_indices = np.flatnonzero(
        np.isin(store.cat_ids.astype(str), np.asarray(sorted(test_cats)))
    )
    mapping_hashes = {"train": set(), "validation": set()}
    for group in idea082.GROUPS:
        pipeline = f"{group}_shuffled"
        matrix, audit = idea082.prepare_pipeline_features(
            pipeline,
            features,
            indices["train"],
            indices["validation"],
            0,
            0,
        )
        assert audit is not None
        assert audit["training_statistics_equal"] is True
        assert np.isnan(matrix[test_indices]).all()
        for role in ("train", "validation"):
            assert audit["roles"][role]["multiset_equal"] is True
            assert audit["roles"][role]["fixed_points"] == 0
            mapping_hashes[role].add(audit["roles"][role]["mapping_sha256"])
    assert all(len(values) == 1 for values in mapping_hashes.values())


def test_initial_logits_parameters_and_real_shuffled_states_are_paired() -> None:
    protocol, store, _, indices, matrices = load_cell()
    probe = indices["validation"][:32]
    audit = idea082.initial_model_audit(
        protocol,
        store,
        matrices,
        indices["train"],
        probe,
        idea082.BASE_SEEDS[0],
    )
    assert audit["parameters"] == idea082.EXPECTED_PARAMETERS
    assert set(audit["max_logit_difference_vs_A0"].values()) == {0.0}
    assert all(audit["common_AST_state_equal_to_A0"].values())
    assert all(audit["real_shuffled_full_state_equal"].values())


def test_all_groups_respect_C1_cap_and_gradient_reachability() -> None:
    protocol, store, _, indices, matrices = load_cell()
    probe = indices["validation"][:32]
    for group in idea082.GROUPS:
        pipeline = f"{group}_real"
        idea082.idea068.idea051.reference.historical.set_seed(idea082.BASE_SEEDS[0])
        model = idea082.build_model(
            pipeline, protocol, store, matrices[pipeline], indices["train"]
        )
        gradient = idea082.gradient_audit(model, matrices[pipeline][probe])
        assert gradient["zero_init_age_output_bias_grad_norm"] > 0.0
        assert gradient["zero_init_age_hidden_weight_grad_norm"] == 0.0
        assert gradient["nonzero_output_probe_age_hidden_weight_grad_norm"] > 0.0
        idea082.idea068.idea051.reference.historical.set_seed(idea082.BASE_SEEDS[0])
        cap_model = idea082.build_model(
            pipeline, protocol, store, matrices[pipeline], indices["train"]
        )
        cap = idea082.cap_audit(cap_model, store, matrices[pipeline], probe)
        assert cap["validation_relative_perturbation_calls"] == len(probe)
        assert cap["validation_relative_perturbation_max"] <= idea082.idea071.CAP + 1e-5


def test_gate_and_descriptive_combo_are_locked() -> None:
    protocol = idea082.read_json(idea082.PROTOCOL_PATH)
    gate = protocol["gate"]
    assert gate["minimum_mean_seed_repeat_macro_f1_delta"] == 0.005
    assert gate["minimum_positive_base_seed_means"] == 2
    assert gate["minimum_positive_seed_repeats"] == 6
    assert gate["minimum_nonnegative_split_cells"] == 8
    assert gate["minimum_worst_split_cell_delta"] == -0.03
    assert gate["utility_minimum_per_base_seed_senior_recall_delta"] == -0.02
    assert gate["combo_incremental_contrasts_are_gate_free_descriptive_only"] is True


def test_formal_run_is_fail_closed_without_director_authorization() -> None:
    args = argparse.Namespace(
        stage="run",
        output_subdir="unused",
        device="cpu",
        resume=False,
        director_authorized=False,
    )
    with pytest.raises(RuntimeError, match="research director"):
        idea082.run(args)


def test_canonical_json_is_lf_only_and_semantically_stable() -> None:
    value = {"status": "GO", "unicode": "猫", "nested": {"x": 1}}
    payload = idea082.canonical_json_bytes(value)
    assert b"\r\n" not in payload
    assert payload.endswith(b"\n")
    assert json.loads(payload.decode("utf-8")) == value
