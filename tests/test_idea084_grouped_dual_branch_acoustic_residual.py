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
RUNNER_PATH = (
    ROOT / "scripts" / "run_meowagenet_idea084_grouped_dual_branch_acoustic_residual.py"
)
SPEC = importlib.util.spec_from_file_location("idea084_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
idea084 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = idea084
SPEC.loader.exec_module(idea084)


def load_cell():
    protocol = idea084.read_json(idea084.PROTOCOL_PATH)
    store = idea084.idea068.idea051.reference.historical.idea019.load_feature_store()
    features, _ = idea084.load_features(protocol, store.call_ids)
    roles = pd.read_csv(ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    indices = idea084.role_cell_indices(store, roles, repeat=0, fold=0)
    return protocol, store, features, indices


def test_protocol_hash_locks_and_fixed_group_reconstruction() -> None:
    protocol = idea084.read_json(idea084.PROTOCOL_PATH)
    idea084.verify_protocol(protocol)
    derivation = protocol["feature_groups"]["hash_derivation"]
    generated_a, generated_b, ranking = idea084.fixed_random_groups(
        derivation["material"]
    )
    assert (generated_a, generated_b) == idea084.R1_GROUPS
    assert ranking == derivation["ranked_indices"]
    assert hashlib.sha256(derivation["material"].encode()).hexdigest() == derivation[
        "material_sha256"
    ]


def test_proposed_and_random_partitions_are_exclusive_exhaustive_and_distinct() -> None:
    for pair in (idea084.P1_GROUPS, idea084.R1_GROUPS):
        assert tuple(map(len, pair)) == (15, 5)
        assert not (set(pair[0]) & set(pair[1]))
        assert set(pair[0]) | set(pair[1]) == set(range(20))
    assert idea084.P1_GROUPS != idea084.R1_GROUPS
    for group in idea084.R1_GROUPS:
        assert set(group) & set(range(15))
        assert set(group) & set(range(15, 20))


def test_parameter_formula_and_fit_budget() -> None:
    protocol = idea084.read_json(idea084.PROTOCOL_PATH)
    model = protocol["model"]
    branch_parameters = (15 * 32 + 32 + 32 * 128 + 128) + (
        5 * 32 + 32 + 32 * 128 + 128
    )
    c1_parameters = 20 * 60 + 60 + 60 * 128 + 128
    assert branch_parameters == 9152
    assert c1_parameters == 9068
    assert model["dual_branch_age_parameters"] == branch_parameters
    assert model["c1_age_branch_parameters"] == c1_parameters
    assert idea084.EXPECTED_PARAMETERS[idea084.PIPELINES[2]] == 99_075 + 9_152
    assert idea084.EXPECTED_PARAMETERS[idea084.PIPELINES[1]] == 99_075 + 9_068
    assert len(idea084.PIPELINES) * len(idea084.BASE_SEEDS) * 3 * 4 == 144


def test_seed_derivation_has_no_prior_full_seed_collision() -> None:
    model = idea084.read_json(idea084.PROTOCOL_PATH)["model"]
    digest = hashlib.sha256(model["seed_derivation_text"].encode()).digest()
    candidates = [
        int.from_bytes(digest[offset : offset + 4], "big") % 10_000
        for offset in range(0, len(digest), 4)
    ]
    assert candidates == model["candidate_sequence"]
    assert tuple(candidates[:3]) == idea084.BASE_SEEDS
    current = {
        seed + 10_000 * repeat + 100 * fold
        for seed in idea084.BASE_SEEDS
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


def test_all_role_cells_are_cat_and_call_disjoint() -> None:
    protocol, store, _, _ = load_cell()
    roles = pd.read_csv(ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    count = 0
    for repeat in range(3):
        for fold in range(4):
            cell = roles[(roles["repeat"] == repeat) & (roles["outer_fold"] == fold)]
            cats = {
                role: set(cell[cell["role"] == role]["cat_id"].astype(str))
                for role in ("train", "validation", "test")
            }
            assert not (cats["train"] & cats["validation"])
            assert not (cats["train"] & cats["test"])
            assert not (cats["validation"] & cats["test"])
            indices = idea084.role_cell_indices(store, roles, repeat, fold)
            assert not np.intersect1d(indices["train"], indices["validation"]).size
            count += 1
    assert count == 12


def test_initial_logits_parameters_common_head_and_P1_R1_trainable_state() -> None:
    protocol, store, features, indices = load_cell()
    audit = idea084.initial_model_audit(
        protocol,
        store,
        features,
        indices["train"],
        indices["validation"][:32],
        idea084.BASE_SEEDS[0],
    )
    assert audit["parameters"] == idea084.EXPECTED_PARAMETERS
    assert set(audit["max_logit_difference_vs_A0"].values()) == {0.0}
    assert all(audit["common_AST_state_equal_to_A0"].values())
    assert audit["P1_R1_trainable_state_equal"] is True
    assert all(audit["zero_initialized_residual_outputs"].values())


def test_paired_C1_is_original_class_and_dual_preprocessing_is_train_only() -> None:
    protocol, store, features, indices = load_cell()
    idea084.idea068.idea051.reference.historical.set_seed(idea084.BASE_SEEDS[0])
    c1 = idea084.build_model(
        idea084.PIPELINES[1], protocol, store, features, indices["train"]
    )
    assert isinstance(c1, idea084.idea071.BoundedWideAdditiveClassifier)
    assert sum(parameter.numel() for parameter in c1.parameters()) == 108_143
    for pipeline in idea084.PIPELINES[2:]:
        idea084.idea068.idea051.reference.historical.set_seed(idea084.BASE_SEEDS[0])
        model = idea084.build_model(
            pipeline, protocol, store, features, indices["train"]
        )
        audit = idea084.preprocessing_audit(model, features, indices["train"])
        assert all(audit["training_role_statistics_equal_recomputed"].values())
        assert audit["validation_rows_used_for_statistics"] is False
        assert audit["test_rows_used_for_statistics"] is False


def test_both_dual_branches_are_gradient_reachable_and_share_one_total_cap() -> None:
    protocol, store, features, indices = load_cell()
    probe = indices["validation"][:32]
    for pipeline in idea084.PIPELINES[2:]:
        idea084.idea068.idea051.reference.historical.set_seed(idea084.BASE_SEEDS[0])
        model = idea084.build_model(
            pipeline, protocol, store, features, indices["train"]
        )
        gradient = idea084.gradient_audit(model, features[probe])
        assert gradient["zero_init_group_a_output_bias_grad_norm"] > 0.0
        assert gradient["zero_init_group_b_output_bias_grad_norm"] > 0.0
        assert gradient["zero_init_group_a_hidden_weight_grad_norm"] == 0.0
        assert gradient["zero_init_group_b_hidden_weight_grad_norm"] == 0.0
        assert gradient["nonzero_output_probe_group_a_hidden_weight_grad_norm"] > 0.0
        assert gradient["nonzero_output_probe_group_b_hidden_weight_grad_norm"] > 0.0
        idea084.idea068.idea051.reference.historical.set_seed(idea084.BASE_SEEDS[0])
        cap_model = idea084.build_model(
            pipeline, protocol, store, features, indices["train"]
        )
        cap = idea084.cap_audit(cap_model, store, features, probe)
        assert cap["validation_relative_perturbation_calls"] == len(probe)
        assert cap["validation_relative_perturbation_max"] <= idea084.idea071.CAP + 1e-5


def test_error_transition_profile_is_paired_and_directional() -> None:
    comparator = pd.DataFrame(
        {
            "base_seed": [1] * 4,
            "repeat": [0] * 4,
            "fold": [0, 0, 1, 1],
            "cat_id": ["a", "b", "c", "d"],
            "true_label": [0, 1, 2, 0],
            "predicted_label": [1, 1, 2, 0],
        }
    )
    candidate = comparator.copy()
    candidate["predicted_label"] = [0, 2, 2, 0]
    audit = idea084.paired_error_transitions(candidate, comparator)
    assert audit == {
        "paired_occurrences": 4,
        "corrected_errors": 1,
        "introduced_errors": 1,
        "net_corrections": 0,
        "unchanged_correct": 2,
        "unchanged_wrong": 0,
    }


def test_comparisons_gates_and_no_single_global_pass_are_locked() -> None:
    protocol = idea084.read_json(idea084.PROTOCOL_PATH)
    assert set(protocol["gate"]["gated_comparisons"]) == set(idea084.COMPARISONS)
    assert protocol["gate"]["no_single_global_pass_flag"] is True
    assert protocol["aggregation"]["report_all_axes_no_single_gate_summary"] is True
    assert protocol["gate"]["report_error_corrections_separately_not_as_a_gate"] is True
    assert protocol["gate"]["minimum_mean_seed_repeat_macro_f1_delta"] == 0.005
    assert protocol["gate"]["minimum_positive_seed_repeats"] == 6
    assert protocol["gate"]["minimum_nonnegative_split_cells"] == 8
    assert protocol["gate"]["minimum_worst_split_cell_delta"] == -0.03


def test_formal_run_is_fail_closed_without_director_authorization() -> None:
    args = argparse.Namespace(
        stage="run",
        output_subdir="unused",
        device="cpu",
        resume=False,
        director_authorized=False,
        max_cells=None,
    )
    with pytest.raises(RuntimeError, match="research director"):
        idea084.run(args)


def test_authorized_run_still_requires_matching_cpu_preflight() -> None:
    args = argparse.Namespace(
        stage="run",
        output_subdir="idea084_test_missing_preflight_do_not_create",
        device="cpu",
        resume=False,
        director_authorized=True,
        max_cells=1,
    )
    with pytest.raises(RuntimeError, match="CPU preflight GO"):
        idea084.run(args)


def test_canonical_json_is_lf_only_and_semantically_stable() -> None:
    value = {"status": "GO", "unicode": "猫", "nested": {"x": 1}}
    payload = idea084.canonical_json_bytes(value)
    assert b"\r\n" not in payload
    assert payload.endswith(b"\n")
    assert json.loads(payload.decode("utf-8")) == value
