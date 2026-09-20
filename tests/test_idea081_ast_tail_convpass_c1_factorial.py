from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_meowagenet_idea081_ast_tail_convpass_c1_factorial.py"
SPEC = importlib.util.spec_from_file_location("idea081_runner", RUNNER)
assert SPEC is not None and SPEC.loader is not None
idea081 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = idea081
SPEC.loader.exec_module(idea081)


def protocol() -> dict:
    return idea081.read_json(idea081.PROTOCOL_PATH)


def test_protocol_locks_factorial_scope_and_budget() -> None:
    value = protocol()
    idea081.verify_protocol(value)
    assert tuple(value["model"]["pipelines"]) == idea081.PIPELINES
    assert value["model"]["total_fits"] == 144
    assert value["model"]["fits_per_pipeline"] == 36
    assert value["model"]["outer_test_predictions"] is False
    assert value["relationship"]["idea077_frozen"] is True
    assert value["relationship"]["idea079_frozen"] is True


def test_new_seed_bank_derives_from_digest_and_excludes_idea080() -> None:
    model = protocol()["model"]
    digest = hashlib.sha256(model["seed_derivation_text"].encode("utf-8")).digest()
    candidates = [
        int.from_bytes(digest[offset : offset + 4], "big") % 10_000
        for offset in range(0, len(digest), 4)
    ]
    assert candidates[:3] == list(idea081.BASE_SEEDS)
    derived = idea081._all_full_seeds(idea081.BASE_SEEDS)
    idea080 = idea081._all_full_seeds([59, 7031, 1855])
    excluded = idea081._all_full_seeds(model["excluded_base_seeds_IDEA065_through_IDEA080"])
    assert len(derived) == 36
    assert derived.isdisjoint(idea080)
    assert derived.isdisjoint(excluded)


def test_convpass_matches_official_parameterization_and_zero_up() -> None:
    module = idea081.ConvPass2D()
    assert idea081.trainable_parameter_count(module) == idea081.CONVPASS_PARAMETERS_EACH
    assert module.adapter_down.weight.shape == (8, 768)
    assert module.adapter_conv.weight.shape == (8, 8, 3, 3)
    assert module.adapter_conv.groups == 1
    assert module.adapter_up.weight.shape == (768, 8)
    torch.testing.assert_close(
        module.adapter_conv.weight[:, :, 1, 1], torch.eye(8), rtol=0, atol=0
    )
    off_center = module.adapter_conv.weight.detach().clone()
    off_center[:, :, 1, 1] = 0
    assert torch.count_nonzero(off_center) == 0
    assert torch.count_nonzero(module.adapter_up.weight) == 0
    assert torch.count_nonzero(module.adapter_up.bias) == 0


def test_zero_up_is_exact_identity_effect_for_all_146_tokens() -> None:
    tokens = torch.randn((2, 146, 768), generator=torch.Generator().manual_seed(81))
    output = idea081.ConvPass2D().eval()(tokens)
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)


def test_dense_3x3_patch_support_and_special_tokens_are_independent() -> None:
    audit = idea081.topology_audit()
    assert audit["dense_conv_weight_shape"] == [8, 8, 3, 3]
    assert audit["patch_impulse_support_flat_indices"] == sorted(
        [frequency * 12 + time for frequency in (4, 5, 6) for time in (4, 5, 6)]
    )
    assert audit["special_zero_token_nonzero_outputs"] == [0]
    assert audit["patch_flatten_order"] == "frequency-major_then_time-minor"


def test_full_convpass_is_parallel_to_both_block12_sublayers() -> None:
    value = protocol()["convpass"]
    assert value["target_block_one_based"] == 12
    assert value["parallel_to"] == ["MSA", "MLP"]
    assert value["modules_per_block"] == 2
    assert value["special_token_policy"] == (
        "CLS and distillation tokens are each processed as an independent 1x1 grid through the shared dense 3x3 convolution"
    )


def test_parameter_and_mac_counts_are_locked() -> None:
    assert idea081.CONVPASS_PARAMETERS_EACH == 6152 + 584 + 6912
    assert idea081.CONVPASS_PARAMETERS == 2 * idea081.CONVPASS_PARAMETERS_EACH
    assert idea081.CONVPASS_MACS_EACH == (
        146 * 768 * 8 + 144 * 8 * 8 * 9 + 2 * 8 * 8 * 9 + 146 * 8 * 768
    )
    assert idea081.EXPECTED_TRAINABLE_PARAMETERS == {
        "A0_frozen_tail": 99075,
        "C1_bounded_age": 108143,
        "V1_tail_full_convpass": 126371,
        "CV1_C1_plus_tail_full_convpass": 135439,
    }


def test_corresponding_factor_states_and_initial_logits_match_on_real_cache() -> None:
    value = protocol()
    store, _ = idea081.load_token_store(value)
    ages = idea081.load_age_features(value, store.call_ids)
    roles = idea081.pd.read_csv(
        idea081.REPO_ROOT / value["data"]["roles_path"], dtype={"cat_id": str}
    )
    indices = idea081.base.fold_indices(store, roles, 0, 0)
    audit = idea081.initialization_audit(
        value, store, ages, indices["train"], indices["validation"], idea081.BASE_SEEDS[0]
    )
    assert audit["trainable_parameters"] == idea081.EXPECTED_TRAINABLE_PARAMETERS
    assert audit["common_head_state_equal_all_four"] is True
    assert audit["C1_CV1_age_state_equal"] is True
    assert audit["V1_CV1_convpass_state_equal"] is True
    assert audit["A0_V1_exact_initial_logits"] is True
    assert audit["C1_CV1_exact_initial_logits"] is True
    assert all(value == 0.0 for value in audit["max_initial_logit_difference_vs_A0"].values())


def test_zero_up_gradient_schedule_is_reachable() -> None:
    audit = idea081.gradient_reachability_audit(idea081.BASE_SEEDS[0])
    assert audit["at_zero_up"]["up"] > 0
    assert audit["at_zero_up"]["down"] == 0
    assert audit["at_zero_up"]["conv"] == 0
    assert min(audit["after_fixed_nonzero_up_probe"].values()) > 0


def test_gate_locks_probability_senior_and_split_checks_for_all_contrasts() -> None:
    gates = protocol()["gate"]
    for name in ("V1_minus_A0", "CV1_minus_C1", "interaction"):
        gate = gates[name]
        assert gate["minimum_nonnegative_split_cells"] == 8
        assert gate["minimum_worst_split_cell_delta"] == -0.03
        assert gate["cross_entropy_must_be_nonworse"] is True
        assert gate["brier_must_be_nonworse"] is True
        assert gate["minimum_per_base_seed_senior_recall_delta"] == -0.02


def test_shared_cache_is_read_only_and_outer_test_is_forbidden() -> None:
    value = protocol()
    assert value["shared_cache"]["reuse_mode"] == "read_only_single_shared_cache_no_duplicate_extraction"
    assert value["shared_cache"]["required_tensor_shape"] == [843, 146, 768]
    assert value["execution"]["current_gpu_allowed"] is False
    assert value["relationship"]["outer_test_accessed"] is False
