from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_idea077_ast_last4_layer_mix.py"
SPEC = importlib.util.spec_from_file_location("idea077_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
idea077 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = idea077
SPEC.loader.exec_module(idea077)


def load_inputs():
    protocol = idea077.read_json(idea077.PROTOCOL_PATH)
    store, cache_audit = idea077.load_store(protocol)
    roles = pd.read_csv(
        ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str}
    )
    indices = idea077.fold_indices(store, roles, 0, 0)
    return protocol, store, cache_audit, roles, indices


def test_protocol_hashes_budget_layers_and_seeds_are_locked() -> None:
    protocol = idea077.read_json(idea077.PROTOCOL_PATH)
    idea077.verify_protocol(protocol)
    assert protocol["representation"]["layers_one_based"] == [9, 10, 11, 12]
    assert protocol["model"]["total_fits"] == 108
    assert protocol["model"]["primary_A0_L1_fits"] == 72
    assert protocol["model"]["outer_test_predictions"] is False
    derived = {
        idea077.full_seed(base, repeat, fold)
        for base in idea077.BASE_SEEDS
        for repeat in range(3)
        for fold in range(4)
    }
    assert len(derived) == 36


def test_existing_cache_has_locked_geometry_and_exact_final_anchor() -> None:
    protocol, store, cache_audit, _, _ = load_inputs()
    assert store.frozen_embeddings.shape == (792, 768)
    assert store.last4_embeddings.shape == (792, 4, 768)
    assert np.array_equal(store.last4_embeddings[:, -1], store.frozen_embeddings)
    tolerance = protocol["representation"]["cached_last_layer_audit_tolerance"]
    assert cache_audit["mean_absolute_difference"] <= tolerance["mean_absolute_maximum"]
    assert cache_audit["maximum_absolute_difference"] <= tolerance["absolute_maximum"]
    assert protocol["representation"]["new_hidden_state_cache_required"] is False


def test_zero_gate_makes_L1_exactly_equal_to_A0() -> None:
    protocol, store, _, _, indices = load_inputs()
    probe = indices["validation"][:32]
    audit = idea077.initialization_audit(
        protocol, store, indices["train"], probe, idea077.BASE_SEEDS[0]
    )
    assert audit["trainable_parameters"] == idea077.EXPECTED_PARAMETERS
    assert audit["shared_head_state_equal"] == {
        "M0_uniform_last4": True,
        "L1_global_layermix": True,
    }
    assert audit["L1_minus_A0_max_initial_logit_difference"] == 0.0
    assert audit["L1_minus_A0_initial_loss_difference"] == 0.0
    assert audit["initial_L1"]["signed_residual_gate"] == 0.0
    assert audit["initial_L1"]["layer_weights_one_based_9_to_12"] == [0.25] * 4


def test_fixed_uniform_and_learned_formulas_are_exact() -> None:
    rng = np.random.default_rng(77)
    final = torch.from_numpy(rng.normal(size=(5, 768)).astype(np.float32))
    last4 = torch.from_numpy(rng.normal(size=(5, 4, 768)).astype(np.float32))
    last4[:, -1] = final
    mean = np.zeros(768, dtype=np.float32)
    scale = np.ones(768, dtype=np.float32)
    m0 = idea077.LayerMixClassifier("M0_uniform_last4", mean, scale, 0.0)
    l1 = idea077.LayerMixClassifier("L1_global_layermix", mean, scale, 0.0)
    torch.testing.assert_close(m0.representation(final, last4), last4.mean(dim=1))
    torch.testing.assert_close(l1.representation(final, last4), final, rtol=0, atol=0)
    assert l1.residual_gate_raw is not None
    with torch.no_grad():
        l1.residual_gate_raw.fill_(0.5)
    expected = final + torch.tanh(torch.tensor(0.5)) * (last4.mean(dim=1) - final)
    torch.testing.assert_close(l1.representation(final, last4), expected)


def test_gate_then_layer_logits_are_gradient_reachable() -> None:
    protocol, store, _, _, indices = load_inputs()
    audit = idea077.gradient_reachability_audit(
        protocol, store, indices["train"], idea077.BASE_SEEDS[0]
    )
    assert audit["gate_gradient_at_zero"] != 0.0
    assert audit["layer_logits_max_gradient_at_zero"] == 0.0
    assert audit["layer_logits_max_gradient_after_gate_probe"] > 0.0


def test_roles_are_cat_disjoint_and_test_is_not_selected() -> None:
    protocol, store, _, roles, _ = load_inputs()
    assert idea077.validate_roles(protocol, store, roles) == 12
    for repeat in range(3):
        for fold in range(4):
            indices = idea077.fold_indices(store, roles, repeat, fold)
            assert set(indices) == {"train", "validation"}
            train_cats = set(store.cat_ids[indices["train"]])
            validation_cats = set(store.cat_ids[indices["validation"]])
            assert train_cats.isdisjoint(validation_cats)


def test_primary_gate_is_not_changed_by_M0() -> None:
    gate = idea077.read_json(idea077.PROTOCOL_PATH)["gate"]
    assert gate["minimum_mean_seed_repeat_L1_minus_A0"] == 0.005
    assert gate["minimum_positive_base_seeds"] == 3
    assert gate["minimum_positive_seed_repeats"] == 6
    assert gate["minimum_nonnegative_split_cells"] == 8
    assert gate["minimum_worst_split_cell_delta"] == -0.03
    assert gate["minimum_per_base_seed_senior_recall_delta"] == -0.02
    assert all("M0" not in key for key in gate)
