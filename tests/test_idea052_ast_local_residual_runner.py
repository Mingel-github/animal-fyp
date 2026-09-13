from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_idea052_ast_local_residual as runner  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea052_ast_local_residual_v1.json"
)
DIAGNOSTIC_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_idea052_local_residual_diagnostics_v1.json"
)


def load_inner_split() -> tuple[object, np.ndarray, np.ndarray]:
    store = runner.load_store()
    roles = pd.read_csv(runner.ROLES_PATH, dtype={"cat_id": str})
    indices = runner.reference.historical.fold_indices(
        store, roles, repeat=0, outer_fold=0, include_test=False
    )
    return store, indices["train"], indices["validation"]


def test_protocol_locks_three_pipeline_36_fit_initial_screen() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == "meowagenet-idea052-ast-local-residual-v1"
    assert tuple(protocol["pipelines"]) == runner.PIPELINES
    assert protocol["fixed_training"]["call_micro_batch_size"] == 8
    assert protocol["fixed_training"]["gradient_accumulation_steps"] == 4
    assert protocol["fixed_training"]["accumulation_window_calls"] == 32
    assert protocol["initial_evaluation"]["total_outer_fits"] == 36
    assert protocol["seed_expansion_gate"]["minimum_mean_macro_f1_gain"] == 0.005
    runner.verify_protocol(protocol)


def test_diagnostic_locates_temporal_variation_and_age_gradient() -> None:
    diagnostic = json.loads(DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
    assert diagnostic["status"] == "complete"
    assert diagnostic["dataset"]["calls"] == 792
    assert diagnostic["dataset"]["cats"] == 111
    assert diagnostic["dataset"]["temporal_tokens"] == 5842
    assert diagnostic["dataset"]["token_count_range"] == [1, 72]
    assert diagnostic["local_variation"]["token_rms_dispersion"]["mean"] == pytest.approx(
        0.44067948140170604
    )
    assert diagnostic["associations_across_792_unique_calls"][
        "ordinal_age_label_vs_token_dispersion"
    ]["rho"] == pytest.approx(0.24696213676716014)
    class_means = diagnostic["class_means"]
    assert (
        class_means["kitten"]["token_rms_dispersion"]
        < class_means["adult"]["token_rms_dispersion"]
        < class_means["senior"]["token_rms_dispersion"]
    )


def test_store_preserves_all_calls_tokens_and_singletons() -> None:
    store = runner.load_store()
    counts = np.asarray([len(indices) for indices in store.call_token_indices])
    assert store.global_embeddings.shape == (792, 768)
    assert store.temporal_tokens.shape == (5842, 768)
    assert len(np.unique(store.cat_ids)) == 111
    assert counts.sum() == 5842
    assert counts.min() == 1
    assert counts.max() == 72


def test_residual_models_start_from_identical_global_logits() -> None:
    protocol = runner.read_json(PROTOCOL_PATH)
    store, train_indices, validation_indices = load_inner_split()
    audit = runner.initialization_audit(
        protocol,
        store,
        train_indices,
        validation_indices,
        torch.device("cpu"),
        seed=17,
    )
    assert audit["shared_initial_state_equal"] is True
    assert audit["zero_initialized_residual_gates"] is True
    assert audit["R1_initial_max_logit_difference_from_R0"] == 0.0
    assert audit["R2_initial_max_logit_difference_from_R0"] == 0.0
    assert audit["single_token_salience_residual_max_abs"] == 0.0


def test_residual_branch_adds_only_128_trainable_scalars() -> None:
    protocol = runner.read_json(PROTOCOL_PATH)
    store, train_indices, _ = load_inner_split()
    models = []
    for pipeline in runner.PIPELINES:
        runner.reference.historical.set_seed(17)
        models.append(runner.build_model(pipeline, protocol, store, train_indices))
    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
    assert counts == [99075, 99203, 99203]
    assert models[0].residual_gate is None
    assert models[1].residual_gate.numel() == 128
    assert models[2].residual_gate.numel() == 128
    common_keys = set(models[0].state_dict())
    for model in models[1:]:
        state = model.state_dict()
        assert all(torch.equal(models[0].state_dict()[key], state[key]) for key in common_keys)


def test_training_loader_covers_every_call_and_token_once() -> None:
    store, train_indices, _ = load_inner_split()
    loader = runner.build_loader(store, train_indices, 8, True, seed=17)
    calls = []
    token_total = 0
    batch_sizes = []
    for batch in loader:
        calls.extend(batch["call_indices"].tolist())
        token_total += int(batch["token_counts"].sum())
        batch_sizes.append(len(batch["labels"]))
    assert sorted(calls) == sorted(train_indices.tolist())
    assert len(calls) == len(set(calls))
    assert token_total == sum(len(store.call_token_indices[index]) for index in train_indices)
    assert all(2 <= size <= 8 for size in batch_sizes)


def test_global_weighted_loss_keeps_fixed_denominator() -> None:
    source = inspect.getsource(runner.train_one_epoch)
    loss_source = inspect.getsource(runner.reference.global_weighted_micro_loss)
    assert "global_weighted_micro_loss" in source
    assert 'values["accumulation_window_calls"]' in source
    assert "call_weights.sum" not in loss_source
    assert "float(accumulation_window_calls)" in loss_source


def test_runner_separates_inner_smoke_and_outer_evaluation() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert 'choices=("smoke", "evaluate")' in source
    assert "include_test=False" in source
    assert "include_test=True" in source
    assert "locked_for_idea052_initial_evaluation" in source
    assert "paired_cat_bootstrap" in source


def test_temporal_pooling_and_best_checkpoint_predictions_are_deterministic() -> None:
    pooling_source = inspect.getsource(runner.GlobalLocalResidualClassifier.hidden_and_residual)
    fit_source = inspect.getsource(runner.fit_inner)
    assert "index_add_" not in pooling_source
    assert "token_groups" in pooling_source
    assert "model.load_state_dict(best_state)" in fit_source
    assert "_, best_calls = predict_calls(model, validation_loader, store, device)" in fit_source
    assert "outer evaluation remains locked" in Path(runner.__file__).read_text(
        encoding="utf-8"
    )
