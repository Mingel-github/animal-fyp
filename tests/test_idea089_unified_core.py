from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_idea089_unified_core as runner


def protocol() -> dict:
    return runner.read_json(runner.PROTOCOL_PATH)


def synthetic_store(seed: int = 19, rows: int = 48) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        frozen_embeddings=rng.normal(size=(rows, 768)).astype(np.float32),
        labels=np.tile(np.arange(3, dtype=np.int64), rows // 3),
        cat_ids=np.asarray([f"cat-{index:03d}" for index in range(rows)]),
        call_ids=np.asarray([f"call-{index:03d}" for index in range(rows)]),
    )


def trainable_keras_parameters(model: tf.keras.Model) -> int:
    return int(sum(int(np.prod(weight.shape)) for weight in model.trainable_weights))


def test_protocol_matrix_budget_and_comparisons_are_frozen() -> None:
    value = protocol()
    assert tuple(value["models"]["pipelines"]) == runner.PIPELINES
    assert value["models"]["trainable_parameter_counts"] == runner.EXPECTED_TRAINABLE_PARAMETERS
    assert value["models"]["total_state_parameter_counts"] == runner.EXPECTED_TOTAL_STATE_PARAMETERS
    assert value["budget"] == {
        "pipelines": 6,
        "repeat_fold_seed_cells_per_pipeline": 36,
        "selection_fits": 216,
        "outer_fits": 216,
        "physical_fits": 432,
        "complete_oof_sets_per_pipeline": 9,
    }
    assert value["evaluation"]["comparisons"] == list(runner.COMPARISONS)
    assert len(runner.COMPARISONS) == 9


def test_full_seed_bank_has_36_unique_cells() -> None:
    value = protocol()
    seeds = {
        runner.full_seed(base, repeat, fold)
        for base in value["training"]["model_seeds"]
        for repeat in value["data"]["repeats"]
        for fold in value["data"]["folds"]
    }
    assert len(seeds) == 36


def test_run_root_is_restricted_to_new_idea089_directory() -> None:
    assert "idea089" in runner.resolve_run_root("meowagenet_idea089_test").name
    with pytest.raises(ValueError, match="IDEA-089"):
        runner.resolve_run_root("meowagenet_idea088_original_style_clean_v1")


def test_initial_six_scope_is_exact_and_full_scope_is_explicit() -> None:
    value = protocol()
    args = SimpleNamespace(
        authorization_scope="initial-six-selection",
        pipeline=None,
        repeat=None,
        fold=None,
        base_seed=None,
        max_fits=None,
    )
    cells, pipelines, limit = runner.authorized_workset(args, value, "selection")
    assert cells == [(0, 0, 17)]
    assert pipelines == list(runner.PIPELINES)
    assert limit == 6
    args.authorization_scope = "full-selection"
    cells, pipelines, limit = runner.authorized_workset(args, value, "selection")
    assert len(cells) == 36
    assert pipelines == list(runner.PIPELINES)
    assert limit is None
    args.authorization_scope = None
    with pytest.raises(RuntimeError, match="authorization-scope"):
        runner.authorized_workset(args, value, "selection")


def test_ast_pairing_zero_columns_and_gradients() -> None:
    value = protocol()
    store = synthetic_store()
    rng = np.random.default_rng(29)
    acoustic = rng.normal(size=(48, 20)).astype(np.float32)
    indices = np.arange(36, dtype=np.int64)
    probe = np.arange(16, dtype=np.int64)
    models = {}
    logits = {}
    for pipeline in runner.AST_PIPELINES:
        runner.set_seed(17)
        model = runner.build_ast_model(pipeline, value, store, acoustic, indices).eval()
        models[pipeline] = model
        with torch.no_grad():
            logits[pipeline] = model(
                torch.from_numpy(store.frozen_embeddings[probe]),
                torch.from_numpy(acoustic[probe]),
            )
        assert sum(parameter.numel() for parameter in model.parameters()) == runner.EXPECTED_TRAINABLE_PARAMETERS[pipeline]
    a0 = models["A0_ast_only"]
    d0 = models["D0_direct_concat"]
    assert torch.equal(d0.concat_linear.weight[:, :768], a0.ast_linear.weight)
    assert torch.count_nonzero(d0.concat_linear.weight[:, 768:]).item() == 0
    for pipeline in runner.AST_PIPELINES[1:]:
        assert torch.max(torch.abs(logits[pipeline] - logits["A0_ast_only"])).item() <= 1e-6
    common_keys = set(a0.state_dict())
    for pipeline in ("U1_wide_unbounded_additive", "C1_bounded_wide_additive"):
        state = models[pipeline].state_dict()
        assert common_keys.issubset(state)
        assert all(torch.equal(a0.state_dict()[key], state[key]) for key in common_keys)
    assert runner.torch_state_digest(models["U1_wide_unbounded_additive"]) == runner.torch_state_digest(models["C1_bounded_wide_additive"])

    for pipeline in ("D0_direct_concat", "U1_wide_unbounded_additive", "C1_bounded_wide_additive"):
        runner.set_seed(17)
        model = runner.build_ast_model(pipeline, value, store, acoustic, indices)
        loss = torch.nn.functional.cross_entropy(
            model(
                torch.from_numpy(store.frozen_embeddings[probe]),
                torch.from_numpy(acoustic[probe]),
            ),
            torch.from_numpy(store.labels[probe]),
        )
        loss.backward()
        gradient = (
            model.concat_linear.weight.grad[:, 768:]
            if pipeline == "D0_direct_concat"
            else model.age_output.weight.grad
        )
        assert torch.isfinite(gradient).all()
        assert torch.max(torch.abs(gradient)).item() > 0.0


def test_vgg_pairing_parameter_scopes_and_f0_gradient() -> None:
    value = protocol()
    runner.set_seed(17)
    vgg128 = runner.build_vgg_model("VGG128_no_f0", 17, value)
    runner.set_seed(17)
    vgg129 = runner.build_vgg_model("VGG129_with_f0", 17, value)
    assert trainable_keras_parameters(vgg128) == 17_155
    assert trainable_keras_parameters(vgg129) == 17_283
    assert vgg128.count_params() == 17_411
    assert vgg129.count_params() == 17_539
    rng = np.random.default_rng(31)
    x128 = rng.normal(size=(18, 128)).astype(np.float32)
    f0 = rng.normal(size=(18, 1)).astype(np.float32)
    x129 = np.concatenate((x128, f0), axis=1)
    left = vgg128(x128, training=False).numpy()
    right = vgg129(x129, training=False).numpy()
    assert np.max(np.abs(left - right)) <= 1e-7
    assert np.count_nonzero(vgg129.get_layer("hidden").get_weights()[0][128]) == 0
    with tf.GradientTape() as tape:
        probabilities = vgg129(x129, training=True)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                tf.convert_to_tensor(np.tile(np.arange(3), 6)), probabilities
            )
        )
    gradients = tape.gradient(loss, vgg129.trainable_weights)
    assert np.max(np.abs(gradients[0].numpy()[128])) > 0.0


def test_strict_improvement_uses_unified_minimum_delta() -> None:
    assert runner.strict_improvement(0.8, float("inf"), 1e-6)
    assert runner.strict_improvement(0.799998, 0.8, 1e-6)
    assert not runner.strict_improvement(0.7999995, 0.8, 1e-6)
    assert not runner.strict_improvement(0.8, 0.8, 1e-6)


def test_stratified_bootstrap_is_deterministic_and_shared() -> None:
    labels = np.repeat(np.arange(3, dtype=np.int64), 4)
    rng = np.random.default_rng(37)
    probabilities = rng.random((9, 12, 3))
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    indices_a = runner.stratified_bootstrap_indices(labels, 40, 20260920)
    indices_b = runner.stratified_bootstrap_indices(labels, 40, 20260920)
    assert np.array_equal(indices_a, indices_b)
    assert all(np.bincount(labels[row], minlength=3).tolist() == [4, 4, 4] for row in indices_a)
    metrics_a = runner.bootstrap_model_metrics(labels, probabilities, indices_a, chunk_size=11)
    metrics_b = runner.bootstrap_model_metrics(labels, probabilities, indices_b, chunk_size=13)
    assert set(metrics_a) == set(metrics_b)
    for name in metrics_a:
        assert np.allclose(metrics_a[name], metrics_b[name], atol=1e-15, rtol=0.0)


def test_real_role_bank_forms_nine_complete_oof_sets_per_pipeline() -> None:
    value = protocol()
    store, acoustic, roles, vggish = runner.load_inputs(value)
    del acoustic
    audits = runner.validate_role_bank(store, roles, vggish, value)
    assert len(audits) == 12
    for repeat in value["data"]["repeats"]:
        seen = []
        for fold in value["data"]["folds"]:
            mapping = runner.role_mapping(roles, repeat, fold)
            seen.extend(cat for cat, role in mapping.items() if role == "test")
        assert len(seen) == 111
        assert len(set(seen)) == 111


def test_outer_fails_closed_without_authorization() -> None:
    args = SimpleNamespace(director_authorized=False, max_fits=None)
    with pytest.raises(RuntimeError, match="director authorization"):
        runner.run_outer(args)


def test_outer_source_locks_checkpoint_before_test_and_rejects_partial_directory() -> None:
    source = inspect.getsource(runner.run_outer)
    assert source.index("write_json(checkpoint_path, checkpoint)") < source.index(
        "test_animals, test_units = predict_dispatch"
    )
    assert "partial outer directory requires audit and may not rerun" in source
    assert '"test_prediction_calls": 1' in source


def test_selection_source_never_predicts_test_role() -> None:
    source = inspect.getsource(runner.run_selection)
    assert 'indices["validation"]' in source
    assert 'indices["test"]' not in source
    assert '"outer_test_accessed": False' in source
