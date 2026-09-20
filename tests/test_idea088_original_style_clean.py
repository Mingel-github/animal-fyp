from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_meowagenet_idea088_original_style_clean.py"
PROTOCOL = (
    ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea088_original_style_clean_v1.json"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("idea088_runner", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


idea088 = load_runner()


def protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_protocol_locks_clean_scope_models_training_and_budget() -> None:
    value = protocol()
    assert value["data"]["calls"] == 792
    assert value["data"]["cats"] == 111
    assert value["data"]["vggish_rows"] == 936
    assert value["data"]["vggish_features"] == 129
    assert value["data"]["expected_cat_class_counts"] == {
        "kitten": 15,
        "adult": 62,
        "senior": 34,
    }
    assert value["data"]["expected_call_class_counts"] == {
        "kitten": 134,
        "adult": 405,
        "senior": 253,
    }
    assert value["splits"]["seeds"] == [7270, 860, 5390, 5191, 5734]
    assert value["splits"]["forced_training_cat_ids"] == ["000A", "046A"]
    assert value["models"]["pipelines"] == list(idea088.PIPELINES)
    fixed = value["training"]
    assert fixed["optimizer"] == "Adamax"
    assert fixed["learning_rate"] == 0.003109800273709165
    assert fixed["epsilon"] == 1e-7
    assert fixed["beta1"] == 0.9 and fixed["beta2"] == 0.999
    assert fixed["dropout"] == 0.44571035356880917
    assert fixed["batch_size"] == 128
    assert fixed["maximum_epochs"] == 1500
    assert fixed["early_stopping_min_delta"] == 0.001
    assert fixed["early_stopping_patience"] == 30
    assert fixed["restore_best_weights"] is True
    assert fixed["gradient_clip"] is None
    assert fixed["automatic_mixed_precision"] is False
    assert value["budget"]["fits"] == 3 * 5 * 4 == 60
    assert value["reporting"]["claims_complete_oof"] is False


def test_runner_never_regenerates_roles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "StratifiedGroupKFold" not in source
    assert "StratifiedKFold" not in source
    assert "never regenerates splits" in source.lower()


def synthetic_roles(expected_cats: list[str]) -> dict:
    forced = ["000A", "046A"]
    others = [cat for cat in expected_cats if cat not in forced]
    cells = []
    for seed in [7270, 860, 5390, 5191, 5734]:
        for fold in range(4):
            test = sorted(cat for index, cat in enumerate(others) if index % 4 == fold)
            train = sorted(set(expected_cats) - set(test))
            cells.append(
                {
                    "cell_id": f"seed_{seed}_fold_{fold}",
                    "split_seed": seed,
                    "fold": fold,
                    "train_cat_ids": train,
                    "test_cat_ids": test,
                    "pre_swap_train_cat_ids": train,
                    "pre_swap_test_cat_ids": test,
                    "swaps": [],
                    "source_reference": "synthetic-test-only",
                }
            )
    return {
        "schema_version": 1.0,
        "protocol_id": "meowagenet-idea088-original-style-clean-v1",
        "source_prediction_rows": 937,
        "projection": {
            "excluded_cat_ids": ["049A"],
            "expected_calls": 792,
            "expected_cats": 111,
            "expected_vggish_rows": 936,
        },
        "vggish_input": {
            "feature_dimensions": 129,
            "feature_columns": [str(index) for index in range(128)] + ["mean_freq"],
        },
        "classifier_labels": {
            "clean_cat_counts": {"kitten": 15, "adult": 62, "senior": 34},
            "clean_call_counts": {"kitten": 134, "adult": 405, "senior": 253},
        },
        "split_seeds": [7270, 860, 5390, 5191, 5734],
        "forced_training_cat_ids": forced,
        "cells": cells,
    }


def test_role_validator_consumes_frozen_cat_lists_and_rejects_leakage() -> None:
    cats = ["000A", "046A"] + [f"T{index:03d}" for index in range(109)]
    manifest = synthetic_roles(sorted(cats))
    cells, audit = idea088.validate_roles_manifest(manifest, protocol(), set(cats))
    assert len(cells) == 20
    assert audit["complete_oof_claim_allowed"] is False
    assert all("000A" in cell["train_cat_ids"] for cell in cells)
    assert all("046A" in cell["train_cat_ids"] for cell in cells)
    broken = json.loads(json.dumps(manifest))
    broken["cells"][0]["test_cat_ids"].append("000A")
    broken["cells"][0]["test_cat_ids"].sort()
    with pytest.raises(RuntimeError, match="leakage|forced-training"):
        idea088.validate_roles_manifest(broken, protocol(), set(cats))


def test_frozen_independent_role_manifest_passes_and_hash_is_locked() -> None:
    value = protocol()
    role_path = ROOT / value["splits"]["roles_path"]
    assert role_path.is_file()
    assert idea088.sha256(role_path) == value["splits"]["roles_sha256"]
    _, _, ast = idea088.load_feature_inputs(value)
    cells, audit = idea088.validate_roles_manifest(
        idea088.read_json(role_path), value, set(ast["store"].cat_ids.astype(str))
    )
    assert len(cells) == 20
    assert audit["forced_training_cats_never_test"] is True


def test_class_weight_and_batch_denominator_match_locked_formula() -> None:
    full_train_labels = np.asarray([0, 0, 0, 0, 1, 2], dtype=np.int64)
    weights_np = idea088.class_weights(full_train_labels)
    np.testing.assert_allclose(weights_np, [0.5, 2.0, 2.0])
    logits = torch.tensor(
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [3.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    labels = torch.from_numpy(np.asarray([0, 0, 1, 2], dtype=np.int64))
    weights = torch.from_numpy(weights_np)
    actual = idea088.weighted_cross_entropy(logits, labels, weights)
    per_unit = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    expected = (per_unit * weights[labels]).sum() / len(labels)
    normalized_by_weight_sum = (per_unit * weights[labels]).sum() / weights[labels].sum()
    assert actual.item() == pytest.approx(expected.item())
    assert actual.item() != pytest.approx(normalized_by_weight_sum.item())


def test_keras_style_min_delta_patience_and_nonfinite_failure() -> None:
    best = float("inf")
    stale = 0
    improved, best, stale, stop = idea088.early_stopping_update(1.0, best, stale, 0.001, 2)
    assert improved and best == 1.0 and stale == 0 and not stop
    improved, best, stale, stop = idea088.early_stopping_update(0.9995, best, stale, 0.001, 2)
    assert not improved and best == 1.0 and stale == 1 and not stop
    improved, best, stale, stop = idea088.early_stopping_update(0.9994, best, stale, 0.001, 2)
    assert not improved and stale == 2 and stop
    callback = idea088.FailOnNonFiniteTrainingLoss()
    with pytest.raises(RuntimeError, match="non-finite"):
        callback.on_epoch_end(4, {"loss": float("nan")})


def test_explicit_restore_uses_recorded_best_even_without_patience_stop() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.weights = [np.asarray([-1.0])]

        def set_weights(self, values) -> None:
            self.weights = [np.asarray(value).copy() for value in values]

    class FakeCallback:
        best_weights = [np.asarray([7.0])]
        stopped_epoch = 0

    model = FakeModel()
    idea088.restore_keras_recorded_best(model, FakeCallback())
    np.testing.assert_array_equal(model.weights[0], [7.0])


def test_short_keras_budget_exit_explicitly_restores_recorded_best() -> None:
    value = protocol()
    rng = np.random.default_rng(88)
    features = rng.normal(size=(12, 129)).astype(np.float32)
    labels = np.tile(np.arange(3, dtype=np.int64), 4)
    sequence = idea088.LoggedVggishSequence(
        features,
        labels,
        np.arange(12, dtype=np.int64),
        batch_size=6,
        base_training_seed=1234,
    )
    model = idea088.formal.build_vggish_model(idea088.vggish_recipe(value), 88)
    callback = idea088.formal.tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=0.001,
        patience=30,
        mode="min",
        restore_best_weights=True,
    )
    history = model.fit(
        sequence,
        epochs=2,
        verbose=0,
        shuffle=False,
        callbacks=[idea088.FailOnNonFiniteTrainingLoss(), callback],
        workers=1,
        use_multiprocessing=False,
        max_queue_size=1,
    )
    assert len(history.history["loss"]) == 2
    assert callback.stopped_epoch == 0
    idea088.restore_keras_recorded_best(model, callback)
    for actual, expected in zip(model.get_weights(), callback.best_weights, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert len(sequence.orders) >= 2
    np.testing.assert_array_equal(np.sort(sequence.orders[0]), np.arange(12))


def test_epoch_orders_are_deterministic_full_and_paired() -> None:
    units = np.arange(593, dtype=np.int64)
    left = idea088.epoch_order(units, 1_007_270, 1)
    right = idea088.epoch_order(units, 1_007_270, 1)
    later = idea088.epoch_order(units, 1_007_270, 2)
    np.testing.assert_array_equal(left, right)
    assert not np.array_equal(left, later)
    np.testing.assert_array_equal(np.sort(left), units)
    assert len(np.unique(left)) == len(units)


def test_batch_size_audit_reveals_one_call_tail_without_dropping() -> None:
    assert idea088.batch_sizes_for_units(593, 128) == [128, 128, 128, 128, 81]
    assert idea088.batch_sizes_for_units(513, 128) == [128, 128, 128, 128, 1]
    assert sum(idea088.batch_sizes_for_units(513, 128)) == 513


def test_vggish_loader_uses_129_features_and_labels_match_ast() -> None:
    vggish, features, ast = idea088.load_feature_inputs(protocol())
    assert vggish["features"].shape == (936, 129)
    assert features.shape == (792, 20)
    assert len(np.unique(vggish["cat_ids"])) == 111
    assert len(np.unique(ast["store"].cat_ids.astype(str))) == 111
    assert ast["audit"]["vggish_feature_columns"][-1] == "mean_freq"
    assert ast["audit"]["vggish_auxiliary_columns_excluded"] == [
        "gender",
        "target",
        "cat_id",
        "age_group",
    ]
    assert ast["audit"]["cat_class_counts"] == [15, 62, 34]
    assert ast["audit"]["call_class_counts"] == [134, 405, 253]


def test_vggish_model_is_existing_keras_structure_with_129_inputs() -> None:
    value = protocol()
    model = idea088.formal.build_vggish_model(idea088.vggish_recipe(value), 7270)
    assert model.input_shape == (None, 129)
    assert model.output_shape == (None, 3)
    assert model.count_params() == 17_539


def test_A0_C1_common_initial_state_logits_and_training_order_pair() -> None:
    value = protocol()
    _, features, ast = idea088.load_feature_inputs(value)
    store = ast["store"]
    roles = idea088.read_json(ROOT / value["splits"]["roles_path"])
    cells, _ = idea088.validate_roles_manifest(
        roles, value, set(store.cat_ids.astype(str))
    )
    train = idea088.indices_for_cats(store.cat_ids, cells[0]["train_cat_ids"])
    seed = idea088.model_seed(cells[0]["split_seed"], cells[0]["fold"])
    models = {
        pipeline: idea088.build_ast_model(
            pipeline, value, store, features, train, seed
        )
        for pipeline in idea088.AST_PIPELINES
    }
    assert len(
        {idea088.torch_state_digest(model, common_only=True) for model in models.values()}
    ) == 1
    probe = train[:16]
    outputs = {}
    for pipeline, model in models.items():
        model.eval()
        with torch.no_grad():
            outputs[pipeline] = model(
                torch.from_numpy(store.frozen_embeddings[probe]),
                torch.from_numpy(features[probe]),
            ).numpy()
    np.testing.assert_array_equal(outputs["A0_ast_only"], outputs["C1_bounded_wide_additive"])
    assert idea088.training_seed(value, 7270, 0) == 1_007_270


def test_cat_probability_mean_and_native_unit_metrics_are_separate() -> None:
    units = pd.DataFrame(
        {
            "unit_index": [0, 1, 2, 3],
            "unit_id": ["a0", "a1", "b0", "c0"],
            "cat_id": ["a", "a", "b", "c"],
            "true_label": [0, 0, 1, 2],
            "prob_kitten": [0.9, 0.1, 0.2, 0.1],
            "prob_adult": [0.05, 0.8, 0.7, 0.1],
            "prob_senior": [0.05, 0.1, 0.1, 0.8],
        }
    )
    cats = idea088.units_to_cats(units)
    result = idea088.prediction_metrics(units, cats)
    assert result["public_cat_probability_mean"]["n"] == 3
    assert result["native_prediction_unit"]["n"] == 4
    assert set(result["public_cat_probability_mean"]["class_recall"]) == set(idea088.CLASS_NAMES)


def test_fit_stage_fails_closed_without_director_authorization() -> None:
    args = argparse.Namespace(director_authorized=False, max_fits=1)
    with pytest.raises(RuntimeError, match="director authorization"):
        idea088.ensure_training_authorized(args)


def test_locked_protocol_dependency_hashes_are_current() -> None:
    value = protocol()
    if value["status"] != "locked_before_cpu_preflight":
        pytest.skip("Protocol is waiting for final role/dependency freeze")
    idea088.verify_protocol(value)
