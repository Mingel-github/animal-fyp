from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_meowagenet_idea087_nested_hpo.py"
PROTOCOL = ROOT / "configs" / "protocol" / "meowagenet_idea087_nested_hpo_v1.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("idea087_runner", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


idea087 = load_runner()


def protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def repeat0_roles() -> pd.DataFrame:
    path = ROOT / protocol()["data"]["roles_path"]
    roles = pd.read_csv(path, dtype={"cat_id": str})
    return roles[roles["repeat"] == 0].copy()


def test_locked_grid_order_and_budget() -> None:
    value = protocol()
    configs = value["search"]["configurations"]
    assert [row["config_id"] for row in configs] == [f"q{i:02d}" for i in range(8)]
    assert configs[0] == {
        "config_id": "q00",
        "learning_rate": 0.006,
        "dropout": 0.44571035356880917,
        "weight_decay": 0.0,
    }
    assert len(configs) * 3 * 4 * 3 * 2 == value["budget"]["inner_fits"] == 576
    assert value["budget"]["outer_refit_fits_maximum"] == 72
    assert value["budget"]["total_fits_maximum"] == 648


def test_all_seed_materials_reproduce_locked_values() -> None:
    value = protocol()
    assert [idea087.derived_uint31(item) for item in value["inner_roles"]["split_seed_materials"]] == value["inner_roles"]["split_seeds"]
    assert [idea087.derived_uint31(item) for item in value["search"]["search_seed_materials"]] == value["search"]["search_base_seeds"]
    assert [idea087.derived_uint31(item) for item in value["refit"]["refit_seed_materials"]] == value["refit"]["refit_base_seeds"]
    assert idea087.full_search_seed(1314160575, 3, 2) == 1314190775
    assert idea087.full_refit_seed(480473385, 3) == 480503385
    audit = idea087.seed_audit(value)
    assert audit["search_unique"] == 24
    assert audit["refit_unique"] == 12
    assert audit["search_refit_overlap"] == []
    assert audit["historical_overlap"] == []


def test_inner_roles_are_stable_complete_grouped_and_stratified() -> None:
    value = protocol()
    roles = repeat0_roles()
    left = idea087.build_inner_roles(value, roles)
    right = idea087.build_inner_roles(value, roles.sample(frac=1.0, random_state=91))
    pd.testing.assert_frame_equal(left, right)
    assert len(left) == 999
    for outer_fold in range(4):
        outer_cell = roles[roles["outer_fold"] == outer_fold]
        outer_dev = set(
            outer_cell[outer_cell["role"].isin(("train", "validation"))]["cat_id"]
        )
        outer_test = set(outer_cell[outer_cell["role"] == "test"]["cat_id"])
        validation_seen: list[str] = []
        expected_validation_sizes = [28, 28, 27] if outer_fold < 3 else [28, 28, 28]
        for inner_fold in range(3):
            cell = left[
                (left["outer_fold"] == outer_fold)
                & (left["inner_fold"] == inner_fold)
            ]
            train = set(cell[cell["role"] == "train"]["cat_id"])
            validation = set(cell[cell["role"] == "validation"]["cat_id"])
            assert not train & validation
            assert not (train | validation) & outer_test
            assert train | validation == outer_dev
            assert len(validation) == expected_validation_sizes[inner_fold]
            assert set(cell[cell["role"] == "train"]["age_group"]) == set(idea087.LABEL_BY_AGE)
            assert set(cell[cell["role"] == "validation"]["age_group"]) == set(idea087.LABEL_BY_AGE)
            validation_seen.extend(sorted(validation))
        assert len(validation_seen) == len(set(validation_seen)) == len(outer_dev)
        assert set(validation_seen) == outer_dev


def test_half_up_refit_epoch_uses_six_one_based_epochs() -> None:
    assert idea087.half_up(2.5) == 3
    assert idea087.half_up(2.49) == 2
    assert idea087.refit_epoch([1, 2, 2, 3, 9, 10]) == 3
    assert idea087.refit_epoch([49, 50, 50, 50, 50, 50]) == 50
    with pytest.raises(RuntimeError):
        idea087.refit_epoch([1, 2, 3])
    with pytest.raises(RuntimeError):
        idea087.refit_epoch([0, 1, 2, 3, 4, 5])


def test_selection_uses_closed_f1_pool_then_brier_sd_accuracy_and_id() -> None:
    rows = []
    for index in range(8):
        rows.append(
            {
                "config_id": f"q{index:02d}",
                "mean_macro_f1": 0.7000,
                "mean_brier": 0.40,
                "macro_f1_sample_sd": 0.02,
                "mean_plain_accuracy": 0.70,
            }
        )
    rows[0]["mean_macro_f1"] = 0.7020
    rows[1]["mean_macro_f1"] = 0.7000 - 5.0e-13
    rows[1]["mean_brier"] = 0.39
    selected, candidates = idea087.select_configuration(rows, 0.002)
    assert "q01" in candidates
    assert selected["config_id"] == "q01"
    rows[1]["macro_f1_sample_sd"] = 0.03
    rows[2]["mean_brier"] = 0.39
    rows[2]["macro_f1_sample_sd"] = 0.02
    rows[2]["mean_plain_accuracy"] = 0.71
    selected, _ = idea087.select_configuration(rows, 0.002)
    assert selected["config_id"] == "q02"


def test_metric_bundle_uses_explicit_three_class_definitions() -> None:
    animals = pd.DataFrame(
        {
            "cat_id": ["a", "b", "c"],
            "true_label": [0, 1, 2],
            "prob_kitten": [0.8, 0.6, 0.1],
            "prob_adult": [0.1, 0.3, 0.2],
            "prob_senior": [0.1, 0.1, 0.7],
        }
    )
    result = idea087.metric_bundle(animals)
    assert result["plain_accuracy"] == pytest.approx(2 / 3)
    assert result["class_recall"] == {"kitten": 1.0, "adult": 0.0, "senior": 1.0}
    assert result["balanced_accuracy"] == pytest.approx(2 / 3)
    expected_brier = np.mean([0.06, 0.86, 0.14])
    assert result["brier"] == pytest.approx(expected_brier)


def test_call_class_weights_are_fitted_only_from_given_training_calls() -> None:
    labels = np.asarray([0, 0, 1, 2, 2, 2], dtype=np.int64)
    weights = idea087.idea068.class_weights(labels)
    np.testing.assert_allclose(weights, [1.0, 2.0, 2.0 / 3.0])
    changed = idea087.idea068.class_weights(np.asarray([0, 1, 1, 2], dtype=np.int64))
    assert not np.array_equal(weights, changed)


def test_model_parameters_initial_logits_and_common_state_are_matched() -> None:
    value = protocol()
    store, features, roles = idea087.load_inputs(value)
    inner = idea087.build_inner_roles(value, roles)
    indices = idea087.inner_indices(store, inner, 0, 0)
    config = idea087.configuration_map(value)["q00"]
    probe = indices["validation"][:16]
    logits = {}
    common = {}
    full = {}
    parameters = {}
    seed = value["search"]["search_base_seeds"][0]
    for pipeline in idea087.PIPELINES:
        idea087.idea068.idea051.reference.historical.set_seed(seed)
        model = idea087.build_model(pipeline, config, store, features, indices["train"])
        parameters[pipeline] = sum(parameter.numel() for parameter in model.parameters())
        common[pipeline] = idea087.state_digest(model, common_only=True)
        full[pipeline] = idea087.state_digest(model)
        model.eval()
        with idea087.torch.no_grad():
            logits[pipeline] = model(
                idea087.torch.from_numpy(store.frozen_embeddings[probe]),
                idea087.torch.from_numpy(features[probe]),
            ).numpy()
    assert parameters == idea087.EXPECTED_PARAMETERS
    assert len(set(common.values())) == 1
    assert full[idea087.PIPELINES[1]] == full[idea087.PIPELINES[2]]
    for pipeline in idea087.PIPELINES[1:]:
        np.testing.assert_array_equal(logits[pipeline], logits[idea087.PIPELINES[0]])


def test_q00_matches_original_idea076_training_values() -> None:
    value = protocol()
    q00 = idea087.configuration_map(value)["q00"]
    source = json.loads(
        (ROOT / value["dependencies"]["idea076_protocol_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert q00["learning_rate"] == source["fixed_training"]["learning_rate"]
    assert q00["dropout"] == source["fixed_training"]["dropout"]
    assert q00["weight_decay"] == 0.0


def test_outer_stage_is_fail_closed_without_authorization() -> None:
    args = argparse.Namespace(
        director_authorized=False,
        selection_lock_sha256=None,
        max_fits=None,
        device="cpu",
        output_subdir="unused",
        resume=False,
    )
    with pytest.raises(RuntimeError, match="director authorization"):
        idea087.run_outer(args)


def test_selection_lock_identity_validator_rejects_duplicates(tmp_path: Path, monkeypatch) -> None:
    value = protocol()
    selection_path = tmp_path / "selection.json"
    rows = [
        {"pipeline": pipeline, "outer_fold": fold}
        for pipeline in idea087.PIPELINES
        for fold in range(4)
    ]
    rows[-1] = dict(rows[0])
    selection = {
        "status": "complete_locked_before_outer_evaluation",
        "protocol_sha256": "p",
        "runner_sha256": "r",
        "tests_sha256": "t",
        "inner_roles_sha256": "i",
        "inner_fits": 576,
        "complete_locks": 12,
        "outer_test_accessed": False,
        "selection_locks": rows,
    }
    idea087.write_json(selection_path, selection)
    monkeypatch.setattr(
        idea087,
        "require_cpu_preflight",
        lambda run_root, protocol: {
            "status": "GO",
            "protocol_sha256": "p",
            "runner_sha256": "r",
            "tests_sha256": "t",
            "inner_roles_sha256": "i",
        },
    )
    monkeypatch.setattr(idea087, "sha256", lambda path: "p" if path == idea087.PROTOCOL_PATH else ("r" if path == Path(idea087.__file__).resolve() else "selection"))
    value["dependencies"]["tests_sha256"] = "t"
    with pytest.raises(RuntimeError, match="identities are incomplete"):
        idea087.validate_selection_lock(selection, selection_path, tmp_path, value)


def test_fixed_epoch_audit_does_not_claim_checkpoint_reload() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"checkpoint_reload_applicable": fixed_epochs is None' in source
    assert "reload_difference = None" in source
