from __future__ import annotations

import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_idea075_dog_C1_age_sensitive_AST as idea075  # noqa: E402


def synthetic_store(dogs: int = 9, units_per_dog: int = 3) -> idea075.DogStore:
    rng = np.random.default_rng(75)
    count = dogs * units_per_dog
    dog_ids = np.repeat([f"dog_{index:03d}" for index in range(dogs)], units_per_dog)
    # Deliberately give every dog more than one label: the dataset must not collapse them.
    labels = np.tile(np.arange(units_per_dog), dogs) % len(idea075.CLASS_NAMES)
    features = rng.normal(size=(count, len(idea075.FEATURE_NAMES))).astype(np.float32)
    features[0, 0] = np.nan
    return idea075.DogStore(
        embeddings=rng.normal(size=(count, 768)).astype(np.float32),
        age_features=features,
        recording_ids=np.asarray([f"unit_{index:04d}" for index in range(count)]),
        dog_ids=np.asarray(dog_ids),
        labels=labels.astype(np.int64),
    )


def test_dog_dataset_preserves_multilabel_units_and_complete_dogs() -> None:
    store = synthetic_store()
    indices = np.arange(len(store.labels))
    dataset = idea075.DogSetDataset(store, indices)
    covered = []
    for item_index in range(len(dataset)):
        item = dataset[item_index]
        assert len(torch.unique(item["labels"])) > 1
        assert set(store.dog_ids[item["indices"].numpy()]) == {item["dog_id"]}
        covered.extend(item["indices"].tolist())
    assert sorted(covered) == indices.tolist()


def test_no_singleton_batches_have_at_most_four_complete_dogs() -> None:
    store = synthetic_store(dogs=9)
    dataset = idea075.DogSetDataset(store, np.arange(len(store.labels)))
    loader = idea075.build_dog_loader(dataset, batch_size=4, shuffle=True, seed=8075)
    batches = list(loader)
    assert len(loader) == len(batches)
    assert [len(batch["dog_ids"]) for batch in batches] == [4, 3, 2]
    assert all(2 <= len(batch["dog_ids"]) <= 4 for batch in batches)
    assert len({dog for batch in batches for dog in batch["dog_ids"]}) == 9
    five = idea075.NoSingletonDogBatchSampler(5, 4, False, 1)
    five_batches = list(five)
    assert len(five) == len(five_batches) == 2
    assert [len(batch) for batch in five_batches] == [3, 2]


def test_A0_U1_C1_initialization_and_parameter_counts_are_locked() -> None:
    store = synthetic_store(dogs=12, units_per_dog=5)
    fit_indices = np.arange(50)
    audit = idea075.initialization_audit(
        store,
        fit_indices,
        np.arange(50, 60),
        dropout=0.44571035356880917,
        seed=8075,
    )
    assert audit["trainable_parameters"] == idea075.EXPECTED_PARAMETERS
    assert audit["shared_state_equal"]
    assert audit["U1_C1_full_state_equal"]
    assert audit["max_logit_difference_vs_A0"] == {
        idea075.PIPELINES[1]: 0.0,
        idea075.PIPELINES[2]: 0.0,
    }


def test_C1_bound_and_detached_rms_anchor_hold() -> None:
    store = synthetic_store(dogs=4, units_per_dog=4)
    model = idea075.build_model(idea075.PIPELINES[2], store, np.arange(12), dropout=0.0)
    assert model.age_output is not None
    with torch.no_grad():
        model.age_output.weight.fill_(1000.0)
        model.age_output.bias.fill_(1000.0)
    model.eval()
    model.reset_perturbation_audit()
    model(torch.from_numpy(store.embeddings[12:]), torch.from_numpy(store.age_features[12:]))
    assert model.audit()["maximum_dimension_budget_violations"] == 0

    ast = torch.from_numpy(store.embeddings[:2]).requires_grad_(True)
    age = torch.from_numpy(store.age_features[:2])
    normalized = (ast - model.ast_mean) / model.ast_scale
    hidden = model.relu(model.ast_linear(normalized))
    anchor = idea075.hidden_rms(hidden)
    assert not anchor.requires_grad
    model(ast, age).sum().backward()
    assert ast.grad is not None
    half_anchor = idea075.hidden_rms(torch.zeros((2, 128), dtype=torch.float16, requires_grad=True))
    assert half_anchor.dtype == torch.float16
    assert torch.isfinite(half_anchor).all()
    assert (half_anchor > 0).all()
    assert not half_anchor.requires_grad


def test_validation_ce_is_mean_of_unit_ce_within_dog_then_equal_dog_mean() -> None:
    # dog_a has three units and dog_b one. A pooled unit CE would weight dog_a 3x.
    probabilities = np.asarray(
        [
            [0.9, 0.025, 0.025, 0.025, 0.025],
            [0.8, 0.05, 0.05, 0.05, 0.05],
            [0.7, 0.075, 0.075, 0.075, 0.075],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ]
    )
    frame = pd.DataFrame(probabilities, columns=idea075.PROBABILITY_COLUMNS)
    frame["dog_id"] = ["dog_a", "dog_a", "dog_a", "dog_b"]
    frame["true_label"] = [0, 0, 0, 0]
    expected = 0.5 * (np.mean(-np.log([0.9, 0.8, 0.7])) - np.log(0.2))
    actual = idea075.dog_equal_unit_cross_entropy(frame)
    pooled = float(np.mean(-np.log([0.9, 0.8, 0.7, 0.2])))
    assert np.isclose(actual, expected)
    assert not np.isclose(actual, pooled)


def test_role_cell_allows_one_dog_to_have_multiple_age_labels() -> None:
    rows = []
    for role_index, role in enumerate(("fit", "validation", "test")):
        for class_index in range(5):
            rows.append(
                {
                    "dog_id": f"{role}_dog_0",
                    "age_group": idea075.CLASS_NAMES[class_index],
                    "role": role,
                }
            )
    frame = pd.DataFrame(rows)
    idea075.validate_role_cell(frame)


def test_reconstructed_locked_roles_have_canonical_LF_hash() -> None:
    protocol = {
        "data": {
            "manifest_path": "runs/idea067_external_ast_representation_v1/canine/manifest.csv",
            "frozen_embedding_path": "runs/idea067_external_ast_representation_v1/canine/ast_standard_recording_embeddings.npz",
            "idea067_oof_path": "runs/idea067_external_ast_representation_v1/canine/grouped_oof_predictions.csv",
        }
    }
    manifest, _ = idea075.load_locked_inputs(protocol)
    roles = idea075.reconstruct_roles(protocol, manifest)
    payload = roles.to_csv(index=False, lineterminator="\n").encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == idea075.EXPECTED_ROLES_SHA256


def _manifest_protocol() -> dict:
    data = {
        "canonical_audio_content_sha256": "audio-lock",
    }
    for stem in (
        "manifest", "frozen_embedding", "idea067_oof", "roles",
        "source_audio_inventory", "age_feature", "feature_summary",
    ):
        data[f"{stem}_path"] = f"artifacts/{stem}.bin"
        data[f"{stem}_sha256"] = f"{stem}-hash"
    return {
        "protocol_id": "idea075-dog-C1-age-sensitive-AST-v1",
        "data": data,
        "model": {"relative_cap": 0.25, "rms_epsilon": 1e-8},
        "acoustic_features": {"feature_names": list(idea075.FEATURE_NAMES)},
        "fixed_training": {"dog_batch_size": 4},
        "gate": idea075.EXPECTED_GATE,
    }


def test_run_manifest_rejects_resume_setting_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(idea075, "PROTOCOL_PATH", protocol_path)
    run_root = tmp_path / "run"
    protocol = _manifest_protocol()
    device = torch.device("cpu")
    first = idea075.ensure_run_manifest(protocol, run_root, resume=False, device=device)
    assert idea075.ensure_run_manifest(protocol, run_root, resume=True, device=device) == first
    changed = _manifest_protocol()
    changed["fixed_training"]["dog_batch_size"] = 3
    with pytest.raises(RuntimeError, match="manifest drift"):
        idea075.ensure_run_manifest(changed, run_root, resume=True, device=device)
    monkeypatch.setattr(idea075.platform, "python_version", lambda: "different-runtime")
    with pytest.raises(RuntimeError, match="manifest drift"):
        idea075.ensure_run_manifest(protocol, run_root, resume=True, device=device)


def test_completed_fit_resume_validates_hash_identity_and_test_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(idea075, "REPO_ROOT", tmp_path)
    store = synthetic_store(dogs=2, units_per_dog=2)
    root = tmp_path / "fit"
    root.mkdir()
    test_path = root / "test.csv"
    validation_path = root / "validation.csv"
    checkpoint_path = root / "checkpoint.pt"
    pretest_path = root / "pretest.json"
    marker_path = root / "marker.json"
    audit_path = root / "audit.json"
    probabilities = np.eye(5, dtype=float)[store.labels[:2]] * 0.8 + 0.04
    predictions = pd.DataFrame(
        {
            "recording_index": [0, 1],
            "recording_id": store.recording_ids[:2],
            "dog_id": store.dog_ids[:2],
            "true_label": store.labels[:2],
            **{column: probabilities[:, index] for index, column in enumerate(idea075.PROBABILITY_COLUMNS)},
        }
    )
    predictions.to_csv(test_path, index=False, lineterminator="\n")
    predictions.to_csv(validation_path, index=False, lineterminator="\n")
    checkpoint_path.write_bytes(b"checkpoint")
    pretest_path.write_text("{}\n", encoding="utf-8")
    marker_path.write_text("{}\n", encoding="utf-8")
    audit_path.write_text("{}\n", encoding="utf-8")
    identity = {
        "pipeline": idea075.PIPELINES[2], "base_seed": 8075,
        "repeat": 0, "outer_fold": 0, "full_seed": 8075,
    }
    summary = {
        "status": "complete", **identity,
        "outer_test_access_count": 1,
        "outer_test_predictions_path": "fit/test.csv",
        "outer_test_predictions_sha256": idea075.sha256(test_path),
        "best_validation_predictions_path": "fit/validation.csv",
        "best_validation_predictions_sha256": idea075.sha256(validation_path),
        "checkpoint_path": "fit/checkpoint.pt",
        "checkpoint_sha256": idea075.sha256(checkpoint_path),
        "pre_test_summary_path": "fit/pretest.json",
        "pre_test_summary_sha256": idea075.sha256(pretest_path),
        "outer_test_access_marker_path": "fit/marker.json",
        "outer_test_access_marker_sha256": idea075.sha256(marker_path),
        "outer_test_audit_path": "fit/audit.json",
        "outer_test_audit_sha256": idea075.sha256(audit_path),
        "checkpoint_reload_max_probability_difference": 0.0,
        "initialization_audit": {
            "trainable_parameters": idea075.EXPECTED_PARAMETERS,
            "shared_state_equal": True,
            "U1_C1_full_state_equal": True,
            "max_logit_difference_vs_A0": {idea075.PIPELINES[1]: 0.0, idea075.PIPELINES[2]: 0.0},
        },
        "model_audit_on_outer_test": {"maximum_dimension_budget_violations": 0},
    }
    summary_path = root / "summary.json"
    idea075.write_json(summary_path, summary)
    idea075.validate_completed_fit(summary_path, identity, store, np.asarray([0, 1]))
    test_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        idea075.validate_completed_fit(summary_path, identity, store, np.asarray([0, 1]))


def test_outer_test_started_without_final_csv_fails_closed_without_reprediction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(idea075, "REPO_ROOT", tmp_path)
    store = synthetic_store(dogs=4, units_per_dog=5)
    paths = idea075._fit_paths(tmp_path / "run", idea075.PIPELINES[0], 8075, 0, 0)
    paths["root"].mkdir(parents=True)
    idea075.write_json(paths["test_marker"], {"status": "started", "pre_test_summary_sha256": "missing"})
    calls = {"predict": 0}

    def forbidden_predict(*args, **kwargs):
        calls["predict"] += 1
        raise AssertionError("outer test was predicted again")

    monkeypatch.setattr(idea075, "predict_units", forbidden_predict)
    with pytest.raises(RuntimeError, match="refusing to predict again"):
        idea075.fit_outer(
            idea075.PIPELINES[0], {}, store,
            np.arange(10), np.arange(10, 15), np.arange(15, 20), np.arange(15),
            torch.device("cpu"), 8075, tmp_path / "run", 8075, 0, 0, {}, True,
        )
    assert calls["predict"] == 0


def test_aggregate_requires_exact_135_fit_factorial(tmp_path: Path) -> None:
    synthetic = []
    for pipeline in idea075.PIPELINES:
        for base_seed in idea075.BASE_SEEDS:
            for repeat in range(3):
                for fold in range(5):
                    synthetic.append(
                        {"pipeline": pipeline, "base_seed": base_seed, "repeat": repeat, "outer_fold": fold}
                    )
    synthetic[-1] = synthetic[0].copy()
    with pytest.raises(RuntimeError, match="135 unique fits"):
        idea075.aggregate(synthetic, {}, tmp_path)
