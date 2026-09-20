from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_idea086_acoustic_set_residual.py"
SPEC = importlib.util.spec_from_file_location("idea086_set_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
idea086 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = idea086
SPEC.loader.exec_module(idea086)


def protocol() -> dict:
    return json.loads(idea086.PROTOCOL_PATH.read_text(encoding="utf-8"))


def synthetic_inputs() -> tuple[SimpleNamespace, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(8601)
    embeddings = rng.normal(size=(7, 768)).astype(np.float32)
    packed = rng.normal(size=(7, 21)).astype(np.float32)
    packed[:, 20] = np.arange(7, dtype=np.float32)
    sequences = []
    for index, frames in enumerate((2, 5, 11, 4, 9, 3, 13)):
        sequence = rng.normal(size=(frames, 6)).astype(np.float32)
        sequence[(index + 1) % frames, 0] = np.nan
        sequence[(index + 1) % frames, 2] = np.nan
        sequences.append(sequence)
    trajectories = idea086.idea085.TemporalRaggedCache.from_sequences(
        [f"call-{index}" for index in range(7)], sequences
    )
    idea086.set_active_trajectories(trajectories)
    return SimpleNamespace(frozen_embeddings=embeddings), packed, np.arange(5)


def build_set():
    store, packed, train = synthetic_inputs()
    idea086.idea068.idea051.reference.historical.set_seed(idea086.BASE_SEEDS[0])
    model = idea086.build_model(idea086.PIPELINE, protocol(), store, packed, train)
    assert isinstance(model, idea086.SetResidualClassifier)
    return model, store, packed, train


def test_protocol_and_locked_parameter_formula() -> None:
    item = protocol()
    idea086.verify_protocol(item)
    assert item["model"]["new_fits"] == 36
    assert item["model"]["reused_reference_fits"] == 144
    assert item["classification_gate"]["gated_comparisons"] == list(
        idea086.GATED_COMPARISONS
    )
    branch = (12 * 48 + 48) + (48 * 48 + 48) + (48 * 128 + 128)
    assert branch == idea086.EXPECTED_BRANCH_PARAMETERS == 9_248
    assert 99_075 + branch == idea086.EXPECTED_PARAMETERS == 108_323


def test_nontrivial_permutation_invariance_context_residual_and_logits() -> None:
    model, store, packed, _ = build_set()
    indices = np.asarray([0, 2, 5], dtype=np.int64)
    result = idea086.permutation_invariance_audit(
        model,
        torch.from_numpy(store.frozen_embeddings[indices]),
        torch.from_numpy(packed[indices]),
        override_projection=True,
    )
    assert result["passed"] is True
    assert result["mixed_lengths"] is True
    assert result["nonzero_frame_encoder_norm"] > 0.0
    assert result["projection_overridden_nonzero"] is True
    assert result["original_projection_weight_norm"] == 0.0
    assert result["active_probe_projection_weight_norm"] > 0.0
    assert result["native_context_norm"] > 0.0
    assert result["native_residual_norm"] > 0.0
    for key, value in result.items():
        if key.endswith("max_abs_difference"):
            assert value <= 2.0e-5


def test_real_cache_permutation_invariance_and_joint_finite_rows() -> None:
    item = protocol()
    store = idea086.idea068.idea051.reference.historical.idea019.load_feature_store()
    packed, _ = idea086.load_features(item, store.call_ids)
    train = np.arange(32, dtype=np.int64)
    model = idea086.build_model(idea086.PIPELINE, item, store, packed, train)
    indices = idea086.audit_probe_indices(model.trajectories, np.arange(40, 80))
    result = idea086.permutation_invariance_audit(
        model,
        torch.from_numpy(store.frozen_embeddings[indices]),
        torch.from_numpy(packed[indices]),
        override_projection=True,
    )
    assert result["passed"] is True
    call_index = int(indices[0])
    native, _ = model._prepared_sequences(torch.from_numpy(packed[[call_index]]), "native")
    shuffled, _ = model._prepared_sequences(
        torch.from_numpy(packed[[call_index]]), "idea085_j1"
    )
    order = idea086.idea085.fixed_joint_permutation(
        model.trajectories.call_ids[call_index], int(model.trajectories.lengths[call_index])
    )
    assert torch.equal(native[0, order], shuffled[0])


def test_padding_zero_init_gradients_cap_and_no_prohibited_modules() -> None:
    model, store, packed, _ = build_set()
    padding = idea086.padding_audit(model, packed)
    assert padding["padding_excluded"] is True
    assert padding["eval_context_max_difference"] <= 1.0e-6

    model, _, packed, _ = build_set()
    gradient = idea086.gradient_audit(model, packed[[0, 2, 5]])
    assert gradient["zero_init_projection_bias_grad_norm"] > 0.0
    assert gradient["zero_init_frame_linear1_weight_grad_norm"] == 0.0
    assert gradient["zero_init_frame_linear2_weight_grad_norm"] == 0.0
    assert gradient["nonzero_projection_frame_linear1_weight_grad_norm"] > 0.0
    assert gradient["nonzero_projection_frame_linear2_weight_grad_norm"] > 0.0

    model, store, packed, _ = build_set()
    cap = idea086.cap_audit(model, store, packed, np.asarray([0, 2, 5]))
    assert cap["validation_relative_perturbation_max"] <= idea086.idea071.CAP + 1e-5
    assert not any(
        isinstance(module, (torch.nn.Conv1d, torch.nn.MultiheadAttention))
        for module in model.modules()
    )


def test_lookup_is_retrieval_only_and_preprocessing_is_train_only() -> None:
    model, _, packed, train = build_set()
    prepared, mask = model._prepared_sequences(torch.from_numpy(packed[[0, 2]]))
    assert prepared.shape[-1] == 12
    assert mask.dtype == torch.bool
    assert model.audit()["time_or_identity_feature_entered_encoder"] is False
    expected = model.trajectories.training_statistics(train)
    actual = tuple(
        getattr(model, name).detach().cpu().numpy()
        for name in ("set_median", "set_mean", "set_scale")
    )
    assert all(np.array_equal(left, right) for left, right in zip(expected, actual))


def test_common_head_zero_logits_preprocessing_and_rng_compatibility() -> None:
    store, packed, train = synthetic_inputs()
    result = idea086.initial_compatibility_audit(
        protocol(), store, packed, train, np.asarray([5, 6]), idea086.BASE_SEEDS[0]
    )
    assert result["parameters"] == idea086.EXPECTED_PARAMETERS
    assert result["common_AST_state_equal_to_A0"] is True
    assert result["common_AST_state_equal_to_T1"] is True
    assert result["zero_init_max_logit_difference_vs_A0"] == 0.0
    assert result["train_only_preprocessing_equal_to_IDEA085_T1"] is True
    assert result["post_build_rng_reset_probe_equal"] is True
    assert result["C1_branch_stacked"] is False


def test_read_only_reuse_manifest_is_complete_and_hashed() -> None:
    item = protocol()
    store = idea086.idea068.idea051.reference.historical.idea019.load_feature_store()
    roles = pd.read_csv(ROOT / item["data"]["roles_path"], dtype={"cat_id": str})
    fits, manifest = idea086.collect_reused_fits(item, store, roles, verify_rows=False)
    assert len(fits) == 144
    assert manifest["fit_summaries"] == 144
    assert manifest["prediction_files"] == 288
    assert len(manifest["entries"]) == 432
    assert len(manifest["content_sha256"]) == 64
    assert manifest["outer_test_accessed"] is False
    assert manifest["all_432_files_match_frozen_independent_audit"] is True


def test_probability_validation_and_call_to_animal_reconstruction() -> None:
    calls = pd.DataFrame(
        {
            "call_id": ["a", "b", "c"], "cat_id": ["x", "x", "y"],
            "true_label": [0, 0, 2], "predicted_label": [0, 0, 2],
            "prob_kitten": [0.8, 0.6, 0.1],
            "prob_adult": [0.1, 0.2, 0.1],
            "prob_senior": [0.1, 0.2, 0.8],
        }
    )
    idea086.validate_prediction_probabilities(calls, "synthetic")
    animals = idea086.reconstruct_animals_from_calls(calls)
    idea086.compare_animal_reconstruction(animals, animals.copy(), "synthetic")
    assert animals.set_index("cat_id").loc["x", "call_count"] == 2
    broken = calls.copy()
    broken.loc[0, "prob_kitten"] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        idea086.validate_prediction_probabilities(broken, "synthetic")


def test_aggregate_reports_four_comparisons_and_only_two_gates(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "cat_id": ["cat-0", "cat-1", "cat-2"],
            "true_label": [0, 1, 2],
            "predicted_label": [0, 1, 2],
            "prob_kitten": [0.8, 0.1, 0.1],
            "prob_adult": [0.1, 0.8, 0.1],
            "prob_senior": [0.1, 0.1, 0.8],
        }
    )
    path = tmp_path / "animals.csv"
    predictions.to_csv(path, index=False)
    new_fits = []
    reused = []
    for base_seed in idea086.BASE_SEEDS:
        for repeat in range(3):
            for fold in range(4):
                new_fits.append(
                    {"pipeline": idea086.PIPELINE, "base_seed": base_seed, "repeat": repeat, "fold": fold, "validation_animal_predictions": str(path)}
                )
                for pipeline in idea086.REFERENCE_PIPELINES:
                    reused.append(
                        {"pipeline": pipeline, "base_seed": base_seed, "repeat": repeat, "fold": fold, "validation_animal_predictions": str(path)}
                    )
    result = idea086.aggregate(new_fits, reused, protocol())
    assert result["new_candidate_fits"] == 36
    assert result["reused_reference_fits"] == 144
    assert result["no_single_global_gate"] is True
    assert set(result["comparison_results"]) == set(idea086.COMPARISONS)
    assert result["animal_occurrence_accounting"][idea086.PIPELINE]["unique_cats"] == 3
    for name, comparison in result["comparison_results"].items():
        if name in idea086.GATED_COMPARISONS:
            assert comparison["classification_gate_applicable"] is True
            assert isinstance(comparison["classification_gate_passed"], bool)
        else:
            assert comparison["classification_gate_applicable"] is False
            assert comparison["classification_gate_passed"] is None
        assert set(comparison["error_correction_profile"]["net_correction_positive_tied_negative"]) == {"positive", "tied", "negative"}


def test_formal_run_and_resume_validation_fail_closed() -> None:
    args = argparse.Namespace(
        stage="run", output_subdir="unused", device="cpu", resume=False,
        director_authorized=False, max_cells=None,
    )
    with pytest.raises(RuntimeError, match="research director"):
        idea086.run(args)
    with pytest.raises(RuntimeError, match="resume identity mismatch"):
        idea086.validate_new_fit(
            {"status": "complete", "pipeline": "wrong"},
            idea086.BASE_SEEDS[0], idea086.BASE_SEEDS[0], 0, 0,
        )
