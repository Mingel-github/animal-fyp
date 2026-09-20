from __future__ import annotations

import argparse
import hashlib
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
RUNNER_PATH = (
    ROOT / "scripts" / "run_meowagenet_idea085_acoustic_temporal_residual.py"
)
SPEC = importlib.util.spec_from_file_location("idea085_temporal_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
idea085 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = idea085
SPEC.loader.exec_module(idea085)


def protocol() -> dict:
    return json.loads(idea085.PROTOCOL_PATH.read_text(encoding="utf-8"))


def synthetic_inputs() -> tuple[SimpleNamespace, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(8501)
    embeddings = rng.normal(size=(6, 768)).astype(np.float32)
    packed = rng.normal(size=(6, 21)).astype(np.float32)
    packed[:, 20] = np.arange(6, dtype=np.float32)
    sequences = []
    for index, frames in enumerate((3, 5, 7, 9, 4, 12)):
        sequence = rng.normal(size=(frames, 6)).astype(np.float32)
        sequence[(index + 1) % frames, 0] = np.nan
        sequence[(index + 2) % frames, 2] = np.nan
        sequences.append(sequence)
    trajectories = idea085.TemporalRaggedCache.from_sequences(
        [f"call-{index}" for index in range(6)], sequences
    )
    idea085.set_active_trajectories(trajectories)
    return SimpleNamespace(frozen_embeddings=embeddings), packed, np.arange(4)


def build_temporal(pipeline: str = idea085.PIPELINES[2]):
    store, packed, train = synthetic_inputs()
    idea085.idea068.idea051.reference.historical.set_seed(idea085.BASE_SEEDS[0])
    model = idea085.build_model(pipeline, protocol(), store, packed, train)
    assert isinstance(model, idea085.TemporalResidualClassifier)
    return model, store, packed, train


def test_protocol_matrix_seed_derivation_and_parameter_formula() -> None:
    item = protocol()
    idea085.verify_protocol(item)
    assert tuple(item["model"]["pipelines"]) == idea085.PIPELINES
    assert tuple(item["model"]["base_seeds"]) == idea085.BASE_SEEDS
    assert len(idea085.PIPELINES) * 3 * 3 * 4 == 144
    digest = hashlib.sha256(item["model"]["seed_derivation_text"].encode()).digest()
    candidates = [
        int.from_bytes(digest[offset : offset + 4], "big") % 10_000
        for offset in range(0, 32, 4)
    ]
    assert candidates == item["model"]["candidate_sequence"]
    assert tuple(candidates[:3]) == idea085.BASE_SEEDS
    branch_parameters = (12 * 32 * 5 + 32) + (32 * 32 * 3 + 32) + (
        32 * 128 + 128
    )
    assert branch_parameters == 9_280
    assert idea085.EXPECTED_PARAMETERS == {
        idea085.PIPELINES[0]: 99_075,
        idea085.PIPELINES[1]: 108_143,
        idea085.PIPELINES[2]: 108_355,
        idea085.PIPELINES[3]: 108_355,
    }


def test_fixed_joint_permutation_is_deterministic_exhaustive_and_joint() -> None:
    trajectories = idea085.TemporalRaggedCache.from_sequences(
        ["missingness-order-audit"],
        [
            np.asarray(
                [
                    [1, 10, np.nan, 100, 1000, 10000],
                    [2, 20, 200, 2000, 20000, 200000],
                    [np.nan, 30, 300, 3000, 30000, 300000],
                    [4, 40, 400, 4000, 40000, 400000],
                    [5, 50, 500, 5000, 50000, 500000],
                ],
                dtype=np.float32,
            )
        ],
    )
    first = idea085.fixed_joint_permutation("missingness-order-audit", 5)
    second = idea085.fixed_joint_permutation("missingness-order-audit", 5)
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(5))
    original = trajectories.sequence(0)
    shuffled = trajectories.sequence(0, shuffled=True)
    assert np.array_equal(original[first], shuffled, equal_nan=True)
    assert np.array_equal(np.isfinite(original[first]), np.isfinite(shuffled))


def test_train_only_preprocessing_and_t1_j1_statistics_match() -> None:
    t1, store, packed, train = build_temporal(idea085.PIPELINES[2])
    idea085.idea068.idea051.reference.historical.set_seed(idea085.BASE_SEEDS[0])
    j1 = idea085.build_model(
        idea085.PIPELINES[3], protocol(), store, packed, train
    )
    expected = t1.trajectories.training_statistics(train)
    actual = tuple(
        getattr(t1, name).detach().cpu().numpy()
        for name in ("temporal_median", "temporal_mean", "temporal_scale")
    )
    assert all(np.array_equal(left, right) for left, right in zip(expected, actual))
    assert all(
        torch.equal(getattr(t1, name), getattr(j1, name))
        for name in ("temporal_median", "temporal_mean", "temporal_scale")
    )
    assert idea085.preprocessing_audit(t1, train)[
        "validation_rows_used_for_statistics"
    ] is False


def test_initial_models_have_locked_counts_common_logits_and_zero_outputs() -> None:
    store, packed, train = synthetic_inputs()
    audit = idea085.initial_model_audit(
        protocol(), store, packed, train, np.asarray([4, 5]), idea085.BASE_SEEDS[0]
    )
    assert audit["parameters"] == idea085.EXPECTED_PARAMETERS
    assert set(audit["max_logit_difference_vs_A0"].values()) == {0.0}
    assert all(audit["common_AST_state_equal_to_A0"].values())
    assert audit["T1_J1_trainable_state_equal"] is True
    assert all(audit["zero_initialized_residual_outputs"].values())


def test_padding_invariance_gradient_reachability_and_shared_cap() -> None:
    model, store, packed, _ = build_temporal()
    padding = idea085.padding_invariance_audit(model, packed)
    assert padding["short_call_frames"] == 3
    assert padding["long_call_frames"] == 12
    assert padding["eval_context_max_difference"] <= 1.0e-6
    gradient = idea085.gradient_audit(model, packed[[4, 5]])
    assert gradient["zero_init_projection_bias_grad_norm"] > 0.0
    assert gradient["zero_init_conv1_weight_grad_norm"] == 0.0
    assert gradient["zero_init_conv2_weight_grad_norm"] == 0.0
    assert gradient["nonzero_projection_probe_conv1_weight_grad_norm"] > 0.0
    assert gradient["nonzero_projection_probe_conv2_weight_grad_norm"] > 0.0
    model, store, packed, _ = build_temporal()
    cap = idea085.cap_audit(model, store, packed, np.asarray([4, 5]))
    assert cap["validation_relative_perturbation_max"] <= idea085.idea071.CAP + 1e-5


def test_lookup_column_is_not_a_c1_or_temporal_model_feature() -> None:
    store, packed, train = synthetic_inputs()
    idea085.idea068.idea051.reference.historical.set_seed(idea085.BASE_SEEDS[0])
    c1 = idea085.build_model(idea085.PIPELINES[1], protocol(), store, packed, train).eval()
    left = torch.from_numpy(packed[[0]].copy())
    right = left.clone()
    right[:, 20] = 5.0
    embedding = torch.from_numpy(store.frozen_embeddings[[0]])
    with torch.no_grad():
        assert torch.equal(c1(embedding, left), c1(embedding, right))
    temporal, _, _, _ = build_temporal()
    prepared, _ = temporal._prepared_sequences(torch.from_numpy(packed[[0]]))
    assert prepared.shape[2] == 12
    assert temporal.audit()["time_or_identity_feature_entered_encoder"] is False


def test_aggregate_smoke_reports_accuracy_all_recalls_and_classification_gate(
    tmp_path: Path,
) -> None:
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
    prediction_path = tmp_path / "synthetic_animal_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    fits = []
    for base_seed in idea085.BASE_SEEDS:
        for repeat in range(3):
            for fold in range(4):
                for pipeline in idea085.PIPELINES:
                    fits.append(
                        {
                            "pipeline": pipeline,
                            "base_seed": base_seed,
                            "repeat": repeat,
                            "fold": fold,
                            "validation_animal_predictions": str(prediction_path),
                        }
                    )
    result = idea085.aggregate(fits, protocol())
    assert result["fits"] == 144
    assert result["paired_fold_comparisons"] == 36
    assert result["seed_repeat_estimates"] == 9
    assert result["split_cell_estimates"] == 12
    assert result["no_single_global_pass_flag"] is True
    means = result["pipeline_seed_repeat_means"]
    assert set(means) == {
        "macro_f1", "plain_accuracy", "balanced_accuracy", "cross_entropy",
        "brier", "kitten_recall", "adult_recall", "senior_recall",
    }
    for comparison in idea085.COMPARISONS:
        item = result["comparison_results"][comparison]
        assert set(item["classification_conditions"]) == {
            "mean_macro_f1_delta", "positive_base_seed_means",
            "positive_seed_repeats", "nonnegative_split_cells",
            "worst_split_cell",
        }
        assert item["gate_passed"] == item["classification_gate_passed"]
        assert item["auxiliary_profile"][
            "auxiliary_axes_are_not_classification_gate_conditions"
        ] is True


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
        idea085.run(args)
