"""Tests for the independent IDEA-089 artifact verifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_idea089_unified_core_results.py"


def load_verifier():
    spec = importlib.util.spec_from_file_location("idea089_result_verifier", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load IDEA-089 result verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_preflight_and_dependency_hashes_pass() -> None:
    verifier = load_verifier()
    protocol = verifier.read_json(verifier.PROTOCOL_PATH)
    assert len(verifier.verify_protocol_dependencies(protocol)) == 12
    result = verifier.verify_preflight(protocol)
    assert result["status"] == "GO"
    assert result["training_started"] is False
    assert result["outer_test_accessed"] is False
    assert result["physical_fits"] == 432


def test_full_selection_coordinate_bank_is_exact() -> None:
    verifier = load_verifier()
    protocol = verifier.read_json(verifier.PROTOCOL_PATH)
    coordinates = verifier.all_selection_coordinates(protocol)
    assert len(coordinates) == 216
    assert len(set(coordinates)) == 216
    assert set(item[0] for item in coordinates) == set(verifier.PIPELINES)
    assert set((item[1], item[2], item[3]) for item in coordinates) == {
        (repeat, fold, seed)
        for repeat in (0, 1, 2)
        for fold in (0, 1, 2, 3)
        for seed in (17, 43, 101)
    }


def test_unit_to_cat_probability_mean_and_metrics() -> None:
    verifier = load_verifier()
    units = pd.DataFrame(
        {
            "unit_id": ["a1", "a2", "b1", "c1"],
            "cat_id": ["a", "a", "b", "c"],
            "true_label": [0, 0, 1, 2],
            "prob_kitten": [0.8, 0.6, 0.1, 0.1],
            "prob_adult": [0.1, 0.2, 0.8, 0.2],
            "prob_senior": [0.1, 0.2, 0.1, 0.7],
        }
    )
    animals = verifier.units_to_animals(units)
    assert animals["unit_count"].tolist() == [2, 1, 1]
    assert np.allclose(
        animals.loc[animals["cat_id"] == "a", list(verifier.PROBABILITY_COLUMNS)],
        [[0.7, 0.15, 0.15]],
    )
    metrics = verifier.metric_bundle(animals)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_strict_best_epoch_recomputation() -> None:
    verifier = load_verifier()
    values = [0.8, 0.7999995, 0.799998, 0.7999975]
    history = [
        {
            "epoch": index + 1,
            "validation_animal_metrics": {"cross_entropy": value},
        }
        for index, value in enumerate(values)
    ]
    best_epoch, best_ce, stale = verifier.best_epoch_from_history(history, 1.0e-6)
    assert best_epoch == 3
    assert best_ce == values[2]
    assert stale == 1


def test_acoustic_preprocessing_is_training_side_and_nan_safe() -> None:
    verifier = load_verifier()
    values = np.asarray(
        [[1.0, np.nan], [3.0, 2.0], [5.0, 6.0]], dtype=np.float32
    )
    median, mean, scale = verifier.acoustic_standardizer(values)
    assert np.array_equal(median, np.asarray([3.0, 4.0], dtype=np.float32))
    assert np.array_equal(mean, np.asarray([3.0, 4.0], dtype=np.float32))
    assert np.isfinite(scale).all()
    assert np.all(scale > 0.0)


def test_vgg_and_ast_deterministic_order_hashes() -> None:
    verifier = load_verifier()
    train_indices = np.asarray([1, 3, 4, 7], dtype=np.int64)
    training_seed = 1_000_017
    vgg_history = []
    for epoch in (1, 2):
        order = np.random.default_rng(training_seed + epoch).permutation(4)
        digest = verifier.hashlib.sha256(
            train_indices[order].astype("<i8").tobytes()
        ).hexdigest()
        vgg_history.append({"unit_order_sha256": digest})
    verifier.verify_history_order(
        vgg_history,
        "VGG128_no_f0",
        train_indices,
        np.asarray(["a"] * 8),
        17,
    )

    cat_ids = np.asarray(["x", "a", "x", "b", "c", "x", "x", "d"])
    cats = np.asarray(["a", "b", "c", "d"])
    generator = verifier.torch.Generator().manual_seed(17)
    coverage = verifier.hashlib.sha256(
        np.sort(train_indices.astype("<i8")).tobytes()
    ).hexdigest()
    ast_history = []
    for _ in (1, 2):
        order = verifier.torch.randperm(4, generator=generator).numpy()
        ast_history.append(
            {
                "train_audit": {
                    "cat_order_sha256": verifier.text_sha256(
                        "\n".join(cats[order].tolist())
                    ),
                    "call_coverage_sha256": coverage,
                    "cats": 4,
                    "calls": 4,
                }
            }
        )
    verifier.verify_history_order(
        ast_history, "A0_ast_only", train_indices, cat_ids, 17
    )


def test_bootstrap_vectorization_matches_direct_metric_average() -> None:
    verifier = load_verifier()
    labels = np.repeat(np.arange(3, dtype=np.int64), 3)
    rng = np.random.default_rng(71)
    probabilities = rng.random((2, 9, 3))
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    samples = verifier.stratified_bootstrap_indices(labels, 7, 20260920)
    vectorized = verifier.bootstrap_model_metrics(
        labels, probabilities, samples, chunk_size=3
    )
    direct = {name: [] for name in vectorized}
    for sample in samples:
        per_oof = []
        for oof in probabilities:
            frame = pd.DataFrame(
                {
                    "cat_id": np.arange(len(sample)).astype(str),
                    "true_label": labels[sample],
                    **{
                        column: oof[sample, index]
                        for index, column in enumerate(verifier.PROBABILITY_COLUMNS)
                    },
                }
            )
            per_oof.append(verifier.flatten_metrics(verifier.metric_bundle(frame)))
        for name in direct:
            direct[name].append(np.mean([row[name] for row in per_oof]))
    for name in vectorized:
        assert np.allclose(vectorized[name], direct[name], atol=1e-15, rtol=0.0)


def test_saved_model_state_digest_matches_runner_algorithm(tmp_path: Path) -> None:
    verifier = load_verifier()
    keras_path = tmp_path / "weights.npz"
    np.savez_compressed(
        keras_path,
        weight_000=np.arange(6, dtype=np.float32).reshape(2, 3),
        weight_001=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )
    digest = verifier.hashlib.sha256()
    with np.load(keras_path, allow_pickle=False) as loaded:
        for key in sorted(loaded.files):
            array = np.ascontiguousarray(loaded[key])
            digest.update(key.encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes())
    assert verifier.saved_model_state_digest(keras_path, "VGG128_no_f0") == digest.hexdigest()

    torch_path = tmp_path / "weights.pt"
    verifier.torch.save(
        {"b": verifier.torch.arange(3), "a": verifier.torch.ones(2)}, torch_path
    )
    torch_digest = verifier.hashlib.sha256()
    state = verifier.torch.load(torch_path, map_location="cpu", weights_only=False)
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        torch_digest.update(name.encode("utf-8"))
        torch_digest.update(str(array.dtype).encode("ascii"))
        torch_digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        torch_digest.update(array.tobytes())
    assert (
        verifier.saved_model_state_digest(torch_path, "A0_ast_only")
        == torch_digest.hexdigest()
    )


def test_first_six_outer_coordinates_are_exact() -> None:
    verifier = load_verifier()
    assert [(pipeline, 0, 0, 17) for pipeline in verifier.PIPELINES] == [
        ("VGG128_no_f0", 0, 0, 17),
        ("VGG129_with_f0", 0, 0, 17),
        ("A0_ast_only", 0, 0, 17),
        ("D0_direct_concat", 0, 0, 17),
        ("U1_wide_unbounded_additive", 0, 0, 17),
        ("C1_bounded_wide_additive", 0, 0, 17),
    ]
