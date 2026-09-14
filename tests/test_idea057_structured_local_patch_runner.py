from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_idea057_structured_local_patch.py"
PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea057_structured_local_patch_v1.json"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("idea057_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_store(runner):
    rng = np.random.default_rng(57)
    calls = 6
    grid = rng.normal(size=(calls, 9, 768)).astype(np.float32)
    return runner.GridStore(
        global_embeddings=rng.normal(size=(calls, 768)).astype(np.float32),
        temporal_tokens=grid.reshape(calls * 9, 768),
        call_token_indices=tuple(
            np.arange(index * 9, (index + 1) * 9, dtype=np.int64)
            for index in range(calls)
        ),
        call_ids=np.asarray([f"call-{index}" for index in range(calls)]),
        cat_ids=np.asarray([f"cat-{index}" for index in range(calls)]),
        labels=np.asarray([0, 1, 2, 0, 1, 2]),
        durations=np.ones(calls, dtype=np.float32),
        valid_frame_fraction=np.full(calls, 0.5, dtype=np.float32),
        mean_abs_fbank=np.full(calls, 0.4, dtype=np.float32),
        fbank_std=np.full(calls, 0.4, dtype=np.float32),
    )


def test_protocol_declares_bounded_inner_selection_and_matched_outer_matrix() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == "meowagenet-idea057-structured-local-patch-v1"
    assert protocol["splits"]["excluded_selection_role"] == "test"
    assert set(protocol["inner_selection"]["candidate_recipes"]) == {
        "middle_frequency_strip",
        "full_3x3_grid",
    }
    assert protocol["inner_selection"]["candidate_fits"] == 24
    assert protocol["initial_evaluation"]["total_outer_fits"] == 36
    assert protocol["feature_preparation"]["outer_test_accessed"] is False
    assert protocol["smoke"]["outer_test_accessed"] is False


def test_only_evaluation_requests_outer_test_indices() -> None:
    runner = load_runner()
    assert "include_test=False" in inspect.getsource(runner.run_selection)
    assert "include_test=False" in inspect.getsource(runner.run_smoke)
    assert "include_test=True" in inspect.getsource(runner.run_evaluation)


def test_m1_and_c1_are_parameter_matched_and_start_at_r0_logits() -> None:
    runner = load_runner()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["_active_recipe_id"] = "middle_frequency_strip"
    store = synthetic_store(runner)
    models = []
    for pipeline in runner.PIPELINES:
        runner.reference.historical.set_seed(17)
        models.append(runner.build_model(pipeline, protocol, store, np.arange(6)).eval())
    counts = [
        runner.reference.historical.idea019.trainable_counts(model)["trainable"]
        for model in models
    ]
    assert counts[1] == counts[2]
    assert counts[1] > counts[0]
    global_embeddings = torch.from_numpy(store.global_embeddings[:2])
    local = torch.from_numpy(store.temporal_tokens[:18])
    token_to_call = torch.arange(2).repeat_interleave(9)
    with torch.no_grad():
        logits = [
            model(global_embeddings, local, token_to_call, 2) for model in models
        ]
    assert torch.equal(logits[0], logits[1])
    assert torch.equal(logits[0], logits[2])


def test_position_removed_control_is_invariant_to_cell_permutation() -> None:
    runner = load_runner()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["_active_recipe_id"] = "full_3x3_grid"
    store = synthetic_store(runner)
    train_indices = np.arange(6)
    runner.reference.historical.set_seed(17)
    m1 = runner.build_model(runner.PIPELINES[1], protocol, store, train_indices).eval()
    runner.reference.historical.set_seed(17)
    c1 = runner.build_model(runner.PIPELINES[2], protocol, store, train_indices).eval()
    with torch.no_grad():
        m1.residual_gate.fill_(0.5)
        c1.residual_gate.fill_(0.5)
    global_embeddings = torch.from_numpy(store.global_embeddings[:2])
    grid = torch.from_numpy(store.temporal_tokens[:18]).reshape(2, 9, 768)
    permutation = torch.tensor([8, 7, 6, 5, 4, 3, 2, 1, 0])
    token_to_call = torch.arange(2).repeat_interleave(9)
    with torch.no_grad():
        m1_original = m1(global_embeddings, grid.reshape(-1, 768), token_to_call, 2)
        m1_permuted = m1(
            global_embeddings, grid[:, permutation].reshape(-1, 768), token_to_call, 2
        )
        c1_original = c1(global_embeddings, grid.reshape(-1, 768), token_to_call, 2)
        c1_permuted = c1(
            global_embeddings, grid[:, permutation].reshape(-1, 768), token_to_call, 2
        )
    assert not torch.allclose(m1_original, m1_permuted)
    assert torch.allclose(c1_original, c1_permuted, atol=1.0e-6, rtol=0.0)


def test_local_initialization_preserves_r0_training_rng_stream() -> None:
    runner = load_runner()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["_active_recipe_id"] = "middle_frequency_strip"
    store = synthetic_store(runner)
    runner.reference.historical.set_seed(17)
    runner.build_model(runner.PIPELINES[0], protocol, store, np.arange(6))
    after_r0 = torch.rand(8)
    runner.reference.historical.set_seed(17)
    runner.build_model(runner.PIPELINES[1], protocol, store, np.arange(6))
    after_m1 = torch.rand(8)
    runner.reference.historical.set_seed(17)
    runner.build_model(runner.PIPELINES[2], protocol, store, np.arange(6))
    after_c1 = torch.rand(8)
    assert torch.equal(after_r0, after_m1)
    assert torch.equal(after_r0, after_c1)


def test_seed_expansion_requires_both_reference_and_mechanism_controls() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    gate = protocol["seed_expansion_gate"]
    assert gate["minimum_mean_macro_f1_gain_over_R0"] == 0.005
    assert gate["minimum_positive_repeats_over_R0"] == 2
    assert gate["minimum_mean_macro_f1_gain_over_C1"] == 0.0
    assert gate["minimum_positive_repeats_over_C1"] == 2
