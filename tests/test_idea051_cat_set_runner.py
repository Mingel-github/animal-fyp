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

import run_meowagenet_idea051_cat_set as runner  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT / "configs" / "protocol" / "meowagenet_idea051_cat_set_v1.json"
)
DIAGNOSTIC_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_idea051_cat_set_diagnostics_v1.json"
)
RUN_ROOT = REPO_ROOT / "runs" / "meowagenet_idea051_cat_set_v1"


def load_store_and_inner_train() -> tuple[object, np.ndarray]:
    store = runner.reference.historical.idea019.load_feature_store()
    roles = pd.read_csv(runner.ROLES_PATH, dtype={"cat_id": str})
    indices = runner.reference.historical.fold_indices(
        store, roles, repeat=0, outer_fold=0, include_test=False
    )
    return store, indices["train"]


def test_protocol_locks_three_pipeline_36_fit_initial_screen() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == "meowagenet-idea051-cat-set-v1"
    assert tuple(protocol["pipelines"]) == runner.PIPELINES
    assert protocol["fixed_training"]["set_cat_batch_size"] == 4
    assert protocol["fixed_training"]["set_loss_denominator_cats"] == 4
    assert protocol["initial_evaluation"]["base_seeds"] == [17]
    assert protocol["initial_evaluation"]["total_outer_fits"] == 36
    assert protocol["seed_expansion_gate"]["minimum_mean_macro_f1_gain"] == 0.005
    runner.verify_protocol(protocol)


def test_diagnostic_establishes_variable_bags_and_call_disagreement() -> None:
    diagnostic = json.loads(DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
    assert diagnostic["status"] == "complete"
    assert diagnostic["dataset_structure"]["calls"] == 792
    assert diagnostic["dataset_structure"]["cats"] == 111
    assert diagnostic["dataset_structure"]["call_count_min"] == 1
    assert diagnostic["dataset_structure"]["call_count_max"] == 45
    assert diagnostic["dataset_structure"]["single_call_cats"] == 20
    assert diagnostic["within_cat_signal"][
        "cat_evaluations_with_any_argmax_disagreement"
    ] == 484
    assert diagnostic["within_cat_signal"][
        "fraction_with_any_argmax_disagreement"
    ] == pytest.approx(0.4844844844844845)


def test_cat_set_dataset_preserves_all_calls_and_singletons() -> None:
    store = runner.reference.historical.idea019.load_feature_store()
    dataset = runner.CatSetDataset(
        store, np.arange(len(store.call_ids), dtype=np.int64)
    )
    sizes = [len(calls) for calls in dataset.call_indices]
    assert len(dataset) == 111
    assert sum(sizes) == 792
    assert min(sizes) == 1
    assert max(sizes) == 45
    assert sum(size == 1 for size in sizes) == 20
    batch = runner.collate_cat_sets([dataset[0], dataset[1], dataset[2], dataset[3]])
    assert len(batch["labels"]) == 4
    assert len(batch["call_indices"]) == sum(sizes[:4])
    assert sorted(batch["call_indices"].tolist()) == sorted(
        np.concatenate(dataset.call_indices[:4]).tolist()
    )


def test_deterministic_cat_batches_preserve_units_without_singletons() -> None:
    first = list(runner.DeterministicNoSingletonBatchSampler(73, 4, 17))
    second = list(runner.DeterministicNoSingletonBatchSampler(73, 4, 17))
    assert first == second
    assert all(2 <= len(batch) <= 4 for batch in first)
    flattened = [index for batch in first for index in batch]
    assert sorted(flattened) == list(range(73))
    assert [len(first[-2]), len(first[-1])] == [3, 2]


def test_cat_class_weights_have_equal_class_totals() -> None:
    store, train = load_store_and_inner_train()
    dataset = runner.CatSetDataset(store, train)
    weights = runner.set_class_weights(dataset)
    cat_weights = weights[dataset.labels]
    expected = len(dataset) / 3.0
    for class_index in range(3):
        assert float(cat_weights[dataset.labels == class_index].sum()) == pytest.approx(
            expected, rel=1.0e-6
        )
    assert float(cat_weights.mean()) == pytest.approx(1.0)


def test_fixed_cat_denominator_matches_four_cat_mean_gradient() -> None:
    generator = torch.Generator().manual_seed(20260913)
    labels = torch.randint(0, 3, (4,), generator=generator)
    full_logits = torch.randn(4, 3, generator=generator, requires_grad=True)
    torch.nn.functional.cross_entropy(full_logits, labels, reduction="mean").backward()
    expected = full_logits.grad.detach().clone()

    actual_logits = full_logits.detach().clone().requires_grad_(True)
    per_cat = torch.nn.functional.cross_entropy(
        actual_logits, labels, reduction="none"
    )
    runner.globally_weighted_set_micro_loss(per_cat, torch.ones(4), 4).backward()
    assert torch.allclose(actual_logits.grad, expected, atol=1.0e-7, rtol=1.0e-6)
    assert "weights.sum" not in inspect.getsource(
        runner.globally_weighted_set_micro_loss
    )


def test_zero_initialized_attention_starts_as_hidden_mean() -> None:
    protocol = runner.read_json(PROTOCOL_PATH)
    store, train = load_store_and_inner_train()
    runner.reference.historical.set_seed(17)
    mean_model = runner.build_model(runner.PIPELINES[1], protocol, store, train)
    runner.reference.historical.set_seed(17)
    attention_model = runner.build_model(runner.PIPELINES[2], protocol, store, train)
    mean_state = mean_model.state_dict()
    attention_state = attention_model.state_dict()
    for key, value in mean_state.items():
        assert torch.equal(value, attention_state[key])
    assert torch.count_nonzero(attention_model.attention.weight) == 0
    assert torch.count_nonzero(attention_model.attention.bias) == 0

    dataset = runner.CatSetDataset(store, train)
    batch = runner.collate_cat_sets([dataset[0], dataset[1], dataset[2], dataset[3]])
    mean_model.eval()
    attention_model.eval()
    with torch.no_grad():
        mean_logits, mean_weights = mean_model(
            batch["embeddings"], batch["instance_to_cat"], 4
        )
        attention_logits, attention_weights = attention_model(
            batch["embeddings"], batch["instance_to_cat"], 4
        )
    assert torch.allclose(mean_logits, attention_logits, atol=1.0e-7, rtol=1.0e-6)
    assert torch.allclose(mean_weights, attention_weights, atol=1.0e-7, rtol=1.0e-6)
    for local_cat in range(4):
        mask = batch["instance_to_cat"] == local_cat
        assert float(attention_weights[mask].sum()) == pytest.approx(1.0)


def test_runner_separates_smoke_lock_and_initial_evaluation() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert 'choices=("smoke", "evaluate")' in source
    assert "include_test=False" in source
    assert "include_test=True" in source
    assert "locked_for_idea051_initial_evaluation" in source
    assert "paired_cat_bootstrap" in source
