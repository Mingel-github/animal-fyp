from __future__ import annotations

import ast
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

import run_meowagenet_ast_cat_balance_global_weighting_v1 as runner  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_ast_cat_balance_global_weighting_v1.json"
)
RUNNER_PATH = (
    REPO_ROOT
    / "scripts"
    / "run_meowagenet_ast_cat_balance_global_weighting_v1.py"
)
RUN_ROOT = REPO_ROOT / "runs" / "meowagenet_ast_cat_balance_global_weighting_v1"
RESULTS_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_ast_cat_balance_global_weighting_v1_results.json"
)


def actual_inner_train() -> tuple[object, np.ndarray]:
    store = runner.historical.idea019.load_feature_store()
    roles = pd.read_csv(runner.ROLES_PATH, dtype={"cat_id": str})
    indices = runner.historical.fold_indices(
        store, roles, repeat=0, outer_fold=0, include_test=False
    )
    return store, indices["train"]


def test_protocol_freezes_global_denominator_and_72_fit_matrix() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == "meowagenet-ast-cat-balance-global-weighting-v1"
    assert tuple(protocol["pipelines"]) == runner.PIPELINES
    assert protocol["fixed_training"]["micro_batch_size"] == 8
    assert protocol["fixed_training"]["gradient_accumulation_steps"] == 4
    assert protocol["fixed_training"]["accumulation_window_calls"] == 32
    assert protocol["fixed_training"]["partial_window_denominator"] == 32
    assert protocol["evaluation"]["base_seeds"] == [17, 43, 101]
    assert protocol["evaluation"]["total_outer_fits"] == 72
    runner.verify_protocol(protocol)


def test_runner_separates_smoke_lock_and_formal_evaluation() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert {
        "run_smoke",
        "verify_execution_lock",
        "run_evaluation",
        "aggregate_evaluation",
        "paired_cat_bootstrap",
        "raw_prediction_inventory",
        "main",
    } <= function_names
    assert 'choices=("smoke", "evaluate")' in source
    assert "include_test=False" in source
    assert "include_test=True" in source
    assert "locked_for_global_weighting_evaluation" in source


def test_global_lookup_totals_match_the_analytic_targets() -> None:
    store, train = actual_inner_train()
    c0 = runner.global_class_balanced_call_weights(store.labels, train)
    c1 = runner.global_cat_and_class_balanced_call_weights(
        store.labels, store.cat_ids, train
    )
    expected_class_total = len(train) / 3.0
    for class_index in range(3):
        class_calls = train[store.labels[train] == class_index]
        assert float(c0[class_calls].sum()) == pytest.approx(expected_class_total)
        assert float(c1[class_calls].sum()) == pytest.approx(expected_class_total)
        c1_cat_totals = [
            float(c1[class_calls[store.cat_ids[class_calls] == cat_id]].sum())
            for cat_id in np.unique(store.cat_ids[class_calls])
        ]
        assert max(c1_cat_totals) == pytest.approx(min(c1_cat_totals), rel=1.0e-6)
    assert float(c0[train].mean()) == pytest.approx(1.0)
    assert float(c1[train].mean()) == pytest.approx(1.0)


def test_deterministic_batches_preserve_every_call_without_singletons() -> None:
    size = 569
    first = list(runner.DeterministicNoSingletonBatchSampler(size, 8, 301))
    second = list(runner.DeterministicNoSingletonBatchSampler(size, 8, 301))
    assert first == second
    assert all(2 <= len(batch) <= 8 for batch in first)
    flattened = [index for batch in first for index in batch]
    assert len(flattened) == size
    assert sorted(flattened) == list(range(size))
    assert [len(first[-2]), len(first[-1])] == [7, 2]


def test_effective_epoch_coefficients_equal_lookup_coefficients_over_32() -> None:
    store, train = actual_inner_train()
    lookup = runner.global_cat_and_class_balanced_call_weights(
        store.labels, store.cat_ids, train
    )
    relative_batches = list(
        runner.DeterministicNoSingletonBatchSampler(len(train), 8, 17)
    )
    processed = [int(train[index]) for batch in relative_batches for index in batch]
    audit = runner.effective_coefficient_audit(
        processed,
        [len(batch) for batch in relative_batches],
        lookup,
        store,
        accumulation_window_calls=32,
        accumulation_steps=4,
    )
    assert audit["processed_calls"] == len(train)
    assert audit["unique_processed_calls"] == len(train)
    for class_index, label in enumerate(runner.LABEL_NAMES):
        class_calls = train[store.labels[train] == class_index]
        assert audit["effective_coefficient_by_class"][label] == pytest.approx(
            float(lookup[class_calls].sum()) / 32.0
        )
        class_cats = np.unique(store.cat_ids[class_calls])
        cat_values = [
            audit["effective_coefficient_by_cat"][str(cat_id)]
            for cat_id in class_cats
        ]
        assert max(cat_values) == pytest.approx(min(cat_values), rel=1.0e-6)


def test_loss_path_uses_fixed_32_denominator() -> None:
    source = inspect.getsource(runner.global_weighted_micro_loss)
    assert "weights.sum" not in source
    assert "accumulation_window_calls" in source
    losses = torch.tensor([1.0, 2.0, 3.0])
    weights = torch.tensor([0.5, 1.0, 2.0])
    actual = runner.global_weighted_micro_loss(losses, weights, 32)
    assert float(actual) == pytest.approx(float((losses * weights).sum()) / 32.0)


def test_unit_weights_match_mean_cross_entropy_gradient_for_32_calls() -> None:
    generator = torch.Generator().manual_seed(20260911)
    labels = torch.randint(0, 3, (32,), generator=generator)
    full_logits = torch.randn(32, 3, generator=generator, requires_grad=True)
    full_loss = torch.nn.functional.cross_entropy(full_logits, labels, reduction="mean")
    full_loss.backward()
    expected_gradient = full_logits.grad.detach().clone()

    micro_logits = full_logits.detach().clone().requires_grad_(True)
    for start in range(0, 32, 8):
        per_call = torch.nn.functional.cross_entropy(
            micro_logits[start : start + 8],
            labels[start : start + 8],
            reduction="none",
        )
        loss = runner.global_weighted_micro_loss(per_call, torch.ones(8), 32)
        loss.backward()
    assert torch.allclose(micro_logits.grad, expected_gradient, atol=1.0e-7, rtol=1.0e-6)


def test_smoke_reloads_checkpoint_and_logs_exact_epoch_coefficients() -> None:
    summary = runner.read_json(RUN_ROOT / "smoke" / "summary.json")
    assert summary["status"] == "complete"
    assert summary["outer_test_accessed"] is False
    assert summary["excluded_from_formal_summary"] is True
    assert summary["checkpoint_reload_passed"] is True
    assert summary["paired_batch_order_audit"]["inner_epochs"] == 2
    fits = [
        runner.read_json(RUN_ROOT / "smoke" / "fits" / pipeline / "fit_summary.json")
        for pipeline in runner.PIPELINES
    ]
    for fit in fits:
        assert fit["checkpoint_reload_max_probability_difference"] == 0.0
        assert fit["validation_animals"] == 17
        for epoch in fit["history"]:
            audit = epoch["effective_coefficients"]
            assert audit["processed_calls"] == 517
            assert audit["unique_processed_calls"] == 517
            target = fit["target_weight_audit"]
            for label in runner.LABEL_NAMES:
                assert audit["effective_coefficient_by_class"][label] == pytest.approx(
                    target["weight_by_class"][label] / 32.0
                )
    for epoch_index in range(2):
        first_hash = fits[0]["history"][epoch_index]["effective_coefficients"][
            "batch_order_sha256"
        ]
        second_hash = fits[1]["history"][epoch_index]["effective_coefficients"][
            "batch_order_sha256"
        ]
        assert first_hash == second_hash


def test_evaluation_completes_the_locked_matrix_and_pairing_audit() -> None:
    summary = runner.read_json(RUN_ROOT / "evaluation" / "summary.json")
    assert summary["status"] == "complete"
    assert summary["completed_outer_fits"] == 72
    assert sum(len(rows) for rows in summary["complete_oof"].values()) == 18
    assert len(summary["paired_C1_vs_C0"]) == 9
    assert all(
        row["n"] == 111
        for rows in summary["complete_oof"].values()
        for row in rows
    )
    batch_audit = summary["paired_batch_order_audit"]
    assert batch_audit["pairs"] == 36
    assert batch_audit["all_common_epoch_hashes_match"] is True


def test_every_formal_epoch_logs_the_exact_global_effective_coefficients() -> None:
    fit_paths = sorted((RUN_ROOT / "evaluation" / "fits").rglob("fit_summary.json"))
    assert len(fit_paths) == 72
    for path in fit_paths:
        fit = runner.read_json(path)
        for phase in ("inner", "outer"):
            target = fit[phase]["target_weight_audit"]
            for epoch in fit[phase]["history"]:
                audit = epoch["effective_coefficients"]
                assert audit["processed_calls"] == audit["unique_processed_calls"]
                assert audit["effective_coefficient_total"] == pytest.approx(
                    target["weight_total"] / 32.0, rel=1.0e-6
                )
                for label in runner.LABEL_NAMES:
                    assert audit["effective_coefficient_by_class"][label] == pytest.approx(
                        target["weight_by_class"][label] / 32.0, rel=1.0e-6
                    )
                assert set(audit["effective_coefficient_by_cat"]) == set(
                    target["weight_by_cat"]
                )
                for cat_id, target_weight in target["weight_by_cat"].items():
                    assert audit["effective_coefficient_by_cat"][cat_id] == pytest.approx(
                        target_weight / 32.0, rel=1.0e-6
                    )


def test_corrected_global_weighting_result_record_matches_run_summary() -> None:
    summary = runner.read_json(RUN_ROOT / "evaluation" / "summary.json")
    result = runner.read_json(RESULTS_PATH)
    c0 = summary["aggregate"]["C0_global_class_balanced"]
    c1 = summary["aggregate"]["C1_global_cat_and_class_balanced"]
    assert result["aggregate"]["C0_global_class_balanced"][
        "macro_f1_mean"
    ] == pytest.approx(c0["macro_f1_mean"])
    assert result["aggregate"]["C1_global_cat_and_class_balanced"][
        "macro_f1_mean"
    ] == pytest.approx(c1["macro_f1_mean"])
    assert result["aggregate"]["C1_minus_C0"]["macro_f1_mean"] == pytest.approx(
        summary["paired_summary"]["mean_C1_minus_C0_macro_f1"]
    )
    assert result["paired_summary"]["positive_comparisons"] == 3
    assert result["stage_decision"]["rule_category"] == "no_improvement_evidence"
    assert result["provenance_sha256"]["evaluation_summary"] == (
        "46affeead827f172eb9acd9dae0ffc26dac7a5d4e9ee5866d0c7f42bc0cfbbe7"
    )
    assert result["provenance_sha256"]["raw_prediction_aggregate"] == (
        "14f01ff60d0a5cde03ef6156af815308291717849f555f1ba66bd31dbc9dc568"
    )
