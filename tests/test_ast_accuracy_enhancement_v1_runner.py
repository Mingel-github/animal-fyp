from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_ast_accuracy_enhancement_v1 as runner  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_ast_accuracy_enhancement_v1.json"
)
RUNNER_PATH = REPO_ROOT / "scripts" / "run_meowagenet_ast_accuracy_enhancement_v1.py"
RUN_ROOT = REPO_ROOT / "runs" / "meowagenet_ast_accuracy_enhancement_v1"
RESULT_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_ast_accuracy_enhancement_v1_results.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_freezes_the_four_factorial_pipelines() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == "meowagenet-ast-accuracy-enhancement-v1"
    assert list(protocol["pipelines"]) == list(runner.PIPELINES)
    assert protocol["fixed_training"]["head_and_fusion_learning_rate"] == 0.006
    assert protocol["scalar_fusion"]["extra_trainable_parameters_vs_A0"] == 12
    assert protocol["exploratory_evaluation"]["total_fits"] == 48
    assert protocol["stage_decision"]["current_scope"] == "seed 17 only"


def test_runner_separates_diagnosis_smoke_lock_and_evaluation() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert {
        "run_diagnostics",
        "run_smoke",
        "verify_execution_lock",
        "run_evaluation",
        "aggregate_evaluation",
        "main",
    } <= function_names
    assert 'choices=("diagnose", "smoke", "evaluate")' in source
    assert "include_test=False" in source
    assert "include_test=True" in source
    assert "locked_for_seed17_exploratory_evaluation" in source


def test_cat_balanced_weights_equalize_class_and_cat_totals() -> None:
    store = runner.idea019.load_feature_store()
    roles = pd.read_csv(runner.ROLES_PATH, dtype={"cat_id": str})
    indices = runner.fold_indices(store, roles, repeat=0, outer_fold=0, include_test=False)
    train = indices["train"]
    weights = runner.cat_balanced_call_weights(store.labels, store.cat_ids, train)
    total = float(weights[train].sum())
    for class_index in range(3):
        class_calls = train[store.labels[train] == class_index]
        assert float(weights[class_calls].sum()) == pytest.approx(total / 3.0)
        cat_totals = [
            float(weights[class_calls[store.cat_ids[class_calls] == cat_id]].sum())
            for cat_id in np.unique(store.cat_ids[class_calls])
        ]
        assert max(cat_totals) == pytest.approx(min(cat_totals), rel=1.0e-6)
    assert weights[train].mean() == pytest.approx(1.0)


def test_scalar_fusion_starts_as_uniform_layer_average() -> None:
    model = runner.ScalarFusionClassifier(nn.Identity())
    instances = torch.arange(2 * 12 * 768, dtype=torch.float32).reshape(2, 12, 768)
    output = model(instances, torch.empty(0, dtype=torch.long), 2)
    assert torch.allclose(output, instances.mean(dim=1), atol=1.0e-5)
    assert model.layer_weights() == pytest.approx([1.0 / 12.0] * 12)


def test_current_protocol_dependencies_validate() -> None:
    protocol = runner.read_json(PROTOCOL_PATH)
    runner.verify_protocol(protocol)


def test_seed17_matrix_has_48_fits_and_12_complete_oof_evaluations() -> None:
    result = runner.read_json(RESULT_PATH)
    summary = runner.read_json(RUN_ROOT / "evaluation" / "summary.json")
    assert result["status"] == "seed17_core_matrix_complete"
    assert summary["status"] == "complete"
    assert summary["completed_fits"] == 48
    assert len(list((RUN_ROOT / "evaluation" / "fits").rglob("fit_summary.json"))) == 48
    for pipeline in runner.PIPELINES:
        assert len(summary["complete_oof"][pipeline]) == 3
        assert all(row["n"] == 111 for row in summary["complete_oof"][pipeline])
    assert result["run_artifact_audit"]["versioned_json_files"] == 60
    assert len(list(RUN_ROOT.rglob("*.json"))) == 60


def test_cat_balancing_is_the_seed17_leading_candidate() -> None:
    result = runner.read_json(RESULT_PATH)
    aggregate = result["aggregate"]
    assert aggregate["A0_final_class_balanced"]["macro_f1_mean"] == pytest.approx(
        0.7487952289623951
    )
    assert aggregate["A1_final_cat_balanced"]["macro_f1_mean"] == pytest.approx(
        0.7764675415415373
    )
    comparison = result["paired_vs_A0"]["A1_final_cat_balanced"]
    assert comparison["mean_macro_f1_difference"] == pytest.approx(
        0.027672312579142228
    )
    assert comparison["positive_repeats"] == 3
    assert result["stage_decision"]["expansion_condition_met"] is True
    assert result["stage_decision"]["leading_pipeline"] == "A1_final_cat_balanced"


def test_a0_reproduces_historical_tuned_ast_and_lock_hashes_match() -> None:
    result = runner.read_json(RESULT_PATH)
    summary = runner.read_json(RUN_ROOT / "evaluation" / "summary.json")
    assert summary["a0_vs_historical_tuned_ast"]["macro_f1_differences"] == [
        0.0,
        0.0,
        0.0,
    ]
    paths = {
        "protocol": PROTOCOL_PATH,
        "runner": RUNNER_PATH,
        "roles": runner.ROLES_PATH,
        "execution_lock": RUN_ROOT / "execution_lock.json",
        "diagnostics_summary": RUN_ROOT / "diagnostics" / "summary.json",
        "smoke_summary": RUN_ROOT / "smoke" / "summary.json",
        "evaluation_summary": RUN_ROOT / "evaluation" / "summary.json",
        "evaluation_run_summary": RUN_ROOT / "evaluation" / "run_summary.json",
    }
    for name, path in paths.items():
        assert path.is_file(), name
        assert sha256(path) == result["provenance_sha256"][name]
    lock = runner.read_json(RUN_ROOT / "execution_lock.json")
    assert lock["runner_sha256"] == sha256(RUNNER_PATH)
    assert lock["protocol_sha256"] == sha256(PROTOCOL_PATH)
