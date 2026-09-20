from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_meowagenet_idea080_ast_last_block_lora_c1_factorial.py"
PROTOCOL_PATH = REPO_ROOT / "configs" / "protocol" / "meowagenet_idea080_ast_last_block_lora_c1_factorial_v1.json"
SPEC = importlib.util.spec_from_file_location("idea080_runner_test", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
idea080 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = idea080
SPEC.loader.exec_module(idea080)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def store_and_roles(protocol: dict):
    store, _ = idea080.idea078.load_store(protocol)
    roles = pd.read_csv(REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    return store, roles


def test_protocol_and_literature_dependencies_are_frozen(protocol: dict) -> None:
    idea080.verify_protocol(protocol)
    assert protocol["priority_and_claim_boundary"]["dog_results_participate"] is False
    assert protocol["priority_and_claim_boundary"]["outer_test_accessed"] is False
    lock = protocol["literature_lock"]
    assert lock["lora_paper"]["arxiv"] == "2106.09685"
    assert lock["petl_ast_paper"]["arxiv"] == "2312.03694"
    for key in ("petl_ast_paper", "soft_mixture_ast_context", "ast_paper"):
        path = Path(lock[key]["local_pdf_path"])
        assert path.is_file()
        assert sha256(path) == lock[key]["local_pdf_sha256"]


def test_seed_derivation_and_full_seed_scope_are_unique(protocol: dict) -> None:
    model = protocol["model"]
    assert model["seed_derivation_sha256"] == hashlib.sha256(
        model["seed_derivation_text"].encode("utf-8")
    ).hexdigest()
    assert tuple(model["base_seeds"]) == idea080.BASE_SEEDS == (59, 7031, 1855)
    excluded = set(model["excluded_base_seeds_IDEA068_through_IDEA079"])
    assert not excluded.intersection(idea080.BASE_SEEDS)
    full = {
        idea080.full_seed(seed, repeat, fold)
        for seed in idea080.BASE_SEEDS
        for repeat in range(3)
        for fold in range(4)
    }
    assert len(full) == 36


def test_lora_is_rank6_qv_only_with_zero_initial_update() -> None:
    tail = idea080.FactorialASTTail(True, initialization_seed=123)
    audit = tail.audit()
    assert audit["target_scope_exact"] is True
    assert audit["target_block_one_based"] == 12
    assert audit["target_projections"] == ["query", "value"]
    assert audit["rank"] == 6
    assert audit["alpha"] == 6
    assert audit["scaling"] == 1.0
    assert audit["dropout"] == 0.0
    assert audit["trainable_parameters"] == 18_432
    attention = tail.block12.attention.attention
    assert isinstance(attention.query, idea080.LoRALinear)
    assert isinstance(attention.value, idea080.LoRALinear)
    assert isinstance(attention.key, torch.nn.Linear)
    assert torch.count_nonzero(attention.query.lora_A.weight).item() > 0
    assert torch.count_nonzero(attention.value.lora_A.weight).item() > 0
    assert torch.count_nonzero(attention.query.lora_B.weight).item() == 0
    assert torch.count_nonzero(attention.value.lora_B.weight).item() == 0


def test_four_pipeline_parameter_budget_and_paired_initialization(
    protocol: dict, store_and_roles
) -> None:
    store, roles = store_and_roles
    indices = idea080.idea078.fold_indices(store, roles, 0, 0)
    probe = indices["validation"][:8]
    audit = idea080.initialization_audit(
        protocol,
        store,
        indices["train"],
        probe,
        idea080.BASE_SEEDS[0],
        torch.device("cpu"),
    )
    assert audit["trainable_parameters"] == idea080.EXPECTED_PARAMETERS
    assert audit["shared_head_state_equal"] is True
    assert audit["C1_CL1_age_state_equal"] is True
    assert audit["L1_CL1_lora_state_equal"] is True
    assert all(value == 0.0 for value in audit["max_initial_logit_difference_vs_A0"].values())
    assert len(set(audit["initial_loss"].values())) == 1
    assert audit["outer_test_accessed"] is False


def test_zero_impact_branches_have_the_locked_gradient_schedule(
    protocol: dict, store_and_roles
) -> None:
    store, roles = store_and_roles
    indices = idea080.idea078.fold_indices(store, roles, 0, 0)
    audit = idea080.gradient_reachability_audit(
        protocol,
        store,
        indices["train"],
        indices["validation"][:8],
        idea080.BASE_SEEDS[0],
        torch.device("cpu"),
    )
    c1 = audit[idea080.PIPELINES[1]]
    l1 = audit[idea080.PIPELINES[2]]
    cl1 = audit[idea080.PIPELINES[3]]
    assert c1["age_output_max_gradient_at_zero"] > 0
    assert c1["age_hidden_max_gradient_at_zero"] == 0
    assert c1["age_hidden_max_gradient_after_output_probe"] > 0
    assert l1["lora_B_max_gradient_at_zero"] > 0
    assert l1["lora_A_max_gradient_at_zero"] == 0
    assert l1["lora_A_max_gradient_after_B_probe"] > 0
    assert cl1["age_output_max_gradient_at_zero"] > 0
    assert cl1["lora_B_max_gradient_at_zero"] > 0
    assert all(item["frozen_tail_parameters_with_grad"] == 0 for item in audit.values())


def test_c1_residual_obeys_the_pointwise_rms_cap(protocol: dict) -> None:
    mean = np.zeros(768, dtype=np.float32)
    scale = np.ones(768, dtype=np.float32)
    ages = np.zeros((16, 20), dtype=np.float32)
    head = idea080.FactorialHead(
        mean,
        scale,
        ages,
        float(protocol["fixed_training"]["dropout"]),
        True,
        1,
        2,
    ).eval()
    with torch.no_grad():
        head.age_output.weight.fill_(100.0)
        head.age_output.bias.fill_(100.0)
        embeddings = torch.randn(16, 768)
        ast = (embeddings - head.ast_mean) / head.ast_scale
        hidden = head.relu(head.ast_linear(ast))
        imputed = torch.zeros(16, 20)
        context = torch.nn.functional.gelu(head.age_hidden(imputed))
        anchor = torch.sqrt(hidden.square().mean(dim=1, keepdim=True) + idea080.RMS_EPSILON)
        residual = idea080.C1_CAP * anchor * torch.tanh(head.age_output(context))
        ratio = torch.linalg.vector_norm(residual, dim=1) / torch.linalg.vector_norm(hidden, dim=1).clamp_min(1e-12)
    assert float(ratio.max()) <= idea080.C1_CAP + 1.0e-6


def test_factorial_interaction_formula_is_exact() -> None:
    row = {}
    bundles = {
        idea080.PIPELINES[0]: {"metrics": {"macro_f1": 0.70, "balanced_accuracy": 0.71, "per_class": {"senior": {"recall": 0.6}}}, "cross_entropy": 0.8, "brier": 0.4},
        idea080.PIPELINES[1]: {"metrics": {"macro_f1": 0.72, "balanced_accuracy": 0.72, "per_class": {"senior": {"recall": 0.6}}}, "cross_entropy": 0.7, "brier": 0.3},
        idea080.PIPELINES[2]: {"metrics": {"macro_f1": 0.73, "balanced_accuracy": 0.73, "per_class": {"senior": {"recall": 0.6}}}, "cross_entropy": 0.7, "brier": 0.3},
        idea080.PIPELINES[3]: {"metrics": {"macro_f1": 0.77, "balanced_accuracy": 0.74, "per_class": {"senior": {"recall": 0.6}}}, "cross_entropy": 0.6, "brier": 0.2},
    }
    idea080.add_metrics(row, bundles)
    expected = (0.77 - 0.72) - (0.73 - 0.70)
    assert row["interaction_macro_f1"] == pytest.approx(expected)


def test_cpu_preflight_artifact_if_present(protocol: dict) -> None:
    path = REPO_ROOT / protocol["outputs"]["cpu_preflight"]
    if not path.exists():
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["status"] == "GO_FOR_FORMAL_GPU_RUN"
    assert result["device"] == "cpu"
    assert result["gpu_used"] is False
    assert result["outer_test_accessed"] is False
    assert result["expected_fits"] == 144
    assert result["initialization"]["L1_CL1_lora_state_equal"] is True
    assert result["initialization"]["C1_CL1_age_state_equal"] is True


def test_formal_run_requires_explicit_director_authorization() -> None:
    args = Namespace(director_authorized=False)
    with pytest.raises(RuntimeError, match="research-director authorization"):
        idea080.run(args)
