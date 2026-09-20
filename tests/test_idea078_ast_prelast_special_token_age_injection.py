from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts"
    / "run_meowagenet_idea078_ast_prelast_special_token_age_injection.py"
)
EXTRACTOR_PATH = ROOT / "scripts" / "extract_ast_prelast_tokens_idea078.py"
SPEC = importlib.util.spec_from_file_location("idea078_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
idea078 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = idea078
SPEC.loader.exec_module(idea078)


class DummyTail(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens[:, 0] + tokens[:, 1]) / 2

    def train(self, mode: bool = True) -> "DummyTail":
        super().train(False)
        return self


class DummyPredictModel(nn.Module):
    def forward(
        self,
        tokens: torch.Tensor,
        segment_to_local_call: torch.Tensor,
        age_features: torch.Tensor,
    ) -> torch.Tensor:
        del tokens, segment_to_local_call
        return torch.zeros((len(age_features), 3), dtype=torch.float32)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol() -> dict:
    return json.loads(idea078.PROTOCOL_PATH.read_text(encoding="utf-8"))


def build_models(seed: int = 9763):
    rng = np.random.default_rng(78)
    ast_mean = rng.normal(size=768).astype(np.float32)
    ast_scale = np.exp(rng.normal(scale=0.1, size=768)).astype(np.float32)
    age_train = rng.normal(size=(40, 20)).astype(np.float32)
    models = {}
    for pipeline in idea078.PIPELINES:
        idea078.set_seed(seed)
        models[pipeline] = idea078.TokenInjectionClassifier(
            pipeline,
            DummyTail(),
            ast_mean,
            ast_scale,
            age_train,
            dropout=0.0,
            bottleneck=10,
        ).eval()
    return models


def test_protocol_hashes_formula_budget_and_priority_are_locked() -> None:
    value = protocol()
    idea078.verify_protocol(value)
    dependencies = value["dependencies"]
    assert sha256(ROOT / dependencies["plan_path"]) == dependencies["plan_sha256"]
    assert sha256(ROOT / dependencies["cache_extractor_path"]) == dependencies["cache_extractor_sha256"]
    assert sha256(RUNNER_PATH) == dependencies["runner_sha256"]
    assert sha256(Path(__file__)) == dependencies["tests_sha256"]
    assert value["priority_and_claim_boundary"]["dataset_priority"].startswith("MeowAgeNet")
    assert value["model"]["age_bottleneck"] == 10
    assert value["model"]["special_token_indices"] == [0, 1]
    assert value["model"]["patch_tokens_modified"] is False
    assert value["model"]["total_fits"] == 108
    assert value["model"]["primary_A0_T1_fits"] == 72


def test_cache_schema_is_float32_mmap_label_free_and_unique() -> None:
    cache = protocol()["cache"]
    assert cache["shape"] == [843, 146, 768]
    assert cache["dtype"] == "float32"
    assert cache["storage"] == "npy_memory_mappable"
    assert cache["index_allow_pickle"] is False
    assert cache["contains_labels"] is False
    assert cache["contains_roles"] is False
    assert cache["contains_cat_ids"] is False
    assert cache["prelast_block_output_one_based"] == 11
    assert cache["last_block_input_one_based"] == 12
    assert cache["special_token_indices"] == [0, 1]
    assert cache["patch_token_indices"] == [2, 145]
    assert cache["raw_token_bytes"] == 378_095_616
    assert cache["manifest_path"].endswith("cache_manifest.json")
    schema = json.loads(
        (ROOT / protocol()["dependencies"]["cache_schema_path"]).read_text(encoding="utf-8")
    )
    assert schema["index"]["keys"] == [
        "segment_call_indices",
        "segment_counts",
        "call_ids",
        "source_paths",
        "prelast_block_output_one_based",
        "last_block_input_one_based",
        "special_token_indices",
        "patch_token_start",
    ]
    assert schema["ownership"]["duplicate_extraction_allowed"] is False


def test_seed_derivation_and_full_seed_collisions_are_locked() -> None:
    model = protocol()["model"]
    text = model["seed_derivation_text"]
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    candidates = [
        struct.unpack(">I", digest[offset : offset + 4])[0] % 10_000
        for offset in range(0, 12, 4)
    ]
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == model["seed_derivation_sha256"]
    assert candidates == list(idea078.BASE_SEEDS)
    derived = {
        idea078.full_seed(base, repeat, fold)
        for base in idea078.BASE_SEEDS
        for repeat in range(3)
        for fold in range(4)
    }
    assert len(derived) == 36
    excluded = {
        idea078.full_seed(base, repeat, fold)
        for base in model["excluded_base_seeds_IDEA068_through_IDEA077"]
        for repeat in range(3)
        for fold in range(5)
    }
    assert not derived & excluded


def test_parameter_counts_head_pairing_and_zero_effect_initialization() -> None:
    models = build_models()
    counts = {
        name: sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        for name, model in models.items()
    }
    assert counts == idea078.EXPECTED_PARAMETERS
    assert idea078.states_equal(
        idea078.head_state(models[idea078.PIPELINES[0]]),
        idea078.head_state(models[idea078.PIPELINES[1]]),
    )
    assert idea078.states_equal(
        idea078.head_state(models[idea078.PIPELINES[0]]),
        idea078.head_state(models[idea078.PIPELINES[2]]),
    )
    assert idea078.states_equal(
        idea078.injection_state(models[idea078.PIPELINES[1]]),
        idea078.injection_state(models[idea078.PIPELINES[2]]),
    )
    rng = np.random.default_rng(780)
    tokens = torch.from_numpy(rng.normal(size=(7, 146, 768)).astype(np.float32))
    mapping = torch.tensor([0, 0, 1, 2, 2, 2, 3], dtype=torch.int64)
    ages = torch.from_numpy(rng.normal(size=(4, 20)).astype(np.float32))
    logits = {
        name: model(tokens, mapping, ages)
        for name, model in models.items()
    }
    for name in idea078.PIPELINES[1:]:
        torch.testing.assert_close(logits[name], logits[idea078.PIPELINES[0]], rtol=0, atol=0)


def test_T1_modifies_only_two_special_tokens_before_tail() -> None:
    models = build_models()
    t1 = models[idea078.PIPELINES[2]]
    assert t1.age_output is not None
    with torch.no_grad():
        t1.age_output.weight.fill_(0.01)
        t1.age_output.bias.fill_(0.02)
    rng = np.random.default_rng(781)
    tokens = torch.from_numpy(rng.normal(size=(5, 146, 768)).astype(np.float32))
    mapping = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int64)
    ages = torch.from_numpy(rng.normal(size=(2, 20)).astype(np.float32))
    delta = t1.age_delta(ages)[mapping]
    conditioned = tokens.clone()
    conditioned[:, 0:2, :] += delta[:, None, :]
    torch.testing.assert_close(conditioned[:, 2:, :], tokens[:, 2:, :], rtol=0, atol=0)
    torch.testing.assert_close(conditioned[:, 0, :] - tokens[:, 0, :], delta)
    torch.testing.assert_close(conditioned[:, 1, :] - tokens[:, 1, :], delta)


def test_zero_output_layer_gradient_schedule_and_frozen_tail() -> None:
    t1 = build_models()[idea078.PIPELINES[2]]
    rng = np.random.default_rng(782)
    tokens = torch.from_numpy(rng.normal(size=(6, 146, 768)).astype(np.float32))
    mapping = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
    ages = torch.from_numpy(rng.normal(size=(3, 20)).astype(np.float32))
    labels = torch.tensor([0, 1, 2], dtype=torch.int64)
    loss = torch.nn.functional.cross_entropy(t1(tokens, mapping, ages), labels)
    loss.backward()
    assert t1.age_output is not None and t1.age_hidden is not None
    assert float(t1.age_output.weight.grad.abs().max()) > 0
    assert float(t1.age_hidden.weight.grad.abs().max()) == 0
    assert all(parameter.grad is None for parameter in t1.tail.parameters())
    t1.zero_grad(set_to_none=True)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(78)
        t1.age_output.weight.copy_(
            torch.randn(t1.age_output.weight.shape, generator=generator) * 1.0e-4
        )
    torch.nn.functional.cross_entropy(t1(tokens, mapping, ages), labels).backward()
    assert float(t1.age_hidden.weight.grad.abs().max()) > 0


def test_capacity_control_and_gates_are_preregistered() -> None:
    value = protocol()
    assert value["model"]["pipelines"][1] == "P1_postpool_age_injection"
    assert value["model"]["trainable_parameters"]["P1_postpool_age_injection"] == value["model"]["trainable_parameters"]["T1_prelast_special_token_injection"]
    assert value["relationship_to_prior_work"]["not_duplicate_of_A1_C1"] is True
    assert value["gate"]["main"]["minimum_mean_T1_minus_A0"] == 0.005
    assert value["gate"]["main"]["minimum_positive_base_seeds"] == 3
    assert value["gate"]["main"]["minimum_positive_seed_repeats"] == 6
    assert value["gate"]["main"]["minimum_nonnegative_split_cells"] == 8
    assert value["gate"]["main"]["minimum_worst_split_cell_delta"] == -0.03
    assert value["gate"]["mechanism"]["interpretable_only_when_main_passes"] is True
    assert value["gate"]["mechanism"]["minimum_positive_base_seeds"] == 2
    assert value["gate"]["mechanism"]["minimum_positive_seed_repeats"] == 5


def test_canonical_json_is_lf_only() -> None:
    payload = idea078.canonical_json_bytes({"中文": [1, 2], "ok": True})
    assert payload.endswith(b"\n")
    assert b"\r\n" not in payload
    assert json.loads(payload.decode("utf-8"))["ok"] is True


def test_prediction_contract_uses_true_label_for_animal_aggregation() -> None:
    tokens = np.zeros((3, 146, 768), dtype=np.float32)
    mapping = np.arange(3, dtype=np.int64)
    counts = np.ones(3, dtype=np.int64)
    store = idea078.TokenStore(
        tokens,
        mapping,
        counts,
        tuple(np.asarray([index], dtype=np.int64) for index in range(3)),
        np.zeros((3, 768), dtype=np.float32),
        np.zeros((3, 20), dtype=np.float32),
        np.asarray(["call-0", "call-1", "call-2"]),
        np.asarray(["cat-0", "cat-1", "cat-2"]),
        np.asarray([0, 1, 2], dtype=np.int64),
    )
    animals, calls = idea078.predict(
        DummyPredictModel(), store, np.arange(3), 3, torch.device("cpu"), 78
    )
    assert "true_label" in calls.columns
    assert "label" not in calls.columns
    assert animals["true_label"].tolist() == [0, 1, 2]
