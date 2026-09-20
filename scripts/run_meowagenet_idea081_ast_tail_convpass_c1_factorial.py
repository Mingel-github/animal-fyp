"""IDEA-081: literature-faithful final-block ConvPass x fixed C1 factorial.

Only the CPU preflight is authorized by the preregistration.  The formal 144-fit
path remains GPU-gated and never reads outer-test predictions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_idea079_spatial_patch_adapter as base  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea081_ast_tail_convpass_c1_factorial_v1.json"
)
PIPELINES = (
    "A0_frozen_tail",
    "C1_bounded_age",
    "V1_tail_full_convpass",
    "CV1_C1_plus_tail_full_convpass",
)
BASE_SEEDS = (9217, 7339, 4211)
PATCH_GRID = (12, 12)
PATCH_TOKENS = 144
SPECIAL_TOKENS = 2
TOTAL_TOKENS = 146
HIDDEN_SIZE = 768
CONVPASS_WIDTH = 8
CONVPASS_SCALE = 0.1
CONVPASS_DROPOUT = 0.1
CONVPASS_PARAMETERS_EACH = 13_648
CONVPASS_PARAMETERS = 27_296
CONVPASS_MACS_EACH = 1_878_144
CONVPASS_MACS = 3_756_288
HEAD_PARAMETERS = 99_075
C1_PARAMETERS = 9_068
C1_HIDDEN = 60
C1_CAP = 0.25
RMS_EPSILON = 1.0e-8
EXPECTED_TRAINABLE_PARAMETERS = {
    PIPELINES[0]: HEAD_PARAMETERS,
    PIPELINES[1]: HEAD_PARAMETERS + C1_PARAMETERS,
    PIPELINES[2]: HEAD_PARAMETERS + CONVPASS_PARAMETERS,
    PIPELINES[3]: HEAD_PARAMETERS + C1_PARAMETERS + CONVPASS_PARAMETERS,
}
LOCKED_STATUS = "locked_before_formal_evaluation"


read_json = base.read_json
write_json = base.write_json
sha256 = base.sha256
repo_relative = base.repo_relative
full_seed = base.full_seed
trainable_parameter_count = base.trainable_parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("cpu-preflight", "run"), required=True)
    parser.add_argument(
        "--output-subdir",
        default="meowagenet_idea081_ast_tail_convpass_c1_factorial_v1",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def has_c1(pipeline: str) -> bool:
    return pipeline in (PIPELINES[1], PIPELINES[3])


def has_convpass(pipeline: str) -> bool:
    return pipeline in (PIPELINES[2], PIPELINES[3])


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _states_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


class QuickGELU(nn.Module):
    """Activation used by the authors' released ConvPass implementation."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.sigmoid(1.702 * value)


class ConvPass2D(nn.Module):
    """Official ConvPass topology adapted only for AST's 12x12 grid and two specials."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE, width: int = CONVPASS_WIDTH) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.width = int(width)
        self.adapter_down = nn.Linear(hidden_size, width)
        self.adapter_conv = nn.Conv2d(width, width, 3, stride=1, padding=1)
        self.adapter_up = nn.Linear(width, hidden_size)
        self.act = QuickGELU()
        self.dropout = nn.Dropout(CONVPASS_DROPOUT)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_conv.weight)
        with torch.no_grad():
            self.adapter_conv.weight[:, :, 1, 1].copy_(torch.eye(width))
        nn.init.zeros_(self.adapter_conv.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[1:] != (
            TOTAL_TOKENS,
            self.hidden_size,
        ):
            raise RuntimeError(f"IDEA-081 token geometry changed: {tuple(hidden_states.shape)}")
        batch = hidden_states.shape[0]
        reduced = self.act(self.adapter_down(hidden_states))
        specials = reduced[:, :SPECIAL_TOKENS].reshape(
            batch * SPECIAL_TOKENS, 1, 1, self.width
        ).permute(0, 3, 1, 2)
        specials = self.adapter_conv(specials).permute(0, 2, 3, 1).reshape(
            batch, SPECIAL_TOKENS, self.width
        )
        patches = reduced[:, SPECIAL_TOKENS:].reshape(
            batch, PATCH_GRID[0], PATCH_GRID[1], self.width
        ).permute(0, 3, 1, 2)
        patches = self.adapter_conv(patches).permute(0, 2, 3, 1).reshape(
            batch, PATCH_TOKENS, self.width
        )
        merged = torch.cat((specials, patches), dim=1)
        return self.adapter_up(self.dropout(self.act(merged)))


class FrozenASTTailWithConvPass(nn.Module):
    """Replay frozen block 12, optionally adding full two-branch ConvPass."""

    def __init__(self, enabled: bool) -> None:
        super().__init__()
        block12, final_layernorm = base._tail_template()
        self.block12 = copy.deepcopy(block12)
        self.final_layernorm = copy.deepcopy(final_layernorm)
        for parameter in self.block12.parameters():
            parameter.requires_grad = False
        for parameter in self.final_layernorm.parameters():
            parameter.requires_grad = False
        self.adapter_attn = ConvPass2D() if enabled else None
        self.adapter_mlp = ConvPass2D() if enabled else None

    def forward(self, tokens_after_block11: torch.Tensor) -> torch.Tensor:
        norm1 = self.block12.layernorm_before(tokens_after_block11)
        attention = self.block12.attention(norm1, None, output_attentions=False)[0]
        hidden = attention + tokens_after_block11
        if self.adapter_attn is not None:
            hidden = hidden + CONVPASS_SCALE * self.adapter_attn(norm1)
        norm2 = self.block12.layernorm_after(hidden)
        output = self.block12.output(self.block12.intermediate(norm2), hidden)
        if self.adapter_mlp is not None:
            output = output + CONVPASS_SCALE * self.adapter_mlp(norm2)
        sequence = self.final_layernorm(output)
        return (sequence[:, 0] + sequence[:, 1]) / 2.0


class FactorialHead(nn.Module):
    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        age_train: np.ndarray,
        dropout: float,
        c1_enabled: bool,
    ) -> None:
        super().__init__()
        safe_scale = np.where(scale > 1.0e-12, scale, 1.0).astype(np.float32)
        self.register_buffer("feature_mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("feature_scale", torch.from_numpy(safe_scale))
        self.ast_linear = nn.Linear(HIDDEN_SIZE, 128)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(128, 3)
        self.c1_enabled = bool(c1_enabled)
        if self.c1_enabled:
            median = np.nanmedian(age_train, axis=0).astype(np.float32)
            imputed = np.where(np.isfinite(age_train), age_train, median)
            age_mean = imputed.mean(axis=0).astype(np.float32)
            age_scale = imputed.std(axis=0).astype(np.float32)
            age_scale = np.where(age_scale > 1.0e-12, age_scale, 1.0).astype(np.float32)
            self.register_buffer("age_median", torch.from_numpy(median))
            self.register_buffer("age_mean", torch.from_numpy(age_mean))
            self.register_buffer("age_scale", torch.from_numpy(age_scale))
            self.age_hidden = nn.Linear(age_train.shape[1], C1_HIDDEN)
            self.age_output = nn.Linear(C1_HIDDEN, 128)
            nn.init.zeros_(self.age_output.weight)
            nn.init.zeros_(self.age_output.bias)
        else:
            self.age_hidden = None
            self.age_output = None

    def shared_state(self) -> dict[str, torch.Tensor]:
        prefixes = ("ast_linear.", "batch_norm.", "output.", "feature_")
        return {
            key: value.detach().cpu().clone()
            for key, value in self.state_dict().items()
            if key.startswith(prefixes)
        }

    def age_state(self) -> dict[str, torch.Tensor]:
        prefixes = ("age_hidden.", "age_output.", "age_median", "age_mean", "age_scale")
        return {
            key: value.detach().cpu().clone()
            for key, value in self.state_dict().items()
            if key.startswith(prefixes)
        }

    def forward(self, embeddings: torch.Tensor, age_features: torch.Tensor) -> torch.Tensor:
        normalized = (embeddings - self.feature_mean) / self.feature_scale
        hidden = self.relu(self.ast_linear(normalized))
        if self.c1_enabled:
            if self.age_hidden is None or self.age_output is None:
                raise RuntimeError("IDEA-081 C1 branch missing")
            imputed = torch.where(torch.isfinite(age_features), age_features, self.age_median)
            standardized = (imputed - self.age_mean) / self.age_scale
            context = torch.nn.functional.gelu(self.age_hidden(standardized))
            anchor = torch.sqrt(hidden.float().square().mean(dim=1, keepdim=True) + RMS_EPSILON)
            anchor = anchor.detach().to(hidden.dtype)
            hidden = hidden + C1_CAP * anchor * torch.tanh(self.age_output(context))
        hidden = self.dropout(self.batch_norm(hidden))
        return self.output(hidden)


class FactorialClassifier(nn.Module):
    def __init__(self, pipeline: str, head: FactorialHead, tail: FrozenASTTailWithConvPass) -> None:
        super().__init__()
        if pipeline not in PIPELINES:
            raise ValueError(pipeline)
        self.pipeline = pipeline
        self.head = head
        self.tail = tail

    def segment_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.tail(tokens)

    def forward(
        self,
        tokens: torch.Tensor,
        segment_to_call: torch.Tensor,
        call_count: int,
        age_features: torch.Tensor,
    ) -> torch.Tensor:
        segments = self.segment_embeddings(tokens)
        calls = torch.zeros(
            (call_count, HIDDEN_SIZE), dtype=segments.dtype, device=segments.device
        )
        calls.index_add_(0, segment_to_call, segments)
        counts = torch.bincount(segment_to_call, minlength=call_count).to(segments.dtype)
        if torch.any(counts == 0):
            raise RuntimeError("IDEA-081 batch contains a call with no segment")
        return self.head(calls / counts[:, None], age_features)

    def convpass_state(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.tail.state_dict().items()
            if key.startswith(("adapter_attn.", "adapter_mlp."))
        }

    def audit(self) -> dict[str, Any]:
        conv_parameters = sum(
            parameter.numel()
            for name, parameter in self.tail.named_parameters()
            if name.startswith(("adapter_attn.", "adapter_mlp.")) and parameter.requires_grad
        )
        age_parameters = sum(
            parameter.numel()
            for name, parameter in self.head.named_parameters()
            if name.startswith(("age_hidden.", "age_output.")) and parameter.requires_grad
        )
        return {
            "pipeline": self.pipeline,
            "trainable_parameters": trainable_parameter_count(self),
            "convpass_parameters": int(conv_parameters),
            "C1_parameters": int(age_parameters),
            "frozen_tail_trainable_parameters": int(
                sum(
                    parameter.numel()
                    for name, parameter in self.tail.named_parameters()
                    if not name.startswith(("adapter_attn.", "adapter_mlp."))
                    and parameter.requires_grad
                )
            ),
        }


def build_model(
    pipeline: str,
    protocol: dict[str, Any],
    final_mean: np.ndarray,
    final_scale: np.ndarray,
    age_train: np.ndarray,
    seed: int,
) -> FactorialClassifier:
    offsets = protocol["fixed_training"]["initialization_seed_offsets"]
    set_seed(seed + int(offsets["head"]))
    head = FactorialHead(
        final_mean,
        final_scale,
        age_train,
        float(protocol["fixed_training"]["dropout"]),
        has_c1(pipeline),
    )
    # Reset each factor's RNG independently so corresponding factorial states match.
    if has_c1(pipeline):
        set_seed(seed + int(offsets["age"]))
        nn.init.kaiming_uniform_(head.age_hidden.weight, a=np.sqrt(5))
        if head.age_hidden.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(head.age_hidden.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(head.age_hidden.bias, -bound, bound)
        nn.init.zeros_(head.age_output.weight)
        nn.init.zeros_(head.age_output.bias)
    set_seed(seed + int(offsets["convpass"]))
    tail = FrozenASTTailWithConvPass(has_convpass(pipeline))
    return FactorialClassifier(pipeline, head, tail)


def load_age_features(protocol: dict[str, Any], call_ids: np.ndarray) -> np.ndarray:
    data = protocol["data"]
    path = REPO_ROOT / data["age_feature_path"]
    if not path.is_file() or sha256(path) != data["age_feature_sha256"]:
        raise RuntimeError("IDEA-081 age-feature checksum mismatch")
    summary_path = REPO_ROOT / data["age_feature_summary_path"]
    if (
        not summary_path.is_file()
        or sha256(summary_path) != data["age_feature_summary_sha256"]
    ):
        raise RuntimeError("IDEA-081 age-feature summary checksum mismatch")
    with np.load(path, allow_pickle=False) as loaded:
        feature_ids = loaded["call_ids"].astype(str)
        features = loaded["features"].astype(np.float32)
        names = loaded["feature_names"].astype(str)
    if not np.array_equal(feature_ids, call_ids.astype(str)):
        raise RuntimeError("IDEA-081 age-feature call order changed")
    if features.shape != (len(call_ids), 20) or len(names) != 20:
        raise RuntimeError("IDEA-081 age-feature geometry changed")
    return features


def literature_source_audit(protocol: dict[str, Any]) -> dict[str, Any]:
    checked = []
    for source in protocol["literature_lock"]["local_pdfs"]:
        path = Path(source["path"])
        if not path.is_file() or sha256(path) != source["sha256"]:
            raise RuntimeError(f"IDEA-081 literature PDF checksum mismatch: {path}")
        checked.append({"path": path.as_posix(), "sha256": source["sha256"]})
    return {"local_pdfs_verified": checked, "count": len(checked)}


def load_token_store(protocol: dict[str, Any]) -> tuple[base.TokenStore, dict[str, Any]]:
    """Use IDEA-079's audited reader without inheriting its protocol-status name."""

    bridge = copy.deepcopy(protocol)
    bridge["status"] = base.LOCKED_STATUS
    return base.load_token_store(bridge)


def _all_full_seeds(bases: list[int] | tuple[int, ...], folds: int = 4) -> set[int]:
    return {
        full_seed(int(base_seed), repeat, fold)
        for base_seed in bases
        for repeat in range(3)
        for fold in range(folds)
    }


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea081-ast-tail-convpass-c1-factorial-v1":
        raise RuntimeError("Unexpected IDEA-081 protocol")
    if protocol.get("status") != LOCKED_STATUS:
        raise RuntimeError("IDEA-081 protocol is not locked")
    model = protocol["model"]
    if tuple(model["pipelines"]) != PIPELINES or tuple(model["base_seeds"]) != BASE_SEEDS:
        raise RuntimeError("IDEA-081 pipeline or seed bank changed")
    if model["repeats"] != [0, 1, 2] or model["folds"] != [0, 1, 2, 3]:
        raise RuntimeError("IDEA-081 repeat/fold scope changed")
    if model["outer_test_predictions"] is not False or int(model["total_fits"]) != 144:
        raise RuntimeError("IDEA-081 fit scope changed")
    if model["trainable_parameters"] != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("IDEA-081 trainable parameter lock changed")
    topology = protocol["convpass"]
    expected = {
        "target_block_one_based": 12,
        "parallel_to": ["MSA", "MLP"],
        "bottleneck_width": CONVPASS_WIDTH,
        "scale": CONVPASS_SCALE,
        "dropout": CONVPASS_DROPOUT,
        "parameters": CONVPASS_PARAMETERS,
        "mac_per_segment": CONVPASS_MACS,
    }
    if any(topology.get(key) != value for key, value in expected.items()):
        raise RuntimeError("IDEA-081 ConvPass topology changed")
    digest = hashlib.sha256(model["seed_derivation_text"].encode("utf-8")).hexdigest()
    if digest != model["seed_derivation_sha256"]:
        raise RuntimeError("IDEA-081 seed digest changed")
    derived = _all_full_seeds(BASE_SEEDS)
    excluded = _all_full_seeds(model["excluded_base_seeds_IDEA065_through_IDEA080"])
    if len(derived) != 36 or derived & excluded:
        raise RuntimeError("IDEA-081 full seeds collide")
    dependencies = protocol["dependencies"]
    for path_key, hash_key in (
        ("literature_report_path", "literature_report_sha256"),
        ("plan_path", "plan_sha256"),
        ("runner_path", "runner_sha256"),
        ("tests_path", "tests_sha256"),
    ):
        path = REPO_ROOT / dependencies[path_key]
        if not path.is_file() or sha256(path) != dependencies[hash_key]:
            raise RuntimeError(f"IDEA-081 dependency checksum mismatch: {path}")


def topology_audit() -> dict[str, Any]:
    module = ConvPass2D().eval()
    parameters = trainable_parameter_count(module)
    if parameters != CONVPASS_PARAMETERS_EACH:
        raise RuntimeError(f"IDEA-081 ConvPass parameter mismatch: {parameters}")
    center = module.adapter_conv.weight[:, :, 1, 1]
    off_center = module.adapter_conv.weight.detach().clone()
    off_center[:, :, 1, 1] = 0
    if not torch.equal(center, torch.eye(CONVPASS_WIDTH)) or torch.count_nonzero(off_center):
        raise RuntimeError("IDEA-081 official identity-center conv init changed")
    impulse = torch.zeros((1, TOTAL_TOKENS, HIDDEN_SIZE))
    with torch.no_grad():
        module.adapter_down.weight.zero_()
        module.adapter_down.bias.zero_()
        module.adapter_down.weight[0, 0] = 1.0
        module.adapter_conv.weight.zero_()
        module.adapter_conv.bias.zero_()
        module.adapter_conv.weight[0, 0, :, :] = 1.0
        module.adapter_up.weight.zero_()
        module.adapter_up.bias.zero_()
        module.adapter_up.weight[0, 0] = 1.0
        impulse[0, SPECIAL_TOKENS + 5 * PATCH_GRID[1] + 5, 0] = 1.0
        output = module(impulse)
    patch_support = torch.nonzero(output[0, SPECIAL_TOKENS:, 0], as_tuple=False).flatten().tolist()
    expected_support = sorted(
        frequency * PATCH_GRID[1] + time
        for frequency in (4, 5, 6)
        for time in (4, 5, 6)
    )
    if sorted(patch_support) != expected_support:
        raise RuntimeError("IDEA-081 12x12 local support/order audit failed")
    special_probe = torch.zeros_like(impulse)
    special_probe[0, 0, 0] = 1.0
    with torch.no_grad():
        special_output = module(special_probe)
    nonzero_tokens = torch.nonzero(special_output[0, :, 0], as_tuple=False).flatten().tolist()
    if nonzero_tokens != [0]:
        raise RuntimeError("IDEA-081 special tokens are not processed independently")
    return {
        "parameters_each": parameters,
        "parameters_two_branches": 2 * parameters,
        "dense_conv_weight_shape": list(module.adapter_conv.weight.shape),
        "patch_impulse_support_flat_indices": sorted(patch_support),
        "special_zero_token_nonzero_outputs": nonzero_tokens,
        "patch_grid": list(PATCH_GRID),
        "patch_flatten_order": "frequency-major_then_time-minor",
    }


def initialization_audit(
    protocol: dict[str, Any],
    store: base.TokenStore,
    age_features: np.ndarray,
    train_indices: np.ndarray,
    probe_indices: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    final_train = store.frozen_embeddings[train_indices]
    models = {
        pipeline: build_model(
            pipeline,
            protocol,
            final_train.mean(axis=0),
            final_train.std(axis=0),
            age_features[train_indices],
            seed,
        ).eval()
        for pipeline in PIPELINES
    }
    parameters = {pipeline: trainable_parameter_count(model) for pipeline, model in models.items()}
    if parameters != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(f"IDEA-081 trainable parameter mismatch: {parameters}")
    head_states = {pipeline: model.head.shared_state() for pipeline, model in models.items()}
    if not all(_states_equal(head_states[PIPELINES[0]], state) for state in head_states.values()):
        raise RuntimeError("IDEA-081 common head states differ")
    age_equal = _states_equal(
        models[PIPELINES[1]].head.age_state(), models[PIPELINES[3]].head.age_state()
    )
    conv_equal = _states_equal(
        models[PIPELINES[2]].convpass_state(), models[PIPELINES[3]].convpass_state()
    )
    if not age_equal or not conv_equal:
        raise RuntimeError("IDEA-081 corresponding factorial states differ")
    rows = [int(index) for index in probe_indices[: min(4, len(probe_indices))]]
    segment_indices = np.concatenate([store.call_segment_indices[index] for index in rows])
    segment_to_call = np.concatenate(
        [np.full(len(store.call_segment_indices[index]), local, dtype=np.int64) for local, index in enumerate(rows)]
    )
    tokens = torch.from_numpy(np.asarray(store.tokens[segment_indices], dtype=np.float32))
    mapping = torch.from_numpy(segment_to_call)
    ages = torch.from_numpy(age_features[rows])
    with torch.no_grad():
        logits = {
            pipeline: model(tokens, mapping, len(rows), ages).cpu()
            for pipeline, model in models.items()
        }
    differences = {
        pipeline: float(torch.max(torch.abs(value - logits[PIPELINES[0]])))
        for pipeline, value in logits.items()
        if pipeline != PIPELINES[0]
    }
    if any(value != 0.0 for value in differences.values()):
        raise RuntimeError(f"IDEA-081 initial logits differ: {differences}")
    return {
        "trainable_parameters": parameters,
        "common_head_state_equal_all_four": True,
        "C1_CV1_age_state_equal": age_equal,
        "V1_CV1_convpass_state_equal": conv_equal,
        "max_initial_logit_difference_vs_A0": differences,
        "A0_V1_exact_initial_logits": differences[PIPELINES[2]] == 0.0,
        "C1_CV1_exact_initial_logits": float(
            torch.max(torch.abs(logits[PIPELINES[1]] - logits[PIPELINES[3]]))
        )
        == 0.0,
        "probe_calls": rows,
        "probe_segments": int(len(segment_indices)),
    }


def gradient_reachability_audit(seed: int) -> dict[str, Any]:
    set_seed(seed)
    module = ConvPass2D().train()
    tokens = torch.randn((2, TOTAL_TOKENS, HIDDEN_SIZE))
    target = torch.randn_like(tokens)
    torch.nn.functional.mse_loss(module(tokens), target).backward()
    zero = {
        "up": float(module.adapter_up.weight.grad.abs().max()),
        "down": float(module.adapter_down.weight.grad.abs().max()),
        "conv": float(module.adapter_conv.weight.grad.abs().max()),
    }
    if zero["up"] <= 0 or zero["down"] != 0 or zero["conv"] != 0:
        raise RuntimeError("IDEA-081 zero-up gradient schedule failed")
    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        module.adapter_up.weight.fill_(1.0e-3)
    torch.nn.functional.mse_loss(module(tokens), target).backward()
    opened = {
        "up": float(module.adapter_up.weight.grad.abs().max()),
        "down": float(module.adapter_down.weight.grad.abs().max()),
        "conv": float(module.adapter_conv.weight.grad.abs().max()),
    }
    if min(opened.values()) <= 0:
        raise RuntimeError("IDEA-081 opened ConvPass is not gradient-reachable")
    return {"at_zero_up": zero, "after_fixed_nonzero_up_probe": opened}


def train_one_epoch(
    model: FactorialClassifier,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    store: base.TokenStore,
    age_features: np.ndarray,
    call_class_weights: torch.Tensor,
    device: torch.device,
    gradient_clip: float,
) -> tuple[float, dict[str, Any]]:
    model.train()
    weighted_total = 0.0
    weight_total = 0.0
    processed_cats: list[str] = []
    processed_calls: list[int] = []
    for cpu_batch in loader:
        batch = base.move_batch(cpu_batch, device)
        call_indices = batch["call_indices"]
        labels = torch.from_numpy(store.labels[call_indices.cpu().numpy()]).to(device)
        ages = torch.from_numpy(age_features[call_indices.cpu().numpy()]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(
                batch["tokens"], batch["segment_to_call"], len(call_indices), ages
            )
            per_call = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            weights = call_class_weights[labels]
            loss = (per_call * weights).sum() / weights.sum()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            gradient_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        weighted_total += float((per_call.detach() * weights).sum())
        weight_total += float(weights.sum())
        processed_cats.extend(str(value) for value in batch["cat_ids"])
        processed_calls.extend(int(value) for value in call_indices.cpu().tolist())
    if len(processed_cats) != len(set(processed_cats)):
        raise RuntimeError("IDEA-081 epoch repeated a cat")
    if len(processed_calls) != len(set(processed_calls)):
        raise RuntimeError("IDEA-081 epoch repeated a call")
    return weighted_total / weight_total, {
        "cats": len(processed_cats),
        "calls": len(processed_calls),
        "cat_order_sha256": hashlib.sha256("\n".join(processed_cats).encode()).hexdigest(),
        "call_coverage_sha256": hashlib.sha256(
            np.sort(np.asarray(processed_calls, dtype="<i8")).tobytes()
        ).hexdigest(),
    }


def predict(
    model: FactorialClassifier,
    store: base.TokenStore,
    age_features: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = base.build_loader(store, indices, batch_size, False, seed)
    model.eval()
    rows = []
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = base.move_batch(cpu_batch, device)
            call_indices = batch["call_indices"].cpu().numpy()
            ages = torch.from_numpy(age_features[call_indices]).to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = model(
                    batch["tokens"], batch["segment_to_call"], len(call_indices), ages
                )
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            for local, call_index in enumerate(call_indices):
                rows.append(
                    {
                        "call_index": int(call_index),
                        "call_id": str(store.call_ids[call_index]),
                        "cat_id": str(store.cat_ids[call_index]),
                        "true_label": int(store.labels[call_index]),
                        **{
                            column: float(probabilities[local, class_index])
                            for class_index, column in enumerate(base.PROBABILITY_COLUMNS)
                        },
                    }
                )
    calls = pd.DataFrame(rows).sort_values("call_index").reset_index(drop=True)
    return base.calls_to_animals(calls), calls


def make_optimizer(model: FactorialClassifier, protocol: dict[str, Any]) -> torch.optim.Optimizer:
    fixed = protocol["fixed_training"]
    groups: list[dict[str, Any]] = []
    if has_convpass(model.pipeline):
        groups.append(
            {
                "params": [
                    parameter
                    for name, parameter in model.tail.named_parameters()
                    if name.startswith(("adapter_attn.", "adapter_mlp."))
                ],
                "lr": float(fixed["convpass_learning_rate"]),
            }
        )
    groups.append({"params": list(model.head.parameters()), "lr": float(fixed["head_learning_rate"])})
    return torch.optim.Adamax(groups, eps=float(fixed["optimizer_epsilon"]))


def fit_inner(
    pipeline: str,
    protocol: dict[str, Any],
    store: base.TokenStore,
    age_features: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    final_train = store.frozen_embeddings[train_indices]
    model = build_model(
        pipeline,
        protocol,
        final_train.mean(axis=0),
        final_train.std(axis=0),
        age_features[train_indices],
        seed,
    ).to(device)
    fixed = protocol["fixed_training"]
    training_seed = seed + int(fixed["post_build_seed_offset"])
    set_seed(training_seed)
    optimizer = make_optimizer(model, protocol)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loader = base.build_loader(
        store, train_indices, int(fixed["cat_batch_size"]), True, seed
    )
    weights = torch.from_numpy(base.class_weights(store.labels[train_indices])).to(device)
    best_loss = float("inf")
    best_epoch = 0
    best_state = _state(model)
    best_animals: pd.DataFrame | None = None
    best_calls: pd.DataFrame | None = None
    history = []
    stale = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(fixed["maximum_epochs"]) + 1):
        train_loss, train_audit = train_one_epoch(
            model,
            loader,
            optimizer,
            scaler,
            store,
            age_features,
            weights,
            device,
            float(fixed["gradient_clip"]),
        )
        animals, calls = predict(
            model,
            store,
            age_features,
            validation_indices,
            int(fixed["cat_batch_size"]) * 2,
            device,
            seed,
        )
        validation_loss = base.animal_cross_entropy(animals)
        metrics = base.animal_metrics(animals)
        history.append(
            {
                "epoch": epoch,
                "train_call_loss": train_loss,
                "train_audit": train_audit,
                "validation_animal_cross_entropy": validation_loss,
                "validation_animal_brier": base.brier(animals),
                "validation_animal_metrics": metrics,
            }
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _state(model)
            best_animals = animals.copy()
            best_calls = calls.copy()
            stale = 0
        else:
            stale += 1
        if stale >= int(fixed["early_stopping_patience"]):
            break
    if best_animals is None or best_calls is None:
        raise RuntimeError("IDEA-081 selected no checkpoint")
    model.load_state_dict(best_state)
    reload_animals, _ = predict(
        model,
        store,
        age_features,
        validation_indices,
        int(fixed["cat_batch_size"]) * 2,
        device,
        seed,
    )
    reload_difference = float(
        np.max(
            np.abs(
                reload_animals[list(base.PROBABILITY_COLUMNS)].to_numpy()
                - best_animals[list(base.PROBABILITY_COLUMNS)].to_numpy()
            )
        )
    )
    return (
        {
            "best_epoch": best_epoch,
            "stopped_epoch": len(history),
            "full_seed": seed,
            "training_seed": training_seed,
            "best_validation_animal_cross_entropy": best_loss,
            "best_validation_animal_brier": base.brier(best_animals),
            "best_validation_animal_metrics": base.animal_metrics(best_animals),
            "checkpoint_reload_max_probability_difference": reload_difference,
            "model": model.audit(),
            "train_seconds": float(time.perf_counter() - started),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "history": history,
            "outer_test_accessed": False,
        },
        best_animals,
        best_calls,
    )


def metric_bundle(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "metrics": base.animal_metrics(frame),
        "cross_entropy": base.animal_cross_entropy(frame),
        "brier": base.brier(frame),
    }


def add_metrics(row: dict[str, Any], bundles: dict[str, Any]) -> None:
    for pipeline in PIPELINES:
        bundle = bundles[pipeline]
        row[f"{pipeline}_macro_f1"] = bundle["metrics"]["macro_f1"]
        row[f"{pipeline}_balanced_accuracy"] = bundle["metrics"]["balanced_accuracy"]
        row[f"{pipeline}_cross_entropy"] = bundle["cross_entropy"]
        row[f"{pipeline}_brier"] = bundle["brier"]
        row[f"{pipeline}_senior_recall"] = bundle["metrics"]["per_class"]["senior"]["recall"]
    a0, c1, v1, cv1 = PIPELINES
    for metric in ("macro_f1", "balanced_accuracy", "senior_recall"):
        row[f"V1_minus_A0_{metric}"] = row[f"{v1}_{metric}"] - row[f"{a0}_{metric}"]
        row[f"CV1_minus_C1_{metric}"] = row[f"{cv1}_{metric}"] - row[f"{c1}_{metric}"]
        row[f"interaction_{metric}"] = row[f"CV1_minus_C1_{metric}"] - row[f"V1_minus_A0_{metric}"]
    for metric in ("cross_entropy", "brier"):
        row[f"V1_minus_A0_{metric}_gain"] = row[f"{a0}_{metric}"] - row[f"{v1}_{metric}"]
        row[f"CV1_minus_C1_{metric}_gain"] = row[f"{c1}_{metric}"] - row[f"{cv1}_{metric}"]
        row[f"interaction_{metric}_gain"] = row[f"CV1_minus_C1_{metric}_gain"] - row[f"V1_minus_A0_{metric}_gain"]


def _gate_conditions(
    name: str,
    seed_repeats: pd.DataFrame,
    split_cells: pd.DataFrame,
    per_seed: dict[str, float],
    per_seed_senior: dict[str, float],
    gate: dict[str, Any],
) -> dict[str, bool]:
    delta = seed_repeats[f"{name}_macro_f1"]
    split = split_cells[f"{name}_macro_f1"]
    ce_gain = seed_repeats[f"{name}_cross_entropy_gain"]
    brier_gain = seed_repeats[f"{name}_brier_gain"]
    return {
        "mean_macro_f1": float(delta.mean()) >= float(gate["minimum_mean_macro_f1"]),
        "positive_base_seeds": sum(value > 0 for value in per_seed.values())
        >= int(gate["minimum_positive_base_seeds"]),
        "positive_seed_repeats": int((delta > 0).sum())
        >= int(gate["minimum_positive_seed_repeats"]),
        "nonnegative_split_cells": int((split >= 0).sum())
        >= int(gate["minimum_nonnegative_split_cells"]),
        "worst_split_cell": float(split.min()) >= float(gate["minimum_worst_split_cell_delta"]),
        "cross_entropy_nonworse": float(ce_gain.mean()) >= 0.0,
        "brier_nonworse": float(brier_gain.mean()) >= 0.0,
        "per_base_seed_senior_recall": all(
            value >= float(gate["minimum_per_base_seed_senior_recall_delta"])
            for value in per_seed_senior.values()
        ),
    }


def aggregate(fits: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    by_key = {
        (fit["pipeline"], fit["base_seed"], fit["repeat"], fit["fold"]): fit
        for fit in fits
    }
    folds: list[dict[str, Any]] = []
    seed_repeats: list[dict[str, Any]] = []
    pooled_all: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
    for base_seed in BASE_SEEDS:
        for repeat in range(3):
            grouped = {pipeline: [] for pipeline in PIPELINES}
            for fold in range(4):
                frames = {
                    pipeline: pd.read_csv(
                        REPO_ROOT
                        / by_key[(pipeline, base_seed, repeat, fold)]["validation_animal_predictions"],
                        dtype={"cat_id": str},
                    )
                    for pipeline in PIPELINES
                }
                row: dict[str, Any] = {"base_seed": base_seed, "repeat": repeat, "fold": fold}
                add_metrics(row, {pipeline: metric_bundle(frame) for pipeline, frame in frames.items()})
                folds.append(row)
                for pipeline, frame in frames.items():
                    tagged = frame.copy()
                    tagged["base_seed"] = base_seed
                    tagged["repeat"] = repeat
                    tagged["fold"] = fold
                    grouped[pipeline].append(tagged)
                    pooled_all[pipeline].append(tagged)
            pooled = {pipeline: pd.concat(parts, ignore_index=True) for pipeline, parts in grouped.items()}
            row = {"base_seed": base_seed, "repeat": repeat}
            add_metrics(row, {pipeline: metric_bundle(frame) for pipeline, frame in pooled.items()})
            seed_repeats.append(row)
    fold_frame = pd.DataFrame(folds)
    sr = pd.DataFrame(seed_repeats)
    comparison_columns = [
        f"{name}_{metric}"
        for name in ("V1_minus_A0", "CV1_minus_C1", "interaction")
        for metric in (
            "macro_f1",
            "balanced_accuracy",
            "senior_recall",
            "cross_entropy_gain",
            "brier_gain",
        )
    ]
    split = fold_frame.groupby(["repeat", "fold"], as_index=False)[comparison_columns].mean()
    pooled_frames = {
        pipeline: pd.concat(parts, ignore_index=True) for pipeline, parts in pooled_all.items()
    }
    per_seed: dict[str, dict[str, float]] = {}
    per_seed_senior: dict[str, dict[str, float]] = {}
    senior_by_seed_pipeline: dict[int, dict[str, float]] = {}
    for seed in BASE_SEEDS:
        senior_by_seed_pipeline[seed] = {
            pipeline: base.animal_metrics(
                frame[frame["base_seed"] == seed]
            )["per_class"]["senior"]["recall"]
            for pipeline, frame in pooled_frames.items()
        }
    for name in ("V1_minus_A0", "CV1_minus_C1", "interaction"):
        per_seed[name] = {
            str(seed): float(sr[sr["base_seed"] == seed][f"{name}_macro_f1"].mean())
            for seed in BASE_SEEDS
        }
    for seed in BASE_SEEDS:
        senior = senior_by_seed_pipeline[seed]
        v1_a0 = senior[PIPELINES[2]] - senior[PIPELINES[0]]
        cv1_c1 = senior[PIPELINES[3]] - senior[PIPELINES[1]]
        values = {
            "V1_minus_A0": v1_a0,
            "CV1_minus_C1": cv1_c1,
            "interaction": cv1_c1 - v1_a0,
        }
        for name, value in values.items():
            per_seed_senior.setdefault(name, {})[str(seed)] = float(value)
    gate = protocol["gate"]
    main = _gate_conditions(
        "V1_minus_A0", sr, split, per_seed["V1_minus_A0"], per_seed_senior["V1_minus_A0"], gate["V1_minus_A0"]
    )
    combination = _gate_conditions(
        "CV1_minus_C1", sr, split, per_seed["CV1_minus_C1"], per_seed_senior["CV1_minus_C1"], gate["CV1_minus_C1"]
    )
    interaction_gate = gate["interaction"]
    interaction = _gate_conditions(
        "interaction", sr, split, per_seed["interaction"], per_seed_senior["interaction"], interaction_gate
    )
    means = {
        metric: {
            pipeline: float(sr[f"{pipeline}_{metric}"].mean()) for pipeline in PIPELINES
        }
        for metric in ("macro_f1", "balanced_accuracy", "cross_entropy", "brier", "senior_recall")
    }
    comparisons = {
        column: {
            "mean": float(sr[column].mean()),
            "sample_sd": float(sr[column].std(ddof=1)),
            "positive": int((sr[column] > 0).sum()),
            "tied": int((sr[column] == 0).sum()),
            "negative": int((sr[column] < 0).sum()),
            "worst": float(sr[column].min()),
            "best": float(sr[column].max()),
        }
        for column in comparison_columns
    }
    return {
        "status": "complete",
        "outer_test_accessed": False,
        "fits": len(fits),
        "seed_repeat_estimates": len(seed_repeats),
        "split_cell_estimates": int(len(split)),
        "fold_results": folds,
        "seed_repeat_results": seed_repeats,
        "split_cell_results": split.to_dict(orient="records"),
        "seed_repeat_equal_weight_means": means,
        "comparisons": comparisons,
        "per_base_seed_macro_f1": per_seed,
        "per_base_seed_senior_recall": per_seed_senior,
        "gate_conditions": {
            "V1_minus_A0": main,
            "CV1_minus_C1": combination,
            "interaction": interaction,
        },
        "V1_minus_A0_gate_passed": bool(all(main.values())),
        "CV1_minus_C1_gate_passed": bool(all(combination.values())),
        "interaction_gate_passed": bool(all(interaction.values())),
        "full_factorial_gate_passed": bool(
            all(main.values()) and all(combination.values()) and all(interaction.values())
        ),
    }


def cpu_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cpu":
        raise RuntimeError("IDEA-081 preflight is CPU-only")
    base.configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    literature = literature_source_audit(protocol)
    source = base.audit_source_fbank(protocol)
    store, cache = load_token_store(protocol)
    age_features = load_age_features(protocol, store.call_ids)
    roles = pd.read_csv(REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    role_cells = base.validate_roles(protocol, store, roles)
    indices = base.fold_indices(store, roles, 0, 0)
    started = time.perf_counter()
    bridge = copy.deepcopy(protocol)
    bridge["status"] = base.LOCKED_STATUS
    reconstructed, reconstruction = base.reconstruct_cached_A0(
        bridge, store, torch.device("cpu")
    )
    initialization = initialization_audit(
        protocol,
        store,
        age_features,
        indices["train"],
        indices["validation"],
        BASE_SEEDS[0],
    )
    return {
        "status": "GO_FOR_FORMAL_GPU_RUN",
        "decision": "GO",
        "reason": "The original full ConvPass topology is exactly expressible inside cached block 12; no approximation is used.",
        "read_only": True,
        "device": "cpu",
        "gpu_used": False,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "source_audit": source,
        "literature_source_audit": literature,
        "cache_audit": cache,
        "role_cells": role_cells,
        "runtime_A0_reconstruction": reconstruction,
        "reconstructed_call_embeddings_shape": list(reconstructed.shape),
        "age_features_shape": list(age_features.shape),
        "topology": topology_audit(),
        "initialization": initialization,
        "gradient_reachability": gradient_reachability_audit(BASE_SEEDS[0]),
        "compute": {
            "convpass_parameters_each": CONVPASS_PARAMETERS_EACH,
            "convpass_parameters_two_branches": CONVPASS_PARAMETERS,
            "convpass_mac_per_segment_each": CONVPASS_MACS_EACH,
            "convpass_mac_per_segment_two_branches": CONVPASS_MACS,
        },
        "base_seeds": list(BASE_SEEDS),
        "unique_full_seeds": 36,
        "formal_fits": 144,
        "cpu_seconds": float(time.perf_counter() - started),
        "outer_test_accessed": False,
    }


def resolve_run_root(output_subdir: str) -> Path:
    runs = (REPO_ROOT / "runs").resolve()
    root = (runs / output_subdir).resolve()
    if runs not in root.parents:
        raise ValueError("--output-subdir must remain below runs")
    return root


def validate_completed_fit(
    fit: dict[str, Any], pipeline: str, base_seed: int, seed: int, repeat: int, fold: int
) -> None:
    expected = {
        "status": "complete",
        "pipeline": pipeline,
        "base_seed": base_seed,
        "full_seed": seed,
        "repeat": repeat,
        "fold": fold,
        "outer_test_accessed": False,
    }
    if any(fit.get(key) != value for key, value in expected.items()):
        raise RuntimeError("IDEA-081 resume fit identity mismatch")
    for prefix in ("validation_animal", "validation_call"):
        path = REPO_ROOT / fit[f"{prefix}_predictions"]
        if not path.is_file() or sha256(path) != fit[f"{prefix}_sha256"]:
            raise RuntimeError(f"IDEA-081 resume prediction mismatch: {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    base.configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    if protocol["execution"].get("current_gpu_allowed") is not True:
        raise RuntimeError("IDEA-081 formal GPU run has not been authorized")
    device = base.resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("IDEA-081 formal 144-fit run is GPU-only")
    store, cache = load_token_store(protocol)
    age_features = load_age_features(protocol, store.call_ids)
    roles = pd.read_csv(REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    base.validate_roles(protocol, store, roles)
    root = resolve_run_root(args.output_subdir)
    preflight_path = root / "cpu_preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("IDEA-081 requires the locked CPU preflight")
    preflight = read_json(preflight_path)
    expected_preflight = {
        "status": "GO_FOR_FORMAL_GPU_RUN",
        "decision": "GO",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "outer_test_accessed": False,
    }
    if any(preflight.get(key) != value for key, value in expected_preflight.items()):
        raise RuntimeError("IDEA-081 CPU preflight is stale")
    root.mkdir(parents=True, exist_ok=True)
    try:
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = None
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "cache_manifest_sha256": protocol["shared_cache"]["cache_manifest_sha256"],
        "git_revision": git_revision,
        "outer_test_accessed": False,
        "model": protocol["model"],
        "cache_audit": cache,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        if not args.resume or read_json(manifest_path) != manifest:
            raise RuntimeError("Existing IDEA-081 run manifest differs")
    else:
        write_json(manifest_path, manifest)
    completed = []
    for base_seed in BASE_SEEDS:
        for repeat in range(3):
            for fold in range(4):
                indices = base.fold_indices(store, roles, repeat, fold)
                seed = full_seed(base_seed, repeat, fold)
                for pipeline in PIPELINES:
                    output = root / "fits" / pipeline / f"base_seed_{base_seed}" / f"repeat_{repeat}" / f"fold_{fold}"
                    summary_path = output / "fit_summary.json"
                    if summary_path.exists():
                        if not args.resume:
                            raise FileExistsError(summary_path)
                        fit = read_json(summary_path)
                        validate_completed_fit(fit, pipeline, base_seed, seed, repeat, fold)
                        completed.append(fit)
                        continue
                    output.mkdir(parents=True, exist_ok=True)
                    audit, animals, calls = fit_inner(
                        pipeline,
                        protocol,
                        store,
                        age_features,
                        indices["train"],
                        indices["validation"],
                        device,
                        seed,
                    )
                    animal_path = output / "validation_animal_predictions.csv"
                    call_path = output / "validation_call_predictions.csv"
                    animals.to_csv(animal_path, index=False, lineterminator="\n")
                    calls.to_csv(call_path, index=False, lineterminator="\n")
                    fit = {
                        "status": "complete",
                        "pipeline": pipeline,
                        "base_seed": base_seed,
                        "full_seed": seed,
                        "repeat": repeat,
                        "fold": fold,
                        "outer_test_accessed": False,
                        "audit": audit,
                        "validation_animal_predictions": repo_relative(animal_path),
                        "validation_animal_sha256": sha256(animal_path),
                        "validation_call_predictions": repo_relative(call_path),
                        "validation_call_sha256": sha256(call_path),
                    }
                    write_json(summary_path, fit)
                    completed.append(fit)
    summary = aggregate(completed, protocol)
    write_json(root / "initial_evaluation_summary.json", summary)
    write_json(
        root / "run_summary.json",
        {
            "status": "complete",
            "completed_fits": len(completed),
            "expected_fits": 144,
            "outer_test_accessed": False,
            "V1_minus_A0_gate_passed": summary["V1_minus_A0_gate_passed"],
            "CV1_minus_C1_gate_passed": summary["CV1_minus_C1_gate_passed"],
            "interaction_gate_passed": summary["interaction_gate_passed"],
            "full_factorial_gate_passed": summary["full_factorial_gate_passed"],
        },
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.stage == "cpu-preflight":
        result = cpu_preflight(args)
        root = resolve_run_root(args.output_subdir)
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "cpu_preflight.json", result)
    else:
        result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
