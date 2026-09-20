"""Run the preregistered IDEA-079 AST spatial patch-token adapter screen.

The structural preflight is intentionally cache-independent and CPU-only.  Formal
cache preflight and training remain hard-blocked until IDEA-078 publishes the
shared block-11 token-cache manifest and IDEA-079 is amended with its immutable
hashes.  IDEA-079 never extracts a second token cache.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import ASTModel


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea079_spatial_patch_adapter_v1.json"
)
PIPELINES = ("A0_cached_tail", "C1_pointwise_patch", "S1_spatial_patch")
BASE_SEEDS = (8058, 2495, 2473)
PATCH_GRID = (12, 12)
PATCH_TOKENS = 144
SPECIAL_TOKENS = 2
TOTAL_TOKENS = 146
HIDDEN_SIZE = 768
BOTTLENECK = 9
ADAPTER_PARAMETERS = 14_691
HEAD_PARAMETERS = 99_075
EXPECTED_TRAINABLE_PARAMETERS = {
    "A0_cached_tail": HEAD_PARAMETERS,
    "C1_pointwise_patch": HEAD_PARAMETERS + ADAPTER_PARAMETERS,
    "S1_spatial_patch": HEAD_PARAMETERS + ADAPTER_PARAMETERS,
}
LABEL_NAMES = ("kitten", "adult", "senior")
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
PENDING_STATUS = "methods_locked_waiting_for_idea078_cache_manifest"
LOCKED_STATUS = "locked_before_initial_evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("structural-preflight", "cache-preflight", "run"),
        required=True,
    )
    parser.add_argument(
        "--output-subdir",
        default="meowagenet_idea079_spatial_patch_adapter_v1",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def full_seed(base_seed: int, repeat: int, fold: int) -> int:
    return int(base_seed + 10_000 * repeat + 100 * fold)


def trainable_parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


class ClassificationHead(nn.Module):
    def __init__(self, mean: np.ndarray, scale: np.ndarray, dropout: float) -> None:
        super().__init__()
        safe_scale = np.where(scale > 1.0e-12, scale, 1.0).astype(np.float32)
        self.register_buffer("feature_mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("feature_scale", torch.from_numpy(safe_scale))
        self.ast_linear = nn.Linear(HIDDEN_SIZE, 128)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(128, 3)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        normalized = (embeddings - self.feature_mean) / self.feature_scale
        hidden = self.relu(self.ast_linear(normalized))
        hidden = self.dropout(self.batch_norm(hidden))
        return self.output(hidden)


class PatchTokenAdapter(nn.Module):
    """Patch-only residual adapter; the two AST special tokens bypass it."""

    def __init__(self, mode: str, hidden_size: int = HIDDEN_SIZE, width: int = BOTTLENECK) -> None:
        super().__init__()
        if mode not in {"pointwise", "spatial"}:
            raise ValueError(mode)
        self.mode = mode
        self.hidden_size = int(hidden_size)
        self.width = int(width)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.down = nn.Linear(hidden_size, width)
        self.activation_in = nn.GELU()
        if mode == "pointwise":
            self.mixer = nn.Conv2d(width, width, kernel_size=1, bias=True)
        else:
            self.mixer = nn.Conv2d(
                width,
                width,
                kernel_size=3,
                padding=1,
                groups=width,
                bias=True,
            )
        self.activation_out = nn.GELU()
        self.up = nn.Linear(width, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise RuntimeError(f"IDEA-079 expects [batch,tokens,hidden], got {hidden_states.shape}")
        if hidden_states.shape[1:] != (TOTAL_TOKENS, self.hidden_size):
            raise RuntimeError(
                "IDEA-079 token geometry changed: "
                f"{tuple(hidden_states.shape[1:])} != {(TOTAL_TOKENS, self.hidden_size)}"
            )
        special = hidden_states[:, :SPECIAL_TOKENS]
        patches = hidden_states[:, SPECIAL_TOKENS:]
        normalized = self.norm(patches)
        compressed = self.activation_in(self.down(normalized))
        grid = compressed.reshape(
            hidden_states.shape[0], PATCH_GRID[0], PATCH_GRID[1], self.width
        ).permute(0, 3, 1, 2)
        mixed = self.mixer(grid)
        mixed = mixed.permute(0, 2, 3, 1).reshape(
            hidden_states.shape[0], PATCH_TOKENS, self.width
        )
        residual = self.up(self.activation_out(mixed))
        adapted_patches = patches + residual
        return torch.cat((special, adapted_patches), dim=1)


class FrozenASTTail(nn.Module):
    def __init__(self, block12: nn.Module, final_layernorm: nn.Module) -> None:
        super().__init__()
        self.block12 = block12
        self.final_layernorm = final_layernorm
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, tokens_after_block11: torch.Tensor) -> torch.Tensor:
        sequence = self.block12(tokens_after_block11, None, False)[0]
        sequence = self.final_layernorm(sequence)
        return (sequence[:, 0] + sequence[:, 1]) / 2.0


@functools.lru_cache(maxsize=1)
def _tail_template() -> tuple[nn.Module, nn.Module]:
    locked = read_json(REPO_ROOT / "configs" / "protocol" / "meowagenet_locked_v1.json")
    ast_config = locked["ast"]
    model = ASTModel.from_pretrained(
        ast_config["checkpoint"],
        revision=ast_config["revision"],
        cache_dir=REPO_ROOT / "data" / "models" / "huggingface",
        use_safetensors=True,
        local_files_only=True,
    ).cpu().eval()
    if len(model.encoder.layer) != 12 or int(model.config.hidden_size) != HIDDEN_SIZE:
        raise RuntimeError("IDEA-079 AST checkpoint architecture changed")
    block12 = copy.deepcopy(model.encoder.layer[11]).cpu().eval()
    final_layernorm = copy.deepcopy(model.layernorm).cpu().eval()
    del model
    for module in (block12, final_layernorm):
        for parameter in module.parameters():
            parameter.requires_grad = False
    return block12, final_layernorm


def build_frozen_tail() -> FrozenASTTail:
    block12, final_layernorm = _tail_template()
    return FrozenASTTail(copy.deepcopy(block12), copy.deepcopy(final_layernorm))


class CachedTailClassifier(nn.Module):
    def __init__(
        self,
        pipeline: str,
        head: ClassificationHead,
        tail: FrozenASTTail,
    ) -> None:
        super().__init__()
        if pipeline not in PIPELINES:
            raise ValueError(pipeline)
        self.pipeline = pipeline
        self.head = head
        self.tail = tail
        if pipeline == "A0_cached_tail":
            self.adapter: PatchTokenAdapter | None = None
        elif pipeline == "C1_pointwise_patch":
            self.adapter = PatchTokenAdapter("pointwise")
        else:
            self.adapter = PatchTokenAdapter("spatial")

    def segment_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.adapter is not None:
            tokens = self.adapter(tokens)
        return self.tail(tokens)

    def forward(
        self,
        tokens: torch.Tensor,
        segment_to_call: torch.Tensor,
        call_count: int,
    ) -> torch.Tensor:
        segment_embeddings = self.segment_embeddings(tokens)
        call_embeddings = torch.zeros(
            (call_count, HIDDEN_SIZE),
            dtype=segment_embeddings.dtype,
            device=segment_embeddings.device,
        )
        call_embeddings.index_add_(0, segment_to_call, segment_embeddings)
        counts = torch.bincount(segment_to_call, minlength=call_count).to(
            segment_embeddings.dtype
        )
        if torch.any(counts == 0):
            raise RuntimeError("IDEA-079 batch contains a call with no segment")
        return self.head(call_embeddings / counts[:, None])

    def audit(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "trainable_parameters": trainable_parameter_count(self),
            "adapter_parameters": 0 if self.adapter is None else trainable_parameter_count(self.adapter),
            "block12_trainable_parameters": trainable_parameter_count(self.tail),
        }


def build_model(
    pipeline: str,
    protocol: dict[str, Any],
    final_mean: np.ndarray,
    final_scale: np.ndarray,
) -> CachedTailClassifier:
    # The head is built first so every pipeline receives exactly the same head
    # tensors under a common seed.  Training RNG is reset after construction.
    head = ClassificationHead(
        final_mean,
        final_scale,
        float(protocol["fixed_training"]["dropout"]),
    )
    return CachedTailClassifier(pipeline, head, build_frozen_tail())


def shared_head_state(model: CachedTailClassifier) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.head.state_dict().items()
    }


def adapter_state(model: CachedTailClassifier) -> dict[str, torch.Tensor]:
    if model.adapter is None:
        return {}
    return {
        key: value.detach().cpu().clone()
        for key, value in model.adapter.state_dict().items()
    }


def audit_source_fbank(protocol: dict[str, Any]) -> dict[str, Any]:
    data = protocol["data"]
    paths_and_hashes = (
        ("fbank_path", "fbank_sha256"),
        ("frozen_embedding_path", "frozen_embedding_sha256"),
        ("roles_path", "roles_sha256"),
        ("audio_manifest_path", "audio_manifest_sha256"),
        ("audio_checksums_path", "audio_checksums_sha256"),
    )
    for path_key, hash_key in paths_and_hashes:
        path = REPO_ROOT / data[path_key]
        if not path.is_file() or sha256(path) != data[hash_key]:
            raise RuntimeError(f"IDEA-079 source checksum mismatch: {path}")
    with np.load(REPO_ROOT / data["fbank_path"], allow_pickle=False) as loaded:
        features_shape = tuple(loaded["features"].shape)
        segment_call_indices = loaded["segment_call_indices"].astype(np.int64)
        segment_counts = loaded["segment_counts"].astype(np.int64)
        call_ids = loaded["call_ids"].astype(str)
        source_paths = loaded["source_paths"].astype(str)
        cat_ids = loaded["cat_ids"].astype(str)
        labels = loaded["labels"].astype(np.int64)
    expected_counts = np.bincount(segment_call_indices, minlength=len(call_ids))
    if features_shape != (843, 128, 128):
        raise RuntimeError(f"IDEA-079 fbank geometry changed: {features_shape}")
    if len(call_ids) != 792 or len(np.unique(cat_ids)) != 111:
        raise RuntimeError("IDEA-079 call/cat population changed")
    if segment_call_indices.shape != (843,) or not np.array_equal(segment_counts, expected_counts):
        raise RuntimeError("IDEA-079 segment-to-call coverage changed")
    if expected_counts.min() != 1 or expected_counts.max() != 6 or expected_counts.sum() != 843:
        raise RuntimeError("IDEA-079 per-call segment range changed")
    if np.count_nonzero(expected_counts > 1) != 42:
        raise RuntimeError("IDEA-079 number of multi-segment calls changed")
    if len(source_paths) != len(call_ids) or any(not value for value in source_paths):
        raise RuntimeError("IDEA-079 source audio path coverage changed")
    if set(np.unique(labels)) != {0, 1, 2}:
        raise RuntimeError("IDEA-079 labels changed")
    ast = read_json(REPO_ROOT / data["locked_protocol_path"])["ast"]
    if (
        float(ast["segment_seconds"]) != 1.28
        or float(ast["segment_hop_seconds"]) != 0.64
        or int(ast["max_length_frames"]) != 128
        or int(ast["num_mel_bins"]) != 128
    ):
        raise RuntimeError("IDEA-079 locked audio window changed")
    return {
        "calls": int(len(call_ids)),
        "cats": int(len(np.unique(cat_ids))),
        "segments": int(len(segment_call_indices)),
        "fbank_shape": list(features_shape),
        "segment_count_range": [int(expected_counts.min()), int(expected_counts.max())],
        "multi_segment_calls": int(np.count_nonzero(expected_counts > 1)),
        "segment_count_sum": int(expected_counts.sum()),
        "source_path_rows": int(len(source_paths)),
        "window_seconds": float(ast["segment_seconds"]),
        "window_hop_seconds": float(ast["segment_hop_seconds"]),
        "fbank_sha256": data["fbank_sha256"],
        "audio_manifest_sha256": data["audio_manifest_sha256"],
        "audio_checksums_sha256": data["audio_checksums_sha256"],
        "roles_sha256": data["roles_sha256"],
    }


def _all_full_seeds(bases: list[int] | tuple[int, ...], folds: int = 4) -> set[int]:
    return {
        full_seed(int(base), repeat, fold)
        for base in bases
        for repeat in range(3)
        for fold in range(folds)
    }


def verify_protocol(protocol: dict[str, Any], allow_pending: bool = False) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea079-spatial-patch-adapter-v1":
        raise RuntimeError("Unexpected IDEA-079 protocol")
    allowed = {LOCKED_STATUS, PENDING_STATUS} if allow_pending else {LOCKED_STATUS}
    if protocol.get("status") not in allowed:
        raise RuntimeError("IDEA-079 protocol status does not permit this stage")
    model = protocol["model"]
    if tuple(model["pipelines"]) != PIPELINES or tuple(model["base_seeds"]) != BASE_SEEDS:
        raise RuntimeError("IDEA-079 pipeline or seed bank changed")
    if model["repeats"] != [0, 1, 2] or model["folds"] != [0, 1, 2, 3]:
        raise RuntimeError("IDEA-079 split scope changed")
    if model["outer_test_predictions"] is not False:
        raise RuntimeError("IDEA-079 must not access outer test predictions")
    if model["trainable_parameters"] != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("IDEA-079 parameter lock changed")
    if int(model["total_fits"]) != 108 or int(model["primary_A0_S1_fits"]) != 72:
        raise RuntimeError("IDEA-079 fit budget changed")
    adapter = protocol["adapter"]
    if (
        adapter["placement_after_block_one_based"] != 11
        or adapter["placement_before_block_one_based"] != 12
        or adapter["patch_grid"] != [12, 12]
        or adapter["patch_tokens"] != PATCH_TOKENS
        or adapter["special_tokens_bypassed"] != SPECIAL_TOKENS
        or adapter["bottleneck_width"] != BOTTLENECK
        or adapter["parameters"] != ADAPTER_PARAMETERS
    ):
        raise RuntimeError("IDEA-079 adapter definition changed")
    digest = hashlib.sha256(model["seed_derivation_text"].encode("utf-8")).hexdigest()
    if digest != model["seed_derivation_sha256"]:
        raise RuntimeError("IDEA-079 seed digest changed")
    derived = _all_full_seeds(BASE_SEEDS)
    excluded = _all_full_seeds(model["excluded_meow_base_seeds_IDEA065_through_IDEA078"])
    excluded |= _all_full_seeds(model["excluded_dog_base_seeds_IDEA075"], folds=5)
    if len(derived) != 36 or derived & excluded:
        raise RuntimeError("IDEA-079 full seeds collide or are not unique")
    dependency_checks = {
        REPO_ROOT / protocol["dependencies"]["idea_path"]: protocol["dependencies"]["idea_sha256"],
        Path(__file__).resolve(): protocol["dependencies"]["runner_sha256"],
    }
    for path, expected in dependency_checks.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"IDEA-079 dependency checksum mismatch: {path}")
    if protocol.get("status") == LOCKED_STATUS:
        shared = protocol["shared_cache"]
        if not shared.get("cache_manifest_sha256") or not shared.get("resolved"):
            raise RuntimeError("IDEA-079 locked protocol lacks resolved IDEA-078 cache hashes")


def adapter_pair_initialization_audit(seed: int) -> dict[str, Any]:
    set_seed(seed)
    pointwise = PatchTokenAdapter("pointwise")
    set_seed(seed)
    spatial = PatchTokenAdapter("spatial")
    shared_equal = {
        "down_weight": torch.equal(pointwise.down.weight, spatial.down.weight),
        "down_bias": torch.equal(pointwise.down.bias, spatial.down.bias),
        "mixer_flat_weight": torch.equal(
            pointwise.mixer.weight.reshape(-1), spatial.mixer.weight.reshape(-1)
        ),
        "mixer_bias": torch.equal(pointwise.mixer.bias, spatial.mixer.bias),
        "up_weight": torch.equal(pointwise.up.weight, spatial.up.weight),
        "up_bias": torch.equal(pointwise.up.bias, spatial.up.bias),
    }
    if not all(shared_equal.values()):
        raise RuntimeError("IDEA-079 controls do not share paired initial draws")
    return {
        "equal_initial_draws": shared_equal,
        "pointwise_mixer_shape": list(pointwise.mixer.weight.shape),
        "spatial_mixer_shape": list(spatial.mixer.weight.shape),
        "pointwise_mixer_parameters": int(sum(p.numel() for p in pointwise.mixer.parameters())),
        "spatial_mixer_parameters": int(sum(p.numel() for p in spatial.mixer.parameters())),
    }


class _IdentityBlock(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor]:
        del head_mask, output_attentions
        return (hidden_states,)


class _PatchMixingBlock(nn.Module):
    """Parameter-free test double that propagates patch changes to special tokens."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor]:
        del head_mask, output_attentions
        output = hidden_states.clone()
        patch_mean = hidden_states[:, SPECIAL_TOKENS:].mean(dim=1, keepdim=True)
        output[:, :SPECIAL_TOKENS] = output[:, :SPECIAL_TOKENS] + patch_mean
        return (output,)


def _synthetic_tail(propagate_patches: bool = False) -> FrozenASTTail:
    block: nn.Module = _PatchMixingBlock() if propagate_patches else _IdentityBlock()
    return FrozenASTTail(block, nn.Identity())


def paired_initialization_audit(
    protocol: dict[str, Any],
    seed: int,
    actual_tail: bool,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(79)
    tokens = torch.randn((4, TOTAL_TOKENS, HIDDEN_SIZE), generator=generator)
    segment_to_call = torch.arange(4, dtype=torch.long)
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    models: dict[str, CachedTailClassifier] = {}
    logits: dict[str, torch.Tensor] = {}
    losses: dict[str, float] = {}
    mean = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    scale = np.ones(HIDDEN_SIZE, dtype=np.float32)
    for pipeline in PIPELINES:
        set_seed(seed)
        head = ClassificationHead(mean, scale, float(protocol["fixed_training"]["dropout"]))
        tail = build_frozen_tail() if actual_tail else _synthetic_tail()
        model = CachedTailClassifier(pipeline, head, tail).eval()
        models[pipeline] = model
        with torch.no_grad():
            logits[pipeline] = model(tokens, segment_to_call, 4)
            losses[pipeline] = float(torch.nn.functional.cross_entropy(logits[pipeline], labels))
    counts = {pipeline: trainable_parameter_count(model) for pipeline, model in models.items()}
    if counts != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(f"IDEA-079 trainable parameter mismatch: {counts}")
    reference_head = shared_head_state(models[PIPELINES[0]])
    head_equal = {
        pipeline: reference_head.keys() == shared_head_state(models[pipeline]).keys()
        and all(
            torch.equal(reference_head[key], shared_head_state(models[pipeline])[key])
            for key in reference_head
        )
        for pipeline in PIPELINES[1:]
    }
    max_logit_differences = {
        pipeline: float(torch.max(torch.abs(logits[pipeline] - logits[PIPELINES[0]])))
        for pipeline in PIPELINES[1:]
    }
    loss_differences = {
        pipeline: abs(losses[pipeline] - losses[PIPELINES[0]])
        for pipeline in PIPELINES[1:]
    }
    special_differences: dict[str, float] = {}
    token_differences: dict[str, float] = {}
    for pipeline in PIPELINES[1:]:
        adapter = models[pipeline].adapter
        assert adapter is not None
        with torch.no_grad():
            adapted = adapter(tokens)
        special_differences[pipeline] = float(
            torch.max(torch.abs(adapted[:, :SPECIAL_TOKENS] - tokens[:, :SPECIAL_TOKENS]))
        )
        token_differences[pipeline] = float(torch.max(torch.abs(adapted - tokens)))
    if (
        not all(head_equal.values())
        or any(value != 0.0 for value in max_logit_differences.values())
        or any(value != 0.0 for value in loss_differences.values())
        or any(value != 0.0 for value in special_differences.values())
        or any(value != 0.0 for value in token_differences.values())
    ):
        raise RuntimeError("IDEA-079 zero-initialization equality failed")
    point_adapter = models[PIPELINES[1]].adapter
    spatial_adapter = models[PIPELINES[2]].adapter
    assert point_adapter is not None and spatial_adapter is not None
    control_initialization_equal = {
        "down_weight": torch.equal(point_adapter.down.weight, spatial_adapter.down.weight),
        "down_bias": torch.equal(point_adapter.down.bias, spatial_adapter.down.bias),
        "mixer_flat_weight": torch.equal(
            point_adapter.mixer.weight.reshape(-1), spatial_adapter.mixer.weight.reshape(-1)
        ),
        "mixer_bias": torch.equal(point_adapter.mixer.bias, spatial_adapter.mixer.bias),
        "up_weight": torch.equal(point_adapter.up.weight, spatial_adapter.up.weight),
        "up_bias": torch.equal(point_adapter.up.bias, spatial_adapter.up.bias),
    }
    if not all(control_initialization_equal.values()):
        raise RuntimeError("IDEA-079 pointwise/spatial paired initialization differs")
    return {
        "actual_ast_tail": actual_tail,
        "trainable_parameters": counts,
        "shared_head_state_equal": head_equal,
        "max_initial_logit_difference_from_A0": max_logit_differences,
        "initial_loss_difference_from_A0": loss_differences,
        "max_special_token_difference_at_injection": special_differences,
        "max_all_token_difference_at_injection": token_differences,
        "control_initialization_equal": control_initialization_equal,
    }


def gradient_reachability_audit(
    protocol: dict[str, Any], seed: int, actual_tail: bool = False
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(7901)
    tokens = torch.randn((4, TOTAL_TOKENS, HIDDEN_SIZE), generator=generator)
    segment_to_call = torch.arange(4, dtype=torch.long)
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    result: dict[str, Any] = {}
    for pipeline in PIPELINES[1:]:
        set_seed(seed)
        model = CachedTailClassifier(
            pipeline,
            ClassificationHead(
                np.zeros(HIDDEN_SIZE, dtype=np.float32),
                np.ones(HIDDEN_SIZE, dtype=np.float32),
                float(protocol["fixed_training"]["dropout"]),
            ),
            build_frozen_tail() if actual_tail else _synthetic_tail(propagate_patches=True),
        ).eval()
        assert model.adapter is not None
        loss = torch.nn.functional.cross_entropy(
            model(tokens, segment_to_call, 4), labels
        )
        loss.backward()
        up_at_zero = float(model.adapter.up.weight.grad.abs().max())
        down_at_zero = float(model.adapter.down.weight.grad.abs().max())
        mixer_at_zero = float(model.adapter.mixer.weight.grad.abs().max())
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.adapter.up.weight.fill_(1.0e-4)
        second_loss = torch.nn.functional.cross_entropy(
            model(tokens, segment_to_call, 4), labels
        )
        second_loss.backward()
        down_after_probe = float(model.adapter.down.weight.grad.abs().max())
        mixer_after_probe = float(model.adapter.mixer.weight.grad.abs().max())
        if (
            not np.isfinite(up_at_zero)
            or up_at_zero <= 0.0
            or down_at_zero != 0.0
            or mixer_at_zero != 0.0
            or down_after_probe <= 0.0
            or mixer_after_probe <= 0.0
        ):
            raise RuntimeError(f"IDEA-079 gradient reachability failed for {pipeline}")
        result[pipeline] = {
            "actual_ast_tail": actual_tail,
            "up_weight_max_gradient_at_zero": up_at_zero,
            "down_weight_max_gradient_at_zero": down_at_zero,
            "mixer_weight_max_gradient_at_zero": mixer_at_zero,
            "down_weight_max_gradient_after_up_probe": down_after_probe,
            "mixer_weight_max_gradient_after_up_probe": mixer_after_probe,
        }
    return result


@dataclass(frozen=True)
class TokenStore:
    tokens: np.ndarray
    segment_call_indices: np.ndarray
    call_segment_indices: tuple[np.ndarray, ...]
    frozen_embeddings: np.ndarray
    call_ids: np.ndarray
    cat_ids: np.ndarray
    labels: np.ndarray
    source_paths: np.ndarray


def _resolved_cache(protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("status") != LOCKED_STATUS:
        raise RuntimeError(
            "IDEA-079 shared cache is unresolved; wait for IDEA-078 final manifest/hash"
        )
    shared = protocol["shared_cache"]
    manifest_path = REPO_ROOT / shared["cache_manifest_path"]
    expected_manifest_hash = shared.get("cache_manifest_sha256")
    resolved = shared.get("resolved")
    if not expected_manifest_hash or not resolved:
        raise RuntimeError("IDEA-079 shared cache placeholders are not resolved")
    if not manifest_path.is_file() or sha256(manifest_path) != expected_manifest_hash:
        raise RuntimeError("IDEA-079 shared IDEA-078 cache manifest mismatch")
    for path_key, hash_key in (
        ("tokens_path", "tokens_sha256"),
        ("index_path", "index_sha256"),
        ("extractor_path", "extractor_sha256"),
        ("cache_schema_path", "cache_schema_sha256"),
        ("read_only_verifier_path", "read_only_verifier_sha256"),
    ):
        path = REPO_ROOT / resolved[path_key]
        if not path.is_file() or sha256(path) != resolved[hash_key]:
            raise RuntimeError(f"IDEA-079 resolved shared-cache dependency mismatch: {path}")
    manifest = read_json(manifest_path)
    manifest_expectations = {
        "status": "complete",
        "artifact": resolved["artifact"],
        "label_information_used": False,
        "roles_used": False,
        "cat_ids_stored": False,
        "labels_stored": False,
        "protocol_id": resolved["owner_protocol_id"],
        "protocol_path": resolved["owner_protocol_path"],
        "protocol_sha256": resolved["owner_protocol_sha256"],
        "cache_schema_path": resolved["cache_schema_path"],
        "cache_schema_sha256": resolved["cache_schema_sha256"],
        "extractor_path": resolved["extractor_path"],
        "extractor_sha256": resolved["extractor_sha256"],
        "read_only_verifier_path": resolved["read_only_verifier_path"],
        "read_only_verifier_sha256": resolved["read_only_verifier_sha256"],
        "token_path": resolved["tokens_path"],
        "token_sha256": resolved["tokens_sha256"],
        "index_path": resolved["index_path"],
        "index_sha256": resolved["index_sha256"],
        "shape": [843, TOTAL_TOKENS, HIDDEN_SIZE],
        "dtype": "float32",
        "storage": "npy_memory_mappable",
        "segments": 843,
        "calls": 792,
        "prelast_block_output_one_based": 11,
        "last_block_input_one_based": 12,
        "special_token_indices": [0, 1],
        "patch_token_indices": [2, 145],
        "window_seconds": 1.28,
        "hop_seconds": 0.64,
    }
    if any(manifest.get(key) != value for key, value in manifest_expectations.items()):
        raise RuntimeError("IDEA-079 IDEA-078 cache manifest schema/invariants changed")
    geometry = manifest.get("geometry", {})
    if (
        geometry.get("target_grid") != [12, 12]
        or geometry.get("patch_tokens") != PATCH_TOKENS
        or geometry.get("special_tokens") != SPECIAL_TOKENS
        or geometry.get("total_tokens") != TOTAL_TOKENS
        or geometry.get("hidden_size") != HIDDEN_SIZE
        or geometry.get("frequency_stride") != 10
        or geometry.get("time_stride") != 10
        or geometry.get("patch_flatten_order") != "frequency-major then time-minor"
    ):
        raise RuntimeError("IDEA-079 IDEA-078 cache geometry changed")
    reconstruction = resolved["reconstruction_vs_locked_A0"]
    manifest_reconstruction = manifest.get("a0_reconstruction", {})
    if (
        float(manifest_reconstruction.get("mean_absolute_error", float("inf")))
        != float(reconstruction["mean_absolute_difference"])
        or float(manifest_reconstruction.get("maximum_absolute_error", float("inf")))
        != float(reconstruction["maximum_absolute_difference"])
        or manifest_reconstruction.get("passed") is not True
    ):
        raise RuntimeError("IDEA-079 resolved A0 audit differs from cache manifest")
    tolerance = shared["reconstruction_tolerance"]
    if (
        float(reconstruction["mean_absolute_difference"])
        > float(tolerance["mean_absolute_maximum"])
        or float(reconstruction["maximum_absolute_difference"])
        > float(tolerance["absolute_maximum"])
    ):
        raise RuntimeError("IDEA-079 shared cache does not reconstruct locked A0")
    return resolved


def load_token_store(protocol: dict[str, Any]) -> tuple[TokenStore, dict[str, Any]]:
    resolved = _resolved_cache(protocol)
    tokens_path = REPO_ROOT / resolved["tokens_path"]
    if resolved["tokens_format"] != "npy_float32_mmap":
        raise RuntimeError("IDEA-079 shared tokens must be a float32 mmap-compatible NPY")
    tokens = np.load(tokens_path, mmap_mode="r", allow_pickle=False)
    if tokens.shape != (843, TOTAL_TOKENS, HIDDEN_SIZE) or tokens.dtype != np.float32:
        raise RuntimeError(f"IDEA-079 shared token geometry changed: {tokens.shape}/{tokens.dtype}")
    with np.load(REPO_ROOT / resolved["index_path"], allow_pickle=False) as index:
        forbidden = [key for key in index.files if "label" in key.lower() or "cat" in key.lower()]
        if forbidden:
            raise RuntimeError(f"IDEA-079 shared cache is not label-free: {forbidden}")
        keys = resolved["index_keys"]
        segment_call_indices = index[keys["segment_call_indices"]].astype(np.int64)
        segment_counts = index[keys["segment_counts"]].astype(np.int64)
        cache_call_ids = index[keys["call_ids"]].astype(str)
        cache_source_paths = index[keys["source_paths"]].astype(str)
        prelast_block = index[keys["prelast_block_output_one_based"]].astype(np.int64)
        last_block = index[keys["last_block_input_one_based"]].astype(np.int64)
        special_indices = index[keys["special_token_indices"]].astype(np.int64)
        patch_token_start = index[keys["patch_token_start"]].astype(np.int64)
    if (
        prelast_block.tolist() != [11]
        or last_block.tolist() != [12]
        or special_indices.tolist() != [0, 1]
        or patch_token_start.tolist() != [2]
    ):
        raise RuntimeError("IDEA-079 shared-cache index token boundary changed")
    data = protocol["data"]
    with np.load(REPO_ROOT / data["fbank_path"], allow_pickle=False) as fbank:
        fbank_segment_call_indices = fbank["segment_call_indices"].astype(np.int64)
        fbank_segment_counts = fbank["segment_counts"].astype(np.int64)
        call_ids = fbank["call_ids"].astype(str)
        cat_ids = fbank["cat_ids"].astype(str)
        labels = fbank["labels"].astype(np.int64)
        source_paths = fbank["source_paths"].astype(str)
    with np.load(REPO_ROOT / data["frozen_embedding_path"], allow_pickle=False) as frozen:
        frozen_embeddings = frozen["embeddings"].astype(np.float32)
        frozen_call_ids = frozen["call_ids"].astype(str)
    if not (
        np.array_equal(segment_call_indices, fbank_segment_call_indices)
        and np.array_equal(segment_counts, fbank_segment_counts)
        and np.array_equal(cache_call_ids, call_ids)
        and np.array_equal(cache_source_paths, source_paths)
        and np.array_equal(frozen_call_ids, call_ids)
    ):
        raise RuntimeError("IDEA-079 shared-cache/fbank/frozen metadata differs")
    expected_counts = np.bincount(segment_call_indices, minlength=len(call_ids))
    if not np.array_equal(expected_counts, segment_counts) or np.any(expected_counts < 1):
        raise RuntimeError("IDEA-079 shared cache loses segment-to-call coverage")
    call_segment_indices = tuple(
        np.flatnonzero(segment_call_indices == call_index).astype(np.int64)
        for call_index in range(len(call_ids))
    )
    store = TokenStore(
        tokens,
        segment_call_indices,
        call_segment_indices,
        frozen_embeddings,
        call_ids,
        cat_ids,
        labels,
        source_paths,
    )
    return store, {
        "tokens_shape": list(tokens.shape),
        "tokens_dtype": str(tokens.dtype),
        "mmap": isinstance(tokens, np.memmap),
        "calls": int(len(call_ids)),
        "segments": int(len(segment_call_indices)),
        "segment_count_range": [int(expected_counts.min()), int(expected_counts.max())],
        "reconstruction_vs_locked_A0": resolved["reconstruction_vs_locked_A0"],
        "cache_manifest_sha256": protocol["shared_cache"]["cache_manifest_sha256"],
    }


def reconstruct_cached_A0(
    protocol: dict[str, Any],
    store: TokenStore,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    tail = build_frozen_tail().to(device).eval()
    batch_size = int(protocol["shared_cache"]["reconstruction_batch_size"])
    call_sums = np.zeros((len(store.call_ids), HIDDEN_SIZE), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(store.tokens), batch_size):
            stop = min(start + batch_size, len(store.tokens))
            batch = torch.from_numpy(
                np.asarray(store.tokens[start:stop], dtype=np.float32).copy()
            ).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                segment_embeddings = tail(batch)
            np.add.at(
                call_sums,
                store.segment_call_indices[start:stop],
                segment_embeddings.float().cpu().numpy(),
            )
    counts = np.asarray(
        [len(indices) for indices in store.call_segment_indices], dtype=np.float64
    )
    reconstructed = (call_sums / counts[:, None]).astype(np.float32)
    difference = np.abs(reconstructed - store.frozen_embeddings)
    audit = {
        "mean_absolute_difference": float(difference.mean()),
        "maximum_absolute_difference": float(difference.max()),
    }
    tolerance = protocol["shared_cache"]["reconstruction_tolerance"]
    if (
        audit["mean_absolute_difference"]
        > float(tolerance["mean_absolute_maximum"])
        or audit["maximum_absolute_difference"]
        > float(tolerance["absolute_maximum"])
    ):
        raise RuntimeError("IDEA-079 runtime cache path does not reconstruct locked A0")
    return reconstructed, audit


def real_cache_initialization_audit(
    protocol: dict[str, Any],
    store: TokenStore,
    train_indices: np.ndarray,
    probe_call_indices: np.ndarray,
    reconstructed: np.ndarray,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    final_train = store.frozen_embeddings[train_indices]
    mean = final_train.mean(axis=0)
    scale = final_train.std(axis=0)
    probe_calls = np.asarray(probe_call_indices[:8], dtype=np.int64)
    segment_rows = [store.call_segment_indices[int(index)] for index in probe_calls]
    probe_segments = np.concatenate(segment_rows)
    segment_to_call = np.concatenate(
        [
            np.full(len(segments), local, dtype=np.int64)
            for local, segments in enumerate(segment_rows)
        ]
    )
    tokens = torch.from_numpy(
        np.asarray(store.tokens[probe_segments], dtype=np.float32).copy()
    ).to(device)
    mapping = torch.from_numpy(segment_to_call).to(device)
    models: dict[str, CachedTailClassifier] = {}
    logits: dict[str, torch.Tensor] = {}
    for pipeline in PIPELINES:
        set_seed(seed)
        model = build_model(pipeline, protocol, mean, scale).to(device).eval()
        models[pipeline] = model
        with torch.inference_mode():
            logits[pipeline] = model(tokens, mapping, len(probe_calls)).float().cpu()
    exact_differences = {
        pipeline: float(
            torch.max(torch.abs(logits[pipeline] - logits["A0_cached_tail"]))
        )
        for pipeline in PIPELINES[1:]
    }
    if any(value != 0.0 for value in exact_differences.values()):
        raise RuntimeError("IDEA-079 real-cache initial pipelines are not exactly equal")
    set_seed(seed)
    audit_head = ClassificationHead(
        mean,
        scale,
        float(protocol["fixed_training"]["dropout"]),
    ).to(device).eval()
    with torch.inference_mode():
        cached_logits = audit_head(
            torch.from_numpy(reconstructed[probe_calls]).to(device)
        ).float().cpu()
        locked_logits = audit_head(
            torch.from_numpy(store.frozen_embeddings[probe_calls]).to(device)
        ).float().cpu()
    locked_difference = float(torch.max(torch.abs(cached_logits - locked_logits)))
    maximum = float(
        protocol["shared_cache"]["reconstruction_tolerance"][
            "maximum_initial_logit_difference"
        ]
    )
    if locked_difference > maximum:
        raise RuntimeError("IDEA-079 cached-tail logits drift too far from locked A0")
    return {
        "probe_calls": int(len(probe_calls)),
        "probe_segments": int(len(probe_segments)),
        "probe_excludes_outer_test": True,
        "max_C1_S1_initial_logit_difference_from_cached_A0": exact_differences,
        "maximum_cached_tail_vs_locked_A0_logit_difference": locked_difference,
        "maximum_allowed_cached_tail_vs_locked_A0_logit_difference": maximum,
        "shared_head_state_equal": {
            pipeline: all(
                torch.equal(
                    shared_head_state(models["A0_cached_tail"])[key],
                    shared_head_state(models[pipeline])[key],
                )
                for key in shared_head_state(models["A0_cached_tail"])
            )
            for pipeline in PIPELINES[1:]
        },
    }


def fold_indices(
    store: TokenStore,
    roles: pd.DataFrame,
    repeat: int,
    fold: int,
) -> dict[str, np.ndarray]:
    cell = roles[(roles["repeat"] == repeat) & (roles["outer_fold"] == fold)]
    if cell["cat_id"].duplicated().any():
        raise RuntimeError("IDEA-079 role cell assigns a cat more than once")
    mapping = dict(zip(cell["cat_id"].astype(str), cell["role"].astype(str)))
    if set(mapping) != set(store.cat_ids.astype(str)):
        raise RuntimeError("IDEA-079 role cell does not cover exactly the cached cats")
    call_roles = np.asarray([mapping[cat_id] for cat_id in store.cat_ids], dtype=str)
    return {
        "train": np.flatnonzero(call_roles == "train"),
        "validation": np.flatnonzero(call_roles == "validation"),
    }


def validate_roles(protocol: dict[str, Any], store: TokenStore, roles: pd.DataFrame) -> int:
    if set(roles["role"].astype(str)) != {"train", "validation", "test"}:
        raise RuntimeError("IDEA-079 role vocabulary changed")
    cells = 0
    for repeat in protocol["model"]["repeats"]:
        for fold in protocol["model"]["folds"]:
            cell = roles[(roles["repeat"] == repeat) & (roles["outer_fold"] == fold)]
            sets = {
                role: set(cell[cell["role"] == role]["cat_id"].astype(str))
                for role in ("train", "validation", "test")
            }
            if any(
                sets[left] & sets[right]
                for left, right in (
                    ("train", "validation"),
                    ("train", "test"),
                    ("validation", "test"),
                )
            ):
                raise RuntimeError("IDEA-079 cat leakage across roles")
            indices = fold_indices(store, roles, repeat, fold)
            if np.intersect1d(indices["train"], indices["validation"]).size:
                raise RuntimeError("IDEA-079 call leakage across roles")
            cells += 1
    return cells


class DeterministicNoSingletonBatchSampler(Sampler[list[int]]):
    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        if size < 2 or batch_size < 2:
            raise ValueError("IDEA-079 batching needs at least two cats")
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.generator = torch.Generator().manual_seed(int(seed))

    def __iter__(self) -> Iterator[list[int]]:
        order = torch.randperm(self.size, generator=self.generator).tolist()
        batches = [
            order[start : start + self.batch_size]
            for start in range(0, self.size, self.batch_size)
        ]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-1].insert(0, batches[-2].pop())
        if any(len(batch) == 1 for batch in batches):
            raise RuntimeError("IDEA-079 singleton cat batch remains")
        yield from batches

    def __len__(self) -> int:
        return math.ceil(self.size / self.batch_size)


class CatTokenDataset(Dataset[tuple[np.ndarray, np.ndarray, np.ndarray, str]]):
    def __init__(self, store: TokenStore, call_indices: np.ndarray) -> None:
        self.store = store
        selected = np.asarray(call_indices, dtype=np.int64)
        self.cat_ids = np.asarray(sorted(np.unique(store.cat_ids[selected]).tolist()))
        self.call_indices: list[np.ndarray] = []
        for cat_id in self.cat_ids:
            calls = np.sort(selected[store.cat_ids[selected] == cat_id])
            labels = np.unique(store.labels[calls])
            if len(labels) != 1:
                raise RuntimeError(f"IDEA-079 cat {cat_id} has inconsistent labels")
            self.call_indices.append(calls)
        if sum(len(value) for value in self.call_indices) != len(selected):
            raise RuntimeError("IDEA-079 cat dataset lost calls")

    def __len__(self) -> int:
        return len(self.cat_ids)

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        calls = self.call_indices[item]
        segment_rows: list[np.ndarray] = []
        segment_to_call: list[np.ndarray] = []
        for local_call, call_index in enumerate(calls):
            segments = self.store.call_segment_indices[int(call_index)]
            segment_rows.append(np.asarray(self.store.tokens[segments], dtype=np.float32))
            segment_to_call.append(np.full(len(segments), local_call, dtype=np.int64))
        return (
            np.concatenate(segment_rows, axis=0),
            np.concatenate(segment_to_call),
            calls,
            str(self.cat_ids[item]),
        )


def collate_cats(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]) -> dict[str, Any]:
    tokens: list[np.ndarray] = []
    segment_to_call: list[np.ndarray] = []
    call_indices: list[np.ndarray] = []
    cats: list[str] = []
    call_offset = 0
    for row_tokens, row_mapping, row_calls, cat_id in rows:
        tokens.append(row_tokens)
        segment_to_call.append(row_mapping + call_offset)
        call_indices.append(row_calls)
        cats.append(cat_id)
        call_offset += len(row_calls)
    return {
        "tokens": torch.from_numpy(np.concatenate(tokens).astype(np.float32, copy=False)),
        "segment_to_call": torch.from_numpy(np.concatenate(segment_to_call)),
        "call_indices": torch.from_numpy(np.concatenate(call_indices).astype(np.int64, copy=False)),
        "cat_ids": cats,
    }


def build_loader(
    store: TokenStore,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = CatTokenDataset(store, indices)
    if shuffle:
        return DataLoader(
            dataset,
            batch_sampler=DeterministicNoSingletonBatchSampler(len(dataset), batch_size, seed),
            num_workers=0,
            collate_fn=collate_cats,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_cats,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "tokens": batch["tokens"].to(device),
        "segment_to_call": batch["segment_to_call"].to(device),
        "call_indices": batch["call_indices"].to(device),
        "cat_ids": batch["cat_ids"],
    }


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("IDEA-079 training role is missing a class")
    return (len(labels) / (3.0 * counts)).astype(np.float32)


def train_one_epoch(
    model: CachedTailClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    store: TokenStore,
    weights: torch.Tensor,
    device: torch.device,
    gradient_clip: float,
) -> tuple[float, dict[str, Any]]:
    model.train()
    model.tail.eval()
    weighted_total = 0.0
    weight_total = 0.0
    processed_cats: list[str] = []
    processed_calls: list[int] = []
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        call_indices = batch["call_indices"]
        labels = torch.from_numpy(store.labels[call_indices.cpu().numpy()]).to(device)
        call_weights = weights[labels]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(
                batch["tokens"], batch["segment_to_call"], len(call_indices)
            )
            per_call = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            loss = (per_call * call_weights).sum() / call_weights.sum()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            gradient_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        weighted_total += float((per_call.detach() * call_weights).sum())
        weight_total += float(call_weights.sum())
        processed_cats.extend(str(value) for value in batch["cat_ids"])
        processed_calls.extend(int(value) for value in call_indices.cpu().tolist())
    if len(processed_cats) != len(set(processed_cats)):
        raise RuntimeError("IDEA-079 epoch repeated a cat")
    if len(processed_calls) != len(set(processed_calls)):
        raise RuntimeError("IDEA-079 epoch repeated a call")
    return weighted_total / weight_total, {
        "cats": len(processed_cats),
        "calls": len(processed_calls),
        "cat_order_sha256": hashlib.sha256(
            "\n".join(processed_cats).encode("utf-8")
        ).hexdigest(),
        "call_coverage_sha256": hashlib.sha256(
            np.sort(np.asarray(processed_calls, dtype="<i8")).tobytes()
        ).hexdigest(),
    }


def calls_to_animals(calls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat_id, group in calls.groupby("cat_id", sort=True):
        labels = group["true_label"].unique()
        if len(labels) != 1:
            raise RuntimeError(f"IDEA-079 cat {cat_id} has conflicting labels")
        probabilities = group[list(PROBABILITY_COLUMNS)].to_numpy(float).mean(axis=0)
        rows.append(
            {
                "cat_id": str(cat_id),
                "true_label": int(labels[0]),
                "call_count": int(len(group)),
                **{
                    column: float(probabilities[index])
                    for index, column in enumerate(PROBABILITY_COLUMNS)
                },
                "predicted_label": int(probabilities.argmax()),
            }
        )
    return pd.DataFrame(rows)


def predict(
    model: CachedTailClassifier,
    store: TokenStore,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = build_loader(store, indices, batch_size, False, seed)
    model.eval()
    rows = []
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(
                    batch["tokens"],
                    batch["segment_to_call"],
                    len(batch["call_indices"]),
                )
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            call_indices = batch["call_indices"].cpu().numpy()
            for local_index, call_index in enumerate(call_indices):
                rows.append(
                    {
                        "call_index": int(call_index),
                        "call_id": str(store.call_ids[call_index]),
                        "cat_id": str(store.cat_ids[call_index]),
                        "true_label": int(store.labels[call_index]),
                        **{
                            column: float(probabilities[local_index, class_index])
                            for class_index, column in enumerate(PROBABILITY_COLUMNS)
                        },
                    }
                )
    calls = pd.DataFrame(rows).sort_values("call_index").reset_index(drop=True)
    return calls_to_animals(calls), calls


def animal_metrics(animals: pd.DataFrame) -> dict[str, Any]:
    labels = animals["true_label"].to_numpy(np.int64)
    predictions = animals["predicted_label"].to_numpy(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1, 2], zero_division=0
    )
    return {
        "macro_f1": float(
            f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(labels, predictions, weights="quadratic")
        ),
        "plain_accuracy": float(accuracy_score(labels, predictions)),
        "per_class": {
            LABEL_NAMES[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(3)
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
        "n": int(len(animals)),
    }


def animal_cross_entropy(animals: pd.DataFrame) -> float:
    probabilities = animals[list(PROBABILITY_COLUMNS)].to_numpy(float)
    labels = animals["true_label"].to_numpy(np.int64)
    return float(
        -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1.0e-12, 1.0)).mean()
    )


def brier(animals: pd.DataFrame) -> float:
    probabilities = animals[list(PROBABILITY_COLUMNS)].to_numpy(float)
    labels = animals["true_label"].to_numpy(np.int64)
    return float(np.mean(np.sum((probabilities - np.eye(3)[labels]) ** 2, axis=1)))


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def make_optimizer(
    model: CachedTailClassifier,
    protocol: dict[str, Any],
) -> torch.optim.Optimizer:
    fixed = protocol["fixed_training"]
    groups: list[dict[str, Any]] = [
        {"params": list(model.head.parameters()), "lr": float(fixed["head_learning_rate"])}
    ]
    if model.adapter is not None:
        groups.insert(
            0,
            {
                "params": list(model.adapter.parameters()),
                "lr": float(fixed["adapter_learning_rate"]),
            },
        )
    return torch.optim.Adamax(groups, eps=float(fixed["optimizer_epsilon"]))


def fit_inner(
    pipeline: str,
    protocol: dict[str, Any],
    store: TokenStore,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    final_train = store.frozen_embeddings[train_indices]
    model = build_model(
        pipeline,
        protocol,
        final_train.mean(axis=0),
        final_train.std(axis=0),
    ).to(device)
    fixed = protocol["fixed_training"]
    training_seed = seed + int(fixed["post_build_seed_offset"])
    set_seed(training_seed)
    optimizer = make_optimizer(model, protocol)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    train_loader = build_loader(
        store,
        train_indices,
        int(fixed["cat_batch_size"]),
        True,
        seed,
    )
    weights = torch.from_numpy(class_weights(store.labels[train_indices])).to(device)
    best_loss = float("inf")
    best_epoch = 1
    best_state = cpu_state_dict(model)
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
            train_loader,
            optimizer,
            scaler,
            store,
            weights,
            device,
            float(fixed["gradient_clip"]),
        )
        animals, calls = predict(
            model,
            store,
            validation_indices,
            int(fixed["cat_batch_size"]) * 2,
            device,
            seed,
        )
        validation_loss = animal_cross_entropy(animals)
        metrics = animal_metrics(animals)
        history.append(
            {
                "epoch": epoch,
                "train_call_loss": train_loss,
                "train_audit": train_audit,
                "validation_animal_cross_entropy": validation_loss,
                "validation_animal_brier": brier(animals),
                "validation_animal_metrics": metrics,
                "model": model.audit(),
            }
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            best_animals = animals.copy()
            best_calls = calls.copy()
            stale = 0
        else:
            stale += 1
        print(
            f"{pipeline} epoch={epoch} call_CE={train_loss:.4f} "
            f"val_CE={validation_loss:.4f} val_F1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if stale >= int(fixed["early_stopping_patience"]):
            break
    if best_animals is None or best_calls is None:
        raise RuntimeError("IDEA-079 selected no checkpoint")
    model.load_state_dict(best_state)
    reload_animals, _ = predict(
        model,
        store,
        validation_indices,
        int(fixed["cat_batch_size"]) * 2,
        device,
        seed,
    )
    reload_difference = float(
        np.abs(
            reload_animals[list(PROBABILITY_COLUMNS)].to_numpy()
            - best_animals[list(PROBABILITY_COLUMNS)].to_numpy()
        ).max()
    )
    audit = {
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "full_seed": seed,
        "training_seed": training_seed,
        "best_validation_animal_cross_entropy": best_loss,
        "best_validation_animal_brier": brier(best_animals),
        "best_validation_animal_metrics": animal_metrics(best_animals),
        "checkpoint_reload_max_probability_difference": reload_difference,
        "best_model": model.audit(),
        "train_seconds": float(time.perf_counter() - started),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "history": history,
        "outer_test_accessed": False,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return audit, best_animals, best_calls


def metric_bundle(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "metrics": animal_metrics(frame),
        "cross_entropy": animal_cross_entropy(frame),
        "brier": brier(frame),
    }


def add_metrics(row: dict[str, Any], bundles: dict[str, Any]) -> None:
    for pipeline in PIPELINES:
        row[f"{pipeline}_macro_f1"] = bundles[pipeline]["metrics"]["macro_f1"]
        row[f"{pipeline}_balanced_accuracy"] = bundles[pipeline]["metrics"]["balanced_accuracy"]
        row[f"{pipeline}_cross_entropy"] = bundles[pipeline]["cross_entropy"]
        row[f"{pipeline}_brier"] = bundles[pipeline]["brier"]
        row[f"{pipeline}_senior_recall"] = bundles[pipeline]["metrics"]["per_class"]["senior"]["recall"]
    row["S1_minus_A0_macro_f1"] = (
        row["S1_spatial_patch_macro_f1"] - row["A0_cached_tail_macro_f1"]
    )
    row["C1_minus_A0_macro_f1"] = (
        row["C1_pointwise_patch_macro_f1"] - row["A0_cached_tail_macro_f1"]
    )
    row["S1_minus_C1_macro_f1"] = (
        row["S1_spatial_patch_macro_f1"] - row["C1_pointwise_patch_macro_f1"]
    )


def aggregate(fits: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    by_key = {
        (fit["pipeline"], fit["base_seed"], fit["repeat"], fit["fold"]): fit
        for fit in fits
    }
    fold_rows = []
    seed_repeat_rows = []
    pooled_all: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
    for base_seed in protocol["model"]["base_seeds"]:
        for repeat in protocol["model"]["repeats"]:
            grouped: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
            for fold in protocol["model"]["folds"]:
                frames = {
                    pipeline: pd.read_csv(
                        REPO_ROOT
                        / by_key[(pipeline, base_seed, repeat, fold)][
                            "validation_animal_predictions"
                        ],
                        dtype={"cat_id": str},
                    )
                    for pipeline in PIPELINES
                }
                row: dict[str, Any] = {
                    "base_seed": base_seed,
                    "repeat": repeat,
                    "fold": fold,
                }
                add_metrics(
                    row,
                    {
                        pipeline: metric_bundle(frame)
                        for pipeline, frame in frames.items()
                    },
                )
                fold_rows.append(row)
                for pipeline, frame in frames.items():
                    tagged = frame.copy()
                    tagged["base_seed"] = base_seed
                    tagged["repeat"] = repeat
                    tagged["fold"] = fold
                    grouped[pipeline].append(tagged)
                    pooled_all[pipeline].append(tagged)
            pooled = {
                pipeline: pd.concat(parts, ignore_index=True)
                for pipeline, parts in grouped.items()
            }
            row = {"base_seed": base_seed, "repeat": repeat}
            add_metrics(
                row,
                {
                    pipeline: metric_bundle(frame)
                    for pipeline, frame in pooled.items()
                },
            )
            seed_repeat_rows.append(row)
    folds = pd.DataFrame(fold_rows)
    seed_repeats = pd.DataFrame(seed_repeat_rows)
    comparison_columns = (
        "S1_minus_A0_macro_f1",
        "C1_minus_A0_macro_f1",
        "S1_minus_C1_macro_f1",
    )
    split_cells = folds.groupby(["repeat", "fold"], as_index=False)[
        list(comparison_columns)
    ].mean()
    mean_metrics = {
        metric: {
            pipeline: float(seed_repeats[f"{pipeline}_{metric}"].mean())
            for pipeline in PIPELINES
        }
        for metric in ("macro_f1", "balanced_accuracy", "cross_entropy", "brier")
    }
    per_seed_delta = {
        str(seed): float(
            seed_repeats[seed_repeats["base_seed"] == seed][
                "S1_minus_A0_macro_f1"
            ].mean()
        )
        for seed in BASE_SEEDS
    }
    pooled_frames = {
        pipeline: pd.concat(parts, ignore_index=True)
        for pipeline, parts in pooled_all.items()
    }
    per_seed_senior = {}
    for seed in BASE_SEEDS:
        senior = {
            pipeline: animal_metrics(frame[frame["base_seed"] == seed])["per_class"][
                "senior"
            ]["recall"]
            for pipeline, frame in pooled_frames.items()
        }
        per_seed_senior[str(seed)] = float(
            senior["S1_spatial_patch"] - senior["A0_cached_tail"]
        )
    candidate_delta = seed_repeats["S1_minus_A0_macro_f1"]
    mechanism_delta = seed_repeats["S1_minus_C1_macro_f1"]
    split_delta = split_cells["S1_minus_A0_macro_f1"]
    gate = protocol["gate"]
    candidate_conditions = {
        "mean_macro_f1_gain": float(candidate_delta.mean())
        >= float(gate["minimum_mean_seed_repeat_S1_minus_A0"]),
        "positive_base_seed_means": sum(value > 0 for value in per_seed_delta.values())
        >= int(gate["minimum_positive_base_seeds"]),
        "positive_seed_repeats": int((candidate_delta > 0).sum())
        >= int(gate["minimum_positive_seed_repeats"]),
        "nonnegative_split_cells": int((split_delta >= 0).sum())
        >= int(gate["minimum_nonnegative_split_cells"]),
        "worst_split_cell": float(split_delta.min())
        >= float(gate["minimum_worst_split_cell_delta"]),
        "cross_entropy_nonworse": mean_metrics["cross_entropy"]["S1_spatial_patch"]
        <= mean_metrics["cross_entropy"]["A0_cached_tail"],
        "brier_nonworse": mean_metrics["brier"]["S1_spatial_patch"]
        <= mean_metrics["brier"]["A0_cached_tail"],
        "per_base_seed_senior_recall": all(
            value >= float(gate["minimum_per_base_seed_senior_recall_delta"])
            for value in per_seed_senior.values()
        ),
    }
    mechanism_conditions = {
        "mean_S1_minus_C1_macro_f1": float(mechanism_delta.mean())
        >= float(gate["minimum_mean_seed_repeat_S1_minus_C1"]),
        "positive_S1_minus_C1_seed_repeats": int((mechanism_delta > 0).sum())
        >= int(gate["minimum_positive_S1_minus_C1_seed_repeats"]),
    }
    comparisons = {}
    for column in comparison_columns:
        values = seed_repeats[column]
        comparisons[column] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            "median": float(values.median()),
            "positive": int((values > 0).sum()),
            "tied": int((values == 0).sum()),
            "negative": int((values < 0).sum()),
            "worst": float(values.min()),
            "best": float(values.max()),
        }
    return {
        "status": "complete",
        "outer_test_accessed": False,
        "fits": len(fits),
        "seed_repeat_estimates": len(seed_repeat_rows),
        "split_cell_estimates": int(len(split_cells)),
        "independence_note": (
            "Repeated validation animals across folds, seeds, and repeats are "
            "descriptive occurrences, not independent samples."
        ),
        "fold_results": fold_rows,
        "seed_repeat_results": seed_repeat_rows,
        "split_cell_results": split_cells.to_dict(orient="records"),
        "seed_repeat_equal_weight_means": mean_metrics,
        "comparisons": comparisons,
        "per_base_seed_S1_minus_A0_macro_f1": per_seed_delta,
        "per_base_seed_S1_minus_A0_senior_recall": per_seed_senior,
        "candidate_gate_conditions": candidate_conditions,
        "mechanism_gate_conditions": mechanism_conditions,
        "candidate_gate_passed": bool(all(candidate_conditions.values())),
        "spatial_mechanism_gate_passed": bool(all(mechanism_conditions.values())),
        "full_gate_passed": bool(
            all(candidate_conditions.values()) and all(mechanism_conditions.values())
        ),
    }


def structural_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cpu":
        raise RuntimeError("IDEA-079 structural preflight is CPU-only")
    configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol, allow_pending=True)
    source_audit = audit_source_fbank(protocol)
    started = time.perf_counter()
    initialization = paired_initialization_audit(
        protocol,
        BASE_SEEDS[0],
        actual_tail=True,
    )
    gradients = gradient_reachability_audit(
        protocol, BASE_SEEDS[0], actual_tail=True
    )
    controls = adapter_pair_initialization_audit(BASE_SEEDS[0])
    return {
        "status": "GO_FOR_SHARED_IDEA078_CACHE_DEPENDENCY",
        "read_only": True,
        "device": "cpu",
        "gpu_used": False,
        "formal_run_ready": protocol["status"] == LOCKED_STATUS,
        "cache_dependency_status": protocol["status"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "source_audit": source_audit,
        "geometry": {
            "block11_cache_tokens": [843, TOTAL_TOKENS, HIDDEN_SIZE],
            "patch_grid": list(PATCH_GRID),
            "patch_tokens": PATCH_TOKENS,
            "special_tokens": SPECIAL_TOKENS,
            "adapter_mac_per_segment": 2_002_320,
            "shared_float32_token_bytes": 378_095_616,
            "shared_float32_token_mib": 360.58,
        },
        "base_seeds": list(BASE_SEEDS),
        "unique_full_seeds": 36,
        "initialization": initialization,
        "gradient_reachability": gradients,
        "matched_controls": controls,
        "cpu_seconds": float(time.perf_counter() - started),
        "outer_test_accessed": False,
    }


def cache_preflight(args: argparse.Namespace) -> dict[str, Any]:
    configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    source_audit = audit_source_fbank(protocol)
    store, cache_audit = load_token_store(protocol)
    roles = pd.read_csv(REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    role_cells = validate_roles(protocol, store, roles)
    device = resolve_device(args.device)
    indices = fold_indices(store, roles, 0, 0)
    reconstructed, runtime_reconstruction = reconstruct_cached_A0(
        protocol, store, device
    )
    initialization = real_cache_initialization_audit(
        protocol,
        store,
        indices["train"],
        indices["validation"],
        reconstructed,
        BASE_SEEDS[0],
        device,
    )
    return {
        "status": "GO_FOR_FORMAL_GPU_RUN",
        "read_only": True,
        "device": str(device),
        "gpu_used": device.type == "cuda",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "source_audit": source_audit,
        "cache_audit": cache_audit,
        "runtime_A0_reconstruction": runtime_reconstruction,
        "real_cache_initialization": initialization,
        "role_cells": role_cells,
        "outer_test_accessed": False,
    }


def resolve_run_root(output_subdir: str) -> Path:
    runs = (REPO_ROOT / "runs").resolve()
    root = (runs / output_subdir).resolve()
    if runs not in root.parents:
        raise ValueError("--output-subdir must stay below runs")
    return root


def validate_completed_fit(
    fit: dict[str, Any],
    pipeline: str,
    base_seed: int,
    seed: int,
    repeat: int,
    fold: int,
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
        raise RuntimeError("IDEA-079 resume fit identity mismatch")
    for prefix in ("validation_animal", "validation_call"):
        path = REPO_ROOT / fit[f"{prefix}_predictions"]
        if not path.is_file() or sha256(path) != fit[f"{prefix}_sha256"]:
            raise RuntimeError(f"IDEA-079 resume prediction mismatch: {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    audit_source_fbank(protocol)
    store, cache_audit = load_token_store(protocol)
    roles = pd.read_csv(REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    validate_roles(protocol, store, roles)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("IDEA-079 formal 108-fit run is queued for GPU, not CPU")
    root = resolve_run_root(args.output_subdir)
    cache_preflight_path = root / "cache_preflight.json"
    if not cache_preflight_path.is_file():
        raise RuntimeError("IDEA-079 formal run requires a completed cache preflight")
    completed_preflight = read_json(cache_preflight_path)
    expected_preflight = {
        "status": "GO_FOR_FORMAL_GPU_RUN",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "outer_test_accessed": False,
    }
    if any(
        completed_preflight.get(key) != value
        for key, value in expected_preflight.items()
    ):
        raise RuntimeError("IDEA-079 cache preflight is stale or incompatible")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "cache_manifest_sha256": protocol["shared_cache"]["cache_manifest_sha256"],
        "git_revision": git_revision(),
        "outer_test_accessed": False,
        "model": protocol["model"],
        "cache_audit": cache_audit,
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
            raise RuntimeError("Existing IDEA-079 run manifest differs")
    else:
        write_json(manifest_path, manifest)
    completed = []
    for base_seed in protocol["model"]["base_seeds"]:
        for repeat in protocol["model"]["repeats"]:
            for fold in protocol["model"]["folds"]:
                indices = fold_indices(store, roles, repeat, fold)
                seed = full_seed(base_seed, repeat, fold)
                for pipeline in PIPELINES:
                    output = (
                        root
                        / "fits"
                        / pipeline
                        / f"base_seed_{base_seed}"
                        / f"repeat_{repeat}"
                        / f"fold_{fold}"
                    )
                    summary_path = output / "fit_summary.json"
                    if summary_path.exists():
                        if not args.resume:
                            raise FileExistsError(summary_path)
                        fit = read_json(summary_path)
                        validate_completed_fit(
                            fit, pipeline, base_seed, seed, repeat, fold
                        )
                        completed.append(fit)
                        continue
                    output.mkdir(parents=True, exist_ok=True)
                    print(
                        f"=== {pipeline} base_seed={base_seed} repeat={repeat} "
                        f"fold={fold} full_seed={seed} ===",
                        flush=True,
                    )
                    audit, animals, calls = fit_inner(
                        pipeline,
                        protocol,
                        store,
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
                        "base_seed": int(base_seed),
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
            "expected_fits": 108,
            "outer_test_accessed": False,
            "candidate_gate_passed": summary["candidate_gate_passed"],
            "spatial_mechanism_gate_passed": summary["spatial_mechanism_gate_passed"],
            "full_gate_passed": summary["full_gate_passed"],
        },
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.stage == "structural-preflight":
        result = structural_preflight(args)
        output_name = "structural_preflight.json"
    elif args.stage == "cache-preflight":
        result = cache_preflight(args)
        output_name = "cache_preflight.json"
    else:
        result = run(args)
        output_name = None
    if output_name is not None:
        output_path = resolve_run_root(args.output_subdir) / output_name
        if output_path.exists() and not args.resume:
            raise FileExistsError(
                f"Refusing to overwrite IDEA-079 preflight artifact: {output_path}"
            )
        result["output"] = repo_relative(output_path)
        write_json(output_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
