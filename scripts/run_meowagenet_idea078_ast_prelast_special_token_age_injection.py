"""Run the preregistered IDEA-078 pre-last AST special-token age injection study."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
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
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import ASTModel


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea078_ast_prelast_special_token_age_injection_v1.json"
)
IDEA077_RUNNER = REPO_ROOT / "scripts" / "run_meowagenet_idea077_ast_last4_layer_mix.py"
GEOMETRY_RUNNER = REPO_ROOT / "scripts" / "extract_ast_layer_embeddings.py"
HF_CACHE = REPO_ROOT / "data" / "models" / "huggingface"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


idea077 = load_module("idea078_idea077", IDEA077_RUNNER)
geometry_helper = load_module("idea078_geometry", GEOMETRY_RUNNER)

PIPELINES = ("A0_token_reconstruction", "P1_postpool_age_injection", "T1_prelast_special_token_injection")
BASE_SEEDS = (9763, 3230, 9726)
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
EXPECTED_PARAMETERS = {
    "A0_token_reconstruction": 99_075,
    "P1_postpool_age_injection": 107_733,
    "T1_prelast_special_token_injection": 107_733,
}
PROBE_CALL_INDICES = (0, 24, 114)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "run"), required=True)
    parser.add_argument(
        "--output-subdir",
        default="meowagenet_idea078_ast_prelast_special_token_age_injection_v1",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
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
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def full_seed(base_seed: int, repeat: int, fold: int) -> int:
    return int(base_seed + 10_000 * repeat + 100 * fold)


class FrozenASTTail(nn.Module):
    def __init__(self, last_block: nn.Module, final_layernorm: nn.Module) -> None:
        super().__init__()
        self.last_block = last_block
        self.final_layernorm = final_layernorm
        for parameter in self.parameters():
            parameter.requires_grad = False
        super().train(False)

    def train(self, mode: bool = True) -> "FrozenASTTail":
        super().train(False)
        return self

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.last_block(tokens, None, False)[0]
        hidden = self.final_layernorm(hidden)
        return (hidden[:, 0] + hidden[:, 1]) / 2


def load_ast_model(protocol: dict[str, Any], device: torch.device) -> ASTModel:
    ast = protocol["ast"]
    model = ASTModel.from_pretrained(
        ast["checkpoint"],
        revision=ast["revision"],
        cache_dir=HF_CACHE,
        use_safetensors=True,
        local_files_only=True,
    )
    locked = read_json(REPO_ROOT / protocol["data"]["locked_ast_protocol_path"])
    geometry_helper.adapt_standard_geometry(model, locked)
    if len(model.encoder.layer) != 12:
        raise RuntimeError("IDEA-078 expected exactly 12 AST blocks")
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.eval().to(device)


def take_tail(model: ASTModel) -> FrozenASTTail:
    return FrozenASTTail(model.encoder.layer[11], model.layernorm)


def manual_prelast_and_pooler(
    model: ASTModel, input_values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = model.embeddings(input_values)
    head_mask = model.get_head_mask(None, model.config.num_hidden_layers)
    for layer_index in range(11):
        layer_head_mask = head_mask[layer_index] if head_mask is not None else None
        hidden = model.encoder.layer[layer_index](hidden, layer_head_mask, False)[0]
    pooled = take_tail(model)(hidden)
    return hidden, pooled


def aggregate_segments(
    segment_values: torch.Tensor,
    segment_to_local_call: torch.Tensor,
    call_count: int,
) -> torch.Tensor:
    result = torch.zeros(
        (call_count, segment_values.shape[-1]),
        device=segment_values.device,
        dtype=segment_values.dtype,
    )
    result.index_add_(0, segment_to_local_call, segment_values)
    counts = torch.bincount(segment_to_local_call, minlength=call_count).to(
        segment_values.dtype
    )
    if torch.any(counts <= 0):
        raise RuntimeError("IDEA-078 call lost all segments")
    return result / counts[:, None]


@dataclass(frozen=True)
class TokenStore:
    prelast_tokens: np.ndarray
    segment_call_indices: np.ndarray
    segment_counts: np.ndarray
    call_segment_indices: tuple[np.ndarray, ...]
    frozen_embeddings: np.ndarray
    age_features: np.ndarray
    call_ids: np.ndarray
    cat_ids: np.ndarray
    labels: np.ndarray


def cache_paths(protocol: dict[str, Any]) -> tuple[Path, Path, Path]:
    root = REPO_ROOT / protocol["cache"]["output_root"]
    return (
        root / protocol["cache"]["token_filename"],
        root / protocol["cache"]["index_filename"],
        root / protocol["cache"]["summary_filename"],
    )


def load_store(protocol: dict[str, Any]) -> tuple[TokenStore, dict[str, Any]]:
    token_path, index_path, manifest_path = cache_paths(protocol)
    if not all(path.is_file() for path in (token_path, index_path, manifest_path)):
        raise FileNotFoundError("IDEA-078 label-free token cache is incomplete")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("label_information_used") is not False:
        raise RuntimeError("IDEA-078 cache manifest is not label-free and complete")
    cache = protocol["cache"]
    if manifest.get("protocol_sha256") != cache["producer_protocol_sha256"]:
        raise RuntimeError("IDEA-078 cache was produced under another pre-cache protocol")
    if sha256(manifest_path) != cache["manifest_sha256"]:
        raise RuntimeError("IDEA-078 cache manifest checksum mismatch")
    if manifest.get("extractor_sha256") != protocol["dependencies"]["cache_extractor_sha256"]:
        raise RuntimeError("IDEA-078 cache extractor hash mismatch")
    if (
        sha256(token_path) != manifest.get("token_sha256")
        or manifest.get("token_sha256") != cache["token_sha256"]
    ):
        raise RuntimeError("IDEA-078 token cache checksum mismatch")
    if (
        sha256(index_path) != manifest.get("index_sha256")
        or manifest.get("index_sha256") != cache["index_sha256"]
    ):
        raise RuntimeError("IDEA-078 token index checksum mismatch")
    tokens = np.load(token_path, mmap_mode="r")
    if list(tokens.shape) != protocol["cache"]["shape"] or tokens.dtype != np.float32:
        raise RuntimeError("IDEA-078 token cache geometry changed")
    with np.load(index_path, allow_pickle=False) as index:
        if set(index.files) != {
            "segment_call_indices",
            "segment_counts",
            "call_ids",
            "source_paths",
            "prelast_block_output_one_based",
            "last_block_input_one_based",
            "special_token_indices",
            "patch_token_start",
        }:
            raise RuntimeError("IDEA-078 token index schema changed")
        segment_call_indices = index["segment_call_indices"].astype(np.int64)
        segment_counts = index["segment_counts"].astype(np.int64)
        call_ids = index["call_ids"].astype(str)
        source_paths = index["source_paths"].astype(str)
        if int(index["prelast_block_output_one_based"][0]) != 11:
            raise RuntimeError("IDEA-078 cached the wrong block boundary")
    with np.load(REPO_ROOT / protocol["data"]["frozen_embedding_path"]) as frozen:
        frozen_embeddings = frozen["embeddings"].astype(np.float32)
        frozen_call_ids = frozen["call_ids"].astype(str)
        cat_ids = frozen["cat_ids"].astype(str)
        labels = frozen["labels"].astype(np.int64)
    with np.load(REPO_ROOT / protocol["data"]["age_feature_path"]) as ages:
        age_features = ages["features"].astype(np.float32)
        age_call_ids = ages["call_ids"].astype(str)
    with np.load(REPO_ROOT / protocol["data"]["fbank_path"], allow_pickle=False) as fbank:
        frozen_source_paths = fbank["source_paths"].astype(str)
    if not (
        np.array_equal(call_ids, frozen_call_ids)
        and np.array_equal(call_ids, age_call_ids)
        and np.array_equal(source_paths, frozen_source_paths)
    ):
        raise RuntimeError("IDEA-078 cache/final/age call order differs")
    if frozen_embeddings.shape != (792, 768) or age_features.shape != (792, 20):
        raise RuntimeError("IDEA-078 final or age feature geometry changed")
    observed = np.bincount(segment_call_indices, minlength=792)
    if not np.array_equal(observed, segment_counts) or int(segment_counts.sum()) != 843:
        raise RuntimeError("IDEA-078 segment mapping is not closed")
    call_segments = tuple(
        np.flatnonzero(segment_call_indices == call_index).astype(np.int64)
        for call_index in range(792)
    )
    return (
        TokenStore(
            tokens,
            segment_call_indices,
            segment_counts,
            call_segments,
            frozen_embeddings,
            age_features,
            call_ids,
            cat_ids,
            labels,
        ),
        manifest,
    )


def fold_indices(
    store: TokenStore, roles: pd.DataFrame, repeat: int, fold: int
) -> dict[str, np.ndarray]:
    cell = roles[(roles["repeat"] == repeat) & (roles["outer_fold"] == fold)]
    if cell["cat_id"].duplicated().any():
        raise RuntimeError("IDEA-078 role cell assigns a cat more than once")
    mapping = dict(zip(cell["cat_id"].astype(str), cell["role"].astype(str)))
    if set(mapping) != set(store.cat_ids.astype(str)):
        raise RuntimeError("IDEA-078 role cell does not cover exactly the cached cats")
    call_roles = np.asarray([mapping[cat_id] for cat_id in store.cat_ids], dtype=str)
    return {
        "train": np.flatnonzero(call_roles == "train"),
        "validation": np.flatnonzero(call_roles == "validation"),
    }


class DeterministicNoSingletonBatchSampler(Sampler[list[int]]):
    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        if size < 2 or batch_size < 2:
            raise ValueError("IDEA-078 batching needs at least two cats")
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
            raise RuntimeError("IDEA-078 singleton cat batch remains")
        yield from batches

    def __len__(self) -> int:
        return math.ceil(self.size / self.batch_size)


class CatDataset(Dataset[np.ndarray]):
    def __init__(self, store: TokenStore, call_indices: np.ndarray) -> None:
        selected = np.asarray(call_indices, dtype=np.int64)
        self.cat_ids = np.asarray(sorted(np.unique(store.cat_ids[selected]).tolist()))
        self.call_indices: list[np.ndarray] = []
        for cat_id in self.cat_ids:
            calls = np.sort(selected[store.cat_ids[selected] == cat_id])
            if len(np.unique(store.labels[calls])) != 1:
                raise RuntimeError(f"IDEA-078 cat {cat_id} has inconsistent labels")
            self.call_indices.append(calls)

    def __len__(self) -> int:
        return len(self.call_indices)

    def __getitem__(self, item: int) -> np.ndarray:
        return self.call_indices[item]


def collate_cats(rows: list[np.ndarray], store: TokenStore) -> dict[str, Any]:
    calls = np.concatenate(rows).astype(np.int64, copy=False)
    segment_parts = [store.call_segment_indices[int(call)] for call in calls]
    segments = np.concatenate(segment_parts).astype(np.int64, copy=False)
    segment_to_call = np.repeat(
        np.arange(len(calls), dtype=np.int64),
        np.asarray([len(part) for part in segment_parts], dtype=np.int64),
    )
    tokens = np.asarray(store.prelast_tokens[segments], dtype=np.float32).copy()
    return {
        "tokens": torch.from_numpy(tokens),
        "segment_to_local_call": torch.from_numpy(segment_to_call),
        "call_indices": torch.from_numpy(calls),
        "age_features": torch.from_numpy(store.age_features[calls].astype(np.float32, copy=False)),
        "cat_ids": [str(value) for value in store.cat_ids[calls]],
        "segment_indices": torch.from_numpy(segments),
    }


def build_loader(
    store: TokenStore,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = CatDataset(store, indices)
    collate = functools.partial(collate_cats, store=store)
    if shuffle:
        return DataLoader(
            dataset,
            batch_sampler=DeterministicNoSingletonBatchSampler(
                len(dataset), batch_size, seed
            ),
            num_workers=0,
            collate_fn=collate,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )


class TokenInjectionClassifier(nn.Module):
    def __init__(
        self,
        pipeline: str,
        tail: FrozenASTTail,
        ast_mean: np.ndarray,
        ast_scale: np.ndarray,
        age_train: np.ndarray,
        dropout: float,
        bottleneck: int,
    ) -> None:
        super().__init__()
        if pipeline not in PIPELINES:
            raise ValueError(pipeline)
        self.pipeline = pipeline
        self.tail = tail
        safe_ast_scale = np.where(ast_scale > 1.0e-12, ast_scale, 1.0).astype(np.float32)
        self.register_buffer("ast_mean", torch.from_numpy(ast_mean.astype(np.float32)))
        self.register_buffer("ast_scale", torch.from_numpy(safe_ast_scale))
        self.ast_linear = nn.Linear(768, 128)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(128, 3)
        if pipeline == PIPELINES[0]:
            self.age_hidden = None
            self.age_output = None
        else:
            with np.errstate(all="ignore"):
                median = np.nanmedian(age_train, axis=0)
            median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
            imputed = np.where(np.isfinite(age_train), age_train, median[None, :])
            age_mean = imputed.mean(axis=0).astype(np.float32)
            age_scale = imputed.std(axis=0).astype(np.float32)
            age_scale = np.where(age_scale > 1.0e-8, age_scale, 1.0).astype(np.float32)
            self.register_buffer("age_median", torch.from_numpy(median))
            self.register_buffer("age_mean", torch.from_numpy(age_mean))
            self.register_buffer("age_scale", torch.from_numpy(age_scale))
            self.age_hidden = nn.Linear(20, bottleneck)
            self.age_output = nn.Linear(bottleneck, 768)
            nn.init.zeros_(self.age_output.weight)
            nn.init.zeros_(self.age_output.bias)
        self.tail.train(False)

    def train(self, mode: bool = True) -> "TokenInjectionClassifier":
        super().train(mode)
        self.tail.train(False)
        return self

    def age_delta(self, age_features: torch.Tensor) -> torch.Tensor:
        if self.age_hidden is None or self.age_output is None:
            return torch.zeros(
                (len(age_features), 768),
                device=age_features.device,
                dtype=age_features.dtype,
            )
        imputed = torch.where(torch.isfinite(age_features), age_features, self.age_median)
        standardized = (imputed - self.age_mean) / self.age_scale
        return self.age_output(torch.nn.functional.gelu(self.age_hidden(standardized)))

    def representation(
        self,
        tokens: torch.Tensor,
        segment_to_local_call: torch.Tensor,
        age_features: torch.Tensor,
    ) -> torch.Tensor:
        call_count = len(age_features)
        delta = self.age_delta(age_features)
        if self.pipeline == PIPELINES[2]:
            conditioned = tokens.clone()
            segment_delta = delta[segment_to_local_call]
            conditioned[:, 0:2, :] = conditioned[:, 0:2, :] + segment_delta[:, None, :]
            segment_representations = self.tail(conditioned)
        else:
            segment_representations = self.tail(tokens)
        calls = aggregate_segments(
            segment_representations, segment_to_local_call, call_count
        )
        if self.pipeline == PIPELINES[1]:
            calls = calls + delta
        return calls

    def forward(
        self,
        tokens: torch.Tensor,
        segment_to_local_call: torch.Tensor,
        age_features: torch.Tensor,
    ) -> torch.Tensor:
        representation = self.representation(tokens, segment_to_local_call, age_features)
        normalized = (representation - self.ast_mean) / self.ast_scale
        hidden = self.relu(self.ast_linear(normalized))
        hidden = self.dropout(self.batch_norm(hidden))
        return self.output(hidden)

    def audit(self) -> dict[str, Any]:
        trainable = int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        frozen_tail = int(sum(p.numel() for p in self.tail.parameters()))
        result: dict[str, Any] = {
            "pipeline": self.pipeline,
            "trainable_parameters": trainable,
            "frozen_tail_parameters": frozen_tail,
            "injection_block_input_one_based": 12,
            "injected_token_indices": [0, 1] if self.pipeline == PIPELINES[2] else [],
            "patch_tokens_modified": False,
        }
        if self.age_output is not None:
            result["age_output_weight_norm"] = float(self.age_output.weight.detach().norm().cpu())
            result["age_output_bias_norm"] = float(self.age_output.bias.detach().norm().cpu())
        return result


def build_model(
    pipeline: str,
    protocol: dict[str, Any],
    store: TokenStore,
    train_indices: np.ndarray,
    tail: FrozenASTTail,
) -> TokenInjectionClassifier:
    final_train = store.frozen_embeddings[train_indices]
    return TokenInjectionClassifier(
        pipeline,
        tail,
        final_train.mean(axis=0),
        final_train.std(axis=0),
        store.age_features[train_indices],
        float(protocol["fixed_training"]["dropout"]),
        int(protocol["model"]["age_bottleneck"]),
    )


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("IDEA-078 training role is missing a class")
    return (len(labels) / (3.0 * counts)).astype(np.float32)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "tokens": batch["tokens"].to(device),
        "segment_to_local_call": batch["segment_to_local_call"].to(device),
        "call_indices": batch["call_indices"].to(device),
        "age_features": batch["age_features"].to(device),
        "cat_ids": batch["cat_ids"],
        "segment_indices": batch["segment_indices"].to(device),
    }


def train_one_epoch(
    model: TokenInjectionClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    store: TokenStore,
    weights: torch.Tensor,
    device: torch.device,
    gradient_clip: float,
) -> tuple[float, dict[str, Any]]:
    model.train()
    weighted_total = 0.0
    weight_total = 0.0
    processed_cats: list[str] = []
    processed_calls: list[int] = []
    processed_segments: list[int] = []
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        call_indices = batch["call_indices"]
        labels = torch.from_numpy(store.labels[call_indices.cpu().numpy()]).to(device)
        call_weights = weights[labels]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(
                batch["tokens"], batch["segment_to_local_call"], batch["age_features"]
            )
            losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            loss = torch.sum(losses * call_weights) / torch.sum(call_weights)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], gradient_clip
        )
        scaler.step(optimizer)
        scaler.update()
        weighted_total += float(torch.sum(losses.detach() * call_weights).cpu())
        weight_total += float(torch.sum(call_weights).cpu())
        processed_cats.extend(batch["cat_ids"])
        processed_calls.extend(call_indices.detach().cpu().tolist())
        processed_segments.extend(batch["segment_indices"].detach().cpu().tolist())
    expected_calls = sorted(loader.dataset.call_indices, key=lambda value: int(value[0]))
    expected_flat = sorted(np.concatenate(expected_calls).tolist())
    if sorted(processed_calls) != expected_flat:
        raise RuntimeError("IDEA-078 training call coverage changed")
    return weighted_total / weight_total, {
        "cats": len(processed_cats),
        "calls": len(processed_calls),
        "segments": len(processed_segments),
        "cat_order_sha256": hashlib.sha256("\n".join(processed_cats).encode()).hexdigest(),
        "call_coverage_sha256": hashlib.sha256(np.asarray(sorted(processed_calls), dtype=np.int32).tobytes()).hexdigest(),
        "segment_coverage_sha256": hashlib.sha256(np.asarray(sorted(processed_segments), dtype=np.int32).tobytes()).hexdigest(),
    }


def predict(
    model: TokenInjectionClassifier,
    store: TokenStore,
    indices: np.ndarray,
    cat_batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    loader = build_loader(store, indices, cat_batch_size, False, seed)
    rows = []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = model(
                    batch["tokens"],
                    batch["segment_to_local_call"],
                    batch["age_features"],
                )
                probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            call_indices = batch["call_indices"].cpu().numpy()
            for local, call_index in enumerate(call_indices):
                rows.append(
                    {
                        "call_index": int(call_index),
                        "call_id": str(store.call_ids[call_index]),
                        "cat_id": str(store.cat_ids[call_index]),
                        "true_label": int(store.labels[call_index]),
                        "prob_kitten": float(probabilities[local, 0]),
                        "prob_adult": float(probabilities[local, 1]),
                        "prob_senior": float(probabilities[local, 2]),
                    }
                )
    calls = pd.DataFrame(rows).sort_values("call_index").reset_index(drop=True)
    animals = idea077.calls_to_animals(calls)
    return animals, calls


def checkpoint_state(model: TokenInjectionClassifier) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("tail.")
    }


def restore_checkpoint(
    model: TokenInjectionClassifier, state: dict[str, torch.Tensor]
) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith("tail.") for key in missing):
        raise RuntimeError("IDEA-078 checkpoint omitted a trainable state")


def fit_inner(
    pipeline: str,
    protocol: dict[str, Any],
    store: TokenStore,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    tail: FrozenASTTail,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    model = build_model(pipeline, protocol, store, train_indices, tail).to(device)
    if model.audit()["trainable_parameters"] != EXPECTED_PARAMETERS[pipeline]:
        raise RuntimeError("IDEA-078 trained parameter count changed")
    fixed = protocol["fixed_training"]
    training_seed = seed + int(fixed["post_build_seed_offset"])
    set_seed(training_seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adamax(
        parameters,
        lr=float(fixed["learning_rate"]),
        eps=float(fixed["optimizer_epsilon"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loader = build_loader(
        store, train_indices, int(fixed["cat_batch_size"]), True, seed
    )
    weights = torch.from_numpy(class_weights(store.labels[train_indices])).to(device)
    best_loss = float("inf")
    best_epoch = 1
    best_state = checkpoint_state(model)
    best_animals: pd.DataFrame | None = None
    best_calls: pd.DataFrame | None = None
    history = []
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, int(fixed["maximum_epochs"]) + 1):
        train_loss, train_audit = train_one_epoch(
            model,
            loader,
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
        validation_loss = idea077.animal_cross_entropy(animals)
        metrics = idea077.animal_metrics(animals)
        history.append(
            {
                "epoch": epoch,
                "train_call_loss": train_loss,
                "train_audit": train_audit,
                "validation_animal_cross_entropy": validation_loss,
                "validation_animal_brier": idea077.brier(animals),
                "validation_animal_metrics": metrics,
                "model": model.audit(),
            }
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = checkpoint_state(model)
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
        raise RuntimeError("IDEA-078 selected no checkpoint")
    restore_checkpoint(model, best_state)
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
        "best_validation_animal_brier": idea077.brier(best_animals),
        "best_validation_animal_metrics": idea077.animal_metrics(best_animals),
        "checkpoint_reload_max_probability_difference": reload_difference,
        "best_model": model.audit(),
        "train_seconds": float(time.perf_counter() - started),
        "history": history,
        "outer_test_accessed": False,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return audit, best_animals, best_calls


def head_state(model: TokenInjectionClassifier) -> dict[str, torch.Tensor]:
    prefixes = ("ast_linear.", "batch_norm.", "output.")
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key.startswith(prefixes)
    }


def injection_state(model: TokenInjectionClassifier) -> dict[str, torch.Tensor]:
    prefixes = ("age_hidden.", "age_output.", "age_median", "age_mean", "age_scale")
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key.startswith(prefixes)
    }


def states_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[key], right[key]) for key in left
    )


def initialization_audit(
    protocol: dict[str, Any],
    store: TokenStore,
    train_indices: np.ndarray,
    probe_indices: np.ndarray,
    tail: FrozenASTTail,
    seed: int,
) -> dict[str, Any]:
    device = next(tail.parameters()).device
    probe_set = set(int(value) for value in probe_indices)
    segment_indices = np.concatenate(
        [store.call_segment_indices[index] for index in probe_indices]
    )
    segment_to_call = np.concatenate(
        [np.full(len(store.call_segment_indices[index]), local, dtype=np.int64) for local, index in enumerate(probe_indices)]
    )
    tokens = torch.from_numpy(
        np.asarray(store.prelast_tokens[segment_indices]).copy()
    ).to(device)
    mapping = torch.from_numpy(segment_to_call).to(device)
    ages = torch.from_numpy(store.age_features[probe_indices]).to(device)
    labels = torch.from_numpy(store.labels[probe_indices]).to(device)
    models: dict[str, TokenInjectionClassifier] = {}
    logits: dict[str, torch.Tensor] = {}
    for pipeline in PIPELINES:
        set_seed(seed)
        model = build_model(pipeline, protocol, store, train_indices, tail).to(device).eval()
        models[pipeline] = model
        logits[pipeline] = model(tokens, mapping, ages)
    parameters = {
        pipeline: model.audit()["trainable_parameters"]
        for pipeline, model in models.items()
    }
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"IDEA-078 parameter mismatch: {parameters}")
    head_equal = all(
        states_equal(head_state(models[PIPELINES[0]]), head_state(models[pipeline]))
        for pipeline in PIPELINES[1:]
    )
    injection_equal = states_equal(
        injection_state(models[PIPELINES[1]]), injection_state(models[PIPELINES[2]])
    )
    max_differences = {
        pipeline: float(torch.max(torch.abs(logits[pipeline] - logits[PIPELINES[0]])))
        for pipeline in PIPELINES[1:]
    }
    losses = {
        pipeline: float(torch.nn.functional.cross_entropy(value, labels))
        for pipeline, value in logits.items()
    }
    if (
        any(value != 0.0 for value in max_differences.values())
        or not head_equal
        or not injection_equal
        or any(losses[pipeline] != losses[PIPELINES[0]] for pipeline in PIPELINES[1:])
    ):
        raise RuntimeError("IDEA-078 paired initialization failed")
    return {
        "probe_calls": len(probe_set),
        "probe_segments": len(segment_indices),
        "trainable_parameters": parameters,
        "shared_head_state_equal": head_equal,
        "P1_T1_injection_state_equal": injection_equal,
        "max_initial_logit_differences_vs_A0": max_differences,
        "initial_loss": losses,
    }


def gradient_reachability_audit(
    protocol: dict[str, Any],
    store: TokenStore,
    train_indices: np.ndarray,
    probe_indices: np.ndarray,
    tail: FrozenASTTail,
    seed: int,
) -> dict[str, Any]:
    device = next(tail.parameters()).device
    segment_indices = np.concatenate([store.call_segment_indices[index] for index in probe_indices])
    mapping = np.concatenate(
        [np.full(len(store.call_segment_indices[index]), local, dtype=np.int64) for local, index in enumerate(probe_indices)]
    )
    tokens = torch.from_numpy(
        np.asarray(store.prelast_tokens[segment_indices]).copy()
    ).to(device)
    mapping_tensor = torch.from_numpy(mapping).to(device)
    ages = torch.from_numpy(store.age_features[probe_indices]).to(device)
    labels = torch.from_numpy(store.labels[probe_indices]).to(device)
    set_seed(seed)
    model = build_model(PIPELINES[2], protocol, store, train_indices, tail).to(device).eval()
    loss = torch.nn.functional.cross_entropy(model(tokens, mapping_tensor, ages), labels)
    loss.backward()
    if model.age_hidden is None or model.age_output is None:
        raise RuntimeError("IDEA-078 T1 age branch is missing")
    output_gradient = float(model.age_output.weight.grad.abs().max())
    hidden_gradient_at_zero = float(model.age_hidden.weight.grad.abs().max())
    tail_gradients = sum(parameter.grad is not None for parameter in model.tail.parameters())
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.age_output.weight.fill_(1.0e-4)
    second_loss = torch.nn.functional.cross_entropy(
        model(tokens, mapping_tensor, ages), labels
    )
    second_loss.backward()
    hidden_gradient_after_probe = float(model.age_hidden.weight.grad.abs().max())
    if (
        not np.isfinite(output_gradient)
        or output_gradient <= 0.0
        or hidden_gradient_at_zero != 0.0
        or not np.isfinite(hidden_gradient_after_probe)
        or hidden_gradient_after_probe <= 0.0
        or tail_gradients != 0
    ):
        raise RuntimeError("IDEA-078 gradient schedule failed")
    return {
        "age_output_max_gradient_at_zero": output_gradient,
        "age_hidden_max_gradient_at_zero": hidden_gradient_at_zero,
        "age_hidden_max_gradient_after_output_probe": hidden_gradient_after_probe,
        "frozen_tail_parameters_with_grad": tail_gradients,
    }


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea078-ast-prelast-special-token-age-injection-v1":
        raise RuntimeError("Unexpected IDEA-078 protocol")
    if protocol.get("status") != "locked_for_formal_evaluation":
        raise RuntimeError("IDEA-078 formal protocol is not locked")
    model = protocol["model"]
    if tuple(model["pipelines"]) != PIPELINES or tuple(model["base_seeds"]) != BASE_SEEDS:
        raise RuntimeError("IDEA-078 pipelines or seeds changed")
    if model["repeats"] != [0, 1, 2] or model["folds"] != [0, 1, 2, 3]:
        raise RuntimeError("IDEA-078 split scope changed")
    if model["outer_test_predictions"] is not False:
        raise RuntimeError("IDEA-078 must not access outer test predictions")
    if model["trainable_parameters"] != EXPECTED_PARAMETERS:
        raise RuntimeError("IDEA-078 parameter lock changed")
    if int(model["total_fits"]) != 108 or int(model["primary_A0_T1_fits"]) != 72:
        raise RuntimeError("IDEA-078 fit budget changed")
    if int(model["age_bottleneck"]) != 10 or model["special_token_indices"] != [0, 1]:
        raise RuntimeError("IDEA-078 injection formula changed")
    if model["patch_tokens_modified"] is not False:
        raise RuntimeError("IDEA-078 patch tokens must not be directly modified")
    full = {
        full_seed(base, repeat, fold)
        for base in BASE_SEEDS
        for repeat in range(3)
        for fold in range(4)
    }
    if len(full) != 36:
        raise RuntimeError("IDEA-078 full seeds are not unique")
    excluded_bases = set(
        int(value) for value in model["excluded_base_seeds_IDEA068_through_IDEA077"]
    )
    excluded = {
        full_seed(base, repeat, fold)
        for base in excluded_bases
        for repeat in range(3)
        for fold in range(5)
    }
    if full & excluded:
        raise RuntimeError("IDEA-078 full seed collision")
    dependencies = protocol["dependencies"]
    checks = {
        "plan_sha256": REPO_ROOT / dependencies["plan_path"],
        "cache_schema_sha256": REPO_ROOT / dependencies["cache_schema_path"],
        "runner_sha256": Path(__file__).resolve(),
        "cache_extractor_sha256": REPO_ROOT / dependencies["cache_extractor_path"],
        "idea077_runner_sha256": IDEA077_RUNNER,
        "geometry_runner_sha256": GEOMETRY_RUNNER,
        "fbank_sha256": REPO_ROOT / protocol["data"]["fbank_path"],
        "locked_embedding_sha256": REPO_ROOT / protocol["data"]["frozen_embedding_path"],
        "age_feature_sha256": REPO_ROOT / protocol["data"]["age_feature_path"],
        "roles_sha256": REPO_ROOT / protocol["data"]["roles_path"],
        "data_manifest_sha256": REPO_ROOT / protocol["data"]["data_manifest_path"],
        "audio_checksums_sha256": REPO_ROOT / protocol["data"]["audio_checksums_path"],
    }
    for field, path in checks.items():
        if sha256(path) != dependencies[field]:
            raise RuntimeError(f"IDEA-078 dependency hash changed: {field}")


def validate_roles(
    protocol: dict[str, Any], call_ids: np.ndarray, cat_ids: np.ndarray, roles: pd.DataFrame
) -> int:
    if len(call_ids) != 792 or len(np.unique(cat_ids)) != 111:
        raise RuntimeError("IDEA-078 expected 792 calls and 111 cats")
    cells = 0
    for repeat in range(3):
        for fold in range(4):
            cell = roles[(roles["repeat"] == repeat) & (roles["outer_fold"] == fold)]
            if len(cell) != 111 or cell["cat_id"].duplicated().any():
                raise RuntimeError("IDEA-078 malformed role cell")
            sets = {
                role: set(cell.loc[cell["role"] == role, "cat_id"].astype(str))
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
                raise RuntimeError("IDEA-078 cat leakage across roles")
            cells += 1
    return cells


def live_cpu_probe(protocol: dict[str, Any]) -> tuple[dict[str, Any], TokenStore, FrozenASTTail]:
    with np.load(REPO_ROOT / protocol["data"]["fbank_path"]) as fbank:
        features = fbank["features"].astype(np.float32)
        mapping = fbank["segment_call_indices"].astype(np.int64)
        counts = fbank["segment_counts"].astype(np.int64)
        call_ids = fbank["call_ids"].astype(str)
        cat_ids = fbank["cat_ids"].astype(str)
        labels = fbank["labels"].astype(np.int64)
    with np.load(REPO_ROOT / protocol["data"]["frozen_embedding_path"]) as frozen:
        locked_final = frozen["embeddings"].astype(np.float32)
        if not np.array_equal(call_ids, frozen["call_ids"].astype(str)):
            raise RuntimeError("IDEA-078 fbank/final call order differs")
    with np.load(REPO_ROOT / protocol["data"]["age_feature_path"]) as ages:
        age_features = ages["features"].astype(np.float32)
        if not np.array_equal(call_ids, ages["call_ids"].astype(str)):
            raise RuntimeError("IDEA-078 fbank/age call order differs")
    probe_calls = np.asarray(PROBE_CALL_INDICES, dtype=np.int64)
    selected_segments = np.flatnonzero(np.isin(mapping, probe_calls))
    local_by_global = {int(value): index for index, value in enumerate(probe_calls)}
    local_mapping = np.asarray(
        [local_by_global[int(mapping[index])] for index in selected_segments], dtype=np.int64
    )
    model = load_ast_model(protocol, torch.device("cpu"))
    batch = torch.from_numpy(features[selected_segments])
    with torch.inference_mode():
        canonical = model(input_values=batch).pooler_output
        prelast, reconstructed = manual_prelast_and_pooler(model, batch)
    same_forward_max = float(torch.max(torch.abs(canonical - reconstructed)))
    call_reconstructed = aggregate_segments(
        reconstructed, torch.from_numpy(local_mapping), len(probe_calls)
    ).cpu().numpy()
    locked_difference = np.abs(call_reconstructed - locked_final[probe_calls])
    tolerance = protocol["preflight"]["cpu_probe_vs_locked_final_tolerance"]
    if same_forward_max != 0.0:
        raise RuntimeError("IDEA-078 manual AST tail differs from canonical CPU forward")
    if (
        float(locked_difference.mean()) > float(tolerance["mean_absolute_maximum"])
        or float(locked_difference.max()) > float(tolerance["absolute_maximum"])
    ):
        raise RuntimeError(
            "IDEA-078 CPU probe differs from locked final pooler: "
            f"mean_abs={float(locked_difference.mean()):.12g}, "
            f"max_abs={float(locked_difference.max()):.12g}, tolerance={tolerance}"
        )
    probe_tokens = prelast.cpu().numpy().astype(np.float32)
    probe_segment_counts = counts[probe_calls]
    call_segments = tuple(
        np.flatnonzero(local_mapping == local).astype(np.int64)
        for local in range(len(probe_calls))
    )
    probe_store = TokenStore(
        probe_tokens,
        local_mapping,
        probe_segment_counts,
        call_segments,
        locked_final[probe_calls],
        age_features[probe_calls],
        call_ids[probe_calls],
        cat_ids[probe_calls],
        labels[probe_calls],
    )
    tail = take_tail(model)
    audit = {
        "probe_call_indices": list(PROBE_CALL_INDICES),
        "probe_calls": int(len(probe_calls)),
        "probe_segments": int(len(selected_segments)),
        "prelast_shape": list(prelast.shape),
        "canonical_vs_manual_tail_max_absolute_difference": same_forward_max,
        "reconstructed_calls_vs_locked_final_mean_absolute_difference": float(
            locked_difference.mean()
        ),
        "reconstructed_calls_vs_locked_final_maximum_absolute_difference": float(
            locked_difference.max()
        ),
        "tolerance": tolerance,
        "elementwise_within_tolerance": True,
    }
    return audit, probe_store, tail


def full_cache_reconstruction_audit(
    protocol: dict[str, Any], store: TokenStore, tail: FrozenASTTail
) -> dict[str, Any]:
    segment_outputs = []
    batch_size = int(protocol["preflight"]["full_cache_cpu_tail_batch_size"])
    with torch.inference_mode():
        for start in range(0, len(store.prelast_tokens), batch_size):
            stop = min(start + batch_size, len(store.prelast_tokens))
            batch = torch.from_numpy(np.asarray(store.prelast_tokens[start:stop]).copy())
            segment_outputs.append(tail(batch).cpu().numpy())
    segments = np.concatenate(segment_outputs).astype(np.float64)
    calls = np.zeros((792, 768), dtype=np.float64)
    np.add.at(calls, store.segment_call_indices, segments)
    calls /= store.segment_counts[:, None]
    difference = np.abs(calls.astype(np.float32) - store.frozen_embeddings)
    tolerance = protocol["preflight"]["full_cache_vs_locked_final_tolerance"]
    if (
        float(difference.mean()) > float(tolerance["mean_absolute_maximum"])
        or float(difference.max()) > float(tolerance["absolute_maximum"])
    ):
        raise RuntimeError("IDEA-078 full cache reconstruction differs from locked final")
    return {
        "segments": 843,
        "calls": 792,
        "mean_absolute_difference": float(difference.mean()),
        "maximum_absolute_difference": float(difference.max()),
        "elementwise_within_tolerance": True,
        "tolerance": tolerance,
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cpu":
        raise RuntimeError("IDEA-078 preparation preflight is CPU-only")
    configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    roles = pd.read_csv(
        REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str}
    )
    live_audit, probe_store, tail = live_cpu_probe(protocol)
    train_roles = roles[(roles["repeat"] == 0) & (roles["outer_fold"] == 0)]
    train_cats = set(train_roles.loc[train_roles["role"] == "train", "cat_id"].astype(str))
    with np.load(REPO_ROOT / protocol["data"]["frozen_embedding_path"]) as frozen:
        all_call_ids = frozen["call_ids"].astype(str)
        all_cat_ids = frozen["cat_ids"].astype(str)
        all_labels = frozen["labels"].astype(np.int64)
        all_final = frozen["embeddings"].astype(np.float32)
    with np.load(REPO_ROOT / protocol["data"]["age_feature_path"]) as ages:
        all_ages = ages["features"].astype(np.float32)
    role_cells = validate_roles(protocol, all_call_ids, all_cat_ids, roles)
    train_indices = np.flatnonzero(np.isin(all_cat_ids, list(train_cats)))
    # Remap compact probe calls to their global indices for model inputs.
    global_probe = np.asarray(PROBE_CALL_INDICES, dtype=np.int64)
    def compact_initialization() -> tuple[dict[str, Any], dict[str, Any]]:
        tokens = torch.from_numpy(probe_store.prelast_tokens.copy())
        mapping = torch.from_numpy(probe_store.segment_call_indices.copy())
        age_probe = torch.from_numpy(all_ages[global_probe])
        labels_probe = torch.from_numpy(all_labels[global_probe])
        models = {}
        logits = {}
        for pipeline in PIPELINES:
            set_seed(BASE_SEEDS[0])
            model = TokenInjectionClassifier(
                pipeline,
                tail,
                all_final[train_indices].mean(axis=0),
                all_final[train_indices].std(axis=0),
                all_ages[train_indices],
                float(protocol["fixed_training"]["dropout"]),
                int(protocol["model"]["age_bottleneck"]),
            ).eval()
            models[pipeline] = model
            logits[pipeline] = model(tokens, mapping, age_probe)
        parameters = {name: model.audit()["trainable_parameters"] for name, model in models.items()}
        head_equal = all(states_equal(head_state(models[PIPELINES[0]]), head_state(models[name])) for name in PIPELINES[1:])
        injection_equal = states_equal(injection_state(models[PIPELINES[1]]), injection_state(models[PIPELINES[2]]))
        diffs = {name: float(torch.max(torch.abs(logits[name] - logits[PIPELINES[0]]))) for name in PIPELINES[1:]}
        losses = {name: float(torch.nn.functional.cross_entropy(value, labels_probe)) for name, value in logits.items()}
        if parameters != EXPECTED_PARAMETERS or not head_equal or not injection_equal or any(value != 0.0 for value in diffs.values()):
            raise RuntimeError("IDEA-078 live initialization audit failed")
        t1 = models[PIPELINES[2]]
        loss = torch.nn.functional.cross_entropy(logits[PIPELINES[2]], labels_probe)
        loss.backward()
        assert t1.age_hidden is not None and t1.age_output is not None
        out_grad = float(t1.age_output.weight.grad.abs().max())
        hidden_zero = float(t1.age_hidden.weight.grad.abs().max())
        tail_grads = sum(parameter.grad is not None for parameter in t1.tail.parameters())
        t1.zero_grad(set_to_none=True)
        with torch.no_grad():
            t1.age_output.weight.fill_(1.0e-4)
        torch.nn.functional.cross_entropy(t1(tokens, mapping, age_probe), labels_probe).backward()
        hidden_after = float(t1.age_hidden.weight.grad.abs().max())
        if out_grad <= 0 or hidden_zero != 0 or hidden_after <= 0 or tail_grads != 0:
            raise RuntimeError("IDEA-078 live gradient audit failed")
        return (
            {
                "trainable_parameters": parameters,
                "shared_head_state_equal": head_equal,
                "P1_T1_injection_state_equal": injection_equal,
                "max_initial_logit_differences_vs_A0": diffs,
                "initial_loss": losses,
            },
            {
                "age_output_max_gradient_at_zero": out_grad,
                "age_hidden_max_gradient_at_zero": hidden_zero,
                "age_hidden_max_gradient_after_output_probe": hidden_after,
                "frozen_tail_parameters_with_grad": tail_grads,
            },
        )
    initialization, gradients = compact_initialization()
    token_path, _, _ = cache_paths(protocol)
    cache_exists = token_path.is_file()
    cache_manifest = None
    full_reconstruction = None
    if cache_exists:
        store, cache_manifest = load_store(protocol)
        model = load_ast_model(protocol, torch.device("cpu"))
        cache_tail = take_tail(model)
        full_reconstruction = full_cache_reconstruction_audit(protocol, store, cache_tail)
        formal_ready = True
        status = "GO_FOR_FORMAL_GPU_RUN"
    else:
        formal_ready = False
        status = "GO_FOR_LABEL_FREE_CACHE_EXTRACTION"
    raw_bytes = int(np.prod(protocol["cache"]["shape"]) * 4)
    return {
        "status": status,
        "read_only": True,
        "device": "cpu",
        "gpu_used": False,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "calls": 792,
        "cats": 111,
        "segments": 843,
        "role_cells": role_cells,
        "outer_test_accessed": False,
        "cache": {
            "required": True,
            "exists": cache_exists,
            "formal_training_ready": formal_ready,
            "schema": protocol["cache"],
            "raw_token_bytes": raw_bytes,
            "raw_token_mib": raw_bytes / (1024**2),
            "manifest": cache_manifest,
        },
        "live_CPU_A0_reconstruction": live_audit,
        "full_cache_A0_reconstruction": full_reconstruction,
        "initialization": initialization,
        "gradient_reachability": gradients,
        "base_seeds": list(BASE_SEEDS),
        "unique_full_seeds": 36,
        "expected_fits": 108,
        "primary_A0_T1_fits": 72,
        "formal_GPU_run_authorized": formal_ready,
    }


def metric_bundle(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "metrics": idea077.animal_metrics(frame),
        "cross_entropy": idea077.animal_cross_entropy(frame),
        "brier": idea077.brier(frame),
    }


def add_metrics(row: dict[str, Any], bundles: dict[str, Any]) -> None:
    for pipeline in PIPELINES:
        row[f"{pipeline}_macro_f1"] = bundles[pipeline]["metrics"]["macro_f1"]
        row[f"{pipeline}_balanced_accuracy"] = bundles[pipeline]["metrics"]["balanced_accuracy"]
        row[f"{pipeline}_cross_entropy"] = bundles[pipeline]["cross_entropy"]
        row[f"{pipeline}_brier"] = bundles[pipeline]["brier"]
        row[f"{pipeline}_senior_recall"] = bundles[pipeline]["metrics"]["per_class"]["senior"]["recall"]
    row["T1_minus_A0_macro_f1"] = row[f"{PIPELINES[2]}_macro_f1"] - row[f"{PIPELINES[0]}_macro_f1"]
    row["P1_minus_A0_macro_f1"] = row[f"{PIPELINES[1]}_macro_f1"] - row[f"{PIPELINES[0]}_macro_f1"]
    row["T1_minus_P1_macro_f1"] = row[f"{PIPELINES[2]}_macro_f1"] - row[f"{PIPELINES[1]}_macro_f1"]


def describe(values: pd.Series) -> dict[str, Any]:
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "median": float(values.median()),
        "positive": int((values > 0).sum()),
        "tied": int((values == 0).sum()),
        "negative": int((values < 0).sum()),
        "worst": float(values.min()),
        "best": float(values.max()),
    }


def aggregate(fits: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    by_key = {
        (fit["pipeline"], fit["base_seed"], fit["repeat"], fit["fold"]): fit
        for fit in fits
    }
    fold_rows = []
    seed_repeat_rows = []
    pooled_all: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
    for base_seed in BASE_SEEDS:
        for repeat in range(3):
            grouped: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
            for fold in range(4):
                frames = {
                    pipeline: pd.read_csv(
                        REPO_ROOT / by_key[(pipeline, base_seed, repeat, fold)]["validation_animal_predictions"],
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
                    row, {pipeline: metric_bundle(frame) for pipeline, frame in frames.items()}
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
                row, {pipeline: metric_bundle(frame) for pipeline, frame in pooled.items()}
            )
            seed_repeat_rows.append(row)
    folds = pd.DataFrame(fold_rows)
    seed_repeats = pd.DataFrame(seed_repeat_rows)
    comparisons = (
        "T1_minus_A0_macro_f1",
        "P1_minus_A0_macro_f1",
        "T1_minus_P1_macro_f1",
    )
    splits = folds.groupby(["repeat", "fold"], as_index=False)[list(comparisons)].mean()
    means = {
        metric: {
            pipeline: float(seed_repeats[f"{pipeline}_{metric}"].mean())
            for pipeline in PIPELINES
        }
        for metric in ("macro_f1", "balanced_accuracy", "cross_entropy", "brier")
    }
    per_seed = {
        contrast: {
            str(seed): float(
                seed_repeats[seed_repeats["base_seed"] == seed][contrast].mean()
            )
            for seed in BASE_SEEDS
        }
        for contrast in comparisons
    }
    pooled_frames = {
        pipeline: pd.concat(parts, ignore_index=True)
        for pipeline, parts in pooled_all.items()
    }
    per_seed_senior = {}
    for seed in BASE_SEEDS:
        senior = {
            pipeline: idea077.animal_metrics(
                frame[frame["base_seed"] == seed]
            )["per_class"]["senior"]["recall"]
            for pipeline, frame in pooled_frames.items()
        }
        per_seed_senior[str(seed)] = float(senior[PIPELINES[2]] - senior[PIPELINES[0]])
    main_delta = seed_repeats["T1_minus_A0_macro_f1"]
    main_split = splits["T1_minus_A0_macro_f1"]
    mechanism_delta = seed_repeats["T1_minus_P1_macro_f1"]
    mechanism_split = splits["T1_minus_P1_macro_f1"]
    gate = protocol["gate"]
    main_conditions = {
        "mean_macro_f1_gain": float(main_delta.mean()) >= float(gate["main"]["minimum_mean_T1_minus_A0"]),
        "positive_base_seed_means": sum(value > 0 for value in per_seed["T1_minus_A0_macro_f1"].values()) >= int(gate["main"]["minimum_positive_base_seeds"]),
        "positive_seed_repeats": int((main_delta > 0).sum()) >= int(gate["main"]["minimum_positive_seed_repeats"]),
        "nonnegative_split_cells": int((main_split >= 0).sum()) >= int(gate["main"]["minimum_nonnegative_split_cells"]),
        "worst_split_cell": float(main_split.min()) >= float(gate["main"]["minimum_worst_split_cell_delta"]),
        "cross_entropy_nonworse": means["cross_entropy"][PIPELINES[2]] <= means["cross_entropy"][PIPELINES[0]],
        "brier_nonworse": means["brier"][PIPELINES[2]] <= means["brier"][PIPELINES[0]],
        "per_base_seed_senior_recall": all(value >= float(gate["main"]["minimum_per_base_seed_senior_recall_delta"]) for value in per_seed_senior.values()),
    }
    main_passed = bool(all(main_conditions.values()))
    mechanism_conditions = {
        "mean_macro_f1_strictly_positive": float(mechanism_delta.mean()) > 0.0,
        "positive_base_seed_means": sum(value > 0 for value in per_seed["T1_minus_P1_macro_f1"].values()) >= int(gate["mechanism"]["minimum_positive_base_seeds"]),
        "positive_seed_repeats": int((mechanism_delta > 0).sum()) >= int(gate["mechanism"]["minimum_positive_seed_repeats"]),
        "nonnegative_split_cells": int((mechanism_split >= 0).sum()) >= int(gate["mechanism"]["minimum_nonnegative_split_cells"]),
        "worst_split_cell": float(mechanism_split.min()) >= float(gate["mechanism"]["minimum_worst_split_cell_delta"]),
        "cross_entropy_nonworse": means["cross_entropy"][PIPELINES[2]] <= means["cross_entropy"][PIPELINES[1]],
        "brier_nonworse": means["brier"][PIPELINES[2]] <= means["brier"][PIPELINES[1]],
    }
    mechanism_raw = bool(all(mechanism_conditions.values()))
    return {
        "status": "complete",
        "outer_test_accessed": False,
        "fits": len(fits),
        "seed_repeat_estimates": len(seed_repeat_rows),
        "split_cell_estimates": int(len(splits)),
        "independence_note": "Repeated validation animals across seeds and repeats are descriptive occurrences, not independent samples.",
        "fold_results": fold_rows,
        "seed_repeat_results": seed_repeat_rows,
        "split_cell_results": splits.to_dict(orient="records"),
        "seed_repeat_equal_weight_means": means,
        "comparisons": {
            contrast: describe(seed_repeats[contrast]) for contrast in comparisons
        },
        "per_base_seed_mean_deltas": per_seed,
        "per_base_seed_T1_minus_A0_senior_recall": per_seed_senior,
        "pooled_validation": {
            pipeline: metric_bundle(frame) for pipeline, frame in pooled_frames.items()
        },
        "main_gate_conditions": main_conditions,
        "main_gate_passed": main_passed,
        "mechanism_gate_conditions": mechanism_conditions,
        "mechanism_gate_raw_passed": mechanism_raw,
        "mechanism_gate_interpretable": main_passed,
        "mechanism_gate_passed": bool(main_passed and mechanism_raw),
        "gate_passed": main_passed,
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
        raise RuntimeError("IDEA-078 resume fit identity mismatch")
    for prefix in ("validation_animal", "validation_call"):
        path = REPO_ROOT / fit[f"{prefix}_predictions"]
        if not path.is_file() or sha256(path) != fit[f"{prefix}_sha256"]:
            raise RuntimeError(f"IDEA-078 resume prediction mismatch: {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    store, cache_manifest = load_store(protocol)
    roles = pd.read_csv(
        REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str}
    )
    validate_roles(protocol, store.call_ids, store.cat_ids, roles)
    device = resolve_device(args.device)
    ast_model = load_ast_model(protocol, device)
    tail = take_tail(ast_model).to(device)
    root = resolve_run_root(args.output_subdir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "git_revision": git_revision(),
        "outer_test_accessed": False,
        "model": protocol["model"],
        "cache_manifest_path": protocol["cache"]["manifest_path"],
        "cache_manifest_sha256": sha256(REPO_ROOT / protocol["cache"]["manifest_path"]),
        "cache_token_sha256": cache_manifest["token_sha256"],
        "cache_index_sha256": cache_manifest["index_sha256"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        if not args.resume or read_json(manifest_path) != manifest:
            raise RuntimeError("Existing IDEA-078 run manifest differs")
    else:
        write_json(manifest_path, manifest)
    completed = []
    for base_seed in BASE_SEEDS:
        for repeat in range(3):
            for fold in range(4):
                indices = fold_indices(store, roles, repeat, fold)
                seed = full_seed(base_seed, repeat, fold)
                init = initialization_audit(
                    protocol,
                    store,
                    indices["train"],
                    indices["validation"][: min(16, len(indices["validation"]))],
                    tail,
                    seed,
                )
                triplet = []
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
                    else:
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
                            tail,
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
                            "initialization_audit": init,
                            "audit": audit,
                            "validation_animal_predictions": repo_relative(animal_path),
                            "validation_animal_sha256": sha256(animal_path),
                            "validation_call_predictions": repo_relative(call_path),
                            "validation_call_sha256": sha256(call_path),
                        }
                        write_json(summary_path, fit)
                    completed.append(fit)
                    triplet.append(fit)
                common_epochs = min(len(item["audit"]["history"]) for item in triplet)
                for epoch in range(common_epochs):
                    reference = triplet[0]["audit"]["history"][epoch]["train_audit"]
                    for candidate in triplet[1:]:
                        current = candidate["audit"]["history"][epoch]["train_audit"]
                        for field in (
                            "cat_order_sha256",
                            "call_coverage_sha256",
                            "segment_coverage_sha256",
                        ):
                            if current[field] != reference[field]:
                                raise RuntimeError("IDEA-078 paired batch coverage differs")
    summary = aggregate(completed, protocol)
    summary_path = root / "initial_evaluation_summary.json"
    expected_bytes = canonical_json_bytes(summary)
    if summary_path.exists():
        if not args.resume:
            raise FileExistsError(summary_path)
        if read_json(summary_path) != summary:
            raise RuntimeError("IDEA-078 resumed aggregate semantic mismatch")
        if summary_path.read_bytes() != expected_bytes:
            raise RuntimeError("IDEA-078 resumed aggregate canonical-byte mismatch")
    else:
        write_json(summary_path, summary)
    compact = {
        "status": "complete",
        "completed_fits": len(completed),
        "expected_fits": 108,
        "outer_test_accessed": False,
        "main_gate_passed": summary["main_gate_passed"],
        "mechanism_gate_interpretable": summary["mechanism_gate_interpretable"],
        "mechanism_gate_passed": summary["mechanism_gate_passed"],
        "gate_passed": summary["gate_passed"],
    }
    compact_path = root / "run_summary.json"
    if compact_path.exists():
        if read_json(compact_path) != compact or compact_path.read_bytes() != canonical_json_bytes(compact):
            raise RuntimeError("IDEA-078 resumed compact summary mismatch")
    else:
        write_json(compact_path, compact)
    return summary


def main() -> None:
    args = parse_args()
    result = preflight(args) if args.stage == "preflight" else run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
