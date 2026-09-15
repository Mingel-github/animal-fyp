"""Run IDEA-039 animal-grouped AST-fbank augmentation experiments."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import statistics
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for local_root in (SCRIPTS_ROOT, SRC_ROOT):
    if str(local_root) not in sys.path:
        sys.path.insert(0, str(local_root))

import run_ast_finetuning as ast_base  # noqa: E402
import run_meowagenet_idea058_top_block_adaptation as base  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea039_grouped_augmentation_v1.json"
)
PLAN_PATH = REPO_ROOT / "plan" / "IDEA-039_grouped_augmentation_policy.md"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
DEFAULT_OUTPUT_SUBDIR = "meowagenet_idea039_grouped_augmentation_v1"
PIPELINES = (
    "R0_cached_frozen_ast",
    "C0_online_identity",
    "A1_fixed_specaugment",
    "A2_nested_policy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "select", "evaluate"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def dependency_paths(protocol: dict[str, Any]) -> dict[str, Path]:
    return {
        key.removesuffix("_path"): REPO_ROOT / value
        for key, value in protocol["dependencies"].items()
        if key.endswith("_path")
    }


def selection_path(run_root: Path) -> Path:
    return run_root / "selection" / "per_fold_policy_locks.json"


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol["protocol_id"] != "meowagenet-idea039-grouped-augmentation-v1":
        raise RuntimeError("Unexpected IDEA-039 protocol ID")
    if base.sha256(PLAN_PATH) != protocol["idea"]["sha256"]:
        raise RuntimeError("IDEA-039 plan checksum mismatch")
    if base.sha256(ROLES_PATH) != protocol["splits"]["roles_sha256"]:
        raise RuntimeError("IDEA-039 split checksum mismatch")
    manifest = REPO_ROOT / protocol["dataset"]["manifest_path"]
    if base.sha256(manifest) != protocol["dataset"]["manifest_sha256"]:
        raise RuntimeError("IDEA-039 dataset manifest checksum mismatch")
    paths = dependency_paths(protocol)
    for key, path in paths.items():
        expected = protocol["dependencies"][f"{key}_sha256"]
        if not path.is_file() or base.sha256(path) != expected:
            raise RuntimeError(f"IDEA-039 dependency checksum mismatch: {path}")
    policies = protocol["augmentation"]["candidate_policies"]
    if list(policies) != [
        "P0_identity",
        "P1_specaugment_light",
        "P2_gain_noise_light",
        "P3_shift_combo_light",
    ]:
        raise RuntimeError("IDEA-039 candidate pool differs from the lock")
    settings = protocol["inner_selection"]
    expected = (
        len(policies)
        * len(protocol["augmentation"]["selection_streams"])
        * len(settings["repeats"])
        * len(settings["outer_folds"])
    )
    if expected != int(settings["logical_candidate_fits"]):
        raise RuntimeError("IDEA-039 selection fit budget is inconsistent")
    evaluation = protocol["initial_evaluation"]
    expected_outer = len(PIPELINES) * len(evaluation["repeats"]) * len(
        evaluation["outer_folds"]
    )
    if expected_outer != int(evaluation["outer_fits"]):
        raise RuntimeError("IDEA-039 outer fit budget is inconsistent")


def _stable_view_seed(
    stream_seed: int, epoch: int, call_index: int, segment_ordinal: int
) -> int:
    payload = f"{stream_seed}|{epoch}|{call_index}|{segment_ordinal}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def valid_frame_count(segment: np.ndarray) -> int:
    valid = np.flatnonzero(np.std(segment, axis=1) > 1.0e-7)
    if len(valid) == 0:
        raise RuntimeError("AST segment has no detectable valid fbank frame")
    if not np.array_equal(valid, np.arange(len(valid))):
        raise RuntimeError("AST valid fbank frames are not a contiguous prefix")
    return int(len(valid))


def augment_segment(
    segment: np.ndarray,
    policy: dict[str, Any],
    stream_seed: int,
    epoch: int,
    call_index: int,
    segment_ordinal: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    original = np.asarray(segment, dtype=np.float32)
    output = original.copy()
    valid_frames = valid_frame_count(original)
    rng = np.random.default_rng(
        _stable_view_seed(stream_seed, epoch, call_index, segment_ordinal)
    )
    applied: list[str] = []
    for operation in policy["operations"]:
        if rng.random() >= float(operation["probability"]):
            continue
        kind = operation["type"]
        if kind == "time_mask":
            limit = min(
                int(operation["maximum_frames"]),
                max(1, int(np.floor(valid_frames * operation["maximum_valid_fraction"]))),
            )
            width = int(rng.integers(1, limit + 1))
            start = int(rng.integers(0, valid_frames - width + 1))
            output[start : start + width, :] = float(operation["fill"])
        elif kind == "frequency_mask":
            width = int(rng.integers(1, int(operation["maximum_bins"]) + 1))
            start = int(rng.integers(0, output.shape[1] - width + 1))
            output[:valid_frames, start : start + width] = float(operation["fill"])
        elif kind == "gain_offset":
            offset = float(rng.uniform(operation["minimum"], operation["maximum"]))
            output[:valid_frames, :] += offset
        elif kind == "gaussian_noise":
            noise = rng.normal(
                0.0,
                float(operation["standard_deviation"]),
                size=(valid_frames, output.shape[1]),
            ).astype(np.float32)
            output[:valid_frames, :] += noise
        elif kind == "valid_time_roll":
            maximum = min(int(operation["maximum_frames"]), valid_frames - 1)
            if maximum > 0:
                shift = int(rng.integers(-maximum, maximum + 1))
                if shift == 0:
                    shift = maximum
                output[:valid_frames, :] = np.roll(
                    output[:valid_frames, :], shift=shift, axis=0
                )
        else:
            raise RuntimeError(f"Unsupported IDEA-039 augmentation operation: {kind}")
        applied.append(kind)
    if not np.array_equal(output[valid_frames:], original[valid_frames:]):
        raise RuntimeError("IDEA-039 augmentation changed padded fbank rows")
    delta = np.abs(output - original)
    return output, {
        "valid_frames": valid_frames,
        "applied_operations": applied,
        "changed_values": int(np.count_nonzero(delta)),
        "maximum_absolute_delta": float(delta.max(initial=0.0)),
    }


def augment_batch(
    cpu_batch: dict[str, torch.Tensor],
    policy: dict[str, Any],
    stream_seed: int,
    epoch: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    instances = cpu_batch["instances"].numpy().copy()
    instance_to_call = cpu_batch["instance_to_call"].numpy()
    call_indices = cpu_batch["call_indices"].numpy()
    ordinals: Counter[int] = Counter()
    changed_values = 0
    maximum_delta = 0.0
    applied = Counter()
    for segment_row, local_call in enumerate(instance_to_call.tolist()):
        call_index = int(call_indices[local_call])
        segment_ordinal = int(ordinals[local_call])
        ordinals[local_call] += 1
        transformed, audit = augment_segment(
            instances[segment_row],
            policy,
            stream_seed,
            epoch,
            call_index,
            segment_ordinal,
        )
        instances[segment_row] = transformed
        changed_values += int(audit["changed_values"])
        maximum_delta = max(maximum_delta, float(audit["maximum_absolute_delta"]))
        applied.update(audit["applied_operations"])
    batch = dict(cpu_batch)
    batch["instances"] = torch.from_numpy(instances)
    return batch, {
        "segments": int(len(instances)),
        "changed_values": int(changed_values),
        "maximum_absolute_delta": float(maximum_delta),
        "applied_operations": dict(sorted(applied.items())),
    }


class OnlineFrozenASTClassifier(nn.Module):
    def __init__(self, locked_protocol: dict[str, Any], head: nn.Module) -> None:
        super().__init__()
        ast_config = locked_protocol["ast"]
        with torch.random.fork_rng(devices=[]):
            self.ast = ast_base.ASTModel.from_pretrained(
                ast_config["checkpoint"],
                revision=ast_config["revision"],
                cache_dir=ast_base.HF_CACHE,
                use_safetensors=True,
            )
            self.geometry_audit = ast_base.adapt_standard_geometry(
                self.ast, locked_protocol
            )
        for parameter in self.ast.parameters():
            parameter.requires_grad = False
        self.ast.eval()
        self.head = head

    def train(self, mode: bool = True) -> "OnlineFrozenASTClassifier":
        super().train(mode)
        self.ast.eval()
        self.head.train(mode)
        return self

    def encode_calls(
        self, instances: torch.Tensor, instance_to_call: torch.Tensor, call_count: int
    ) -> torch.Tensor:
        with torch.autocast(device_type=instances.device.type, enabled=False):
            with torch.no_grad():
                segment_embeddings = self.ast(
                    input_values=instances.float()
                ).pooler_output
        call_embeddings = torch.zeros(
            (call_count, segment_embeddings.shape[-1]),
            dtype=segment_embeddings.dtype,
            device=segment_embeddings.device,
        )
        call_embeddings.index_add_(0, instance_to_call, segment_embeddings)
        counts = torch.bincount(instance_to_call, minlength=call_count).to(
            segment_embeddings.dtype
        )
        return call_embeddings / counts[:, None]

    def forward(
        self, instances: torch.Tensor, instance_to_call: torch.Tensor, call_count: int
    ) -> torch.Tensor:
        return self.head(self.encode_calls(instances, instance_to_call, call_count))


def build_online_model(
    protocol: dict[str, Any], store: Any, train_indices: np.ndarray
) -> OnlineFrozenASTClassifier:
    embeddings = store.frozen_embeddings[train_indices]
    head = ast_base.ClassificationHead(
        mean=embeddings.mean(axis=0),
        scale=embeddings.std(axis=0),
        dropout=float(protocol["fixed_training"]["dropout"]),
    )
    locked = base.read_json(dependency_paths(protocol)["locked_protocol"])
    return OnlineFrozenASTClassifier(locked, head)


def parameter_audit(model: nn.Module) -> dict[str, Any]:
    trainable = {
        name: int(parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return {
        "trainable": int(sum(trainable.values())),
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_encoder": int(
            sum(count for name, count in trainable.items() if name.startswith("ast."))
        ),
        "trainable_head": int(
            sum(count for name, count in trainable.items() if name.startswith("head."))
        ),
        "trainable_parameter_names": sorted(trainable),
    }


def training_values(protocol: dict[str, Any]) -> dict[str, Any]:
    fixed = protocol["fixed_training"]
    return {
        "head_learning_rate": float(fixed["head_learning_rate"]),
        "head_weight_decay": float(fixed["head_weight_decay"]),
        "optimizer_epsilon": float(fixed["optimizer_epsilon"]),
        "batch_size": int(fixed["micro_batch_size"]),
        "evaluation_batch_size": int(fixed["evaluation_batch_size"]),
        "accumulation_steps": int(fixed["gradient_accumulation_steps"]),
        "accumulation_window_calls": int(fixed["accumulation_window_calls"]),
        "gradient_clip": float(fixed["gradient_clip"]),
        "max_epochs": int(fixed["maximum_epochs"]),
        "patience": int(fixed["early_stopping_patience"]),
    }


def build_online_loader(
    store: Any,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = ast_base.CallDataset(store, indices, "online")
    if shuffle:
        sampler = base.weighting.DeterministicNoSingletonBatchSampler(
            len(indices), batch_size, seed
        )
        return DataLoader(
            dataset, batch_sampler=sampler, num_workers=0, collate_fn=ast_base.collate_calls
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ast_base.collate_calls,
    )


def encode_augmented_calls(
    model: OnlineFrozenASTClassifier,
    loader: DataLoader,
    store: Any,
    device: torch.device,
    policy: dict[str, Any],
    stream_seed: int,
    epoch: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode one deterministic augmented view per call with a frozen AST."""

    model.ast.eval()
    embeddings = store.frozen_embeddings.copy()
    changed_values = 0
    maximum_delta = 0.0
    applied = Counter()
    encoded_calls = 0
    with torch.inference_mode():
        for original_batch in loader:
            augmented_batch, augmentation_audit = augment_batch(
                original_batch, policy, stream_seed, epoch
            )
            changed_values += int(augmentation_audit["changed_values"])
            maximum_delta = max(
                maximum_delta,
                float(augmentation_audit["maximum_absolute_delta"]),
            )
            applied.update(augmentation_audit["applied_operations"])
            batch = ast_base.move_batch(augmented_batch, device)
            call_embeddings = model.encode_calls(
                batch["instances"],
                batch["instance_to_call"],
                len(batch["labels"]),
            ).cpu().numpy()
            call_indices = batch["call_indices"].cpu().numpy().astype(np.int64)
            embeddings[call_indices] = call_embeddings
            encoded_calls += len(call_indices)
    return embeddings, {
        "encoded_calls": int(encoded_calls),
        "stream_seed": int(stream_seed),
        "changed_values": int(changed_values),
        "maximum_absolute_delta": float(maximum_delta),
        "applied_operations": dict(sorted(applied.items())),
        "encoder_batch_size": int(loader.batch_size or 0),
    }


def frozen_embedding_loader(
    store: Any,
    embeddings: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tuple[Any, DataLoader]:
    epoch_store = replace(store, frozen_embeddings=embeddings)
    loader = base.build_loader(
        epoch_store,
        indices,
        base.PIPELINES[0],
        batch_size,
        shuffle,
        seed,
    )
    return epoch_store, loader


def train_one_epoch_online(
    model: OnlineFrozenASTClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    weight_lookup: torch.Tensor,
    weight_lookup_numpy: np.ndarray,
    store: Any,
    device: torch.device,
    values: dict[str, Any],
    scaler: torch.cuda.amp.GradScaler,
    policy: dict[str, Any],
    stream_seed: int,
    epoch: int,
) -> tuple[float, dict[str, Any]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    weighted_total = 0.0
    weight_total = 0.0
    processed_indices: list[int] = []
    batch_sizes: list[int] = []
    changed_values = 0
    maximum_delta = 0.0
    applied = Counter()
    for step, original_batch in enumerate(loader):
        indices = original_batch["call_indices"].numpy().astype(np.int64).tolist()
        processed_indices.extend(indices)
        batch_sizes.append(len(indices))
        augmented_batch, augmentation_audit = augment_batch(
            original_batch, policy, stream_seed, epoch
        )
        changed_values += int(augmentation_audit["changed_values"])
        maximum_delta = max(
            maximum_delta, float(augmentation_audit["maximum_absolute_delta"])
        )
        applied.update(augmentation_audit["applied_operations"])
        batch = ast_base.move_batch(augmented_batch, device)
        weights = weight_lookup[batch["call_indices"]]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(
                batch["instances"], batch["instance_to_call"], len(batch["labels"])
            )
            per_call_loss = torch.nn.functional.cross_entropy(
                logits, batch["labels"], reduction="none"
            )
            loss = base.weighting.global_weighted_micro_loss(
                per_call_loss, weights, values["accumulation_window_calls"]
            )
        scaler.scale(loss).backward()
        should_step = (
            (step + 1) % values["accumulation_steps"] == 0
            or step + 1 == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), values["gradient_clip"]
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        weighted_total += float((per_call_loss.detach() * weights).sum())
        weight_total += float(weights.sum())
    audit = base.weighting.effective_coefficient_audit(
        processed_indices,
        batch_sizes,
        weight_lookup_numpy,
        store,
        values["accumulation_window_calls"],
        values["accumulation_steps"],
    )
    audit["augmentation"] = {
        "stream_seed": int(stream_seed),
        "changed_values": int(changed_values),
        "maximum_absolute_delta": float(maximum_delta),
        "applied_operations": dict(sorted(applied.items())),
    }
    return weighted_total / weight_total, audit


def fit_inner_online(
    policy_id: str,
    protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    training_seed: int,
    stream_seed: int,
    max_epochs_override: int | None = None,
) -> tuple[int, dict[str, Any], dict[str, torch.Tensor], pd.DataFrame]:
    base.split_utils.set_seed(training_seed)
    values = training_values(protocol)
    maximum_epochs = int(max_epochs_override or values["max_epochs"])
    policy = protocol["augmentation"]["candidate_policies"][policy_id]
    model = build_online_model(protocol, store, train_indices).to(device)
    head_model = ast_base.FrozenClassifier(model.head)
    parameters = parameter_audit(model)
    optimizer = torch.optim.Adamax(
        model.head.parameters(),
        lr=values["head_learning_rate"],
        eps=values["optimizer_epsilon"],
        weight_decay=values["head_weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    weights_numpy = base.weighting.global_class_balanced_call_weights(
        store.labels, train_indices
    )
    weights = torch.from_numpy(weights_numpy).to(device)
    encoding_loader = build_online_loader(
        store,
        train_indices,
        values["evaluation_batch_size"],
        False,
        training_seed,
    )
    validation_encoding_loader = build_online_loader(
        store,
        validation_indices,
        values["evaluation_batch_size"],
        False,
        training_seed,
    )
    identity_policy = protocol["augmentation"]["candidate_policies"]["P0_identity"]
    validation_embeddings, validation_encoding_audit = encode_augmented_calls(
        model,
        validation_encoding_loader,
        store,
        device,
        identity_policy,
        0,
        0,
    )
    validation_store, validation_loader = frozen_embedding_loader(
        store,
        validation_embeddings,
        validation_indices,
        values["evaluation_batch_size"],
        False,
        training_seed,
    )
    identity_training_cache: tuple[np.ndarray, dict[str, Any]] | None = None
    best_loss = float("inf")
    best_epoch = 1
    best_metrics: dict[str, Any] = {}
    best_state = base.portable_state_dict(model)
    history = []
    epochs_without_improvement = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, maximum_epochs + 1):
        if policy_id == "P0_identity" and identity_training_cache is not None:
            training_embeddings, augmentation_audit = identity_training_cache
        else:
            training_embeddings, augmentation_audit = encode_augmented_calls(
                model,
                encoding_loader,
                store,
                device,
                policy,
                stream_seed,
                epoch,
            )
            if policy_id == "P0_identity":
                identity_training_cache = (training_embeddings, augmentation_audit)
        epoch_store, train_loader = frozen_embedding_loader(
            store,
            training_embeddings,
            train_indices,
            values["batch_size"],
            True,
            training_seed,
        )
        train_loss, train_audit = base.train_one_epoch(
            head_model,
            train_loader,
            optimizer,
            weights,
            weights_numpy,
            epoch_store,
            device,
            values,
            scaler,
        )
        train_audit["augmentation"] = augmentation_audit
        call_loss, validation_calls = ast_base.predict_calls(
            head_model, validation_loader, validation_store, device
        )
        validation_animals = base.evaluation.calls_to_animals(validation_calls)
        animal_loss = base.evaluation.animal_cross_entropy(validation_animals)
        metrics = base.evaluation.animal_metrics(validation_animals)
        history.append(
            {
                "epoch": int(epoch),
                "train_weighted_loss": float(train_loss),
                "validation_call_cross_entropy": float(call_loss),
                "validation_animal_cross_entropy": float(animal_loss),
                "validation_animal_macro_f1": metrics["macro_f1"],
                "validation_animal_balanced_accuracy": metrics["balanced_accuracy"],
                "validation_animal_qwk": metrics["quadratic_weighted_kappa"],
                "validation_plain_accuracy": metrics["plain_accuracy"],
                "train_unit_audit": train_audit,
            }
        )
        if animal_loss < best_loss - 1.0e-6:
            best_loss = animal_loss
            best_epoch = epoch
            best_metrics = metrics
            best_state = base.portable_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"IDEA039 {policy_id} inner epoch={epoch} train={train_loss:.4f} "
            f"animal_CE={animal_loss:.4f} animal_F1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if max_epochs_override is None and epochs_without_improvement >= values["patience"]:
            break
    base.load_portable_state_dict(model, best_state)
    _, best_calls = ast_base.predict_calls(
        head_model, validation_loader, validation_store, device
    )
    audit = {
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(len(history)),
        "best_validation_animal_cross_entropy": float(best_loss),
        "best_validation_animal_metrics": best_metrics,
        "history": history,
        "target_weight_audit": base.weighting.target_weight_audit(
            weights_numpy, store, train_indices
        ),
        "train_seconds": float(time.perf_counter() - started),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "parameters": parameters,
        "policy_id": policy_id,
        "stream_seed": int(stream_seed),
        "validation_identity_encoding": validation_encoding_audit,
        "frozen_encoder_epoch_cache": True,
        "checkpoint_selection": "minimum_unweighted_animal_cross_entropy",
    }
    del model, head_model, optimizer, weights
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, audit, best_state, best_calls


def fit_outer_online(
    policy_id: str,
    protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    epochs: int,
    device: torch.device,
    training_seed: int,
    stream_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base.split_utils.set_seed(training_seed)
    values = training_values(protocol)
    policy = protocol["augmentation"]["candidate_policies"][policy_id]
    model = build_online_model(protocol, store, train_indices).to(device)
    head_model = ast_base.FrozenClassifier(model.head)
    parameters = parameter_audit(model)
    optimizer = torch.optim.Adamax(
        model.head.parameters(),
        lr=values["head_learning_rate"],
        eps=values["optimizer_epsilon"],
        weight_decay=values["head_weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    weights_numpy = base.weighting.global_class_balanced_call_weights(
        store.labels, train_indices
    )
    weights = torch.from_numpy(weights_numpy).to(device)
    encoding_loader = build_online_loader(
        store,
        train_indices,
        values["evaluation_batch_size"],
        False,
        training_seed,
    )
    identity_training_cache: tuple[np.ndarray, dict[str, Any]] | None = None
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(epochs) + 1):
        if policy_id == "P0_identity" and identity_training_cache is not None:
            training_embeddings, augmentation_audit = identity_training_cache
        else:
            training_embeddings, augmentation_audit = encode_augmented_calls(
                model,
                encoding_loader,
                store,
                device,
                policy,
                stream_seed,
                epoch,
            )
            if policy_id == "P0_identity":
                identity_training_cache = (training_embeddings, augmentation_audit)
        epoch_store, train_loader = frozen_embedding_loader(
            store,
            training_embeddings,
            train_indices,
            values["batch_size"],
            True,
            training_seed,
        )
        train_loss, train_audit = base.train_one_epoch(
            head_model,
            train_loader,
            optimizer,
            weights,
            weights_numpy,
            epoch_store,
            device,
            values,
            scaler,
        )
        train_audit["augmentation"] = augmentation_audit
        history.append(
            {
                "epoch": int(epoch),
                "train_weighted_loss": float(train_loss),
                "train_unit_audit": train_audit,
            }
        )
        print(
            f"IDEA039 {policy_id} outer epoch={epoch}/{epochs} train={train_loss:.4f}",
            flush=True,
        )
    test_encoding_loader = build_online_loader(
        store, test_indices, values["evaluation_batch_size"], False, training_seed
    )
    identity_policy = protocol["augmentation"]["candidate_policies"]["P0_identity"]
    test_embeddings, test_encoding_audit = encode_augmented_calls(
        model,
        test_encoding_loader,
        store,
        device,
        identity_policy,
        0,
        0,
    )
    test_store, test_loader = frozen_embedding_loader(
        store,
        test_embeddings,
        test_indices,
        values["evaluation_batch_size"],
        False,
        training_seed,
    )
    call_loss, calls = ast_base.predict_calls(
        head_model, test_loader, test_store, device
    )
    animals = base.evaluation.calls_to_animals(calls)
    audit = {
        "epochs": int(epochs),
        "history": history,
        "test_call_cross_entropy": float(call_loss),
        "test_animal_cross_entropy": base.evaluation.animal_cross_entropy(animals),
        "test_animal_metrics": base.evaluation.animal_metrics(animals),
        "target_weight_audit": base.weighting.target_weight_audit(
            weights_numpy, store, train_indices
        ),
        "train_and_predict_seconds": float(time.perf_counter() - started),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "parameters": parameters,
        "policy_id": policy_id,
        "stream_seed": int(stream_seed),
        "test_identity_encoding": test_encoding_audit,
        "frozen_encoder_epoch_cache": True,
        "checkpoint_selection": "median_inner_stream_best_epoch",
    }
    del model, head_model, optimizer, weights
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return animals, calls, audit


def r0_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    compatible = copy.deepcopy(protocol)
    compatible["inner_selection"] = {
        "candidate_recipes": {
            "r0_compat": {"block_count": 1, "encoder_learning_rate": 0.0}
        },
        "fixed_candidate_constraints": {
            "update_final_layernorm": False,
            "encoder_weight_decay": 0.0,
            "head_weight_decay": float(protocol["fixed_training"]["head_weight_decay"]),
            "gradient_clip": float(protocol["fixed_training"]["gradient_clip"]),
        },
    }
    compatible["_active_recipe_id"] = "r0_compat"
    return compatible


def outer_stream_seed(protocol: dict[str, Any], repeat: int, outer_fold: int) -> int:
    return int(protocol["augmentation"]["outer_stream_seed_base"]) + repeat * 10000 + outer_fold * 100


def stream_seed(base_stream: int, repeat: int, outer_fold: int) -> int:
    return int(base_stream) + repeat * 10000 + outer_fold * 100


def environment_lock(protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "protocol_sha256": base.sha256(PROTOCOL_PATH),
        "runner_sha256": base.sha256(Path(__file__).resolve()),
        "plan_sha256": base.sha256(PLAN_PATH),
        "dependencies": {
            key: base.sha256(path) for key, path in dependency_paths(protocol).items()
        },
    }


def online_cache_difference(
    protocol: dict[str, Any],
    store: Any,
    indices: np.ndarray,
    device: torch.device,
) -> float:
    base.split_utils.set_seed(17)
    model = build_online_model(protocol, store, indices).to(device)
    model.eval()
    loader = build_online_loader(store, indices, 32, False, 17)
    maximum = 0.0
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = ast_base.move_batch(cpu_batch, device)
            embeddings = model.encode_calls(
                batch["instances"], batch["instance_to_call"], len(batch["labels"])
            ).cpu().numpy()
            expected = store.frozen_embeddings[
                batch["call_indices"].cpu().numpy().astype(np.int64)
            ]
            maximum = max(maximum, float(np.max(np.abs(embeddings - expected))))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return maximum


def run_smoke(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    device: torch.device,
) -> None:
    smoke = protocol["smoke"]
    indices = base.split_utils.fold_indices(
        store,
        roles,
        int(smoke["repeat"]),
        int(smoke["outer_fold"]),
        include_test=False,
    )
    sample = store.features[store.call_segment_indices[int(indices["train"][0])][0]]
    policies = protocol["augmentation"]["candidate_policies"]
    transform_audit = {}
    for policy_id, policy in policies.items():
        first, first_audit = augment_segment(sample, policy, 39001, 1, 0, 0)
        second, _ = augment_segment(sample, policy, 39001, 1, 0, 0)
        other, _ = augment_segment(sample, policy, 39002, 1, 0, 0)
        transform_audit[policy_id] = {
            "same_stream_bitwise_equal": bool(np.array_equal(first, second)),
            "different_stream_bitwise_equal": bool(np.array_equal(first, other)),
            "changed_values": first_audit["changed_values"],
            "maximum_absolute_delta": first_audit["maximum_absolute_delta"],
        }
    if not np.array_equal(
        augment_segment(sample, policies["P0_identity"], 39001, 1, 0, 0)[0], sample
    ):
        raise RuntimeError("IDEA-039 identity policy changed fbank input")
    maximum_embedding_difference = online_cache_difference(
        protocol,
        store,
        np.concatenate((indices["train"], indices["validation"])),
        device,
    )
    if maximum_embedding_difference > 3.0e-5:
        raise RuntimeError(
            f"Online identity embedding differs from cache: {maximum_embedding_difference}"
        )
    seed = base.split_utils.full_seed(
        int(smoke["base_seed"]), int(smoke["repeat"]), int(smoke["outer_fold"])
    )
    fits = []
    for policy_id in ("P0_identity", protocol["augmentation"]["fixed_policy_id"]):
        _, audit, state, _ = fit_inner_online(
            policy_id,
            protocol,
            store,
            indices["train"],
            indices["validation"],
            device,
            seed,
            stream_seed(39001, int(smoke["repeat"]), int(smoke["outer_fold"])),
            max_epochs_override=int(smoke["epochs"]),
        )
        del state
        fits.append(audit)
    if fits[0]["history"][0]["train_unit_audit"]["batch_order_sha256"] != fits[1]["history"][0]["train_unit_audit"]["batch_order_sha256"]:
        raise RuntimeError("IDEA-039 smoke batch orders differ")
    smoke_root = run_root / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    env = environment_lock(protocol, device)
    base.write_json(run_root / "environment_lock.json", env)
    result = {
        "status": "complete",
        "stage": "idea039_inner_only_smoke",
        "outer_test_accessed": False,
        "repeat": int(smoke["repeat"]),
        "outer_fold": int(smoke["outer_fold"]),
        "transform_audit": transform_audit,
        "online_identity_cache_max_abs_difference": maximum_embedding_difference,
        "training_fits": fits,
        "environment_lock_sha256": base.sha256(run_root / "environment_lock.json"),
    }
    base.write_json(smoke_root / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def rank_policies(
    policy_rows: dict[str, list[dict[str, Any]]], policies: dict[str, Any]
) -> list[dict[str, Any]]:
    summaries = []
    for policy_id, rows in policy_rows.items():
        macro = [row["best_validation_animal_metrics"]["macro_f1"] for row in rows]
        ce = [row["best_validation_animal_cross_entropy"] for row in rows]
        summaries.append(
            {
                "policy_id": policy_id,
                "mean_animal_macro_f1": float(np.mean(macro)),
                "sample_sd_animal_macro_f1": float(np.std(macro, ddof=1)),
                "mean_animal_cross_entropy": float(np.mean(ce)),
                "best_epochs": [int(row["best_epoch"]) for row in rows],
                "median_best_epoch": int(statistics.median(row["best_epoch"] for row in rows)),
                "severity_rank": int(policies[policy_id]["severity_rank"]),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            -round(row["mean_animal_macro_f1"], 6),
            row["mean_animal_cross_entropy"],
            row["severity_rank"],
            row["policy_id"],
        ),
    )


def run_selection(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    output_path = selection_path(run_root)
    if output_path.is_file() and not resume:
        raise FileExistsError(output_path)
    settings = protocol["inner_selection"]
    policies = protocol["augmentation"]["candidate_policies"]
    streams = protocol["augmentation"]["selection_streams"]
    rows_by_fold: dict[tuple[int, int], dict[str, list[dict[str, Any]]]] = {
        (int(repeat), int(outer_fold)): {policy_id: [] for policy_id in policies}
        for repeat in settings["repeats"]
        for outer_fold in settings["outer_folds"]
    }
    all_rows = []
    for repeat in settings["repeats"]:
        for outer_fold in settings["outer_folds"]:
            indices = base.split_utils.fold_indices(
                store, roles, int(repeat), int(outer_fold), include_test=False
            )
            training_seed = base.split_utils.full_seed(
                int(settings["base_seed"]), int(repeat), int(outer_fold)
            )
            for stream_base in streams:
                augmentation_seed = stream_seed(
                    int(stream_base), int(repeat), int(outer_fold)
                )
                for policy_id in policies:
                    fit_path = (
                        run_root
                        / "selection"
                        / "fits"
                        / policy_id
                        / f"stream_{stream_base}"
                        / f"repeat_{repeat}"
                        / f"fold_{outer_fold}.json"
                    )
                    if fit_path.is_file():
                        if not resume:
                            raise FileExistsError(fit_path)
                        fit = base.read_json(fit_path)
                    else:
                        print(
                            f"IDEA039 SELECT policy={policy_id} stream={stream_base} "
                            f"repeat={repeat} fold={outer_fold}",
                            flush=True,
                        )
                        _, audit, state, _ = fit_inner_online(
                            policy_id,
                            protocol,
                            store,
                            indices["train"],
                            indices["validation"],
                            device,
                            training_seed,
                            augmentation_seed,
                        )
                        del state
                        fit = {
                            "status": "complete",
                            "stage": "idea039_inner_policy_selection",
                            "outer_test_accessed": False,
                            "repeat": int(repeat),
                            "outer_fold": int(outer_fold),
                            "policy_id": policy_id,
                            "stream_base": int(stream_base),
                            "augmentation_seed": int(augmentation_seed),
                            "training_seed": int(training_seed),
                            **audit,
                        }
                        base.write_json(fit_path, fit)
                    if fit["outer_test_accessed"] is not False:
                        raise RuntimeError("IDEA-039 selection fit accessed outer test")
                    rows_by_fold[(int(repeat), int(outer_fold))][policy_id].append(fit)
                    all_rows.append(fit)
    locks = []
    for (repeat, outer_fold), policy_rows in sorted(rows_by_fold.items()):
        ranking = rank_policies(policy_rows, policies)
        selected = ranking[0]
        per_stream_rankings = []
        for stream_index, stream_base in enumerate(streams):
            rows = [policy_rows[policy_id][stream_index] for policy_id in policies]
            stream_ranking = sorted(
                rows,
                key=lambda row: (
                    -round(row["best_validation_animal_metrics"]["macro_f1"], 6),
                    row["best_validation_animal_cross_entropy"],
                    policies[row["policy_id"]]["severity_rank"],
                    row["policy_id"],
                ),
            )
            per_stream_rankings.append(
                {
                    "stream_base": int(stream_base),
                    "ranking": [row["policy_id"] for row in stream_ranking],
                }
            )
        top_two_count = sum(
            selected["policy_id"] in row["ranking"][:2] for row in per_stream_rankings
        )
        locks.append(
            {
                "repeat": repeat,
                "outer_fold": outer_fold,
                "selected_policy_id": selected["policy_id"],
                "selected_epoch": selected["median_best_epoch"],
                "selected_mean_inner_animal_macro_f1": selected["mean_animal_macro_f1"],
                "selected_mean_inner_animal_cross_entropy": selected["mean_animal_cross_entropy"],
                "selected_top_two_stream_count": int(top_two_count),
                "stream_stable": bool(top_two_count >= 2),
                "aggregate_ranking": [row["policy_id"] for row in ranking],
                "candidate_summaries": {
                    row["policy_id"]: row for row in ranking
                },
                "per_stream_rankings": per_stream_rankings,
            }
        )
    counts = Counter(row["selected_policy_id"] for row in locks)
    record = {
        "status": "complete",
        "stage": "idea039_per_fold_policy_selection",
        "outer_test_accessed": False,
        "protocol_id": protocol["protocol_id"],
        "selection_boundary": "per_repeat_outer_fold",
        "logical_candidate_fits_completed": len(all_rows),
        "fold_locks": len(locks),
        "stream_stable_fold_locks": int(sum(row["stream_stable"] for row in locks)),
        "selection_rule": settings["selection_rule"],
        "selected_policy_counts": dict(sorted(counts.items())),
        "per_fold_policy_locks": locks,
    }
    base.write_json(output_path, record)
    print(json.dumps(record, indent=2), flush=True)


def fold_lock(selection: dict[str, Any], repeat: int, outer_fold: int) -> dict[str, Any]:
    matches = [
        row
        for row in selection["per_fold_policy_locks"]
        if int(row["repeat"]) == repeat and int(row["outer_fold"]) == outer_fold
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one IDEA-039 policy lock")
    return matches[0]


def issue_or_verify_execution_lock(
    run_root: Path, protocol: dict[str, Any], selection: dict[str, Any]
) -> dict[str, Any]:
    path = run_root / "execution_lock.json"
    if path.is_file():
        lock = base.read_json(path)
    else:
        revision = base.git_revision()
        if revision is None:
            raise RuntimeError("IDEA-039 evaluation requires a git revision")
        for locked_path in (PLAN_PATH, PROTOCOL_PATH, Path(__file__).resolve()):
            if base.git_blob_object_id(revision, locked_path) != base.worktree_blob_object_id(locked_path):
                raise RuntimeError(f"IDEA-039 locked file differs from commit: {locked_path}")
        lock = {
            "schema_version": "1.0",
            "status": "locked_for_idea039_evaluation",
            "code_commit": revision,
            "protocol_sha256": base.sha256(PROTOCOL_PATH),
            "runner_sha256": base.sha256(Path(__file__).resolve()),
            "plan_sha256": base.sha256(PLAN_PATH),
            "selection_sha256": base.sha256(selection_path(run_root)),
            "per_fold_policy_locks": selection["per_fold_policy_locks"],
        }
        base.write_json(path, lock)
    if lock["protocol_sha256"] != base.sha256(PROTOCOL_PATH):
        raise RuntimeError("IDEA-039 protocol differs from execution lock")
    if lock["runner_sha256"] != base.sha256(Path(__file__).resolve()):
        raise RuntimeError("IDEA-039 runner differs from execution lock")
    if lock["selection_sha256"] != base.sha256(selection_path(run_root)):
        raise RuntimeError("IDEA-039 selection differs from execution lock")
    return lock


def policy_for_pipeline(
    pipeline: str, protocol: dict[str, Any], lock: dict[str, Any]
) -> tuple[str, int]:
    if pipeline == PIPELINES[1]:
        policy_id = "P0_identity"
    elif pipeline == PIPELINES[2]:
        policy_id = protocol["augmentation"]["fixed_policy_id"]
    elif pipeline == PIPELINES[3]:
        policy_id = lock["selected_policy_id"]
    else:
        raise ValueError(pipeline)
    summary = lock["candidate_summaries"][policy_id]
    return policy_id, int(summary["median_best_epoch"])


def contrast_rows(
    metrics: dict[str, list[dict[str, Any]]],
    animals: dict[tuple[str, int], pd.DataFrame],
    left_pipeline: str,
    right_pipeline: str,
    repeats: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for repeat in repeats:
        left_metrics = metrics[left_pipeline][repeat]
        right_metrics = metrics[right_pipeline][repeat]
        merged = animals[(left_pipeline, repeat)].merge(
            animals[(right_pipeline, repeat)],
            on=["cat_id", "true_label", "call_count"],
            suffixes=("_left", "_right"),
        )
        rows.append(
            {
                "repeat": repeat,
                "left_macro_f1": left_metrics["macro_f1"],
                "right_macro_f1": right_metrics["macro_f1"],
                "macro_f1_difference": right_metrics["macro_f1"] - left_metrics["macro_f1"],
                "balanced_accuracy_difference": right_metrics["balanced_accuracy"] - left_metrics["balanced_accuracy"],
                "qwk_difference": right_metrics["quadratic_weighted_kappa"] - left_metrics["quadratic_weighted_kappa"],
                "plain_accuracy_difference": right_metrics["plain_accuracy"] - left_metrics["plain_accuracy"],
                "animal_cross_entropy_difference": right_metrics["animal_cross_entropy"] - left_metrics["animal_cross_entropy"],
                "changed_animals": int((merged["predicted_label_left"] != merged["predicted_label_right"]).sum()),
                "gained_correct_animals": int(((merged["predicted_label_right"] == merged["true_label"]) & (merged["predicted_label_left"] != merged["true_label"])).sum()),
                "lost_correct_animals": int(((merged["predicted_label_right"] != merged["true_label"]) & (merged["predicted_label_left"] == merged["true_label"])).sum()),
            }
        )
    return rows


def gate_for_pipeline(
    summary: dict[str, Any], protocol: dict[str, Any], pipeline: str
) -> dict[str, Any]:
    reference = PIPELINES[0]
    name = f"{pipeline}_minus_{reference}"
    paired = summary["paired_summary"][name]
    aggregate = summary["aggregate"]
    ce = summary["animal_cross_entropy"]
    rule = protocol["seed_expansion_gate"]
    class_recall_drops = {
        label: float(
            aggregate[reference]["mean_per_class"][label]["recall"]
            - aggregate[pipeline]["mean_per_class"][label]["recall"]
        )
        for label in base.LABEL_NAMES
    }
    checks = {
        "mean_macro_f1_gain": paired["mean_macro_f1_difference"] >= float(rule["minimum_mean_macro_f1_gain_over_R0"]),
        "positive_repeats": paired["positive_repeats"] >= int(rule["minimum_positive_repeats_over_R0"]),
        "balanced_accuracy_cost": paired["mean_balanced_accuracy_difference"] >= -float(rule["maximum_tolerated_mean_balanced_accuracy_drop"]),
        "qwk_cost": paired["mean_qwk_difference"] >= -float(rule["maximum_tolerated_mean_qwk_drop"]),
        "animal_ce_cost": ce[pipeline]["mean"] - ce[reference]["mean"] <= float(rule["maximum_tolerated_mean_animal_ce_increase"]),
        "class_recall_cost": max(class_recall_drops.values()) <= float(rule["maximum_tolerated_class_recall_drop"]),
    }
    if pipeline == PIPELINES[3]:
        stable = int(summary["policy_selection"]["stream_stable_fold_locks"])
        checks["selector_stream_stability"] = stable >= int(rule["minimum_stable_A2_fold_locks"])
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "mean_macro_f1_difference": paired["mean_macro_f1_difference"],
        "positive_repeats": paired["positive_repeats"],
        "mean_animal_cross_entropy_difference": ce[pipeline]["mean"] - ce[reference]["mean"],
        "class_recall_drop_from_R0": class_recall_drops,
    }


def aggregate_evaluation(
    evaluation_root: Path, protocol: dict[str, Any], selection: dict[str, Any]
) -> dict[str, Any]:
    settings = protocol["initial_evaluation"]
    base_seed = int(settings["base_seeds"][0])
    repeats = [int(value) for value in settings["repeats"]]
    metrics: dict[str, list[dict[str, Any]]] = {pipeline: [] for pipeline in PIPELINES}
    animals_by_key: dict[tuple[str, int], pd.DataFrame] = {}
    fits_by_pipeline: dict[str, list[dict[str, Any]]] = {pipeline: [] for pipeline in PIPELINES}
    for repeat in repeats:
        for pipeline in PIPELINES:
            call_frames = []
            for outer_fold in settings["outer_folds"]:
                fit_root = evaluation_root / "fits" / pipeline / f"base_seed_{base_seed}" / f"repeat_{repeat}" / f"fold_{outer_fold}"
                call_frames.append(pd.read_csv(fit_root / "outer_test_call_predictions.csv", dtype={"cat_id": str}))
                fits_by_pipeline[pipeline].append(base.read_json(fit_root / "fit_summary.json"))
            calls = pd.concat(call_frames, ignore_index=True).sort_values("call_index").reset_index(drop=True)
            if len(calls) != 792 or calls["call_index"].nunique() != 792:
                raise RuntimeError("IDEA-039 complete OOF must contain 792 calls")
            animals = base.evaluation.calls_to_animals(calls)
            if len(animals) != 111 or animals["cat_id"].nunique() != 111:
                raise RuntimeError("IDEA-039 complete OOF must contain 111 cats")
            row = {
                "base_seed": base_seed,
                "repeat": repeat,
                "animal_cross_entropy": base.evaluation.animal_cross_entropy(animals),
                **base.evaluation.animal_metrics(animals),
            }
            metrics[pipeline].append(row)
            animals_by_key[(pipeline, repeat)] = animals
            output_root = evaluation_root / "oof" / pipeline
            output_root.mkdir(parents=True, exist_ok=True)
            calls.to_csv(output_root / f"repeat_{repeat}_calls.csv", index=False)
            animals.to_csv(output_root / f"repeat_{repeat}_animals.csv", index=False)
    contrasts = (
        (PIPELINES[0], PIPELINES[1]),
        (PIPELINES[0], PIPELINES[2]),
        (PIPELINES[0], PIPELINES[3]),
        (PIPELINES[2], PIPELINES[3]),
    )
    paired = {}
    bootstrap = {}
    for left, right in contrasts:
        name = f"{right}_minus_{left}"
        paired[name] = contrast_rows(metrics, animals_by_key, left, right, repeats)
        bootstrap[name] = base.evaluation.paired_cat_bootstrap(
            animals_by_key, left, right, protocol
        )
    aggregate = {
        pipeline: base.evaluation.aggregate_metrics(rows)
        for pipeline, rows in metrics.items()
    }
    ce = {
        pipeline: {
            "by_repeat": [float(row["animal_cross_entropy"]) for row in rows],
            "mean": float(np.mean([row["animal_cross_entropy"] for row in rows])),
            "sample_sd": float(np.std([row["animal_cross_entropy"] for row in rows], ddof=1)),
        }
        for pipeline, rows in metrics.items()
    }
    training = {}
    for pipeline, fits in fits_by_pipeline.items():
        epochs = [int(fit["selected_epoch"]) for fit in fits]
        training[pipeline] = {
            "selected_epochs": epochs,
            "mean_selected_epoch": float(np.mean(epochs)),
            "selected_epoch_range": [int(min(epochs)), int(max(epochs))],
            "mean_inner_train_seconds": float(np.mean([fit["inner"].get("train_seconds", 0.0) for fit in fits])),
            "mean_outer_train_and_predict_seconds": float(np.mean([fit["outer"]["train_and_predict_seconds"] for fit in fits])),
            "maximum_peak_vram_bytes": int(max(max(fit["inner"].get("peak_vram_bytes", 0), fit["outer"]["peak_vram_bytes"]) for fit in fits)),
            "parameters": fits[0]["outer"]["parameters"],
        }
    summary = {
        "status": "complete",
        "stage": "idea039_grouped_augmentation_evaluation",
        "outer_test_accessed": True,
        "protocol_id": protocol["protocol_id"],
        "primary_unit": "animal",
        "primary_metric": "macro_f1",
        "metrics_by_repeat": metrics,
        "aggregate": aggregate,
        "animal_cross_entropy": ce,
        "paired": paired,
        "paired_summary": {name: base.paired_summary(rows) for name, rows in paired.items()},
        "paired_cat_bootstrap": bootstrap,
        "training_and_parameters": training,
        "complete_oof": {"calls_per_pipeline_repeat": 792, "animals_per_pipeline_repeat": 111, "evaluations": 12},
        "policy_selection": {
            "selected_policy_counts": selection["selected_policy_counts"],
            "stream_stable_fold_locks": selection["stream_stable_fold_locks"],
            "per_fold_policy_locks": selection["per_fold_policy_locks"],
        },
    }
    c0_name = f"{PIPELINES[1]}_minus_{PIPELINES[0]}"
    c0 = summary["paired_summary"][c0_name]
    mean_agreement = 1.0 - c0["mean_changed_animals"] / 111.0
    path_rule = protocol["path_equivalence_gate"]
    summary["path_equivalence_gate"] = {
        "passed": bool(abs(c0["mean_macro_f1_difference"]) <= float(path_rule["maximum_absolute_mean_macro_f1_difference_C0_vs_R0"]) and mean_agreement >= float(path_rule["minimum_mean_prediction_agreement_C0_vs_R0"])),
        "C0_minus_R0_mean_macro_f1": c0["mean_macro_f1_difference"],
        "mean_prediction_agreement": mean_agreement,
    }
    summary["H039_A_fixed_augmentation_gate"] = gate_for_pipeline(summary, protocol, PIPELINES[2])
    summary["H039_B_nested_selector_gate"] = gate_for_pipeline(summary, protocol, PIPELINES[3])
    return summary


def run_evaluation(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    selection = base.read_json(selection_path(run_root))
    if selection["status"] != "complete" or selection["outer_test_accessed"] is not False:
        raise RuntimeError("IDEA-039 selection lock is incomplete")
    execution_lock = issue_or_verify_execution_lock(run_root, protocol, selection)
    evaluation_root = run_root / "evaluation"
    summary_path = evaluation_root / "summary.json"
    if summary_path.is_file() and not resume:
        raise FileExistsError(summary_path)
    evaluation_root.mkdir(parents=True, exist_ok=True)
    settings = protocol["initial_evaluation"]
    completed = []
    for base_seed in settings["base_seeds"]:
        for repeat in settings["repeats"]:
            for outer_fold in settings["outer_folds"]:
                lock = fold_lock(selection, int(repeat), int(outer_fold))
                indices = base.split_utils.fold_indices(
                    store, roles, int(repeat), int(outer_fold), include_test=True
                )
                outer_train = np.concatenate((indices["train"], indices["validation"]))
                training_seed = base.split_utils.full_seed(
                    int(base_seed), int(repeat), int(outer_fold)
                )
                for pipeline in PIPELINES:
                    output_dir = evaluation_root / "fits" / pipeline / f"base_seed_{base_seed}" / f"repeat_{repeat}" / f"fold_{outer_fold}"
                    fit_path = output_dir / "fit_summary.json"
                    if fit_path.is_file():
                        if not resume:
                            raise FileExistsError(fit_path)
                        completed.append(base.read_json(fit_path))
                        continue
                    print(
                        f"IDEA039 EVAL pipeline={pipeline} repeat={repeat} fold={outer_fold}",
                        flush=True,
                    )
                    if pipeline == PIPELINES[0]:
                        compatible = r0_protocol(protocol)
                        best_epoch, inner, state, _ = base.fit_inner(
                            base.PIPELINES[0], compatible, store, indices["train"], indices["validation"], device, training_seed
                        )
                        del state
                        animals, calls, outer = base.fit_outer_and_predict(
                            base.PIPELINES[0], compatible, store, outer_train, indices["test"], best_epoch, device, training_seed
                        )
                        policy_id = "cached_identity"
                    else:
                        policy_id, best_epoch = policy_for_pipeline(pipeline, protocol, lock)
                        inner = {
                            "selection_source": "three_stream_current_fold_policy_selection",
                            "best_epoch": int(best_epoch),
                            "candidate_summary": lock["candidate_summaries"][policy_id],
                            "train_seconds": float(sum(
                                base.read_json(
                                    run_root / "selection" / "fits" / policy_id / f"stream_{stream_base}" / f"repeat_{repeat}" / f"fold_{outer_fold}.json"
                                )["train_seconds"]
                                for stream_base in protocol["augmentation"]["selection_streams"]
                            )),
                            "peak_vram_bytes": int(max(
                                base.read_json(
                                    run_root / "selection" / "fits" / policy_id / f"stream_{stream_base}" / f"repeat_{repeat}" / f"fold_{outer_fold}.json"
                                )["peak_vram_bytes"]
                                for stream_base in protocol["augmentation"]["selection_streams"]
                            )),
                        }
                        animals, calls, outer = fit_outer_online(
                            policy_id,
                            protocol,
                            store,
                            outer_train,
                            indices["test"],
                            best_epoch,
                            device,
                            training_seed,
                            outer_stream_seed(protocol, int(repeat), int(outer_fold)),
                        )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    animal_path = output_dir / "outer_test_animal_predictions.csv"
                    call_path = output_dir / "outer_test_call_predictions.csv"
                    animals.to_csv(animal_path, index=False)
                    calls.to_csv(call_path, index=False)
                    fit = {
                        "status": "complete",
                        "stage": "idea039_grouped_augmentation_evaluation",
                        "outer_test_accessed": True,
                        "pipeline": pipeline,
                        "policy_id": policy_id,
                        "base_seed": int(base_seed),
                        "repeat": int(repeat),
                        "outer_fold": int(outer_fold),
                        "training_seed": int(training_seed),
                        "selected_epoch": int(best_epoch),
                        "inner": inner,
                        "outer": outer,
                        "animal_prediction_path": base.repo_relative(animal_path),
                        "call_prediction_path": base.repo_relative(call_path),
                    }
                    base.write_json(fit_path, fit)
                    completed.append(fit)
    summary = aggregate_evaluation(evaluation_root, protocol, selection)
    inventory = base.evaluation.raw_prediction_inventory(evaluation_root)
    inventory_path = evaluation_root / "raw_prediction_inventory.json"
    base.write_json(inventory_path, inventory)
    summary.update(
        {
            "code_commit": execution_lock["code_commit"],
            "execution_lock_sha256": base.sha256(run_root / "execution_lock.json"),
            "environment_lock_sha256": base.sha256(run_root / "environment_lock.json"),
            "selection_sha256": base.sha256(selection_path(run_root)),
            "protocol_sha256": execution_lock["protocol_sha256"],
            "runner_sha256": execution_lock["runner_sha256"],
            "raw_prediction_inventory": {
                "path": base.repo_relative(inventory_path),
                "inventory_sha256": base.sha256(inventory_path),
                "files": inventory["files"],
                "bytes": inventory["bytes"],
                "aggregate_sha256": inventory["aggregate_sha256"],
            },
        }
    )
    base.write_json(summary_path, summary)
    base.write_json(REPO_ROOT / protocol["outputs"]["result_metadata"], summary)
    base.write_json(
        evaluation_root / "run_summary.json",
        {
            "status": "complete",
            "completed_fits": len(completed),
            "expected_fits": int(settings["outer_fits"]),
            "summary_sha256": base.sha256(summary_path),
            "raw_prediction_aggregate_sha256": inventory["aggregate_sha256"],
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    protocol = base.read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    device = base.resolve_device(args.device)
    run_root = REPO_ROOT / "runs" / args.output_subdir
    run_root.mkdir(parents=True, exist_ok=True)
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    store = ast_base.load_feature_store()
    if len(store.call_ids) != 792 or len(np.unique(store.cat_ids)) != 111:
        raise RuntimeError("IDEA-039 feature store inventory mismatch")
    if args.stage == "smoke":
        run_smoke(run_root, protocol, roles, store, device)
    elif args.stage == "select":
        run_selection(run_root, protocol, roles, store, device, args.resume)
    else:
        run_evaluation(run_root, protocol, roles, store, device, args.resume)


if __name__ == "__main__":
    main()
