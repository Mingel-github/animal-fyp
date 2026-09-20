"""Run IDEA-085 minimal local acoustic temporal residual experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_idea068_age_sensitive_ast as idea068  # noqa: E402
import run_meowagenet_idea071_bounded_dual_path_fusion as idea071  # noqa: E402
import run_meowagenet_idea082_age_acoustic_group_ablation as idea082  # noqa: E402
import run_meowagenet_idea084_grouped_dual_branch_acoustic_residual as idea084  # noqa: E402


# Preserve the upstream IDEA-084 identities before run() installs its execution
# hooks.  verify_protocol() is also called from the patched IDEA-084 skeleton,
# so reading the mutable module globals there would accidentally make IDEA-085
# validate itself as its own dependency.
IDEA084_PROTOCOL_PATH = Path(idea084.PROTOCOL_PATH).resolve()
IDEA084_RUNNER_PATH = Path(idea084.__file__).resolve()


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea085_acoustic_temporal_residual_v1.json"
)
PIPELINES = (
    "A0_ast_only",
    "C1_bounded_wide_additive",
    "T1_acoustic_temporal_residual",
    "J1_frame_shuffled_temporal_residual",
)
FEATURE_NAMES = (
    "log_f0",
    "voiced_probability",
    "periodicity",
    "log_rms",
    "spectral_tilt",
    "spectral_flatness",
)
BASE_SEEDS = (2713, 5395, 5226)
PERMUTATION_MATERIAL = "IDEA-085-joint-frame-permutation-v1"
EXPECTED_PARAMETERS = {
    PIPELINES[0]: 99_075,
    PIPELINES[1]: 108_143,
    PIPELINES[2]: 108_355,
    PIPELINES[3]: 108_355,
}
COMPARISONS = {
    "T1_minus_C1": (PIPELINES[2], PIPELINES[1]),
    "T1_minus_A0": (PIPELINES[2], PIPELINES[0]),
    "T1_minus_J1": (PIPELINES[2], PIPELINES[3]),
}
DESCRIPTIVE_COMPARISONS = {
    "C1_minus_A0_descriptive": (PIPELINES[1], PIPELINES[0])
}
CLASS_NAMES = ("kitten", "adult", "senior")
_IDEA068_BUILD_MODEL = idea068.build_model
_ACTIVE_TRAJECTORIES: "TemporalRaggedCache | None" = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "run"), required=True)
    parser.add_argument(
        "--output-subdir",
        default="meowagenet_idea085_acoustic_temporal_residual_v1",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--director-authorized", action="store_true")
    parser.add_argument("--max-cells", type=int, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return idea068.sha256(path)


def permutation_seed(call_id: str) -> int:
    digest = hashlib.sha256(
        f"{PERMUTATION_MATERIAL}|call_id={call_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def fixed_joint_permutation(call_id: str, frames: int) -> np.ndarray:
    if frames < 1:
        raise ValueError("IDEA-085 trajectories must contain at least one frame")
    return np.random.default_rng(permutation_seed(call_id)).permutation(frames)


@dataclass(frozen=True)
class TemporalRaggedCache:
    call_ids: tuple[str, ...]
    offsets: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.offsets.dtype != np.int64:
            raise TypeError("IDEA-085 offsets must be int64")
        if self.values.dtype != np.float32:
            raise TypeError("IDEA-085 values must be float32")
        if self.values.ndim != 2 or self.values.shape[1] != len(FEATURE_NAMES):
            raise ValueError("IDEA-085 values must have shape [total_frames, 6]")
        if self.offsets.shape != (len(self.call_ids) + 1,):
            raise ValueError("IDEA-085 offsets length differs from call IDs")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != len(self.values):
            raise ValueError("IDEA-085 offsets do not span values")
        if np.any(np.diff(self.offsets) <= 0):
            raise ValueError("IDEA-085 every call must have at least one frame")
        if len(set(self.call_ids)) != len(self.call_ids):
            raise ValueError("IDEA-085 call IDs must be unique")

    @classmethod
    def from_sequences(
        cls, call_ids: Iterable[str], sequences: Iterable[np.ndarray]
    ) -> "TemporalRaggedCache":
        ids = tuple(str(value) for value in call_ids)
        arrays = [np.asarray(value, dtype=np.float32) for value in sequences]
        offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
        for index, array in enumerate(arrays):
            if array.ndim != 2 or array.shape[1] != len(FEATURE_NAMES):
                raise ValueError("Synthetic IDEA-085 sequence has wrong shape")
            offsets[index + 1] = offsets[index] + len(array)
        values = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
        return cls(ids, offsets, values)

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def sequence(self, call_index: int, shuffled: bool = False) -> np.ndarray:
        start = int(self.offsets[call_index])
        stop = int(self.offsets[call_index + 1])
        sequence = self.values[start:stop]
        if shuffled:
            sequence = sequence[
                fixed_joint_permutation(self.call_ids[call_index], len(sequence))
            ]
        return np.asarray(sequence, dtype=np.float32)

    def training_statistics(
        self, train_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frames = np.concatenate(
            [self.sequence(int(index), shuffled=False) for index in train_indices],
            axis=0,
        ).astype(np.float64)
        finite = np.isfinite(frames)
        with np.errstate(all="ignore"):
            median = np.nanmedian(np.where(finite, frames, np.nan), axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        imputed = np.where(finite, frames, median[None, :])
        mean = imputed.mean(axis=0)
        scale = imputed.std(axis=0)
        scale = np.where(scale > 1.0e-8, scale, 1.0)
        return tuple(
            np.asarray(value, dtype=np.float32)
            for value in (median, mean, scale)
        )  # type: ignore[return-value]


def set_active_trajectories(store: TemporalRaggedCache) -> None:
    global _ACTIVE_TRAJECTORIES
    _ACTIVE_TRAJECTORIES = store


class C1LookupCompatibleClassifier(idea071.BoundedWideAdditiveClassifier):
    """Original C1 parameters/preprocessing with a trailing lookup-only column ignored."""

    def forward(
        self, ast_embeddings: torch.Tensor, packed_features: torch.Tensor
    ) -> torch.Tensor:
        return super().forward(ast_embeddings, packed_features[:, :20])


class TemporalResidualClassifier(idea071.PerturbationAuditMixin, torch.nn.Module):
    def __init__(
        self,
        pipeline: str,
        ast_mean: np.ndarray,
        ast_scale: np.ndarray,
        trajectories: TemporalRaggedCache,
        train_indices: np.ndarray,
        dropout: float,
    ) -> None:
        super().__init__()
        if pipeline not in PIPELINES[2:]:
            raise ValueError(pipeline)
        self.pipeline = pipeline
        self.shuffle_frames = pipeline == PIPELINES[3]
        self.trajectories = trajectories
        safe_ast_scale = np.where(ast_scale > 1.0e-12, ast_scale, 1.0).astype(
            np.float32
        )
        self.register_buffer("ast_mean", torch.from_numpy(ast_mean.astype(np.float32)))
        self.register_buffer("ast_scale", torch.from_numpy(safe_ast_scale))
        # Preserve the exact common AST head construction order.
        self.ast_linear = torch.nn.Linear(768, 128)
        self.relu = torch.nn.ReLU()
        self.batch_norm = torch.nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01)
        self.dropout = torch.nn.Dropout(dropout)
        self.output = torch.nn.Linear(128, 3)

        median, mean, scale = trajectories.training_statistics(train_indices)
        self.register_buffer("temporal_median", torch.from_numpy(median))
        self.register_buffer("temporal_mean", torch.from_numpy(mean))
        self.register_buffer("temporal_scale", torch.from_numpy(scale))
        self.temporal_conv1 = torch.nn.Conv1d(12, 32, kernel_size=5, padding=2)
        self.temporal_conv2 = torch.nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.temporal_projection = torch.nn.Linear(32, 128)
        torch.nn.init.zeros_(self.temporal_projection.weight)
        torch.nn.init.zeros_(self.temporal_projection.bias)
        self.reset_perturbation_audit()

    @staticmethod
    def _lookup_indices(packed_features: torch.Tensor) -> list[int]:
        if packed_features.ndim != 2 or packed_features.shape[1] < 21:
            raise ValueError("IDEA-085 packed features are missing lookup column")
        lookup = packed_features[:, 20]
        rounded = torch.round(lookup).to(dtype=torch.long)
        if not torch.equal(lookup, rounded.to(dtype=lookup.dtype)):
            raise ValueError("IDEA-085 call lookup is not an exact integer")
        return [int(value) for value in rounded.detach().cpu().tolist()]

    def _prepared_sequences(
        self, packed_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences: list[torch.Tensor] = []
        lengths: list[int] = []
        for call_index in self._lookup_indices(packed_features):
            raw_numpy = self.trajectories.sequence(
                call_index, shuffled=self.shuffle_frames
            )
            raw = torch.from_numpy(np.ascontiguousarray(raw_numpy)).to(
                device=packed_features.device, dtype=torch.float32
            )
            finite = torch.isfinite(raw)
            imputed = torch.where(finite, raw, self.temporal_median)
            standardized = (imputed - self.temporal_mean) / self.temporal_scale
            model_input = torch.cat(
                [standardized, finite.to(dtype=standardized.dtype)], dim=1
            )
            sequences.append(model_input)
            lengths.append(len(model_input))
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=0.0
        )
        length_tensor = torch.tensor(
            lengths, device=padded.device, dtype=torch.long
        )
        positions = torch.arange(padded.shape[1], device=padded.device)[None, :]
        mask = positions < length_tensor[:, None]
        return padded, mask

    def temporal_context(self, packed_features: torch.Tensor) -> torch.Tensor:
        padded, mask = self._prepared_sequences(packed_features)
        mask_channel = mask[:, None, :].to(dtype=padded.dtype)
        encoded = torch.nn.functional.gelu(
            self.temporal_conv1(padded.transpose(1, 2))
        )
        encoded = encoded * mask_channel
        encoded = torch.nn.functional.gelu(self.temporal_conv2(encoded))
        encoded = encoded * mask_channel
        denominator = mask_channel.sum(dim=2).clamp_min(1.0)
        return encoded.sum(dim=2) / denominator

    def temporal_output(self, packed_features: torch.Tensor) -> torch.Tensor:
        return self.temporal_projection(self.temporal_context(packed_features))

    def forward(
        self, ast_embeddings: torch.Tensor, packed_features: torch.Tensor
    ) -> torch.Tensor:
        ast = (ast_embeddings - self.ast_mean) / self.ast_scale
        hidden = self.relu(self.ast_linear(ast))
        residual = (
            idea071.CAP
            * idea071.hidden_rms(hidden)
            * torch.tanh(self.temporal_output(packed_features))
        )
        self.record_perturbation(hidden, residual)
        hidden = hidden + residual
        hidden = self.batch_norm(hidden)
        hidden = self.dropout(hidden)
        return self.output(hidden)

    def audit(self) -> dict[str, Any]:
        return {
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in self.parameters())
            ),
            "fusion": "bounded_rms_relative_local_acoustic_temporal_residual",
            "cap": idea071.CAP,
            "rms_epsilon": idea071.RMS_EPSILON,
            "raw_channels": 6,
            "finite_indicators": 6,
            "local_receptive_field_frames": 7,
            "call_lookup_used_only_for_ragged_retrieval": True,
            "time_or_identity_feature_entered_encoder": False,
            **self.perturbation_audit(),
        }


def build_model(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    packed_features: np.ndarray,
    train_indices: np.ndarray,
) -> torch.nn.Module:
    embeddings = store.frozen_embeddings[train_indices]
    if pipeline == PIPELINES[0]:
        return _IDEA068_BUILD_MODEL(
            pipeline, protocol, store, packed_features, train_indices
        )
    common = {
        "ast_mean": embeddings.mean(axis=0),
        "ast_scale": embeddings.std(axis=0),
        "dropout": float(protocol["fixed_training"]["dropout"]),
    }
    if pipeline == PIPELINES[1]:
        return C1LookupCompatibleClassifier(
            age_train=packed_features[train_indices, :20], **common
        )
    if pipeline in PIPELINES[2:]:
        if _ACTIVE_TRAJECTORIES is None:
            raise RuntimeError("IDEA-085 trajectory cache is not active")
        return TemporalResidualClassifier(
            pipeline=pipeline,
            trajectories=_ACTIVE_TRAJECTORIES,
            train_indices=train_indices,
            **common,
        )
    raise ValueError(pipeline)


def load_features(
    protocol: dict[str, Any], call_ids: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    dense_path = REPO_ROOT / protocol["data"]["summary_feature_path"]
    if not dense_path.is_file() or sha256(dense_path) != protocol["data"]["summary_feature_sha256"]:
        raise RuntimeError("IDEA-085 original C1 summary feature cache changed")
    dense_loaded = np.load(dense_path, allow_pickle=False)
    if tuple(dense_loaded["feature_names"].astype(str)) != idea068.FEATURE_NAMES:
        raise RuntimeError("IDEA-085 original C1 feature schema changed")
    if not np.array_equal(dense_loaded["call_ids"].astype(str), call_ids.astype(str)):
        raise RuntimeError("IDEA-085 original C1 feature call order differs from AST")
    dense = dense_loaded["features"].astype(np.float32)
    if dense.shape != (len(call_ids), 20):
        raise RuntimeError("IDEA-085 original C1 feature matrix has wrong shape")

    trajectory_path = REPO_ROOT / protocol["data"]["trajectory_path"]
    expected_sha = protocol["data"]["trajectory_sha256"]
    if expected_sha.startswith("__PENDING_"):
        raise RuntimeError("IDEA-085 trajectory cache SHA is not frozen")
    if not trajectory_path.is_file() or sha256(trajectory_path) != expected_sha:
        raise RuntimeError("IDEA-085 trajectory cache checksum mismatch")
    loaded = np.load(trajectory_path, allow_pickle=False)
    required = {"call_ids", "offsets", "values", "feature_names"}
    if not required.issubset(loaded.files):
        raise RuntimeError("IDEA-085 trajectory cache contract is incomplete")
    stored_ids = loaded["call_ids"].astype(str)
    if not np.array_equal(stored_ids, call_ids.astype(str)):
        raise RuntimeError("IDEA-085 trajectory and AST call order differs")
    if tuple(loaded["feature_names"].astype(str)) != FEATURE_NAMES:
        raise RuntimeError("IDEA-085 trajectory feature names changed")
    offsets = loaded["offsets"]
    values = loaded["values"]
    if offsets.dtype != np.int64 or values.dtype != np.float32:
        raise RuntimeError("IDEA-085 trajectory dtypes changed")
    trajectories = TemporalRaggedCache(
        tuple(stored_ids.tolist()), offsets.copy(), values.copy()
    )
    set_active_trajectories(trajectories)
    lookup = np.arange(len(call_ids), dtype=np.float32)[:, None]
    packed = np.concatenate([dense, lookup], axis=1)
    finite_rates = np.isfinite(values).mean(axis=0)
    summary = {
        "feature_sha256": expected_sha,
        "trajectory_path": protocol["data"]["trajectory_path"],
        "calls": int(len(call_ids)),
        "total_frames": int(len(values)),
        "frame_length_min": int(trajectories.lengths.min()),
        "frame_length_median": float(np.median(trajectories.lengths)),
        "frame_length_max": int(trajectories.lengths.max()),
        "maximum_duration_seconds_on_10ms_grid": float(
            trajectories.lengths.max() * 0.010
        ),
        "finite_rate_by_channel": {
            name: float(finite_rates[index])
            for index, name in enumerate(FEATURE_NAMES)
        },
        "optional_time_seconds_present_but_not_loaded": "time_seconds" in loaded.files,
        "call_lookup_numeric_column_is_identity_only_not_model_input": True,
        "C1_first_20_columns_byte_equal_source": bool(
            np.array_equal(packed[:, :20], dense, equal_nan=True)
        ),
    }
    return packed.astype(np.float32), summary


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea085-acoustic-temporal-residual-v1":
        raise RuntimeError("Unexpected IDEA-085 protocol")
    if protocol.get("status") != "locked_for_cpu_preflight_before_initial_evaluation":
        raise RuntimeError("IDEA-085 protocol is not result-blind locked")
    data = protocol["data"]
    data_paths = {
        "roles_sha256": REPO_ROOT / data["roles_path"],
        "frozen_embedding_sha256": REPO_ROOT / data["frozen_embedding_path"],
        "summary_feature_sha256": REPO_ROOT / data["summary_feature_path"],
        "trajectory_sha256": REPO_ROOT / data["trajectory_path"],
    }
    for key, path in data_paths.items():
        if not path.is_file() or data.get(key) != sha256(path):
            raise RuntimeError(f"IDEA-085 locked data checksum changed: {key}")
    model = protocol["model"]
    if tuple(model["pipelines"]) != PIPELINES or tuple(model["base_seeds"]) != BASE_SEEDS:
        raise RuntimeError("IDEA-085 pipeline or seed lock changed")
    if model["repeats"] != [0, 1, 2] or model["folds"] != [0, 1, 2, 3]:
        raise RuntimeError("IDEA-085 split scope changed")
    if int(model["fits_per_pipeline"]) != 36 or int(model["total_fits"]) != 144:
        raise RuntimeError("IDEA-085 fit budget changed")
    if float(model["cap"]) != idea071.CAP or float(model["rms_epsilon"]) != idea071.RMS_EPSILON:
        raise RuntimeError("IDEA-085 residual bound changed")
    expected = {
        PIPELINES[0]: int(model["a0_trainable_parameters"]),
        PIPELINES[1]: int(model["c1_total_trainable_parameters"]),
        PIPELINES[2]: int(model["temporal_total_trainable_parameters"]),
        PIPELINES[3]: int(model["temporal_total_trainable_parameters"]),
    }
    if expected != EXPECTED_PARAMETERS:
        raise RuntimeError("IDEA-085 parameter lock changed")
    if int(model["temporal_branch_parameters"]) != 9_280:
        raise RuntimeError("IDEA-085 temporal branch parameter lock changed")
    trajectory = protocol["acoustic_trajectories"]
    if tuple(trajectory["feature_names"]) != FEATURE_NAMES:
        raise RuntimeError("IDEA-085 trajectory channel order changed")
    if (
        int(trajectory["raw_channels"]) != 6
        or int(trajectory["finite_indicators"]) != 6
        or int(trajectory["model_input_channels"]) != 12
    ):
        raise RuntimeError("IDEA-085 temporal input shape changed")
    if (
        int(trajectory["sample_rate_hz"]) != 16_000
        or int(trajectory["frame_length"]) != 1_024
        or int(trajectory["hop_length"]) != 160
    ):
        raise RuntimeError("IDEA-085 extraction grid changed")
    permutation = protocol["feature_groups"]["J1_fixed_joint_frame_permutation"]
    if permutation["material"] != PERMUTATION_MATERIAL:
        raise RuntimeError("IDEA-085 permutation material changed")
    if hashlib.sha256(PERMUTATION_MATERIAL.encode()).hexdigest() != permutation["material_sha256"]:
        raise RuntimeError("IDEA-085 permutation material digest changed")

    digest = hashlib.sha256(model["seed_derivation_text"].encode()).digest()
    if digest.hex() != model["seed_derivation_sha256"]:
        raise RuntimeError("IDEA-085 seed material digest changed")
    candidates = [
        int.from_bytes(digest[offset : offset + 4], "big") % 10_000
        for offset in range(0, 32, 4)
    ]
    if candidates != model["candidate_sequence"] or tuple(candidates[:3]) != BASE_SEEDS:
        raise RuntimeError("IDEA-085 seed derivation changed")
    current_full = {
        base + 10_000 * repeat + 100 * fold
        for base in BASE_SEEDS
        for repeat in range(3)
        for fold in range(4)
    }
    prior_full = {
        int(base) + 10_000 * repeat + 100 * fold
        for base in model["known_prior_base_seeds"]
        for repeat in range(3)
        for fold in range(5)
    }
    if len(current_full) != 36 or current_full & prior_full:
        raise RuntimeError("IDEA-085 full seed collision detected")

    source_protocol = read_json(IDEA084_PROTOCOL_PATH)
    shared = (
        "ast_head",
        "age_hidden_units",
        "dropout",
        "optimizer",
        "learning_rate",
        "optimizer_epsilon",
        "gradient_clip",
        "maximum_epochs",
        "early_stopping_patience",
        "cat_batch_size",
        "loss",
        "checkpoint_selection",
        "post_build_seed_offset",
        "paired_batch_order",
    )
    for key in shared:
        if protocol["fixed_training"][key] != source_protocol["fixed_training"][key]:
            raise RuntimeError(f"IDEA-085 changed locked training field: {key}")
    if protocol["determinism"] != source_protocol["determinism"]:
        raise RuntimeError("IDEA-085 determinism changed")
    dependencies = protocol["dependencies"]
    paths = {
        "idea068_protocol_sha256": idea068.PROTOCOL_PATH,
        "idea068_runner_sha256": Path(idea068.__file__).resolve(),
        "idea071_protocol_sha256": idea071.PROTOCOL_PATH,
        "idea071_runner_sha256": Path(idea071.__file__).resolve(),
        "idea082_protocol_sha256": idea082.PROTOCOL_PATH,
        "idea082_runner_sha256": Path(idea082.__file__).resolve(),
        "idea084_protocol_sha256": IDEA084_PROTOCOL_PATH,
        "idea084_runner_sha256": IDEA084_RUNNER_PATH,
        "idea085_runner_sha256": Path(__file__).resolve(),
        "idea085_tests_sha256": REPO_ROOT
        / "tests"
        / "test_idea085_acoustic_temporal_residual.py",
    }
    for key, path in paths.items():
        if dependencies.get(key) != sha256(path):
            raise RuntimeError(f"IDEA-085 dependency checksum changed: {key}")


def initial_model_audit(
    protocol: dict[str, Any],
    store: Any,
    features: np.ndarray,
    train_indices: np.ndarray,
    probe_indices: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    models: dict[str, torch.nn.Module] = {}
    logits: dict[str, np.ndarray] = {}
    parameters: dict[str, int] = {}
    for pipeline in PIPELINES:
        idea068.idea051.reference.historical.set_seed(seed)
        model = build_model(pipeline, protocol, store, features, train_indices).eval()
        models[pipeline] = model
        parameters[pipeline] = sum(value.numel() for value in model.parameters())
        with torch.no_grad():
            logits[pipeline] = model(
                torch.from_numpy(store.frozen_embeddings[probe_indices]),
                torch.from_numpy(features[probe_indices]),
            ).numpy()
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"IDEA-085 parameter audit mismatch: {parameters}")
    differences = {
        pipeline: float(np.max(np.abs(logits[pipeline] - logits[PIPELINES[0]])))
        for pipeline in PIPELINES[1:]
    }
    if any(value != 0.0 for value in differences.values()):
        raise RuntimeError("IDEA-085 pipelines differ at zero-residual initialization")
    common_equal = {
        pipeline: idea084.common_state_equal(models[PIPELINES[0]], models[pipeline])
        for pipeline in PIPELINES[1:]
    }
    if not all(common_equal.values()):
        raise RuntimeError("IDEA-085 common AST state differs")
    t1_j1_equal = idea084.trainable_state_equal(models[PIPELINES[2]], models[PIPELINES[3]])
    if not t1_j1_equal:
        raise RuntimeError("IDEA-085 T1/J1 trainable initial state differs")
    zero_outputs = {
        PIPELINES[1]: bool(
            torch.count_nonzero(models[PIPELINES[1]].age_output.weight) == 0
            and torch.count_nonzero(models[PIPELINES[1]].age_output.bias) == 0
        ),
        PIPELINES[2]: bool(
            torch.count_nonzero(models[PIPELINES[2]].temporal_projection.weight) == 0
            and torch.count_nonzero(models[PIPELINES[2]].temporal_projection.bias) == 0
        ),
        PIPELINES[3]: bool(
            torch.count_nonzero(models[PIPELINES[3]].temporal_projection.weight) == 0
            and torch.count_nonzero(models[PIPELINES[3]].temporal_projection.bias) == 0
        ),
    }
    if not all(zero_outputs.values()):
        raise RuntimeError("IDEA-085 residual output is not zero initialized")
    return {
        "parameters": parameters,
        "max_logit_difference_vs_A0": differences,
        "common_AST_state_equal_to_A0": common_equal,
        "T1_J1_trainable_state_equal": t1_j1_equal,
        "zero_initialized_residual_outputs": zero_outputs,
    }


def preprocessing_audit(
    model: TemporalResidualClassifier, train_indices: np.ndarray
) -> dict[str, Any]:
    expected = model.trajectories.training_statistics(train_indices)
    actual = (
        model.temporal_median.detach().cpu().numpy(),
        model.temporal_mean.detach().cpu().numpy(),
        model.temporal_scale.detach().cpu().numpy(),
    )
    equal = all(np.array_equal(left, right) for left, right in zip(expected, actual))
    if not equal:
        raise RuntimeError("IDEA-085 temporal preprocessing is not train-role local")
    return {
        "training_role_statistics_equal_recomputed": True,
        "validation_rows_used_for_statistics": False,
        "test_rows_used_for_statistics": False,
        "per_call_centering_used": False,
        "finite_indicators_standardized": False,
    }


def gradient_audit(
    model: TemporalResidualClassifier, packed_probe: np.ndarray
) -> dict[str, float]:
    packed = torch.from_numpy(packed_probe)
    model.zero_grad(set_to_none=True)
    output = model.temporal_output(packed)
    output.sum().backward()
    projection_grad = float(model.temporal_projection.bias.grad.norm())
    conv1_zero = float(model.temporal_conv1.weight.grad.norm())
    conv2_zero = float(model.temporal_conv2.weight.grad.norm())
    if projection_grad <= 0.0 or conv1_zero != 0.0 or conv2_zero != 0.0:
        raise RuntimeError("IDEA-085 zero-init gradient pattern changed")
    with torch.no_grad():
        model.temporal_projection.weight.fill_(0.01)
    model.zero_grad(set_to_none=True)
    model.temporal_output(packed).sum().backward()
    conv1_probe = float(model.temporal_conv1.weight.grad.norm())
    conv2_probe = float(model.temporal_conv2.weight.grad.norm())
    if min(conv1_probe, conv2_probe) <= 0.0:
        raise RuntimeError("IDEA-085 temporal convolution is not gradient reachable")
    return {
        "zero_init_projection_bias_grad_norm": projection_grad,
        "zero_init_conv1_weight_grad_norm": conv1_zero,
        "zero_init_conv2_weight_grad_norm": conv2_zero,
        "nonzero_projection_probe_conv1_weight_grad_norm": conv1_probe,
        "nonzero_projection_probe_conv2_weight_grad_norm": conv2_probe,
    }


def cap_audit(
    model: TemporalResidualClassifier,
    store: Any,
    features: np.ndarray,
    probe_indices: np.ndarray,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        model.temporal_projection.weight.fill_(100.0)
        model.temporal_projection.bias.fill_(100.0)
    model.reset_perturbation_audit()
    with torch.no_grad():
        model(
            torch.from_numpy(store.frozen_embeddings[probe_indices]),
            torch.from_numpy(features[probe_indices]),
        )
    result = model.perturbation_audit()
    if result["validation_relative_perturbation_max"] > idea071.CAP + 1.0e-5:
        raise RuntimeError("IDEA-085 temporal residual cap failed")
    return result


def padding_invariance_audit(
    model: TemporalResidualClassifier, features: np.ndarray
) -> dict[str, Any]:
    lengths = model.trajectories.lengths
    short_index = int(np.argmin(lengths))
    long_index = int(np.argmax(lengths))
    model.eval()
    with torch.no_grad():
        alone = model.temporal_context(
            torch.from_numpy(features[[short_index]])
        )[0]
        batched = model.temporal_context(
            torch.from_numpy(features[[short_index, long_index]])
        )[0]
    maximum_difference = float(torch.max(torch.abs(alone - batched)))
    if maximum_difference > 1.0e-6:
        raise RuntimeError("IDEA-085 padding changes a call representation")
    return {
        "short_call_frames": int(lengths[short_index]),
        "long_call_frames": int(lengths[long_index]),
        "eval_context_max_difference": maximum_difference,
        "padding_zeroed_after_each_convolution": True,
        "padding_excluded_from_masked_mean": True,
    }


def shuffle_audit(trajectories: TemporalRaggedCache) -> dict[str, Any]:
    candidates = np.flatnonzero(trajectories.lengths > 3)
    if not len(candidates):
        raise RuntimeError("IDEA-085 has no call long enough for shuffle audit")
    index = int(candidates[0])
    original = trajectories.sequence(index, shuffled=False)
    permutation = fixed_joint_permutation(trajectories.call_ids[index], len(original))
    shuffled = trajectories.sequence(index, shuffled=True)
    if sorted(permutation.tolist()) != list(range(len(original))):
        raise RuntimeError("IDEA-085 J1 permutation is not exhaustive")
    if not np.array_equal(original[permutation], shuffled, equal_nan=True):
        raise RuntimeError("IDEA-085 J1 does not preserve joint frame tuples")
    if np.array_equal(permutation, np.arange(len(original))):
        raise RuntimeError("IDEA-085 shuffle audit selected an identity permutation")
    repeated = fixed_joint_permutation(trajectories.call_ids[index], len(original))
    if not np.array_equal(permutation, repeated):
        raise RuntimeError("IDEA-085 J1 permutation is not deterministic")
    return {
        "audited_call_id": trajectories.call_ids[index],
        "frames": int(len(original)),
        "permutation_seed": permutation_seed(trajectories.call_ids[index]),
        "deterministic": True,
        "non_identity": True,
        "joint_six_channel_and_finite_tuple_multiset_preserved": True,
        "length_preserved": True,
        "padding_permuted": False,
        "labels_used": False,
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cpu":
        raise RuntimeError("IDEA-085 preflight is CPU-only")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before IDEA-085 CPU preflight")
    idea068.configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    store = idea068.idea051.reference.historical.idea019.load_feature_store()
    if len(store.call_ids) != 792 or len(np.unique(store.cat_ids.astype(str))) != 111:
        raise RuntimeError("IDEA-085 expected 792 calls from 111 cats")
    features, feature_summary = load_features(protocol, store.call_ids)
    if features.shape != (792, 21) or not feature_summary["C1_first_20_columns_byte_equal_source"]:
        raise RuntimeError("IDEA-085 packed lookup changed original C1 features")
    roles = pd.read_csv(REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    if set(roles["role"].astype(str)) != {"train", "validation", "test"}:
        raise RuntimeError("IDEA-085 role vocabulary changed")
    first_indices: dict[str, np.ndarray] | None = None
    role_cells = 0
    for repeat in protocol["model"]["repeats"]:
        for fold in protocol["model"]["folds"]:
            cell = roles[(roles["repeat"] == repeat) & (roles["outer_fold"] == fold)]
            cats = {
                role: set(cell[cell["role"] == role]["cat_id"].astype(str))
                for role in ("train", "validation", "test")
            }
            if any(
                cats[left] & cats[right]
                for left, right in (
                    ("train", "validation"),
                    ("train", "test"),
                    ("validation", "test"),
                )
            ):
                raise RuntimeError("IDEA-085 cat leakage across roles")
            indices = idea084.role_cell_indices(store, roles, repeat, fold)
            if np.intersect1d(indices["train"], indices["validation"]).size:
                raise RuntimeError("IDEA-085 call leakage across train/validation")
            if first_indices is None:
                first_indices = indices
            role_cells += 1
    if first_indices is None:
        raise RuntimeError("IDEA-085 found no role cells")
    probe_indices = first_indices["validation"][: min(16, len(first_indices["validation"]))]
    initialization = initial_model_audit(
        protocol, store, features, first_indices["train"], probe_indices, BASE_SEEDS[0]
    )
    models: dict[str, TemporalResidualClassifier] = {}
    preprocessing: dict[str, Any] = {}
    gradients: dict[str, Any] = {}
    caps: dict[str, Any] = {}
    padding: dict[str, Any] = {}
    for pipeline in PIPELINES[2:]:
        idea068.idea051.reference.historical.set_seed(BASE_SEEDS[0])
        model = build_model(pipeline, protocol, store, features, first_indices["train"])
        if not isinstance(model, TemporalResidualClassifier):
            raise RuntimeError("IDEA-085 temporal model type changed")
        models[pipeline] = model
        preprocessing[pipeline] = preprocessing_audit(model, first_indices["train"])
        gradients[pipeline] = gradient_audit(model, features[probe_indices])
        idea068.idea051.reference.historical.set_seed(BASE_SEEDS[0])
        cap_model = build_model(pipeline, protocol, store, features, first_indices["train"])
        caps[pipeline] = cap_audit(cap_model, store, features, probe_indices)
        idea068.idea051.reference.historical.set_seed(BASE_SEEDS[0])
        padding_model = build_model(pipeline, protocol, store, features, first_indices["train"])
        padding[pipeline] = padding_invariance_audit(padding_model, features)
    stats_equal = all(
        torch.equal(
            getattr(models[PIPELINES[2]], name), getattr(models[PIPELINES[3]], name)
        )
        for name in ("temporal_median", "temporal_mean", "temporal_scale")
    )
    if not stats_equal:
        raise RuntimeError("IDEA-085 T1/J1 training statistics differ")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized during IDEA-085 CPU preflight")
    result = {
        "status": "GO",
        "scope": "CPU preflight only; formal GPU remains unauthorized",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "feature_sha256": feature_summary["feature_sha256"],
        "calls": 792,
        "cats": 111,
        "role_cells": role_cells,
        "base_seeds": list(BASE_SEEDS),
        "unique_full_seeds": 36,
        "pipelines": list(PIPELINES),
        "expected_fits": 144,
        "outer_test_predictions_or_metrics_accessed": False,
        "cuda_initialized": False,
        "device": "cpu",
        "trajectory_quality": feature_summary,
        "initialization": initialization,
        "preprocessing": preprocessing,
        "T1_J1_train_only_statistics_equal": stats_equal,
        "gradient_reachability": gradients,
        "cap_saturation_probes": caps,
        "padding_invariance": padding,
        "joint_shuffle": shuffle_audit(models[PIPELINES[2]].trajectories),
    }
    output_path = idea084.resolve_run_root(args.output_subdir) / "cpu_preflight.json"
    idea084.write_json(output_path, result)
    return result


def require_matching_cpu_preflight(
    protocol: dict[str, Any], output_subdir: str
) -> tuple[dict[str, Any], Path]:
    path = idea084.resolve_run_root(output_subdir) / "cpu_preflight.json"
    if not path.is_file():
        raise RuntimeError("IDEA-085 formal run requires CPU preflight GO")
    result = read_json(path)
    expected = {
        "status": "GO",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "expected_fits": 144,
        "outer_test_predictions_or_metrics_accessed": False,
        "cuda_initialized": False,
        "device": "cpu",
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"IDEA-085 CPU preflight mismatch: {key}")
    if result.get("pipelines") != list(PIPELINES) or result.get("base_seeds") != list(BASE_SEEDS):
        raise RuntimeError("IDEA-085 CPU preflight matrix mismatch")
    if protocol["execution_gate"].get("gpu_authorized") is not False:
        raise RuntimeError("IDEA-085 protocol GPU flag must remain false")
    return result, path


def metric_bundle(frame: pd.DataFrame) -> dict[str, Any]:
    return idea082.metric_bundle(frame)


def add_pipeline_metrics(row: dict[str, Any], bundles: dict[str, Any]) -> None:
    for pipeline in PIPELINES:
        metrics = bundles[pipeline]["metrics"]
        row[f"{pipeline}_macro_f1"] = metrics["macro_f1"]
        row[f"{pipeline}_plain_accuracy"] = metrics["plain_accuracy"]
        row[f"{pipeline}_balanced_accuracy"] = metrics["balanced_accuracy"]
        row[f"{pipeline}_cross_entropy"] = bundles[pipeline]["cross_entropy"]
        row[f"{pipeline}_brier"] = bundles[pipeline]["brier"]
        for class_name in CLASS_NAMES:
            row[f"{pipeline}_{class_name}_recall"] = metrics["per_class"][class_name]["recall"]
    for name, (candidate, comparator) in {**COMPARISONS, **DESCRIPTIVE_COMPARISONS}.items():
        for metric in ("macro_f1", "plain_accuracy", "balanced_accuracy"):
            row[f"{name}_{metric}"] = row[f"{candidate}_{metric}"] - row[f"{comparator}_{metric}"]
        row[f"{name}_cross_entropy_gain"] = row[f"{comparator}_cross_entropy"] - row[f"{candidate}_cross_entropy"]
        row[f"{name}_brier_gain"] = row[f"{comparator}_brier"] - row[f"{candidate}_brier"]
        for class_name in CLASS_NAMES:
            row[f"{name}_{class_name}_recall"] = row[f"{candidate}_{class_name}_recall"] - row[f"{comparator}_{class_name}_recall"]


def aggregate(fits: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    fold_results: list[dict[str, Any]] = []
    seed_repeat_results: list[dict[str, Any]] = []
    pooled_all: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
    for base_seed in BASE_SEEDS:
        for repeat in protocol["model"]["repeats"]:
            seed_repeat_frames = {pipeline: [] for pipeline in PIPELINES}
            for fold in protocol["model"]["folds"]:
                bundles: dict[str, Any] = {}
                for pipeline in PIPELINES:
                    fit = next(
                        item for item in fits
                        if item["pipeline"] == pipeline
                        and item["base_seed"] == base_seed
                        and item["repeat"] == repeat
                        and item["fold"] == fold
                    )
                    animals = idea084.load_animals(fit)
                    bundles[pipeline] = metric_bundle(animals)
                    tagged = animals.copy()
                    tagged["base_seed"] = base_seed
                    tagged["repeat"] = repeat
                    tagged["fold"] = fold
                    seed_repeat_frames[pipeline].append(tagged)
                    pooled_all[pipeline].append(tagged)
                row: dict[str, Any] = {"base_seed": base_seed, "repeat": repeat, "fold": fold}
                add_pipeline_metrics(row, bundles)
                fold_results.append(row)
            pooled = {
                pipeline: pd.concat(parts, ignore_index=True)
                for pipeline, parts in seed_repeat_frames.items()
            }
            row = {"base_seed": base_seed, "repeat": repeat}
            add_pipeline_metrics(
                row, {pipeline: metric_bundle(frame) for pipeline, frame in pooled.items()}
            )
            for name, (candidate, comparator) in COMPARISONS.items():
                for key, value in idea084.paired_error_transitions(
                    pooled[candidate], pooled[comparator]
                ).items():
                    row[f"{name}_{key}"] = value
            seed_repeat_results.append(row)

    folds = pd.DataFrame(fold_results)
    seed_repeats = pd.DataFrame(seed_repeat_results)
    contrast_columns = [f"{name}_macro_f1" for name in COMPARISONS]
    split_cells = (
        folds.groupby(["repeat", "fold"], as_index=False)[contrast_columns]
        .mean().sort_values(["repeat", "fold"]).reset_index(drop=True)
    )
    pooled_frames = {
        pipeline: pd.concat(parts, ignore_index=True)
        for pipeline, parts in pooled_all.items()
    }
    metric_names = (
        "macro_f1", "plain_accuracy", "balanced_accuracy", "cross_entropy", "brier",
        "kitten_recall", "adult_recall", "senior_recall",
    )
    pipeline_means = {
        metric: {
            pipeline: float(seed_repeats[f"{pipeline}_{metric}"].mean())
            for pipeline in PIPELINES
        }
        for metric in metric_names
    }
    gate = protocol["classification_gate"]
    comparison_results: dict[str, Any] = {}
    for name, (candidate, comparator) in COMPARISONS.items():
        values = seed_repeats[f"{name}_macro_f1"]
        split_values = split_cells[f"{name}_macro_f1"]
        per_seed = {
            str(seed): float(seed_repeats[seed_repeats["base_seed"] == seed][f"{name}_macro_f1"].mean())
            for seed in BASE_SEEDS
        }
        conditions = {
            "mean_macro_f1_delta": float(values.mean()) >= float(gate["minimum_mean_seed_repeat_macro_f1_delta"]),
            "positive_base_seed_means": sum(value > 0 for value in per_seed.values()) >= int(gate["minimum_positive_base_seed_means"]),
            "positive_seed_repeats": int((values > 0).sum()) >= int(gate["minimum_positive_seed_repeats"]),
            "nonnegative_split_cells": int((split_values >= 0).sum()) >= int(gate["minimum_nonnegative_split_cells"]),
            "worst_split_cell": float(split_values.min()) >= float(gate["minimum_worst_split_cell_delta"]),
        }
        correction_keys = (
            "paired_occurrences", "corrected_errors", "introduced_errors",
            "net_corrections", "unchanged_correct", "unchanged_wrong",
        )
        correction_profile = {
            key: {
                "mean_per_seed_repeat": float(seed_repeats[f"{name}_{key}"].mean()),
                "total_descriptive_repeated_occurrences": int(seed_repeats[f"{name}_{key}"].sum()),
            }
            for key in correction_keys
        }
        correction_profile["net_correction_positive_tied_negative"] = {
            "positive": int((seed_repeats[f"{name}_net_corrections"] > 0).sum()),
            "tied": int((seed_repeats[f"{name}_net_corrections"] == 0).sum()),
            "negative": int((seed_repeats[f"{name}_net_corrections"] < 0).sum()),
        }
        comparison_results[name] = {
            "candidate": candidate,
            "comparator": comparator,
            "macro_f1": idea084.contrast_summary(values),
            "per_base_seed_mean_delta": per_seed,
            "split_cell_nonnegative": int((split_values >= 0).sum()),
            "split_cell_worst": float(split_values.min()),
            "classification_conditions": conditions,
            "classification_gate_passed": bool(all(conditions.values())),
            "gate_passed": bool(all(conditions.values())),
            "auxiliary_profile": {
                "mean_plain_accuracy_delta": pipeline_means["plain_accuracy"][candidate] - pipeline_means["plain_accuracy"][comparator],
                "mean_balanced_accuracy_delta": pipeline_means["balanced_accuracy"][candidate] - pipeline_means["balanced_accuracy"][comparator],
                "mean_cross_entropy_gain": pipeline_means["cross_entropy"][comparator] - pipeline_means["cross_entropy"][candidate],
                "mean_brier_gain": pipeline_means["brier"][comparator] - pipeline_means["brier"][candidate],
                "mean_class_recall_delta": {
                    class_name: pipeline_means[f"{class_name}_recall"][candidate] - pipeline_means[f"{class_name}_recall"][comparator]
                    for class_name in CLASS_NAMES
                },
                "auxiliary_axes_are_not_classification_gate_conditions": True,
            },
            "error_correction_profile": correction_profile,
            "interpretation_boundary": (
                "Real order versus one fixed joint frame permutation; not complete causality and not pure-F0 isolation."
                if name == "T1_minus_J1"
                else "Same 111 cats; not independent external confirmation."
            ),
        }
    descriptive_results = {}
    for name, (candidate, comparator) in DESCRIPTIVE_COMPARISONS.items():
        descriptive_results[name] = {
            "candidate": candidate,
            "comparator": comparator,
            "macro_f1": idea084.contrast_summary(seed_repeats[f"{name}_macro_f1"]),
            "mean_plain_accuracy_delta": pipeline_means["plain_accuracy"][candidate] - pipeline_means["plain_accuracy"][comparator],
            "mean_balanced_accuracy_delta": pipeline_means["balanced_accuracy"][candidate] - pipeline_means["balanced_accuracy"][comparator],
            "mean_cross_entropy_gain": pipeline_means["cross_entropy"][comparator] - pipeline_means["cross_entropy"][candidate],
            "mean_brier_gain": pipeline_means["brier"][comparator] - pipeline_means["brier"][candidate],
            "status": "contemporaneous_context_only",
        }
    return {
        "status": "complete",
        "outer_test_accessed": False,
        "fits": len(fits),
        "paired_fold_comparisons": len(fold_results),
        "seed_repeat_estimates": len(seed_repeat_results),
        "split_cell_estimates": len(split_cells),
        "independence_note": "Fold deltas and repeated animal occurrences reuse the same 111 cats and are descriptive, not independent samples.",
        "no_single_global_pass_flag": True,
        "pipeline_seed_repeat_means": pipeline_means,
        "comparison_results": comparison_results,
        "descriptive_results": descriptive_results,
        "fold_results": fold_results,
        "seed_repeat_results": seed_repeat_results,
        "split_cell_results": split_cells.to_dict(orient="records"),
        "pooled_validation": {
            pipeline: {"animal_occurrences": len(frame), **metric_bundle(frame)}
            for pipeline, frame in pooled_frames.items()
        },
    }


def validate_completed_fit(
    fit: dict[str, Any], pipeline: str, base_seed: int, full_seed: int,
    repeat: int, fold: int,
) -> None:
    expected = {
        "status": "complete", "pipeline": pipeline, "base_seed": base_seed,
        "full_seed": full_seed, "repeat": repeat, "fold": fold,
        "outer_test_accessed": False,
    }
    for key, value in expected.items():
        if fit.get(key) != value:
            raise RuntimeError(f"IDEA-085 resume identity mismatch: {key}")
    for prefix in ("validation_animal", "validation_call"):
        path = REPO_ROOT / fit[f"{prefix}_predictions"]
        if not path.is_file() or sha256(path) != fit[f"{prefix}_sha256"]:
            raise RuntimeError(f"IDEA-085 resume prediction hash mismatch: {path}")


def install_base_runner_hooks() -> None:
    # IDEA-084 already supplies the locked training/checkpoint/resume skeleton.
    # These hooks replace only experiment-specific identities, model construction,
    # preflight matching, and aggregation.
    idea084.__file__ = str(Path(__file__).resolve())
    idea084.PROTOCOL_PATH = PROTOCOL_PATH
    idea084.PIPELINES = PIPELINES
    idea084.BASE_SEEDS = BASE_SEEDS
    idea084.EXPECTED_PARAMETERS = EXPECTED_PARAMETERS
    idea084.COMPARISONS = COMPARISONS
    idea084.DESCRIPTIVE_COMPARISONS = DESCRIPTIVE_COMPARISONS
    idea084.build_model = build_model
    idea084.load_features = load_features
    idea084.verify_protocol = verify_protocol
    idea084.initial_model_audit = initial_model_audit
    idea084.require_matching_cpu_preflight = require_matching_cpu_preflight
    idea084.validate_completed_fit = validate_completed_fit
    idea084.metric_bundle = metric_bundle
    idea084.add_pipeline_metrics = add_pipeline_metrics
    idea084.aggregate = aggregate


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.director_authorized:
        raise RuntimeError(
            "IDEA-085 formal run is blocked until the research director explicitly authorizes it"
        )
    install_base_runner_hooks()
    return idea084.run(args)


def main() -> None:
    args = parse_args()
    result = preflight(args) if args.stage == "preflight" else run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
