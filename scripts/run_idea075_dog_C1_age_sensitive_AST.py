"""Prepare and run the IDEA-075 dog age-sensitive AST benchmark.

The observation is a bark unit, while splitting and minibatching are by dog.
A dog may legitimately contribute bark units carrying multiple age-group labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import sklearn
from scipy.signal import resample_poly
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_idea068_age_sensitive_ast as idea068  # noqa: E402


PROTOCOL_PATH = REPO_ROOT / "configs" / "protocol" / "idea075_dog_C1_age_sensitive_AST_v1.json"
PIPELINES = ("A0_ast_only", "U1_unbounded_age_residual", "C1_bounded_age_residual")
CLASS_NAMES = ("puppy", "juvenile", "adolescent", "adult", "senior")
PROBABILITY_COLUMNS = tuple(f"prob_{name}" for name in CLASS_NAMES)
FEATURE_NAMES = idea068.FEATURE_NAMES
BASE_SEEDS = (8075, 4270, 1872)
SPLIT_SEEDS = (17, 43, 101)
EXPECTED_PARAMETERS = {
    PIPELINES[0]: 99_333,
    PIPELINES[1]: 108_401,
    PIPELINES[2]: 108_401,
}
AGE_HIDDEN_UNITS = 60
CAP = 0.25
RMS_EPSILON = 1.0e-8
DEFAULT_RUN_SUBDIR = "idea075_dog_C1_age_sensitive_AST_v1"
EXPECTED_ROLES_SHA256 = "c269b009fdae3747fbeec31f3d252fad2c8e86656e2447cfe98a6f77ab7e7da5"
EXPECTED_GATE = {
    "tier1_C1_package_vs_A0": {
        "minimum_mean_macro_f1_gain": 0.005,
        "required_positive_base_seeds": 3,
        "minimum_positive_seed_repeats": 6,
        "minimum_nonnegative_split_cells": 10,
        "minimum_worst_split_cell_delta": -0.03,
        "require_mean_CE_nonworse": True,
        "require_mean_Brier_nonworse": True,
        "minimum_mean_balanced_accuracy_delta": 0.0,
        "minimum_each_base_seed_senior_recall_delta": -0.02,
    },
    "tier2_C1_bound_vs_U1": {
        "require_mean_macro_f1_strictly_positive": True,
        "minimum_strictly_positive_base_seeds": 2,
        "minimum_strictly_positive_seed_repeats": 5,
        "require_mean_CE_nonworse": True,
        "require_mean_Brier_nonworse": True,
    },
    "tier3_external_task_floor": {
        "required_C1_seed_repeats_strictly_above_outer_train_prior_dummy": 9,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("extract-features", "run"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-subdir", default=DEFAULT_RUN_SUBDIR)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_run_root(output_subdir: str) -> Path:
    runs_root = (REPO_ROOT / "runs").resolve()
    run_root = (runs_root / output_subdir).resolve()
    if runs_root not in run_root.parents:
        raise ValueError("--output-subdir must stay below runs")
    return run_root


def configure_determinism() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _path_and_hash(section: dict[str, Any], path_key: str, hash_key: str) -> Path:
    path = REPO_ROOT / section[path_key]
    expected = str(section.get(hash_key, ""))
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected and sha256(path) != expected:
        raise RuntimeError(f"IDEA-075 checksum mismatch: {path}")
    return path


def verify_protocol(protocol: dict[str, Any], require_prepared: bool = False) -> None:
    if protocol.get("protocol_id") != "idea075-dog-C1-age-sensitive-AST-v1":
        raise RuntimeError("Unexpected IDEA-075 protocol")
    if tuple(protocol["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-075 pipeline matrix changed")
    expected_status = "locked_before_initial_evaluation" if require_prepared else "locked_before_asset_preparation"
    if protocol.get("status") != expected_status:
        raise RuntimeError(f"IDEA-075 protocol has wrong stage status; expected {expected_status}")
    if tuple(protocol["classes"]) != CLASS_NAMES:
        raise RuntimeError("IDEA-075 class order changed")
    if tuple(protocol["base_seeds"]) != BASE_SEEDS:
        raise RuntimeError("IDEA-075 base seeds changed")
    if tuple(protocol["outer_splits"]["split_seeds"]) != SPLIT_SEEDS:
        raise RuntimeError("IDEA-075 split seeds changed")
    if tuple(protocol["acoustic_features"]["feature_names"]) != FEATURE_NAMES:
        raise RuntimeError("IDEA-075 acoustic feature definition changed")
    acoustics = protocol["acoustic_features"]
    expected_acoustics = {
        "sample_rate_hz": 16000,
        "fmin_hz": 60.0,
        "fmax_hz": 2000.0,
        "frame_length": 1024,
        "hop_length": 160,
    }
    for key, expected in expected_acoustics.items():
        if float(acoustics[key]) != float(expected):
            raise RuntimeError(f"IDEA-075 acoustic setting changed: {key}")
    if tuple(float(value) for value in acoustics["spectral_tilt_band_hz"]) != (200.0, 4000.0):
        raise RuntimeError("IDEA-075 spectral tilt band changed")
    fixed = protocol["fixed_training"]
    expected_fixed = {
        "age_hidden_units": AGE_HIDDEN_UNITS,
        "dog_batch_size": 4,
        "maximum_epochs": 50,
        "early_stopping_patience": 8,
    }
    for key, expected in expected_fixed.items():
        if int(fixed[key]) != expected:
            raise RuntimeError(f"IDEA-075 fixed setting changed: {key}")
    expected_float_fixed = {
        "dropout": 0.44571035356880917,
        "learning_rate": 0.006,
        "optimizer_epsilon": 1.0e-7,
        "gradient_clip": 1.0,
        "post_build_seed_offset": 1_000_000,
    }
    for key, expected in expected_float_fixed.items():
        if float(fixed[key]) != expected:
            raise RuntimeError(f"IDEA-075 fixed setting changed: {key}")
    if fixed["optimizer"] != "Adamax":
        raise RuntimeError("IDEA-075 optimizer changed")
    if float(protocol["model"]["relative_cap"]) != CAP:
        raise RuntimeError("IDEA-075 C1 cap changed")
    if float(protocol["model"]["rms_epsilon"]) != RMS_EPSILON:
        raise RuntimeError("IDEA-075 RMS epsilon changed")
    data = protocol["data"]
    _path_and_hash(data, "manifest_path", "manifest_sha256")
    _path_and_hash(data, "frozen_embedding_path", "frozen_embedding_sha256")
    _path_and_hash(data, "idea067_oof_path", "idea067_oof_sha256")
    if protocol.get("gate") != EXPECTED_GATE:
        raise RuntimeError("IDEA-075 preregistered gate changed")
    dependencies = protocol.get("dependencies", {})
    for stem in (
        "plan",
        "idea067_protocol",
        "idea067_runner",
        "idea068_protocol",
        "idea068_runner",
        "runner",
    ):
        path_key, hash_key = f"{stem}_path", f"{stem}_sha256"
        if path_key in dependencies:
            _path_and_hash(dependencies, path_key, hash_key)
    if require_prepared:
        for path_key, hash_key in (
            ("roles_path", "roles_sha256"),
            ("source_audio_inventory_path", "source_audio_inventory_sha256"),
            ("age_feature_path", "age_feature_sha256"),
            ("feature_summary_path", "feature_summary_sha256"),
        ):
            _path_and_hash(data, path_key, hash_key)


def load_audio(path: Path, target_rate: int) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if waveform.size == 0:
        raise RuntimeError(f"Empty audio: {path}")
    if sample_rate != target_rate:
        divisor = math.gcd(int(sample_rate), target_rate)
        waveform = resample_poly(
            waveform, target_rate // divisor, int(sample_rate) // divisor
        ).astype(np.float32)
    return waveform.astype(np.float32, copy=False)


def load_locked_inputs(protocol: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    manifest = pd.read_csv(REPO_ROOT / protocol["data"]["manifest_path"], dtype={"recording_id": str, "animal_id": str})
    if manifest["recording_id"].duplicated().any():
        raise RuntimeError("Duplicate recording_id in canine manifest")
    frozen = np.load(REPO_ROOT / protocol["data"]["frozen_embedding_path"], allow_pickle=False)
    recording_ids = frozen["recording_ids"].astype(str)
    if not np.array_equal(manifest["recording_id"].to_numpy(str), recording_ids):
        raise RuntimeError("Manifest and frozen AST recording order differ")
    if not np.array_equal(manifest["animal_id"].to_numpy(str), frozen["animal_ids"].astype(str)):
        raise RuntimeError("Manifest and frozen AST dog order differ")
    if not np.array_equal(manifest["audio_path"].to_numpy(str), frozen["source_paths"].astype(str)):
        raise RuntimeError("Manifest and frozen AST source paths differ")
    label_map = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = manifest["label_name"].map(label_map)
    if labels.isna().any() or not np.array_equal(labels.to_numpy(np.int64), frozen["labels"].astype(np.int64)):
        raise RuntimeError("Manifest and frozen AST labels differ")
    if tuple(frozen["label_names"].astype(str)) != CLASS_NAMES:
        raise RuntimeError("Frozen AST class order changed")
    if frozen["embeddings"].shape != (2290, 768):
        raise RuntimeError("Unexpected canine AST matrix shape")
    return manifest, {key: frozen[key] for key in frozen.files}


def reconstruct_roles(protocol: dict[str, Any], manifest: pd.DataFrame) -> pd.DataFrame:
    oof = pd.read_csv(REPO_ROOT / protocol["data"]["idea067_oof_path"], dtype={"recording_id": str, "animal_id": str})
    expected_pairs = {(index, seed) for index, seed in enumerate(SPLIT_SEEDS)}
    if set(map(tuple, oof[["repeat", "seed"]].drop_duplicates().to_numpy())) != expected_pairs:
        raise RuntimeError("IDEA-067 repeat/seed mapping changed")
    label_map = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = manifest["label_name"].map(label_map).to_numpy(np.int64)
    dogs = manifest["animal_id"].to_numpy(str)
    ids = manifest["recording_id"].to_numpy(str)
    rows: list[pd.DataFrame] = []
    for repeat, split_seed in enumerate(SPLIT_SEEDS):
        repeat_oof = oof[oof["repeat"] == repeat]
        by_id = repeat_oof.set_index("recording_id", verify_integrity=True)
        if set(by_id.index) != set(ids):
            raise RuntimeError("IDEA-067 OOF IDs do not cover canine manifest")
        outer_fold_by_row = by_id.loc[ids, "fold"].to_numpy(np.int64)
        if not np.array_equal(by_id.loc[ids, "animal_id"].to_numpy(str), dogs):
            raise RuntimeError("IDEA-067 OOF dog IDs changed")
        if not np.array_equal(by_id.loc[ids, "true_label"].to_numpy(np.int64), labels):
            raise RuntimeError("IDEA-067 OOF true labels changed")
        for outer_fold in range(5):
            test_mask = outer_fold_by_row == outer_fold
            outer_train = np.flatnonzero(~test_mask)
            splitter = StratifiedGroupKFold(
                n_splits=4,
                shuffle=True,
                random_state=split_seed + 1000 + outer_fold,
            )
            fit_local, validation_local = next(
                splitter.split(outer_train, labels[outer_train], dogs[outer_train])
            )
            role = np.full(len(manifest), "test", dtype="U10")
            role[outer_train[fit_local]] = "fit"
            role[outer_train[validation_local]] = "validation"
            frame = pd.DataFrame(
                {
                    "repeat": repeat,
                    "outer_split_seed": split_seed,
                    "outer_fold": outer_fold,
                    "inner_seed": split_seed + 1000 + outer_fold,
                    "recording_id": ids,
                    "dog_id": dogs,
                    "age_group": manifest["label_name"].to_numpy(str),
                    "role": role,
                }
            )
            validate_role_cell(frame)
            rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    if len(result) != 15 * len(manifest):
        raise RuntimeError("IDEA-075 role table size mismatch")
    return result


def validate_role_cell(frame: pd.DataFrame) -> None:
    role_dogs = {
        role: set(frame.loc[frame["role"] == role, "dog_id"].astype(str))
        for role in ("fit", "validation", "test")
    }
    if role_dogs["fit"] & role_dogs["validation"] or role_dogs["fit"] & role_dogs["test"] or role_dogs["validation"] & role_dogs["test"]:
        raise RuntimeError("Dog leakage across IDEA-075 roles")
    for role in role_dogs:
        labels = frame.loc[frame["role"] == role, "age_group"]
        normalized = set(labels.astype(str)) if not np.issubdtype(labels.dtype, np.number) else {CLASS_NAMES[int(value)] for value in labels}
        if normalized != set(CLASS_NAMES):
            raise RuntimeError(f"IDEA-075 role is missing a class: {role}")


def _prepared_paths(run_root: Path) -> dict[str, Path]:
    return {
        "roles": run_root / "preflight" / "roles.csv",
        "inventory": run_root / "preflight" / "source_audio_inventory.csv",
        "features": run_root / "features" / "dog_age_sensitive_acoustic_features.npz",
        "summary": run_root / "features" / "extraction_summary.json",
    }


def prepare_features_and_roles(protocol: dict[str, Any], run_root: Path, resume: bool = False) -> dict[str, Any]:
    paths = _prepared_paths(run_root)
    if any(path.exists() for path in paths.values()):
        if not resume:
            raise FileExistsError("IDEA-075 prepare outputs already or partially exist")
        if all(path.is_file() for path in paths.values()):
            summary = read_json(paths["summary"])
            hashes = {name: sha256(path) for name, path in paths.items() if name != "summary"}
            if summary.get("artifact_sha256") != hashes:
                raise RuntimeError("Prepared IDEA-075 artifact hash mismatch")
            return summary
    manifest, frozen = load_locked_inputs(protocol)
    roles = reconstruct_roles(protocol, manifest)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    roles.to_csv(paths["roles"], index=False, lineterminator="\n")
    if sha256(paths["roles"]) != EXPECTED_ROLES_SHA256:
        raise RuntimeError("IDEA-075 reconstructed roles hash changed")
    settings = protocol["acoustic_features"]
    target_rate = int(settings["sample_rate_hz"])
    feature_rows: list[np.ndarray] = []
    inventory_rows: list[dict[str, Any]] = []
    analysis_frames: list[int] = []
    voiced_frames: list[int] = []
    started = time.perf_counter()
    canonical_digest = hashlib.sha256()
    manifest_rows = list(manifest.itertuples(index=False))
    # Audit every raw source before extracting the first feature. The canonical
    # content lock deliberately excludes paths and audio-decoder behavior.
    for row in manifest_rows:
        audio_path = REPO_ROOT / str(row.audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        source_hash = sha256(audio_path)
        info = sf.info(audio_path)
        if int(info.samplerate) != 16000 or int(info.channels) != 1 or str(info.subtype) != "PCM_16":
            raise RuntimeError(f"Unexpected locked canine audio header: {audio_path}")
        canonical_digest.update(str(row.recording_id).encode("utf-8"))
        canonical_digest.update(b"\0")
        canonical_digest.update(bytes.fromhex(source_hash))
        inventory_rows.append(
            {
                "recording_id": str(row.recording_id),
                "dog_id": str(row.animal_id),
                "audio_path": str(row.audio_path),
                "sha256": source_hash,
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "frames": int(info.frames),
                "subtype": str(info.subtype),
            }
        )
    canonical_hash = canonical_digest.hexdigest()
    expected_canonical_hash = str(protocol["data"].get("canonical_audio_content_sha256", ""))
    if not expected_canonical_hash or canonical_hash != expected_canonical_hash:
        raise RuntimeError("IDEA-075 canonical source-audio content hash mismatch")
    if pd.DataFrame(inventory_rows)["sha256"].duplicated().any():
        raise RuntimeError("Duplicate canine source-audio SHA-256")
    for row_index, row in enumerate(manifest_rows):
        audio_path = REPO_ROOT / str(row.audio_path)
        waveform = load_audio(audio_path, target_rate)
        values = idea068.extract_call_features(waveform, target_rate, settings)
        if np.isinf(values).any():
            raise RuntimeError(f"Infinite acoustic feature: {row.recording_id}")
        frame_count = 1 + len(waveform) // int(settings["hop_length"])
        voiced_fraction = float(values[idea068.VOICED_FRACTION_INDEX])
        feature_rows.append(values)
        analysis_frames.append(frame_count)
        voiced_frames.append(int(round(voiced_fraction * frame_count)) if np.isfinite(voiced_fraction) else 0)
        if (row_index + 1) % 100 == 0 or row_index + 1 == len(manifest):
            print(f"IDEA-075 extracted {row_index + 1}/{len(manifest)} bark units", flush=True)
    features = np.stack(feature_rows).astype(np.float32)
    if features.shape != (2290, len(FEATURE_NAMES)):
        raise RuntimeError("IDEA-075 acoustic feature matrix shape changed")
    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(paths["inventory"], index=False, lineterminator="\n")
    np.savez_compressed(
        paths["features"],
        features=features,
        feature_names=np.asarray(FEATURE_NAMES, dtype="U"),
        recording_ids=frozen["recording_ids"].astype("U"),
        dog_ids=frozen["animal_ids"].astype("U"),
        analysis_frame_counts=np.asarray(analysis_frames, dtype=np.int32),
        voiced_f0_frame_counts=np.asarray(voiced_frames, dtype=np.int32),
    )
    artifact_hashes = {
        "roles": sha256(paths["roles"]),
        "inventory": sha256(paths["inventory"]),
        "features": sha256(paths["features"]),
    }
    summary = {
        "status": "complete",
        "label_information_used_for_feature_extraction": False,
        "role_labels_used_only_for_grouped_splitting": True,
        "recordings": len(manifest),
        "dogs": int(manifest["animal_id"].nunique()),
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "missing_values_by_feature": {
            name: int(np.isnan(features[:, index]).sum()) for index, name in enumerate(FEATURE_NAMES)
        },
        "fully_finite_recordings": int(np.isfinite(features).all(axis=1).sum()),
        "canonical_audio_content_sha256": canonical_hash,
        "all_missing_features": [
            name for index, name in enumerate(FEATURE_NAMES) if np.isnan(features[:, index]).all()
        ],
        "analysis_frame_count_min_median_max": [
            int(np.min(analysis_frames)), float(np.median(analysis_frames)), int(np.max(analysis_frames))
        ],
        "voiced_f0_frame_count_min_median_max": [
            int(np.min(voiced_frames)), float(np.median(voiced_frames)), int(np.max(voiced_frames))
        ],
        "roles_path": paths["roles"].relative_to(REPO_ROOT).as_posix(),
        "source_audio_inventory_path": paths["inventory"].relative_to(REPO_ROOT).as_posix(),
        "feature_path": paths["features"].relative_to(REPO_ROOT).as_posix(),
        "artifact_sha256": artifact_hashes,
        "runner_sha256": sha256(Path(__file__).resolve()),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(paths["summary"], summary)
    return summary


@dataclass(frozen=True)
class DogStore:
    embeddings: np.ndarray
    age_features: np.ndarray
    recording_ids: np.ndarray
    dog_ids: np.ndarray
    labels: np.ndarray


class DogSetDataset(torch.utils.data.Dataset):
    """One item per complete dog; unit labels are retained without aggregation."""

    def __init__(self, store: DogStore, indices: Sequence[int]) -> None:
        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1 or len(selected) == 0:
            raise ValueError("DogSetDataset needs non-empty one-dimensional indices")
        by_dog: dict[str, list[int]] = {}
        for index in selected:
            by_dog.setdefault(str(store.dog_ids[index]), []).append(int(index))
        self.store = store
        self.dogs = tuple(sorted(by_dog))
        self.indices_by_dog = {
            dog: np.asarray(sorted(by_dog[dog]), dtype=np.int64) for dog in self.dogs
        }
        covered = np.concatenate([self.indices_by_dog[dog] for dog in self.dogs])
        if not np.array_equal(np.sort(covered), np.sort(selected)):
            raise RuntimeError("DogSetDataset coverage mismatch")

    def __len__(self) -> int:
        return len(self.dogs)

    def __getitem__(self, item: int) -> dict[str, Any]:
        dog_id = self.dogs[item]
        indices = self.indices_by_dog[dog_id]
        return {
            "dog_id": dog_id,
            "indices": torch.from_numpy(indices),
            "embeddings": torch.from_numpy(self.store.embeddings[indices]),
            "age_features": torch.from_numpy(self.store.age_features[indices]),
            "labels": torch.from_numpy(self.store.labels[indices]),
        }


def collate_dogs(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise RuntimeError("Empty dog batch")
    counts = [len(item["indices"]) for item in items]
    return {
        "dog_ids": [item["dog_id"] for item in items],
        "dog_index": torch.repeat_interleave(torch.arange(len(items)), torch.tensor(counts)),
        "indices": torch.cat([item["indices"] for item in items]),
        "embeddings": torch.cat([item["embeddings"] for item in items]),
        "age_features": torch.cat([item["age_features"] for item in items]),
        "labels": torch.cat([item["labels"] for item in items]),
    }


class NoSingletonDogBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Yield at most ``batch_size`` dogs and merge a final singleton batch."""

    def __init__(self, size: int, batch_size: int, shuffle: bool, seed: int) -> None:
        if size < 2 or batch_size < 2:
            raise ValueError("No-singleton dog batching requires at least two dogs")
        self.size, self.batch_size, self.shuffle, self.seed = size, batch_size, shuffle, seed
        self.epoch = 0

    def __len__(self) -> int:
        quotient, remainder = divmod(self.size, self.batch_size)
        return quotient + bool(remainder)

    def __iter__(self) -> Iterable[list[int]]:
        indices = np.arange(self.size)
        if self.shuffle:
            indices = np.random.default_rng(self.seed + self.epoch).permutation(indices)
        self.epoch += 1
        batches = [indices[start : start + self.batch_size].tolist() for start in range(0, self.size, self.batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            singleton = batches.pop()[0]
            moved = batches[-1].pop()
            batches.append([moved, singleton])
        if any(len(batch) < 2 or len(batch) > self.batch_size for batch in batches):
            raise RuntimeError("Invalid no-singleton dog batch")
        yield from batches


def build_dog_loader(dataset: DogSetDataset, batch_size: int, shuffle: bool, seed: int) -> torch.utils.data.DataLoader:
    sampler = NoSingletonDogBatchSampler(len(dataset), batch_size, shuffle, seed)
    return torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_dogs, num_workers=0)


def hidden_rms(hidden: torch.Tensor) -> torch.Tensor:
    """IDEA-071 RMS anchor: FP32 reduction, detached, then restored dtype."""
    return (
        torch.sqrt(hidden.float().square().mean(dim=1, keepdim=True) + RMS_EPSILON)
        .detach()
        .to(hidden.dtype)
    )


class DogAgeClassifier(torch.nn.Module):
    def __init__(
        self,
        pipeline: str,
        ast_mean: np.ndarray,
        ast_scale: np.ndarray,
        age_train: np.ndarray,
        dropout: float,
    ) -> None:
        super().__init__()
        if pipeline not in PIPELINES:
            raise ValueError(pipeline)
        self.pipeline = pipeline
        safe_scale = np.where(ast_scale > 1.0e-12, ast_scale, 1.0).astype(np.float32)
        self.register_buffer("ast_mean", torch.from_numpy(ast_mean.astype(np.float32)))
        self.register_buffer("ast_scale", torch.from_numpy(safe_scale))
        self.ast_linear = torch.nn.Linear(768, 128)
        self.relu = torch.nn.ReLU()
        self.batch_norm = torch.nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01)
        self.dropout = torch.nn.Dropout(dropout)
        self.output = torch.nn.Linear(128, 5)
        self._ratios: list[np.ndarray] = []
        self._budget_violations = 0
        if pipeline == PIPELINES[0]:
            self.age_hidden = None
            self.age_output = None
            return
        with np.errstate(all="ignore"):
            age_median = np.nanmedian(age_train, axis=0)
        all_missing = ~np.isfinite(age_median)
        age_median = np.where(all_missing, 0.0, age_median).astype(np.float32)
        imputed = np.where(np.isfinite(age_train), age_train, age_median[None, :])
        age_mean = imputed.mean(axis=0).astype(np.float32)
        age_scale = imputed.std(axis=0).astype(np.float32)
        age_mean[all_missing] = 0.0
        age_scale[all_missing] = 1.0
        age_scale = np.where(age_scale > 1.0e-8, age_scale, 1.0).astype(np.float32)
        self.register_buffer("age_median", torch.from_numpy(age_median))
        self.register_buffer("age_mean", torch.from_numpy(age_mean))
        self.register_buffer("age_scale", torch.from_numpy(age_scale))
        self.register_buffer("age_all_missing", torch.from_numpy(all_missing))
        self.age_hidden = torch.nn.Linear(len(FEATURE_NAMES), AGE_HIDDEN_UNITS)
        self.age_output = torch.nn.Linear(AGE_HIDDEN_UNITS, 128)
        torch.nn.init.zeros_(self.age_output.weight)
        torch.nn.init.zeros_(self.age_output.bias)

    def reset_perturbation_audit(self) -> None:
        self._ratios = []
        self._budget_violations = 0

    def forward(self, ast_embeddings: torch.Tensor, age_features: torch.Tensor) -> torch.Tensor:
        ast = (ast_embeddings - self.ast_mean) / self.ast_scale
        hidden = self.relu(self.ast_linear(ast))
        if self.pipeline != PIPELINES[0]:
            if self.age_hidden is None or self.age_output is None:
                raise RuntimeError("Missing IDEA-075 age branch")
            imputed = torch.where(torch.isfinite(age_features), age_features, self.age_median)
            standardized = (imputed - self.age_mean) / self.age_scale
            raw = self.age_output(torch.nn.functional.gelu(self.age_hidden(standardized)))
            if self.pipeline == PIPELINES[2]:
                anchor = hidden_rms(hidden)
                residual = CAP * anchor * torch.tanh(raw)
                with torch.no_grad():
                    limit = CAP * anchor
                    self._budget_violations += int((residual.abs() > limit + 1.0e-6).sum().item())
                    ratio = torch.linalg.vector_norm(residual, dim=1) / torch.clamp(torch.linalg.vector_norm(hidden, dim=1), min=1.0e-12)
                    self._ratios.append(ratio.detach().float().cpu().numpy())
            else:
                residual = raw
            hidden = hidden + residual
        return self.output(self.dropout(self.batch_norm(hidden)))

    def audit(self) -> dict[str, Any]:
        ratios = np.concatenate(self._ratios) if self._ratios else np.asarray([], dtype=float)
        result: dict[str, Any] = {
            "trainable_parameters": int(sum(parameter.numel() for parameter in self.parameters())),
            "maximum_dimension_budget_violations": int(self._budget_violations),
        }
        if hasattr(self, "age_all_missing"):
            result["fit_role_all_missing_features"] = [
                FEATURE_NAMES[index]
                for index, missing in enumerate(self.age_all_missing.detach().cpu().numpy())
                if bool(missing)
            ]
        if ratios.size:
            result["residual_to_hidden_norm_ratio"] = {
                "count": int(len(ratios)),
                "mean": float(ratios.mean()),
                "q50": float(np.quantile(ratios, 0.5)),
                "q90": float(np.quantile(ratios, 0.9)),
                "q99": float(np.quantile(ratios, 0.99)),
                "max": float(ratios.max()),
            }
        return result


def build_model(pipeline: str, store: DogStore, fit_indices: np.ndarray, dropout: float) -> DogAgeClassifier:
    ast = store.embeddings[fit_indices]
    return DogAgeClassifier(pipeline, ast.mean(axis=0), ast.std(axis=0), store.age_features[fit_indices], dropout)


def initialization_audit(store: DogStore, fit_indices: np.ndarray, probe_indices: np.ndarray, dropout: float, seed: int) -> dict[str, Any]:
    states: dict[str, dict[str, torch.Tensor]] = {}
    logits: dict[str, np.ndarray] = {}
    parameters: dict[str, int] = {}
    for pipeline in PIPELINES:
        set_seed(seed)
        model = build_model(pipeline, store, fit_indices, dropout).eval()
        states[pipeline] = {key: value.detach().clone() for key, value in model.state_dict().items()}
        parameters[pipeline] = sum(parameter.numel() for parameter in model.parameters())
        with torch.no_grad():
            logits[pipeline] = model(torch.from_numpy(store.embeddings[probe_indices]), torch.from_numpy(store.age_features[probe_indices])).numpy()
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"IDEA-075 parameter mismatch: {parameters}")
    common_keys = set(states[PIPELINES[0]])
    shared_equal = all(
        torch.equal(states[PIPELINES[0]][key], states[pipeline][key])
        for pipeline in PIPELINES[1:]
        for key in common_keys
    )
    if not shared_equal:
        raise RuntimeError("IDEA-075 shared model states differ at initialization")
    u1_c1_equal = states[PIPELINES[1]].keys() == states[PIPELINES[2]].keys() and all(
        torch.equal(states[PIPELINES[1]][key], states[PIPELINES[2]][key]) for key in states[PIPELINES[1]]
    )
    if not u1_c1_equal:
        raise RuntimeError("IDEA-075 U1/C1 states differ at initialization")
    differences = {
        pipeline: float(np.max(np.abs(logits[pipeline] - logits[PIPELINES[0]]))) for pipeline in PIPELINES[1:]
    }
    if any(value != 0.0 for value in differences.values()):
        raise RuntimeError("IDEA-075 initial logits differ")
    return {
        "trainable_parameters": parameters,
        "shared_state_equal": True,
        "U1_C1_full_state_equal": True,
        "max_logit_difference_vs_A0": differences,
    }


def dog_equal_unit_cross_entropy(predictions: pd.DataFrame) -> float:
    probabilities = predictions[list(PROBABILITY_COLUMNS)].to_numpy(float)
    labels = predictions["true_label"].to_numpy(np.int64)
    unit_ce = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1.0e-7, 1.0))
    frame = pd.DataFrame({"dog_id": predictions["dog_id"].astype(str), "unit_ce": unit_ce})
    return float(frame.groupby("dog_id", sort=False)["unit_ce"].mean().mean())


def load_prepared_store(protocol: dict[str, Any]) -> tuple[DogStore, pd.DataFrame]:
    manifest, frozen = load_locked_inputs(protocol)
    data = protocol["data"]
    loaded = np.load(REPO_ROOT / data["age_feature_path"], allow_pickle=False)
    if tuple(loaded["feature_names"].astype(str)) != FEATURE_NAMES:
        raise RuntimeError("IDEA-075 stored acoustic feature names changed")
    ids = frozen["recording_ids"].astype(str)
    if not np.array_equal(loaded["recording_ids"].astype(str), ids):
        raise RuntimeError("IDEA-075 acoustic/AST recording order differs")
    if not np.array_equal(loaded["dog_ids"].astype(str), frozen["animal_ids"].astype(str)):
        raise RuntimeError("IDEA-075 acoustic/AST dog order differs")
    features = loaded["features"].astype(np.float32)
    if features.shape != (len(ids), len(FEATURE_NAMES)) or np.isinf(features).any():
        raise RuntimeError("Invalid IDEA-075 feature matrix")
    store = DogStore(
        embeddings=frozen["embeddings"].astype(np.float32),
        age_features=features,
        recording_ids=ids,
        dog_ids=frozen["animal_ids"].astype(str),
        labels=frozen["labels"].astype(np.int64),
    )
    roles = pd.read_csv(
        REPO_ROOT / data["roles_path"],
        dtype={"recording_id": str, "dog_id": str},
    )
    if len(roles) != 15 * len(ids):
        raise RuntimeError("IDEA-075 locked role table size changed")
    for (_, _), frame in roles.groupby(["repeat", "outer_fold"], sort=True):
        validate_role_cell(frame)
        if not np.array_equal(frame["recording_id"].to_numpy(str), ids):
            raise RuntimeError("IDEA-075 role/AST recording order differs")
        if not np.array_equal(frame["dog_id"].to_numpy(str), store.dog_ids):
            raise RuntimeError("IDEA-075 role/AST dog order differs")
        role_labels = frame["age_group"].map({name: index for index, name in enumerate(CLASS_NAMES)}).to_numpy(np.int64)
        if not np.array_equal(role_labels, store.labels):
            raise RuntimeError("IDEA-075 role/AST labels differ")
    return store, roles


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("An IDEA-075 fit role is missing a class")
    return (len(labels) / (len(CLASS_NAMES) * counts)).astype(np.float32)


def train_one_epoch(
    model: DogAgeClassifier,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    unit_class_weights: torch.Tensor,
    device: torch.device,
    gradient_clip: float,
) -> tuple[float, dict[str, Any]]:
    model.train()
    losses: list[float] = []
    processed_dogs: list[str] = []
    processed_units: list[int] = []
    dog_batch_sizes: list[int] = []
    for cpu_batch in loader:
        embeddings = cpu_batch["embeddings"].to(device)
        ages = cpu_batch["age_features"].to(device)
        labels = cpu_batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(embeddings, ages)
            per_unit = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            weights = unit_class_weights[labels]
            loss = (per_unit * weights).sum() / weights.sum()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        processed_dogs.extend(str(value) for value in cpu_batch["dog_ids"])
        processed_units.extend(int(value) for value in cpu_batch["indices"].numpy())
        dog_batch_sizes.append(len(cpu_batch["dog_ids"]))
    if len(processed_dogs) != len(set(processed_dogs)):
        raise RuntimeError("An IDEA-075 epoch repeated a dog")
    if len(processed_units) != len(set(processed_units)):
        raise RuntimeError("An IDEA-075 epoch repeated a bark unit")
    if any(size < 2 or size > 4 for size in dog_batch_sizes):
        raise RuntimeError("IDEA-075 produced an invalid dog batch")
    return float(np.mean(losses)), {
        "dogs": len(processed_dogs),
        "bark_units": len(processed_units),
        "batch_sizes_dogs": dog_batch_sizes,
        "dog_order_sha256": hashlib.sha256("\n".join(processed_dogs).encode("utf-8")).hexdigest(),
        "recording_coverage_sha256": hashlib.sha256(
            np.sort(np.asarray(processed_units, dtype="<i8")).tobytes()
        ).hexdigest(),
    }


def predict_units(
    model: DogAgeClassifier,
    store: DogStore,
    indices: np.ndarray,
    device: torch.device,
    dog_batch_size: int,
    seed: int,
) -> pd.DataFrame:
    dataset = DogSetDataset(store, indices)
    loader = build_dog_loader(dataset, dog_batch_size, False, seed)
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(batch["embeddings"].to(device), batch["age_features"].to(device))
                probabilities = torch.softmax(logits, dim=1).float().cpu().numpy()
            for local, index in enumerate(batch["indices"].numpy().astype(np.int64)):
                rows.append(
                    {
                        "recording_index": int(index),
                        "recording_id": str(store.recording_ids[index]),
                        "dog_id": str(store.dog_ids[index]),
                        "true_label": int(store.labels[index]),
                        **{
                            column: float(probabilities[local, class_index])
                            for class_index, column in enumerate(PROBABILITY_COLUMNS)
                        },
                    }
                )
    result = pd.DataFrame(rows).sort_values("recording_index").reset_index(drop=True)
    if not np.array_equal(result["recording_index"].to_numpy(np.int64), np.sort(indices)):
        raise RuntimeError("IDEA-075 prediction coverage mismatch")
    return result


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _fit_paths(run_root: Path, pipeline: str, base_seed: int, repeat: int, outer_fold: int) -> dict[str, Path]:
    root = run_root / "fits" / pipeline / f"seed_{base_seed}" / f"repeat_{repeat}" / f"fold_{outer_fold}"
    return {
        "root": root,
        "summary": root / "fit_summary.json",
        "pretest": root / "pre_test_summary.json",
        "test_marker": root / "outer_test_access_marker.json",
        "test_audit": root / "outer_test_audit.json",
        "test_temp": root / "outer_test_predictions.csv.tmp",
        "test": root / "outer_test_predictions.csv",
        "validation": root / "best_validation_predictions.csv",
        "checkpoint": root / "best_checkpoint.pt",
    }


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def runtime_environment_lock(device: torch.device) -> dict[str, Any]:
    return {
        "requested_and_resolved_device_type": device.type,
        "torch_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if device.type == "cuda" else None,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
    }


def expected_run_manifest(protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    data = protocol["data"]
    prepared = {}
    for stem in (
        "manifest",
        "frozen_embedding",
        "idea067_oof",
        "roles",
        "source_audio_inventory",
        "age_feature",
        "feature_summary",
    ):
        prepared[stem] = {"path": data[f"{stem}_path"], "sha256": data[f"{stem}_sha256"]}
    locked_settings = {
        "pipelines": list(PIPELINES),
        "classes": list(CLASS_NAMES),
        "base_seeds": list(BASE_SEEDS),
        "split_seeds": list(SPLIT_SEEDS),
        "model": protocol["model"],
        "acoustic_features": protocol["acoustic_features"],
        "fixed_training": protocol["fixed_training"],
        "gate": protocol["gate"],
    }
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "canonical_audio_content_sha256": data["canonical_audio_content_sha256"],
        "prepared_artifacts": prepared,
        "locked_settings_sha256": _canonical_json_sha256(locked_settings),
        "runtime_environment": runtime_environment_lock(device),
        "expected_neural_fits": 135,
    }


def ensure_run_manifest(
    protocol: dict[str, Any], run_root: Path, resume: bool, device: torch.device
) -> dict[str, Any]:
    path = run_root / "run_manifest.json"
    expected = expected_run_manifest(protocol, device)
    if path.exists():
        actual = read_json(path)
        if actual != expected:
            raise RuntimeError("IDEA-075 run manifest drift detected")
        return actual
    if resume and (run_root / "fits").exists():
        raise RuntimeError("IDEA-075 partial fits exist without a run manifest")
    write_json(path, expected)
    return expected


def validate_completed_fit(
    summary_path: Path,
    expected_identity: dict[str, Any] | None = None,
    store: DogStore | None = None,
    expected_test_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    summary = read_json(summary_path)
    if summary.get("status") != "complete" or int(summary.get("outer_test_access_count", -1)) != 1:
        raise RuntimeError(f"Incomplete IDEA-075 fit: {summary_path}")
    for stem in (
        "outer_test_predictions",
        "best_validation_predictions",
        "checkpoint",
        "pre_test_summary",
        "outer_test_access_marker",
        "outer_test_audit",
    ):
        path = REPO_ROOT / summary[f"{stem}_path"]
        if not path.is_file() or sha256(path) != summary[f"{stem}_sha256"]:
            raise RuntimeError(f"IDEA-075 completed-fit checksum mismatch: {path}")
    if expected_identity and any(summary.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError(f"IDEA-075 resumed-fit identity mismatch: {summary_path}")
    initialization = summary.get("initialization_audit", {})
    if initialization.get("trainable_parameters") != EXPECTED_PARAMETERS:
        raise RuntimeError("IDEA-075 completed-fit parameter initialization audit changed")
    if initialization.get("shared_state_equal") is not True or initialization.get("U1_C1_full_state_equal") is not True:
        raise RuntimeError("IDEA-075 completed-fit paired initialization audit failed")
    if any(float(value) != 0.0 for value in initialization.get("max_logit_difference_vs_A0", {}).values()):
        raise RuntimeError("IDEA-075 completed-fit initial logits differ")
    if float(summary.get("checkpoint_reload_max_probability_difference", float("inf"))) != 0.0:
        raise RuntimeError("IDEA-075 completed-fit checkpoint reload changed predictions")
    if summary.get("pipeline") == PIPELINES[2] and int(summary.get("model_audit_on_outer_test", {}).get("maximum_dimension_budget_violations", -1)) != 0:
        raise RuntimeError("IDEA-075 completed C1 fit violated its residual budget")
    if store is not None and expected_test_indices is not None:
        predictions = pd.read_csv(
            REPO_ROOT / summary["outer_test_predictions_path"],
            dtype={"recording_id": str, "dog_id": str},
        ).sort_values("recording_index")
        expected = np.sort(np.asarray(expected_test_indices, dtype=np.int64))
        if not np.array_equal(predictions["recording_index"].to_numpy(np.int64), expected):
            raise RuntimeError("IDEA-075 resumed-fit outer-test index coverage changed")
        if predictions["recording_id"].duplicated().any() or not np.array_equal(
            predictions["recording_id"].to_numpy(str), store.recording_ids[expected]
        ):
            raise RuntimeError("IDEA-075 resumed-fit outer-test ID coverage changed")
        if not np.array_equal(predictions["dog_id"].to_numpy(str), store.dog_ids[expected]) or not np.array_equal(
            predictions["true_label"].to_numpy(np.int64), store.labels[expected]
        ):
            raise RuntimeError("IDEA-075 resumed-fit outer-test dog/label coverage changed")
    return summary


def finalize_pretested_fit(
    paths: dict[str, Path],
    expected_identity: dict[str, Any],
    store: DogStore,
    test_indices: np.ndarray,
) -> dict[str, Any]:
    required = ("pretest", "test_marker", "test_audit", "test", "validation", "checkpoint")
    if not all(paths[name].is_file() for name in required):
        raise RuntimeError("IDEA-075 outer-test access started but durable final artifacts are incomplete; refusing to predict again")
    marker = read_json(paths["test_marker"])
    if marker.get("status") != "started" or marker.get("pre_test_summary_sha256") != sha256(paths["pretest"]):
        raise RuntimeError("IDEA-075 outer-test access marker is invalid")
    pretest = read_json(paths["pretest"])
    if pretest.get("status") != "ready_for_single_outer_test_access" or any(pretest.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError("IDEA-075 pre-test summary identity mismatch")
    for stem in ("best_validation_predictions", "checkpoint"):
        artifact = REPO_ROOT / pretest[f"{stem}_path"]
        if not artifact.is_file() or sha256(artifact) != pretest[f"{stem}_sha256"]:
            raise RuntimeError(f"IDEA-075 pre-test artifact checksum mismatch: {artifact}")
    audit = read_json(paths["test_audit"])
    if audit.get("outer_test_predictions_sha256") != sha256(paths["test"]):
        raise RuntimeError("IDEA-075 durable outer-test prediction hash mismatch")
    summary = {
        **pretest,
        "status": "complete",
        "outer_test_access_count": 1,
        "outer_test_predictions_path": _relative(paths["test"]),
        "outer_test_predictions_sha256": sha256(paths["test"]),
        "model_audit_on_outer_test": audit["model_audit_on_outer_test"],
        "pre_test_summary_path": _relative(paths["pretest"]),
        "pre_test_summary_sha256": sha256(paths["pretest"]),
        "outer_test_access_marker_path": _relative(paths["test_marker"]),
        "outer_test_access_marker_sha256": sha256(paths["test_marker"]),
        "outer_test_audit_path": _relative(paths["test_audit"]),
        "outer_test_audit_sha256": sha256(paths["test_audit"]),
    }
    write_json(paths["summary"], summary)
    return validate_completed_fit(paths["summary"], expected_identity, store, test_indices)


def fit_outer(
    pipeline: str,
    protocol: dict[str, Any],
    store: DogStore,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    outer_train_indices: np.ndarray,
    device: torch.device,
    full_seed: int,
    run_root: Path,
    base_seed: int,
    repeat: int,
    outer_fold: int,
    initialization: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    paths = _fit_paths(run_root, pipeline, base_seed, repeat, outer_fold)
    expected_identity = {
        "pipeline": pipeline,
        "base_seed": base_seed,
        "repeat": repeat,
        "outer_fold": outer_fold,
        "full_seed": full_seed,
    }
    if paths["summary"].is_file():
        if not resume:
            raise FileExistsError(paths["summary"])
        return validate_completed_fit(paths["summary"], expected_identity, store, test_indices)
    if paths["test_marker"].exists():
        if not resume:
            raise RuntimeError("IDEA-075 outer-test access marker exists without a completed fit")
        return finalize_pretested_fit(paths, expected_identity, store, test_indices)
    if paths["pretest"].exists() or paths["validation"].exists() or paths["checkpoint"].exists():
        raise RuntimeError("IDEA-075 partial pre-test artifacts exist without an access marker; refusing to overwrite")
    if paths["test"].exists() or paths["test_temp"].exists() or paths["test_audit"].exists():
        raise RuntimeError("IDEA-075 outer-test artifact exists without its access marker")
    paths["root"].mkdir(parents=True, exist_ok=True)
    fixed = protocol["fixed_training"]
    set_seed(full_seed)
    model = build_model(pipeline, store, fit_indices, float(fixed["dropout"])).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS[pipeline]:
        raise RuntimeError("IDEA-075 fit parameter count changed")
    training_seed = full_seed + int(fixed.get("post_build_seed_offset", 1_000_000))
    set_seed(training_seed)
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=float(fixed["learning_rate"]),
        eps=float(fixed["optimizer_epsilon"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loader = build_dog_loader(
        DogSetDataset(store, fit_indices),
        int(fixed["dog_batch_size"]),
        True,
        training_seed,
    )
    weights = torch.from_numpy(class_weights(store.labels[fit_indices])).to(device)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: pd.DataFrame | None = None
    history: list[dict[str, Any]] = []
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, int(fixed["maximum_epochs"]) + 1):
        train_loss, epoch_audit = train_one_epoch(
            model,
            loader,
            optimizer,
            scaler,
            weights,
            device,
            float(fixed["gradient_clip"]),
        )
        validation = predict_units(
            model,
            store,
            validation_indices,
            device,
            int(fixed["dog_batch_size"]),
            full_seed,
        )
        validation_loss = dog_equal_unit_cross_entropy(validation)
        validation_f1 = metric_bundle(validation)["macro_f1"]
        history.append(
            {
                "epoch": epoch,
                "train_weighted_unit_cross_entropy": train_loss,
                "train_audit": epoch_audit,
                "validation_dog_equal_unit_cross_entropy": validation_loss,
                "validation_unit_macro_f1": validation_f1,
            }
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            best_validation = validation.copy()
            stale = 0
        else:
            stale += 1
        print(
            f"{pipeline} seed={base_seed} repeat={repeat} fold={outer_fold} "
            f"epoch={epoch} train_CE={train_loss:.4f} val_dog_CE={validation_loss:.4f} "
            f"val_F1={validation_f1:.4f}",
            flush=True,
        )
        if stale >= int(fixed["early_stopping_patience"]):
            break
    if best_state is None or best_validation is None:
        raise RuntimeError("No IDEA-075 checkpoint selected")
    model.load_state_dict(best_state)
    reload_validation = predict_units(
        model,
        store,
        validation_indices,
        device,
        int(fixed["dog_batch_size"]),
        full_seed,
    )
    reload_difference = float(
        np.abs(
            reload_validation[list(PROBABILITY_COLUMNS)].to_numpy()
            - best_validation[list(PROBABILITY_COLUMNS)].to_numpy()
        ).max()
    )
    if reload_difference != 0.0:
        raise RuntimeError("IDEA-075 checkpoint reload changed validation predictions")
    outer_counts = np.bincount(store.labels[outer_train_indices], minlength=5).astype(np.float64)
    dummy_probabilities = outer_counts / outer_counts.sum()
    best_validation.to_csv(paths["validation"], index=False, lineterminator="\n")
    torch.save(best_state, paths["checkpoint"])
    pretest = {
        "status": "ready_for_single_outer_test_access",
        **expected_identity,
        "training_seed": training_seed,
        "fit_bark_units": int(len(fit_indices)),
        "fit_dogs": int(len(np.unique(store.dog_ids[fit_indices]))),
        "validation_bark_units": int(len(validation_indices)),
        "validation_dogs": int(len(np.unique(store.dog_ids[validation_indices]))),
        "test_bark_units": int(len(test_indices)),
        "test_dogs": int(len(np.unique(store.dog_ids[test_indices]))),
        "dummy_prior_source": "complete outer-train (fit plus validation)",
        "dummy_probabilities": dummy_probabilities.tolist(),
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "best_validation_dog_equal_unit_cross_entropy": best_loss,
        "checkpoint_reload_max_probability_difference": reload_difference,
        "initialization_audit": initialization,
        "history": history,
        "best_validation_predictions_path": _relative(paths["validation"]),
        "best_validation_predictions_sha256": sha256(paths["validation"]),
        "checkpoint_path": _relative(paths["checkpoint"]),
        "checkpoint_sha256": sha256(paths["checkpoint"]),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(paths["pretest"], pretest)
    write_json(
        paths["test_marker"],
        {
            "status": "started",
            **expected_identity,
            "pre_test_summary_sha256": sha256(paths["pretest"]),
        },
    )
    model.reset_perturbation_audit()
    # This is the only access to the outer-test features by the neural pipeline.
    test_predictions = predict_units(
        model,
        store,
        test_indices,
        device,
        int(fixed["dog_batch_size"]),
        full_seed,
    )
    model_test_audit = model.audit()
    if pipeline == PIPELINES[2] and int(model_test_audit["maximum_dimension_budget_violations"]) != 0:
        raise RuntimeError("IDEA-075 C1 residual budget violation")
    for class_index, name in enumerate(CLASS_NAMES):
        test_predictions[f"dummy_prob_{name}"] = float(dummy_probabilities[class_index])
    test_predictions.to_csv(paths["test_temp"], index=False, lineterminator="\n")
    write_json(
        paths["test_audit"],
        {
            "model_audit_on_outer_test": model_test_audit,
            "outer_test_predictions_sha256": sha256(paths["test_temp"]),
        },
    )
    os.replace(paths["test_temp"], paths["test"])
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return finalize_pretested_fit(paths, expected_identity, store, test_indices)


def metric_bundle(frame: pd.DataFrame, probability_columns: Sequence[str] = PROBABILITY_COLUMNS) -> dict[str, Any]:
    probabilities = frame[list(probability_columns)].to_numpy(float)
    labels = frame["true_label"].to_numpy(np.int64)
    predictions = probabilities.argmax(axis=1)
    ce = float(np.mean(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1.0e-7, 1.0))))
    targets = np.eye(len(CLASS_NAMES), dtype=float)[labels]
    recalls = recall_score(labels, predictions, labels=np.arange(5), average=None, zero_division=0)
    unit_correct = (predictions == labels).astype(float)
    dog_accuracy = pd.DataFrame({"dog_id": frame["dog_id"].astype(str), "correct": unit_correct}).groupby("dog_id")["correct"].mean().mean()
    grouped = frame.assign(_pred=[row for row in probabilities]).groupby(["dog_id", "true_label"], sort=False)["_pred"].apply(lambda values: np.mean(np.stack(values), axis=0)).reset_index()
    grouped_probabilities = np.stack(grouped["_pred"])
    grouped_labels = grouped["true_label"].to_numpy(np.int64)
    grouped_predictions = grouped_probabilities.argmax(axis=1)
    return {
        "macro_f1": float(f1_score(labels, predictions, labels=np.arange(5), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "cross_entropy": ce,
        "multiclass_brier": float(np.mean(np.sum((probabilities - targets) ** 2, axis=1))),
        "per_class_recall": {name: float(recalls[index]) for index, name in enumerate(CLASS_NAMES)},
        "mean_per_dog_bark_unit_accuracy": float(dog_accuracy),
        "dog_age_group_macro_f1": float(f1_score(grouped_labels, grouped_predictions, labels=np.arange(5), average="macro", zero_division=0)),
        "dog_age_group_balanced_accuracy": float(balanced_accuracy_score(grouped_labels, grouped_predictions)),
        "bark_units": int(len(frame)),
        "dogs": int(frame["dog_id"].nunique()),
        "dog_age_group_cells": int(len(grouped)),
    }


def _flat_metrics(prefix: str, bundle: dict[str, Any]) -> dict[str, Any]:
    result = {
        f"{prefix}_macro_f1": bundle["macro_f1"],
        f"{prefix}_balanced_accuracy": bundle["balanced_accuracy"],
        f"{prefix}_cross_entropy": bundle["cross_entropy"],
        f"{prefix}_multiclass_brier": bundle["multiclass_brier"],
        f"{prefix}_mean_per_dog_bark_unit_accuracy": bundle["mean_per_dog_bark_unit_accuracy"],
        f"{prefix}_dog_age_group_macro_f1": bundle["dog_age_group_macro_f1"],
        f"{prefix}_dog_age_group_balanced_accuracy": bundle["dog_age_group_balanced_accuracy"],
    }
    result.update({f"{prefix}_recall_{name}": value for name, value in bundle["per_class_recall"].items()})
    return result


def _sign_counts(values: Sequence[float], tolerance: float = 1.0e-12) -> dict[str, int]:
    array = np.asarray(values, dtype=float)
    return {
        "positive": int((array > tolerance).sum()),
        "tie": int((np.abs(array) <= tolerance).sum()),
        "negative": int((array < -tolerance).sum()),
    }


def aggregate(fits: list[dict[str, Any]], protocol: dict[str, Any], run_root: Path) -> dict[str, Any]:
    by_key = {
        (fit["pipeline"], int(fit["base_seed"]), int(fit["repeat"]), int(fit["outer_fold"])): fit
        for fit in fits
    }
    expected_keys = {
        (pipeline, base_seed, repeat, outer_fold)
        for pipeline in PIPELINES
        for base_seed in BASE_SEEDS
        for repeat in range(3)
        for outer_fold in range(5)
    }
    if len(fits) != 135 or set(by_key) != expected_keys:
        raise RuntimeError("IDEA-075 aggregate does not contain 135 unique fits")
    prediction_cache: dict[tuple[str, int, int, int], pd.DataFrame] = {}
    fold_rows: list[dict[str, Any]] = []
    for key, fit in by_key.items():
        frame = pd.read_csv(REPO_ROOT / fit["outer_test_predictions_path"], dtype={"recording_id": str, "dog_id": str})
        required_columns = {"recording_index", "recording_id", "dog_id", "true_label", *PROBABILITY_COLUMNS}
        if not required_columns.issubset(frame.columns) or frame["recording_id"].duplicated().any():
            raise RuntimeError("IDEA-075 outer-test prediction schema/IDs changed")
        probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(float)
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-5, rtol=0):
            raise RuntimeError("IDEA-075 outer-test probabilities are invalid")
        if key[0] == PIPELINES[2]:
            dummy_columns = [f"dummy_prob_{name}" for name in CLASS_NAMES]
            dummy = frame[dummy_columns].to_numpy(float)
            if not np.isfinite(dummy).all() or not np.allclose(dummy.sum(axis=1), 1.0, atol=1.0e-12, rtol=0):
                raise RuntimeError("IDEA-075 dummy probabilities are invalid")
        prediction_cache[key] = frame
        bundle = metric_bundle(frame)
        fold_rows.append(
            {
                "pipeline": key[0], "base_seed": key[1], "repeat": key[2], "outer_fold": key[3],
                **_flat_metrics("unit", bundle),
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    pooled_frames: dict[tuple[str, int, int], pd.DataFrame] = {}
    seed_repeat_rows: list[dict[str, Any]] = []
    for pipeline in PIPELINES:
        for base_seed in BASE_SEEDS:
            for repeat in range(3):
                parts = [prediction_cache[(pipeline, base_seed, repeat, fold)] for fold in range(5)]
                pooled = pd.concat(parts, ignore_index=True).sort_values("recording_index").reset_index(drop=True)
                if len(pooled) != 2290 or pooled["recording_id"].duplicated().any():
                    raise RuntimeError("IDEA-075 pooled OOF coverage mismatch")
                pooled_frames[(pipeline, base_seed, repeat)] = pooled
                bundle = metric_bundle(pooled)
                row = {"pipeline": pipeline, "base_seed": base_seed, "repeat": repeat, **_flat_metrics("unit", bundle)}
                if pipeline == PIPELINES[2]:
                    dummy_columns = [f"dummy_prob_{name}" for name in CLASS_NAMES]
                    row.update(_flat_metrics("dummy", metric_bundle(pooled, dummy_columns)))
                seed_repeat_rows.append(row)
    seed_repeat = pd.DataFrame(seed_repeat_rows)
    comparisons = {
        "C1_minus_A0": (PIPELINES[2], PIPELINES[0]),
        "U1_minus_A0": (PIPELINES[1], PIPELINES[0]),
        "C1_minus_U1": (PIPELINES[2], PIPELINES[1]),
    }
    delta_rows: list[dict[str, Any]] = []
    dog_rows: list[dict[str, Any]] = []
    for name, (candidate, reference) in comparisons.items():
        for base_seed in BASE_SEEDS:
            for repeat in range(3):
                left_row = seed_repeat[(seed_repeat.pipeline == candidate) & (seed_repeat.base_seed == base_seed) & (seed_repeat.repeat == repeat)].iloc[0]
                right_row = seed_repeat[(seed_repeat.pipeline == reference) & (seed_repeat.base_seed == base_seed) & (seed_repeat.repeat == repeat)].iloc[0]
                delta_rows.append(
                    {
                        "comparison": name,
                        "base_seed": base_seed,
                        "repeat": repeat,
                        "macro_f1_delta": float(left_row.unit_macro_f1 - right_row.unit_macro_f1),
                        "balanced_accuracy_delta": float(left_row.unit_balanced_accuracy - right_row.unit_balanced_accuracy),
                        "cross_entropy_delta": float(left_row.unit_cross_entropy - right_row.unit_cross_entropy),
                        "multiclass_brier_delta": float(left_row.unit_multiclass_brier - right_row.unit_multiclass_brier),
                        "senior_recall_delta": float(left_row.unit_recall_senior - right_row.unit_recall_senior),
                    }
                )
                left = pooled_frames[(candidate, base_seed, repeat)].set_index("recording_id")
                right = pooled_frames[(reference, base_seed, repeat)].set_index("recording_id")
                if not left.index.equals(right.index):
                    right = right.loc[left.index]
                for dog_id, ids in left.groupby("dog_id").groups.items():
                    li = left.loc[ids]
                    ri = right.loc[ids]
                    labels = li["true_label"].to_numpy(np.int64)
                    left_prob = li[list(PROBABILITY_COLUMNS)].to_numpy(float)
                    right_prob = ri[list(PROBABILITY_COLUMNS)].to_numpy(float)
                    dog_rows.append(
                        {
                            "comparison": name, "base_seed": base_seed, "repeat": repeat, "dog_id": dog_id,
                            "cross_entropy_delta_candidate_minus_reference": float(np.mean(-np.log(np.clip(left_prob[np.arange(len(labels)), labels], 1e-7, 1))) - np.mean(-np.log(np.clip(right_prob[np.arange(len(labels)), labels], 1e-7, 1)))),
                            "accuracy_delta_candidate_minus_reference": float(np.mean(left_prob.argmax(1) == labels) - np.mean(right_prob.argmax(1) == labels)),
                        }
                    )
    deltas = pd.DataFrame(delta_rows)
    dog_deltas = pd.DataFrame(dog_rows)
    split_rows: list[dict[str, Any]] = []
    for name, (candidate, reference) in comparisons.items():
        for repeat in range(3):
            for fold in range(5):
                values = []
                for base_seed in BASE_SEEDS:
                    left = fold_metrics[(fold_metrics.pipeline == candidate) & (fold_metrics.base_seed == base_seed) & (fold_metrics.repeat == repeat) & (fold_metrics.outer_fold == fold)].iloc[0]
                    right = fold_metrics[(fold_metrics.pipeline == reference) & (fold_metrics.base_seed == base_seed) & (fold_metrics.repeat == repeat) & (fold_metrics.outer_fold == fold)].iloc[0]
                    values.append(float(left.unit_macro_f1 - right.unit_macro_f1))
                split_rows.append({"comparison": name, "repeat": repeat, "outer_fold": fold, "macro_f1_delta_mean_across_base_seeds": float(np.mean(values))})
    split_cells = pd.DataFrame(split_rows)
    comparison_summary: dict[str, Any] = {}
    for name in comparisons:
        current = deltas[deltas.comparison == name]
        base_means = current.groupby("base_seed")["macro_f1_delta"].mean()
        cells = split_cells[split_cells.comparison == name]["macro_f1_delta_mean_across_base_seeds"]
        dog_means = dog_deltas[dog_deltas.comparison == name].groupby("dog_id", sort=True)[
            ["cross_entropy_delta_candidate_minus_reference", "accuracy_delta_candidate_minus_reference"]
        ].mean()
        ce_values = dog_means["cross_entropy_delta_candidate_minus_reference"]
        accuracy_values = dog_means["accuracy_delta_candidate_minus_reference"]
        comparison_summary[name] = {
            "mean_macro_f1_delta": float(current.macro_f1_delta.mean()),
            "base_seed_macro_f1_deltas": {str(key): float(value) for key, value in base_means.items()},
            "seed_repeat_macro_f1_signs": _sign_counts(current.macro_f1_delta),
            "split_cell_macro_f1_signs": _sign_counts(cells),
            "worst_split_cell_macro_f1_delta": float(cells.min()),
            "mean_balanced_accuracy_delta": float(current.balanced_accuracy_delta.mean()),
            "mean_cross_entropy_delta": float(current.cross_entropy_delta.mean()),
            "mean_multiclass_brier_delta": float(current.multiclass_brier_delta.mean()),
            "per_dog_cross_entropy_delta_signs_candidate_minus_reference": _sign_counts(ce_values),
            "per_dog_accuracy_delta_signs_candidate_minus_reference": _sign_counts(accuracy_values),
            "per_dog_count": int(len(dog_means)),
        }
    c1_a0 = deltas[deltas.comparison == "C1_minus_A0"]
    c1_u1 = deltas[deltas.comparison == "C1_minus_U1"]
    c1_a0_base = c1_a0.groupby("base_seed")["macro_f1_delta"].mean()
    c1_u1_base = c1_u1.groupby("base_seed")["macro_f1_delta"].mean()
    c1_a0_cells = split_cells[split_cells.comparison == "C1_minus_A0"]["macro_f1_delta_mean_across_base_seeds"]
    c1_metrics = seed_repeat[seed_repeat.pipeline == PIPELINES[2]]
    a0_metrics = seed_repeat[seed_repeat.pipeline == PIPELINES[0]]
    u1_metrics = seed_repeat[seed_repeat.pipeline == PIPELINES[1]]
    senior_by_seed = {
        base_seed: float(c1_a0[c1_a0.base_seed == base_seed].senior_recall_delta.mean())
        for base_seed in BASE_SEEDS
    }
    tier1_checks = {
        "mean_macro_f1_gain_at_least_0_005": float(c1_a0.macro_f1_delta.mean()) >= 0.005,
        "all_base_seed_means_strictly_positive": bool((c1_a0_base > 0).all()),
        "at_least_6_of_9_seed_repeats_positive": int((c1_a0.macro_f1_delta > 0).sum()) >= 6,
        "at_least_10_of_15_split_cells_nonnegative": int((c1_a0_cells >= 0).sum()) >= 10,
        "worst_split_cell_at_least_minus_0_03": float(c1_a0_cells.min()) >= -0.03,
        "mean_CE_not_worse": float(c1_metrics.unit_cross_entropy.mean()) <= float(a0_metrics.unit_cross_entropy.mean()),
        "mean_Brier_not_worse": float(c1_metrics.unit_multiclass_brier.mean()) <= float(a0_metrics.unit_multiclass_brier.mean()),
        "mean_balanced_accuracy_delta_nonnegative": float(c1_a0.balanced_accuracy_delta.mean()) >= 0,
        "each_base_seed_senior_recall_delta_at_least_minus_0_02": all(value >= -0.02 for value in senior_by_seed.values()),
    }
    tier2_checks = {
        "mean_C1_minus_U1_macro_f1_strictly_positive": float(c1_u1.macro_f1_delta.mean()) > 0,
        "at_least_2_of_3_base_seed_means_strictly_positive": int((c1_u1_base > 0).sum()) >= 2,
        "at_least_5_of_9_seed_repeats_strictly_positive": int((c1_u1.macro_f1_delta > 0).sum()) >= 5,
        "mean_CE_not_worse_than_U1": float(c1_metrics.unit_cross_entropy.mean()) <= float(u1_metrics.unit_cross_entropy.mean()),
        "mean_Brier_not_worse_than_U1": float(c1_metrics.unit_multiclass_brier.mean()) <= float(u1_metrics.unit_multiclass_brier.mean()),
    }
    dummy_deltas = c1_metrics.unit_macro_f1.to_numpy(float) - c1_metrics.dummy_macro_f1.to_numpy(float)
    tier3_checks = {"C1_above_outer_train_prior_dummy_in_all_9_seed_repeats": bool((dummy_deltas > 0).all())}
    for frame, name in (
        (fold_metrics, "fold_metrics.csv"),
        (seed_repeat, "seed_repeat_metrics.csv"),
        (deltas, "seed_repeat_deltas.csv"),
        (split_cells, "split_cell_deltas.csv"),
        (dog_deltas, "per_dog_pair_deltas.csv"),
    ):
        frame.to_csv(run_root / name, index=False, lineterminator="\n")
    c1_audits = [fit["model_audit_on_outer_test"] for fit in fits if fit["pipeline"] == PIPELINES[2]]
    return {
        "status": "complete",
        "analysis_unit": "9 equal-weight base_seed x split_repeat pooled OOF units",
        "pipeline_mean_metrics": {
            pipeline: {
                metric: float(seed_repeat[seed_repeat.pipeline == pipeline][metric].mean())
                for metric in ("unit_macro_f1", "unit_balanced_accuracy", "unit_cross_entropy", "unit_multiclass_brier", "unit_recall_senior", "unit_mean_per_dog_bark_unit_accuracy", "unit_dog_age_group_macro_f1")
            }
            for pipeline in PIPELINES
        },
        "comparisons": comparison_summary,
        "gate": {
            "tier1_C1_package_vs_A0": {"passed": all(tier1_checks.values()), "checks": tier1_checks, "senior_recall_delta_by_base_seed": senior_by_seed},
            "tier2_C1_bound_vs_U1": {"interpretable": all(tier1_checks.values()), "passed": all(tier1_checks.values()) and all(tier2_checks.values()), "checks": tier2_checks},
            "tier3_external_task_floor": {"passed": all(tier3_checks.values()), "checks": tier3_checks, "C1_minus_dummy_macro_f1": dummy_deltas.tolist()},
        },
        "C1_mechanism_audit": {
            "fits": len(c1_audits),
            "maximum_dimension_budget_violations": max(int(audit["maximum_dimension_budget_violations"]) for audit in c1_audits),
            "mean_residual_to_hidden_norm_ratio": float(np.mean([audit.get("residual_to_hidden_norm_ratio", {}).get("mean", np.nan) for audit in c1_audits])),
        },
        "artifacts": {
            name: {"path": _relative(run_root / filename), "sha256": sha256(run_root / filename)}
            for name, filename in (
                ("fold_metrics", "fold_metrics.csv"),
                ("seed_repeat_metrics", "seed_repeat_metrics.csv"),
                ("seed_repeat_deltas", "seed_repeat_deltas.csv"),
                ("split_cell_deltas", "split_cell_deltas.csv"),
                ("per_dog_pair_deltas", "per_dog_pair_deltas.csv"),
            )
        },
    }


def run_benchmark(args: argparse.Namespace, protocol: dict[str, Any], run_root: Path) -> dict[str, Any]:
    summary_path = run_root / "evaluation_summary.json"
    previous_summary_bytes: bytes | None = None
    if summary_path.exists():
        if not args.resume:
            raise FileExistsError(summary_path)
        previous_summary_bytes = summary_path.read_bytes()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    store, roles = load_prepared_store(protocol)
    run_manifest = ensure_run_manifest(protocol, run_root, args.resume, device)
    previous_fit_summary_hashes: dict[str, str] = {}
    if previous_summary_bytes is not None:
        completed = json.loads(previous_summary_bytes.decode("utf-8"))
        if completed.get("status") != "complete" or completed.get("protocol_sha256") != sha256(PROTOCOL_PATH) or completed.get("runner_sha256") != sha256(Path(__file__).resolve()):
            raise RuntimeError("IDEA-075 existing evaluation summary identity mismatch")
        locked_artifacts = list(completed.get("artifacts", {}).values()) + [
            {"path": completed["pairing_audit_path"], "sha256": completed["pairing_audit_sha256"]},
            {"path": completed["run_manifest_path"], "sha256": completed["run_manifest_sha256"]},
        ]
        for artifact in locked_artifacts:
            path = REPO_ROOT / artifact["path"]
            if not path.is_file() or sha256(path) != artifact["sha256"]:
                raise RuntimeError(f"IDEA-075 existing evaluation artifact mismatch: {path}")
        fit_index_artifact = completed["artifacts"]["fit_index"]
        fit_index = pd.read_csv(REPO_ROOT / fit_index_artifact["path"])
        previous_fit_summary_hashes = dict(zip(fit_index["fit_summary_path"], fit_index["fit_summary_sha256"]))
        if len(previous_fit_summary_hashes) != 135:
            raise RuntimeError("IDEA-075 existing fit index does not contain 135 fits")
    fits: list[dict[str, Any]] = []
    pairing_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for base_seed in BASE_SEEDS:
        for repeat in range(3):
            for outer_fold in range(5):
                role = roles[(roles.repeat == repeat) & (roles.outer_fold == outer_fold)]
                fit_indices = np.flatnonzero(role.role.to_numpy(str) == "fit")
                validation_indices = np.flatnonzero(role.role.to_numpy(str) == "validation")
                test_indices = np.flatnonzero(role.role.to_numpy(str) == "test")
                outer_train_indices = np.concatenate([fit_indices, validation_indices])
                full_seed = base_seed + 10_000 * repeat + 100 * outer_fold
                initialization = initialization_audit(
                    store,
                    fit_indices,
                    validation_indices[: min(32, len(validation_indices))],
                    float(protocol["fixed_training"]["dropout"]),
                    full_seed,
                )
                cell_fits = []
                for pipeline in PIPELINES:
                    pending_summary_path = _fit_paths(run_root, pipeline, base_seed, repeat, outer_fold)["summary"]
                    pending_relative = _relative(pending_summary_path)
                    if previous_fit_summary_hashes and (
                        pending_relative not in previous_fit_summary_hashes
                        or not pending_summary_path.is_file()
                        or sha256(pending_summary_path) != previous_fit_summary_hashes[pending_relative]
                    ):
                        raise RuntimeError(f"IDEA-075 existing fit-summary hash mismatch: {pending_summary_path}")
                    result = fit_outer(
                        pipeline, protocol, store, fit_indices, validation_indices, test_indices,
                        outer_train_indices, device, full_seed, run_root, base_seed, repeat,
                        outer_fold, initialization, args.resume,
                    )
                    fits.append(result)
                    cell_fits.append(result)
                common_epochs = min(len(fit["history"]) for fit in cell_fits)
                dog_order_equal = all(
                    len({fit["history"][epoch]["train_audit"]["dog_order_sha256"] for fit in cell_fits}) == 1
                    for epoch in range(common_epochs)
                )
                coverage_equal = all(
                    len({fit["history"][epoch]["train_audit"]["recording_coverage_sha256"] for fit in cell_fits}) == 1
                    for epoch in range(common_epochs)
                )
                if not dog_order_equal or not coverage_equal:
                    raise RuntimeError("IDEA-075 paired training batches diverged")
                pairing_rows.append(
                    {
                        "base_seed": base_seed, "repeat": repeat, "outer_fold": outer_fold,
                        "full_seed": full_seed, "common_epochs": common_epochs,
                        "dog_order_hashes_equal": dog_order_equal,
                        "recording_coverage_hashes_equal": coverage_equal,
                    }
                )
    pairing_path = run_root / "pairing_audit.csv"
    pd.DataFrame(pairing_rows).to_csv(pairing_path, index=False, lineterminator="\n")
    fit_index_path = run_root / "fit_index.csv"
    fit_index_rows = []
    for fit in fits:
        fit_summary_path = _fit_paths(
            run_root,
            fit["pipeline"],
            int(fit["base_seed"]),
            int(fit["repeat"]),
            int(fit["outer_fold"]),
        )["summary"]
        fit_index_rows.append(
            {
                "pipeline": fit["pipeline"],
                "base_seed": fit["base_seed"],
                "repeat": fit["repeat"],
                "outer_fold": fit["outer_fold"],
                "fit_summary_path": _relative(fit_summary_path),
                "fit_summary_sha256": sha256(fit_summary_path),
            }
        )
    pd.DataFrame(fit_index_rows).to_csv(fit_index_path, index=False, lineterminator="\n")
    summary = aggregate(fits, protocol, run_root)
    summary["artifacts"]["fit_index"] = {"path": _relative(fit_index_path), "sha256": sha256(fit_index_path)}
    summary.update(
        {
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "fits_completed": len(fits),
            "outer_test_predictions_per_fit": 1,
            "outer_test_accessed_during_selection": False,
            "outer_test_access_count": len(fits),
            "run_manifest_path": _relative(run_root / "run_manifest.json"),
            "run_manifest_sha256": sha256(run_root / "run_manifest.json"),
            "pairing_audit_path": _relative(pairing_path),
            "pairing_audit_sha256": sha256(pairing_path),
            "elapsed_seconds": float(time.perf_counter() - started),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": str(device),
                "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            },
        }
    )
    if run_manifest["expected_neural_fits"] != len(fits):
        raise RuntimeError("IDEA-075 neural fit budget mismatch")
    if previous_summary_bytes is not None:
        previous = json.loads(previous_summary_bytes.decode("utf-8"))
        summary["elapsed_seconds"] = previous["elapsed_seconds"]
        proposed = (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        if proposed != previous_summary_bytes:
            raise RuntimeError("IDEA-075 resumed aggregate differs from locked evaluation summary")
    else:
        write_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol, require_prepared=args.stage == "run")
    configure_determinism()
    run_root = resolve_run_root(args.output_subdir)
    if args.stage == "extract-features":
        result = prepare_features_and_roles(protocol, run_root, args.resume)
    else:
        result = run_benchmark(args, protocol, run_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
