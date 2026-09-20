"""Extract age-sensitive acoustics and run the IDEA-068 inner-only screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_idea051_cat_set as idea051  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea068_age_sensitive_ast_v1.json"
)
PIPELINES = (
    "A0_ast_only",
    "A1_age_residual",
    "A2_confidence_gated_age_residual",
)
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
FEATURE_NAMES = (
    "log_f0_q10",
    "log_f0_median",
    "log_f0_q90",
    "log_f0_iqr",
    "log_f0_std",
    "log_f0_mad",
    "log_f0_slope",
    "log_f0_abs_delta_median",
    "period_variation_proxy",
    "voiced_fraction",
    "voiced_probability_mean",
    "voiced_probability_std",
    "periodicity_median",
    "hnr_db_median",
    "amplitude_variation_proxy",
    "log_rms_median",
    "log_rms_iqr",
    "spectral_tilt_median",
    "spectral_tilt_iqr",
    "spectral_flatness_median",
)
VOICED_FRACTION_INDEX = FEATURE_NAMES.index("voiced_fraction")
VOICED_PROBABILITY_INDEX = FEATURE_NAMES.index("voiced_probability_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("extract", "screen"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output-subdir", default="meowagenet_idea068_age_sensitive_ast_v1"
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_determinism() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea068-age-sensitive-ast-v1":
        raise RuntimeError("Unexpected IDEA-068 protocol")
    if protocol.get("status") != "locked_before_feature_extraction_and_inner_screen":
        raise RuntimeError("IDEA-068 protocol is not locked")
    if tuple(protocol["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-068 pipeline matrix changed")
    if tuple(protocol["acoustic_features"]["feature_names"]) != FEATURE_NAMES:
        raise RuntimeError("IDEA-068 acoustic feature set changed")
    screen = protocol["screen"]
    expected_fits = (
        len(PIPELINES)
        * len(screen["repeats"])
        * len(screen["outer_folds_used_as_inner_role_definitions"])
    )
    if expected_fits != int(screen["total_fits"]):
        raise RuntimeError("IDEA-068 fit budget is inconsistent")
    dependencies = protocol["dependencies"]
    checks = {
        REPO_ROOT / dependencies["idea_path"]: dependencies["idea_sha256"],
        REPO_ROOT / protocol["data"]["manifest_path"]: protocol["data"][
            "manifest_sha256"
        ],
        REPO_ROOT / protocol["data"]["roles_path"]: protocol["data"][
            "roles_sha256"
        ],
        REPO_ROOT / protocol["data"]["frozen_embedding_path"]: protocol["data"][
            "frozen_embedding_sha256"
        ],
        REPO_ROOT / dependencies["idea051_runner_path"]: dependencies[
            "idea051_runner_sha256"
        ],
        Path(__file__).resolve(): dependencies["runner_sha256"],
    }
    for path, expected_sha in checks.items():
        if not path.is_file() or sha256(path) != expected_sha:
            raise RuntimeError(f"IDEA-068 dependency checksum mismatch: {path}")


def resolve_run_root(output_subdir: str) -> Path:
    run_root = (idea051.RUNS_ROOT / output_subdir).resolve()
    if idea051.RUNS_ROOT.resolve() not in run_root.parents:
        raise ValueError("--output-subdir must stay below runs")
    return run_root


def load_audio(path: Path, target_rate: int) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if sample_rate != target_rate:
        divisor = math.gcd(sample_rate, target_rate)
        waveform = resample_poly(
            waveform, target_rate // divisor, sample_rate // divisor
        ).astype(np.float32)
    if waveform.size == 0:
        raise RuntimeError(f"Empty audio: {path}")
    return waveform.astype(np.float32, copy=False)


def finite_quantile(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan")
    return float(np.quantile(finite, q))


def finite_median(values: np.ndarray) -> float:
    return finite_quantile(values, 0.5)


def consecutive_pair_mask(valid: np.ndarray) -> np.ndarray:
    valid = np.asarray(valid, dtype=bool)
    if len(valid) < 2:
        return np.zeros(0, dtype=bool)
    return valid[:-1] & valid[1:]


def frame_periodicity(
    waveform: np.ndarray,
    f0: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    padded = np.pad(waveform, (frame_length // 2, frame_length // 2))
    result = np.full(len(f0), np.nan, dtype=np.float64)
    window = np.hanning(frame_length).astype(np.float64)
    for frame_index, frequency in enumerate(f0):
        if not np.isfinite(frequency) or frequency <= 0:
            continue
        lag = int(round(sample_rate / float(frequency)))
        if lag < 1 or lag >= frame_length // 2:
            continue
        start = frame_index * hop_length
        frame = padded[start : start + frame_length].astype(np.float64, copy=False)
        if len(frame) != frame_length:
            continue
        frame = (frame - frame.mean()) * window
        left = frame[:-lag]
        right = frame[lag:]
        denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
        if denominator <= 1.0e-12:
            continue
        result[frame_index] = float(np.clip(np.dot(left, right) / denominator, 0, 0.999))
    return result


def extract_call_features(
    waveform: np.ndarray, sample_rate: int, settings: dict[str, Any]
) -> np.ndarray:
    frame_length = int(settings["frame_length"])
    hop_length = int(settings["hop_length"])
    f0, _, voiced_probability = librosa.pyin(
        waveform,
        fmin=float(settings["fmin_hz"]),
        fmax=float(settings["fmax_hz"]),
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )
    f0 = np.asarray(f0, dtype=np.float64)
    voiced_probability = np.asarray(voiced_probability, dtype=np.float64)
    magnitude = np.abs(
        librosa.stft(
            waveform,
            n_fft=frame_length,
            hop_length=hop_length,
            win_length=frame_length,
            center=True,
        )
    ).astype(np.float64)
    rms = librosa.feature.rms(S=magnitude, frame_length=frame_length)[0]
    flatness = librosa.feature.spectral_flatness(S=magnitude + 1.0e-10)[0]
    frame_count = min(len(f0), magnitude.shape[1], len(rms), len(flatness))
    f0 = f0[:frame_count]
    voiced_probability = voiced_probability[:frame_count]
    magnitude = magnitude[:, :frame_count]
    rms = np.asarray(rms[:frame_count], dtype=np.float64)
    flatness = np.asarray(flatness[:frame_count], dtype=np.float64)
    voiced = np.isfinite(f0) & (f0 > 0)
    log_f0 = np.full(frame_count, np.nan, dtype=np.float64)
    log_f0[voiced] = np.log(f0[voiced])

    q10 = finite_quantile(log_f0, 0.10)
    q25 = finite_quantile(log_f0, 0.25)
    q50 = finite_quantile(log_f0, 0.50)
    q75 = finite_quantile(log_f0, 0.75)
    q90 = finite_quantile(log_f0, 0.90)
    finite_log_f0 = log_f0[voiced]
    log_std = float(np.std(finite_log_f0)) if len(finite_log_f0) else float("nan")
    log_mad = (
        float(np.median(np.abs(finite_log_f0 - np.median(finite_log_f0))))
        if len(finite_log_f0)
        else float("nan")
    )
    if voiced.sum() >= 2:
        time_axis = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)[voiced]
        log_slope = float(np.polyfit(time_axis, finite_log_f0, 1)[0])
    else:
        log_slope = float("nan")
    pair_mask = consecutive_pair_mask(voiced)
    log_delta = np.abs(np.diff(log_f0))[pair_mask]
    periods = np.full(frame_count, np.nan, dtype=np.float64)
    periods[voiced] = 1.0 / f0[voiced]
    period_delta = np.abs(np.diff(periods))[pair_mask]
    period_median = finite_median(periods)
    period_variation = (
        finite_median(period_delta) / period_median
        if np.isfinite(period_median) and period_median > 0
        else float("nan")
    )

    periodicity = frame_periodicity(
        waveform, f0, sample_rate, frame_length, hop_length
    )[:frame_count]
    valid_periodicity = periodicity[np.isfinite(periodicity)]
    hnr = np.full_like(valid_periodicity, np.nan)
    if len(valid_periodicity):
        clipped = np.clip(valid_periodicity, 1.0e-6, 0.999)
        hnr = 10.0 * np.log10(clipped / (1.0 - clipped))

    log_rms = np.full(frame_count, np.nan, dtype=np.float64)
    log_rms[voiced] = np.log(rms[voiced] + 1.0e-10)
    rms_pair_delta = np.abs(np.diff(rms))[pair_mask]
    rms_median = finite_median(rms[voiced])
    amplitude_variation = (
        finite_median(rms_pair_delta) / rms_median
        if np.isfinite(rms_median) and rms_median > 0
        else float("nan")
    )

    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=frame_length)
    frequency_mask = (frequencies >= 200.0) & (frequencies <= 4000.0)
    log_frequency = np.log(frequencies[frequency_mask])
    centered_frequency = log_frequency - log_frequency.mean()
    denominator = float(np.dot(centered_frequency, centered_frequency))
    log_magnitude = np.log(magnitude[frequency_mask] + 1.0e-10)
    centered_magnitude = log_magnitude - log_magnitude.mean(axis=0, keepdims=True)
    spectral_tilt = (centered_frequency[:, None] * centered_magnitude).sum(axis=0) / denominator
    voiced_tilt = spectral_tilt[voiced]
    voiced_flatness = flatness[voiced]

    finite_voicing = voiced_probability[np.isfinite(voiced_probability)]
    features = np.asarray(
        [
            q10,
            q50,
            q90,
            q75 - q25 if np.isfinite(q75) and np.isfinite(q25) else np.nan,
            log_std,
            log_mad,
            log_slope,
            finite_median(log_delta),
            period_variation,
            float(voiced.mean()) if frame_count else np.nan,
            float(np.mean(finite_voicing)) if len(finite_voicing) else np.nan,
            float(np.std(finite_voicing)) if len(finite_voicing) else np.nan,
            finite_median(valid_periodicity),
            finite_median(hnr),
            amplitude_variation,
            finite_median(log_rms),
            (
                finite_quantile(log_rms, 0.75) - finite_quantile(log_rms, 0.25)
                if np.isfinite(finite_quantile(log_rms, 0.75))
                and np.isfinite(finite_quantile(log_rms, 0.25))
                else np.nan
            ),
            finite_median(voiced_tilt),
            (
                finite_quantile(voiced_tilt, 0.75)
                - finite_quantile(voiced_tilt, 0.25)
                if np.isfinite(finite_quantile(voiced_tilt, 0.75))
                and np.isfinite(finite_quantile(voiced_tilt, 0.25))
                else np.nan
            ),
            finite_median(voiced_flatness),
        ],
        dtype=np.float32,
    )
    if features.shape != (len(FEATURE_NAMES),):
        raise RuntimeError("IDEA-068 feature shape changed")
    return features


def extract_features(protocol: dict[str, Any], run_root: Path) -> dict[str, Any]:
    output_dir = run_root / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "age_sensitive_acoustic_features.npz"
    summary_path = output_dir / "extraction_summary.json"
    if feature_path.exists() or summary_path.exists():
        raise FileExistsError("IDEA-068 feature output already exists")
    frozen = np.load(REPO_ROOT / protocol["data"]["frozen_embedding_path"])
    call_ids = frozen["call_ids"].astype(str)
    manifest = pd.read_csv(REPO_ROOT / protocol["data"]["manifest_path"])
    manifest = manifest[
        manifest["analysis_include"].astype(str).str.lower() == "true"
    ].copy()
    by_name = manifest.set_index("filename", verify_integrity=True)
    if set(call_ids) != set(by_name.index.astype(str)):
        raise RuntimeError("IDEA-068 manifest and frozen call IDs differ")
    settings = protocol["acoustic_features"]
    sample_rate = int(settings["sample_rate_hz"])
    rows = []
    source_hashes_verified = 0
    started = time.perf_counter()
    for index, call_id in enumerate(call_ids):
        row = by_name.loc[call_id]
        audio_path = REPO_ROOT / str(row["local_relpath"])
        if not audio_path.is_file() or sha256(audio_path) != str(row["sha256"]):
            raise RuntimeError(f"IDEA-068 audio checksum mismatch: {audio_path}")
        source_hashes_verified += 1
        waveform = load_audio(audio_path, sample_rate)
        rows.append(extract_call_features(waveform, sample_rate, settings))
        if (index + 1) % 50 == 0 or index + 1 == len(call_ids):
            print(f"IDEA-068 extracted {index + 1}/{len(call_ids)} calls", flush=True)
    features = np.stack(rows).astype(np.float32)
    np.savez_compressed(
        feature_path,
        features=features,
        feature_names=np.asarray(FEATURE_NAMES, dtype="U"),
        call_ids=np.asarray(call_ids, dtype="U"),
    )
    summary = {
        "status": "complete",
        "label_information_used": False,
        "calls": int(len(call_ids)),
        "features": int(features.shape[1]),
        "feature_names": list(FEATURE_NAMES),
        "source_hashes_verified": source_hashes_verified,
        "missing_values_by_feature": {
            name: int(np.isnan(features[:, index]).sum())
            for index, name in enumerate(FEATURE_NAMES)
        },
        "fully_finite_calls": int(np.isfinite(features).all(axis=1).sum()),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "feature_path": feature_path.relative_to(REPO_ROOT).as_posix(),
        "feature_sha256": sha256(feature_path),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(summary_path, summary)
    return summary


def load_age_features(
    protocol: dict[str, Any], run_root: Path, call_ids: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    summary_path = run_root / "features" / "extraction_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = read_json(summary_path)
    if summary.get("status") != "complete":
        raise RuntimeError("IDEA-068 feature extraction is incomplete")
    if summary.get("protocol_sha256") != sha256(PROTOCOL_PATH):
        raise RuntimeError("IDEA-068 features were extracted under another protocol")
    feature_path = REPO_ROOT / summary["feature_path"]
    if not feature_path.is_file() or sha256(feature_path) != summary["feature_sha256"]:
        raise RuntimeError("IDEA-068 acoustic feature checksum mismatch")
    loaded = np.load(feature_path)
    if tuple(loaded["feature_names"].astype(str)) != FEATURE_NAMES:
        raise RuntimeError("IDEA-068 stored feature names changed")
    if not np.array_equal(loaded["call_ids"].astype(str), call_ids.astype(str)):
        raise RuntimeError("IDEA-068 feature call order differs from AST")
    features = loaded["features"].astype(np.float32)
    if features.shape != (len(call_ids), len(FEATURE_NAMES)):
        raise RuntimeError("IDEA-068 stored feature matrix has the wrong shape")
    return features, summary


class AgeResidualClassifier(torch.nn.Module):
    def __init__(
        self,
        pipeline: str,
        ast_mean: np.ndarray,
        ast_scale: np.ndarray,
        age_train: np.ndarray,
        dropout: float,
        age_hidden_units: int,
    ) -> None:
        super().__init__()
        if pipeline not in PIPELINES:
            raise ValueError(pipeline)
        self.pipeline = pipeline
        safe_ast_scale = np.where(ast_scale > 1.0e-12, ast_scale, 1.0).astype(
            np.float32
        )
        self.register_buffer("ast_mean", torch.from_numpy(ast_mean.astype(np.float32)))
        self.register_buffer("ast_scale", torch.from_numpy(safe_ast_scale))
        self.ast_linear = torch.nn.Linear(768, 128)
        self.relu = torch.nn.ReLU()
        self.batch_norm = torch.nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01)
        self.dropout = torch.nn.Dropout(dropout)
        self.output = torch.nn.Linear(128, 3)
        if pipeline == PIPELINES[0]:
            self.age_hidden = None
            self.age_output = None
            self.gate_logits = None
            return
        with np.errstate(all="ignore"):
            age_median = np.nanmedian(age_train, axis=0)
        age_median = np.where(np.isfinite(age_median), age_median, 0.0).astype(np.float32)
        imputed = np.where(np.isfinite(age_train), age_train, age_median[None, :])
        age_mean = imputed.mean(axis=0).astype(np.float32)
        age_scale = imputed.std(axis=0).astype(np.float32)
        age_scale = np.where(age_scale > 1.0e-8, age_scale, 1.0).astype(np.float32)
        self.register_buffer("age_median", torch.from_numpy(age_median))
        self.register_buffer("age_mean", torch.from_numpy(age_mean))
        self.register_buffer("age_scale", torch.from_numpy(age_scale))
        self.age_hidden = torch.nn.Linear(len(FEATURE_NAMES), age_hidden_units)
        self.age_output = torch.nn.Linear(age_hidden_units, 128)
        torch.nn.init.zeros_(self.age_output.weight)
        torch.nn.init.zeros_(self.age_output.bias)
        self.gate_logits = (
            torch.nn.Parameter(torch.zeros(128, dtype=torch.float32))
            if pipeline == PIPELINES[2]
            else None
        )

    def forward(
        self, ast_embeddings: torch.Tensor, age_features: torch.Tensor
    ) -> torch.Tensor:
        ast = (ast_embeddings - self.ast_mean) / self.ast_scale
        hidden = self.relu(self.ast_linear(ast))
        if self.pipeline != PIPELINES[0]:
            if self.age_hidden is None or self.age_output is None:
                raise RuntimeError("IDEA-068 age branch is missing")
            imputed = torch.where(torch.isfinite(age_features), age_features, self.age_median)
            standardized = (imputed - self.age_mean) / self.age_scale
            residual = self.age_output(torch.nn.functional.gelu(self.age_hidden(standardized)))
            if self.pipeline == PIPELINES[2]:
                voiced_fraction = torch.clamp(
                    imputed[:, VOICED_FRACTION_INDEX], min=0.0, max=1.0
                )
                voiced_probability = torch.clamp(
                    imputed[:, VOICED_PROBABILITY_INDEX], min=0.0, max=1.0
                )
                reliability = torch.sqrt(voiced_fraction * voiced_probability)
                residual = residual * reliability[:, None] * torch.sigmoid(
                    self.gate_logits
                )[None, :]
            hidden = hidden + residual
        hidden = self.batch_norm(hidden)
        hidden = self.dropout(hidden)
        return self.output(hidden)

    def audit(self) -> dict[str, Any]:
        result = {
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in self.parameters())
            )
        }
        if self.gate_logits is not None:
            gate = torch.sigmoid(self.gate_logits.detach()).cpu().numpy()
            result["gate_mean"] = float(gate.mean())
            result["gate_min"] = float(gate.min())
            result["gate_max"] = float(gate.max())
        return result


def build_model(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    age_features: np.ndarray,
    train_indices: np.ndarray,
) -> AgeResidualClassifier:
    embeddings = store.frozen_embeddings[train_indices]
    return AgeResidualClassifier(
        pipeline=pipeline,
        ast_mean=embeddings.mean(axis=0),
        ast_scale=embeddings.std(axis=0),
        age_train=age_features[train_indices],
        dropout=float(protocol["fixed_training"]["dropout"]),
        age_hidden_units=int(protocol["fixed_training"]["age_hidden_units"]),
    )


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("An IDEA-068 training role is missing a class")
    return (len(labels) / (3.0 * counts)).astype(np.float32)


def train_one_epoch(
    model: AgeResidualClassifier,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    store: Any,
    age_features: np.ndarray,
    call_class_weights: torch.Tensor,
    device: torch.device,
    gradient_clip: float,
) -> tuple[float, dict[str, Any]]:
    model.train()
    losses = []
    processed_cats: list[str] = []
    processed_calls: list[int] = []
    batch_sizes: list[int] = []
    for cpu_batch in loader:
        batch = idea051.move_set_batch(cpu_batch, device)
        call_indices = cpu_batch["call_indices"].numpy().astype(np.int64)
        labels = torch.from_numpy(store.labels[call_indices]).to(device)
        age = torch.from_numpy(age_features[call_indices]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(batch["embeddings"], age)
            per_call = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            weights = call_class_weights[labels]
            loss = (per_call * weights).sum() / weights.sum()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        processed_cats.extend(str(value) for value in cpu_batch["cat_ids"])
        processed_calls.extend(int(value) for value in call_indices)
        batch_sizes.append(len(cpu_batch["cat_ids"]))
    if len(processed_cats) != len(set(processed_cats)):
        raise RuntimeError("An IDEA-068 epoch repeated a cat")
    if len(processed_calls) != len(set(processed_calls)):
        raise RuntimeError("An IDEA-068 epoch repeated a call")
    return float(np.mean(losses)), {
        "cats": len(processed_cats),
        "calls": len(processed_calls),
        "batch_sizes_cats": batch_sizes,
        "cat_order_sha256": hashlib.sha256(
            "\n".join(processed_cats).encode("utf-8")
        ).hexdigest(),
        "call_coverage_sha256": hashlib.sha256(
            np.sort(np.asarray(processed_calls, dtype="<i8")).tobytes()
        ).hexdigest(),
    }


def predict(
    model: AgeResidualClassifier,
    store: Any,
    age_features: np.ndarray,
    indices: np.ndarray,
    cat_batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = idea051.CatSetDataset(store, indices)
    loader = idea051.build_set_loader(dataset, cat_batch_size, False, seed)
    model.eval()
    rows = []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = idea051.move_set_batch(cpu_batch, device)
            call_indices = cpu_batch["call_indices"].numpy().astype(np.int64)
            age = torch.from_numpy(age_features[call_indices]).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(batch["embeddings"], age)
                probabilities = torch.softmax(logits, dim=1).float().cpu().numpy()
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
    animals = idea051.calls_to_animals(calls)
    return animals, calls


def brier(animals: pd.DataFrame) -> float:
    probabilities = animals[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    labels = animals["true_label"].to_numpy(dtype=np.int64)
    targets = np.eye(3, dtype=float)[labels]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def fit_inner(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    age_features: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    idea051.reference.historical.set_seed(seed)
    fixed = protocol["fixed_training"]
    model = build_model(
        pipeline, protocol, store, age_features, train_indices
    ).to(device)
    # Equalize dropout/AMP RNG streams after pipeline-specific module creation.
    training_seed = seed + int(fixed["post_build_seed_offset"])
    idea051.reference.historical.set_seed(training_seed)
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=float(fixed["learning_rate"]),
        eps=float(fixed["optimizer_epsilon"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    dataset = idea051.CatSetDataset(store, train_indices)
    loader = idea051.build_set_loader(
        dataset, int(fixed["cat_batch_size"]), True, seed
    )
    call_weights = torch.from_numpy(class_weights(store.labels[train_indices])).to(
        device
    )
    best_loss = float("inf")
    best_epoch = 1
    best_state = idea051.cpu_state_dict(model)
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
            age_features,
            call_weights,
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
        validation_loss = idea051.animal_cross_entropy(animals)
        metrics = idea051.animal_metrics(animals)
        history.append(
            {
                "epoch": epoch,
                "train_call_loss": train_loss,
                "train_audit": train_audit,
                "validation_animal_cross_entropy": validation_loss,
                "validation_animal_brier": brier(animals),
                "validation_animal_metrics": metrics,
            }
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = idea051.cpu_state_dict(model)
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
        raise RuntimeError("No IDEA-068 checkpoint was selected")
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
        np.abs(
            reload_animals[list(PROBABILITY_COLUMNS)].to_numpy()
            - best_animals[list(PROBABILITY_COLUMNS)].to_numpy()
        ).max()
    )
    audit = {
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "base_seed": seed,
        "training_seed": training_seed,
        "best_validation_animal_cross_entropy": best_loss,
        "best_validation_animal_brier": brier(best_animals),
        "best_validation_animal_metrics": idea051.animal_metrics(best_animals),
        "checkpoint_reload_max_probability_difference": reload_difference,
        "model": model.audit(),
        "train_seconds": float(time.perf_counter() - started),
        "history": history,
        "outer_test_accessed": False,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return audit, best_animals, best_calls


def aggregate(fits: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    by_key = {
        (fit["pipeline"], fit["repeat"], fit["outer_fold"]): fit for fit in fits
    }
    fold_rows = []
    pooled: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
    for repeat in protocol["screen"]["repeats"]:
        for outer_fold in protocol["screen"][
            "outer_folds_used_as_inner_role_definitions"
        ]:
            frames = {}
            metrics = {}
            for pipeline in PIPELINES:
                fit = by_key[(pipeline, repeat, outer_fold)]
                frame = pd.read_csv(
                    REPO_ROOT / fit["validation_animal_predictions"],
                    dtype={"cat_id": str},
                )
                tagged = frame.copy()
                tagged["repeat"] = repeat
                tagged["outer_fold"] = outer_fold
                pooled[pipeline].append(tagged)
                frames[pipeline] = frame
                metrics[pipeline] = idea051.animal_metrics(frame)
            control = PIPELINES[0]
            row: dict[str, Any] = {"repeat": repeat, "outer_fold": outer_fold}
            for pipeline in PIPELINES:
                row[f"{pipeline}_macro_f1"] = metrics[pipeline]["macro_f1"]
                row[f"{pipeline}_cross_entropy"] = idea051.animal_cross_entropy(
                    frames[pipeline]
                )
                row[f"{pipeline}_brier"] = brier(frames[pipeline])
                row[f"{pipeline}_senior_recall"] = metrics[pipeline]["per_class"][
                    "senior"
                ]["recall"]
                if pipeline != control:
                    row[f"{pipeline}_delta_macro_f1"] = (
                        metrics[pipeline]["macro_f1"] - metrics[control]["macro_f1"]
                    )
            fold_rows.append(row)
    folds = pd.DataFrame(fold_rows)
    pooled_frames = {
        pipeline: pd.concat(parts, ignore_index=True) for pipeline, parts in pooled.items()
    }
    pooled_metrics = {
        pipeline: {
            "animal_occurrences": int(len(frame)),
            "metrics": idea051.animal_metrics(frame),
            "cross_entropy": idea051.animal_cross_entropy(frame),
            "brier": brier(frame),
        }
        for pipeline, frame in pooled_frames.items()
    }
    mean_fold_macro_f1 = {
        pipeline: float(folds[f"{pipeline}_macro_f1"].mean())
        for pipeline in PIPELINES
    }
    candidates = {}
    gate = protocol["gate"]
    for pipeline in PIPELINES[1:]:
        delta_column = f"{pipeline}_delta_macro_f1"
        per_repeat_senior_delta = {}
        for repeat in protocol["screen"]["repeats"]:
            control_repeat = pooled_frames[PIPELINES[0]][
                pooled_frames[PIPELINES[0]]["repeat"] == repeat
            ]
            candidate_repeat = pooled_frames[pipeline][
                pooled_frames[pipeline]["repeat"] == repeat
            ]
            per_repeat_senior_delta[str(repeat)] = float(
                idea051.animal_metrics(candidate_repeat)["per_class"]["senior"][
                    "recall"
                ]
                - idea051.animal_metrics(control_repeat)["per_class"]["senior"][
                    "recall"
                ]
            )
        conditions = {
            "mean_fold_macro_f1_gain": float(folds[delta_column].mean())
            >= float(gate["minimum_mean_fold_macro_f1_gain"]),
            "positive_folds": int((folds[delta_column] > 0).sum())
            >= int(gate["minimum_positive_folds"]),
            "pooled_cross_entropy_nonworse": pooled_metrics[pipeline][
                "cross_entropy"
            ]
            <= pooled_metrics[PIPELINES[0]]["cross_entropy"],
            "pooled_brier_nonworse": pooled_metrics[pipeline]["brier"]
            <= pooled_metrics[PIPELINES[0]]["brier"],
            "worst_fold_macro_f1": float(folds[delta_column].min())
            >= float(gate["minimum_worst_fold_macro_f1_delta"]),
            "per_repeat_senior_recall": all(
                delta >= float(gate["minimum_per_repeat_senior_recall_delta"])
                for delta in per_repeat_senior_delta.values()
            ),
        }
        candidates[pipeline] = {
            "mean_fold_macro_f1_delta": float(folds[delta_column].mean()),
            "positive_folds": int((folds[delta_column] > 0).sum()),
            "worst_fold_macro_f1_delta": float(folds[delta_column].min()),
            "per_repeat_senior_recall_delta": per_repeat_senior_delta,
            "gate_conditions": conditions,
            "gate_passed": bool(all(conditions.values())),
        }
    return {
        "status": "complete",
        "outer_test_accessed": False,
        "fits": len(fits),
        "fold_results": fold_rows,
        "mean_fold_macro_f1": mean_fold_macro_f1,
        "pooled_validation": pooled_metrics,
        "candidates": candidates,
        "A2_minus_A1_mean_fold_macro_f1": float(
            folds[f"{PIPELINES[2]}_macro_f1"].mean()
            - folds[f"{PIPELINES[1]}_macro_f1"].mean()
        ),
        "any_candidate_gate_passed": bool(
            any(value["gate_passed"] for value in candidates.values())
        ),
    }


def screen(
    protocol: dict[str, Any], run_root: Path, device_name: str, resume: bool
) -> dict[str, Any]:
    configure_determinism()
    run_root.mkdir(parents=True, exist_ok=True)
    store = idea051.reference.historical.idea019.load_feature_store()
    if len(store.call_ids) != int(protocol["data"]["calls"]):
        raise RuntimeError("IDEA-068 analysis call count changed")
    age_features, extraction_summary = load_age_features(
        protocol, run_root, store.call_ids
    )
    device = idea051.reference.historical.idea019.resolve_device(device_name)
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "feature_sha256": extraction_summary["feature_sha256"],
        "outer_test_accessed": False,
        "pipelines": list(PIPELINES),
        "screen": protocol["screen"],
        "determinism": protocol["determinism"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "librosa": librosa.__version__,
            "device": str(device),
        },
    }
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.exists():
        if not resume or read_json(manifest_path) != manifest:
            raise RuntimeError("Existing IDEA-068 run manifest differs")
    else:
        write_json(manifest_path, manifest)
    roles = pd.read_csv(
        REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str}
    )
    completed = []
    screen_spec = protocol["screen"]
    for repeat in screen_spec["repeats"]:
        for outer_fold in screen_spec["outer_folds_used_as_inner_role_definitions"]:
            indices = idea051.reference.historical.fold_indices(
                store, roles, repeat, outer_fold, include_test=False
            )
            seed = idea051.reference.historical.full_seed(
                int(screen_spec["base_seed"]), repeat, outer_fold
            )
            pair = []
            for pipeline in PIPELINES:
                output_dir = (
                    run_root
                    / "fits"
                    / pipeline
                    / f"repeat_{repeat}"
                    / f"fold_{outer_fold}"
                )
                summary_path = output_dir / "fit_summary.json"
                if summary_path.exists():
                    if not resume:
                        raise FileExistsError(summary_path)
                    fit = read_json(summary_path)
                    completed.append(fit)
                    pair.append(fit)
                    continue
                print(
                    f"=== {pipeline} repeat={repeat} fold={outer_fold} seed={seed} ===",
                    flush=True,
                )
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
                output_dir.mkdir(parents=True, exist_ok=True)
                animal_path = output_dir / "validation_animal_predictions.csv"
                call_path = output_dir / "validation_call_predictions.csv"
                animals.to_csv(animal_path, index=False)
                calls.to_csv(call_path, index=False)
                fit = {
                    "status": "complete",
                    "pipeline": pipeline,
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "base_seed": int(screen_spec["base_seed"]),
                    "full_seed": seed,
                    "outer_test_accessed": False,
                    "train_calls": int(len(indices["train"])),
                    "validation_calls": int(len(indices["validation"])),
                    "validation_cats": int(animals["cat_id"].nunique()),
                    "validation_animal_predictions": animal_path.relative_to(
                        REPO_ROOT
                    ).as_posix(),
                    "validation_call_predictions": call_path.relative_to(
                        REPO_ROOT
                    ).as_posix(),
                    "audit": audit,
                }
                write_json(summary_path, fit)
                completed.append(fit)
                pair.append(fit)
            common_epochs = min(len(item["audit"]["history"]) for item in pair)
            for epoch in range(common_epochs):
                reference_audit = pair[0]["audit"]["history"][epoch]["train_audit"]
                for candidate in pair[1:]:
                    candidate_audit = candidate["audit"]["history"][epoch][
                        "train_audit"
                    ]
                    if (
                        reference_audit["cat_order_sha256"]
                        != candidate_audit["cat_order_sha256"]
                        or reference_audit["call_coverage_sha256"]
                        != candidate_audit["call_coverage_sha256"]
                    ):
                        raise RuntimeError("IDEA-068 paired batch order differs")
    summary = aggregate(completed, protocol)
    write_json(run_root / "inner_screen_summary.json", summary)
    write_json(
        run_root / "run_summary.json",
        {
            "status": "complete",
            "completed_fits": len(completed),
            "expected_fits": int(screen_spec["total_fits"]),
            "any_candidate_gate_passed": summary["any_candidate_gate_passed"],
        },
    )
    return summary


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = resolve_run_root(args.output_subdir)
    if args.stage == "extract":
        result = extract_features(protocol, run_root)
    else:
        result = screen(protocol, run_root, args.device, args.resume)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
