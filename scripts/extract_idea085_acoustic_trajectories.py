"""Extract the locked IDEA-085 six-channel frame-level acoustic cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "configs/protocol/meowagenet_idea085_trajectory_extraction_v1.json"
)
DEFAULT_RUN_ROOT = (
    REPO_ROOT / "runs/meowagenet_idea085_acoustic_temporal_residual_v1"
)
FEATURE_NAMES = (
    "log_f0",
    "voiced_probability",
    "periodicity",
    "log_rms",
    "spectral_tilt",
    "spectral_flatness",
)
IDEA068_FEATURE_NAMES = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-subdir",
        default="meowagenet_idea085_acoustic_temporal_residual_v1",
    )
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def resolve_run_root(output_subdir: str) -> Path:
    runs_root = (REPO_ROOT / "runs").resolve()
    run_root = (runs_root / output_subdir).resolve()
    if runs_root not in run_root.parents:
        raise ValueError("--output-subdir must stay below runs")
    return run_root


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea085-trajectory-extraction-v1":
        raise RuntimeError("Unexpected IDEA-085 extraction protocol")
    if protocol.get("status") != "locked_before_trajectory_extraction":
        raise RuntimeError("IDEA-085 extraction protocol is not locked")
    extraction = protocol["trajectory_extraction"]
    expected = {
        "sample_rate_hz": 16_000,
        "fmin_hz": 60.0,
        "fmax_hz": 2_000.0,
        "frame_length": 1_024,
        "hop_length": 160,
        "center": True,
    }
    for key, value in expected.items():
        if extraction.get(key) != value:
            raise RuntimeError(f"IDEA-085 extraction setting changed: {key}")
    if tuple(extraction.get("feature_names", ())) != FEATURE_NAMES:
        raise RuntimeError("IDEA-085 channel order changed")
    if extraction.get("preserve_original_frame_grid") is not True:
        raise RuntimeError("IDEA-085 must preserve the original frame grid")
    if extraction.get("concatenate_only_valid_f0_frames") is not False:
        raise RuntimeError("IDEA-085 must not concatenate only valid-F0 frames")
    if extraction.get("per_call_f0_normalization") is not False:
        raise RuntimeError("IDEA-085 must not normalize F0 per call")
    if extraction.get("label_information_used") is not False:
        raise RuntimeError("IDEA-085 extraction must be label blind")
    if protocol["data"].get("calls") != 792 or protocol["data"].get("cats") != 111:
        raise RuntimeError("IDEA-085 data scope changed")
    dependencies = protocol["dependencies"]
    checks = {
        REPO_ROOT / protocol["data"]["manifest_path"]: protocol["data"][
            "manifest_sha256"
        ],
        REPO_ROOT / protocol["data"]["frozen_embedding_path"]: protocol["data"][
            "frozen_embedding_sha256"
        ],
        REPO_ROOT / dependencies["idea068_protocol_path"]: dependencies[
            "idea068_protocol_sha256"
        ],
        REPO_ROOT / dependencies["idea068_runner_path"]: dependencies[
            "idea068_runner_sha256"
        ],
        REPO_ROOT / dependencies["idea068_feature_path"]: dependencies[
            "idea068_feature_sha256"
        ],
        REPO_ROOT / dependencies["idea068_feature_summary_path"]: dependencies[
            "idea068_feature_summary_sha256"
        ],
    }
    optional = {
        Path(__file__).resolve(): dependencies["extractor_sha256"],
        REPO_ROOT / dependencies["tests_path"]: dependencies["tests_sha256"],
    }
    checks.update(
        {
            path: expected_hash
            for path, expected_hash in optional.items()
            if not str(expected_hash).startswith("PENDING_")
        }
    )
    for path, expected_hash in checks.items():
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"IDEA-085 dependency checksum mismatch: {path}")


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


def frame_periodicity(
    waveform: np.ndarray,
    f0: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    """The IDEA-068 normalized-autocorrelation periodicity, frame for frame."""

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
        result[frame_index] = float(
            np.clip(np.dot(left, right) / denominator, 0.0, 0.999)
        )
    return result


def extract_call_trajectory(
    waveform: np.ndarray, sample_rate: int, settings: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    frame_length = int(settings["frame_length"])
    hop_length = int(settings["hop_length"])
    f0, _, voiced_probability = librosa.pyin(
        waveform,
        fmin=float(settings["fmin_hz"]),
        fmax=float(settings["fmax_hz"]),
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        center=bool(settings["center"]),
    )
    f0 = np.asarray(f0, dtype=np.float64)
    voiced_probability = np.asarray(voiced_probability, dtype=np.float64)
    magnitude = np.abs(
        librosa.stft(
            waveform,
            n_fft=frame_length,
            hop_length=hop_length,
            win_length=frame_length,
            center=bool(settings["center"]),
        )
    ).astype(np.float64)
    rms = np.asarray(
        librosa.feature.rms(S=magnitude, frame_length=frame_length)[0],
        dtype=np.float64,
    )
    flatness = np.asarray(
        librosa.feature.spectral_flatness(S=magnitude + 1.0e-10)[0],
        dtype=np.float64,
    )
    frame_count = min(len(f0), len(voiced_probability), magnitude.shape[1], len(rms), len(flatness))
    if frame_count <= 0:
        raise RuntimeError("IDEA-085 extractor produced no frames")
    f0 = f0[:frame_count]
    voiced_probability = voiced_probability[:frame_count]
    magnitude = magnitude[:, :frame_count]
    rms = rms[:frame_count]
    flatness = flatness[:frame_count]

    voiced = np.isfinite(f0) & (f0 > 0)
    log_f0 = np.full(frame_count, np.nan, dtype=np.float64)
    log_f0[voiced] = np.log(f0[voiced])
    periodicity = frame_periodicity(
        waveform, f0, sample_rate, frame_length, hop_length
    )[:frame_count]

    # Unlike the IDEA-068 summary statistics, retain these three channels on every
    # computable frame, including unvoiced frames. The voiced mask is recoverable
    # from finite log_f0 when reproducing the old summaries.
    log_rms = np.log(rms + 1.0e-10)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=frame_length)
    low_hz, high_hz = (float(x) for x in settings["spectral_tilt_band_hz"])
    frequency_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    log_frequency = np.log(frequencies[frequency_mask])
    centered_frequency = log_frequency - log_frequency.mean()
    denominator = float(np.dot(centered_frequency, centered_frequency))
    log_magnitude = np.log(magnitude[frequency_mask] + 1.0e-10)
    centered_magnitude = log_magnitude - log_magnitude.mean(axis=0, keepdims=True)
    spectral_tilt = (
        centered_frequency[:, None] * centered_magnitude
    ).sum(axis=0) / denominator

    values = np.column_stack(
        (
            log_f0,
            voiced_probability,
            periodicity,
            log_rms,
            spectral_tilt,
            flatness,
        )
    ).astype(np.float32)
    hop_seconds = hop_length / float(sample_rate)
    time_seconds = np.arange(frame_count, dtype=np.float64) * hop_seconds
    if values.shape != (frame_count, len(FEATURE_NAMES)):
        raise RuntimeError("IDEA-085 trajectory shape changed")
    return values, time_seconds


def finite_quantile(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, q)) if len(finite) else float("nan")


def finite_median(values: np.ndarray) -> float:
    return finite_quantile(values, 0.5)


def reconstruct_idea068_features(values: np.ndarray) -> np.ndarray:
    """Reconstruct the old 20-D summaries from one six-channel trajectory."""

    log_f0, voiced_probability, periodicity, log_rms, spectral_tilt, flatness = (
        values[:, index].astype(np.float64) for index in range(6)
    )
    frame_count = len(values)
    voiced = np.isfinite(log_f0)
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
    pair_mask = voiced[:-1] & voiced[1:] if frame_count >= 2 else np.zeros(0, dtype=bool)
    log_delta = np.abs(np.diff(log_f0))[pair_mask]
    f0 = np.full(frame_count, np.nan, dtype=np.float64)
    f0[voiced] = np.exp(log_f0[voiced])
    periods = np.full(frame_count, np.nan, dtype=np.float64)
    periods[voiced] = 1.0 / f0[voiced]
    period_delta = np.abs(np.diff(periods))[pair_mask]
    period_median = finite_median(periods)
    period_variation = (
        finite_median(period_delta) / period_median
        if np.isfinite(period_median) and period_median > 0
        else float("nan")
    )
    valid_periodicity = periodicity[np.isfinite(periodicity)]
    hnr = np.full_like(valid_periodicity, np.nan)
    if len(valid_periodicity):
        clipped = np.clip(valid_periodicity, 1.0e-6, 0.999)
        hnr = 10.0 * np.log10(clipped / (1.0 - clipped))
    rms = np.exp(log_rms) - 1.0e-10
    rms_pair_delta = np.abs(np.diff(rms))[pair_mask]
    rms_median = finite_median(rms[voiced])
    amplitude_variation = (
        finite_median(rms_pair_delta) / rms_median
        if np.isfinite(rms_median) and rms_median > 0
        else float("nan")
    )
    finite_voicing = voiced_probability[np.isfinite(voiced_probability)]
    voiced_log_rms = log_rms[voiced]
    voiced_tilt = spectral_tilt[voiced]
    voiced_flatness = flatness[voiced]
    return np.asarray(
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
            finite_median(voiced_log_rms),
            (
                finite_quantile(voiced_log_rms, 0.75)
                - finite_quantile(voiced_log_rms, 0.25)
                if np.isfinite(finite_quantile(voiced_log_rms, 0.75))
                and np.isfinite(finite_quantile(voiced_log_rms, 0.25))
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


def cache_paths(run_root: Path) -> tuple[Path, Path, Path]:
    feature_dir = run_root / "features"
    return (
        feature_dir / "acoustic_trajectories.npz",
        feature_dir / "trajectory_schema.json",
        feature_dir / "extraction_summary.json",
    )


def load_and_validate_cache(
    protocol: dict[str, Any], run_root: Path, verify_hash: bool = True
) -> dict[str, Any]:
    feature_path, schema_path, summary_path = cache_paths(run_root)
    for path in (feature_path, schema_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = read_json(summary_path)
    schema = read_json(schema_path)
    if summary.get("status") != "complete":
        raise RuntimeError("IDEA-085 cache is incomplete")
    if verify_hash and sha256(feature_path) != summary.get("feature_sha256"):
        raise RuntimeError("IDEA-085 cache hash mismatch")
    if verify_hash and sha256(schema_path) != summary.get("schema_sha256"):
        raise RuntimeError("IDEA-085 schema hash mismatch")
    if summary.get("protocol_sha256") != sha256(PROTOCOL_PATH):
        raise RuntimeError("IDEA-085 cache protocol hash mismatch")
    if tuple(schema.get("feature_names", ())) != FEATURE_NAMES:
        raise RuntimeError("IDEA-085 schema channel order mismatch")
    with np.load(feature_path) as loaded:
        required = {"call_ids", "offsets", "values", "feature_names", "time_seconds"}
        if set(loaded.files) != required:
            raise RuntimeError(f"IDEA-085 cache keys changed: {loaded.files}")
        call_ids = loaded["call_ids"]
        offsets = loaded["offsets"]
        values = loaded["values"]
        feature_names = loaded["feature_names"]
        time_seconds = loaded["time_seconds"]
        if call_ids.dtype.kind != "U" or feature_names.dtype.kind != "U":
            raise RuntimeError("IDEA-085 string arrays must be Unicode")
        if offsets.dtype != np.int64 or values.dtype != np.float32:
            raise RuntimeError("IDEA-085 numeric dtypes changed")
        if call_ids.shape != (792,) or offsets.shape != (793,):
            raise RuntimeError("IDEA-085 cache call/offset shape changed")
        if values.ndim != 2 or values.shape[1] != 6:
            raise RuntimeError("IDEA-085 values shape changed")
        if time_seconds.shape != (len(values),):
            raise RuntimeError("IDEA-085 time grid shape changed")
        if tuple(feature_names.astype(str)) != FEATURE_NAMES:
            raise RuntimeError("IDEA-085 stored channel order changed")
        if offsets[0] != 0 or offsets[-1] != len(values) or np.any(np.diff(offsets) <= 0):
            raise RuntimeError("IDEA-085 offsets are invalid")
        hop_seconds = float(protocol["trajectory_extraction"]["hop_length"]) / float(
            protocol["trajectory_extraction"]["sample_rate_hz"]
        )
        for index in range(len(call_ids)):
            left, right = int(offsets[index]), int(offsets[index + 1])
            expected_time = np.arange(right - left, dtype=np.float64) * hop_seconds
            if not np.array_equal(time_seconds[left:right], expected_time):
                raise RuntimeError(f"IDEA-085 time grid differs: {call_ids[index]}")
    return summary


def build_schema(protocol: dict[str, Any]) -> dict[str, Any]:
    extraction = protocol["trajectory_extraction"]
    return {
        "schema_version": "1.0",
        "cache_id": "meowagenet-idea085-acoustic-trajectories-v1",
        "alignment_key": "call_id",
        "arrays": {
            "call_ids": {"shape": "[N]", "dtype": "Unicode"},
            "offsets": {"shape": "[N+1]", "dtype": "int64"},
            "values": {"shape": "[total_frames,6]", "dtype": "float32"},
            "feature_names": {"shape": "[6]", "dtype": "Unicode"},
            "time_seconds": {
                "shape": "[total_frames]",
                "dtype": "float64",
                "model_input": False,
            },
        },
        "feature_names": list(FEATURE_NAMES),
        "channel_semantics": {
            "log_f0": "Natural log of pYIN F0; NaN on unvoiced/missing frames.",
            "voiced_probability": "Raw pYIN voiced probability on the original grid; NaN preserved.",
            "periodicity": "IDEA-068 normalized autocorrelation at the pYIN period; NaN without valid F0.",
            "log_rms": "Natural log RMS plus 1e-10 on every computable frame, including unvoiced frames.",
            "spectral_tilt": "IDEA-068 log-magnitude versus log-frequency slope over 200-4000 Hz on every computable frame.",
            "spectral_flatness": "librosa spectral flatness on every computable frame.",
        },
        "missingness": "NaN is retained; models must consume separate finite indicators.",
        "grid": {
            "sample_rate_hz": extraction["sample_rate_hz"],
            "frame_length": extraction["frame_length"],
            "hop_length": extraction["hop_length"],
            "hop_seconds": extraction["hop_length"] / extraction["sample_rate_hz"],
            "center": extraction["center"],
            "valid_f0_frames_concatenated": False,
        },
        "label_information_used": False,
        "scope_note": "RMS, spectral tilt, and flatness retain all computable frames; IDEA-068 summarized these channels only on the finite-F0 voiced subset.",
    }


def extraction_summary_from_arrays(
    protocol: dict[str, Any],
    call_ids: np.ndarray,
    cat_ids: np.ndarray,
    offsets: np.ndarray,
    values: np.ndarray,
    feature_path: Path,
    schema_path: Path,
    source_hashes_verified: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    lengths = np.diff(offsets)
    finite = np.isfinite(values)
    all_unvoiced = []
    for index, call_id in enumerate(call_ids):
        left, right = int(offsets[index]), int(offsets[index + 1])
        if not np.isfinite(values[left:right, 0]).any():
            all_unvoiced.append(str(call_id))
    longest_index = int(np.argmax(lengths))
    return {
        "status": "complete",
        "label_information_used": False,
        "calls": int(len(call_ids)),
        "cats": int(len(np.unique(cat_ids.astype(str)))),
        "source_hashes_verified": int(source_hashes_verified),
        "total_frames": int(len(values)),
        "feature_names": list(FEATURE_NAMES),
        "sequence_length_frames": {
            "min": int(lengths.min()),
            "median": float(np.median(lengths)),
            "max": int(lengths.max()),
        },
        "longest_sequence": {
            "call_id": str(call_ids[longest_index]),
            "frames": int(lengths[longest_index]),
            "duration_on_grid_seconds": float(
                (lengths[longest_index] - 1)
                * protocol["trajectory_extraction"]["hop_length"]
                / protocol["trajectory_extraction"]["sample_rate_hz"]
            ),
        },
        "fully_unvoiced_or_no_valid_f0_calls": {
            "count": len(all_unvoiced),
            "call_ids": all_unvoiced,
        },
        "finite_by_channel": {
            name: {
                "count": int(finite[:, index].sum()),
                "missing": int((~finite[:, index]).sum()),
                "rate": float(finite[:, index].mean()),
            }
            for index, name in enumerate(FEATURE_NAMES)
        },
        "original_10ms_grid_preserved": True,
        "nan_preserved": True,
        "finite_indicators_stored_in_cache": False,
        "finite_indicators_required_in_model": True,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "extractor_sha256": sha256(Path(__file__).resolve()),
        "feature_path": feature_path.relative_to(REPO_ROOT).as_posix(),
        "feature_sha256": sha256(feature_path),
        "schema_path": schema_path.relative_to(REPO_ROOT).as_posix(),
        "schema_sha256": sha256(schema_path),
        "elapsed_seconds": float(elapsed_seconds),
    }


def old_summary_consistency(
    protocol: dict[str, Any], call_ids: np.ndarray, offsets: np.ndarray, values: np.ndarray
) -> dict[str, Any]:
    old_path = REPO_ROOT / protocol["dependencies"]["idea068_feature_path"]
    with np.load(old_path) as old:
        old_ids = old["call_ids"].astype(str)
        old_names = tuple(old["feature_names"].astype(str))
        old_values = old["features"].astype(np.float32)
    if old_names != IDEA068_FEATURE_NAMES:
        raise RuntimeError("IDEA-068 feature order changed")
    if not np.array_equal(old_ids, call_ids.astype(str)):
        raise RuntimeError("IDEA-068 and IDEA-085 call order differs")
    lengths = np.diff(offsets)
    sample = set(np.linspace(0, len(call_ids) - 1, 24, dtype=int).tolist())
    sample.add(int(np.argmax(lengths)))
    for index in range(len(call_ids)):
        left, right = int(offsets[index]), int(offsets[index + 1])
        if not np.isfinite(values[left:right, 0]).any():
            sample.add(index)
    sample_indices = sorted(sample)
    reconstructed = np.stack(
        [
            reconstruct_idea068_features(
                values[int(offsets[index]) : int(offsets[index + 1])]
            )
            for index in sample_indices
        ]
    )
    expected = old_values[sample_indices]
    if not np.array_equal(np.isnan(reconstructed), np.isnan(expected)):
        raise RuntimeError("IDEA-085 reconstruction changed IDEA-068 missingness")
    finite = np.isfinite(reconstructed) & np.isfinite(expected)
    absolute = np.zeros_like(reconstructed, dtype=np.float64)
    absolute[finite] = np.abs(
        reconstructed[finite].astype(np.float64) - expected[finite].astype(np.float64)
    )
    if not np.allclose(reconstructed, expected, atol=2.0e-4, rtol=2.0e-4, equal_nan=True):
        maximum = float(absolute.max())
        raise RuntimeError(f"IDEA-085 cannot reconstruct IDEA-068 sample: {maximum}")
    return {
        "sample_calls": len(sample_indices),
        "sample_call_ids": [str(call_ids[index]) for index in sample_indices],
        "atol": 2.0e-4,
        "rtol": 2.0e-4,
        "maximum_absolute_difference": float(absolute.max()),
        "per_feature_maximum_absolute_difference": {
            name: float(absolute[:, index].max())
            for index, name in enumerate(IDEA068_FEATURE_NAMES)
        },
        "missingness_exact": True,
        "status": "PASS",
    }


def extract(protocol: dict[str, Any], run_root: Path) -> dict[str, Any]:
    verify_protocol(protocol)
    feature_path, schema_path, summary_path = cache_paths(run_root)
    if any(path.exists() for path in (feature_path, schema_path, summary_path)):
        raise FileExistsError("IDEA-085 canonical cache output already exists")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path = REPO_ROOT / protocol["data"]["frozen_embedding_path"]
    with np.load(frozen_path) as frozen:
        call_ids = frozen["call_ids"].astype(str)
        cat_ids = frozen["cat_ids"].astype(str)
    if len(call_ids) != 792 or len(np.unique(cat_ids)) != 111:
        raise RuntimeError("IDEA-085 frozen identity scope differs")
    manifest = pd.read_csv(REPO_ROOT / protocol["data"]["manifest_path"])
    manifest = manifest[
        manifest["analysis_include"].astype(str).str.lower() == "true"
    ].copy()
    by_name = manifest.set_index("filename", verify_integrity=True)
    if set(call_ids) != set(by_name.index.astype(str)):
        raise RuntimeError("IDEA-085 manifest and frozen call IDs differ")
    manifest_cat = by_name.loc[call_ids, "analysis_cat_id"].astype(str).to_numpy()
    if not np.array_equal(manifest_cat, cat_ids):
        raise RuntimeError("IDEA-085 manifest and frozen cat IDs differ")

    settings = protocol["trajectory_extraction"]
    sample_rate = int(settings["sample_rate_hz"])
    trajectories: list[np.ndarray] = []
    times: list[np.ndarray] = []
    source_hashes_verified = 0
    started = time.perf_counter()
    for index, call_id in enumerate(call_ids):
        row = by_name.loc[call_id]
        audio_path = REPO_ROOT / str(row["local_relpath"])
        if not audio_path.is_file() or sha256(audio_path) != str(row["sha256"]):
            raise RuntimeError(f"IDEA-085 audio checksum mismatch: {audio_path}")
        source_hashes_verified += 1
        waveform = load_audio(audio_path, sample_rate)
        values, time_seconds = extract_call_trajectory(waveform, sample_rate, settings)
        trajectories.append(values)
        times.append(time_seconds)
        if (index + 1) % 50 == 0 or index + 1 == len(call_ids):
            print(f"IDEA-085 extracted {index + 1}/{len(call_ids)} calls", flush=True)

    lengths = np.asarray([len(item) for item in trajectories], dtype=np.int64)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    values = np.concatenate(trajectories, axis=0).astype(np.float32, copy=False)
    time_seconds = np.concatenate(times).astype(np.float64, copy=False)
    schema = build_schema(protocol)
    write_json(schema_path, schema)
    np.savez_compressed(
        feature_path,
        call_ids=np.asarray(call_ids, dtype="U"),
        offsets=offsets,
        values=values,
        feature_names=np.asarray(FEATURE_NAMES, dtype="U"),
        time_seconds=time_seconds,
    )
    summary = extraction_summary_from_arrays(
        protocol,
        call_ids,
        cat_ids,
        offsets,
        values,
        feature_path,
        schema_path,
        source_hashes_verified,
        time.perf_counter() - started,
    )
    summary["idea068_reconstruction_audit"] = old_summary_consistency(
        protocol, call_ids, offsets, values
    )
    write_json(summary_path, summary)
    load_and_validate_cache(protocol, run_root)
    return summary


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = resolve_run_root(args.output_subdir)
    if args.verify_existing:
        result = load_and_validate_cache(protocol, run_root)
    else:
        result = extract(protocol, run_root)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
