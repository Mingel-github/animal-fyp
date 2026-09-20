from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/extract_idea085_acoustic_trajectories.py"
SPEC = importlib.util.spec_from_file_location("idea085_trajectory_extractor", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
idea085 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = idea085
SPEC.loader.exec_module(idea085)


def protocol() -> dict:
    return json.loads(idea085.PROTOCOL_PATH.read_text(encoding="utf-8"))


def first_audio() -> tuple[Path, np.ndarray, dict]:
    item = protocol()
    with np.load(ROOT / item["data"]["frozen_embedding_path"]) as frozen:
        call_id = str(frozen["call_ids"].astype(str)[0])
    manifest = pd.read_csv(ROOT / item["data"]["manifest_path"])
    row = manifest[manifest["filename"].astype(str) == call_id].iloc[0]
    path = ROOT / str(row["local_relpath"])
    waveform = idea085.load_audio(
        path, int(item["trajectory_extraction"]["sample_rate_hz"])
    )
    return path, waveform, item


def test_protocol_locks_identity_grid_channels_and_dependencies() -> None:
    item = protocol()
    idea085.verify_protocol(item)
    extraction = item["trajectory_extraction"]
    assert extraction["sample_rate_hz"] == 16_000
    assert extraction["frame_length"] == 1_024
    assert extraction["hop_length"] == 160
    assert extraction["fmin_hz"] == 60.0
    assert extraction["fmax_hz"] == 2_000.0
    assert extraction["center"] is True
    assert tuple(extraction["feature_names"]) == idea085.FEATURE_NAMES
    assert extraction["preserve_original_frame_grid"] is True
    assert extraction["concatenate_only_valid_f0_frames"] is False
    assert extraction["per_call_f0_normalization"] is False
    assert extraction["label_information_used"] is False


def test_real_call_preserves_10ms_grid_and_all_frame_channels() -> None:
    _, waveform, item = first_audio()
    settings = item["trajectory_extraction"]
    values, time_seconds = idea085.extract_call_trajectory(
        waveform, int(settings["sample_rate_hz"]), settings
    )
    assert values.dtype == np.float32
    assert values.ndim == 2 and values.shape[1] == 6
    assert len(values) == len(time_seconds) and len(values) > 1
    assert np.array_equal(
        time_seconds,
        np.arange(len(values), dtype=np.float64) * 0.01,
    )
    assert np.isfinite(values[:, 3:]).all()
    assert np.array_equal(np.isfinite(values[:, 0]), np.isfinite(values[:, 2]))


def test_silence_keeps_frames_and_nan_f0_without_masking_energy_spectrum() -> None:
    item = protocol()
    settings = item["trajectory_extraction"]
    waveform = np.zeros(int(settings["sample_rate_hz"] * 0.25), dtype=np.float32)
    values, time_seconds = idea085.extract_call_trajectory(
        waveform, int(settings["sample_rate_hz"]), settings
    )
    assert len(values) > 1
    assert np.isnan(values[:, 0]).all()
    assert np.isnan(values[:, 2]).all()
    assert np.isfinite(values[:, 3:]).all()
    assert np.allclose(np.diff(time_seconds), 0.01, atol=1.0e-15, rtol=0.0)


def test_new_trajectory_reconstructs_cached_idea068_statistics_for_real_call() -> None:
    _, waveform, item = first_audio()
    settings = item["trajectory_extraction"]
    values, _ = idea085.extract_call_trajectory(
        waveform, int(settings["sample_rate_hz"]), settings
    )
    reconstructed = idea085.reconstruct_idea068_features(values)
    with np.load(ROOT / item["data"]["frozen_embedding_path"]) as frozen:
        call_id = str(frozen["call_ids"].astype(str)[0])
    with np.load(ROOT / item["dependencies"]["idea068_feature_path"]) as old:
        call_ids = old["call_ids"].astype(str)
        expected = old["features"][np.where(call_ids == call_id)[0][0]].astype(np.float32)
    assert np.array_equal(np.isnan(reconstructed), np.isnan(expected))
    assert np.allclose(
        reconstructed, expected, atol=2.0e-4, rtol=2.0e-4, equal_nan=True
    )


def test_schema_explicitly_marks_time_as_non_model_and_missingness_as_nan() -> None:
    schema = idea085.build_schema(protocol())
    assert tuple(schema["feature_names"]) == idea085.FEATURE_NAMES
    assert schema["arrays"]["time_seconds"]["model_input"] is False
    assert schema["missingness"].startswith("NaN is retained")
    assert schema["grid"]["hop_seconds"] == 0.01
    assert schema["grid"]["valid_f0_frames_concatenated"] is False
    assert schema["label_information_used"] is False


def test_canonical_full_cache_when_present() -> None:
    feature_path, _, _ = idea085.cache_paths(idea085.DEFAULT_RUN_ROOT)
    if not feature_path.is_file():
        pytest.skip("canonical IDEA-085 cache has not been extracted yet")
    summary = idea085.load_and_validate_cache(protocol(), idea085.DEFAULT_RUN_ROOT)
    assert summary["calls"] == 792
    assert summary["cats"] == 111
    assert summary["source_hashes_verified"] == 792
    assert summary["idea068_reconstruction_audit"]["status"] == "PASS"
