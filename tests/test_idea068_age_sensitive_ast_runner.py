from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_idea068_age_sensitive_ast as runner  # noqa: E402


def test_protocol_locks_label_free_three_pipeline_inner_screen() -> None:
    protocol = json.loads(runner.PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["relationship"]["outer_test_access"] is False
    assert protocol["acoustic_features"]["label_information_used"] is False
    assert tuple(protocol["pipelines"]) == runner.PIPELINES
    assert tuple(protocol["acoustic_features"]["feature_names"]) == runner.FEATURE_NAMES
    assert protocol["screen"]["total_fits"] == 36
    runner.verify_protocol(protocol)


def test_feature_extractor_recovers_synthetic_pitch_and_finite_shape() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = np.sin(2.0 * np.pi * 700.0 * time).astype(np.float32)
    features = runner.extract_call_features(
        waveform,
        sample_rate,
        {
            "fmin_hz": 60.0,
            "fmax_hz": 2000.0,
            "frame_length": 1024,
            "hop_length": 160,
        },
    )
    assert features.shape == (20,)
    assert np.isfinite(features).all()
    median_f0 = np.exp(features[runner.FEATURE_NAMES.index("log_f0_median")])
    assert median_f0 == pytest.approx(700.0, abs=10.0)


def test_zero_initialized_age_residuals_match_ast_logits() -> None:
    rng = np.random.default_rng(19)
    ast = rng.normal(size=(32, 768)).astype(np.float32)
    age = rng.normal(size=(32, len(runner.FEATURE_NAMES))).astype(np.float32)
    age[0, 0] = np.nan
    outputs = []
    parameter_counts = []
    for pipeline in runner.PIPELINES:
        torch.manual_seed(17)
        model = runner.AgeResidualClassifier(
            pipeline,
            ast.mean(axis=0),
            ast.std(axis=0),
            age,
            dropout=0.44571035356880917,
            age_hidden_units=32,
        ).eval()
        outputs.append(
            model(torch.from_numpy(ast[:8]), torch.from_numpy(age[:8]))
            .detach()
            .numpy()
        )
        parameter_counts.append(model.audit()["trainable_parameters"])
    np.testing.assert_array_equal(outputs[1], outputs[0])
    np.testing.assert_array_equal(outputs[2], outputs[0])
    assert parameter_counts == [99075, 103971, 104099]
