from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_idea070_bounded_age_modulation as runner  # noqa: E402


def test_protocol_locks_parameter_matched_108_fit_screen() -> None:
    protocol = json.loads(runner.PROTOCOL_PATH.read_text(encoding="utf-8"))
    runner.verify_protocol(protocol)
    assert tuple(protocol["model"]["base_seeds"]) == runner.BASE_SEEDS
    assert protocol["model"]["total_fits"] == 108
    assert protocol["model"]["outer_test_predictions"] is False
    assert protocol["model"]["modulation_cap"] == runner.MODULATION_CAP


def test_bounded_modulation_is_identity_initialized_and_parameter_matched() -> None:
    rng = np.random.default_rng(70)
    ast = rng.normal(size=(40, 768)).astype(np.float32)
    age = rng.normal(size=(40, len(runner.idea068.FEATURE_NAMES))).astype(np.float32)
    outputs = []
    counts = []
    for pipeline in runner.PIPELINES:
        torch.manual_seed(2326)
        if pipeline == runner.PIPELINES[2]:
            model = runner.BoundedAgeModulationClassifier(
                ast.mean(axis=0), ast.std(axis=0), age, 0.44571035356880917, 32
            ).eval()
        else:
            model = runner.idea068.AgeResidualClassifier(
                pipeline, ast.mean(axis=0), ast.std(axis=0), age, 0.44571035356880917, 32
            ).eval()
        outputs.append(
            model(torch.from_numpy(ast[:8]), torch.from_numpy(age[:8]))
            .detach()
            .numpy()
        )
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    np.testing.assert_array_equal(outputs[1], outputs[0])
    np.testing.assert_array_equal(outputs[2], outputs[0])
    assert counts == [99075, 103971, 103971]


def test_modulation_scale_is_structurally_bounded() -> None:
    raw = torch.linspace(-100.0, 100.0, 1000)
    scale = runner.MODULATION_CAP * torch.tanh(raw)
    assert float(scale.min()) >= -0.25
    assert float(scale.max()) <= 0.25
    multiplier = 1.0 + scale
    assert float(multiplier.min()) >= 0.75
    assert float(multiplier.max()) <= 1.25

