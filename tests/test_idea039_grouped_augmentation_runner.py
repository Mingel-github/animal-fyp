from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_meowagenet_idea039_grouped_augmentation as runner  # noqa: E402


def protocol():
    return json.loads(runner.PROTOCOL_PATH.read_text(encoding="utf-8"))


def synthetic_segment(valid_frames: int = 32) -> np.ndarray:
    rng = np.random.default_rng(123)
    segment = np.full((128, 128), -1.2776, dtype=np.float32)
    segment[:valid_frames] = rng.normal(size=(valid_frames, 128)).astype(np.float32)
    return segment


def test_protocol_has_finite_grouped_candidate_matrix() -> None:
    data = protocol()
    runner.verify_protocol(data)
    assert data["splits"]["independence_unit"] == "cat_id"
    assert data["splits"]["augmentation_roles"] == ["train"]
    assert data["splits"]["identity_roles"] == ["validation", "test"]
    assert len(data["augmentation"]["candidate_policies"]) == 4
    assert data["inner_selection"]["logical_candidate_fits"] == 144
    assert data["initial_evaluation"]["outer_fits"] == 48


def test_identity_is_bitwise_and_stochastic_policies_are_deterministic() -> None:
    data = protocol()
    segment = synthetic_segment()
    identity = data["augmentation"]["candidate_policies"]["P0_identity"]
    output, audit = runner.augment_segment(segment, identity, 39001, 1, 7, 0)
    assert np.array_equal(output, segment)
    assert audit["changed_values"] == 0

    for policy_id in (
        "P1_specaugment_light",
        "P2_gain_noise_light",
        "P3_shift_combo_light",
    ):
        policy = data["augmentation"]["candidate_policies"][policy_id]
        first, _ = runner.augment_segment(segment, policy, 39001, 2, 7, 0)
        second, _ = runner.augment_segment(segment, policy, 39001, 2, 7, 0)
        assert np.array_equal(first, second)


def test_every_policy_preserves_padding_and_shape() -> None:
    data = protocol()
    segment = synthetic_segment(valid_frames=24)
    padding = segment[24:].copy()
    for policy in data["augmentation"]["candidate_policies"].values():
        output, _ = runner.augment_segment(segment, policy, 39003, 4, 18, 0)
        assert output.shape == (128, 128)
        assert output.dtype == np.float32
        assert np.array_equal(output[24:], padding)


def test_policy_ranking_uses_stream_mean_then_ce_and_severity() -> None:
    policies = protocol()["augmentation"]["candidate_policies"]

    def row(policy_id: str, macro_f1: float, ce: float, epoch: int = 5):
        return {
            "policy_id": policy_id,
            "best_epoch": epoch,
            "best_validation_animal_cross_entropy": ce,
            "best_validation_animal_metrics": {"macro_f1": macro_f1},
        }

    rows = {
        "P0_identity": [row("P0_identity", 0.70, 0.8)] * 3,
        "P1_specaugment_light": [row("P1_specaugment_light", 0.72, 0.7)] * 3,
        "P2_gain_noise_light": [row("P2_gain_noise_light", 0.72, 0.7)] * 3,
        "P3_shift_combo_light": [row("P3_shift_combo_light", 0.69, 0.6)] * 3,
    }
    ranking = runner.rank_policies(rows, policies)
    assert [item["policy_id"] for item in ranking] == [
        "P2_gain_noise_light",
        "P1_specaugment_light",
        "P0_identity",
        "P3_shift_combo_light",
    ]


def test_fold_policy_mapping_keeps_fixed_and_nested_claims_separate() -> None:
    data = protocol()
    lock = {
        "selected_policy_id": "P2_gain_noise_light",
        "candidate_summaries": {
            "P0_identity": {"median_best_epoch": 4},
            "P1_specaugment_light": {"median_best_epoch": 6},
            "P2_gain_noise_light": {"median_best_epoch": 8},
        },
    }
    assert runner.policy_for_pipeline(runner.PIPELINES[1], data, lock) == (
        "P0_identity",
        4,
    )
    assert runner.policy_for_pipeline(runner.PIPELINES[2], data, lock) == (
        "P1_specaugment_light",
        6,
    )
    assert runner.policy_for_pipeline(runner.PIPELINES[3], data, lock) == (
        "P2_gain_noise_light",
        8,
    )
