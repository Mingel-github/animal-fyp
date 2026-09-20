from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_idea088_original_style_clean_results.py"
PROTOCOL_PATH = ROOT / "configs/protocol/meowagenet_idea088_original_style_clean_v1.json"


def load_verifier():
    spec = importlib.util.spec_from_file_location("idea088_independent_verifier", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


verifier = load_verifier()


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_verifier_does_not_import_training_runner() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import run_meowagenet_idea088_original_style_clean" not in source
    assert "from run_meowagenet_idea088_original_style_clean" not in source


def test_frozen_locks_inputs_and_roles_are_independently_valid() -> None:
    value = protocol()
    locks = verifier.verify_locks(value)
    inputs = verifier.load_inputs(value)
    roles = verifier.read_json(ROOT / value["splits"]["roles_path"])
    cells, multiplicity = verifier.validate_roles(
        value, roles, set(inputs["ast"]["cat_ids"].astype(str))
    )
    assert locks["dependency_mismatches"] == 0
    assert locks["checkpoint_before_prediction_static"] is True
    assert len(cells) == 20
    assert inputs["vggish"]["features"].shape == (936, 129)
    assert inputs["ast"]["embeddings"].shape == (792, 768)
    assert all(row["unique_test_cats"] == 109 for row in multiplicity.values())
    assert all(
        row["omitted_forced_training_cats"] == ["000A", "046A"]
        for row in multiplicity.values()
    )


def test_class_weight_epoch_order_and_early_stop_formulas() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 2], dtype=np.int64)
    np.testing.assert_allclose(verifier.class_weights(labels), [0.5, 2.0, 2.0])
    indices = np.arange(20, dtype=np.int64)
    left = verifier.epoch_order(indices, 1_007_270, 1)
    right = verifier.epoch_order(indices, 1_007_270, 1)
    np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(np.sort(left), indices)
    replay = verifier.replay_early_stopping(
        [
            {"epoch": 1, "training_loss": 1.0},
            {"epoch": 2, "training_loss": 0.9995},
            {"epoch": 3, "training_loss": 0.9994},
        ]
    )
    assert replay["best_epoch"] == 1
    assert replay["best_training_loss"] == 1.0
    assert replay["stale_epochs"] == 2
    assert replay["stop_epoch"] is None


def test_unit_to_cat_probability_mean_and_metrics_are_independent() -> None:
    units = pd.DataFrame(
        {
            "unit_index": [0, 1, 2, 3],
            "unit_id": ["a0", "a1", "b0", "c0"],
            "cat_id": ["a", "a", "b", "c"],
            "true_label": [0, 0, 1, 2],
            "prob_kitten": [0.9, 0.1, 0.2, 0.1],
            "prob_adult": [0.05, 0.8, 0.7, 0.1],
            "prob_senior": [0.05, 0.1, 0.1, 0.8],
        }
    )
    cats = verifier.units_to_cats(units)
    np.testing.assert_allclose(
        cats.loc[cats["cat_id"] == "a", list(verifier.PROBABILITY_COLUMNS)].iloc[0],
        [0.5, 0.425, 0.075],
    )
    metrics = verifier.metric_bundle(
        cats["true_label"].to_numpy(np.int64),
        cats[list(verifier.PROBABILITY_COLUMNS)].to_numpy(float),
    )
    assert metrics["n"] == 3
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)


def test_historical_protection_snapshot_is_unchanged() -> None:
    result = verifier.protection_snapshot()
    assert result == {
        "file_count": 3965,
        "combined_sha256": verifier.EXPECTED_HISTORICAL_SHA256,
        "matches_pre_execution_snapshot": True,
    }


def test_full_result_matrix_rebuilds_from_raw_predictions() -> None:
    result = verifier.verify_full()
    assert result["status"] == "PASS_FULL"
    assert result["fits_verified"] == 60
    assert result["fit_artifact_hashes_verified"] == 360
    assert result["paired_A0_C1_common_initial_states"] == 20
    assert result["paired_A0_C1_order_prefixes"] == 20
    assert result["maximum_fit_metric_abs_difference"] < 1.0e-8
    assert result["maximum_aggregate_abs_difference"] < 2.0e-9
    rebuilt = result["rebuilt_results"]
    assert rebuilt["pipelines"]["vggish_mlp"]["public_cat_probability_mean"][
        "macro_f1"
    ]["mean_over_five_seed_means"] == pytest.approx(0.6989719842024775)
    assert rebuilt["pipelines"]["A0_ast_only"]["public_cat_probability_mean"][
        "macro_f1"
    ]["mean_over_five_seed_means"] == pytest.approx(0.7094176232277063)
    assert rebuilt["pipelines"]["C1_bounded_wide_additive"][
        "public_cat_probability_mean"
    ]["macro_f1"]["mean_over_five_seed_means"] == pytest.approx(0.7222633650086638)
    paired = rebuilt["paired_C1_minus_A0"]
    assert paired["public_cat_macro_f1"]["mean"] == pytest.approx(
        0.012845741780957515
    )
    assert paired["native_call_macro_f1"]["mean"] == pytest.approx(
        -0.00023356005093582433
    )
    recalls = result["independent_class_recall_summary"]
    assert recalls["C1_bounded_wide_additive"]["public_cat_probability_mean"][
        "kitten"
    ]["mean_over_five_seed_means"] == pytest.approx(0.8225)
    assert recalls["C1_bounded_wide_additive"]["public_cat_probability_mean"][
        "adult"
    ]["mean_over_five_seed_means"] == pytest.approx(0.7865319383488888)
    assert recalls["C1_bounded_wide_additive"]["public_cat_probability_mean"][
        "senior"
    ]["mean_over_five_seed_means"] == pytest.approx(0.6007070707070707)
