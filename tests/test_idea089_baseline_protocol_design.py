"""Independent tests for the IDEA-089 design audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_idea089_baseline_protocol_design.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("idea089_design_audit", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load IDEA-089 design audit")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audited():
    module = load_audit_module()
    return module, module.run_audit()


def test_frozen_age_boundaries_and_label_order(audited) -> None:
    module, result = audited
    assert module.age_group(0.0) == "kitten"
    assert module.age_group(0.499999) == "kitten"
    assert module.age_group(0.5) == "adult"
    assert module.age_group(9.999999) == "adult"
    assert module.age_group(10.0) == "senior"
    assert module.age_group(19.999999) == "senior"
    with pytest.raises(ValueError):
        module.age_group(20.0)
    assert result["clean_data"]["unified_label_order"] == [
        "kitten",
        "adult",
        "senior",
    ]


def test_role_cells_and_complete_oof_contract(audited) -> None:
    _, result = audited
    roles = result["roles"]
    assert roles["selected_repeats"] == [0, 1, 2]
    assert roles["selected_rows"] == 3 * 4 * 111
    assert len(roles["cells"]) == 12
    for cell in roles["cells"]:
        counts = {role: values["cats"] for role, values in cell["roles"].items()}
        assert counts in (
            {"train": 66, "validation": 17, "test": 28},
            {"train": 67, "validation": 17, "test": 27},
        )
        assert sum(values["calls"] for values in cell["roles"].values()) == 792


def test_data_vgg_and_cache_identity_contract(audited) -> None:
    _, result = audited
    assert result["clean_data"]["cat_labels"] == {
        "adult": 62,
        "kitten": 15,
        "senior": 34,
    }
    assert result["clean_data"]["call_labels"] == {
        "adult": 405,
        "kitten": 134,
        "senior": 253,
    }
    assert result["vgg"]["clean_rows"] == 936
    assert result["vgg"]["cat_label_mismatches_vs_call_manifest"] == 0
    assert result["caches"]["ast_shape"] == [792, 768]
    assert result["caches"]["acoustic_shape"] == [792, 20]
    assert result["caches"]["call_order_exactly_aligned"] is True
    assert result["caches"]["acoustic_missing_values_total"] == 340


def test_parameter_and_fit_budget_arithmetic(audited) -> None:
    module, result = audited
    assert module.trainable_parameter_counts() == {
        "A0": 99075,
        "direct_concat": 101635,
        "U1": 108143,
        "C1": 108143,
    }
    arithmetic = result["frozen_design_arithmetic"]
    assert arithmetic["selection_fits"] == 216
    assert arithmetic["outer_refit_fits"] == 216
    assert arithmetic["total_real_fits"] == 432
    assert arithmetic["complete_oof_scores_per_model"] == 9
    assert arithmetic["prediction_occurrences_per_model"] == 999
    assert arithmetic["distinct_animals"] == 111


def test_historical_reuse_and_unit_boundaries_are_explicit(audited) -> None:
    _, result = audited
    history = result["historical_source_evidence"]
    assert history["formal_v2_1_vgg_dimensions"] == 128
    assert history["idea088_author_style_vgg_dimensions"] == 129
    assert "cannot populate" in history["reuse_boundary"]
    assert "no reliable row-to-call mapping" in result["vgg"]["unit"]
    assert "only 111 distinct animals" in result["claim_boundaries"]["independence"]
    assert result["status"] == "PASS_DESIGN_PREFLIGHT_ONLY_NO_TRAINING_AUTHORIZATION"
