from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_idea088_original_roles.py"
ARTIFACT = (
    ROOT
    / "runs"
    / "meowagenet_idea088_original_style_clean_v1"
    / "roles"
    / "original_clean_roles.json"
)
REPORT = ROOT / "reports" / "79_IDEA-088_original_roles_and_design_audit.md"


def load_module():
    spec = importlib.util.spec_from_file_location("idea088_roles", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


idea088 = load_module()


def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_exactly_matches_direct_log_parse() -> None:
    expected = idea088.build_payload()
    assert ARTIFACT.is_file()
    assert artifact() == expected
    assert expected["status"] == "frozen_roles_only_training_not_authorized"
    assert expected["schema_version"] == 1.0
    assert expected["protocol_id"] == "meowagenet-idea088-original-style-clean-v1"
    assert expected["source_commit"] == "3d02295bef1500d2b2500a124596f77010181391"
    assert expected["source_prediction_rows"] == 937
    assert expected["split_seeds"] == [7270, 860, 5390, 5191, 5734]
    assert expected["forced_training_cat_ids"] == ["000A", "046A"]
    assert len(expected["cells"]) == 20
    assert [cell["fold"] for cell in expected["cells"]] == [0, 1, 2, 3] * 5


def test_cells_preserve_logged_post_swap_roles_and_delete_only_049a() -> None:
    value = artifact()
    parsed = idea088.parse_logged_cells(idea088.discover_original_log())
    clean_ids = set()
    for cell, source in zip(value["cells"], parsed, strict=True):
        train = set(cell["train_cat_ids"])
        test = set(cell["test_cat_ids"])
        pre_train = set(cell["pre_swap_train_cat_ids"])
        pre_test = set(cell["pre_swap_test_cat_ids"])

        assert not train & test
        assert len(train | test) == 111
        assert not pre_train & pre_test
        assert len(pre_train | pre_test) == 111
        assert train == set(source["train_cat_ids"]) - {"049A"}
        assert test == set(source["test_cat_ids"]) - {"049A"}
        assert pre_train == set(source["pre_swap_train_cat_ids"]) - {"049A"}
        assert pre_test == set(source["pre_swap_test_cat_ids"]) - {"049A"}
        for field in (
            "train_cat_ids",
            "test_cat_ids",
            "pre_swap_train_cat_ids",
            "pre_swap_test_cat_ids",
        ):
            assert cell[field] == sorted(set(cell[field]))
            assert "049A" not in cell[field]
        assert {"000A", "046A"}.issubset(train)
        assert not {"000A", "046A"} & test
        swapped_in = {swap["forced_cat_id"] for swap in cell["swaps"]}
        swapped_out = {swap["replacement_cat_id"] for swap in cell["swaps"]}
        assert train - pre_train == swapped_in
        assert pre_train - train == swapped_out
        assert test - pre_test == swapped_out
        assert pre_test - test == swapped_in
        clean_ids |= train | test
    assert len(clean_ids) == 111
    assert "049A" not in clean_ids


def test_each_cell_has_full_clean_coverage_and_all_classes() -> None:
    value = artifact()
    assert value["dataset_counts"] == {
        "official": {"cats": 112, "calls": 793, "vggish_rows": 937},
        "clean": {
            "cats": 111,
            "calls": 792,
            "vggish_rows": 936,
            "class_cats": {"kitten": 15, "adult": 62, "senior": 34},
        },
    }
    assert value["classifier_labels"]["clean_call_counts"] == {
        "kitten": 134,
        "adult": 405,
        "senior": 253,
    }
    assert value["vggish_input"]["feature_dimensions"] == 129
    assert value["vggish_input"]["feature_columns"][-1] == "mean_freq"
    for cell in value["audit"]["per_cell"]:
        train = cell["train"]
        test = cell["test"]
        assert train["cats"] + test["cats"] == 111
        assert train["calls"] + test["calls"] == 792
        assert train["vggish_rows"] + test["vggish_rows"] == 936
        assert all(train["class_cats"][label] > 0 for label in idea088.LABELS)
        assert all(test["class_cats"][label] > 0 for label in idea088.LABELS)


def test_per_seed_counts_record_non_oof_original_swap_behavior() -> None:
    value = artifact()
    expected_repeated = {
        "7270": {"026C": 2, "110A": 2},
        "860": {"087A": 2, "110A": 2},
        "5390": {"022A": 2, "048A": 2},
        "5191": {"010A": 2, "045A": 2},
        "5734": {"002A": 2, "111A": 2},
    }
    for seed, summary in value["audit"]["per_seed_test_multiplicity"].items():
        counts = summary["test_count_by_cat"]
        assert len(counts) == 111
        assert sum(counts.values()) == 111
        assert sum(count > 0 for count in counts.values()) == 109
        assert summary["never_tested_cat_ids"] == ["000A", "046A"]
        assert summary["repeated_test_cat_ids"] == expected_repeated[seed]
        assert summary["complete_oof_partition"] is False

        reconstructed = Counter()
        for cell in value["cells"]:
            if str(cell["split_seed"]) == seed:
                reconstructed.update(cell["test_cat_ids"])
        assert counts == {cat_id: reconstructed[cat_id] for cat_id in counts}


def test_source_hashes_are_bound_and_split_regeneration_is_absent() -> None:
    value = artifact()
    assert len(value["source_evidence"]) == 4
    for source in value["source_evidence"]:
        assert len(source["sha256"]) == 64
        assert source["size_bytes"] > 0
    script_text = SCRIPT.read_text(encoding="utf-8")
    forbidden = ("StratifiedGroupKFold", "StratifiedKFold", "train_test_split")
    assert not any(token in script_text for token in forbidden)


def test_frozen_writer_verifies_and_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "idea088.json"
    assert idea088.write_or_verify(path, "same\n") == "written"
    assert idea088.write_or_verify(path, "same\n") == "verified"
    with pytest.raises(RuntimeError, match="Do not overwrite"):
        idea088.write_or_verify(path, "different\n")


def test_design_audit_report_records_metric_and_training_gates() -> None:
    text = REPORT.read_text(encoding="utf-8")
    required = (
        "049A",
        "000A",
        "046A",
        "并非完整 OOF",
        "936",
        "792",
        "cat probability-mean macro-F1",
        "129",
        "adult",
        "kitten",
        "senior",
        "60 fits",
        "未获授权",
    )
    assert all(token in text for token in required)
