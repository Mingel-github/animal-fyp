from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "protocol" / "meowagenet_formal_v2.json"
OUTER_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_outer_folds.csv"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
RANDOM_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_random_adapter_pairs.csv"


def load_protocol() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_formal_route_and_exclusions_are_frozen() -> None:
    config = load_protocol()
    assert config["status"] == "frozen_before_formal_v2_outcomes"
    assert config["research_program"]["primary_goal"] == "IDEA-048"
    assert config["research_program"]["method_route"] == "IDEA-019"
    excluded = config["formal_matrix"]["excluded_from_formal_v2"]
    assert "IDEA-003 ordinal learning" in excluded
    assert config["execution"]["formal_outcomes_accessed_at_freeze"] is False


def test_repeated_outer_manifest_is_complete() -> None:
    config = load_protocol()
    outer = pd.read_csv(OUTER_PATH, dtype={"cat_id": str})
    expected_cats = config["dataset"]["expected_cats"]
    expected_repeats = config["splits"]["repeat_count"]
    assert len(outer) == expected_cats * expected_repeats
    for repeat in range(expected_repeats):
        repeated = outer[outer["repeat"] == repeat]
        assert repeated["cat_id"].nunique() == expected_cats
        assert set(repeated["outer_fold"]) == set(range(config["splits"]["outer_folds"]))


def test_nested_roles_are_cat_disjoint_and_complete() -> None:
    config = load_protocol()
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    expected_cats = config["dataset"]["expected_cats"]
    for repeat in range(config["splits"]["repeat_count"]):
        repeat_roles = roles[roles["repeat"] == repeat]
        test_counts = repeat_roles[repeat_roles["role"] == "test"].groupby("cat_id").size()
        assert len(test_counts) == expected_cats
        assert (test_counts == 1).all()
        for outer_fold in range(config["splits"]["outer_folds"]):
            fold = repeat_roles[repeat_roles["outer_fold"] == outer_fold]
            assert len(fold) == expected_cats
            assert fold["cat_id"].nunique() == expected_cats
            assert set(fold["role"]) == {"train", "validation", "test"}


def test_random_adapter_controls_are_precomputed_and_valid() -> None:
    config = load_protocol()
    random_pairs = pd.read_csv(RANDOM_PATH)
    random_config = next(
        row
        for row in config["formal_matrix"]["required_diagnostics"]
        if row["pipeline_id"] == "ast_adapter_random_controls"
    )
    expected_rows = (
        config["splits"]["repeat_count"]
        * config["splits"]["outer_folds"]
        * random_config["draws_per_repeat_fold"]
    )
    assert len(random_pairs) == expected_rows
    assert random_pairs["layer_1_one_based"].between(1, 12).all()
    assert random_pairs["layer_2_one_based"].between(1, 12).all()
    assert (random_pairs["layer_1_one_based"] < random_pairs["layer_2_one_based"]).all()
    group_sizes = random_pairs.groupby(["repeat", "outer_fold"]).size()
    assert (group_sizes == random_config["draws_per_repeat_fold"]).all()
