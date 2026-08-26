from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_outer_fold_manifest_has_one_assignment_per_analysis_cat() -> None:
    outer = pd.read_csv(REPO_ROOT / "splits" / "meowagenet_outer_folds_v1.csv")
    assert len(outer) == 111
    assert outer["cat_id"].nunique() == 111
    assert set(outer["outer_fold"]) == {0, 1, 2, 3}
    assert int(outer["call_count"].sum()) == 792
    for _, fold in outer.groupby("outer_fold"):
        assert set(fold["age_group"]) == {"kitten", "adult", "senior"}


def test_nested_roles_are_animal_disjoint_and_complete() -> None:
    roles = pd.read_csv(REPO_ROOT / "splits" / "meowagenet_nested_roles_v1.csv")
    for outer_fold, fold in roles.groupby("outer_fold"):
        assert len(fold) == 111, outer_fold
        assert fold["cat_id"].nunique() == 111
        assert set(fold["role"]) == {"train", "validation", "test"}
        assert int(fold["call_count"].sum()) == 792
        assert len(fold[fold["role"] == "test"]) in {27, 28}
