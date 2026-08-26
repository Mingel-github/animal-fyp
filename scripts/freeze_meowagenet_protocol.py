"""Generate deterministic animal-disjoint outer and inner split manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "protocol" / "meowagenet_locked_v1.json"
CAT_MANIFEST_PATH = REPO_ROOT / "metadata" / "datasets" / "meowagenet" / "cat_id_manifest.csv"
OUTER_PATH = REPO_ROOT / "splits" / "meowagenet_outer_folds_v1.csv"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_nested_roles_v1.csv"
SUMMARY_PATH = REPO_ROOT / "splits" / "meowagenet_fold_summary_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cats = pd.read_csv(CAT_MANIFEST_PATH, dtype={"published_cat_id": str})
    cats = cats[cats["analysis_include"]].copy()
    cats = cats.rename(
        columns={
            "published_cat_id": "cat_id",
            "age_group_published": "age_group",
            "raw_file_rows": "call_count",
        }
    )
    cats = cats[["cat_id", "age_group", "call_count"]].sort_values("cat_id").reset_index(drop=True)
    expected_cats = config["dataset"]["expected_cats"]
    if len(cats) != expected_cats or cats["cat_id"].nunique() != expected_cats:
        raise ValueError(f"Expected {expected_cats} analysis cats, found {len(cats)}")

    split_config = config["splits"]
    outer = StratifiedKFold(
        n_splits=split_config["outer_folds"],
        shuffle=True,
        random_state=split_config["outer_seed"],
    )
    cats["outer_fold"] = -1
    for fold, (_, test_indices) in enumerate(outer.split(cats["cat_id"], cats["age_group"])):
        cats.loc[test_indices, "outer_fold"] = fold
    if (cats["outer_fold"] < 0).any():
        raise RuntimeError("At least one cat was not assigned to an outer fold")

    role_rows: list[dict[str, object]] = []
    fold_summaries: list[dict[str, object]] = []
    for outer_fold in range(split_config["outer_folds"]):
        test_mask = cats["outer_fold"] == outer_fold
        outer_train = cats[~test_mask].reset_index(drop=True)
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=split_config["inner_validation_fraction"],
            random_state=split_config["inner_seed_base"] + outer_fold,
        )
        train_indices, validation_indices = next(
            inner.split(outer_train["cat_id"], outer_train["age_group"])
        )
        train_ids = set(outer_train.loc[train_indices, "cat_id"])
        validation_ids = set(outer_train.loc[validation_indices, "cat_id"])
        test_ids = set(cats.loc[test_mask, "cat_id"])
        if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
            raise RuntimeError(f"Animal overlap detected for outer fold {outer_fold}")

        for row in cats.itertuples(index=False):
            role = "test" if row.cat_id in test_ids else "validation" if row.cat_id in validation_ids else "train"
            role_rows.append(
                {
                    "outer_fold": outer_fold,
                    "cat_id": row.cat_id,
                    "age_group": row.age_group,
                    "call_count": int(row.call_count),
                    "role": role,
                }
            )
        fold_rows = pd.DataFrame(role_rows)[lambda frame: frame["outer_fold"] == outer_fold]
        by_role = {}
        for role, group in fold_rows.groupby("role"):
            by_role[role] = {
                "cats": int(len(group)),
                "calls": int(group["call_count"].sum()),
                "class_cats": {
                    label: int((group["age_group"] == label).sum()) for label in config["labels"]
                },
            }
        fold_summaries.append({"outer_fold": outer_fold, "roles": by_role})

    OUTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    cats.to_csv(OUTER_PATH, index=False, lineterminator="\n")
    roles = pd.DataFrame(role_rows).sort_values(["outer_fold", "role", "cat_id"])
    roles.to_csv(ROLES_PATH, index=False, lineterminator="\n")
    summary = {
        "protocol_id": config["protocol_id"],
        "config_sha256": sha256(CONFIG_PATH),
        "cat_manifest_sha256": sha256(CAT_MANIFEST_PATH),
        "outer_manifest_sha256": sha256(OUTER_PATH),
        "nested_roles_sha256": sha256(ROLES_PATH),
        "cats": int(len(cats)),
        "calls": int(cats["call_count"].sum()),
        "class_cats": {
            label: int((cats["age_group"] == label).sum()) for label in config["labels"]
        },
        "folds": fold_summaries,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
