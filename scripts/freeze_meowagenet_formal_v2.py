"""Freeze and verify deterministic repeated animal-disjoint formal-v2 splits."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "protocol" / "meowagenet_formal_v2.json"
CAT_MANIFEST_PATH = (
    REPO_ROOT / "metadata" / "datasets" / "meowagenet" / "cat_id_manifest.csv"
)
OUTER_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_outer_folds.csv"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
RANDOM_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_random_adapter_pairs.csv"
SUMMARY_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_split_summary.json"
FREEZE_RECORD_PATH = (
    REPO_ROOT / "metadata" / "experiments" / "meowagenet_formal_v2_freeze.json"
)
REPORT_PATH = REPO_ROOT / "reports" / "09_formal_protocol_v2_freeze.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_or_verify(path: Path, content: str) -> None:
    encoded = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(
                f"Frozen artifact differs: {path}. Create a new protocol version instead of overwriting it."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def csv_text(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n")


def load_analysis_cats(config: dict[str, object]) -> pd.DataFrame:
    cats = pd.read_csv(CAT_MANIFEST_PATH, dtype={"published_cat_id": str})
    cats = cats[cats["analysis_include"]].copy()
    cats = cats.rename(
        columns={
            "published_cat_id": "cat_id",
            "age_group_published": "age_group",
            "raw_file_rows": "call_count",
        }
    )
    cats = cats[["cat_id", "age_group", "call_count"]]
    cats = cats.sort_values("cat_id").reset_index(drop=True)
    expected_cats = int(config["dataset"]["expected_cats"])
    expected_calls = int(config["dataset"]["expected_calls"])
    if len(cats) != expected_cats or cats["cat_id"].nunique() != expected_cats:
        raise ValueError(f"Expected {expected_cats} analysis cats, found {len(cats)}")
    if int(cats["call_count"].sum()) != expected_calls:
        raise ValueError(
            f"Expected {expected_calls} analysis calls, found {int(cats['call_count'].sum())}"
        )
    if sha256(CAT_MANIFEST_PATH) != config["dataset"]["cat_manifest_sha256"]:
        raise ValueError("Cat manifest checksum differs from formal-v2 protocol")
    return cats


def make_split_frames(
    config: dict[str, object], cats: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    split_config = config["splits"]
    outer_rows: list[dict[str, object]] = []
    role_rows: list[dict[str, object]] = []
    repeat_summaries: list[dict[str, object]] = []

    for repeat_index, outer_seed in enumerate(split_config["outer_split_seeds"]):
        repeated = cats.copy()
        splitter = StratifiedKFold(
            n_splits=int(split_config["outer_folds"]),
            shuffle=True,
            random_state=int(outer_seed),
        )
        repeated["outer_fold"] = -1
        for outer_fold, (_, test_indices) in enumerate(
            splitter.split(repeated["cat_id"], repeated["age_group"])
        ):
            repeated.loc[test_indices, "outer_fold"] = outer_fold
        if (repeated["outer_fold"] < 0).any():
            raise RuntimeError(f"Unassigned cat in repeat {repeat_index}")

        for row in repeated.itertuples(index=False):
            outer_rows.append(
                {
                    "repeat": repeat_index,
                    "outer_seed": int(outer_seed),
                    "cat_id": row.cat_id,
                    "age_group": row.age_group,
                    "call_count": int(row.call_count),
                    "outer_fold": int(row.outer_fold),
                }
            )

        fold_summaries: list[dict[str, object]] = []
        for outer_fold in range(int(split_config["outer_folds"])):
            test_mask = repeated["outer_fold"] == outer_fold
            outer_train = repeated[~test_mask].reset_index(drop=True)
            inner_seed = int(outer_seed) + 1000 + outer_fold
            inner = StratifiedShuffleSplit(
                n_splits=1,
                test_size=float(split_config["inner_validation_fraction"]),
                random_state=inner_seed,
            )
            train_indices, validation_indices = next(
                inner.split(outer_train["cat_id"], outer_train["age_group"])
            )
            train_ids = set(outer_train.loc[train_indices, "cat_id"])
            validation_ids = set(outer_train.loc[validation_indices, "cat_id"])
            test_ids = set(repeated.loc[test_mask, "cat_id"])
            if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
                raise RuntimeError(
                    f"Animal overlap in repeat {repeat_index}, fold {outer_fold}"
                )

            current_rows: list[dict[str, object]] = []
            for row in repeated.itertuples(index=False):
                if row.cat_id in test_ids:
                    role = "test"
                elif row.cat_id in validation_ids:
                    role = "validation"
                else:
                    role = "train"
                current_rows.append(
                    {
                        "repeat": repeat_index,
                        "outer_seed": int(outer_seed),
                        "inner_seed": inner_seed,
                        "outer_fold": outer_fold,
                        "cat_id": row.cat_id,
                        "age_group": row.age_group,
                        "call_count": int(row.call_count),
                        "role": role,
                    }
                )
            role_rows.extend(current_rows)
            fold_frame = pd.DataFrame(current_rows)
            by_role: dict[str, object] = {}
            for role, group in fold_frame.groupby("role"):
                by_role[role] = {
                    "cats": int(len(group)),
                    "calls": int(group["call_count"].sum()),
                    "class_cats": {
                        label: int((group["age_group"] == label).sum())
                        for label in config["dataset"]["class_cats"]
                    },
                }
            fold_summaries.append(
                {
                    "outer_fold": outer_fold,
                    "outer_seed": int(outer_seed),
                    "inner_seed": inner_seed,
                    "roles": by_role,
                }
            )
        repeat_summaries.append(
            {
                "repeat": repeat_index,
                "outer_seed": int(outer_seed),
                "folds": fold_summaries,
            }
        )

    outer = pd.DataFrame(outer_rows).sort_values(["repeat", "cat_id"])
    roles = pd.DataFrame(role_rows).sort_values(
        ["repeat", "outer_fold", "role", "cat_id"]
    )
    return outer, roles, repeat_summaries


def make_random_controls(config: dict[str, object]) -> pd.DataFrame:
    random_config = next(
        row
        for row in config["formal_matrix"]["required_diagnostics"]
        if row["pipeline_id"] == "ast_adapter_random_controls"
    )
    all_pairs = list(itertools.combinations(range(1, 13), 2))
    rows: list[dict[str, int]] = []
    for repeat_index in range(int(config["splits"]["repeat_count"])):
        for outer_fold in range(int(config["splits"]["outer_folds"])):
            seed = 880301 + 100 * repeat_index + outer_fold
            sampled = random.Random(seed).sample(
                all_pairs, int(random_config["draws_per_repeat_fold"])
            )
            for draw, (layer_1, layer_2) in enumerate(sampled):
                rows.append(
                    {
                        "repeat": repeat_index,
                        "outer_fold": outer_fold,
                        "draw": draw,
                        "placement_seed": seed,
                        "layer_1_one_based": layer_1,
                        "layer_2_one_based": layer_2,
                    }
                )
    return pd.DataFrame(rows).sort_values(["repeat", "outer_fold", "draw"])


def validate_frames(
    config: dict[str, object], outer: pd.DataFrame, roles: pd.DataFrame
) -> None:
    expected_cats = int(config["dataset"]["expected_cats"])
    repeat_count = int(config["splits"]["repeat_count"])
    outer_folds = int(config["splits"]["outer_folds"])
    if len(outer) != expected_cats * repeat_count:
        raise RuntimeError("Unexpected formal-v2 outer manifest row count")
    if len(roles) != expected_cats * repeat_count * outer_folds:
        raise RuntimeError("Unexpected formal-v2 nested-role row count")
    for repeat_index in range(repeat_count):
        repeated = outer[outer["repeat"] == repeat_index]
        if repeated["cat_id"].nunique() != expected_cats:
            raise RuntimeError(f"Duplicate or missing cats in repeat {repeat_index}")
        test_roles = roles[
            (roles["repeat"] == repeat_index) & (roles["role"] == "test")
        ]
        counts = test_roles.groupby("cat_id").size()
        if len(counts) != expected_cats or not (counts == 1).all():
            raise RuntimeError(
                f"Each cat must appear in one test fold in repeat {repeat_index}"
            )


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["protocol_id"] != "meowagenet-formal-v2":
        raise ValueError("Unexpected protocol ID")
    if config["status"] != "frozen_before_formal_v2_outcomes":
        raise ValueError("Protocol is not in the expected frozen state")
    cats = load_analysis_cats(config)
    outer, roles, repeat_summaries = make_split_frames(config, cats)
    random_controls = make_random_controls(config)
    validate_frames(config, outer, roles)

    write_or_verify(OUTER_PATH, csv_text(outer))
    write_or_verify(ROLES_PATH, csv_text(roles))
    write_or_verify(RANDOM_PATH, csv_text(random_controls))

    summary = {
        "schema_version": 2,
        "protocol_id": config["protocol_id"],
        "status": "frozen",
        "config_sha256": sha256(CONFIG_PATH),
        "cat_manifest_sha256": sha256(CAT_MANIFEST_PATH),
        "outer_manifest_sha256": sha256(OUTER_PATH),
        "nested_roles_sha256": sha256(ROLES_PATH),
        "random_adapter_pairs_sha256": sha256(RANDOM_PATH),
        "cats": int(len(cats)),
        "calls": int(cats["call_count"].sum()),
        "repeat_count": int(config["splits"]["repeat_count"]),
        "outer_folds_per_repeat": int(config["splits"]["outer_folds"]),
        "model_seeds_per_pipeline": len(config["model_randomness"]["base_model_seeds"]),
        "complete_oof_evaluations_per_pipeline": int(
            config["model_randomness"]["complete_oof_evaluations_per_pipeline"]
        ),
        "class_cats": {
            label: int((cats["age_group"] == label).sum())
            for label in config["dataset"]["class_cats"]
        },
        "repeats": repeat_summaries,
    }
    write_or_verify(SUMMARY_PATH, json.dumps(summary, indent=2) + "\n")

    freeze_record = {
        "schema_version": 2,
        "decision_id": "meowagenet-formal-v2-freeze",
        "date": config["frozen_date"],
        "status": "frozen_unexecuted",
        "route": {
            "research_goal": "IDEA-048",
            "method_route": "IDEA-019 low-parameter AST adaptation",
            "primary_candidate": "ast_probe_guided_adapter",
            "idea003": "excluded_from_formal_v2",
        },
        "prior_access": config["prior_access"],
        "artifacts": {
            "protocol": {
                "path": str(CONFIG_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": sha256(CONFIG_PATH),
            },
            "split_summary": {
                "path": str(SUMMARY_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": sha256(SUMMARY_PATH),
            },
            "outer_folds": {
                "path": str(OUTER_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": sha256(OUTER_PATH),
            },
            "nested_roles": {
                "path": str(ROLES_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": sha256(ROLES_PATH),
            },
            "random_adapter_pairs": {
                "path": str(RANDOM_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": sha256(RANDOM_PATH),
            },
            "freeze_script": {
                "path": str(Path(__file__).relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": sha256(Path(__file__)),
            },
            "protocol_report": str(REPORT_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        },
        "formal_outcomes_accessed": False,
        "execution_location": "external experiment environment",
        "change_policy": "Do not overwrite v2. Record deviations or create a later protocol version.",
    }
    write_or_verify(FREEZE_RECORD_PATH, json.dumps(freeze_record, indent=2) + "\n")
    print(json.dumps(freeze_record, indent=2))


if __name__ == "__main__":
    main()
