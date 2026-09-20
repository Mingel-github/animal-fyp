"""Freeze IDEA-088 roles by parsing the archived original-style run log.

This script deliberately does not regenerate any folds.  It parses the actual
post-swap train/test cat arrays emitted by the 2026-08-25 reproduction, then
projects each role cell onto the clean 111-cat view by deleting only 049A.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_LOG_RELATIVE_PATH = Path(
    "runs/reproduction/2026-08-25-original/categorical/"
    "vggish-categorical-logs.txt"
)
DATA_MANIFEST_PATH = (
    REPO_ROOT / "metadata" / "datasets" / "meowagenet" / "data_manifest.csv"
)
CAT_MANIFEST_PATH = (
    REPO_ROOT / "metadata" / "datasets" / "meowagenet" / "cat_id_manifest.csv"
)
VGGISH_CSV_PATH = (
    REPO_ROOT
    / "data"
    / "meowagenet"
    / "official-3d02295bef15"
    / "embeddings"
    / "vggish_looped_embeddings.csv"
)
OUTPUT_PATH = (
    REPO_ROOT
    / "runs"
    / "meowagenet_idea088_original_style_clean_v1"
    / "roles"
    / "original_clean_roles.json"
)

SEEDS = (7270, 860, 5390, 5191, 5734)
FOLDS_PER_SEED = 4
EXCLUDED_PUBLISHED_CAT_IDS = ("049A",)
FORCED_TRAIN_CAT_IDS = ("000A", "046A")
LABELS = ("kitten", "adult", "senior")
EXPECTED_CLEAN_CATS = 111
EXPECTED_CLEAN_CALLS = 792
EXPECTED_CLEAN_VGGISH_ROWS = 936
SOURCE_COMMIT = "3d02295bef1500d2b2500a124596f77010181391"

OUTER_FOLD_PATTERN = re.compile(r"^outer_fold\s+([1-4])\s*$", re.MULTILINE)
PRE_SWAP_PATTERN = re.compile(
    r"^Unique Training/Validation Group IDs:\s*$\s*"
    r"\[(?P<train>.*?)\]\s*"
    r"^Unique Test Group IDs:\s*$\s*"
    r"\[(?P<test>.*?)\]",
    flags=re.MULTILINE | re.DOTALL,
)
MOVE_PATTERN = re.compile(
    r"^Moved to Training/Validation Set:\s*$\s*(?P<moved_train>.*?)\s*"
    r"^Removed from Training/Validation Set:\s*$\s*(?P<removed_train>.*?)\s*"
    r"^Moved to Test Set:\s*$\s*(?P<moved_test>.*?)\s*"
    r"^Removed from Test Set\s*$\s*(?P<removed_test>.*?)\s*"
    r"^AFTER SWAP - Unique Training/Validation Group IDs:",
    flags=re.MULTILINE | re.DOTALL,
)
POST_SWAP_PATTERN = re.compile(
    r"^AFTER SWAP - Unique Training/Validation Group IDs:\s*$\s*"
    r"\[(?P<train>.*?)\]\s*"
    r"^AFTER SWAP - Unique Test Group IDs:\s*$\s*"
    r"\[(?P<test>.*?)\]",
    flags=re.MULTILINE | re.DOTALL,
)
CAT_ID_PATTERN = re.compile(r"'([^']+)'")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def is_true(value: str) -> bool:
    return value.strip().lower() == "true"


def age_group_from_age(value: str | float) -> str:
    age = float(value)
    if 0.0 <= age < 0.5:
        return "kitten"
    if 0.5 <= age < 10.0:
        return "adult"
    if 10.0 <= age < 20.0:
        return "senior"
    raise ValueError(f"Age is outside the original categorical ranges: {age}")


def discover_original_log() -> Path:
    candidates = [
        REPO_ROOT / ORIGINAL_LOG_RELATIVE_PATH,
        REPO_ROOT.parent.parent / "animal-fyp" / ORIGINAL_LOG_RELATIVE_PATH,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find the archived original reproduction log. Checked:\n"
        f"{rendered}\nPass --source-log explicitly."
    )


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def sorted_cat_ids(value: str) -> list[str]:
    ids = CAT_ID_PATTERN.findall(value)
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate cat IDs in logged set: {ids}")
    return sorted(ids)


def parse_logged_cells(log_path: Path) -> list[dict[str, object]]:
    text = log_path.read_text(encoding="utf-8")
    matches = list(OUTER_FOLD_PATTERN.finditer(text))
    expected = len(SEEDS) * FOLDS_PER_SEED
    if len(matches) != expected:
        raise ValueError(f"Expected {expected} logged role cells, found {len(matches)}")

    cells: list[dict[str, object]] = []
    for source_order, match in enumerate(matches):
        block_start = match.start()
        block_end = matches[source_order + 1].start() if source_order + 1 < len(matches) else len(text)
        block = text[block_start:block_end]
        pre = PRE_SWAP_PATTERN.search(block)
        moves = MOVE_PATTERN.search(block)
        post = POST_SWAP_PATTERN.search(block)
        if pre is None or moves is None or post is None:
            raise ValueError(f"Incomplete role evidence in source cell {source_order}")
        seed_index, expected_fold = divmod(source_order, FOLDS_PER_SEED)
        source_fold_one_based = int(match.group(1))
        if source_fold_one_based != expected_fold + 1:
            raise ValueError(
                "Unexpected fold order at source cell "
                f"{source_order}: found {source_fold_one_based}, "
                f"expected {expected_fold + 1}"
            )
        cells.append(
            {
                "source_order": source_order,
                "split_seed": SEEDS[seed_index],
                "fold": expected_fold,
                "source_fold_one_based": source_fold_one_based,
                "pre_swap_train_cat_ids": sorted_cat_ids(pre.group("train")),
                "pre_swap_test_cat_ids": sorted_cat_ids(pre.group("test")),
                "train_cat_ids": sorted_cat_ids(post.group("train")),
                "test_cat_ids": sorted_cat_ids(post.group("test")),
                "moved_to_train": sorted_cat_ids(moves.group("moved_train")),
                "removed_from_train": sorted_cat_ids(moves.group("removed_train")),
                "moved_to_test": sorted_cat_ids(moves.group("moved_test")),
                "removed_from_test": sorted_cat_ids(moves.group("removed_test")),
                "source_reference": {
                    "logical_path": ORIGINAL_LOG_RELATIVE_PATH.as_posix(),
                    "line_start": line_number(text, block_start),
                    "line_end": line_number(text, block_end - 1),
                    "post_swap_line_start": line_number(
                        text, block_start + post.start()
                    ),
                    "post_swap_line_end": line_number(text, block_start + post.end()),
                    "swap_evidence_line_start": line_number(
                        text, block_start + moves.start()
                    ),
                    "swap_evidence_line_end": line_number(
                        text, block_start + moves.end()
                    ),
                },
            }
        )
    return cells


def load_dataset_metadata(
    data_manifest_path: Path,
    cat_manifest_path: Path,
    vggish_csv_path: Path,
) -> dict[str, object]:
    data_rows = read_csv(data_manifest_path)
    cat_rows = read_csv(cat_manifest_path)
    vggish_rows = read_csv(vggish_csv_path)

    published_cat_ids = sorted({row["published_cat_id"] for row in cat_rows})
    clean_cat_rows = [row for row in cat_rows if is_true(row["analysis_include"])]
    clean_cat_ids = sorted(row["analysis_cat_id"] for row in clean_cat_rows)
    if len(clean_cat_ids) != len(set(clean_cat_ids)):
        raise ValueError("Clean cat manifest contains duplicate analysis_cat_id values")

    clean_data_rows = [row for row in data_rows if is_true(row["analysis_include"])]
    call_counts = Counter(row["analysis_cat_id"] for row in clean_data_rows)
    vggish_counts = Counter(
        row["cat_id"]
        for row in vggish_rows
        if row["cat_id"] not in EXCLUDED_PUBLISHED_CAT_IDS
    )
    age_groups_by_cat: dict[str, set[str]] = {}
    for row in clean_data_rows:
        expected_label = age_group_from_age(row["age_years_filename"])
        if row["age_group_filename"] != expected_label:
            raise ValueError(
                f"Manifest label differs from original age boundary for {row['filename']}"
            )
        age_groups_by_cat.setdefault(row["analysis_cat_id"], set()).add(expected_label)
    multi_label_cats = {
        cat_id: sorted(labels)
        for cat_id, labels in age_groups_by_cat.items()
        if len(labels) != 1
    }
    if multi_label_cats:
        raise ValueError(f"Cats span multiple classifier labels: {multi_label_cats}")
    age_group_by_cat = {
        cat_id: next(iter(labels)) for cat_id, labels in age_groups_by_cat.items()
    }

    vggish_age_groups_by_cat: dict[str, set[str]] = {}
    for row in vggish_rows:
        if row["cat_id"] in EXCLUDED_PUBLISHED_CAT_IDS:
            continue
        vggish_age_groups_by_cat.setdefault(row["cat_id"], set()).add(
            age_group_from_age(row["target"])
        )

    if set(clean_cat_ids) != set(call_counts):
        raise ValueError("Clean data-manifest cats differ from clean cat-manifest cats")
    if set(clean_cat_ids) != set(vggish_counts):
        raise ValueError("Clean VGGish cats differ from clean cat-manifest cats")
    if any(len(labels) != 1 for labels in vggish_age_groups_by_cat.values()):
        raise ValueError("At least one VGGish cat spans multiple classifier labels")
    if {
        cat_id: next(iter(labels))
        for cat_id, labels in vggish_age_groups_by_cat.items()
    } != age_group_by_cat:
        raise ValueError("VGGish target labels differ from clean audio-manifest labels")
    if set(age_group_by_cat.values()) != set(LABELS):
        raise ValueError("Unexpected clean age-group labels")
    if len(clean_cat_ids) != EXPECTED_CLEAN_CATS:
        raise ValueError(
            f"Expected {EXPECTED_CLEAN_CATS} clean cats, found {len(clean_cat_ids)}"
        )
    if sum(call_counts.values()) != EXPECTED_CLEAN_CALLS:
        raise ValueError(
            f"Expected {EXPECTED_CLEAN_CALLS} clean calls, "
            f"found {sum(call_counts.values())}"
        )
    if sum(vggish_counts.values()) != EXPECTED_CLEAN_VGGISH_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_CLEAN_VGGISH_ROWS} clean VGGish rows, "
            f"found {sum(vggish_counts.values())}"
        )

    return {
        "published_cat_ids": published_cat_ids,
        "clean_cat_ids": clean_cat_ids,
        "call_counts": dict(call_counts),
        "vggish_counts": dict(vggish_counts),
        "age_group_by_cat": age_group_by_cat,
        "official_counts": {
            "cats": len(published_cat_ids),
            "calls": len(data_rows),
            "vggish_rows": len(vggish_rows),
        },
        "vggish_feature_columns": [str(index) for index in range(128)]
        + ["mean_freq"],
    }


def role_counts(cat_ids: Iterable[str], metadata: dict[str, object]) -> dict[str, object]:
    ids = list(cat_ids)
    call_counts: dict[str, int] = metadata["call_counts"]  # type: ignore[assignment]
    vggish_counts: dict[str, int] = metadata["vggish_counts"]  # type: ignore[assignment]
    age_group_by_cat: dict[str, str] = metadata["age_group_by_cat"]  # type: ignore[assignment]
    return {
        "cats": len(ids),
        "calls": sum(call_counts[cat_id] for cat_id in ids),
        "vggish_rows": sum(vggish_counts[cat_id] for cat_id in ids),
        "class_cats": {
            label: sum(age_group_by_cat[cat_id] == label for cat_id in ids)
            for label in LABELS
        },
        "class_calls": {
            label: sum(
                call_counts[cat_id]
                for cat_id in ids
                if age_group_by_cat[cat_id] == label
            )
            for label in LABELS
        },
        "class_vggish_rows": {
            label: sum(
                vggish_counts[cat_id]
                for cat_id in ids
                if age_group_by_cat[cat_id] == label
            )
            for label in LABELS
        },
    }


def validate_and_project_cells(
    logged_cells: list[dict[str, object]], metadata: dict[str, object]
) -> list[dict[str, object]]:
    published_cat_ids = set(metadata["published_cat_ids"])
    clean_cat_ids = set(metadata["clean_cat_ids"])
    projected: list[dict[str, object]] = []

    age_group_by_cat: dict[str, str] = metadata["age_group_by_cat"]  # type: ignore[assignment]
    for source in logged_cells:
        pre_train = list(source["pre_swap_train_cat_ids"])
        pre_test = list(source["pre_swap_test_cat_ids"])
        post_train = list(source["train_cat_ids"])
        post_test = list(source["test_cat_ids"])
        pre_train_set = set(pre_train)
        pre_test_set = set(pre_test)
        post_train_set = set(post_train)
        post_test_set = set(post_test)
        if pre_train_set & pre_test_set:
            raise ValueError(f"Pre-swap role overlap in source cell {source['source_order']}")
        if pre_train_set | pre_test_set != published_cat_ids:
            raise ValueError(
                f"Pre-swap role union differs from 112 published cats in cell "
                f"{source['source_order']}"
            )
        if post_train_set & post_test_set:
            raise ValueError(f"Original role overlap in source cell {source['source_order']}")
        if post_train_set | post_test_set != published_cat_ids:
            raise ValueError(
                f"Post-swap role union differs from 112 published cats in cell "
                f"{source['source_order']}"
            )

        moved_to_train = set(source["moved_to_train"])
        removed_from_train = set(source["removed_from_train"])
        moved_to_test = set(source["moved_to_test"])
        removed_from_test = set(source["removed_from_test"])
        if moved_to_train != post_train_set - pre_train_set:
            raise ValueError(f"Moved-to-train evidence mismatch in cell {source['source_order']}")
        if removed_from_train != pre_train_set - post_train_set:
            raise ValueError(
                f"Removed-from-train evidence mismatch in cell {source['source_order']}"
            )
        if moved_to_test != post_test_set - pre_test_set:
            raise ValueError(f"Moved-to-test evidence mismatch in cell {source['source_order']}")
        if removed_from_test != pre_test_set - post_test_set:
            raise ValueError(
                f"Removed-from-test evidence mismatch in cell {source['source_order']}"
            )
        if moved_to_train != removed_from_test or removed_from_train != moved_to_test:
            raise ValueError(f"Swap directions are inconsistent in cell {source['source_order']}")
        if not moved_to_train.issubset(FORCED_TRAIN_CAT_IDS):
            raise ValueError(f"Unexpected forced cat in cell {source['source_order']}")

        clean_pre_train = sorted(pre_train_set - set(EXCLUDED_PUBLISHED_CAT_IDS))
        clean_pre_test = sorted(pre_test_set - set(EXCLUDED_PUBLISHED_CAT_IDS))
        clean_train = sorted(post_train_set - set(EXCLUDED_PUBLISHED_CAT_IDS))
        clean_test = sorted(post_test_set - set(EXCLUDED_PUBLISHED_CAT_IDS))
        clean_train_set = set(clean_train)
        clean_test_set = set(clean_test)
        if clean_train_set & clean_test_set:
            raise ValueError(f"Clean role overlap in source cell {source['source_order']}")
        if clean_train_set | clean_test_set != clean_cat_ids:
            raise ValueError(
                f"Clean role union differs from 111 analysis cats in cell "
                f"{source['source_order']}"
            )
        if not set(FORCED_TRAIN_CAT_IDS).issubset(clean_train_set):
            raise ValueError(
                f"Forced-train cats missing from train in cell {source['source_order']}"
            )
        if set(FORCED_TRAIN_CAT_IDS) & clean_test_set:
            raise ValueError(
                f"Forced-train cat appears in test in cell {source['source_order']}"
            )

        swaps: list[dict[str, object]] = []
        used_replacements: set[str] = set()
        for forced_cat_id in FORCED_TRAIN_CAT_IDS:
            if forced_cat_id not in moved_to_train:
                continue
            candidates = sorted(
                cat_id
                for cat_id in removed_from_train
                if age_group_by_cat[cat_id] == age_group_by_cat[forced_cat_id]
            )
            if len(candidates) != 1:
                raise ValueError(
                    f"Cannot uniquely recover replacement for {forced_cat_id} in "
                    f"cell {source['source_order']}: {candidates}"
                )
            replacement_cat_id = candidates[0]
            if replacement_cat_id in used_replacements:
                raise ValueError(f"Replacement reused inside cell {source['source_order']}")
            used_replacements.add(replacement_cat_id)
            swaps.append(
                {
                    "forced_cat_id": forced_cat_id,
                    "replacement_cat_id": replacement_cat_id,
                    "source_reference": {
                        "logical_path": source["source_reference"]["logical_path"],
                        "line_start": source["source_reference"][
                            "swap_evidence_line_start"
                        ],
                        "line_end": source["source_reference"][
                            "swap_evidence_line_end"
                        ],
                    },
                }
            )
        if used_replacements != removed_from_train:
            raise ValueError(
                f"Unmapped replacement cats in cell {source['source_order']}: "
                f"{sorted(removed_from_train - used_replacements)}"
            )

        train_counts = role_counts(clean_train, metadata)
        test_counts = role_counts(clean_test, metadata)
        for role, counts in (("train", train_counts), ("test", test_counts)):
            if not all(counts["class_cats"][label] > 0 for label in LABELS):
                raise ValueError(
                    f"Missing age class in {role} for cell {source['source_order']}"
                )
        if train_counts["calls"] + test_counts["calls"] != EXPECTED_CLEAN_CALLS:
            raise ValueError(f"Call coverage mismatch in cell {source['source_order']}")
        if (
            train_counts["vggish_rows"] + test_counts["vggish_rows"]
            != EXPECTED_CLEAN_VGGISH_ROWS
        ):
            raise ValueError(f"VGGish coverage mismatch in cell {source['source_order']}")

        projected.append(
            {
                "cell_id": f"seed_{source['split_seed']}_fold_{source['fold']}",
                "split_seed": source["split_seed"],
                "fold": source["fold"],
                "train_cat_ids": clean_train,
                "test_cat_ids": clean_test,
                "pre_swap_train_cat_ids": clean_pre_train,
                "pre_swap_test_cat_ids": clean_pre_test,
                "swaps": swaps,
                "source_reference": source["source_reference"],
                "source_order": source["source_order"],
            }
        )
    return projected


def make_per_seed_test_counts(
    cells: list[dict[str, object]], clean_cat_ids: list[str]
) -> dict[str, object]:
    output: dict[str, object] = {}
    for seed in SEEDS:
        seed_cells = [cell for cell in cells if cell["split_seed"] == seed]
        if len(seed_cells) != FOLDS_PER_SEED:
            raise ValueError(f"Expected four cells for seed {seed}")
        counts = Counter(
            cat_id
            for cell in seed_cells
            for cat_id in cell["test_cat_ids"]
        )
        complete_counts = {cat_id: int(counts[cat_id]) for cat_id in clean_cat_ids}
        missing = [cat_id for cat_id, count in complete_counts.items() if count == 0]
        repeated = {
            cat_id: count for cat_id, count in complete_counts.items() if count > 1
        }
        if missing != list(FORCED_TRAIN_CAT_IDS):
            raise ValueError(f"Unexpected never-tested cats for seed {seed}: {missing}")
        if len(repeated) != 2 or set(repeated.values()) != {2}:
            raise ValueError(f"Unexpected repeated test cats for seed {seed}: {repeated}")
        output[str(seed)] = {
            "test_count_by_cat": complete_counts,
            "test_entries": sum(complete_counts.values()),
            "unique_test_cats": sum(count > 0 for count in complete_counts.values()),
            "never_tested_cat_ids": missing,
            "repeated_test_cat_ids": repeated,
            "complete_oof_partition": False,
        }
    return output


def logical_source(path: Path, logical_path: str) -> dict[str, object]:
    return {
        "logical_path": logical_path,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def build_payload(
    source_log_path: Path = None,
    data_manifest_path: Path = DATA_MANIFEST_PATH,
    cat_manifest_path: Path = CAT_MANIFEST_PATH,
    vggish_csv_path: Path = VGGISH_CSV_PATH,
) -> dict[str, object]:
    source_log_path = source_log_path or discover_original_log()
    metadata = load_dataset_metadata(
        data_manifest_path=data_manifest_path,
        cat_manifest_path=cat_manifest_path,
        vggish_csv_path=vggish_csv_path,
    )
    logged_cells = parse_logged_cells(source_log_path)
    cells = validate_and_project_cells(logged_cells, metadata)
    per_seed_test_counts = make_per_seed_test_counts(
        cells, list(metadata["clean_cat_ids"])
    )

    source_evidence = [
        {
            "kind": "actual_post_swap_roles_and_swap_differences",
            **logical_source(source_log_path, ORIGINAL_LOG_RELATIVE_PATH.as_posix()),
        },
        {
            "kind": "clean_call_membership_and_classifier_age_labels",
            **logical_source(
                data_manifest_path,
                "metadata/datasets/meowagenet/data_manifest.csv",
            ),
        },
        {
            "kind": "published_to_analysis_cat_projection",
            **logical_source(
                cat_manifest_path,
                "metadata/datasets/meowagenet/cat_id_manifest.csv",
            ),
        },
        {
            "kind": "original_vggish_prediction_rows",
            **logical_source(
                vggish_csv_path,
                "data/meowagenet/official-3d02295bef15/embeddings/"
                "vggish_looped_embeddings.csv",
            ),
        },
    ]
    per_cell_audit: list[dict[str, object]] = []
    for cell in cells:
        train_counts = role_counts(cell["train_cat_ids"], metadata)
        test_counts = role_counts(cell["test_cat_ids"], metadata)
        per_cell_audit.append(
            {
                "cell_id": cell["cell_id"],
                "split_seed": cell["split_seed"],
                "fold": cell["fold"],
                "train": train_counts,
                "test": test_counts,
                "train_test_disjoint": not bool(
                    set(cell["train_cat_ids"]) & set(cell["test_cat_ids"])
                ),
                "union_cats": len(
                    set(cell["train_cat_ids"]) | set(cell["test_cat_ids"])
                ),
                "projection_removed_from_pre_swap_role": {
                    "train": [
                        cat_id
                        for cat_id in EXCLUDED_PUBLISHED_CAT_IDS
                        if cat_id
                        in logged_cells[int(cell["source_order"])][
                            "pre_swap_train_cat_ids"
                        ]
                    ],
                    "test": [
                        cat_id
                        for cat_id in EXCLUDED_PUBLISHED_CAT_IDS
                        if cat_id
                        in logged_cells[int(cell["source_order"])][
                            "pre_swap_test_cat_ids"
                        ]
                    ],
                },
                "projection_removed_from_post_swap_role": {
                    "train": [
                        cat_id
                        for cat_id in EXCLUDED_PUBLISHED_CAT_IDS
                        if cat_id
                        in logged_cells[int(cell["source_order"])]["train_cat_ids"]
                    ],
                    "test": [
                        cat_id
                        for cat_id in EXCLUDED_PUBLISHED_CAT_IDS
                        if cat_id
                        in logged_cells[int(cell["source_order"])]["test_cat_ids"]
                    ],
                },
            }
        )

    cell_ids = [cell["cell_id"] for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("IDEA-088 cell IDs are not unique")

    return {
        "schema_version": 1.0,
        "protocol_id": "meowagenet-idea088-original-style-clean-v1",
        "source_commit": SOURCE_COMMIT,
        "source_prediction_rows": 937,
        "status": "frozen_roles_only_training_not_authorized",
        "projection": {
            "excluded_cat_ids": list(EXCLUDED_PUBLISHED_CAT_IDS),
            "expected_calls": EXPECTED_CLEAN_CALLS,
            "expected_cats": EXPECTED_CLEAN_CATS,
            "expected_vggish_rows": EXPECTED_CLEAN_VGGISH_ROWS,
            "method": (
                "Parse actual logged roles and swaps; delete only 049A from every "
                "pre/post role array; sort and deduplicate; never repartition cats."
            ),
        },
        "split_seeds": list(SEEDS),
        "forced_training_cat_ids": list(FORCED_TRAIN_CAT_IDS),
        "folds_per_seed": FOLDS_PER_SEED,
        "fold_indexing": "zero_based; archived source log is one_based",
        "classifier_labels": {
            "ordered_names_for_idea088": list(LABELS),
            "age_ranges_years_half_open": {
                "kitten": [0.0, 0.5],
                "adult": [0.5, 10.0],
                "senior": [10.0, 20.0],
            },
            "clean_cat_counts": {
                label: sum(
                    metadata["age_group_by_cat"][cat_id] == label
                    for cat_id in metadata["clean_cat_ids"]
                )
                for label in LABELS
            },
            "clean_call_counts": {
                label: sum(
                    metadata["call_counts"][cat_id]
                    for cat_id in metadata["clean_cat_ids"]
                    if metadata["age_group_by_cat"][cat_id] == label
                )
                for label in LABELS
            },
        },
        "vggish_input": {
            "feature_dimensions": len(metadata["vggish_feature_columns"]),
            "feature_columns": metadata["vggish_feature_columns"],
            "disclosure": (
                "The archived final model used iloc[:, :-4] after appending age_group, "
                "therefore retaining 128 embedding columns plus mean_freq (129 total)."
            ),
        },
        "dataset_counts": {
            "official": metadata["official_counts"],
            "clean": {
                "cats": len(metadata["clean_cat_ids"]),
                "calls": sum(metadata["call_counts"].values()),
                "vggish_rows": sum(metadata["vggish_counts"].values()),
                "class_cats": {
                    label: sum(
                        metadata["age_group_by_cat"][cat_id] == label
                        for cat_id in metadata["clean_cat_ids"]
                    )
                    for label in LABELS
                },
            },
        },
        "cells": cells,
        "audit": {
            "source_sha256": {
                row["kind"]: row["sha256"] for row in source_evidence
            },
            "cell_count": len(cells),
            "unique_cell_ids": len(set(cell_ids)),
            "per_cell": per_cell_audit,
            "per_seed_test_multiplicity": per_seed_test_counts,
            "global_validation": {
                "every_cell_train_test_disjoint": True,
                "every_cell_union_is_clean_111": True,
                "excluded_049A_absent_from_all_role_arrays": True,
                "every_cell_contains_all_three_classes_in_train_and_test": True,
                "every_cell_covers_clean_792_calls": True,
                "every_cell_covers_clean_936_vggish_rows": True,
                "forced_training_cats_never_tested": True,
                "per_seed_complete_oof_partition": False,
                "per_seed_oof_warning": (
                    "000A and 046A are never tested; two same-class replacement "
                    "cats are tested twice per seed. Equal-average 20 fold metrics "
                    "and do not claim complete OOF."
                ),
            },
            "runner_contract": {
                "train_cat_ids_field": "cells[].train_cat_ids",
                "test_cat_ids_field": "cells[].test_cat_ids",
                "pre_swap_fields_are_audit_only": True,
                "prohibitions": [
                    "do_not_regenerate_splits",
                    "do_not_restore_049A",
                    "do_not_train_on_pre_swap_roles",
                    "do_not_claim_complete_per_seed_oof",
                ],
            },
        },
        "source_evidence": source_evidence,
    }


def render_payload(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def write_or_verify(path: Path, content: str) -> str:
    encoded = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(
                f"Frozen IDEA-088 role artifact differs: {path}. "
                "Do not overwrite; investigate the source mismatch."
            )
        return "verified"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return "written"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-log", type=Path, default=None)
    parser.add_argument("--data-manifest", type=Path, default=DATA_MANIFEST_PATH)
    parser.add_argument("--cat-manifest", type=Path, default=CAT_MANIFEST_PATH)
    parser.add_argument("--vggish-csv", type=Path, default=VGGISH_CSV_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Validate and print the artifact without writing it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(
        source_log_path=args.source_log,
        data_manifest_path=args.data_manifest,
        cat_manifest_path=args.cat_manifest,
        vggish_csv_path=args.vggish_csv,
    )
    content = render_payload(payload)
    if args.stdout_only:
        print(content, end="")
        return
    action = write_or_verify(args.output, content)
    print(
        json.dumps(
            {
                "status": action,
                "output": str(args.output),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "cells": len(payload["cells"]),
                "training_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
