"""Independent, read-only design audit for IDEA-089.

This audit intentionally does not import the IDEA-089 runner.  It checks the
frozen source assets and the arithmetic assumptions needed before any training
is authorized.  An optional JSON output is permitted only at an IDEA-089 path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ROLES_PATH = ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
DATA_MANIFEST_PATH = (
    ROOT / "metadata" / "datasets" / "meowagenet" / "data_manifest.csv"
)
CAT_MANIFEST_PATH = (
    ROOT / "metadata" / "datasets" / "meowagenet" / "cat_id_manifest.csv"
)
VGG_PATH = (
    ROOT
    / "data"
    / "meowagenet"
    / "official-3d02295bef15"
    / "embeddings"
    / "vggish_looped_embeddings.csv"
)
AST_PATH = (
    ROOT
    / "runs"
    / "ast_locked_v1"
    / "gpu_rerun_2026-08-26"
    / "ast_standard_call_embeddings.npz"
)
ACOUSTIC_PATH = (
    ROOT
    / "runs"
    / "meowagenet_idea068_age_sensitive_ast_v1"
    / "features"
    / "age_sensitive_acoustic_features.npz"
)
ACOUSTIC_SUMMARY_PATH = ACOUSTIC_PATH.with_name("extraction_summary.json")
V21_RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_formal_v2_1.py"
IDEA088_FREEZER_PATH = ROOT / "scripts" / "freeze_idea088_original_roles.py"
V21_RESULTS_PATH = (
    ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_formal_v2_1_core_results.json"
)

EXPECTED_HASHES = {
    "roles": "87deda39808297e1af5b71283e1d7487a7b88d9288cb492c488e3e64fb91c433",
    "data_manifest": "68e5131dc5d3cd611ecdda30e5176a6dcc90c0ea2500a6d8a4d9b066ce11a72f",
    "cat_manifest": "5ae4c48c2266c4452b69a161c9e56bc506b35bff5fe3eaa53a3f69a91f641f5e",
    "vgg": "0c4856c12bb3b829a9731b426619d5f82d3510307374e756555fc8eb0349e641",
    "ast": "1c763169e9a9306cc46898808c571b0de7e41db0272e963c30fb88d9b4142399",
    "acoustic": "ba951db756436ea00adb5c7241ea0264de9fa0f6674a84f274812434d93e26ba",
    "acoustic_summary": "022cc2d54fc2af25430e327ebb60d8daff733d627c76a522d3baf81f3b4b2cda",
    "v21_runner": "ed3fc855539d37963aa51d17e54f5ce59acba1d023567289b89d5d1f313e6b25",
    "idea088_freezer": "da595817a34dcb344448783eafa2193ca2461ae6b9c29270401d03196384ae98",
    "v21_results": "20f9ed949381bc67f208640064ae75a0a91322dd02ab965b3af607763480ff1e",
}

PATHS = {
    "roles": ROLES_PATH,
    "data_manifest": DATA_MANIFEST_PATH,
    "cat_manifest": CAT_MANIFEST_PATH,
    "vgg": VGG_PATH,
    "ast": AST_PATH,
    "acoustic": ACOUSTIC_PATH,
    "acoustic_summary": ACOUSTIC_SUMMARY_PATH,
    "v21_runner": V21_RUNNER_PATH,
    "idea088_freezer": IDEA088_FREEZER_PATH,
    "v21_results": V21_RESULTS_PATH,
}

SELECTED_REPEATS = (0, 1, 2)
FOLDS = (0, 1, 2, 3)
BASE_SEEDS = (17, 43, 101)
LABEL_ORDER = ("kitten", "adult", "senior")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABEL_ORDER)}
EXPECTED_OUTER_SEEDS = {0: 104729, 1: 130363, 2: 155921}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_true(value: object) -> bool:
    return str(value).strip().lower() == "true"


def age_group(age_years: float) -> str:
    age = float(age_years)
    if 0.0 <= age < 0.5:
        return "kitten"
    if 0.5 <= age < 10.0:
        return "adult"
    if 10.0 <= age < 20.0:
        return "senior"
    raise ValueError(f"Age outside frozen three-class range: {age}")


def trainable_parameter_counts() -> dict[str, int]:
    # Dense layers include bias; BatchNorm has two trainable vectors.
    a0 = 768 * 128 + 128 + 2 * 128 + 128 * 3 + 3
    direct = 788 * 128 + 128 + 2 * 128 + 128 * 3 + 3
    branch = 20 * 60 + 60 + 60 * 128 + 128
    return {
        "A0": a0,
        "direct_concat": direct,
        "U1": a0 + branch,
        "C1": a0 + branch,
    }


def _clean_call_data() -> tuple[pd.DataFrame, dict[str, str], dict[str, int]]:
    data = pd.read_csv(DATA_MANIFEST_PATH, dtype=str, keep_default_na=False)
    clean = data[data["analysis_include"].map(is_true)].copy()
    if len(clean) != 792 or clean["analysis_cat_id"].nunique() != 111:
        raise AssertionError("Clean call manifest is not 792 calls / 111 cats")
    if "049A" in set(clean["analysis_cat_id"]):
        raise AssertionError("Excluded duplicate alias 049A remains in clean calls")

    labels_by_cat: dict[str, str] = {}
    for cat_id, part in clean.groupby("analysis_cat_id", sort=True):
        labels = set(part["age_group_filename"])
        if len(labels) != 1:
            raise AssertionError(f"Cat {cat_id} has non-unique semantic labels")
        labels_by_cat[str(cat_id)] = labels.pop()
    call_counts = clean["analysis_cat_id"].value_counts().astype(int).to_dict()
    if Counter(labels_by_cat.values()) != Counter(
        {"kitten": 15, "adult": 62, "senior": 34}
    ):
        raise AssertionError("Clean cat label counts changed")
    if Counter(clean["age_group_filename"]) != Counter(
        {"kitten": 134, "adult": 405, "senior": 253}
    ):
        raise AssertionError("Clean call label counts changed")
    return clean, labels_by_cat, call_counts


def audit_roles(labels_by_cat: dict[str, str], call_counts: dict[str, int]) -> dict:
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    selected = roles[roles["repeat"].isin(SELECTED_REPEATS)].copy()
    if len(roles) != 2220 or len(selected) != 1332:
        raise AssertionError("Formal-v2 role row count changed")
    if set(selected["cat_id"]) != set(labels_by_cat):
        raise AssertionError("Selected roles and clean cat identities differ")

    cells: list[dict] = []
    for repeat in SELECTED_REPEATS:
        repeat_rows = selected[selected["repeat"] == repeat]
        if set(repeat_rows["outer_seed"].astype(int)) != {EXPECTED_OUTER_SEEDS[repeat]}:
            raise AssertionError(f"Unexpected outer seed for repeat {repeat}")
        test_occurrences = Counter()
        for fold in FOLDS:
            cell = repeat_rows[repeat_rows["outer_fold"] == fold]
            if len(cell) != 111 or cell["cat_id"].nunique() != 111:
                raise AssertionError(f"Role cell {repeat}/{fold} is not 111 cats")
            role_sets = {
                role: set(cell.loc[cell["role"] == role, "cat_id"])
                for role in ("train", "validation", "test")
            }
            if any(
                role_sets[left] & role_sets[right]
                for left, right in (
                    ("train", "validation"),
                    ("train", "test"),
                    ("validation", "test"),
                )
            ):
                raise AssertionError(f"Role leakage in cell {repeat}/{fold}")
            if set().union(*role_sets.values()) != set(labels_by_cat):
                raise AssertionError(f"Role union changed in cell {repeat}/{fold}")
            for row in cell.itertuples(index=False):
                cat_id = str(row.cat_id)
                if row.age_group != labels_by_cat[cat_id]:
                    raise AssertionError(f"Role label mismatch for {cat_id}")
                if int(row.call_count) != call_counts[cat_id]:
                    raise AssertionError(f"Role call count mismatch for {cat_id}")
            test_occurrences.update(role_sets["test"])
            role_summary = {}
            for role, cat_ids in role_sets.items():
                role_summary[role] = {
                    "cats": len(cat_ids),
                    "calls": sum(call_counts[cat_id] for cat_id in cat_ids),
                    "cat_labels": dict(
                        sorted(Counter(labels_by_cat[c] for c in cat_ids).items())
                    ),
                }
            cells.append({"repeat": repeat, "fold": fold, "roles": role_summary})
        if set(test_occurrences) != set(labels_by_cat) or set(test_occurrences.values()) != {1}:
            raise AssertionError(f"Repeat {repeat} is not complete 111-cat OOF")
    return {
        "source_rows_all_five_repeats": len(roles),
        "selected_rows": len(selected),
        "selected_repeats": list(SELECTED_REPEATS),
        "folds_per_repeat": len(FOLDS),
        "complete_oof_cats_per_repeat": 111,
        "outer_seeds": EXPECTED_OUTER_SEEDS,
        "cells": cells,
    }


def audit_vgg(labels_by_cat: dict[str, str]) -> dict:
    frame = pd.read_csv(VGG_PATH, dtype={"cat_id": str})
    expected_columns = [str(index) for index in range(128)] + [
        "mean_freq",
        "gender",
        "target",
        "cat_id",
    ]
    if list(frame.columns) != expected_columns:
        raise AssertionError("VGG CSV column contract changed")
    clean = frame[frame["cat_id"].isin(labels_by_cat)].copy()
    if len(frame) != 937 or frame["cat_id"].nunique() != 112:
        raise AssertionError("Published VGG source is not 937 rows / 112 IDs")
    if len(clean) != 936 or clean["cat_id"].nunique() != 111:
        raise AssertionError("Clean VGG source is not 936 rows / 111 cats")
    clean["semantic_label"] = clean["target"].map(age_group)
    vgg_labels_by_cat: dict[str, str] = {}
    for cat_id, part in clean.groupby("cat_id", sort=True):
        labels = set(part["semantic_label"])
        if len(labels) != 1:
            raise AssertionError(f"VGG cat {cat_id} has non-unique labels")
        vgg_labels_by_cat[str(cat_id)] = labels.pop()
    mismatches = {
        cat_id: (labels_by_cat[cat_id], vgg_labels_by_cat[cat_id])
        for cat_id in labels_by_cat
        if labels_by_cat[cat_id] != vgg_labels_by_cat[cat_id]
    }
    if mismatches:
        raise AssertionError(f"VGG/call semantic cat labels differ: {mismatches}")
    mean_freq = clean["mean_freq"].to_numpy(dtype=np.float64)
    if not np.isfinite(mean_freq).all():
        raise AssertionError("VGG mean_freq contains non-finite values")
    return {
        "published_rows": len(frame),
        "published_cat_ids": frame["cat_id"].nunique(),
        "clean_rows": len(clean),
        "clean_cat_ids": clean["cat_id"].nunique(),
        "clean_row_labels": dict(sorted(Counter(clean["semantic_label"]).items())),
        "cat_label_mismatches_vs_call_manifest": 0,
        "vgg128_columns": [str(index) for index in range(128)],
        "vgg129_extra_column": "mean_freq",
        "mean_freq": {
            "all_finite": True,
            "mean": float(mean_freq.mean()),
            "sample_sd": float(mean_freq.std(ddof=1)),
            "minimum": float(mean_freq.min()),
            "maximum": float(mean_freq.max()),
        },
        "unit": "embedding/window row; no reliable row-to-call mapping in the CSV",
    }


def audit_caches(clean_calls: pd.DataFrame, labels_by_cat: dict[str, str]) -> dict:
    with np.load(AST_PATH, allow_pickle=False) as ast:
        if ast["embeddings"].shape != (792, 768):
            raise AssertionError("AST embedding shape changed")
        if not np.isfinite(ast["embeddings"]).all():
            raise AssertionError("AST embeddings contain non-finite values")
        call_ids = ast["call_ids"].astype(str)
        cat_ids = ast["cat_ids"].astype(str)
        labels = ast["labels"].astype(int)
    manifest_call_ids = clean_calls["filename"].to_numpy(dtype=str)
    if not np.array_equal(call_ids, manifest_call_ids):
        raise AssertionError("AST call order differs from clean manifest")
    if any(labels_by_cat[cat_id] != LABEL_ORDER[label] for cat_id, label in zip(cat_ids, labels)):
        raise AssertionError("AST integer labels differ from unified semantic labels")

    with np.load(ACOUSTIC_PATH, allow_pickle=False) as acoustic:
        features = acoustic["features"]
        feature_names = acoustic["feature_names"].astype(str)
        acoustic_call_ids = acoustic["call_ids"].astype(str)
    if features.shape != (792, 20) or len(set(feature_names)) != 20:
        raise AssertionError("Acoustic cache shape/name contract changed")
    if not np.array_equal(acoustic_call_ids, call_ids):
        raise AssertionError("Acoustic and AST call orders differ")
    if np.isinf(features).any() or not np.isfinite(features).any(axis=0).all():
        raise AssertionError("Acoustic cache cannot be safely training-side imputed")
    missing_by_feature = {
        name: int(np.isnan(features[:, index]).sum())
        for index, name in enumerate(feature_names)
    }
    summary = json.loads(ACOUSTIC_SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary["label_information_used"] is not False:
        raise AssertionError("Acoustic cache is not documented as label-blind")
    if summary["missing_values_by_feature"] != missing_by_feature:
        raise AssertionError("Acoustic cache missingness differs from extraction summary")
    return {
        "ast_shape": [792, 768],
        "acoustic_shape": [792, 20],
        "call_order_exactly_aligned": True,
        "acoustic_feature_names": feature_names.tolist(),
        "acoustic_missing_values_total": int(np.isnan(features).sum()),
        "acoustic_fully_finite_calls": int(np.isfinite(features).all(axis=1).sum()),
        "label_information_used_in_acoustic_extraction": False,
        "required_preprocessing_boundary": (
            "Median imputation and standardization must be fit on the current "
            "training side independently in selection and refit."
        ),
    }


def audit_historical_sources() -> dict:
    v21_code = V21_RUNNER_PATH.read_text(encoding="utf-8")
    idea088_code = IDEA088_FREEZER_PATH.read_text(encoding="utf-8")
    if 'feature_columns = [str(index) for index in range(128)]' not in v21_code:
        raise AssertionError("Cannot locate formal-v2.1 VGG128 source evidence")
    if 'monitor="val_loss"' not in v21_code or 'history.history["val_loss"]' not in v21_code:
        raise AssertionError("Cannot locate formal-v2.1 row-loss selection evidence")
    if '+ ["mean_freq"]' not in idea088_code:
        raise AssertionError("Cannot locate IDEA-088 VGG129 source evidence")
    results = json.loads(V21_RESULTS_PATH.read_text(encoding="utf-8"))
    execution = results["execution"]
    integrity = results["integrity_audit"]
    if execution["repeat_indices"] != list(SELECTED_REPEATS):
        raise AssertionError("Historical formal-v2.1 repeats changed")
    if execution["model_seeds"] != list(BASE_SEEDS):
        raise AssertionError("Historical formal-v2.1 model seeds changed")
    if integrity["completed_fold_level_fits"] != 108 or integrity["complete_oof_files"] != 27:
        raise AssertionError("Historical formal-v2.1 completion record changed")
    return {
        "formal_v2_1_vgg_dimensions": 128,
        "formal_v2_1_vgg_epoch_monitor": "Keras row-level val_loss",
        "idea088_author_style_vgg_dimensions": 129,
        "idea088_extra_column": "mean_freq",
        "historical_formal_v2_1_completed_fits": 108,
        "historical_formal_v2_1_complete_oof_files": 27,
        "reuse_boundary": (
            "Historical metrics, weights, predictions, and selected epochs are "
            "background only and cannot populate the IDEA-089 main table."
        ),
    }


def run_audit() -> dict:
    observed_hashes = {name: sha256(path) for name, path in PATHS.items()}
    if observed_hashes != EXPECTED_HASHES:
        changed = {
            name: {"expected": EXPECTED_HASHES[name], "observed": observed_hashes[name]}
            for name in EXPECTED_HASHES
            if observed_hashes[name] != EXPECTED_HASHES[name]
        }
        raise AssertionError(f"Frozen source hash mismatch: {changed}")

    clean_calls, labels_by_cat, call_counts = _clean_call_data()
    cat_manifest = pd.read_csv(CAT_MANIFEST_PATH, dtype=str, keep_default_na=False)
    clean_cat_manifest = cat_manifest[cat_manifest["analysis_include"].map(is_true)]
    if set(clean_cat_manifest["analysis_cat_id"]) != set(labels_by_cat):
        raise AssertionError("Clean cat and call manifests disagree")

    parameters = trainable_parameter_counts()
    expected_parameters = {"A0": 99075, "direct_concat": 101635, "U1": 108143, "C1": 108143}
    if parameters != expected_parameters:
        raise AssertionError(f"Parameter arithmetic mismatch: {parameters}")

    models = 6
    selection_fits = models * len(SELECTED_REPEATS) * len(FOLDS) * len(BASE_SEEDS)
    if selection_fits != 216:
        raise AssertionError("Selection fit budget arithmetic changed")

    return {
        "schema_version": "1.0",
        "audit_id": "IDEA-089-baseline-protocol-design-independent-audit",
        "status": "PASS_DESIGN_PREFLIGHT_ONLY_NO_TRAINING_AUTHORIZATION",
        "source_sha256": observed_hashes,
        "clean_data": {
            "calls": len(clean_calls),
            "cats": len(labels_by_cat),
            "cat_labels": dict(sorted(Counter(labels_by_cat.values()).items())),
            "call_labels": dict(sorted(Counter(clean_calls["age_group_filename"]).items())),
            "excluded_duplicate_alias": "049A",
            "unified_label_order": list(LABEL_ORDER),
        },
        "roles": audit_roles(labels_by_cat, call_counts),
        "vgg": audit_vgg(labels_by_cat),
        "caches": audit_caches(clean_calls, labels_by_cat),
        "historical_source_evidence": audit_historical_sources(),
        "frozen_design_arithmetic": {
            "models_including_vgg128_control": models,
            "base_seeds": list(BASE_SEEDS),
            "selection_fits": selection_fits,
            "outer_refit_fits": selection_fits,
            "total_real_fits": selection_fits * 2,
            "complete_oof_scores_per_model": len(SELECTED_REPEATS) * len(BASE_SEEDS),
            "prediction_occurrences_per_model": 111 * len(SELECTED_REPEATS) * len(BASE_SEEDS),
            "distinct_animals": 111,
            "trainable_parameters": parameters,
        },
        "claim_boundaries": {
            "mean_freq": (
                "The source column is named mean_freq. Its extraction and a "
                "row-to-call mapping are not independently available here, so "
                "it is not asserted to be a verified per-call F0 measurement."
            ),
            "vgg_vs_ast": (
                "VGG uses 936 embedding/window rows; AST uses 792 calls. Only "
                "cat identity and semantic label are aligned, not row identity."
            ),
            "independence": (
                "Each model has 999 prediction occurrences across nine complete "
                "OOF sets, but only 111 distinct animals."
            ),
            "training_gate": "This audit does not authorize or execute training.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output; its path must contain 'idea089'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit()
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        if "idea089" not in str(output).lower():
            raise ValueError("Refusing to write outside an IDEA-089-named path")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
