"""Read-only independent audit for an IDEA-082 protocol and CPU preflight.

The audit deliberately does not import the candidate runner and never launches a
fit.  It verifies locked assets, the preregistered 5/10/5 feature partition,
fold-local preprocessing evidence, cat-level isolation, matched pairing, the
outer-test lock, and role-local deterministic shuffled controls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_REVIEW = (
    ROOT
    / "metadata"
    / "experiments"
    / "idea082_independent_feature_grouping_review_v1.json"
)
REFERENCE_RUNNER = ROOT / "scripts" / "run_meowagenet_idea068_age_sensitive_ast.py"
FEATURE_CACHE = (
    ROOT
    / "runs"
    / "meowagenet_idea068_age_sensitive_ast_v1"
    / "features"
    / "age_sensitive_acoustic_features.npz"
)
EXTRACTION_SUMMARY = FEATURE_CACHE.parent / "extraction_summary.json"
ROLES = ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"

EXPECTED_HASHES = {
    "reference_runner": "3eab879d1a39bfdc4c3d56d3e105d628c0a38222e550760965f71706cccfe745",
    "feature_cache": "ba951db756436ea00adb5c7241ea0264de9fa0f6674a84f274812434d93e26ba",
    "extraction_summary": "022cc2d54fc2af25430e327ebb60d8daff733d627c76a522d3baf81f3b4b2cda",
    "roles": "87deda39808297e1af5b71283e1d7487a7b88d9288cb492c488e3e64fb91c433",
}
EXPECTED_FEATURES = (
    "log_f0_q10",
    "log_f0_median",
    "log_f0_q90",
    "log_f0_iqr",
    "log_f0_std",
    "log_f0_mad",
    "log_f0_slope",
    "log_f0_abs_delta_median",
    "period_variation_proxy",
    "voiced_fraction",
    "voiced_probability_mean",
    "voiced_probability_std",
    "periodicity_median",
    "hnr_db_median",
    "amplitude_variation_proxy",
    "log_rms_median",
    "log_rms_iqr",
    "spectral_tilt_median",
    "spectral_tilt_iqr",
    "spectral_flatness_median",
)
EXPECTED_GROUPS = {
    "G1": (0, 1, 2, 3, 6),
    "G2": (4, 5, 7, 8, 9, 10, 11, 12, 13, 14),
    "G3": (15, 16, 17, 18, 19),
    # G12 is the exact G1 union G2, represented in the locked native schema
    # order so that indices and feature names remain directly auditable.
    "G12": tuple(range(15)),
}
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def walk(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, path + (str(index),))


def values_for_keys(data: Any, predicate) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for path, value in walk(data):
        if path and predicate(norm(path[-1])):
            rows.append((".".join(path), value))
    return rows


def has_scalar(data: Any, expected: object) -> bool:
    return any(value == expected for _, value in walk(data) if not isinstance(value, (dict, list)))


def bool_evidence(
    data: Any,
    alternatives: tuple[tuple[str, ...], ...],
    expected: bool,
) -> list[str]:
    evidence = []
    for path, value in walk(data):
        if not path or not isinstance(value, bool) or value is not expected:
            continue
        key = norm(path[-1])
        if any(all(token in key for token in tokens) for tokens in alternatives):
            evidence.append(".".join(path))
    return evidence


def numeric_zero_evidence(
    data: Any, alternatives: tuple[tuple[str, ...], ...]
) -> list[str]:
    evidence = []
    for path, value in walk(data):
        if not path or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        key = norm(path[-1])
        if value == 0 and any(all(token in key for token in tokens) for tokens in alternatives):
            evidence.append(".".join(path))
    return evidence


def feature_indices(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, dict):
        preferred = (
            "indices",
            "feature_indices",
            "columns",
            "features",
            "feature_names",
        )
        by_norm = {norm(key): child for key, child in value.items()}
        for key in preferred:
            if key in by_norm:
                return feature_indices(by_norm[key])
        return None
    if not isinstance(value, list):
        return None
    if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return tuple(int(item) for item in value)
    if all(isinstance(item, str) and item in EXPECTED_FEATURES for item in value):
        return tuple(EXPECTED_FEATURES.index(str(item)) for item in value)
    return None


def group_key_matches(key: str, group: str) -> bool:
    parts = norm(key).split("_")
    return group.lower() in parts or norm(key) == group.lower()


def find_group_evidence(data: Any, group: str) -> list[tuple[str, tuple[int, ...]]]:
    evidence = []
    for path, value in walk(data):
        if not path or not group_key_matches(path[-1], group):
            continue
        indices = feature_indices(value)
        if indices is not None:
            evidence.append((".".join(path), indices))
    return evidence


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
    *,
    evidence: Any | None = None,
) -> None:
    row: dict[str, Any] = {
        "name": name,
        "passed": bool(passed),
        "detail": detail,
    }
    if evidence is not None:
        row["evidence"] = evidence
    checks.append(row)


def replay_role_mappings(protocol: dict[str, Any]) -> dict[tuple[int, int, str], dict[str, Any]]:
    """Recreate every role-local derangement without importing candidate code."""

    embedding_path = ROOT / protocol["data"]["frozen_embedding_path"]
    with np.load(embedding_path, allow_pickle=False) as loaded:
        cat_ids = loaded["cat_ids"].astype(str)
    role_rows: dict[tuple[int, int, str], set[str]] = {}
    with ROLES.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["repeat"]), int(row["outer_fold"]), str(row["role"]))
            role_rows.setdefault(key, set()).add(str(row["cat_id"]))
    result: dict[tuple[int, int, str], dict[str, Any]] = {}
    for repeat in protocol["model"]["repeats"]:
        for fold in protocol["model"]["folds"]:
            role_cats = {
                role: role_rows[(int(repeat), int(fold), role)]
                for role in ("train", "validation", "test")
            }
            if any(
                role_cats[left] & role_cats[right]
                for left, right in (
                    ("train", "validation"),
                    ("train", "test"),
                    ("validation", "test"),
                )
            ):
                raise RuntimeError(f"Cat leakage in roles repeat={repeat} fold={fold}")
            for role in ("train", "validation"):
                destinations = np.flatnonzero(
                    np.isin(cat_ids, np.asarray(sorted(role_cats[role])))
                ).astype(np.int64)
                destinations.sort()
                material = (
                    f"IDEA-082-shuffle-v1|repeat={repeat}|fold={fold}|role={role}"
                )
                digest = hashlib.sha256(material.encode("utf-8")).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
                order = destinations[rng.permutation(len(destinations))]
                source = np.roll(order, 1)
                sorting = np.argsort(order)
                destination_sorted = order[sorting]
                source_sorted = source[sorting]
                pairs = np.column_stack([destination_sorted, source_sorted]).astype("<i8")
                result[(int(repeat), int(fold), role)] = {
                    "mapping_sha256": hashlib.sha256(pairs.tobytes()).hexdigest(),
                    "fixed_points": int(np.sum(destination_sorted == source_sorted)),
                    "role_size": int(len(destinations)),
                }
    return result


def audit(
    protocol_path: Path,
    preflight_path: Path,
    runner_path: Path,
) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    preflight = read_json(preflight_path)
    review = read_json(REFERENCE_REVIEW)
    checks: list[dict[str, Any]] = []

    locked_paths = {
        "reference_runner": REFERENCE_RUNNER,
        "feature_cache": FEATURE_CACHE,
        "extraction_summary": EXTRACTION_SUMMARY,
        "roles": ROLES,
    }
    observed_hashes = {}
    for name, path in locked_paths.items():
        observed = sha256(path) if path.exists() else None
        observed_hashes[name] = observed
        add_check(
            checks,
            f"locked_{name}_sha256",
            observed == EXPECTED_HASHES[name],
            f"Locked {name} must remain byte-identical.",
            evidence={"path": str(path), "expected": EXPECTED_HASHES[name], "observed": observed},
        )

    with np.load(FEATURE_CACHE, allow_pickle=False) as loaded:
        cache_names = tuple(loaded["feature_names"].astype(str).tolist())
        cache_shape = tuple(int(value) for value in loaded["features"].shape)
    add_check(
        checks,
        "cache_schema",
        cache_names == EXPECTED_FEATURES and cache_shape == (792, 20),
        "The locked cache must contain the exact 792x20 ordered IDEA-068 schema.",
        evidence={"shape": cache_shape, "feature_names": cache_names},
    )

    extraction = read_json(EXTRACTION_SUMMARY)
    add_check(
        checks,
        "label_free_extraction",
        extraction.get("label_information_used") is False
        and extraction.get("source_hashes_verified") == 792,
        "Feature extraction must remain label-free with every source hash verified.",
        evidence={
            "label_information_used": extraction.get("label_information_used"),
            "source_hashes_verified": extraction.get("source_hashes_verified"),
        },
    )

    feature_name_candidates = values_for_keys(
        protocol, lambda key: key in {"feature_names", "acoustic_feature_names"}
    )
    exact_feature_paths = [
        path
        for path, value in feature_name_candidates
        if isinstance(value, list) and tuple(value) == EXPECTED_FEATURES
    ]
    candidate_group_keys = {
        "G1": "G1_f0",
        "G2": "G2_stability",
        "G3": "G3_spectral_energy",
        "G12": "G12_f0_stability",
    }
    group_name_evidence: dict[str, bool] = {}
    for short, key in candidate_group_keys.items():
        entry = protocol.get("feature_groups", {}).get(key, {})
        expected_names = [EXPECTED_FEATURES[index] for index in EXPECTED_GROUPS[short]]
        group_name_evidence[short] = entry.get("feature_names") == expected_names
    add_check(
        checks,
        "protocol_feature_order",
        bool(exact_feature_paths) or all(group_name_evidence.values()),
        "Protocol must lock the exact schema directly or through index/name-consistent locked groups.",
        evidence={
            "full_schema_paths": exact_feature_paths,
            "group_index_name_consistency": group_name_evidence,
        },
    )

    for group, expected in EXPECTED_GROUPS.items():
        evidence = find_group_evidence(protocol, group)
        exact = [(path, indices) for path, indices in evidence if indices == expected]
        add_check(
            checks,
            f"protocol_{group.lower()}_indices",
            bool(exact),
            f"{group} must equal the preregistered ordered indices {list(expected)}.",
            evidence=[{"path": path, "indices": indices} for path, indices in evidence],
        )

    primary_union = set(EXPECTED_GROUPS["G1"]) | set(EXPECTED_GROUPS["G2"]) | set(EXPECTED_GROUPS["G3"])
    primary_total = len(EXPECTED_GROUPS["G1"]) + len(EXPECTED_GROUPS["G2"]) + len(EXPECTED_GROUPS["G3"])
    add_check(
        checks,
        "reference_partition_disjoint_exhaustive",
        primary_union == set(range(20)) and primary_total == 20,
        "The independent 5/10/5 primary partition must be disjoint and exhaustive.",
        evidence=review.get("primary_groups"),
    )

    for label, expected in (
        ("feature_cache", EXPECTED_HASHES["feature_cache"]),
        ("roles", EXPECTED_HASHES["roles"]),
    ):
        add_check(
            checks,
            f"protocol_declares_{label}_sha256",
            has_scalar(protocol, expected),
            f"Protocol must declare the locked {label} SHA-256.",
            evidence=expected,
        )
        transitive_protocol_binding = has_scalar(preflight, sha256(protocol_path))
        add_check(
            checks,
            f"preflight_confirms_{label}_sha256",
            has_scalar(preflight, expected)
            or (label == "roles" and transitive_protocol_binding),
            f"Preflight must confirm {label} directly or bind the exact protocol that locks it.",
            evidence={
                "expected": expected,
                "direct": has_scalar(preflight, expected),
                "exact_protocol_bound": transitive_protocol_binding,
            },
        )

    protocol_hash = sha256(protocol_path)
    runner_hash = sha256(runner_path)
    add_check(
        checks,
        "preflight_protocol_hash",
        has_scalar(preflight, protocol_hash),
        "Preflight must bind to the exact candidate protocol bytes.",
        evidence=protocol_hash,
    )
    add_check(
        checks,
        "preflight_runner_hash",
        has_scalar(preflight, runner_hash),
        "Preflight must bind to the exact candidate runner bytes.",
        evidence=runner_hash,
    )

    for source_name, source in (("protocol", protocol), ("preflight", preflight)):
        flags = values_for_keys(source, lambda key: "outer_test" in key)
        valid = bool(flags) and all(value is False for _, value in flags)
        add_check(
            checks,
            f"{source_name}_outer_test_lock",
            valid,
            f"Every {source_name} outer-test flag must exist and be false.",
            evidence=flags,
        )

    runner_text = runner_path.read_text(encoding="utf-8").lower()
    add_check(
        checks,
        "cpu_preflight_status",
        preflight.get("status") == "GO"
        and preflight.get("device") == "cpu"
        and preflight.get("cuda_initialized") is False,
        "The submitted preflight must be a CPU-only GO with CUDA uninitialized.",
        evidence={
            "status": preflight.get("status"),
            "device": preflight.get("device"),
            "cuda_initialized": preflight.get("cuda_initialized"),
        },
    )

    preprocess_static = {
        "nanmedian_on_age_train": "np.nanmedian(age_train" in runner_text,
        "train_subset_passed_to_model": "age_train=age_features[train_indices]" in runner_text,
        "train_ast_mean": "ast_mean=embeddings.mean(axis=0)" in runner_text,
        "train_ast_scale": "ast_scale=embeddings.std(axis=0)" in runner_text,
        "frozen_validation_transform": "self.age_median" in runner_text
        and "self.age_mean" in runner_text
        and "self.age_scale" in runner_text,
    }
    shuffle_cells = preflight.get("shuffle_cells", [])
    train_stats_equal = (
        isinstance(shuffle_cells, list)
        and len(shuffle_cells) == 12
        and all(row.get("all_group_training_statistics_equal") is True for row in shuffle_cells)
    )
    add_check(
        checks,
        "fold_local_imputation_standardization",
        all(preprocess_static.values()) and train_stats_equal,
        "Runner and preflight must prove training-fold-only imputation and standardization.",
        evidence={"static": preprocess_static, "all_12_shuffle_train_stats_equal": train_stats_equal},
    )

    replay_error = None
    replayed: dict[tuple[int, int, str], dict[str, Any]] = {}
    try:
        replayed = replay_role_mappings(protocol)
    except Exception as error:  # pragma: no cover - surfaced in audit JSON
        replay_error = f"{type(error).__name__}: {error}"
    cat_isolation = replay_error is None and len(replayed) == 24 and preflight.get("role_cells") == 12
    add_check(
        checks,
        "cat_level_role_isolation",
        cat_isolation,
        "Independent replay of all 12 role cells must find no cross-role cat leakage.",
        evidence={"replayed_role_mappings": len(replayed), "error": replay_error},
    )

    initialization = preflight.get("initialization", {})
    logits = initialization.get("max_logit_difference_vs_A0", {})
    common_states = initialization.get("common_AST_state_equal_to_A0", {})
    pair_states = initialization.get("real_shuffled_full_state_equal", {})
    init_evidence = (
        len(logits) == 8
        and all(float(value) == 0.0 for value in logits.values())
        and len(common_states) == 8
        and all(value is True for value in common_states.values())
        and len(pair_states) == 4
        and all(value is True for value in pair_states.values())
    )
    batch_static = {
        "common_full_seed": "full_seed," in runner_text and "fit_inner(" in runner_text,
        "epoch_cat_order_hash_checked": "cat_order_sha256" in runner_text,
        "epoch_call_coverage_hash_checked": "call_coverage_sha256" in runner_text,
        "fail_closed_on_difference": "paired batch order differs" in runner_text,
    }
    add_check(
        checks,
        "paired_common_initialization",
        init_evidence,
        "Preflight must prove matched common initialization across paired variants.",
        evidence={"zero_logits": logits, "common_states": common_states, "pair_states": pair_states},
    )
    add_check(
        checks,
        "paired_cat_batch_order",
        all(batch_static.values()),
        "Runner must use one full seed per cell and fail closed on epoch batch-order hash differences.",
        evidence=batch_static,
    )

    observed_mappings: dict[tuple[int, int, str], str] = {}
    shuffle_rows_valid = isinstance(shuffle_cells, list) and len(shuffle_cells) == 12
    if shuffle_rows_valid:
        for row in shuffle_cells:
            repeat = int(row.get("repeat", -1))
            fold = int(row.get("fold", -1))
            mappings = row.get("mapping_sha256", {})
            fixed = row.get("fixed_points", {})
            if set(mappings) != {"train", "validation"} or set(fixed) != {"train", "validation"}:
                shuffle_rows_valid = False
                continue
            for role in ("train", "validation"):
                value = mappings[role]
                if not isinstance(value, str) or not HEX64.fullmatch(value) or fixed[role] != 0:
                    shuffle_rows_valid = False
                else:
                    observed_mappings[(repeat, fold, role)] = value.lower()
            if row.get("all_group_multisets_equal") is not True:
                shuffle_rows_valid = False
    replay_matches = (
        replay_error is None
        and len(observed_mappings) == 24
        and all(
            observed_mappings.get(key) == value["mapping_sha256"]
            and value["fixed_points"] == 0
            for key, value in replayed.items()
        )
    )
    shuffle_protocol = protocol.get("shuffle_control", {})
    no_cross_role = (
        shuffle_protocol.get("roles_permuted_independently") == ["train", "validation"]
        and shuffle_protocol.get("cross_role_mapping") is False
        and shuffle_protocol.get("test_touched") is False
        and shuffle_protocol.get("label_blind") is True
    )
    add_check(
        checks,
        "shuffled_control_within_role",
        shuffle_rows_valid and no_cross_role,
        "Shuffled controls must contain valid train/validation-only role mappings for all cells.",
        evidence={"rows": len(shuffle_cells) if isinstance(shuffle_cells, list) else None, "protocol": shuffle_protocol},
    )
    add_check(
        checks,
        "shuffled_control_deterministic",
        replay_matches,
        "Independent SHA-seeded replay must reproduce every submitted mapping hash.",
        evidence={"replayed": len(replayed), "matched": replay_matches, "error": replay_error},
    )
    add_check(
        checks,
        "shuffled_control_no_cross_role",
        no_cross_role and cat_isolation,
        "Protocol and independent role replay must reject cross-role/full-data/test permutation.",
        evidence={"protocol_lock": no_cross_role, "cat_role_isolation": cat_isolation},
    )
    add_check(
        checks,
        "shuffled_control_permutation_hashes",
        len(observed_mappings) == 24 and len(set(observed_mappings.values())) == 24,
        "Preflight must record one distinct mapping hash for each of 24 role cells.",
        evidence={"count": len(observed_mappings), "unique": len(set(observed_mappings.values()))},
    )

    aggregation = protocol.get("aggregation", {})
    paired_summary = (
        "base_seed-by-repeat" in str(aggregation.get("primary_metric_unit", ""))
        and "repeat-by-fold" in str(aggregation.get("split_cell_unit", ""))
        and "seed_repeat_results" in runner_text
        and "split_cell_results" in runner_text
    )
    animal_level = (
        "animal" in str(aggregation.get("calibration_and_BA_gate_unit", "")).lower()
        and "animal_metrics" in runner_text
        and "validation_animal_predictions" in runner_text
    )
    add_check(
        checks,
        "paired_summary_cells",
        paired_summary,
        "Preflight must lock seed/repeat/fold as the paired reporting cell.",
        evidence=aggregation,
    )
    add_check(
        checks,
        "animal_level_metrics",
        animal_level,
        "Preflight must confirm cat/animal-level validation metrics.",
        evidence={"aggregation": aggregation, "runner_markers": animal_level},
    )

    static_terms = {
        "runner_mentions_train_indices": "train_indices" in runner_text,
        "runner_mentions_nanmedian": "nanmedian" in runner_text,
        "runner_mentions_age_train": "age_train" in runner_text,
        "runner_mentions_cat_id": "cat_id" in runner_text,
        "runner_mentions_permutation": "permutation" in runner_text,
        "runner_mentions_outer_test_lock": "outer_test_accessed" in runner_text,
    }
    add_check(
        checks,
        "runner_static_safety_markers",
        all(static_terms.values()),
        "Candidate runner must expose reviewable markers for fold-local preprocessing, cat grouping, permutation, and the outer-test lock.",
        evidence=static_terms,
    )

    failed = [row["name"] for row in checks if not row["passed"]]
    return {
        "audit_id": "idea082-independent-protocol-preflight-audit-v1",
        "status": "GO" if not failed else "NO_GO",
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "preflight_path": str(preflight_path),
        "preflight_sha256": sha256(preflight_path),
        "runner_path": str(runner_path),
        "runner_sha256": runner_hash,
        "outer_test_accessed": False,
        "gpu_used": False,
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_checks": failed,
        "observed_locked_hashes": observed_hashes,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    result = audit(resolve(args.protocol), resolve(args.preflight), resolve(args.runner))
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if result["status"] != "GO":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
