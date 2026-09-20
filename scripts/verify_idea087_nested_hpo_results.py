"""Independent IDEA-087 selection and outer-results verification.

This verifier deliberately does not import the IDEA-087 runner.  It rebuilds
roles from the locked source table, rebuilds animal predictions from raw call
predictions, repeats the selection rule, and recomputes the final OOF metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/protocol/meowagenet_idea087_nested_hpo_v1.json"
RUN_ROOT = ROOT / "runs/meowagenet_idea087_nested_hpo_v1"
PIPELINES = (
    "A0_ast_only",
    "U1_wide_unbounded_additive",
    "C1_bounded_wide_additive",
)
POLICIES = ("original", "selected")
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
LABEL_BY_AGE = {"kitten": 0, "adult": 1, "senior": 2}
TOLERANCE = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("selection", "full"), required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def close(left: float, right: float, context: str, tolerance: float = TOLERANCE) -> float:
    difference = abs(float(left) - float(right))
    if not math.isfinite(difference) or difference > tolerance:
        raise RuntimeError(f"{context}: {left} != {right}")
    return difference


def compare_nested(
    rebuilt: Any, saved: Any, context: str, tolerance: float = TOLERANCE
) -> float:
    """Compare a rebuilt JSON-like value while allowing tiny float differences."""
    if isinstance(rebuilt, dict):
        if not isinstance(saved, dict):
            raise RuntimeError(f"{context}: saved value is not an object")
        missing = set(rebuilt) - set(saved)
        if missing:
            raise RuntimeError(f"{context}: saved object lacks keys {sorted(missing)}")
        return max(
            (
                compare_nested(rebuilt[key], saved[key], f"{context}.{key}", tolerance)
                for key in rebuilt
            ),
            default=0.0,
        )
    if isinstance(rebuilt, list):
        if not isinstance(saved, list) or len(rebuilt) != len(saved):
            raise RuntimeError(f"{context}: list shape differs")
        return max(
            (
                compare_nested(left, right, f"{context}[{index}]", tolerance)
                for index, (left, right) in enumerate(zip(rebuilt, saved))
            ),
            default=0.0,
        )
    if isinstance(rebuilt, bool) or isinstance(saved, bool):
        if rebuilt is not saved:
            raise RuntimeError(f"{context}: {rebuilt!r} != {saved!r}")
        return 0.0
    if isinstance(rebuilt, (int, float, np.integer, np.floating)) and isinstance(
        saved, (int, float, np.integer, np.floating)
    ):
        return close(float(rebuilt), float(saved), context, tolerance)
    if rebuilt != saved:
        raise RuntimeError(f"{context}: {rebuilt!r} != {saved!r}")
    return 0.0


def full_search_seed(base_seed: int, outer_fold: int, inner_fold: int) -> int:
    return int(base_seed) + 10_000 * int(outer_fold) + 100 * int(inner_fold)


def full_refit_seed(base_seed: int, outer_fold: int) -> int:
    return int(base_seed) + 10_000 * int(outer_fold)


def refit_epoch(best_epochs: Iterable[int], maximum_epochs: int = 50) -> int:
    values = sorted(int(value) for value in best_epochs)
    if len(values) != 6 or values[0] < 1 or values[-1] > maximum_epochs:
        raise RuntimeError("independent refit epoch requires six valid 1-based epochs")
    median = (values[2] + values[3]) / 2.0
    return int(min(max(math.floor(median + 0.5), 1), maximum_epochs))


def metric_bundle(animals: pd.DataFrame) -> dict[str, Any]:
    frame = animals.sort_values("cat_id").reset_index(drop=True)
    labels = frame["true_label"].to_numpy(dtype=np.int64)
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    recalls = recall_score(
        labels, predictions, labels=[0, 1, 2], average=None, zero_division=0
    )
    targets = np.eye(3, dtype=np.float64)[labels]
    clipped = np.clip(probabilities, 1.0e-12, 1.0)
    return {
        "n": int(len(frame)),
        "plain_accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(np.mean(recalls)),
        "class_recall": {
            "kitten": float(recalls[0]),
            "adult": float(recalls[1]),
            "senior": float(recalls[2]),
        },
        "cross_entropy": float(
            -np.mean(np.log(clipped[np.arange(len(labels)), labels]))
        ),
        "brier": float(np.mean(np.sum((probabilities - targets) ** 2, axis=1))),
    }


def configuration_map(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = protocol["search"]["configurations"]
    result = {str(row["config_id"]): row for row in rows}
    if len(rows) != 8 or len(result) != 8 or "q00" not in result:
        raise RuntimeError("IDEA-087 configuration matrix is not the locked 8-row grid")
    return result


def inner_fit_path(
    pipeline: str,
    config_id: str,
    outer_fold: int,
    inner_fold: int,
    base_seed: int,
) -> Path:
    return (
        RUN_ROOT
        / "inner"
        / pipeline
        / config_id
        / f"outer_{outer_fold}"
        / f"inner_{inner_fold}"
        / f"search_seed_{base_seed}"
        / "fit_summary.json"
    )


def outer_fit_path(
    pipeline: str, policy: str, outer_fold: int, base_seed: int
) -> Path:
    return (
        RUN_ROOT
        / "outer"
        / pipeline
        / policy
        / f"outer_{outer_fold}"
        / f"refit_seed_{base_seed}"
        / "fit_summary.json"
    )


def load_locked_inputs(
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roles_path = ROOT / protocol["data"]["roles_path"]
    manifest_path = ROOT / protocol["data"]["dataset_manifest_path"]
    if sha256(roles_path) != protocol["data"]["roles_sha256"]:
        raise RuntimeError("IDEA-087 roles hash differs")
    if sha256(manifest_path) != protocol["data"]["dataset_manifest_sha256"]:
        raise RuntimeError("IDEA-087 dataset manifest hash differs")
    roles = pd.read_csv(roles_path, dtype={"cat_id": str})
    roles = roles[roles["repeat"] == int(protocol["data"]["roles_repeat"])].copy()
    if len(roles) != 444:
        raise RuntimeError("IDEA-087 repeat-0 roles must have 444 rows")
    manifest = pd.read_csv(manifest_path)
    included = manifest[manifest["analysis_include"].astype(bool)].copy().reset_index(
        drop=True
    )
    included["call_index"] = np.arange(len(included), dtype=np.int64)
    included["call_id"] = included["filename"].astype(str)
    included["cat_id"] = included["analysis_cat_id"].astype(str)
    included["true_label"] = included["age_group_filename"].map(LABEL_BY_AGE)
    if len(included) != 792 or included["cat_id"].nunique() != 111:
        raise RuntimeError("IDEA-087 manifest is not the locked 792-call/111-cat set")
    if included["true_label"].isna().any():
        raise RuntimeError("IDEA-087 manifest contains an unknown age class")
    included["true_label"] = included["true_label"].astype(np.int64)
    cat_truth = (
        included.groupby("cat_id", sort=True)
        .agg(
            true_label=("true_label", "first"),
            label_count=("true_label", "nunique"),
            call_count=("call_id", "size"),
        )
        .reset_index()
    )
    if (cat_truth["label_count"] != 1).any():
        raise RuntimeError("IDEA-087 manifest assigns multiple labels to one cat")
    for row in roles.itertuples(index=False):
        truth = cat_truth[cat_truth["cat_id"] == str(row.cat_id)]
        if len(truth) != 1:
            raise RuntimeError(f"IDEA-087 role cat missing from manifest: {row.cat_id}")
        expected_label = LABEL_BY_AGE[str(row.age_group)]
        if (
            int(truth.iloc[0]["true_label"]) != expected_label
            or int(truth.iloc[0]["call_count"]) != int(row.call_count)
        ):
            raise RuntimeError(f"IDEA-087 role metadata differs for cat {row.cat_id}")
    return roles, included, cat_truth


def rebuild_inner_roles(
    protocol: dict[str, Any], outer_roles: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outer_fold, split_seed in zip(
        protocol["data"]["outer_folds"], protocol["inner_roles"]["split_seeds"]
    ):
        cell = outer_roles[outer_roles["outer_fold"] == int(outer_fold)].copy()
        development = cell[cell["role"].isin(("train", "validation"))].copy()
        development = development.sort_values("cat_id").reset_index(drop=True)
        splitter = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=int(split_seed)
        )
        labels = development["age_group"].astype(str).to_numpy()
        held_out: list[str] = []
        for inner_fold, (train_pos, validation_pos) in enumerate(
            splitter.split(np.zeros(len(development)), labels)
        ):
            for role_name, positions in (
                ("train", train_pos),
                ("validation", validation_pos),
            ):
                selected = development.iloc[positions]
                if set(selected["age_group"].astype(str)) != set(LABEL_BY_AGE):
                    raise RuntimeError("independently rebuilt inner role lacks a class")
                for row in selected.itertuples(index=False):
                    rows.append(
                        {
                            "outer_fold": int(outer_fold),
                            "inner_fold": int(inner_fold),
                            "split_seed": int(split_seed),
                            "cat_id": str(row.cat_id),
                            "age_group": str(row.age_group),
                            "call_count": int(row.call_count),
                            "role": role_name,
                        }
                    )
            held_out.extend(development.iloc[validation_pos]["cat_id"].astype(str))
        if len(held_out) != len(set(held_out)) or set(held_out) != set(
            development["cat_id"].astype(str)
        ):
            raise RuntimeError("independent inner roles do not form complete OOF")
    result = pd.DataFrame(rows).sort_values(
        ["outer_fold", "inner_fold", "role", "cat_id"]
    ).reset_index(drop=True)
    if len(result) != 999:
        raise RuntimeError("independent inner role row count is not 999")
    return result


def training_identity(
    calls: pd.DataFrame, cat_ids: Iterable[str]
) -> dict[str, Any]:
    cats = sorted(set(str(value) for value in cat_ids))
    selected = calls[calls["cat_id"].isin(cats)].copy()
    labels = selected["true_label"].to_numpy(dtype=np.int64)
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("independent training identity found a missing class")
    indices = selected["call_index"].to_numpy(dtype="<i8")
    weights = (len(labels) / (3.0 * counts)).astype(np.float32)
    return {
        "cats": len(cats),
        "calls": int(len(selected)),
        "cat_ids_sha256": text_sha256("\n".join(cats)),
        "call_indices_sha256": hashlib.sha256(np.sort(indices).tobytes()).hexdigest(),
        "call_class_counts": counts.astype(int).tolist(),
        "call_class_weights": weights.astype(float).tolist(),
    }


def validate_probabilities(frame: pd.DataFrame, context: str) -> None:
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"{context}: nonfinite probabilities")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise RuntimeError(f"{context}: probability outside [0,1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0):
        raise RuntimeError(f"{context}: probabilities are not normalized")


def validate_and_rebuild_predictions(
    fit: dict[str, Any],
    expected_cats: set[str],
    calls_truth: pd.DataFrame,
    context: str,
) -> tuple[pd.DataFrame, float]:
    animal_path = ROOT / fit["prediction_animal_path"]
    call_path = ROOT / fit["prediction_call_path"]
    if sha256(animal_path) != fit["prediction_animal_sha256"]:
        raise RuntimeError(f"{context}: animal prediction hash differs")
    if sha256(call_path) != fit["prediction_call_sha256"]:
        raise RuntimeError(f"{context}: call prediction hash differs")
    animals = pd.read_csv(animal_path, dtype={"cat_id": str})
    calls = pd.read_csv(call_path, dtype={"cat_id": str, "call_id": str})
    if calls["call_index"].duplicated().any() or calls["call_id"].duplicated().any():
        raise RuntimeError(f"{context}: duplicate call prediction")
    if animals["cat_id"].duplicated().any():
        raise RuntimeError(f"{context}: duplicate animal prediction")
    if set(calls["cat_id"].astype(str)) != expected_cats:
        raise RuntimeError(f"{context}: call prediction role differs")
    if set(animals["cat_id"].astype(str)) != expected_cats:
        raise RuntimeError(f"{context}: animal prediction role differs")
    expected_calls = calls_truth[calls_truth["cat_id"].isin(expected_cats)].copy()
    expected_calls = expected_calls.sort_values("call_index").reset_index(drop=True)
    observed_calls = calls.sort_values("call_index").reset_index(drop=True)
    for column in ("call_index", "call_id", "cat_id", "true_label"):
        if not np.array_equal(
            observed_calls[column].to_numpy(), expected_calls[column].to_numpy()
        ):
            raise RuntimeError(f"{context}: call identity differs in {column}")
    validate_probabilities(calls, f"{context}/calls")
    validate_probabilities(animals, f"{context}/animals")
    grouped = (
        observed_calls.groupby("cat_id", sort=True)
        .agg(
            true_label=("true_label", "first"),
            label_count=("true_label", "nunique"),
            call_count=("call_id", "size"),
            prob_kitten=("prob_kitten", "mean"),
            prob_adult=("prob_adult", "mean"),
            prob_senior=("prob_senior", "mean"),
        )
        .reset_index()
    )
    if (grouped.pop("label_count") != 1).any():
        raise RuntimeError(f"{context}: a cat has multiple call labels")
    grouped["predicted_label"] = grouped[list(PROBABILITY_COLUMNS)].to_numpy().argmax(
        axis=1
    )
    saved = animals.sort_values("cat_id").reset_index(drop=True)
    grouped = grouped.sort_values("cat_id").reset_index(drop=True)
    for column in ("cat_id", "true_label", "call_count", "predicted_label"):
        if not np.array_equal(saved[column].to_numpy(), grouped[column].to_numpy()):
            raise RuntimeError(f"{context}: call-to-cat mismatch in {column}")
    maximum = float(
        np.abs(
            saved[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
            - grouped[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
        ).max(initial=0.0)
    )
    if maximum > TOLERANCE:
        raise RuntimeError(f"{context}: call-to-cat probability mismatch {maximum}")
    return grouped, maximum


def validate_common_fit_fields(
    fit: dict[str, Any],
    protocol: dict[str, Any],
    expected: dict[str, Any],
    train_identity: dict[str, Any],
    context: str,
) -> None:
    common = {
        **expected,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": protocol["dependencies"]["runner_sha256"],
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "train_role": train_identity,
    }
    compare_nested(common, fit, context, tolerance=1.0e-10)


def verify_inner_history(
    fit: dict[str, Any], protocol: dict[str, Any], metrics: dict[str, Any], context: str
) -> float:
    audit = fit["audit"]
    history = audit["history"]
    patience = int(protocol["fixed_training"]["early_stopping_patience"])
    best_loss = float("inf")
    best_epoch = 1
    stale = 0
    for expected_epoch, row in enumerate(history, start=1):
        if int(row["epoch"]) != expected_epoch:
            raise RuntimeError(f"{context}: non-consecutive history epochs")
        value = float(row["validation_animal_cross_entropy"])
        if value < best_loss - 1.0e-6:
            best_loss = value
            best_epoch = expected_epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience and expected_epoch != len(history):
            raise RuntimeError(f"{context}: history continued after patience")
    if int(audit["best_epoch"]) != best_epoch:
        raise RuntimeError(f"{context}: independently selected best epoch differs")
    if int(audit["stopped_epoch"]) != len(history):
        raise RuntimeError(f"{context}: stopped epoch differs from history")
    if audit.get("fixed_epoch_training") is not False:
        raise RuntimeError(f"{context}: inner fit claims fixed-epoch training")
    return compare_nested(metrics, audit["evaluation_metrics"], f"{context}.metrics")


def select_configuration(
    scores: list[dict[str, Any]], tolerance: float
) -> tuple[dict[str, Any], list[str]]:
    best_f1 = max(float(row["mean_macro_f1"]) for row in scores)
    candidates = [
        row
        for row in scores
        if float(row["mean_macro_f1"]) >= best_f1 - tolerance - 1.0e-12
    ]
    candidates.sort(
        key=lambda row: (
            float(row["mean_brier"]),
            float(row["macro_f1_sample_sd"]),
            -float(row["mean_plain_accuracy"]),
            str(row["config_id"]),
        )
    )
    return candidates[0], [str(row["config_id"]) for row in candidates]


def verify_locks(protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("status") != "locked_before_cpu_preflight":
        raise RuntimeError("IDEA-087 protocol is not locked")
    dependencies = protocol["dependencies"]
    for path_key, hash_key in (
        ("plan_path", "plan_sha256"),
        ("runner_path", "runner_sha256"),
        ("tests_path", "tests_sha256"),
    ):
        path = ROOT / dependencies[path_key]
        if not path.is_file() or sha256(path) != dependencies[hash_key]:
            raise RuntimeError(f"IDEA-087 dependency lock differs: {path_key}")
    preflight_path = RUN_ROOT / protocol["outputs"]["cpu_preflight"]
    preflight = read_json(preflight_path)
    expected = {
        "status": "GO",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": dependencies["runner_sha256"],
        "tests_sha256": dependencies["tests_sha256"],
        "expected_inner_fits": 576,
        "maximum_outer_refits": 72,
        "outer_test_predictions_generated": False,
        "device": "cpu",
    }
    compare_nested(expected, preflight, "cpu_preflight")
    return preflight


def rebuild_selection() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(PROTOCOL_PATH)
    preflight = verify_locks(protocol)
    roles, calls_truth, _ = load_locked_inputs(protocol)
    inner_roles = rebuild_inner_roles(protocol, roles)
    saved_roles_path = RUN_ROOT / protocol["outputs"]["inner_roles"]
    saved_roles = pd.read_csv(saved_roles_path, dtype={"cat_id": str})
    pd.testing.assert_frame_equal(inner_roles, saved_roles, check_dtype=False)
    if sha256(saved_roles_path) != preflight["inner_roles_sha256"]:
        raise RuntimeError("IDEA-087 inner roles changed after preflight")
    configs = configuration_map(protocol)
    fit_lookup: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    animal_lookup: dict[tuple[str, str, int, int, int], pd.DataFrame] = {}
    evidence_lines: list[str] = []
    max_metric_difference = 0.0
    max_call_to_cat_difference = 0.0
    prediction_files = 0
    for outer_fold in protocol["data"]["outer_folds"]:
        for inner_fold in range(3):
            cell = inner_roles[
                (inner_roles["outer_fold"] == int(outer_fold))
                & (inner_roles["inner_fold"] == int(inner_fold))
            ]
            train_cats = set(cell[cell["role"] == "train"]["cat_id"].astype(str))
            validation_cats = set(
                cell[cell["role"] == "validation"]["cat_id"].astype(str)
            )
            train_id = training_identity(calls_truth, train_cats)
            validation_hash = text_sha256("\n".join(sorted(validation_cats)))
            for base_seed in protocol["search"]["search_base_seeds"]:
                for config in protocol["search"]["configurations"]:
                    for pipeline in PIPELINES:
                        path = inner_fit_path(
                            pipeline,
                            str(config["config_id"]),
                            int(outer_fold),
                            inner_fold,
                            int(base_seed),
                        )
                        fit = read_json(path)
                        context = (
                            f"inner/{pipeline}/{config['config_id']}/outer={outer_fold}/"
                            f"inner={inner_fold}/seed={base_seed}"
                        )
                        expected = {
                            "status": "complete",
                            "stage": "inner",
                            "pipeline": pipeline,
                            "config_id": config["config_id"],
                            "config": config,
                            "outer_fold": int(outer_fold),
                            "inner_fold": inner_fold,
                            "search_base_seed": int(base_seed),
                            "full_seed": full_search_seed(
                                int(base_seed), int(outer_fold), inner_fold
                            ),
                            "validation_cat_ids_sha256": validation_hash,
                            "outer_test_accessed": False,
                        }
                        validate_common_fit_fields(
                            fit, protocol, expected, train_id, context
                        )
                        animals, reconstruction_difference = (
                            validate_and_rebuild_predictions(
                                fit, validation_cats, calls_truth, context
                            )
                        )
                        metrics = metric_bundle(animals)
                        max_metric_difference = max(
                            max_metric_difference,
                            verify_inner_history(fit, protocol, metrics, context),
                        )
                        max_call_to_cat_difference = max(
                            max_call_to_cat_difference, reconstruction_difference
                        )
                        key = (
                            pipeline,
                            str(config["config_id"]),
                            int(outer_fold),
                            inner_fold,
                            int(base_seed),
                        )
                        fit_lookup[key] = fit
                        animal_lookup[key] = animals
                        relative = path.relative_to(ROOT).as_posix()
                        evidence_lines.append(f"{relative}:{sha256(path)}")
                        prediction_files += 2
    if len(fit_lookup) != 576 or prediction_files != 1152:
        raise RuntimeError("IDEA-087 independent inner matrix is incomplete")
    saved_lock_path = RUN_ROOT / protocol["outputs"]["selection_lock"]
    saved_lock = read_json(saved_lock_path)
    expected_top = {
        "status": "complete_locked_before_outer_evaluation",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": protocol["dependencies"]["runner_sha256"],
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "inner_roles_sha256": sha256(saved_roles_path),
        "inner_fits": 576,
        "inner_fit_evidence_sha256": text_sha256(
            "\n".join(sorted(evidence_lines))
        ),
        "complete_locks": 12,
        "outer_test_accessed": False,
    }
    compare_nested(expected_top, saved_lock, "selection_lock")
    saved_rows = {
        (str(row["pipeline"]), int(row["outer_fold"])): row
        for row in saved_lock["selection_locks"]
    }
    if len(saved_rows) != 12:
        raise RuntimeError("IDEA-087 saved selection identities are incomplete")
    rebuilt_rows: list[dict[str, Any]] = []
    tolerance = float(protocol["selection"]["macro_f1_candidate_tolerance"])
    for pipeline in PIPELINES:
        for outer_fold in protocol["data"]["outer_folds"]:
            development_cats = set(
                roles[
                    (roles["outer_fold"] == int(outer_fold))
                    & roles["role"].isin(("train", "validation"))
                ]["cat_id"].astype(str)
            )
            scores: list[dict[str, Any]] = []
            for config_id, config in configs.items():
                seed_evidence = []
                best_epochs = []
                for base_seed in protocol["search"]["search_base_seeds"]:
                    parts = []
                    for inner_fold in range(3):
                        key = (
                            pipeline,
                            config_id,
                            int(outer_fold),
                            inner_fold,
                            int(base_seed),
                        )
                        parts.append(animal_lookup[key])
                        best_epochs.append(int(fit_lookup[key]["audit"]["best_epoch"]))
                    oof = pd.concat(parts, ignore_index=True)
                    if oof["cat_id"].duplicated().any() or set(
                        oof["cat_id"].astype(str)
                    ) != development_cats:
                        raise RuntimeError("independent inner seed OOF is incomplete")
                    bundle = metric_bundle(oof)
                    seed_evidence.append(
                        {
                            "search_base_seed": int(base_seed),
                            "cats": int(len(oof)),
                            "cat_ids_sha256": text_sha256(
                                "\n".join(sorted(oof["cat_id"].astype(str)))
                            ),
                            "metrics": bundle,
                        }
                    )
                f1_values = np.asarray(
                    [row["metrics"]["macro_f1"] for row in seed_evidence],
                    dtype=np.float64,
                )
                scores.append(
                    {
                        "config_id": config_id,
                        "config": config,
                        "mean_macro_f1": float(f1_values.mean()),
                        "macro_f1_sample_sd": float(f1_values.std(ddof=1)),
                        "mean_brier": float(
                            np.mean([row["metrics"]["brier"] for row in seed_evidence])
                        ),
                        "mean_plain_accuracy": float(
                            np.mean(
                                [
                                    row["metrics"]["plain_accuracy"]
                                    for row in seed_evidence
                                ]
                            )
                        ),
                        "seed_oof": seed_evidence,
                        "best_epochs_1_based": best_epochs,
                        "refit_epoch": refit_epoch(best_epochs),
                    }
                )
            selected, pool = select_configuration(scores, tolerance)
            original = next(row for row in scores if row["config_id"] == "q00")
            rebuilt = {
                "pipeline": pipeline,
                "outer_fold": int(outer_fold),
                "selected_config_id": selected["config_id"],
                "selected_refit_epoch": selected["refit_epoch"],
                "original_config_id": "q00",
                "original_refit_epoch": original["refit_epoch"],
                "candidate_pool_config_ids": pool,
                "config_scores": scores,
            }
            max_metric_difference = max(
                max_metric_difference,
                compare_nested(
                    rebuilt,
                    saved_rows[(pipeline, int(outer_fold))],
                    f"selection/{pipeline}/outer={outer_fold}",
                ),
            )
            rebuilt_rows.append(rebuilt)
    audit = {
        "schema_version": "1.0",
        "status": "PASS",
        "mode": "selection",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": protocol["dependencies"]["runner_sha256"],
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "selection_lock_sha256": sha256(saved_lock_path),
        "inner_roles_sha256": sha256(saved_roles_path),
        "inner_fits_verified": len(fit_lookup),
        "prediction_files_verified": prediction_files,
        "call_to_cat_reconstructions": len(fit_lookup),
        "maximum_call_to_cat_probability_difference": max_call_to_cat_difference,
        "maximum_saved_metric_difference": max_metric_difference,
        "selection_locks_rebuilt": len(rebuilt_rows),
        "selected_configs": [
            {
                "pipeline": row["pipeline"],
                "outer_fold": row["outer_fold"],
                "selected_config_id": row["selected_config_id"],
                "selected_refit_epoch": row["selected_refit_epoch"],
                "original_refit_epoch": row["original_refit_epoch"],
            }
            for row in rebuilt_rows
        ],
        "outer_test_accessed": False,
    }
    return audit, saved_lock


def flatten_metrics(bundle: dict[str, Any]) -> dict[str, float]:
    return {
        "plain_accuracy": float(bundle["plain_accuracy"]),
        "macro_f1": float(bundle["macro_f1"]),
        "balanced_accuracy": float(bundle["balanced_accuracy"]),
        "kitten_recall": float(bundle["class_recall"]["kitten"]),
        "adult_recall": float(bundle["class_recall"]["adult"]),
        "senior_recall": float(bundle["class_recall"]["senior"]),
        "cross_entropy": float(bundle["cross_entropy"]),
        "brier": float(bundle["brier"]),
    }


def comparison_summary(
    rows: list[dict[str, Any]], candidate: str, control: str
) -> dict[str, Any]:
    direction_tolerance = 1.0e-12

    def directions(values: np.ndarray) -> dict[str, int]:
        return {
            "positive": int((values > direction_tolerance).sum()),
            "tied": int((np.abs(values) <= direction_tolerance).sum()),
            "negative": int((values < -direction_tolerance).sum()),
        }

    result: dict[str, Any] = {"candidate": candidate, "control": control}
    for metric in (
        "plain_accuracy",
        "macro_f1",
        "balanced_accuracy",
        "kitten_recall",
        "adult_recall",
        "senior_recall",
    ):
        values = np.asarray(
            [row[f"{candidate}_{metric}"] - row[f"{control}_{metric}"] for row in rows],
            dtype=np.float64,
        )
        result[f"{metric}_delta"] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            **directions(values),
            "worst": float(values.min()),
            "best": float(values.max()),
        }
    for metric in ("cross_entropy", "brier"):
        values = np.asarray(
            [row[f"{control}_{metric}"] - row[f"{candidate}_{metric}"] for row in rows],
            dtype=np.float64,
        )
        result[f"{metric}_gain"] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            **directions(values),
        }
    result["direction_tolerance"] = direction_tolerance
    return result


def verify_full() -> dict[str, Any]:
    selection_audit, selection = rebuild_selection()
    protocol = read_json(PROTOCOL_PATH)
    roles, calls_truth, _ = load_locked_inputs(protocol)
    configs = configuration_map(protocol)
    selection_sha = selection_audit["selection_lock_sha256"]
    locks = {
        (str(row["pipeline"]), int(row["outer_fold"])): row
        for row in selection["selection_locks"]
    }
    animals_by_fit: dict[tuple[str, str, int, int], pd.DataFrame] = {}
    physical = 0
    aliases = 0
    prediction_files: set[str] = set()
    max_call_to_cat_difference = 0.0
    max_metric_difference = float(selection_audit["maximum_saved_metric_difference"])
    for pipeline in PIPELINES:
        for policy in POLICIES:
            for outer_fold in protocol["data"]["outer_folds"]:
                lock = locks[(pipeline, int(outer_fold))]
                config_id = str(lock[f"{policy}_config_id"])
                epoch = int(lock[f"{policy}_refit_epoch"])
                development_cats = set(
                    roles[
                        (roles["outer_fold"] == int(outer_fold))
                        & roles["role"].isin(("train", "validation"))
                    ]["cat_id"].astype(str)
                )
                test_cats = set(
                    roles[
                        (roles["outer_fold"] == int(outer_fold))
                        & (roles["role"] == "test")
                    ]["cat_id"].astype(str)
                )
                train_id = training_identity(calls_truth, development_cats)
                for base_seed in protocol["refit"]["refit_base_seeds"]:
                    path = outer_fit_path(
                        pipeline, policy, int(outer_fold), int(base_seed)
                    )
                    fit = read_json(path)
                    context = (
                        f"outer/{pipeline}/{policy}/outer={outer_fold}/seed={base_seed}"
                    )
                    expected = {
                        "stage": "outer",
                        "pipeline": pipeline,
                        "policy": policy,
                        "config_id": config_id,
                        "config": configs[config_id],
                        "refit_epoch": epoch,
                        "outer_fold": int(outer_fold),
                        "refit_base_seed": int(base_seed),
                        "full_seed": full_refit_seed(
                            int(base_seed), int(outer_fold)
                        ),
                        "selection_lock_sha256": selection_sha,
                        "outer_test_accessed": True,
                    }
                    validate_common_fit_fields(
                        fit, protocol, expected, train_id, context
                    )
                    status = str(fit.get("status"))
                    if status == "complete":
                        if fit.get("physical_fit") is not True:
                            raise RuntimeError(f"{context}: complete fit is not physical")
                        physical += 1
                    elif status == "alias":
                        if fit.get("physical_fit") is not False:
                            raise RuntimeError(f"{context}: alias claims a physical fit")
                        source_path = ROOT / fit["alias_source_summary_path"]
                        if sha256(source_path) != fit["alias_source_summary_sha256"]:
                            raise RuntimeError(f"{context}: alias source hash differs")
                        source = read_json(source_path)
                        expected_source_path = outer_fit_path(
                            pipeline, "original", int(outer_fold), int(base_seed)
                        )
                        if source_path.resolve() != expected_source_path.resolve():
                            raise RuntimeError(f"{context}: alias source path is not original")
                        if source.get("policy") != "original":
                            raise RuntimeError(f"{context}: alias source policy is not original")
                        for key in (
                            "pipeline",
                            "config_id",
                            "config",
                            "refit_epoch",
                            "outer_fold",
                            "refit_base_seed",
                            "full_seed",
                            "selection_lock_sha256",
                            "train_role",
                            "prediction_animal_path",
                            "prediction_animal_sha256",
                            "prediction_call_path",
                            "prediction_call_sha256",
                            "audit",
                        ):
                            compare_nested(
                                source[key], fit[key], f"{context}.alias.{key}"
                            )
                        if source.get("status") != "complete" or source.get(
                            "physical_fit"
                        ) is not True:
                            raise RuntimeError(f"{context}: alias source is not physical")
                        aliases += 1
                    else:
                        raise RuntimeError(f"{context}: invalid outer fit status {status}")
                    animals, reconstruction_difference = (
                        validate_and_rebuild_predictions(
                            fit, test_cats, calls_truth, context
                        )
                    )
                    metrics = metric_bundle(animals)
                    audit = fit["audit"]
                    if (
                        audit.get("fixed_epoch_training") is not True
                        or int(audit["best_epoch"]) != epoch
                        or int(audit["stopped_epoch"]) != epoch
                        or len(audit["history"]) != epoch
                    ):
                        raise RuntimeError(f"{context}: fixed-epoch audit differs")
                    max_metric_difference = max(
                        max_metric_difference,
                        compare_nested(
                            metrics, audit["evaluation_metrics"], f"{context}.metrics"
                        ),
                    )
                    max_call_to_cat_difference = max(
                        max_call_to_cat_difference, reconstruction_difference
                    )
                    prediction_files.add(str(fit["prediction_animal_path"]))
                    prediction_files.add(str(fit["prediction_call_path"]))
                    animals_by_fit[
                        (pipeline, policy, int(outer_fold), int(base_seed))
                    ] = animals
    if physical + aliases != 72 or len(animals_by_fit) != 72:
        raise RuntimeError("IDEA-087 logical outer matrix is not 72 records")
    selected_q00_cells = sum(
        int(str(row["selected_config_id"]) == "q00")
        for row in selection["selection_locks"]
    )
    expected_aliases = 3 * selected_q00_cells
    if aliases != expected_aliases or physical != 72 - expected_aliases:
        raise RuntimeError("IDEA-087 physical/alias count differs from selection")
    if len(prediction_files) != 2 * physical:
        raise RuntimeError("IDEA-087 unique outer prediction count differs from physical fits")
    all_cats = set(calls_truth["cat_id"].astype(str))
    seed_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for base_seed in protocol["refit"]["refit_base_seeds"]:
        seed_row: dict[str, Any] = {"refit_base_seed": int(base_seed)}
        for outer_fold in protocol["data"]["outer_folds"]:
            cell_row: dict[str, Any] = {
                "refit_base_seed": int(base_seed),
                "outer_fold": int(outer_fold),
            }
            for pipeline in PIPELINES:
                for policy in POLICIES:
                    name = f"{pipeline}_{policy}"
                    flat = flatten_metrics(
                        metric_bundle(
                            animals_by_fit[
                                (pipeline, policy, int(outer_fold), int(base_seed))
                            ]
                        )
                    )
                    for metric, value in flat.items():
                        cell_row[f"{name}_{metric}"] = value
            cell_rows.append(cell_row)
        for pipeline in PIPELINES:
            for policy in POLICIES:
                name = f"{pipeline}_{policy}"
                oof = pd.concat(
                    [
                        animals_by_fit[(pipeline, policy, int(fold), int(base_seed))]
                        for fold in protocol["data"]["outer_folds"]
                    ],
                    ignore_index=True,
                )
                if (
                    len(oof) != 111
                    or oof["cat_id"].duplicated().any()
                    or set(oof["cat_id"].astype(str)) != all_cats
                ):
                    raise RuntimeError("independent outer folds do not form 111-cat OOF")
                for metric, value in flatten_metrics(metric_bundle(oof)).items():
                    seed_row[f"{name}_{metric}"] = value
        seed_rows.append(seed_row)
    means: dict[str, Any] = {}
    metric_names = tuple(flatten_metrics(metric_bundle(next(iter(animals_by_fit.values())))))
    for pipeline in PIPELINES:
        for policy in POLICIES:
            name = f"{pipeline}_{policy}"
            means[name] = {
                metric: float(
                    np.mean([row[f"{name}_{metric}"] for row in seed_rows])
                )
                for metric in metric_names
            }
    comparisons: dict[str, Any] = {}
    for pipeline in PIPELINES:
        comparisons[f"{pipeline}_tuned_minus_fixed_seed_oof"] = comparison_summary(
            seed_rows, f"{pipeline}_selected", f"{pipeline}_original"
        )
        comparisons[f"{pipeline}_tuned_minus_fixed_cells"] = comparison_summary(
            cell_rows, f"{pipeline}_selected", f"{pipeline}_original"
        )
    for policy, label in (("original", "fixed"), ("selected", "tuned")):
        for pipeline in PIPELINES[1:]:
            comparisons[
                f"{pipeline}_{label}_minus_A0_{label}_seed_oof"
            ] = comparison_summary(
                seed_rows, f"{pipeline}_{policy}", f"{PIPELINES[0]}_{policy}"
            )
            comparisons[
                f"{pipeline}_{label}_minus_A0_{label}_cells"
            ] = comparison_summary(
                cell_rows, f"{pipeline}_{policy}", f"{PIPELINES[0]}_{policy}"
            )
        comparisons[f"C1_minus_U1_{label}_seed_oof"] = comparison_summary(
            seed_rows, f"{PIPELINES[2]}_{policy}", f"{PIPELINES[1]}_{policy}"
        )
        comparisons[f"C1_minus_U1_{label}_cells"] = comparison_summary(
            cell_rows, f"{PIPELINES[2]}_{policy}", f"{PIPELINES[1]}_{policy}"
        )
    summary_path = RUN_ROOT / protocol["outputs"]["summary"]
    saved_summary = read_json(summary_path)
    rebuilt_core = {
        "status": "complete",
        "exploratory_not_independent_confirmation": True,
        "no_global_gate": True,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": protocol["dependencies"]["runner_sha256"],
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "selection_lock_sha256": selection_sha,
        "physical_outer_fits": physical,
        "outer_alias_records": aliases,
        "maximum_outer_refits": 72,
        "refit_seed_oof_estimates": 3,
        "paired_refit_seed_by_outer_fold_cells": 12,
        "repeated_animal_occurrences_per_pipeline_policy": 333,
        "unique_cats": 111,
        "pipeline_policy_seed_oof_means": means,
        "comparisons": comparisons,
        "seed_oof_results": seed_rows,
        "cell_results": cell_rows,
        "outer_test_accessed": True,
    }
    max_metric_difference = max(
        max_metric_difference,
        compare_nested(rebuilt_core, saved_summary, "initial_evaluation_summary"),
    )
    authorization_path = RUN_ROOT / "outer_authorization_record.json"
    authorization = read_json(authorization_path)
    expected_authorization = {
        "status": "director_authorized_outer_transition",
        "director_authorized": True,
        "selection_lock_sha256": selection_sha,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": protocol["dependencies"]["runner_sha256"],
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "outer_test_accessed_by_this_stage": True,
    }
    compare_nested(expected_authorization, authorization, "outer_authorization")
    final_manifest_path = RUN_ROOT / "final_stage_manifest.json"
    final_manifest = read_json(final_manifest_path)
    expected_final_manifest = {
        "status": "complete",
        "outer_authorization_record_sha256": sha256(authorization_path),
        "selection_lock_sha256": selection_sha,
        "summary_sha256": sha256(summary_path),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": protocol["dependencies"]["runner_sha256"],
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "outer_test_accessed": True,
    }
    compare_nested(expected_final_manifest, final_manifest, "final_stage_manifest")
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "mode": "full",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": protocol["dependencies"]["runner_sha256"],
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "selection_lock_sha256": selection_sha,
        "summary_sha256": sha256(summary_path),
        "inner_fits_verified": selection_audit["inner_fits_verified"],
        "inner_prediction_files_verified": selection_audit[
            "prediction_files_verified"
        ],
        "selection_locks_rebuilt": selection_audit["selection_locks_rebuilt"],
        "outer_logical_records_verified": len(animals_by_fit),
        "outer_physical_fits": physical,
        "outer_alias_records": aliases,
        "outer_unique_prediction_files_verified": len(prediction_files),
        "outer_call_to_cat_reconstructions": len(animals_by_fit),
        "maximum_call_to_cat_probability_difference": max(
            float(selection_audit["maximum_call_to_cat_probability_difference"]),
            max_call_to_cat_difference,
        ),
        "maximum_saved_metric_difference": max_metric_difference,
        "pipeline_policy_seed_oof_means": means,
        "comparisons": comparisons,
        "outer_test_accessed_for_final_scoring_only": True,
        "claim_boundary": "Internal nested optimization on the historically studied 111 cats; not independent-animal confirmation.",
    }


def main() -> None:
    args = parse_args()
    if args.mode == "selection":
        audit, _ = rebuild_selection()
        output = RUN_ROOT / "selection/independent_selection_audit.json"
    else:
        audit = verify_full()
        output = RUN_ROOT / "independent_results_audit.json"
    write_json(output, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
