"""Independent file and gate audit for completed IDEA-082 results."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "meowagenet_idea082_age_acoustic_group_ablation_v1"
RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_idea082_age_acoustic_group_ablation.py"
PROTOCOL_PATH = ROOT / "configs" / "protocol" / "meowagenet_idea082_age_acoustic_group_ablation_v1.json"
OUTPUT = RUN_ROOT / "independent_results_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_runner():
    spec = importlib.util.spec_from_file_location("idea082_result_audit_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load IDEA-082 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assert_finite(value, path: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise RuntimeError(f"Non-finite numeric value at {path}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")


def close(left: float, right: float, name: str) -> None:
    if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"Metric mismatch for {name}: {left} != {right}")


def bundle(runner, frame: pd.DataFrame) -> dict:
    return {
        "metrics": runner.idea068.idea051.animal_metrics(frame),
        "cross_entropy": runner.idea068.idea051.animal_cross_entropy(frame),
        "brier": runner.idea068.brier(frame),
    }


def add_metrics(runner, row: dict, bundles: dict) -> None:
    for pipeline in runner.PIPELINES:
        row[f"{pipeline}_macro_f1"] = bundles[pipeline]["metrics"]["macro_f1"]
        row[f"{pipeline}_balanced_accuracy"] = bundles[pipeline]["metrics"][
            "balanced_accuracy"
        ]
        row[f"{pipeline}_cross_entropy"] = bundles[pipeline]["cross_entropy"]
        row[f"{pipeline}_brier"] = bundles[pipeline]["brier"]
        row[f"{pipeline}_senior_recall"] = bundles[pipeline]["metrics"][
            "per_class"
        ]["senior"]["recall"]
    for group in runner.GROUPS:
        real = f"{group}_real"
        shuffled = f"{group}_shuffled"
        row[f"{group}_information_macro_f1"] = (
            row[f"{real}_macro_f1"] - row[f"{shuffled}_macro_f1"]
        )
        row[f"{group}_utility_macro_f1"] = (
            row[f"{real}_macro_f1"] - row["A0_ast_only_macro_f1"]
        )
    row["G12_real_minus_G1_real_macro_f1"] = (
        row["G12_f0_stability_real_macro_f1"] - row["G1_f0_real_macro_f1"]
    )
    row["G12_real_minus_G2_real_macro_f1"] = (
        row["G12_f0_stability_real_macro_f1"]
        - row["G2_stability_real_macro_f1"]
    )
    row["G12_redundancy_interaction_macro_f1"] = (
        row["G12_f0_stability_real_macro_f1"]
        - row["G1_f0_real_macro_f1"]
        - row["G2_stability_real_macro_f1"]
        + row["A0_ast_only_macro_f1"]
    )


def compare_rows(recomputed: list[dict], reported: list[dict], identity: tuple[str, ...]) -> None:
    if len(recomputed) != len(reported):
        raise RuntimeError("Reported/recomputed row counts differ")
    indexed = {tuple(row[key] for key in identity): row for row in reported}
    for row in recomputed:
        key = tuple(row[name] for name in identity)
        if key not in indexed:
            raise RuntimeError(f"Missing reported row: {key}")
        target = indexed[key]
        if row.keys() != target.keys():
            raise RuntimeError(f"Reported/recomputed columns differ: {key}")
        for name, value in row.items():
            if name in identity:
                if value != target[name]:
                    raise RuntimeError(f"Identity mismatch: {key}/{name}")
            else:
                close(value, target[name], f"{key}/{name}")


def main() -> None:
    runner = load_runner()
    protocol = read_json(PROTOCOL_PATH)
    manifest = read_json(RUN_ROOT / "run_manifest.json")
    summary_path = RUN_ROOT / "initial_evaluation_summary.json"
    compact_path = RUN_ROOT / "run_summary.json"
    summary = read_json(summary_path)
    compact = read_json(compact_path)
    authorization = read_json(RUN_ROOT / "gpu_authorization.json")
    preflight = read_json(RUN_ROOT / "cpu_preflight.json")
    first_cell = read_json(RUN_ROOT / "first_complete_cell_audit.json")
    if manifest["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise RuntimeError("Manifest/protocol hash mismatch")
    if manifest["runner_sha256"] != sha256(RUNNER_PATH):
        raise RuntimeError("Manifest/runner hash mismatch")
    if authorization["protocol_sha256"] != manifest["protocol_sha256"]:
        raise RuntimeError("Authorization/protocol hash mismatch")
    if preflight["status"] != "GO" or first_cell["status"] != "PASS_RESUME_AUTHORIZED":
        raise RuntimeError("Preflight or first-cell audit did not pass")
    expected_identities = {
        (pipeline, base_seed, repeat, fold)
        for pipeline in runner.PIPELINES
        for base_seed in runner.BASE_SEEDS
        for repeat in range(3)
        for fold in range(4)
    }
    paths = list((RUN_ROOT / "fits").rglob("fit_summary.json"))
    if len(paths) != 324:
        raise RuntimeError(f"Expected 324 fit summaries, found {len(paths)}")
    fits: dict[tuple[str, int, int, int], dict] = {}
    animals_by_fit: dict[tuple[str, int, int, int], pd.DataFrame] = {}
    calls_by_fit: dict[tuple[str, int, int, int], pd.DataFrame] = {}
    checkpoint_reload_max = 0.0
    mapping_hashes: dict[tuple[int, int, str], set[str]] = {}
    for path in paths:
        fit = read_json(path)
        identity = (
            fit["pipeline"],
            int(fit["base_seed"]),
            int(fit["repeat"]),
            int(fit["fold"]),
        )
        if identity not in expected_identities or identity in fits:
            raise RuntimeError(f"Unexpected or duplicate fit identity: {identity}")
        full_seed = identity[1] + 10_000 * identity[2] + 100 * identity[3]
        if fit.get("status") != "complete" or fit.get("full_seed") != full_seed:
            raise RuntimeError(f"Incomplete or mis-seeded fit: {identity}")
        if fit.get("outer_test_accessed") is not False or fit["audit"].get(
            "outer_test_accessed"
        ) is not False:
            raise RuntimeError(f"Outer-test flag changed: {identity}")
        assert_finite(fit["audit"], str(identity))
        expected_parameters = runner.EXPECTED_PARAMETERS[identity[0]]
        if fit["audit"]["model"]["trainable_parameters"] != expected_parameters:
            raise RuntimeError(f"Parameter mismatch: {identity}")
        reload_difference = float(
            fit["audit"]["checkpoint_reload_max_probability_difference"]
        )
        checkpoint_reload_max = max(checkpoint_reload_max, reload_difference)
        if reload_difference > 1.0e-6:
            raise RuntimeError(f"Checkpoint reload mismatch: {identity}")
        initialization = fit["initialization"]
        if any(initialization["max_logit_difference_vs_A0"].values()):
            raise RuntimeError(f"Initial logits changed: {identity}")
        if not all(initialization["common_AST_state_equal_to_A0"].values()):
            raise RuntimeError(f"Common AST state changed: {identity}")
        if not all(initialization["real_shuffled_full_state_equal"].values()):
            raise RuntimeError(f"Real/shuffled state changed: {identity}")
        shuffle = fit.get("shuffle_audit")
        if identity[0].endswith("_shuffled"):
            if shuffle is None or shuffle["test_rows_materialized"] is not False:
                raise RuntimeError(f"Missing shuffle audit: {identity}")
            for role in ("train", "validation"):
                role_audit = shuffle["roles"][role]
                if role_audit["fixed_points"] != 0 or not role_audit["multiset_equal"]:
                    raise RuntimeError(f"Invalid derangement: {identity}/{role}")
                mapping_hashes.setdefault((identity[2], identity[3], role), set()).add(
                    role_audit["mapping_sha256"]
                )
            if not shuffle["training_statistics_equal"]:
                raise RuntimeError(f"Training statistics changed: {identity}")
        elif shuffle is not None:
            raise RuntimeError(f"Unexpected shuffle audit: {identity}")
        animal_path = ROOT / fit["validation_animal_predictions"]
        call_path = ROOT / fit["validation_call_predictions"]
        if sha256(animal_path) != fit["validation_animal_sha256"]:
            raise RuntimeError(f"Animal hash mismatch: {identity}")
        if sha256(call_path) != fit["validation_call_sha256"]:
            raise RuntimeError(f"Call hash mismatch: {identity}")
        animals = pd.read_csv(animal_path, dtype={"cat_id": str})
        calls = pd.read_csv(call_path, dtype={"cat_id": str, "call_id": str})
        for frame, id_column in ((animals, "cat_id"), (calls, "call_id")):
            probabilities = frame[list(runner.idea068.PROBABILITY_COLUMNS)].to_numpy(float)
            if not np.isfinite(probabilities).all():
                raise RuntimeError(f"Non-finite probability: {identity}")
            if float(np.max(np.abs(probabilities.sum(axis=1) - 1.0))) > 1.0e-5:
                raise RuntimeError(f"Unnormalized probability: {identity}")
            if frame[id_column].duplicated().any():
                raise RuntimeError(f"Duplicate prediction identity: {identity}")
        metrics = bundle(runner, animals)
        close(
            metrics["metrics"]["macro_f1"],
            fit["audit"]["best_validation_animal_metrics"]["macro_f1"],
            f"fit macro-F1 {identity}",
        )
        close(
            metrics["cross_entropy"],
            fit["audit"]["best_validation_animal_cross_entropy"],
            f"fit CE {identity}",
        )
        close(
            metrics["brier"],
            fit["audit"]["best_validation_animal_brier"],
            f"fit Brier {identity}",
        )
        fits[identity] = fit
        animals_by_fit[identity] = animals
        calls_by_fit[identity] = calls
    if set(fits) != expected_identities:
        raise RuntimeError("Fit identity matrix is incomplete")
    if any(len(values) != 1 for values in mapping_hashes.values()):
        raise RuntimeError("Groups used different shuffle mappings within a cell")
    for repeat in range(3):
        for fold in range(4):
            expected_cell = preflight["shuffle_cells"][repeat * 4 + fold]
            for role in ("train", "validation"):
                observed = next(iter(mapping_hashes[(repeat, fold, role)]))
                if observed != expected_cell["mapping_sha256"][role]:
                    raise RuntimeError("Runtime shuffle hash differs from preflight")
    for base_seed in runner.BASE_SEEDS:
        for repeat in range(3):
            for fold in range(4):
                reference_a = animals_by_fit[(runner.PIPELINES[0], base_seed, repeat, fold)]
                reference_c = calls_by_fit[(runner.PIPELINES[0], base_seed, repeat, fold)]
                histories = {}
                for pipeline in runner.PIPELINES:
                    identity = (pipeline, base_seed, repeat, fold)
                    if not reference_a[["cat_id", "true_label"]].equals(
                        animals_by_fit[identity][["cat_id", "true_label"]]
                    ):
                        raise RuntimeError(f"Animal population differs: {identity}")
                    if not reference_c[["call_id", "cat_id", "true_label"]].equals(
                        calls_by_fit[identity][["call_id", "cat_id", "true_label"]]
                    ):
                        raise RuntimeError(f"Call population differs: {identity}")
                    histories[pipeline] = fits[identity]["audit"]["history"]
                common_epochs = min(len(value) for value in histories.values())
                for epoch in range(common_epochs):
                    reference = histories[runner.PIPELINES[0]][epoch]["train_audit"]
                    for pipeline in runner.PIPELINES[1:]:
                        current = histories[pipeline][epoch]["train_audit"]
                        if (
                            reference["cat_order_sha256"]
                            != current["cat_order_sha256"]
                            or reference["call_coverage_sha256"]
                            != current["call_coverage_sha256"]
                        ):
                            raise RuntimeError("Paired batch coverage differs")
    fold_rows = []
    seed_rows = []
    pooled_all = {pipeline: [] for pipeline in runner.PIPELINES}
    for base_seed in runner.BASE_SEEDS:
        for repeat in range(3):
            pooled_seed = {pipeline: [] for pipeline in runner.PIPELINES}
            for fold in range(4):
                bundles = {}
                for pipeline in runner.PIPELINES:
                    frame = animals_by_fit[(pipeline, base_seed, repeat, fold)].copy()
                    frame["base_seed"] = base_seed
                    frame["repeat"] = repeat
                    frame["fold"] = fold
                    pooled_seed[pipeline].append(frame)
                    pooled_all[pipeline].append(frame)
                    bundles[pipeline] = bundle(runner, frame)
                row = {"base_seed": base_seed, "repeat": repeat, "fold": fold}
                add_metrics(runner, row, bundles)
                fold_rows.append(row)
            pooled = {
                pipeline: pd.concat(parts, ignore_index=True)
                for pipeline, parts in pooled_seed.items()
            }
            row = {"base_seed": base_seed, "repeat": repeat}
            add_metrics(
                runner,
                row,
                {pipeline: bundle(runner, frame) for pipeline, frame in pooled.items()},
            )
            seed_rows.append(row)
    compare_rows(fold_rows, summary["fold_results"], ("base_seed", "repeat", "fold"))
    compare_rows(seed_rows, summary["seed_repeat_results"], ("base_seed", "repeat"))
    seed_frame = pd.DataFrame(seed_rows)
    fold_frame = pd.DataFrame(fold_rows)
    split_columns = [
        f"{group}_{kind}_macro_f1"
        for group in runner.GROUPS
        for kind in ("information", "utility")
    ]
    split_frame = (
        fold_frame.groupby(["repeat", "fold"], as_index=False)[split_columns]
        .mean()
        .sort_values(["repeat", "fold"])
        .reset_index(drop=True)
    )
    compare_rows(
        split_frame.to_dict(orient="records"),
        summary["split_cell_results"],
        ("repeat", "fold"),
    )
    pipeline_means = {
        metric: {
            pipeline: float(seed_frame[f"{pipeline}_{metric}"].mean())
            for pipeline in runner.PIPELINES
        }
        for metric in ("macro_f1", "balanced_accuracy", "cross_entropy", "brier")
    }
    gate = protocol["gate"]
    pooled_frames = {
        pipeline: pd.concat(parts, ignore_index=True)
        for pipeline, parts in pooled_all.items()
    }
    recomputed_groups = {}
    for group in runner.GROUPS:
        real = f"{group}_real"
        shuffled = f"{group}_shuffled"
        senior = {}
        for base_seed in runner.BASE_SEEDS:
            selected_real = pooled_frames[real][
                pooled_frames[real]["base_seed"] == base_seed
            ]
            selected_a0 = pooled_frames[runner.PIPELINES[0]][
                pooled_frames[runner.PIPELINES[0]]["base_seed"] == base_seed
            ]
            senior[str(base_seed)] = float(
                runner.idea068.idea051.animal_metrics(selected_real)["per_class"][
                    "senior"
                ]["recall"]
                - runner.idea068.idea051.animal_metrics(selected_a0)["per_class"][
                    "senior"
                ]["recall"]
            )
        gates = {}
        for kind, comparator in (("information", shuffled), ("utility", runner.PIPELINES[0])):
            column = f"{group}_{kind}_macro_f1"
            values = seed_frame[column]
            split_values = split_frame[column]
            per_seed = {
                str(seed): float(seed_frame[seed_frame["base_seed"] == seed][column].mean())
                for seed in runner.BASE_SEEDS
            }
            conditions = {
                "mean_macro_f1_delta": float(values.mean())
                >= gate["minimum_mean_seed_repeat_macro_f1_delta"],
                "positive_base_seed_means": sum(value > 0 for value in per_seed.values())
                >= gate["minimum_positive_base_seed_means"],
                "positive_seed_repeats": int((values > 0).sum())
                >= gate["minimum_positive_seed_repeats"],
                "nonnegative_split_cells": int((split_values >= 0).sum())
                >= gate["minimum_nonnegative_split_cells"],
                "worst_split_cell": float(split_values.min())
                >= gate["minimum_worst_split_cell_delta"],
                "mean_cross_entropy_nonworse": pipeline_means["cross_entropy"][real]
                <= pipeline_means["cross_entropy"][comparator],
                "mean_brier_nonworse": pipeline_means["brier"][real]
                <= pipeline_means["brier"][comparator],
                "mean_balanced_accuracy_nonworse": pipeline_means[
                    "balanced_accuracy"
                ][real]
                >= pipeline_means["balanced_accuracy"][comparator],
            }
            if kind == "utility":
                conditions["per_base_seed_senior_recall_safety"] = all(
                    value >= gate["utility_minimum_per_base_seed_senior_recall_delta"]
                    for value in senior.values()
                )
            reported = summary["group_results"][group]["contrasts"][kind]
            if conditions != reported["conditions"]:
                raise RuntimeError(f"Gate condition mismatch: {group}/{kind}")
            if bool(all(conditions.values())) != reported["gate_passed"]:
                raise RuntimeError(f"Gate result mismatch: {group}/{kind}")
            gates[kind] = bool(all(conditions.values()))
        supported = gates["information"] and gates["utility"]
        if supported != summary["group_results"][group]["source_supported"]:
            raise RuntimeError(f"Source support mismatch: {group}")
        recomputed_groups[group] = {
            "information_gate_passed": gates["information"],
            "utility_gate_passed": gates["utility"],
            "source_supported": supported,
        }
    if compact["completed_fits"] != 324 or compact["expected_fits"] != 324:
        raise RuntimeError("Compact fit counts differ")
    if compact["source_supported"] != {
        group: value["source_supported"] for group, value in recomputed_groups.items()
    }:
        raise RuntimeError("Compact source support differs")
    forbidden = [
        path.relative_to(RUN_ROOT).as_posix()
        for path in RUN_ROOT.rglob("*")
        if path.is_file()
        and ("outer_test" in path.name.lower() or "test_predictions" in path.name.lower())
    ]
    if forbidden:
        raise RuntimeError(f"Unexpected outer-test-like outputs: {forbidden}")
    result = {
        "status": "PASS",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(RUNNER_PATH),
        "summary_sha256": sha256(summary_path),
        "run_summary_sha256": sha256(compact_path),
        "fit_summaries": len(paths),
        "expected_identity_matrix_complete": True,
        "prediction_hashes_valid": True,
        "probabilities_finite_and_normalized": True,
        "checkpoint_reload_max_probability_difference": checkpoint_reload_max,
        "initialization_audits_valid": True,
        "role_local_derangements_match_preflight": True,
        "prediction_populations_equal_within_all_cells": True,
        "paired_batch_coverage_valid": True,
        "fold_metrics_recomputed": len(fold_rows),
        "seed_repeat_metrics_recomputed": len(seed_rows),
        "split_cells_recomputed": len(split_frame),
        "gate_results_recomputed": recomputed_groups,
        "outer_test_accessed": False,
        "outer_test_like_outputs": forbidden,
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
