from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for search_root in (SCRIPTS_ROOT, SRC_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_meowagenet_idea058_top_block_adaptation as base  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea058_strict_nested_v1.json"
)
PLAN_PATH = REPO_ROOT / "plan" / "POST_IDEA058_next_stage_plan.md"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
RUNS_ROOT = REPO_ROOT / "runs"
DEFAULT_OUTPUT_SUBDIR = "meowagenet_idea058_strict_nested_v1"
PIPELINES = base.PIPELINES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the IDEA-058 per-outer-fold strict nested confirmation"
    )
    parser.add_argument(
        "--stage", choices=("select", "smoke", "evaluate", "all"), default="all"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def selection_path(run_root: Path) -> Path:
    return run_root / "selection" / "per_fold_recipe_locks.json"


def dependency_paths(protocol: dict[str, Any]) -> dict[str, Path]:
    return {
        key.removesuffix("_path"): REPO_ROOT / value
        for key, value in protocol["dependencies"].items()
        if key.endswith("_path")
    }


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol["protocol_id"] != "meowagenet-idea058-strict-nested-v1":
        raise RuntimeError("Unexpected IDEA-058 strict nested protocol")
    if protocol["splits"]["selection_boundary"] != "per_repeat_outer_fold":
        raise RuntimeError("Strict selection boundary is not per outer fold")
    selection = protocol["inner_selection"]
    expected_candidate_fits = (
        len(selection["candidate_recipes"])
        * len(selection["repeats"])
        * len(selection["outer_folds"])
    )
    if expected_candidate_fits != int(selection["candidate_fits"]):
        raise RuntimeError("Strict candidate-fit budget is inconsistent")
    for recipe_id, recipe in selection["candidate_recipes"].items():
        if int(recipe["block_count"]) not in (1, 2):
            raise RuntimeError(f"Invalid block count for {recipe_id}")
        if float(recipe["encoder_learning_rate"]) not in (3.0e-6, 1.0e-5):
            raise RuntimeError(f"Invalid encoder learning rate for {recipe_id}")
    initial = protocol["initial_evaluation"]
    expected_outer_fits = (
        len(PIPELINES)
        * len(initial["base_seeds"])
        * len(initial["repeats"])
        * len(initial["outer_folds"])
    )
    if expected_outer_fits != int(initial["total_outer_fits"]):
        raise RuntimeError("Strict outer-fit budget is inconsistent")
    fixed = protocol["fixed_training"]
    if int(fixed["micro_batch_size"]) * int(
        fixed["gradient_accumulation_steps"]
    ) != int(fixed["accumulation_window_calls"]):
        raise RuntimeError("Strict accumulation window is inconsistent")
    checks = {
        PLAN_PATH: protocol["idea"]["sha256"],
        ROLES_PATH: protocol["splits"]["roles_sha256"],
    }
    for key, path in dependency_paths(protocol).items():
        checks[path] = protocol["dependencies"][f"{key}_sha256"]
    for path, expected in checks.items():
        if not path.is_file() or base.sha256(path) != expected:
            raise RuntimeError(f"Strict dependency checksum mismatch: {path}")


def rank_fold_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -round(float(row["best_validation_animal_metrics"]["macro_f1"]), 6),
            float(row["best_validation_animal_cross_entropy"]),
            int(row["parameters"]["trainable"]),
            str(row["recipe_id"]),
        ),
    )


def fold_recipe(
    selection: dict[str, Any], repeat: int, outer_fold: int
) -> dict[str, Any]:
    matches = [
        row
        for row in selection["per_fold_recipe_locks"]
        if int(row["repeat"]) == int(repeat)
        and int(row["outer_fold"]) == int(outer_fold)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one recipe lock for repeat={repeat} fold={outer_fold}"
        )
    return matches[0]


def run_selection(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    output_path = selection_path(run_root)
    if output_path.is_file() and not resume:
        raise FileExistsError(output_path)
    settings = protocol["inner_selection"]
    rows_by_fold: dict[tuple[int, int], list[dict[str, Any]]] = {
        (int(repeat), int(outer_fold)): []
        for repeat in settings["repeats"]
        for outer_fold in settings["outer_folds"]
    }
    all_rows = []
    for recipe_id in settings["candidate_recipes"]:
        protocol["_active_recipe_id"] = recipe_id
        for repeat in settings["repeats"]:
            for outer_fold in settings["outer_folds"]:
                fit_path = (
                    run_root
                    / "selection"
                    / "fits"
                    / recipe_id
                    / f"repeat_{repeat}"
                    / f"fold_{outer_fold}.json"
                )
                if fit_path.is_file():
                    if not resume:
                        raise FileExistsError(fit_path)
                    fit = base.read_json(fit_path)
                else:
                    indices = base.split_utils.fold_indices(
                        store,
                        roles,
                        int(repeat),
                        int(outer_fold),
                        include_test=False,
                    )
                    seed = base.split_utils.full_seed(
                        int(settings["base_seed"]), int(repeat), int(outer_fold)
                    )
                    print(
                        f"IDEA058 STRICT SELECT recipe={recipe_id} repeat={repeat} "
                        f"fold={outer_fold} seed={seed}",
                        flush=True,
                    )
                    _, audit, state, _ = base.fit_inner(
                        PIPELINES[1],
                        protocol,
                        store,
                        indices["train"],
                        indices["validation"],
                        device,
                        seed,
                    )
                    del state
                    fit = {
                        "status": "complete",
                        "stage": "idea058_strict_nested_inner_selection",
                        "outer_test_accessed": False,
                        "selection_boundary": "current_repeat_outer_fold",
                        "recipe_id": recipe_id,
                        "recipe": settings["candidate_recipes"][recipe_id],
                        "repeat": int(repeat),
                        "outer_fold": int(outer_fold),
                        "full_seed": int(seed),
                        **audit,
                    }
                    base.write_json(fit_path, fit)
                if fit["outer_test_accessed"] is not False:
                    raise RuntimeError("Strict candidate fit accessed outer test")
                rows_by_fold[(int(repeat), int(outer_fold))].append(fit)
                all_rows.append(fit)
    locks = []
    for (repeat, outer_fold), rows in sorted(rows_by_fold.items()):
        if len(rows) != len(settings["candidate_recipes"]):
            raise RuntimeError("Strict fold candidate matrix is incomplete")
        ranking = rank_fold_candidates(rows)
        selected = ranking[0]
        locks.append(
            {
                "repeat": repeat,
                "outer_fold": outer_fold,
                "selected_recipe_id": selected["recipe_id"],
                "selected_recipe": selected["recipe"],
                "selected_inner_animal_macro_f1": selected[
                    "best_validation_animal_metrics"
                ]["macro_f1"],
                "selected_inner_animal_cross_entropy": selected[
                    "best_validation_animal_cross_entropy"
                ],
                "candidate_ranking": [row["recipe_id"] for row in ranking],
                "candidate_scores": {
                    row["recipe_id"]: {
                        "animal_macro_f1": row["best_validation_animal_metrics"][
                            "macro_f1"
                        ],
                        "animal_cross_entropy": row[
                            "best_validation_animal_cross_entropy"
                        ],
                        "best_epoch": row["best_epoch"],
                        "trainable_parameters": row["parameters"]["trainable"],
                    }
                    for row in ranking
                },
            }
        )
    counts = Counter(row["selected_recipe_id"] for row in locks)
    record = {
        "status": "complete",
        "stage": "idea058_strict_nested_per_fold_selection",
        "outer_test_accessed": False,
        "protocol_id": protocol["protocol_id"],
        "selection_boundary": "per_repeat_outer_fold",
        "candidate_fits_completed": len(all_rows),
        "fold_locks": len(locks),
        "selection_rule": settings["selection_rule"],
        "selected_recipe_counts": dict(sorted(counts.items())),
        "per_fold_recipe_locks": locks,
        "candidate_fit_aggregate": {
            recipe_id: base.summarize_selection_rows(
                [row for row in all_rows if row["recipe_id"] == recipe_id]
            )
            for recipe_id in settings["candidate_recipes"]
        },
    }
    base.write_json(output_path, record)
    print(json.dumps(record, indent=2), flush=True)


def environment_lock(protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    paths = dependency_paths(protocol)
    return {
        "schema_version": "1.0",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "git_revision": base.git_revision(),
        "protocol_sha256": base.sha256(PROTOCOL_PATH),
        "runner_sha256": base.sha256(Path(__file__)),
        "idea_sha256": base.sha256(PLAN_PATH),
        "roles_sha256": base.sha256(ROLES_PATH),
        "fbank_sha256": base.sha256(paths["fbank"]),
        "global_embedding_sha256": base.sha256(paths["global_embedding"]),
        "ast_checkpoint": protocol["fixed_training"]["checkpoint"],
        "ast_checkpoint_revision": protocol["fixed_training"]["revision"],
    }


def run_smoke(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    output_path = run_root / "smoke" / "summary.json"
    if output_path.is_file() and not resume:
        raise FileExistsError(output_path)
    selection = base.read_json(selection_path(run_root))
    settings = protocol["smoke"]
    recipe_lock = fold_recipe(
        selection, int(settings["repeat"]), int(settings["outer_fold"])
    )
    protocol["_active_recipe_id"] = recipe_lock["selected_recipe_id"]
    indices = base.split_utils.fold_indices(
        store,
        roles,
        int(settings["repeat"]),
        int(settings["outer_fold"]),
        include_test=False,
    )
    seed = base.split_utils.full_seed(
        int(settings["base_seed"]),
        int(settings["repeat"]),
        int(settings["outer_fold"]),
    )
    initial = base.initialization_audit(
        protocol,
        store,
        indices["train"],
        indices["validation"],
        device,
        seed,
    )
    fits = []
    for pipeline in PIPELINES:
        best_epoch, audit, state, before = base.fit_inner(
            pipeline,
            protocol,
            store,
            indices["train"],
            indices["validation"],
            device,
            seed,
            max_epochs_override=int(settings["epochs"]),
        )
        checkpoint_path = run_root / "smoke" / f"{pipeline}_checkpoint.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "pipeline": pipeline,
                "selected_recipe_id": recipe_lock["selected_recipe_id"],
                "best_epoch": int(best_epoch),
                "state_dict": state,
            },
            checkpoint_path,
        )
        difference = base.reload_probability_difference(
            pipeline,
            protocol,
            store,
            indices["train"],
            indices["validation"],
            checkpoint_path,
            before,
            device,
            seed,
        )
        if difference > 1.0e-6:
            raise RuntimeError(f"Strict checkpoint reload mismatch for {pipeline}")
        fits.append(
            {
                "pipeline": pipeline,
                "best_epoch": int(best_epoch),
                "checkpoint_path": base.repo_relative(checkpoint_path),
                "checkpoint_sha256": base.sha256(checkpoint_path),
                "reload_probability_max_abs_difference": difference,
                "audit": audit,
            }
        )
        del state
    environment_path = run_root / "environment_lock.json"
    base.write_json(environment_path, environment_lock(protocol, device))
    summary = {
        "status": "complete",
        "stage": "idea058_strict_nested_inner_only_smoke",
        "outer_test_accessed": False,
        "selection_boundary": "current_repeat_outer_fold",
        "recipe_lock": recipe_lock,
        "repeat": int(settings["repeat"]),
        "outer_fold": int(settings["outer_fold"]),
        "epochs": int(settings["epochs"]),
        "initialization_audit": initial,
        "fits": fits,
        "environment_lock_path": base.repo_relative(environment_path),
        "environment_lock_sha256": base.sha256(environment_path),
    }
    base.write_json(output_path, summary)
    revision = base.git_revision()
    if revision is None:
        raise RuntimeError("Strict execution lock requires a Git revision")
    lock = {
        "schema_version": "1.0",
        "status": "locked_for_idea058_strict_nested_evaluation",
        "code_commit": revision,
        "selection_boundary": "per_repeat_outer_fold",
        "per_fold_selection_sha256": base.sha256(selection_path(run_root)),
        "per_fold_recipe_locks": selection["per_fold_recipe_locks"],
        "smoke_summary_sha256": base.sha256(output_path),
        "environment_lock_sha256": base.sha256(environment_path),
        "protocol_sha256": base.sha256(PROTOCOL_PATH),
        "runner_sha256": base.sha256(Path(__file__)),
        "idea_sha256": base.sha256(PLAN_PATH),
        "roles_sha256": base.sha256(ROLES_PATH),
        "initial_evaluation": protocol["initial_evaluation"],
        "git_blobs": {
            base.repo_relative(path): base.git_blob_object_id(revision, path)
            for path in (PROTOCOL_PATH, Path(__file__).resolve(), PLAN_PATH)
        },
    }
    base.write_json(run_root / "execution_lock.json", lock)
    print(json.dumps(summary, indent=2), flush=True)


def verify_execution_lock(
    run_root: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    lock_path = run_root / "execution_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("Run strict nested smoke before outer evaluation")
    lock = base.read_json(lock_path)
    if lock["status"] != "locked_for_idea058_strict_nested_evaluation":
        raise RuntimeError("Strict execution lock status is invalid")
    checks = {
        PROTOCOL_PATH: lock["protocol_sha256"],
        Path(__file__).resolve(): lock["runner_sha256"],
        PLAN_PATH: lock["idea_sha256"],
        ROLES_PATH: lock["roles_sha256"],
        selection_path(run_root): lock["per_fold_selection_sha256"],
        run_root / "smoke" / "summary.json": lock["smoke_summary_sha256"],
        run_root / "environment_lock.json": lock["environment_lock_sha256"],
    }
    for path, expected in checks.items():
        if not path.is_file() or base.sha256(path) != expected:
            raise RuntimeError(f"Strict execution-lock file changed: {path}")
    for path in (PROTOCOL_PATH, Path(__file__).resolve(), PLAN_PATH):
        if base.git_blob_object_id(
            lock["code_commit"], path
        ) != base.worktree_blob_object_id(path):
            raise RuntimeError(f"Strict locked commit blob differs: {path}")
    if lock["initial_evaluation"] != protocol["initial_evaluation"]:
        raise RuntimeError("Strict evaluation matrix differs from lock")
    selection = base.read_json(selection_path(run_root))
    if lock["per_fold_recipe_locks"] != selection["per_fold_recipe_locks"]:
        raise RuntimeError("Strict per-fold recipe locks differ")
    return lock


def parameter_profiles(completed: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for pipeline in PIPELINES:
        fits = [fit for fit in completed if fit["pipeline"] == pipeline]
        profiles = {}
        for fit in fits:
            recipe_id = (
                "fixed_frozen_head"
                if pipeline == PIPELINES[0]
                else fit["selected_recipe_id"]
            )
            parameters = fit["outer"]["parameters"]
            existing = profiles.get(recipe_id)
            if existing is not None and existing != parameters:
                raise RuntimeError("Parameter profile changed within strict recipe")
            profiles[recipe_id] = parameters
        counts = [fit["outer"]["parameters"]["trainable"] for fit in fits]
        result[pipeline] = {
            "trainable_parameter_range": [int(min(counts)), int(max(counts))],
            "profiles": profiles,
        }
    return result


def exploratory_comparison(
    strict_summary: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    exploratory = base.read_json(dependency_paths(protocol)["exploratory_results"])
    pipelines = {}
    for pipeline in PIPELINES:
        old = exploratory["aggregate"][pipeline]["macro_f1_mean"]
        new = strict_summary["aggregate"][pipeline]["macro_f1_mean"]
        pipelines[pipeline] = {
            "exploratory_macro_f1_mean": old,
            "strict_macro_f1_mean": new,
            "strict_minus_exploratory": new - old,
        }
    return {
        "exploratory_protocol_id": exploratory["protocol_id"],
        "selection_boundary_change": "global_across_12_inner_splits_to_per_repeat_outer_fold",
        "pipelines": pipelines,
    }


def run_evaluation(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    lock = verify_execution_lock(run_root, protocol)
    selection = base.read_json(selection_path(run_root))
    evaluation_root = run_root / "evaluation"
    summary_path = evaluation_root / "summary.json"
    if summary_path.is_file() and not resume:
        raise FileExistsError(summary_path)
    evaluation_root.mkdir(parents=True, exist_ok=True)
    base.write_json(
        evaluation_root / "run_manifest.json",
        {
            "status": "running",
            "stage": "idea058_strict_nested_evaluation",
            "outer_test_accessed": True,
            "code_commit": lock["code_commit"],
            "selection_boundary": lock["selection_boundary"],
            "per_fold_selection_sha256": lock["per_fold_selection_sha256"],
            "execution_lock_sha256": base.sha256(run_root / "execution_lock.json"),
            "device": str(device),
        },
    )
    completed = []
    order_audits = []
    settings = protocol["initial_evaluation"]
    for base_seed in settings["base_seeds"]:
        for repeat in settings["repeats"]:
            for outer_fold in settings["outer_folds"]:
                recipe_lock = fold_recipe(selection, int(repeat), int(outer_fold))
                protocol["_active_recipe_id"] = recipe_lock["selected_recipe_id"]
                indices = base.split_utils.fold_indices(
                    store, roles, int(repeat), int(outer_fold), include_test=True
                )
                outer_train = np.concatenate(
                    (indices["train"], indices["validation"])
                )
                seed = base.split_utils.full_seed(
                    int(base_seed), int(repeat), int(outer_fold)
                )
                fit_group = []
                for pipeline in PIPELINES:
                    output_dir = (
                        evaluation_root
                        / "fits"
                        / pipeline
                        / f"base_seed_{base_seed}"
                        / f"repeat_{repeat}"
                        / f"fold_{outer_fold}"
                    )
                    fit_path = output_dir / "fit_summary.json"
                    if fit_path.is_file():
                        if not resume:
                            raise FileExistsError(fit_path)
                        fit = base.read_json(fit_path)
                        if fit["selected_recipe_id"] != recipe_lock["selected_recipe_id"]:
                            raise RuntimeError("Resumed fit uses a different fold recipe")
                        completed.append(fit)
                        fit_group.append(fit)
                        continue
                    print(
                        f"IDEA058 STRICT EVAL {pipeline} recipe="
                        f"{recipe_lock['selected_recipe_id']} base_seed={base_seed} "
                        f"repeat={repeat} fold={outer_fold} seed={seed}",
                        flush=True,
                    )
                    best_epoch, inner, state, _ = base.fit_inner(
                        pipeline,
                        protocol,
                        store,
                        indices["train"],
                        indices["validation"],
                        device,
                        seed,
                    )
                    del state
                    animals, calls, outer = base.fit_outer_and_predict(
                        pipeline,
                        protocol,
                        store,
                        outer_train,
                        indices["test"],
                        best_epoch,
                        device,
                        seed,
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    animal_path = output_dir / "outer_test_animal_predictions.csv"
                    call_path = output_dir / "outer_test_call_predictions.csv"
                    animals.to_csv(animal_path, index=False)
                    calls.to_csv(call_path, index=False)
                    fit = {
                        "status": "complete",
                        "stage": "idea058_strict_nested_evaluation",
                        "outer_test_accessed": True,
                        "selection_boundary": "current_repeat_outer_fold",
                        "pipeline": pipeline,
                        "selected_recipe_id": recipe_lock["selected_recipe_id"],
                        "selected_recipe": recipe_lock["selected_recipe"],
                        "base_seed": int(base_seed),
                        "repeat": int(repeat),
                        "outer_fold": int(outer_fold),
                        "full_seed": int(seed),
                        "selected_epoch": int(best_epoch),
                        "inner": inner,
                        "outer": outer,
                        "animal_prediction_path": base.repo_relative(animal_path),
                        "call_prediction_path": base.repo_relative(call_path),
                    }
                    base.write_json(fit_path, fit)
                    completed.append(fit)
                    fit_group.append(fit)
                order_audits.append(
                    {
                        "base_seed": int(base_seed),
                        "repeat": int(repeat),
                        "outer_fold": int(outer_fold),
                        "selected_recipe_id": recipe_lock["selected_recipe_id"],
                        **base.assert_batch_orders(fit_group),
                    }
                )
    summary = base.aggregate_evaluation(evaluation_root, protocol)
    summary["stage"] = "idea058_strict_nested_evaluation"
    summary["selection_boundary"] = "per_repeat_outer_fold"
    summary["per_fold_recipe_locks"] = selection["per_fold_recipe_locks"]
    summary["selected_recipe_counts"] = selection["selected_recipe_counts"]
    summary["parameter_profiles"] = parameter_profiles(completed)
    for pipeline in PIPELINES[1:]:
        summary["training_and_parameters"][pipeline].pop("parameters")
        summary["training_and_parameters"][pipeline]["parameters_vary_by_fold"] = True
    summary["comparison_with_exploratory_idea058"] = exploratory_comparison(
        summary, protocol
    )
    summary["idea058_strict_seed_expansion_gate"] = summary.pop(
        "idea058_seed_expansion_gate"
    )
    inventory = base.evaluation.raw_prediction_inventory(evaluation_root)
    inventory_path = evaluation_root / "raw_prediction_inventory.json"
    base.write_json(inventory_path, inventory)
    summary.update(
        {
            "per_fold_selection_sha256": lock["per_fold_selection_sha256"],
            "code_commit": lock["code_commit"],
            "execution_lock_sha256": base.sha256(run_root / "execution_lock.json"),
            "environment_lock_sha256": base.sha256(
                run_root / "environment_lock.json"
            ),
            "runner_sha256": lock["runner_sha256"],
            "protocol_sha256": lock["protocol_sha256"],
            "batch_order_audit": {
                "fold_groups": len(order_audits),
                "all_common_epoch_hashes_match": True,
                "details": order_audits,
            },
            "raw_prediction_inventory": {
                "path": base.repo_relative(inventory_path),
                "inventory_sha256": base.sha256(inventory_path),
                "files": inventory["files"],
                "bytes": inventory["bytes"],
                "aggregate_sha256": inventory["aggregate_sha256"],
            },
        }
    )
    base.write_json(summary_path, summary)
    base.write_json(
        evaluation_root / "run_summary.json",
        {
            "status": "complete",
            "completed_fits": len(completed),
            "expected_fits": int(settings["total_outer_fits"]),
            "summary_path": base.repo_relative(summary_path),
            "summary_sha256": base.sha256(summary_path),
            "raw_prediction_inventory_sha256": base.sha256(inventory_path),
            "raw_prediction_aggregate_sha256": inventory["aggregate_sha256"],
        },
    )
    base.write_json(REPO_ROOT / protocol["outputs"]["result_metadata"], summary)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    protocol = base.read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    device = base.resolve_device(args.device)
    run_root = RUNS_ROOT / args.output_subdir
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    store = base.ast_base.load_feature_store()
    if len(store.call_ids) != int(protocol["dataset"]["calls"]):
        raise RuntimeError("Strict call count differs from protocol")
    if len(np.unique(store.cat_ids)) != int(protocol["dataset"]["cats"]):
        raise RuntimeError("Strict cat count differs from protocol")
    if args.stage in ("select", "all"):
        run_selection(run_root, protocol, roles, store, device, args.resume)
    if args.stage in ("smoke", "all"):
        run_smoke(run_root, protocol, roles, store, device, args.resume)
    if args.stage in ("evaluate", "all"):
        run_evaluation(run_root, protocol, roles, store, device, args.resume)


if __name__ == "__main__":
    main()
