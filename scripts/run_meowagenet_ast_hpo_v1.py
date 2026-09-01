"""Inner-only HPO and locked exploratory evaluation for two AST pipelines.

The search stage never requests outer-test indices.  It writes a selected
configuration file before the evaluate stage can produce outer predictions.
Historical formal-v2.1 recipes, runners, and results remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for local_root in (SCRIPTS_ROOT, SRC_ROOT):
    if str(local_root) not in sys.path:
        sys.path.insert(0, str(local_root))

import run_idea019_peft_placement as idea019  # noqa: E402


PROTOCOL_PATH = REPO_ROOT / "configs" / "protocol" / "meowagenet_ast_hpo_v1.json"
LOCKED_MODEL_PROTOCOL_PATH = (
    REPO_ROOT / "configs" / "protocol" / "meowagenet_locked_v1.json"
)
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
RUNS_ROOT = REPO_ROOT / "runs"
PIPELINES = ("ast_head_only", "ast_probe_guided_adapter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("search", "evaluate"), required=True)
    parser.add_argument("--output-subdir", default="meowagenet_ast_hpo_v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol["protocol_id"] != "meowagenet-ast-hpo-v1":
        raise RuntimeError("Unexpected AST HPO protocol")
    if sha256(ROLES_PATH) != protocol["splits"]["roles_sha256"]:
        raise RuntimeError("Formal nested-role checksum mismatch")
    dependencies = protocol["dependencies"]
    expected = {
        idea019.FEATURE_PATH: dependencies["fbank_sha256"],
        idea019.FROZEN_EMBEDDING_PATH: dependencies["frozen_embedding_sha256"],
        idea019.LAYER_EMBEDDING_PATH: dependencies["layer_embedding_sha256"],
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"AST HPO dependency checksum mismatch: {path}")
    search = protocol["search"]
    for key in ("head_only_trials", "adapter_trials"):
        trials = search[key]
        if len(trials) != int(search["trials_per_pipeline"]):
            raise RuntimeError(f"Unexpected search budget for {key}")
        identifiers = [trial["trial_id"] for trial in trials]
        if len(identifiers) != len(set(identifiers)):
            raise RuntimeError(f"Duplicate trial ID in {key}")


def stage_manifest(
    stage: str,
    protocol: dict[str, Any],
    device: torch.device,
    selection_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "stage": stage,
        "outer_test_accessed": stage == "evaluate",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "roles_sha256": sha256(ROLES_PATH),
        "selection_path": repo_relative(selection_path) if selection_path else None,
        "selection_sha256": sha256(selection_path) if selection_path else None,
        "git_revision_at_start": git_revision(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
        },
    }


def full_seed(base_seed: int, repeat: int, outer_fold: int) -> int:
    return int(base_seed + 10_000 * repeat + 100 * outer_fold)


def fold_indices(
    store: Any,
    roles: pd.DataFrame,
    repeat: int,
    outer_fold: int,
    include_test: bool,
) -> dict[str, np.ndarray]:
    fold_roles = roles[
        (roles["repeat"] == repeat) & (roles["outer_fold"] == outer_fold)
    ]
    mapping = dict(zip(fold_roles["cat_id"], fold_roles["role"], strict=True))
    if len(mapping) != 111:
        raise RuntimeError("Each AST HPO fold must assign all 111 cats")
    call_roles = np.asarray([mapping[cat_id] for cat_id in store.cat_ids])
    selected = {
        "train": np.flatnonzero(call_roles == "train"),
        "validation": np.flatnonzero(call_roles == "validation"),
    }
    if include_test:
        selected["test"] = np.flatnonzero(call_roles == "test")
    return selected


def probe_seed_for_fold(roles: pd.DataFrame, repeat: int, outer_fold: int) -> int:
    rows = roles[(roles["repeat"] == repeat) & (roles["outer_fold"] == outer_fold)]
    values = rows["inner_seed"].unique()
    if len(values) != 1:
        raise RuntimeError("Each AST HPO fold must have one frozen inner seed")
    return int(values[0])


def load_or_compute_probe(
    run_root: Path,
    layer_store: Any,
    train_indices: np.ndarray,
    repeat: int,
    outer_fold: int,
    seed: int,
) -> dict[str, Any]:
    path = run_root / "probes" / f"repeat_{repeat}_fold_{outer_fold}.json"
    if path.is_file():
        probe = read_json(path)
        if int(probe["seed"]) != seed:
            raise RuntimeError("Cached layer probe seed differs")
        return probe
    probe = idea019.probe_layer_utilities(
        layer_store, train_indices, cv_folds=3, seed=seed
    )
    probe["outer_test_accessed"] = False
    write_json(path, probe)
    return probe


def mode_and_placement(
    pipeline: str, probe: dict[str, Any]
) -> tuple[str, tuple[int, int] | None]:
    if pipeline == "ast_head_only":
        return "head_only", None
    if pipeline == "ast_probe_guided_adapter":
        return "adapter_probe_guided", idea019.placement_for_mode(
            "adapter_probe_guided", probe
        )
    raise RuntimeError(f"Unsupported AST HPO pipeline: {pipeline}")


def protocol_for_trial(
    locked_protocol: dict[str, Any], trial: dict[str, Any]
) -> dict[str, Any]:
    current = copy.deepcopy(locked_protocol)
    current["classifier_head"]["dropout"] = float(trial["dropout"])
    return current


def training_args(protocol: dict[str, Any], trial: dict[str, Any]) -> SimpleNamespace:
    fixed = protocol["fixed_training"]
    return SimpleNamespace(
        adapter_width=int(fixed["adapter_width"]),
        adapter_learning_rate=float(trial.get("adapter_learning_rate", 0.001)),
        head_learning_rate=float(trial["head_learning_rate"]),
        batch_size=int(fixed["micro_batch_size"]),
        accumulation_steps=int(fixed["gradient_accumulation_steps"]),
        max_epochs=int(fixed["maximum_epochs"]),
        patience=int(fixed["early_stopping_patience"]),
        gradient_clip=float(fixed["gradient_clip"]),
        probe_cv_folds=3,
    )


def trials_for_pipeline(protocol: dict[str, Any], pipeline: str) -> list[dict[str, Any]]:
    key = "head_only_trials" if pipeline == "ast_head_only" else "adapter_trials"
    return list(protocol["search"][key])


def rank_trials(
    search_root: Path, protocol: dict[str, Any], pipeline: str
) -> list[dict[str, Any]]:
    folds = list(protocol["search"]["outer_folds_used_as_inner_development_splits"])
    ranking = []
    for trial in trials_for_pipeline(protocol, pipeline):
        summaries = [
            read_json(
                search_root
                / pipeline
                / trial["trial_id"]
                / f"fold_{fold}"
                / "fit_summary.json"
            )
            for fold in folds
        ]
        metrics = [summary["audit"]["best_validation_animal_metrics"] for summary in summaries]
        row = {
            "trial_id": trial["trial_id"],
            "parameters": trial,
            "fold_macro_f1": [float(metric["macro_f1"]) for metric in metrics],
            "fold_balanced_accuracy": [
                float(metric["balanced_accuracy"]) for metric in metrics
            ],
            "fold_qwk": [
                float(metric["quadratic_weighted_kappa"]) for metric in metrics
            ],
            "fold_validation_loss": [
                float(summary["audit"]["best_validation_loss"])
                for summary in summaries
            ],
        }
        row["mean_macro_f1"] = float(np.mean(row["fold_macro_f1"]))
        row["mean_balanced_accuracy"] = float(
            np.mean(row["fold_balanced_accuracy"])
        )
        row["mean_qwk"] = float(np.mean(row["fold_qwk"]))
        row["mean_validation_loss"] = float(np.mean(row["fold_validation_loss"]))
        ranking.append(row)
    ranking.sort(
        key=lambda row: (
            -row["mean_macro_f1"],
            -row["mean_balanced_accuracy"],
            -row["mean_qwk"],
            row["mean_validation_loss"],
            row["trial_id"],
        )
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    return ranking


def run_search(
    run_root: Path,
    protocol: dict[str, Any],
    locked_protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    layer_store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    search_root = run_root / "search"
    selection_path = search_root / "selection.json"
    if selection_path.is_file():
        if not resume:
            raise FileExistsError("AST HPO selection already exists; pass --resume to verify")
        print(selection_path.read_text(encoding="utf-8"), flush=True)
        return
    write_json(search_root / "run_manifest.json", stage_manifest("search", protocol, device))
    repeat = int(protocol["search"]["development_repeat"])
    base_seed = int(protocol["search"]["base_seed"])
    folds = list(protocol["search"]["outer_folds_used_as_inner_development_splits"])
    completed = []
    for outer_fold in folds:
        indices = fold_indices(store, roles, repeat, outer_fold, include_test=False)
        probe_seed = probe_seed_for_fold(roles, repeat, outer_fold)
        probe = load_or_compute_probe(
            run_root,
            layer_store,
            indices["train"],
            repeat,
            outer_fold,
            probe_seed,
        )
        seed = full_seed(base_seed, repeat, outer_fold)
        for pipeline in PIPELINES:
            mode, placement = mode_and_placement(pipeline, probe)
            for trial in trials_for_pipeline(protocol, pipeline):
                output_dir = search_root / pipeline / trial["trial_id"] / f"fold_{outer_fold}"
                summary_path = output_dir / "fit_summary.json"
                if summary_path.is_file():
                    if not resume:
                        raise FileExistsError(summary_path)
                    completed.append(read_json(summary_path))
                    continue
                print(
                    f"SEARCH {pipeline} {trial['trial_id']} fold={outer_fold} seed={seed}",
                    flush=True,
                )
                best_epoch, audit = idea019.fit_inner(
                    mode,
                    placement,
                    protocol_for_trial(locked_protocol, trial),
                    store,
                    indices["train"],
                    indices["validation"],
                    training_args(protocol, trial),
                    device,
                    seed,
                )
                summary = {
                    "status": "complete",
                    "stage": "inner_only_search",
                    "outer_test_accessed": False,
                    "pipeline": pipeline,
                    "trial_id": trial["trial_id"],
                    "parameters": trial,
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "base_seed": base_seed,
                    "full_seed": seed,
                    "placement_layers_one_based": (
                        [layer + 1 for layer in placement] if placement else []
                    ),
                    "best_epoch": best_epoch,
                    "audit": audit,
                }
                write_json(summary_path, summary)
                completed.append(summary)
    rankings = {
        pipeline: rank_trials(search_root, protocol, pipeline) for pipeline in PIPELINES
    }
    selected = {pipeline: rankings[pipeline][0] for pipeline in PIPELINES}
    baseline_ids = {
        "ast_head_only": "h_baseline",
        "ast_probe_guided_adapter": "a_baseline",
    }
    selection = {
        "schema_version": "1.0",
        "status": "locked_before_exploratory_outer_evaluation",
        "outer_test_accessed": False,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "development_repeat": repeat,
        "folds": folds,
        "base_seed": base_seed,
        "completed_inner_fits": len(completed),
        "rankings": rankings,
        "selected": selected,
        "selected_minus_baseline_inner_macro_f1": {},
    }
    for pipeline in PIPELINES:
        baseline = next(
            row for row in rankings[pipeline] if row["trial_id"] == baseline_ids[pipeline]
        )
        selection["selected_minus_baseline_inner_macro_f1"][pipeline] = float(
            selected[pipeline]["mean_macro_f1"] - baseline["mean_macro_f1"]
        )
    write_json(selection_path, selection)
    write_json(
        search_root / "run_summary.json",
        {
            "status": "complete",
            "outer_test_accessed": False,
            "completed_inner_fits": len(completed),
            "selection_path": repo_relative(selection_path),
            "selection_sha256": sha256(selection_path),
        },
    )
    print(json.dumps(selection, indent=2), flush=True)


def aggregate_evaluation(
    evaluation_root: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    evaluation = protocol["exploratory_evaluation"]
    base_seed = int(evaluation["base_seed"])
    metrics_by_pipeline: dict[str, list[dict[str, Any]]] = {}
    for pipeline in PIPELINES:
        rows = []
        for repeat in evaluation["repeats"]:
            parts = []
            for outer_fold in evaluation["outer_folds"]:
                path = (
                    evaluation_root
                    / "fits"
                    / pipeline
                    / f"repeat_{repeat}"
                    / f"fold_{outer_fold}"
                    / f"base_seed_{base_seed}"
                    / "outer_test_call_predictions.csv"
                )
                parts.append(pd.read_csv(path, dtype={"cat_id": str}))
            calls = pd.concat(parts, ignore_index=True)
            if len(calls) != 792 or calls["call_index"].nunique() != 792:
                raise RuntimeError("AST HPO complete OOF must cover 792 calls")
            metrics, animals = idea019.evaluate_frame(calls)
            if len(animals) != 111 or animals["cat_id"].nunique() != 111:
                raise RuntimeError("AST HPO complete OOF must cover 111 cats")
            animals.insert(0, "base_seed", base_seed)
            animals.insert(0, "repeat", repeat)
            animals.insert(0, "pipeline", pipeline)
            output = evaluation_root / "oof" / pipeline / f"repeat_{repeat}_animals.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            animals.to_csv(output, index=False)
            row = {"repeat": int(repeat), **metrics}
            rows.append(row)
        metrics_by_pipeline[pipeline] = rows
    aggregate = {}
    for pipeline, rows in metrics_by_pipeline.items():
        f1 = [float(row["macro_f1"]) for row in rows]
        aggregate[pipeline] = {
            "macro_f1_mean": float(np.mean(f1)),
            "macro_f1_sample_sd": float(np.std(f1, ddof=1)),
            "balanced_accuracy_mean": float(
                np.mean([row["balanced_accuracy"] for row in rows])
            ),
            "qwk_mean": float(
                np.mean([row["quadratic_weighted_kappa"] for row in rows])
            ),
        }
    head = [row["macro_f1"] for row in metrics_by_pipeline["ast_head_only"]]
    adapter = [
        row["macro_f1"] for row in metrics_by_pipeline["ast_probe_guided_adapter"]
    ]
    differences = [float(candidate - reference) for reference, candidate in zip(head, adapter)]
    return {
        "status": "complete",
        "pipelines": metrics_by_pipeline,
        "aggregate": aggregate,
        "adapter_minus_head_only_macro_f1": {
            "paired_differences": differences,
            "mean_difference": float(np.mean(differences)),
            "positive_repeats": int(sum(value > 0 for value in differences)),
        },
    }


def run_evaluation(
    run_root: Path,
    protocol: dict[str, Any],
    locked_protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    layer_store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    selection_path = run_root / "search" / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("Run --stage search before exploratory evaluation")
    selection = read_json(selection_path)
    if selection["status"] != "locked_before_exploratory_outer_evaluation":
        raise RuntimeError("AST HPO selection is not locked")
    if selection["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise RuntimeError("AST HPO protocol changed after selection")
    if selection["runner_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("AST HPO runner changed after selection")
    evaluation_root = run_root / "evaluation"
    summary_path = evaluation_root / "summary.json"
    if summary_path.is_file():
        if not resume:
            raise FileExistsError("AST HPO evaluation exists; pass --resume to verify")
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return
    write_json(
        evaluation_root / "run_manifest.json",
        stage_manifest("evaluate", protocol, device, selection_path),
    )
    evaluation = protocol["exploratory_evaluation"]
    base_seed = int(evaluation["base_seed"])
    completed = []
    for repeat in evaluation["repeats"]:
        for outer_fold in evaluation["outer_folds"]:
            indices = fold_indices(store, roles, int(repeat), int(outer_fold), include_test=True)
            probe_seed = probe_seed_for_fold(roles, int(repeat), int(outer_fold))
            probe = load_or_compute_probe(
                run_root,
                layer_store,
                indices["train"],
                int(repeat),
                int(outer_fold),
                probe_seed,
            )
            seed = full_seed(base_seed, int(repeat), int(outer_fold))
            for pipeline in PIPELINES:
                selected = selection["selected"][pipeline]
                trial = selected["parameters"]
                mode, placement = mode_and_placement(pipeline, probe)
                output_dir = (
                    evaluation_root
                    / "fits"
                    / pipeline
                    / f"repeat_{repeat}"
                    / f"fold_{outer_fold}"
                    / f"base_seed_{base_seed}"
                )
                fit_path = output_dir / "fit_summary.json"
                if fit_path.is_file():
                    if not resume:
                        raise FileExistsError(fit_path)
                    completed.append(read_json(fit_path))
                    continue
                print(
                    f"EVAL {pipeline} {trial['trial_id']} repeat={repeat} "
                    f"fold={outer_fold} seed={seed}",
                    flush=True,
                )
                current_protocol = protocol_for_trial(locked_protocol, trial)
                args = training_args(protocol, trial)
                best_epoch, inner_audit = idea019.fit_inner(
                    mode,
                    placement,
                    current_protocol,
                    store,
                    indices["train"],
                    indices["validation"],
                    args,
                    device,
                    seed,
                )
                outer_train = np.concatenate((indices["train"], indices["validation"]))
                test_frame, outer_audit = idea019.fit_outer_and_predict(
                    mode,
                    placement,
                    current_protocol,
                    store,
                    outer_train,
                    indices["test"],
                    best_epoch,
                    args,
                    device,
                    seed,
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                prediction_path = output_dir / "outer_test_call_predictions.csv"
                test_frame.to_csv(prediction_path, index=False)
                fit = {
                    "status": "complete",
                    "stage": "locked_exploratory_evaluation",
                    "outer_test_accessed": True,
                    "pipeline": pipeline,
                    "trial_id": trial["trial_id"],
                    "parameters": trial,
                    "repeat": int(repeat),
                    "outer_fold": int(outer_fold),
                    "base_seed": base_seed,
                    "full_seed": seed,
                    "placement_layers_one_based": (
                        [layer + 1 for layer in placement] if placement else []
                    ),
                    "selected_epoch": best_epoch,
                    "inner": inner_audit,
                    "outer": outer_audit,
                    "prediction_path": repo_relative(prediction_path),
                }
                write_json(fit_path, fit)
                completed.append(fit)
    summary = aggregate_evaluation(evaluation_root, protocol)
    summary["selection_path"] = repo_relative(selection_path)
    summary["selection_sha256"] = sha256(selection_path)
    summary["completed_fits"] = len(completed)
    write_json(summary_path, summary)
    write_json(
        evaluation_root / "run_summary.json",
        {
            "status": "complete",
            "completed_fits": len(completed),
            "expected_fits": int(evaluation["total_fits"]),
            "summary_path": repo_relative(summary_path),
            "fits": completed,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_PATH)
    locked_protocol = read_json(LOCKED_MODEL_PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = (RUNS_ROOT / args.output_subdir).resolve()
    if RUNS_ROOT.resolve() not in run_root.parents:
        raise ValueError("--output-subdir must stay below runs")
    run_root.mkdir(parents=True, exist_ok=True)
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    store = idea019.load_feature_store()
    layer_store = idea019.load_layer_store(store)
    device = idea019.resolve_device(args.device)
    print(
        f"AST HPO stage={args.stage}; device={device}; "
        f"device_name={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}",
        flush=True,
    )
    if args.stage == "search":
        run_search(
            run_root,
            protocol,
            locked_protocol,
            roles,
            store,
            layer_store,
            device,
            args.resume,
        )
    else:
        run_evaluation(
            run_root,
            protocol,
            locked_protocol,
            roles,
            store,
            layer_store,
            device,
            args.resume,
        )


if __name__ == "__main__":
    main()
