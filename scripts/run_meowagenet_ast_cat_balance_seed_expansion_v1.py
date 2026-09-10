"""Expand the matched A0/A1 frozen-AST comparison to base seeds 43 and 101."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for local_root in (SCRIPTS_ROOT, SRC_ROOT):
    if str(local_root) not in sys.path:
        sys.path.insert(0, str(local_root))

import run_meowagenet_ast_accuracy_enhancement_v1 as core  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_ast_cat_balance_seed_expansion_v1.json"
)
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
RUNS_ROOT = REPO_ROOT / "runs"
SEED17_RUN_ROOT = RUNS_ROOT / "meowagenet_ast_accuracy_enhancement_v1"
SEED17_RESULT_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_ast_accuracy_enhancement_v1_results.json"
)
PIPELINES = ("A0_final_class_balanced", "A1_final_cat_balanced")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "evaluate"), required=True)
    parser.add_argument(
        "--output-subdir", default="meowagenet_ast_cat_balance_seed_expansion_v1"
    )
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
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


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
    if protocol["protocol_id"] != "meowagenet-ast-cat-balance-seed-expansion-v1":
        raise RuntimeError("Unexpected AST cat-balance seed-expansion protocol")
    if tuple(protocol["pipelines"]) != PIPELINES:
        raise RuntimeError("Expansion pipeline pair changed")
    if tuple(protocol["evaluation"]["base_seeds"]) != (43, 101):
        raise RuntimeError("Expansion seeds changed")
    if sha256(ROLES_PATH) != protocol["splits"]["roles_sha256"]:
        raise RuntimeError("Formal nested-role checksum mismatch")
    dependencies = protocol["dependencies"]
    dependency_paths = {
        "seed17_protocol": core.PROTOCOL_PATH,
        "seed17_runner": SCRIPTS_ROOT
        / "run_meowagenet_ast_accuracy_enhancement_v1.py",
        "seed17_result": SEED17_RESULT_PATH,
        "seed17_execution_lock": SEED17_RUN_ROOT / "execution_lock.json",
        "seed17_evaluation_summary": SEED17_RUN_ROOT
        / "evaluation"
        / "summary.json",
        "fbank": core.idea019.FEATURE_PATH,
        "final_embedding": core.idea019.FROZEN_EMBEDDING_PATH,
    }
    for key, path in dependency_paths.items():
        if not path.is_file() or sha256(path) != dependencies[f"{key}_sha256"]:
            raise RuntimeError(f"Expansion dependency checksum mismatch: {path}")
    evaluation = protocol["evaluation"]
    expected_fits = (
        len(PIPELINES)
        * len(evaluation["base_seeds"])
        * len(evaluation["repeats"])
        * len(evaluation["outer_folds"])
    )
    if expected_fits != int(evaluation["total_outer_fits"]):
        raise RuntimeError("Expansion fit budget is inconsistent")


def stage_manifest(stage: str, protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "stage": stage,
        "outer_test_accessed": stage == "evaluate",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "roles_sha256": sha256(ROLES_PATH),
        "seed17_result_sha256": sha256(SEED17_RESULT_PATH),
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


def run_smoke(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    smoke_root = run_root / "smoke"
    lock_path = run_root / "execution_lock.json"
    if lock_path.is_file() and not resume:
        raise FileExistsError("Expansion execution lock exists; pass --resume to verify")
    write_json(smoke_root / "run_manifest.json", stage_manifest("smoke", protocol, device))
    smoke = protocol["smoke"]
    base_seed = int(smoke["base_seed"])
    repeat = int(smoke["repeat"])
    outer_fold = int(smoke["outer_fold"])
    seed = core.full_seed(base_seed, repeat, outer_fold)
    indices = core.fold_indices(
        store, roles, repeat, outer_fold, include_test=False
    )
    completed = []
    for pipeline in PIPELINES:
        output_dir = smoke_root / "fits" / pipeline
        fit_path = output_dir / "fit_summary.json"
        if fit_path.is_file():
            if not resume:
                raise FileExistsError(fit_path)
            fit = read_json(fit_path)
        else:
            print(
                f"EXPANSION SMOKE {pipeline} base_seed={base_seed} "
                f"repeat={repeat} fold={outer_fold} seed={seed}",
                flush=True,
            )
            best_epoch, inner = core.fit_inner(
                pipeline,
                protocol,
                store,
                None,
                indices["train"],
                indices["validation"],
                device,
                seed,
            )
            fit = {
                "status": "complete",
                "stage": "inner_only_smoke",
                "outer_test_accessed": False,
                "pipeline": pipeline,
                "base_seed": base_seed,
                "repeat": repeat,
                "outer_fold": outer_fold,
                "full_seed": seed,
                "best_epoch": best_epoch,
                "inner": inner,
            }
            write_json(fit_path, fit)
        completed.append((fit_path, fit))
    if lock_path.is_file():
        lock = read_json(lock_path)
        if (
            lock["protocol_sha256"] != sha256(PROTOCOL_PATH)
            or lock["runner_sha256"] != sha256(Path(__file__))
        ):
            raise RuntimeError("Expansion execution lock hash mismatch")
    else:
        lock = {
            "schema_version": "1.0",
            "status": "locked_for_seed43_101_evaluation",
            "outer_test_accessed": False,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "runner_sha256": sha256(Path(__file__)),
            "roles_sha256": sha256(ROLES_PATH),
            "seed17_result_sha256": sha256(SEED17_RESULT_PATH),
            "pipelines": list(PIPELINES),
            "evaluation": protocol["evaluation"],
            "fixed_training": protocol["fixed_training"],
            "smoke_fit_sha256": {
                fit["pipeline"]: sha256(path) for path, fit in completed
            },
        }
        write_json(lock_path, lock)
    summary = {
        "status": "complete",
        "outer_test_accessed": False,
        "completed_inner_fits": len(completed),
        "execution_lock": repo_relative(lock_path),
        "execution_lock_sha256": sha256(lock_path),
        "pipelines": {
            fit["pipeline"]: {
                "best_epoch": fit["best_epoch"],
                "best_validation_loss": fit["inner"]["best_validation_loss"],
                "best_validation_animal_metrics": fit["inner"][
                    "best_validation_animal_metrics"
                ],
            }
            for _, fit in completed
        },
    }
    write_json(smoke_root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def verify_execution_lock(run_root: Path) -> dict[str, Any]:
    lock_path = run_root / "execution_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("Run expansion smoke before evaluation")
    lock = read_json(lock_path)
    if lock["status"] != "locked_for_seed43_101_evaluation":
        raise RuntimeError("Unexpected expansion execution-lock status")
    if lock["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise RuntimeError("Expansion protocol changed after lock")
    if lock["runner_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("Expansion runner changed after lock")
    if lock["seed17_result_sha256"] != sha256(SEED17_RESULT_PATH):
        raise RuntimeError("Seed-17 result changed after expansion lock")
    return lock


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    macro_f1 = [float(row["macro_f1"]) for row in rows]
    return {
        "macro_f1_mean": float(np.mean(macro_f1)),
        "macro_f1_sample_sd": float(np.std(macro_f1, ddof=1)),
        "balanced_accuracy_mean": float(
            np.mean([row["balanced_accuracy"] for row in rows])
        ),
        "qwk_mean": float(
            np.mean([row["quadratic_weighted_kappa"] for row in rows])
        ),
        "plain_accuracy_mean": float(
            np.mean(
                [
                    np.trace(np.asarray(row["confusion_matrix"])) / int(row["n"])
                    for row in rows
                ]
            )
        ),
        "mean_per_class_recall": {
            label: float(np.mean([row["per_class_recall"][label] for row in rows]))
            for label in core.LABEL_NAMES
        },
    }


def aggregate_evaluation(
    evaluation_root: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    evaluation = protocol["evaluation"]
    metrics_by_pipeline: dict[str, list[dict[str, Any]]] = {
        pipeline: [] for pipeline in PIPELINES
    }
    animals_by_key: dict[tuple[str, int, int], pd.DataFrame] = {}
    for base_seed in evaluation["base_seeds"]:
        for pipeline in PIPELINES:
            for repeat in evaluation["repeats"]:
                parts = []
                for outer_fold in evaluation["outer_folds"]:
                    path = (
                        evaluation_root
                        / "fits"
                        / pipeline
                        / f"base_seed_{base_seed}"
                        / f"repeat_{repeat}"
                        / f"fold_{outer_fold}"
                        / "outer_test_call_predictions.csv"
                    )
                    parts.append(pd.read_csv(path, dtype={"cat_id": str}))
                calls = pd.concat(parts, ignore_index=True).sort_values("call_index")
                if len(calls) != 792 or calls["call_index"].nunique() != 792:
                    raise RuntimeError("Expanded complete OOF must cover 792 calls")
                metrics, animals = core.idea019.evaluate_frame(calls)
                if len(animals) != 111 or animals["cat_id"].nunique() != 111:
                    raise RuntimeError("Expanded complete OOF must cover 111 cats")
                row = {"base_seed": int(base_seed), "repeat": int(repeat), **metrics}
                metrics_by_pipeline[pipeline].append(row)
                animals.insert(0, "base_seed", base_seed)
                animals.insert(0, "repeat", repeat)
                animals.insert(0, "pipeline", pipeline)
                output = (
                    evaluation_root
                    / "oof"
                    / pipeline
                    / f"base_seed_{base_seed}_repeat_{repeat}_animals.csv"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                animals.to_csv(output, index=False)
                animals_by_key[(pipeline, int(base_seed), int(repeat))] = animals
    new_aggregate = {
        pipeline: mean_metrics(rows) for pipeline, rows in metrics_by_pipeline.items()
    }
    paired_new = []
    change_frames = []
    for base_seed in evaluation["base_seeds"]:
        for repeat in evaluation["repeats"]:
            a0_metric = next(
                row
                for row in metrics_by_pipeline[PIPELINES[0]]
                if row["base_seed"] == base_seed and row["repeat"] == repeat
            )
            a1_metric = next(
                row
                for row in metrics_by_pipeline[PIPELINES[1]]
                if row["base_seed"] == base_seed and row["repeat"] == repeat
            )
            left = animals_by_key[(PIPELINES[0], int(base_seed), int(repeat))][
                ["cat_id", "true_label", "predicted_label"]
            ].rename(columns={"predicted_label": "a0_prediction"})
            right = animals_by_key[(PIPELINES[1], int(base_seed), int(repeat))][
                ["cat_id", "true_label", "predicted_label"]
            ].rename(columns={"predicted_label": "a1_prediction"})
            merged = left.merge(right, on=["cat_id", "true_label"], validate="one_to_one")
            merged["a0_correct"] = merged["a0_prediction"] == merged["true_label"]
            merged["a1_correct"] = merged["a1_prediction"] == merged["true_label"]
            changed = merged[merged["a0_prediction"] != merged["a1_prediction"]].copy()
            changed.insert(0, "repeat", repeat)
            changed.insert(0, "base_seed", base_seed)
            change_frames.append(changed)
            paired_new.append(
                {
                    "base_seed": int(base_seed),
                    "repeat": int(repeat),
                    "A0_macro_f1": float(a0_metric["macro_f1"]),
                    "A1_macro_f1": float(a1_metric["macro_f1"]),
                    "A1_minus_A0_macro_f1": float(
                        a1_metric["macro_f1"] - a0_metric["macro_f1"]
                    ),
                    "changed_animals": int(len(changed)),
                    "gained_correct_animals": int(
                        ((~changed["a0_correct"]) & changed["a1_correct"]).sum()
                    ),
                    "lost_correct_animals": int(
                        (changed["a0_correct"] & (~changed["a1_correct"])).sum()
                    ),
                }
            )
    change_path = evaluation_root / "paired_prediction_changes_A1_vs_A0.csv"
    pd.concat(change_frames, ignore_index=True).to_csv(change_path, index=False)
    seed17 = read_json(SEED17_RUN_ROOT / "evaluation" / "summary.json")
    seed17_rows = {
        pipeline: [
            {"base_seed": 17, **row} for row in seed17["complete_oof"][pipeline]
        ]
        for pipeline in PIPELINES
    }
    combined_rows = {
        pipeline: seed17_rows[pipeline] + metrics_by_pipeline[pipeline]
        for pipeline in PIPELINES
    }
    combined_aggregate = {
        pipeline: mean_metrics(rows) for pipeline, rows in combined_rows.items()
    }
    seed17_differences = seed17["paired_vs_A0"][
        "A1_final_cat_balanced_minus_A0_final_class_balanced"
    ]["macro_f1_differences"]
    combined_paired = [
        {"base_seed": 17, "repeat": repeat, "A1_minus_A0_macro_f1": float(value)}
        for repeat, value in enumerate(seed17_differences)
    ] + paired_new
    combined_differences = [
        float(row["A1_minus_A0_macro_f1"]) for row in combined_paired
    ]
    per_seed = {}
    for base_seed in (17, 43, 101):
        values = [
            row["A1_minus_A0_macro_f1"]
            for row in combined_paired
            if row["base_seed"] == base_seed
        ]
        per_seed[str(base_seed)] = {
            "paired_macro_f1_differences": values,
            "mean_difference": float(np.mean(values)),
            "positive_repeats": int(sum(value > 0 for value in values)),
        }
    positive_comparisons = int(sum(value > 0 for value in combined_differences))
    combined_mean_difference = float(np.mean(combined_differences))
    confirmation_met = combined_mean_difference > 0 and positive_comparisons >= 6
    return {
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "new_completed_outer_fits": int(evaluation["total_outer_fits"]),
        "new_complete_oof": metrics_by_pipeline,
        "new_seed_aggregate": new_aggregate,
        "new_paired_A1_vs_A0": paired_new,
        "combined_seed17_43_101": {
            "complete_oof_count_per_pipeline": 9,
            "aggregate": combined_aggregate,
            "paired_macro_f1_differences": combined_differences,
            "mean_A1_minus_A0_macro_f1": combined_mean_difference,
            "positive_comparisons": positive_comparisons,
            "per_seed": per_seed,
        },
        "stage_decision": {
            "rule": protocol["combined_decision"]["confirmation_rule"],
            "confirmation_met": confirmation_met,
            "leading_pipeline": PIPELINES[1] if combined_mean_difference > 0 else PIPELINES[0],
            "interpretation": (
                protocol["combined_decision"]["interpretation_if_met"]
                if confirmation_met
                else protocol["combined_decision"]["interpretation_if_unmet"]
            ),
        },
        "artifacts": {
            "paired_prediction_changes": repo_relative(change_path),
        },
    }


def run_evaluation(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    lock = verify_execution_lock(run_root)
    evaluation_root = run_root / "evaluation"
    summary_path = evaluation_root / "summary.json"
    if summary_path.is_file():
        if not resume:
            raise FileExistsError("Expansion evaluation exists; pass --resume to verify")
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return
    write_json(
        evaluation_root / "run_manifest.json",
        {
            **stage_manifest("evaluate", protocol, device),
            "execution_lock_path": repo_relative(run_root / "execution_lock.json"),
            "execution_lock_sha256": sha256(run_root / "execution_lock.json"),
        },
    )
    evaluation = protocol["evaluation"]
    completed = []
    for base_seed in evaluation["base_seeds"]:
        for repeat in evaluation["repeats"]:
            for outer_fold in evaluation["outer_folds"]:
                indices = core.fold_indices(
                    store, roles, int(repeat), int(outer_fold), include_test=True
                )
                outer_train = np.concatenate(
                    (indices["train"], indices["validation"])
                )
                seed = core.full_seed(int(base_seed), int(repeat), int(outer_fold))
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
                        completed.append(read_json(fit_path))
                        continue
                    print(
                        f"EXPANSION EVAL {pipeline} base_seed={base_seed} "
                        f"repeat={repeat} fold={outer_fold} seed={seed}",
                        flush=True,
                    )
                    best_epoch, inner = core.fit_inner(
                        pipeline,
                        protocol,
                        store,
                        None,
                        indices["train"],
                        indices["validation"],
                        device,
                        seed,
                    )
                    test_frame, outer = core.fit_outer_and_predict(
                        pipeline,
                        protocol,
                        store,
                        None,
                        outer_train,
                        indices["test"],
                        best_epoch,
                        device,
                        seed,
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    prediction_path = output_dir / "outer_test_call_predictions.csv"
                    test_frame.to_csv(prediction_path, index=False)
                    fit = {
                        "status": "complete",
                        "stage": "locked_seed43_101_evaluation",
                        "outer_test_accessed": True,
                        "pipeline": pipeline,
                        "base_seed": int(base_seed),
                        "repeat": int(repeat),
                        "outer_fold": int(outer_fold),
                        "full_seed": seed,
                        "selected_epoch": best_epoch,
                        "inner": inner,
                        "outer": outer,
                        "prediction_path": repo_relative(prediction_path),
                    }
                    write_json(fit_path, fit)
                    completed.append(fit)
    summary = aggregate_evaluation(evaluation_root, protocol)
    summary["execution_lock_sha256"] = sha256(run_root / "execution_lock.json")
    summary["runner_sha256"] = lock["runner_sha256"]
    summary["protocol_sha256"] = lock["protocol_sha256"]
    write_json(summary_path, summary)
    write_json(
        evaluation_root / "run_summary.json",
        {
            "status": "complete",
            "completed_fits": len(completed),
            "expected_fits": int(evaluation["total_outer_fits"]),
            "summary_path": repo_relative(summary_path),
            "summary_sha256": sha256(summary_path),
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = (RUNS_ROOT / args.output_subdir).resolve()
    if RUNS_ROOT.resolve() not in run_root.parents:
        raise ValueError("--output-subdir must stay below runs")
    run_root.mkdir(parents=True, exist_ok=True)
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    store = core.idea019.load_feature_store()
    device = core.idea019.resolve_device(args.device)
    print(
        f"AST cat-balance expansion stage={args.stage}; device={device}; "
        f"device_name={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}",
        flush=True,
    )
    if args.stage == "smoke":
        run_smoke(run_root, protocol, roles, store, device, args.resume)
    else:
        run_evaluation(run_root, protocol, roles, store, device, args.resume)


if __name__ == "__main__":
    main()
