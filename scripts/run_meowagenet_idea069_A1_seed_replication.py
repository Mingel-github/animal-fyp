"""Run the IDEA-069 independent-seed replication of the IDEA-068 A1 residual."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_idea068_age_sensitive_ast as idea068  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea069_A1_seed_replication_v1.json"
)
PIPELINES = ("A0_ast_only", "A1_age_residual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output-subdir", default="meowagenet_idea069_A1_seed_replication_v1"
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea069-A1-seed-replication-v1":
        raise RuntimeError("Unexpected IDEA-069 protocol")
    if protocol.get("status") != "locked_before_independent_seed_replication":
        raise RuntimeError("IDEA-069 protocol is not locked")
    if tuple(protocol["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-069 pipeline matrix changed")
    screen = protocol["screen"]
    if screen["base_seeds"] != [613, 887, 1201]:
        raise RuntimeError("IDEA-069 seed bank changed")
    expected = (
        len(PIPELINES)
        * len(screen["base_seeds"])
        * len(screen["repeats"])
        * len(screen["outer_folds_used_as_inner_role_definitions"])
    )
    if expected != int(screen["total_fits"]):
        raise RuntimeError("IDEA-069 fit budget is inconsistent")
    dependencies = protocol["dependencies"]
    checks = {
        REPO_ROOT / dependencies["idea_path"]: dependencies["idea_sha256"],
        REPO_ROOT / dependencies["idea068_protocol_path"]: dependencies[
            "idea068_protocol_sha256"
        ],
        REPO_ROOT / dependencies["idea068_runner_path"]: dependencies[
            "idea068_runner_sha256"
        ],
        REPO_ROOT / protocol["data"]["roles_path"]: protocol["data"][
            "roles_sha256"
        ],
        REPO_ROOT / protocol["data"]["frozen_embedding_path"]: protocol["data"][
            "frozen_embedding_sha256"
        ],
        REPO_ROOT / protocol["data"]["feature_path"]: protocol["data"][
            "feature_sha256"
        ],
        REPO_ROOT / protocol["data"]["feature_extraction_summary_path"]: protocol[
            "data"
        ]["feature_extraction_summary_sha256"],
        Path(__file__).resolve(): dependencies["runner_sha256"],
    }
    for path, expected_sha in checks.items():
        if not path.is_file() or sha256(path) != expected_sha:
            raise RuntimeError(f"IDEA-069 dependency checksum mismatch: {path}")


def resolve_run_root(output_subdir: str) -> Path:
    run_root = (idea068.idea051.RUNS_ROOT / output_subdir).resolve()
    if idea068.idea051.RUNS_ROOT.resolve() not in run_root.parents:
        raise ValueError("--output-subdir must stay below runs")
    return run_root


def load_age_features(
    protocol: dict[str, Any], call_ids: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    feature_path = REPO_ROOT / protocol["data"]["feature_path"]
    extraction_summary = read_json(
        REPO_ROOT / protocol["data"]["feature_extraction_summary_path"]
    )
    if extraction_summary.get("status") != "complete":
        raise RuntimeError("IDEA-069 source feature extraction is incomplete")
    if extraction_summary.get("feature_sha256") != protocol["data"]["feature_sha256"]:
        raise RuntimeError("IDEA-069 feature summary and protocol differ")
    loaded = np.load(feature_path)
    if tuple(loaded["feature_names"].astype(str)) != idea068.FEATURE_NAMES:
        raise RuntimeError("IDEA-069 stored acoustic feature set changed")
    if not np.array_equal(loaded["call_ids"].astype(str), call_ids.astype(str)):
        raise RuntimeError("IDEA-069 feature call order differs from AST")
    features = loaded["features"].astype(np.float32)
    if features.shape != (len(call_ids), len(idea068.FEATURE_NAMES)):
        raise RuntimeError("IDEA-069 feature matrix has the wrong shape")
    return features, extraction_summary


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "metrics": idea068.idea051.animal_metrics(frame),
        "cross_entropy": idea068.idea051.animal_cross_entropy(frame),
        "brier": idea068.brier(frame),
    }


def aggregate(fits: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    by_key = {
        (
            int(fit["base_seed"]),
            fit["pipeline"],
            int(fit["repeat"]),
            int(fit["outer_fold"]),
        ): fit
        for fit in fits
    }
    fold_rows = []
    pooled: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
    screen = protocol["screen"]
    for base_seed in screen["base_seeds"]:
        for repeat in screen["repeats"]:
            for outer_fold in screen["outer_folds_used_as_inner_role_definitions"]:
                frames = {}
                values = {}
                for pipeline in PIPELINES:
                    fit = by_key[(base_seed, pipeline, repeat, outer_fold)]
                    frame = pd.read_csv(
                        REPO_ROOT / fit["validation_animal_predictions"],
                        dtype={"cat_id": str},
                    )
                    tagged = frame.copy()
                    tagged["base_seed"] = base_seed
                    tagged["repeat"] = repeat
                    tagged["outer_fold"] = outer_fold
                    pooled[pipeline].append(tagged)
                    frames[pipeline] = frame
                    values[pipeline] = metrics(frame)
                control_f1 = values[PIPELINES[0]]["metrics"]["macro_f1"]
                candidate_f1 = values[PIPELINES[1]]["metrics"]["macro_f1"]
                fold_rows.append(
                    {
                        "base_seed": base_seed,
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "control_macro_f1": control_f1,
                        "candidate_macro_f1": candidate_f1,
                        "delta_macro_f1": candidate_f1 - control_f1,
                        "control_cross_entropy": values[PIPELINES[0]][
                            "cross_entropy"
                        ],
                        "candidate_cross_entropy": values[PIPELINES[1]][
                            "cross_entropy"
                        ],
                        "control_brier": values[PIPELINES[0]]["brier"],
                        "candidate_brier": values[PIPELINES[1]]["brier"],
                        "control_senior_recall": values[PIPELINES[0]]["metrics"][
                            "per_class"
                        ]["senior"]["recall"],
                        "candidate_senior_recall": values[PIPELINES[1]][
                            "metrics"
                        ]["per_class"]["senior"]["recall"],
                    }
                )
    folds = pd.DataFrame(fold_rows)
    pooled_frames = {
        pipeline: pd.concat(parts, ignore_index=True) for pipeline, parts in pooled.items()
    }
    pooled_metrics = {
        pipeline: {
            "animal_occurrences": int(len(frame)),
            **metrics(frame),
        }
        for pipeline, frame in pooled_frames.items()
    }
    seed_results = {}
    seed_repeat_results = []
    for base_seed in screen["base_seeds"]:
        selected = folds[folds["base_seed"] == base_seed]
        seed_frames = {
            pipeline: frame[frame["base_seed"] == base_seed]
            for pipeline, frame in pooled_frames.items()
        }
        seed_metrics = {pipeline: metrics(frame) for pipeline, frame in seed_frames.items()}
        senior_delta = (
            seed_metrics[PIPELINES[1]]["metrics"]["per_class"]["senior"]["recall"]
            - seed_metrics[PIPELINES[0]]["metrics"]["per_class"]["senior"]["recall"]
        )
        seed_results[str(base_seed)] = {
            "mean_fold_macro_f1": {
                PIPELINES[0]: float(selected["control_macro_f1"].mean()),
                PIPELINES[1]: float(selected["candidate_macro_f1"].mean()),
                "delta": float(selected["delta_macro_f1"].mean()),
            },
            "positive_folds": int((selected["delta_macro_f1"] > 0).sum()),
            "tied_folds": int((selected["delta_macro_f1"] == 0).sum()),
            "negative_folds": int((selected["delta_macro_f1"] < 0).sum()),
            "pooled_validation": seed_metrics,
            "senior_recall_delta": float(senior_delta),
        }
        for repeat in screen["repeats"]:
            cell = selected[selected["repeat"] == repeat]
            seed_repeat_results.append(
                {
                    "base_seed": base_seed,
                    "repeat": repeat,
                    "mean_fold_macro_f1_delta": float(
                        cell["delta_macro_f1"].mean()
                    ),
                    "positive_folds": int((cell["delta_macro_f1"] > 0).sum()),
                    "tied_folds": int((cell["delta_macro_f1"] == 0).sum()),
                    "negative_folds": int((cell["delta_macro_f1"] < 0).sum()),
                }
            )
    seed_repeat_frame = pd.DataFrame(seed_repeat_results)
    gate = protocol["gate"]
    ce_delta = (
        pooled_metrics[PIPELINES[1]]["cross_entropy"]
        - pooled_metrics[PIPELINES[0]]["cross_entropy"]
    )
    conditions = {
        "mean_fold_macro_f1_gain": float(folds["delta_macro_f1"].mean())
        >= float(gate["minimum_mean_fold_macro_f1_gain"]),
        "every_seed_mean_not_materially_negative": all(
            value["mean_fold_macro_f1"]["delta"]
            >= float(gate["minimum_each_seed_mean_macro_f1_delta"])
            for value in seed_results.values()
        ),
        "positive_seed_repeat_cells": int(
            (seed_repeat_frame["mean_fold_macro_f1_delta"] > 0).sum()
        )
        >= int(gate["minimum_positive_seed_repeat_cells"]),
        "positive_folds": int((folds["delta_macro_f1"] > 0).sum())
        >= int(gate["minimum_positive_folds"]),
        "nonnegative_folds": int((folds["delta_macro_f1"] >= 0).sum())
        >= int(gate["minimum_nonnegative_folds"]),
        "pooled_cross_entropy_within_tolerance": ce_delta
        <= float(gate["maximum_pooled_cross_entropy_delta"]),
        "pooled_brier_nonworse": pooled_metrics[PIPELINES[1]]["brier"]
        <= pooled_metrics[PIPELINES[0]]["brier"],
        "each_seed_senior_recall": all(
            value["senior_recall_delta"]
            >= float(gate["minimum_each_seed_senior_recall_delta"])
            for value in seed_results.values()
        ),
    }
    return {
        "status": "complete",
        "outer_test_accessed": False,
        "fits": len(fits),
        "paired_fold_comparisons": int(len(folds)),
        "fold_results": fold_rows,
        "overall": {
            "mean_fold_macro_f1": {
                PIPELINES[0]: float(folds["control_macro_f1"].mean()),
                PIPELINES[1]: float(folds["candidate_macro_f1"].mean()),
                "delta": float(folds["delta_macro_f1"].mean()),
            },
            "positive_folds": int((folds["delta_macro_f1"] > 0).sum()),
            "tied_folds": int((folds["delta_macro_f1"] == 0).sum()),
            "negative_folds": int((folds["delta_macro_f1"] < 0).sum()),
            "pooled_validation": pooled_metrics,
            "cross_entropy_delta": float(ce_delta),
            "brier_delta": float(
                pooled_metrics[PIPELINES[1]]["brier"]
                - pooled_metrics[PIPELINES[0]]["brier"]
            ),
        },
        "seed_results": seed_results,
        "seed_repeat_results": seed_repeat_results,
        "positive_seed_repeat_cells": int(
            (seed_repeat_frame["mean_fold_macro_f1_delta"] > 0).sum()
        ),
        "gate_conditions": conditions,
        "gate_passed": bool(all(conditions.values())),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    idea068.configure_determinism()
    run_root = resolve_run_root(args.output_subdir)
    run_root.mkdir(parents=True, exist_ok=True)
    store = idea068.idea051.reference.historical.idea019.load_feature_store()
    age_features, extraction_summary = load_age_features(protocol, store.call_ids)
    device = idea068.idea051.reference.historical.idea019.resolve_device(args.device)
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "feature_sha256": extraction_summary["feature_sha256"],
        "outer_test_accessed": False,
        "pipelines": list(PIPELINES),
        "screen": protocol["screen"],
        "determinism": protocol["determinism"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
    }
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.exists():
        if not args.resume or read_json(manifest_path) != manifest:
            raise RuntimeError("Existing IDEA-069 run manifest differs")
    else:
        write_json(manifest_path, manifest)
    roles = pd.read_csv(
        REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str}
    )
    completed = []
    screen_spec = protocol["screen"]
    for base_seed in screen_spec["base_seeds"]:
        for repeat in screen_spec["repeats"]:
            for outer_fold in screen_spec[
                "outer_folds_used_as_inner_role_definitions"
            ]:
                indices = idea068.idea051.reference.historical.fold_indices(
                    store, roles, repeat, outer_fold, include_test=False
                )
                full_seed = idea068.idea051.reference.historical.full_seed(
                    int(base_seed), repeat, outer_fold
                )
                pair = []
                for pipeline in PIPELINES:
                    output_dir = (
                        run_root
                        / "fits"
                        / pipeline
                        / f"base_seed_{base_seed}"
                        / f"repeat_{repeat}"
                        / f"fold_{outer_fold}"
                    )
                    summary_path = output_dir / "fit_summary.json"
                    if summary_path.exists():
                        if not args.resume:
                            raise FileExistsError(summary_path)
                        fit = read_json(summary_path)
                        completed.append(fit)
                        pair.append(fit)
                        continue
                    print(
                        f"=== {pipeline} base_seed={base_seed} repeat={repeat} "
                        f"fold={outer_fold} full_seed={full_seed} ===",
                        flush=True,
                    )
                    audit, animals, calls = idea068.fit_inner(
                        pipeline,
                        protocol,
                        store,
                        age_features,
                        indices["train"],
                        indices["validation"],
                        device,
                        full_seed,
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    animal_path = output_dir / "validation_animal_predictions.csv"
                    call_path = output_dir / "validation_call_predictions.csv"
                    animals.to_csv(animal_path, index=False)
                    calls.to_csv(call_path, index=False)
                    fit = {
                        "status": "complete",
                        "pipeline": pipeline,
                        "base_seed": int(base_seed),
                        "full_seed": int(full_seed),
                        "repeat": int(repeat),
                        "outer_fold": int(outer_fold),
                        "outer_test_accessed": False,
                        "train_calls": int(len(indices["train"])),
                        "validation_calls": int(len(indices["validation"])),
                        "validation_cats": int(animals["cat_id"].nunique()),
                        "validation_animal_predictions": animal_path.relative_to(
                            REPO_ROOT
                        ).as_posix(),
                        "validation_call_predictions": call_path.relative_to(
                            REPO_ROOT
                        ).as_posix(),
                        "audit": audit,
                    }
                    write_json(summary_path, fit)
                    completed.append(fit)
                    pair.append(fit)
                common_epochs = min(len(item["audit"]["history"]) for item in pair)
                for epoch in range(common_epochs):
                    left = pair[0]["audit"]["history"][epoch]["train_audit"]
                    right = pair[1]["audit"]["history"][epoch]["train_audit"]
                    if (
                        left["cat_order_sha256"] != right["cat_order_sha256"]
                        or left["call_coverage_sha256"]
                        != right["call_coverage_sha256"]
                    ):
                        raise RuntimeError("IDEA-069 paired batch order differs")
    summary = aggregate(completed, protocol)
    write_json(run_root / "seed_replication_summary.json", summary)
    write_json(
        run_root / "run_summary.json",
        {
            "status": "complete",
            "completed_fits": len(completed),
            "expected_fits": int(screen_spec["total_fits"]),
            "gate_passed": summary["gate_passed"],
        },
    )
    return summary


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
