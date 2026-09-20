"""Run the deterministic IDEA-069 A0-vs-A1 independent-seed replication."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

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
    / "meowagenet_idea069_age_residual_seed_replication_v1.json"
)
IDEA068_PROTOCOL_PATH = idea068.PROTOCOL_PATH
PIPELINES = ("A0_ast_only", "A1_age_residual")
BASE_SEEDS = (151, 307, 509)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-subdir",
        default="meowagenet_idea069_age_residual_seed_replication_v1",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea069-age-residual-seed-replication-v1":
        raise RuntimeError("Unexpected IDEA-069 protocol")
    if protocol.get("status") != "locked_before_seed_replication":
        raise RuntimeError("IDEA-069 protocol is not locked")
    model = protocol["model"]
    if tuple(model["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-069 pipeline matrix changed")
    if tuple(model["base_seeds"]) != BASE_SEEDS:
        raise RuntimeError("IDEA-069 base-seed bank changed")
    if model["repeats"] != [0, 1, 2] or model["folds"] != [0, 1, 2, 3]:
        raise RuntimeError("IDEA-069 split scope changed")
    if model.get("outer_test_predictions") is not False:
        raise RuntimeError("IDEA-069 must not access outer-test predictions")
    expected_fits = (
        len(PIPELINES)
        * len(model["base_seeds"])
        * len(model["repeats"])
        * len(model["folds"])
    )
    if expected_fits != 72 or expected_fits != int(model["total_fits"]):
        raise RuntimeError("IDEA-069 fit budget is inconsistent")
    full_seeds = {
        int(base_seed) + 10_000 * int(repeat) + 100 * int(fold)
        for base_seed in model["base_seeds"]
        for repeat in model["repeats"]
        for fold in model["folds"]
    }
    if len(full_seeds) != 36:
        raise RuntimeError("IDEA-069 derived full seeds are not unique")
    source_protocol = read_json(IDEA068_PROTOCOL_PATH)
    if protocol["fixed_training"] != source_protocol["fixed_training"]:
        raise RuntimeError("IDEA-069 changed the locked IDEA-068 training recipe")
    if protocol["determinism"] != source_protocol["determinism"]:
        raise RuntimeError("IDEA-069 changed the locked IDEA-068 determinism recipe")
    dependencies = protocol["dependencies"]
    checks = {
        REPO_ROOT / dependencies["idea_path"]: dependencies["idea_sha256"],
        IDEA068_PROTOCOL_PATH: dependencies["idea068_protocol_sha256"],
        Path(idea068.__file__).resolve(): dependencies["idea068_runner_sha256"],
        REPO_ROOT / protocol["data"]["roles_path"]: protocol["data"][
            "roles_sha256"
        ],
        REPO_ROOT / protocol["data"]["frozen_embedding_path"]: protocol["data"][
            "frozen_embedding_sha256"
        ],
        REPO_ROOT / protocol["data"]["fbank_path"]: protocol["data"][
            "fbank_sha256"
        ],
        REPO_ROOT / protocol["data"]["feature_path"]: protocol["data"][
            "feature_sha256"
        ],
        REPO_ROOT / protocol["data"]["feature_summary_path"]: protocol["data"][
            "feature_summary_sha256"
        ],
        Path(__file__).resolve(): dependencies["runner_sha256"],
    }
    for path, expected_sha in checks.items():
        if not path.is_file() or idea068.sha256(path) != expected_sha:
            raise RuntimeError(f"IDEA-069 dependency checksum mismatch: {path}")


def resolve_run_root(output_subdir: str) -> Path:
    run_root = (idea068.idea051.RUNS_ROOT / output_subdir).resolve()
    if idea068.idea051.RUNS_ROOT.resolve() not in run_root.parents:
        raise ValueError("--output-subdir must stay below runs")
    return run_root


def load_features(
    protocol: dict[str, Any], call_ids: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    summary = read_json(REPO_ROOT / protocol["data"]["feature_summary_path"])
    if summary.get("status") != "complete" or summary.get("label_information_used") is not False:
        raise RuntimeError("IDEA-069 source features failed their extraction audit")
    expected_summary = protocol["data"]["feature_extraction_audit"]
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"IDEA-069 feature extraction audit changed: {key}")
    feature_path = REPO_ROOT / protocol["data"]["feature_path"]
    if idea068.sha256(feature_path) != summary.get("feature_sha256"):
        raise RuntimeError("IDEA-069 source feature hash differs from extraction audit")
    loaded = np.load(feature_path)
    if tuple(loaded["feature_names"].astype(str)) != idea068.FEATURE_NAMES:
        raise RuntimeError("IDEA-069 feature definition changed")
    if not np.array_equal(loaded["call_ids"].astype(str), call_ids.astype(str)):
        raise RuntimeError("IDEA-069 feature and AST call order differs")
    features = loaded["features"].astype(np.float32)
    if features.shape != (len(call_ids), len(idea068.FEATURE_NAMES)):
        raise RuntimeError("IDEA-069 feature matrix has the wrong shape")
    return features, summary


def initial_logit_max_difference(
    protocol: dict[str, Any],
    store: Any,
    features: np.ndarray,
    train_indices: np.ndarray,
    probe_indices: np.ndarray,
    seed: int,
) -> float:
    """Confirm zero-init makes A1 exactly equal to A0 before optimization."""
    logits = []
    for pipeline in PIPELINES:
        idea068.idea051.reference.historical.set_seed(seed)
        model = idea068.build_model(
            pipeline, protocol, store, features, train_indices
        ).eval()
        with torch.no_grad():
            logits.append(
                model(
                    torch.from_numpy(store.frozen_embeddings[probe_indices]),
                    torch.from_numpy(features[probe_indices]),
                ).cpu().numpy()
            )
    difference = float(np.max(np.abs(logits[0] - logits[1])))
    if difference != 0.0:
        raise RuntimeError("IDEA-069 zero-init A0/A1 logits are not identical")
    return difference


def validate_completed_fit(
    fit: dict[str, Any],
    pipeline: str,
    base_seed: int,
    full_seed: int,
    repeat: int,
    fold: int,
) -> None:
    expected = {
        "status": "complete",
        "pipeline": pipeline,
        "base_seed": base_seed,
        "full_seed": full_seed,
        "repeat": repeat,
        "fold": fold,
        "outer_test_accessed": False,
    }
    for key, value in expected.items():
        if fit.get(key) != value:
            raise RuntimeError(f"IDEA-069 resume identity mismatch for {key}")
    if fit.get("initial_pair_max_logit_difference") != 0.0:
        raise RuntimeError("IDEA-069 resume zero-init audit mismatch")
    for prefix in ("validation_animal", "validation_call"):
        path = REPO_ROOT / fit[f"{prefix}_predictions"]
        if not path.is_file() or idea068.sha256(path) != fit[f"{prefix}_sha256"]:
            raise RuntimeError(f"IDEA-069 resume prediction hash mismatch: {path}")


def load_animals(fit: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(
        REPO_ROOT / fit["validation_animal_predictions"], dtype={"cat_id": str}
    )


def aggregate(fits: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    model = protocol["model"]
    by_key = {
        (fit["pipeline"], fit["base_seed"], fit["repeat"], fit["fold"]): fit
        for fit in fits
    }
    fold_results = []
    seed_repeat_results = []
    pooled_all: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
    for base_seed in model["base_seeds"]:
        for repeat in model["repeats"]:
            seed_repeat_frames: dict[str, list[pd.DataFrame]] = {
                pipeline: [] for pipeline in PIPELINES
            }
            for fold in model["folds"]:
                frames = {
                    pipeline: load_animals(
                        by_key[(pipeline, base_seed, repeat, fold)]
                    )
                    for pipeline in PIPELINES
                }
                metrics = {
                    pipeline: idea068.idea051.animal_metrics(frame)
                    for pipeline, frame in frames.items()
                }
                fold_results.append(
                    {
                        "base_seed": base_seed,
                        "repeat": repeat,
                        "fold": fold,
                        "control_macro_f1": metrics[PIPELINES[0]]["macro_f1"],
                        "candidate_macro_f1": metrics[PIPELINES[1]]["macro_f1"],
                        "delta_macro_f1": metrics[PIPELINES[1]]["macro_f1"]
                        - metrics[PIPELINES[0]]["macro_f1"],
                        "control_cross_entropy": idea068.idea051.animal_cross_entropy(
                            frames[PIPELINES[0]]
                        ),
                        "candidate_cross_entropy": idea068.idea051.animal_cross_entropy(
                            frames[PIPELINES[1]]
                        ),
                        "control_brier": idea068.brier(frames[PIPELINES[0]]),
                        "candidate_brier": idea068.brier(frames[PIPELINES[1]]),
                    }
                )
                for pipeline, frame in frames.items():
                    tagged = frame.copy()
                    tagged["base_seed"] = base_seed
                    tagged["repeat"] = repeat
                    tagged["fold"] = fold
                    seed_repeat_frames[pipeline].append(tagged)
                    pooled_all[pipeline].append(tagged)
            pooled = {
                pipeline: pd.concat(parts, ignore_index=True)
                for pipeline, parts in seed_repeat_frames.items()
            }
            metrics = {
                pipeline: idea068.idea051.animal_metrics(frame)
                for pipeline, frame in pooled.items()
            }
            seed_repeat_results.append(
                {
                    "base_seed": base_seed,
                    "repeat": repeat,
                    "control_macro_f1": metrics[PIPELINES[0]]["macro_f1"],
                    "candidate_macro_f1": metrics[PIPELINES[1]]["macro_f1"],
                    "delta_macro_f1": metrics[PIPELINES[1]]["macro_f1"]
                    - metrics[PIPELINES[0]]["macro_f1"],
                    "control_cross_entropy": idea068.idea051.animal_cross_entropy(
                        pooled[PIPELINES[0]]
                    ),
                    "candidate_cross_entropy": idea068.idea051.animal_cross_entropy(
                        pooled[PIPELINES[1]]
                    ),
                    "control_brier": idea068.brier(pooled[PIPELINES[0]]),
                    "candidate_brier": idea068.brier(pooled[PIPELINES[1]]),
                    "control_senior_recall": metrics[PIPELINES[0]]["per_class"][
                        "senior"
                    ]["recall"],
                    "candidate_senior_recall": metrics[PIPELINES[1]]["per_class"][
                        "senior"
                    ]["recall"],
                }
            )
    folds = pd.DataFrame(fold_results)
    seed_repeats = pd.DataFrame(seed_repeat_results)
    split_cells = (
        folds.groupby(["repeat", "fold"], as_index=False)
        .agg(
            control_macro_f1=("control_macro_f1", "mean"),
            candidate_macro_f1=("candidate_macro_f1", "mean"),
            delta_macro_f1=("delta_macro_f1", "mean"),
        )
        .sort_values(["repeat", "fold"])
        .reset_index(drop=True)
    )
    pooled_frames = {
        pipeline: pd.concat(parts, ignore_index=True)
        for pipeline, parts in pooled_all.items()
    }
    pooled_metrics = {
        pipeline: {
            "animal_occurrences": int(len(frame)),
            "metrics": idea068.idea051.animal_metrics(frame),
            "cross_entropy": idea068.idea051.animal_cross_entropy(frame),
            "brier": idea068.brier(frame),
        }
        for pipeline, frame in pooled_frames.items()
    }
    per_base_seed_mean_delta = {
        str(seed): float(
            seed_repeats.loc[
                seed_repeats["base_seed"] == seed, "delta_macro_f1"
            ].mean()
        )
        for seed in model["base_seeds"]
    }
    per_base_seed_senior_delta = {}
    for seed in model["base_seeds"]:
        metrics = {}
        for pipeline, frame in pooled_frames.items():
            selected = frame[frame["base_seed"] == seed]
            metrics[pipeline] = idea068.idea051.animal_metrics(selected)
        per_base_seed_senior_delta[str(seed)] = float(
            metrics[PIPELINES[1]]["per_class"]["senior"]["recall"]
            - metrics[PIPELINES[0]]["per_class"]["senior"]["recall"]
        )
    gate = protocol["gate"]
    conditions = {
        "mean_seed_repeat_macro_f1_gain": float(
            seed_repeats["delta_macro_f1"].mean()
        )
        >= float(gate["minimum_mean_seed_repeat_macro_f1_gain"]),
        "positive_seed_repeats": int((seed_repeats["delta_macro_f1"] > 0).sum())
        >= int(gate["minimum_positive_seed_repeats"]),
        "every_base_seed_mean_positive": all(
            value > float(gate["minimum_each_base_seed_mean_delta"])
            for value in per_base_seed_mean_delta.values()
        ),
        "nonnegative_split_cells": int(
            (split_cells["delta_macro_f1"] >= 0).sum()
        )
        >= int(gate["minimum_nonnegative_split_cells"]),
        "worst_split_cell_macro_f1": float(split_cells["delta_macro_f1"].min())
        >= float(gate["minimum_worst_split_cell_macro_f1_delta"]),
        "pooled_cross_entropy_nonworse": pooled_metrics[PIPELINES[1]][
            "cross_entropy"
        ]
        <= pooled_metrics[PIPELINES[0]]["cross_entropy"],
        "pooled_brier_nonworse": pooled_metrics[PIPELINES[1]]["brier"]
        <= pooled_metrics[PIPELINES[0]]["brier"],
        "per_base_seed_senior_recall": all(
            value >= float(gate["minimum_per_base_seed_senior_recall_delta"])
            for value in per_base_seed_senior_delta.values()
        ),
    }
    return {
        "status": "complete",
        "outer_test_accessed": False,
        "fits": len(fits),
        "paired_fold_comparisons": len(fold_results),
        "seed_repeat_estimates": len(seed_repeat_results),
        "split_cell_estimates": int(len(split_cells)),
        "independence_note": (
            "The 36 fold deltas and pooled animal occurrences repeat the same "
            "animals across seeds/repeats and are descriptive, not 36 independent samples."
        ),
        "fold_results": fold_results,
        "seed_repeat_results": seed_repeat_results,
        "split_cell_results": split_cells.to_dict(orient="records"),
        "fold_level": {
            "control_mean_macro_f1": float(folds["control_macro_f1"].mean()),
            "candidate_mean_macro_f1": float(folds["candidate_macro_f1"].mean()),
            "mean_delta_macro_f1": float(folds["delta_macro_f1"].mean()),
            "positive": int((folds["delta_macro_f1"] > 0).sum()),
            "tied": int((folds["delta_macro_f1"] == 0).sum()),
            "negative": int((folds["delta_macro_f1"] < 0).sum()),
        },
        "seed_repeat_level": {
            "control_mean_macro_f1": float(seed_repeats["control_macro_f1"].mean()),
            "candidate_mean_macro_f1": float(
                seed_repeats["candidate_macro_f1"].mean()
            ),
            "mean_delta_macro_f1": float(seed_repeats["delta_macro_f1"].mean()),
            "delta_sample_sd": float(seed_repeats["delta_macro_f1"].std(ddof=1)),
            "median_delta": float(seed_repeats["delta_macro_f1"].median()),
            "positive": int((seed_repeats["delta_macro_f1"] > 0).sum()),
            "tied": int((seed_repeats["delta_macro_f1"] == 0).sum()),
            "negative": int((seed_repeats["delta_macro_f1"] < 0).sum()),
            "worst_delta": float(seed_repeats["delta_macro_f1"].min()),
            "best_delta": float(seed_repeats["delta_macro_f1"].max()),
        },
        "split_cell_level": {
            "mean_delta_macro_f1": float(split_cells["delta_macro_f1"].mean()),
            "positive": int((split_cells["delta_macro_f1"] > 0).sum()),
            "tied": int((split_cells["delta_macro_f1"] == 0).sum()),
            "negative": int((split_cells["delta_macro_f1"] < 0).sum()),
            "nonnegative": int((split_cells["delta_macro_f1"] >= 0).sum()),
            "worst_delta": float(split_cells["delta_macro_f1"].min()),
            "best_delta": float(split_cells["delta_macro_f1"].max()),
        },
        "per_base_seed_mean_delta": per_base_seed_mean_delta,
        "per_base_seed_senior_recall_delta": per_base_seed_senior_delta,
        "pooled_validation": pooled_metrics,
        "gate_conditions": conditions,
        "gate_passed": bool(all(conditions.values())),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    idea068.configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = resolve_run_root(args.output_subdir)
    run_root.mkdir(parents=True, exist_ok=True)
    store = idea068.idea051.reference.historical.idea019.load_feature_store()
    if len(store.call_ids) != 792 or len(np.unique(store.cat_ids.astype(str))) != 111:
        raise RuntimeError("IDEA-069 expected exactly 792 calls from 111 cats")
    features, feature_summary = load_features(protocol, store.call_ids)
    roles = pd.read_csv(
        REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str}
    )
    device = idea068.idea051.reference.historical.idea019.resolve_device(args.device)
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": idea068.sha256(PROTOCOL_PATH),
        "runner_sha256": idea068.sha256(Path(__file__).resolve()),
        "source_feature_sha256": feature_summary["feature_sha256"],
        "outer_test_accessed": False,
        "pipelines": list(PIPELINES),
        "model": protocol["model"],
        "determinism": protocol["determinism"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
    }
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.exists():
        if not args.resume or read_json(manifest_path) != manifest:
            raise RuntimeError("Existing IDEA-069 run manifest differs")
    else:
        write_json(manifest_path, manifest)
    completed = []
    model = protocol["model"]
    for base_seed in model["base_seeds"]:
        for repeat in model["repeats"]:
            for fold in model["folds"]:
                indices = idea068.idea051.reference.historical.fold_indices(
                    store, roles, repeat, fold, include_test=False
                )
                full_seed = idea068.idea051.reference.historical.full_seed(
                    int(base_seed), repeat, fold
                )
                initial_difference = initial_logit_max_difference(
                    protocol,
                    store,
                    features,
                    indices["train"],
                    indices["validation"][: min(32, len(indices["validation"]))],
                    full_seed,
                )
                pair = []
                for pipeline in PIPELINES:
                    output_dir = (
                        run_root
                        / "fits"
                        / pipeline
                        / f"base_seed_{base_seed}"
                        / f"repeat_{repeat}"
                        / f"fold_{fold}"
                    )
                    summary_path = output_dir / "fit_summary.json"
                    if summary_path.exists():
                        if not args.resume:
                            raise FileExistsError(summary_path)
                        fit = read_json(summary_path)
                        validate_completed_fit(
                            fit, pipeline, base_seed, full_seed, repeat, fold
                        )
                        completed.append(fit)
                        pair.append(fit)
                        continue
                    print(
                        f"=== {pipeline} base_seed={base_seed} repeat={repeat} "
                        f"fold={fold} full_seed={full_seed} ===",
                        flush=True,
                    )
                    audit, animals, calls = idea068.fit_inner(
                        pipeline,
                        protocol,
                        store,
                        features,
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
                        "fold": int(fold),
                        "outer_test_accessed": False,
                        "initial_pair_max_logit_difference": initial_difference,
                        "train_calls": int(len(indices["train"])),
                        "validation_calls": int(len(indices["validation"])),
                        "validation_cats": int(animals["cat_id"].nunique()),
                        "validation_animal_predictions": animal_path.relative_to(
                            REPO_ROOT
                        ).as_posix(),
                        "validation_animal_sha256": idea068.sha256(animal_path),
                        "validation_call_predictions": call_path.relative_to(
                            REPO_ROOT
                        ).as_posix(),
                        "validation_call_sha256": idea068.sha256(call_path),
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
            "expected_fits": int(model["total_fits"]),
            "gate_passed": summary["gate_passed"],
        },
    )
    return summary


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
