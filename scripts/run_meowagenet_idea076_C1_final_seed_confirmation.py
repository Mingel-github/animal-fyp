"""Run IDEA-076 final new-seed confirmation of bounded additive C1 on MeowAgeNet."""

from __future__ import annotations

import argparse
import hashlib
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
import run_meowagenet_idea069_age_residual_seed_replication as idea069  # noqa: E402
import run_meowagenet_idea071_bounded_dual_path_fusion as idea071  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea076_C1_final_seed_confirmation_v1.json"
)
PIPELINES = (
    "A0_ast_only",
    "U1_wide_unbounded_additive",
    "C1_bounded_wide_additive",
)
BASE_SEEDS = (8807, 268, 6915, 5994, 9330, 4322)
EXPECTED_PARAMETERS = {
    "A0_ast_only": 99_075,
    "U1_wide_unbounded_additive": 108_143,
    "C1_bounded_wide_additive": 108_143,
}
_IDEA068_BUILD_MODEL = idea068.build_model
_IDEA068_PREDICT = idea068.predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "run"), required=True)
    parser.add_argument(
        "--output-subdir", default="meowagenet_idea076_C1_final_seed_confirmation_v1"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON with canonical LF newlines on every platform."""
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


class WideUnboundedAdditiveClassifier(idea068.AgeResidualClassifier):
    """Parameter-matched unbounded additive control for bounded C1."""

    def __init__(
        self,
        ast_mean: np.ndarray,
        ast_scale: np.ndarray,
        age_train: np.ndarray,
        dropout: float,
    ) -> None:
        super().__init__(
            pipeline="A1_age_residual",
            ast_mean=ast_mean,
            ast_scale=ast_scale,
            age_train=age_train,
            dropout=dropout,
            age_hidden_units=idea071.CAPACITY_CONTROL_HIDDEN_UNITS,
        )
        self.pipeline = "U1_wide_unbounded_additive"

    def audit(self) -> dict[str, Any]:
        result = super().audit()
        result["fusion"] = "wide_unbounded_additive"
        return result


def build_model(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    age_features: np.ndarray,
    train_indices: np.ndarray,
) -> torch.nn.Module:
    if pipeline == PIPELINES[0]:
        return _IDEA068_BUILD_MODEL(
            pipeline, protocol, store, age_features, train_indices
        )
    embeddings = store.frozen_embeddings[train_indices]
    common = {
        "ast_mean": embeddings.mean(axis=0),
        "ast_scale": embeddings.std(axis=0),
        "age_train": age_features[train_indices],
        "dropout": float(protocol["fixed_training"]["dropout"]),
    }
    if pipeline == PIPELINES[1]:
        return WideUnboundedAdditiveClassifier(**common)
    if pipeline == PIPELINES[2]:
        return idea071.BoundedWideAdditiveClassifier(**common)
    raise ValueError(pipeline)


def predict_with_perturbation_reset(
    model: torch.nn.Module,
    store: Any,
    age_features: np.ndarray,
    indices: np.ndarray,
    cat_batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if hasattr(model, "reset_perturbation_audit"):
        model.reset_perturbation_audit()
    return _IDEA068_PREDICT(
        model, store, age_features, indices, cat_batch_size, device, seed
    )


def fit_inner(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    features: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    original_build = idea068.build_model
    original_predict = idea068.predict
    idea068.build_model = build_model
    idea068.predict = predict_with_perturbation_reset
    try:
        return idea068.fit_inner(
            pipeline,
            protocol,
            store,
            features,
            train_indices,
            validation_indices,
            device,
            seed,
        )
    finally:
        idea068.build_model = original_build
        idea068.predict = original_predict


def initial_logit_differences(
    protocol: dict[str, Any],
    store: Any,
    features: np.ndarray,
    train_indices: np.ndarray,
    probe_indices: np.ndarray,
    seed: int,
) -> dict[str, float]:
    logits = {}
    parameters = {}
    for pipeline in PIPELINES:
        idea068.idea051.reference.historical.set_seed(seed)
        model = build_model(pipeline, protocol, store, features, train_indices).eval()
        parameters[pipeline] = sum(parameter.numel() for parameter in model.parameters())
        with torch.no_grad():
            logits[pipeline] = model(
                torch.from_numpy(store.frozen_embeddings[probe_indices]),
                torch.from_numpy(features[probe_indices]),
            ).cpu().numpy()
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"IDEA-076 parameter audit mismatch: {parameters}")
    differences = {
        pipeline: float(np.max(np.abs(logits[pipeline] - logits[PIPELINES[0]])))
        for pipeline in PIPELINES[1:]
    }
    if any(value != 0.0 for value in differences.values()):
        raise RuntimeError("IDEA-076 pipelines are not identical at initialization")
    return differences


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea076-C1-final-seed-confirmation-v1":
        raise RuntimeError("Unexpected IDEA-076 protocol")
    if protocol.get("status") != "locked_before_initial_evaluation":
        raise RuntimeError("IDEA-076 protocol is not locked")
    model = protocol["model"]
    if tuple(model["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-076 pipeline matrix changed")
    if tuple(model["base_seeds"]) != BASE_SEEDS:
        raise RuntimeError("IDEA-076 base-seed bank changed")
    if model["repeats"] != [0, 1, 2] or model["folds"] != [0, 1, 2, 3]:
        raise RuntimeError("IDEA-076 split scope changed")
    if model.get("outer_test_predictions") is not False:
        raise RuntimeError("IDEA-076 must not access outer-test predictions")
    if float(model["cap"]) != idea071.CAP:
        raise RuntimeError("IDEA-076 cap changed")
    if float(model["rms_epsilon"]) != idea071.RMS_EPSILON:
        raise RuntimeError("IDEA-076 RMS epsilon changed")
    if int(model["wide_hidden_units"]) != idea071.CAPACITY_CONTROL_HIDDEN_UNITS:
        raise RuntimeError("IDEA-076 width changed")
    if model["trainable_parameters"] != EXPECTED_PARAMETERS:
        raise RuntimeError("IDEA-076 parameter lock changed")
    expected_fits = len(PIPELINES) * len(BASE_SEEDS) * 3 * 4
    if expected_fits != 216 or int(model["total_fits"]) != expected_fits:
        raise RuntimeError("IDEA-076 fit budget is inconsistent")
    full_seeds = {
        base + 10_000 * repeat + 100 * fold
        for base in BASE_SEEDS
        for repeat in range(3)
        for fold in range(4)
    }
    if len(full_seeds) != 72:
        raise RuntimeError("IDEA-076 derived full seeds are not unique")
    excluded_full_seeds = {
        int(base) + 10_000 * repeat + 100 * fold
        for base in model["excluded_meow_base_seeds_IDEA068_through_IDEA074"]
        for repeat in range(3)
        for fold in range(4)
    } | {
        int(base) + 10_000 * repeat + 100 * fold
        for base in model["excluded_dog_base_seeds_IDEA075"]
        for repeat in range(3)
        for fold in range(5)
    }
    if full_seeds & excluded_full_seeds:
        raise RuntimeError("IDEA-076 derived full seed collides with IDEA-068 through IDEA-075")
    digest = hashlib.sha256(model["seed_derivation_text"].encode("utf-8")).hexdigest()
    if digest != model["seed_derivation_sha256"]:
        raise RuntimeError("IDEA-076 seed derivation digest changed")
    source_protocol = read_json(idea068.PROTOCOL_PATH)
    if protocol["fixed_training"] != source_protocol["fixed_training"]:
        raise RuntimeError("IDEA-076 changed the locked IDEA-068 training recipe")
    if protocol["determinism"] != source_protocol["determinism"]:
        raise RuntimeError("IDEA-076 changed the locked IDEA-068 determinism recipe")
    dependencies = protocol["dependencies"]
    checks = {
        REPO_ROOT / dependencies["idea_path"]: dependencies["idea_sha256"],
        REPO_ROOT / dependencies["evidence_ledger_path"]: dependencies["evidence_ledger_sha256"],
        idea068.PROTOCOL_PATH: dependencies["idea068_protocol_sha256"],
        Path(idea068.__file__).resolve(): dependencies["idea068_runner_sha256"],
        idea071.PROTOCOL_PATH: dependencies["idea071_protocol_sha256"],
        Path(idea071.__file__).resolve(): dependencies["idea071_runner_sha256"],
        REPO_ROOT / dependencies["idea071_summary_path"]: dependencies["idea071_summary_sha256"],
        REPO_ROOT / dependencies["idea072_summary_path"]: dependencies["idea072_summary_sha256"],
        REPO_ROOT / dependencies["idea073_summary_path"]: dependencies["idea073_summary_sha256"],
        REPO_ROOT / dependencies["idea074_summary_path"]: dependencies["idea074_summary_sha256"],
        REPO_ROOT / protocol["data"]["roles_path"]: protocol["data"]["roles_sha256"],
        REPO_ROOT / protocol["data"]["frozen_embedding_path"]: protocol["data"]["frozen_embedding_sha256"],
        REPO_ROOT / protocol["data"]["fbank_path"]: protocol["data"]["fbank_sha256"],
        REPO_ROOT / protocol["data"]["feature_path"]: protocol["data"]["feature_sha256"],
        REPO_ROOT / protocol["data"]["feature_summary_path"]: protocol["data"]["feature_summary_sha256"],
        Path(__file__).resolve(): dependencies["runner_sha256"],
    }
    for path, expected_sha in checks.items():
        if not path.is_file() or idea068.sha256(path) != expected_sha:
            raise RuntimeError(f"IDEA-076 dependency checksum mismatch: {path}")


def resolve_run_root(output_subdir: str) -> Path:
    run_root = (idea068.idea051.RUNS_ROOT / output_subdir).resolve()
    if idea068.idea051.RUNS_ROOT.resolve() not in run_root.parents:
        raise ValueError("--output-subdir must stay below runs")
    return run_root


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
            raise RuntimeError(f"IDEA-076 resume identity mismatch for {key}")
    if any(value != 0.0 for value in fit["initial_logit_differences"].values()):
        raise RuntimeError("IDEA-076 resume initial-logit audit mismatch")
    for prefix in ("validation_animal", "validation_call"):
        path = REPO_ROOT / fit[f"{prefix}_predictions"]
        if not path.is_file() or idea068.sha256(path) != fit[f"{prefix}_sha256"]:
            raise RuntimeError(f"IDEA-076 resume prediction hash mismatch: {path}")


def load_animals(fit: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(
        REPO_ROOT / fit["validation_animal_predictions"], dtype={"cat_id": str}
    )


def metric_bundle(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "metrics": idea068.idea051.animal_metrics(frame),
        "cross_entropy": idea068.idea051.animal_cross_entropy(frame),
        "brier": idea068.brier(frame),
    }


def add_pipeline_metrics(row: dict[str, Any], bundles: dict[str, Any]) -> None:
    for pipeline in PIPELINES:
        row[f"{pipeline}_macro_f1"] = bundles[pipeline]["metrics"]["macro_f1"]
        row[f"{pipeline}_balanced_accuracy"] = bundles[pipeline]["metrics"][
            "balanced_accuracy"
        ]
        row[f"{pipeline}_cross_entropy"] = bundles[pipeline]["cross_entropy"]
        row[f"{pipeline}_brier"] = bundles[pipeline]["brier"]
        row[f"{pipeline}_senior_recall"] = bundles[pipeline]["metrics"][
            "per_class"
        ]["senior"]["recall"]
    row["C1_minus_A0_macro_f1"] = (
        row[f"{PIPELINES[2]}_macro_f1"] - row[f"{PIPELINES[0]}_macro_f1"]
    )
    row["C1_minus_U1_macro_f1"] = (
        row[f"{PIPELINES[2]}_macro_f1"] - row[f"{PIPELINES[1]}_macro_f1"]
    )
    row["U1_minus_A0_macro_f1"] = (
        row[f"{PIPELINES[1]}_macro_f1"] - row[f"{PIPELINES[0]}_macro_f1"]
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
                    pipeline: load_animals(by_key[(pipeline, base_seed, repeat, fold)])
                    for pipeline in PIPELINES
                }
                bundles = {
                    pipeline: metric_bundle(frame) for pipeline, frame in frames.items()
                }
                row: dict[str, Any] = {
                    "base_seed": base_seed,
                    "repeat": repeat,
                    "fold": fold,
                }
                add_pipeline_metrics(row, bundles)
                fold_results.append(row)
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
            bundles = {
                pipeline: metric_bundle(frame) for pipeline, frame in pooled.items()
            }
            row = {"base_seed": base_seed, "repeat": repeat}
            add_pipeline_metrics(row, bundles)
            seed_repeat_results.append(row)
    folds = pd.DataFrame(fold_results)
    seed_repeats = pd.DataFrame(seed_repeat_results)
    split_columns = ["C1_minus_A0_macro_f1", "C1_minus_U1_macro_f1"]
    split_cells = (
        folds.groupby(["repeat", "fold"], as_index=False)[split_columns]
        .mean()
        .sort_values(["repeat", "fold"])
        .reset_index(drop=True)
    )
    pooled_frames = {
        pipeline: pd.concat(parts, ignore_index=True)
        for pipeline, parts in pooled_all.items()
    }
    pooled_metrics = {
        pipeline: {"animal_occurrences": int(len(frame)), **metric_bundle(frame)}
        for pipeline, frame in pooled_frames.items()
    }
    per_seed_deltas = {}
    per_seed_senior = {}
    for seed in model["base_seeds"]:
        selected_rows = seed_repeats[seed_repeats["base_seed"] == seed]
        per_seed_deltas[str(seed)] = {
            name: float(selected_rows[name].mean()) for name in split_columns
        }
        selected_frames = {
            pipeline: frame[frame["base_seed"] == seed]
            for pipeline, frame in pooled_frames.items()
        }
        senior_recalls = {
            pipeline: idea068.idea051.animal_metrics(frame)["per_class"]["senior"]
            ["recall"]
            for pipeline, frame in selected_frames.items()
        }
        per_seed_senior[str(seed)] = {
            "C1_minus_A0": float(
                senior_recalls[PIPELINES[2]] - senior_recalls[PIPELINES[0]]
            ),
        }
    delta_a0 = seed_repeats["C1_minus_A0_macro_f1"]
    delta_u1 = seed_repeats["C1_minus_U1_macro_f1"]
    split_delta = split_cells["C1_minus_A0_macro_f1"]
    gate = protocol["gate"]
    mean_cross_entropy = {
        pipeline: float(seed_repeats[f"{pipeline}_cross_entropy"].mean())
        for pipeline in PIPELINES
    }
    mean_brier = {
        pipeline: float(seed_repeats[f"{pipeline}_brier"].mean())
        for pipeline in PIPELINES
    }
    mean_balanced_accuracy = {
        pipeline: float(seed_repeats[f"{pipeline}_balanced_accuracy"].mean())
        for pipeline in PIPELINES
    }
    main_conditions = {
        "mean_C1_minus_A0_macro_f1": float(delta_a0.mean())
        >= float(gate["minimum_mean_seed_repeat_C1_minus_A0"]),
        "positive_base_seed_means_C1_minus_A0": sum(
            values["C1_minus_A0_macro_f1"] > 0.0
            for values in per_seed_deltas.values()
        ) >= int(gate["minimum_positive_base_seeds_C1_minus_A0"]),
        "positive_C1_minus_A0_seed_repeats": int((delta_a0 > 0).sum())
        >= int(gate["minimum_positive_seed_repeats_C1_minus_A0"]),
        "nonnegative_C1_minus_A0_split_cells": int((split_delta >= 0).sum())
        >= int(gate["minimum_nonnegative_split_cells_C1_minus_A0"]),
        "worst_C1_minus_A0_split_cell": float(split_delta.min())
        >= float(gate["minimum_worst_split_cell_delta"]),
        "mean_cross_entropy_vs_A0": mean_cross_entropy[PIPELINES[2]]
        <= mean_cross_entropy[PIPELINES[0]],
        "mean_brier_vs_A0": mean_brier[PIPELINES[2]]
        <= mean_brier[PIPELINES[0]],
        "mean_balanced_accuracy_vs_A0": mean_balanced_accuracy[PIPELINES[2]]
        >= mean_balanced_accuracy[PIPELINES[0]],
        "per_base_seed_senior_recall_vs_A0": all(
            value["C1_minus_A0"]
            >= float(gate["minimum_per_base_seed_senior_recall_delta_vs_A0"])
            for value in per_seed_senior.values()
        ),
    }
    mechanism_conditions = {
        "mean_C1_minus_U1_strictly_positive": float(delta_u1.mean()) > 0.0,
        "positive_base_seed_means_C1_minus_U1": sum(
            values["C1_minus_U1_macro_f1"] > 0.0
            for values in per_seed_deltas.values()
        ) >= int(gate["minimum_positive_base_seeds_C1_minus_U1"]),
        "positive_C1_minus_U1_seed_repeats": int((delta_u1 > 0).sum())
        >= int(gate["minimum_positive_seed_repeats_C1_minus_U1"]),
        "mean_cross_entropy_vs_U1": mean_cross_entropy[PIPELINES[2]]
        <= mean_cross_entropy[PIPELINES[1]],
        "mean_brier_vs_U1": mean_brier[PIPELINES[2]]
        <= mean_brier[PIPELINES[1]],
    }
    main_passed = bool(all(main_conditions.values()))
    mechanism_raw_passed = bool(all(mechanism_conditions.values()))
    pipeline_means = {
        pipeline: float(seed_repeats[f"{pipeline}_macro_f1"].mean())
        for pipeline in PIPELINES
    }
    comparison_columns = split_columns + ["U1_minus_A0_macro_f1"]
    comparisons = {}
    for name in comparison_columns:
        values = seed_repeats[name]
        comparisons[name] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            "median": float(values.median()),
            "positive": int((values > 0).sum()),
            "tied": int((values == 0).sum()),
            "negative": int((values < 0).sum()),
            "worst": float(values.min()),
            "best": float(values.max()),
        }
    return {
        "status": "complete",
        "outer_test_accessed": False,
        "fits": len(fits),
        "paired_fold_comparisons": len(fold_results),
        "seed_repeat_estimates": len(seed_repeat_results),
        "split_cell_estimates": int(len(split_cells)),
        "independence_note": (
            "Fold deltas and pooled animal occurrences repeat animals across "
            "seeds/repeats and are descriptive, not independent samples."
        ),
        "fold_results": fold_results,
        "seed_repeat_results": seed_repeat_results,
        "split_cell_results": split_cells.to_dict(orient="records"),
        "seed_repeat_level": {
            "pipeline_mean_macro_f1": pipeline_means,
            "pipeline_mean_balanced_accuracy": mean_balanced_accuracy,
            "pipeline_mean_cross_entropy": mean_cross_entropy,
            "pipeline_mean_brier": mean_brier,
            "comparisons": comparisons,
        },
        "split_cell_level": {
            "C1_minus_A0_nonnegative": int((split_delta >= 0).sum()),
            "C1_minus_A0_positive": int((split_delta > 0).sum()),
            "C1_minus_A0_tied": int((split_delta == 0).sum()),
            "C1_minus_A0_negative": int((split_delta < 0).sum()),
            "C1_minus_A0_worst": float(split_delta.min()),
            "C1_minus_A0_best": float(split_delta.max()),
        },
        "per_base_seed_mean_deltas": per_seed_deltas,
        "per_base_seed_senior_recall_deltas": per_seed_senior,
        "pooled_validation": pooled_metrics,
        "gate_conditions": {
            "C1_vs_A0_main": main_conditions,
            "C1_vs_U1_mechanism": mechanism_conditions,
        },
        "main_gate_passed": main_passed,
        "bounded_mechanism_interpretable": main_passed,
        "bounded_mechanism_raw_conditions_passed": mechanism_raw_passed,
        "bounded_mechanism_passed": bool(main_passed and mechanism_raw_passed),
        "gate_passed": main_passed,
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Perform a read-only, fail-closed audit before any run directory is written."""
    idea068.configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    store = idea068.idea051.reference.historical.idea019.load_feature_store()
    if len(store.call_ids) != 792 or len(np.unique(store.cat_ids.astype(str))) != 111:
        raise RuntimeError("IDEA-076 expected exactly 792 calls from 111 cats")
    features, feature_summary = idea069.load_features(protocol, store.call_ids)
    roles = pd.read_csv(
        REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str}
    )
    if set(roles["role"].astype(str)) != {"train", "validation", "test"}:
        raise RuntimeError("IDEA-076 nested role vocabulary changed")
    role_cells = 0
    for repeat in protocol["model"]["repeats"]:
        for fold in protocol["model"]["folds"]:
            cell = roles[(roles["repeat"] == repeat) & (roles["outer_fold"] == fold)]
            cats = {
                role: set(cell[cell["role"] == role]["cat_id"].astype(str))
                for role in ("train", "validation", "test")
            }
            if any(cats[left] & cats[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
                raise RuntimeError("IDEA-076 cat leakage across nested roles")
            indices = idea068.idea051.reference.historical.fold_indices(
                store, roles, repeat, fold, include_test=False
            )
            if np.intersect1d(indices["train"], indices["validation"]).size:
                raise RuntimeError("IDEA-076 call leakage across train/validation")
            role_cells += 1
    probe_indices = np.arange(0, 32, dtype=np.int64)
    train_indices = np.arange(32, 532, dtype=np.int64)
    initial = initial_logit_differences(
        protocol, store, features, train_indices, probe_indices, BASE_SEEDS[0]
    )
    states = {}
    for pipeline in (PIPELINES[1], PIPELINES[2]):
        idea068.idea051.reference.historical.set_seed(BASE_SEEDS[0])
        states[pipeline] = build_model(
            pipeline, protocol, store, features, train_indices
        ).state_dict()
    if states[PIPELINES[1]].keys() != states[PIPELINES[2]].keys() or not all(
        torch.equal(states[PIPELINES[1]][key], states[PIPELINES[2]][key])
        for key in states[PIPELINES[1]]
    ):
        raise RuntimeError("IDEA-076 U1/C1 initial states differ")
    c1 = build_model(PIPELINES[2], protocol, store, features, train_indices).eval()
    torch.nn.init.constant_(c1.age_output.weight, 100.0)
    torch.nn.init.constant_(c1.age_output.bias, 100.0)
    c1.reset_perturbation_audit()
    with torch.no_grad():
        c1(
            torch.from_numpy(store.frozen_embeddings[probe_indices]),
            torch.from_numpy(features[probe_indices]),
        )
    perturbation = c1.perturbation_audit()
    if perturbation["validation_relative_perturbation_max"] > idea071.CAP + 1e-5:
        raise RuntimeError("IDEA-076 C1 perturbation budget failed")
    device = idea068.idea051.reference.historical.idea019.resolve_device(args.device)
    return {
        "status": "GO",
        "read_only": True,
        "protocol_sha256": idea068.sha256(PROTOCOL_PATH),
        "runner_sha256": idea068.sha256(Path(__file__).resolve()),
        "feature_sha256": feature_summary["feature_sha256"],
        "calls": int(len(store.call_ids)),
        "cats": int(len(np.unique(store.cat_ids.astype(str)))),
        "role_cells": role_cells,
        "base_seeds": list(BASE_SEEDS),
        "unique_full_seeds": 72,
        "expected_fits": 216,
        "outer_test_accessed": False,
        "initial_logit_differences": initial,
        "U1_C1_initial_state_equal": True,
        "C1_budget_max_under_saturation_probe": perturbation[
            "validation_relative_perturbation_max"
        ],
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    idea068.configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = resolve_run_root(args.output_subdir)
    run_root.mkdir(parents=True, exist_ok=True)
    store = idea068.idea051.reference.historical.idea019.load_feature_store()
    if len(store.call_ids) != 792 or len(np.unique(store.cat_ids.astype(str))) != 111:
        raise RuntimeError("IDEA-076 expected exactly 792 calls from 111 cats")
    features, feature_summary = idea069.load_features(protocol, store.call_ids)
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
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.exists():
        if not args.resume or read_json(manifest_path) != manifest:
            raise RuntimeError("Existing IDEA-076 run manifest differs")
    else:
        write_json(manifest_path, manifest)
    completed = []
    for base_seed in protocol["model"]["base_seeds"]:
        for repeat in protocol["model"]["repeats"]:
            for fold in protocol["model"]["folds"]:
                indices = idea068.idea051.reference.historical.fold_indices(
                    store, roles, repeat, fold, include_test=False
                )
                full_seed = idea068.idea051.reference.historical.full_seed(
                    int(base_seed), repeat, fold
                )
                initial_differences = initial_logit_differences(
                    protocol,
                    store,
                    features,
                    indices["train"],
                    indices["validation"][: min(32, len(indices["validation"]))],
                    full_seed,
                )
                triplet = []
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
                    else:
                        print(
                            f"=== {pipeline} base_seed={base_seed} repeat={repeat} "
                            f"fold={fold} full_seed={full_seed} ===",
                            flush=True,
                        )
                        audit, animals, calls = fit_inner(
                            pipeline,
                            protocol,
                            store,
                            features,
                            indices["train"],
                            indices["validation"],
                            device,
                            full_seed,
                        )
                        expected_parameters = EXPECTED_PARAMETERS[pipeline]
                        if audit["model"]["trainable_parameters"] != expected_parameters:
                            raise RuntimeError("IDEA-076 trained parameter audit mismatch")
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
                            "initial_logit_differences": initial_differences,
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
                    triplet.append(fit)
                common_epochs = min(len(item["audit"]["history"]) for item in triplet)
                for epoch in range(common_epochs):
                    reference = triplet[0]["audit"]["history"][epoch]["train_audit"]
                    for candidate in triplet[1:]:
                        current = candidate["audit"]["history"][epoch]["train_audit"]
                        if (
                            reference["cat_order_sha256"] != current["cat_order_sha256"]
                            or reference["call_coverage_sha256"]
                            != current["call_coverage_sha256"]
                        ):
                            raise RuntimeError("IDEA-076 paired batch order differs")
    summary = aggregate(completed, protocol)
    summary_path = run_root / "initial_evaluation_summary.json"
    summary_bytes = canonical_json_bytes(summary)
    if summary_path.exists():
        if not args.resume:
            raise FileExistsError(summary_path)
        existing_bytes = summary_path.read_bytes()
        if read_json(summary_path) != summary:
            raise RuntimeError("IDEA-076 resumed aggregate semantic mismatch")
        if existing_bytes != summary_bytes:
            raise RuntimeError("IDEA-076 resumed aggregate canonical-byte mismatch")
    else:
        write_json(summary_path, summary)
    compact = {
        "status": "complete",
        "completed_fits": len(completed),
        "expected_fits": int(protocol["model"]["total_fits"]),
        "main_gate_passed": summary["main_gate_passed"],
        "bounded_mechanism_interpretable": summary[
            "bounded_mechanism_interpretable"
        ],
        "bounded_mechanism_passed": summary["bounded_mechanism_passed"],
        "gate_passed": summary["gate_passed"],
    }
    compact_path = run_root / "run_summary.json"
    if compact_path.exists():
        if read_json(compact_path) != compact or compact_path.read_bytes() != canonical_json_bytes(compact):
            raise RuntimeError("IDEA-076 resumed compact summary mismatch")
    else:
        write_json(compact_path, compact)
    return summary


def main() -> None:
    args = parse_args()
    result = preflight(args) if args.stage == "preflight" else run(args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
