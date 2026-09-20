"""Run IDEA-070 bounded age-conditioned AST modulation."""

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
import run_meowagenet_idea069_age_residual_seed_replication as idea069  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea070_bounded_age_modulation_v1.json"
)
PIPELINES = ("A0_ast_only", "A1_age_residual", "B1_bounded_age_modulation")
BASE_SEEDS = (2326, 3550, 4426)
MODULATION_CAP = 0.25
_IDEA068_BUILD_MODEL = idea068.build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-subdir", default="meowagenet_idea070_bounded_age_modulation_v1"
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


class BoundedAgeModulationClassifier(idea068.AgeResidualClassifier):
    """Age-conditioned feature-wise scaling with a fixed identity-centered cap."""

    def __init__(
        self,
        ast_mean: np.ndarray,
        ast_scale: np.ndarray,
        age_train: np.ndarray,
        dropout: float,
        age_hidden_units: int,
    ) -> None:
        super().__init__(
            pipeline="A1_age_residual",
            ast_mean=ast_mean,
            ast_scale=ast_scale,
            age_train=age_train,
            dropout=dropout,
            age_hidden_units=age_hidden_units,
        )
        self.pipeline = "B1_bounded_age_modulation"

    def forward(
        self, ast_embeddings: torch.Tensor, age_features: torch.Tensor
    ) -> torch.Tensor:
        ast = (ast_embeddings - self.ast_mean) / self.ast_scale
        hidden = self.relu(self.ast_linear(ast))
        if self.age_hidden is None or self.age_output is None:
            raise RuntimeError("IDEA-070 age modulation branch is missing")
        imputed = torch.where(
            torch.isfinite(age_features), age_features, self.age_median
        )
        standardized = (imputed - self.age_mean) / self.age_scale
        raw_scale = self.age_output(
            torch.nn.functional.gelu(self.age_hidden(standardized))
        )
        scale = MODULATION_CAP * torch.tanh(raw_scale)
        hidden = hidden * (1.0 + scale)
        hidden = self.batch_norm(hidden)
        hidden = self.dropout(hidden)
        return self.output(hidden)

    def audit(self) -> dict[str, Any]:
        result = super().audit()
        if self.age_output is None:
            raise RuntimeError("IDEA-070 age modulation branch is missing")
        result.update(
            {
                "modulation_cap": MODULATION_CAP,
                "age_output_weight_l2": float(
                    torch.linalg.vector_norm(self.age_output.weight.detach()).cpu()
                ),
                "age_output_bias_l2": float(
                    torch.linalg.vector_norm(self.age_output.bias.detach()).cpu()
                ),
            }
        )
        return result


def build_model(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    age_features: np.ndarray,
    train_indices: np.ndarray,
) -> torch.nn.Module:
    if pipeline in PIPELINES[:2]:
        return _IDEA068_BUILD_MODEL(
            pipeline, protocol, store, age_features, train_indices
        )
    if pipeline != PIPELINES[2]:
        raise ValueError(pipeline)
    embeddings = store.frozen_embeddings[train_indices]
    return BoundedAgeModulationClassifier(
        ast_mean=embeddings.mean(axis=0),
        ast_scale=embeddings.std(axis=0),
        age_train=age_features[train_indices],
        dropout=float(protocol["fixed_training"]["dropout"]),
        age_hidden_units=int(protocol["fixed_training"]["age_hidden_units"]),
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
    original = idea068.build_model
    idea068.build_model = build_model
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
        idea068.build_model = original


def initial_logit_differences(
    protocol: dict[str, Any],
    store: Any,
    features: np.ndarray,
    train_indices: np.ndarray,
    probe_indices: np.ndarray,
    seed: int,
) -> dict[str, float]:
    logits = {}
    for pipeline in PIPELINES:
        idea068.idea051.reference.historical.set_seed(seed)
        model = build_model(
            pipeline, protocol, store, features, train_indices
        ).eval()
        with torch.no_grad():
            logits[pipeline] = model(
                torch.from_numpy(store.frozen_embeddings[probe_indices]),
                torch.from_numpy(features[probe_indices]),
            ).cpu().numpy()
    differences = {
        pipeline: float(np.max(np.abs(logits[pipeline] - logits[PIPELINES[0]])))
        for pipeline in PIPELINES[1:]
    }
    if any(value != 0.0 for value in differences.values()):
        raise RuntimeError("IDEA-070 pipelines are not identical at initialization")
    return differences


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea070-bounded-age-modulation-v1":
        raise RuntimeError("Unexpected IDEA-070 protocol")
    if protocol.get("status") != "locked_before_initial_evaluation":
        raise RuntimeError("IDEA-070 protocol is not locked")
    model = protocol["model"]
    if tuple(model["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-070 pipeline matrix changed")
    if tuple(model["base_seeds"]) != BASE_SEEDS:
        raise RuntimeError("IDEA-070 base-seed bank changed")
    if model["repeats"] != [0, 1, 2] or model["folds"] != [0, 1, 2, 3]:
        raise RuntimeError("IDEA-070 split scope changed")
    if model.get("outer_test_predictions") is not False:
        raise RuntimeError("IDEA-070 must not access outer-test predictions")
    if float(model["modulation_cap"]) != MODULATION_CAP:
        raise RuntimeError("IDEA-070 modulation cap changed")
    expected_fits = len(PIPELINES) * len(BASE_SEEDS) * 3 * 4
    if expected_fits != 108 or int(model["total_fits"]) != expected_fits:
        raise RuntimeError("IDEA-070 fit budget is inconsistent")
    full_seeds = {
        base + 10_000 * repeat + 100 * fold
        for base in BASE_SEEDS
        for repeat in range(3)
        for fold in range(4)
    }
    if len(full_seeds) != 36:
        raise RuntimeError("IDEA-070 derived full seeds are not unique")
    source_protocol = read_json(idea068.PROTOCOL_PATH)
    if protocol["fixed_training"] != source_protocol["fixed_training"]:
        raise RuntimeError("IDEA-070 changed the locked IDEA-068 training recipe")
    if protocol["determinism"] != source_protocol["determinism"]:
        raise RuntimeError("IDEA-070 changed the locked IDEA-068 determinism recipe")
    dependencies = protocol["dependencies"]
    checks = {
        REPO_ROOT / dependencies["idea_path"]: dependencies["idea_sha256"],
        idea068.PROTOCOL_PATH: dependencies["idea068_protocol_sha256"],
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
            raise RuntimeError(f"IDEA-070 dependency checksum mismatch: {path}")


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
            raise RuntimeError(f"IDEA-070 resume identity mismatch for {key}")
    if any(value != 0.0 for value in fit["initial_logit_differences"].values()):
        raise RuntimeError("IDEA-070 resume initial-logit audit mismatch")
    for prefix in ("validation_animal", "validation_call"):
        path = REPO_ROOT / fit[f"{prefix}_predictions"]
        if not path.is_file() or idea068.sha256(path) != fit[f"{prefix}_sha256"]:
            raise RuntimeError(f"IDEA-070 resume prediction hash mismatch: {path}")


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
                bundles = {
                    pipeline: metric_bundle(frame) for pipeline, frame in frames.items()
                }
                row: dict[str, Any] = {
                    "base_seed": base_seed,
                    "repeat": repeat,
                    "fold": fold,
                }
                for pipeline in PIPELINES:
                    row[f"{pipeline}_macro_f1"] = bundles[pipeline]["metrics"][
                        "macro_f1"
                    ]
                    row[f"{pipeline}_cross_entropy"] = bundles[pipeline][
                        "cross_entropy"
                    ]
                    row[f"{pipeline}_brier"] = bundles[pipeline]["brier"]
                row["B1_minus_A0_macro_f1"] = (
                    row[f"{PIPELINES[2]}_macro_f1"]
                    - row[f"{PIPELINES[0]}_macro_f1"]
                )
                row["B1_minus_A1_macro_f1"] = (
                    row[f"{PIPELINES[2]}_macro_f1"]
                    - row[f"{PIPELINES[1]}_macro_f1"]
                )
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
            for pipeline in PIPELINES:
                row[f"{pipeline}_macro_f1"] = bundles[pipeline]["metrics"][
                    "macro_f1"
                ]
                row[f"{pipeline}_cross_entropy"] = bundles[pipeline][
                    "cross_entropy"
                ]
                row[f"{pipeline}_brier"] = bundles[pipeline]["brier"]
                row[f"{pipeline}_senior_recall"] = bundles[pipeline]["metrics"][
                    "per_class"
                ]["senior"]["recall"]
            row["B1_minus_A0_macro_f1"] = (
                row[f"{PIPELINES[2]}_macro_f1"]
                - row[f"{PIPELINES[0]}_macro_f1"]
            )
            row["B1_minus_A1_macro_f1"] = (
                row[f"{PIPELINES[2]}_macro_f1"]
                - row[f"{PIPELINES[1]}_macro_f1"]
            )
            seed_repeat_results.append(row)
    folds = pd.DataFrame(fold_results)
    seed_repeats = pd.DataFrame(seed_repeat_results)
    split_cells = (
        folds.groupby(["repeat", "fold"], as_index=False)
        .agg(
            B1_minus_A0_macro_f1=("B1_minus_A0_macro_f1", "mean"),
            B1_minus_A1_macro_f1=("B1_minus_A1_macro_f1", "mean"),
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
            **metric_bundle(frame),
        }
        for pipeline, frame in pooled_frames.items()
    }
    per_base_seed_mean_delta = {
        str(seed): float(
            seed_repeats.loc[
                seed_repeats["base_seed"] == seed, "B1_minus_A0_macro_f1"
            ].mean()
        )
        for seed in model["base_seeds"]
    }
    per_base_seed_senior_delta = {}
    for seed in model["base_seeds"]:
        selected = {
            pipeline: frame[frame["base_seed"] == seed]
            for pipeline, frame in pooled_frames.items()
        }
        per_base_seed_senior_delta[str(seed)] = float(
            idea068.idea051.animal_metrics(selected[PIPELINES[2]])["per_class"][
                "senior"
            ]["recall"]
            - idea068.idea051.animal_metrics(selected[PIPELINES[0]])["per_class"][
                "senior"
            ]["recall"]
        )
    gate = protocol["gate"]
    delta_a0 = seed_repeats["B1_minus_A0_macro_f1"]
    delta_a1 = seed_repeats["B1_minus_A1_macro_f1"]
    split_delta = split_cells["B1_minus_A0_macro_f1"]
    conditions = {
        "mean_B1_minus_A0_macro_f1": float(delta_a0.mean())
        >= float(gate["minimum_mean_seed_repeat_B1_minus_A0"]),
        "every_base_seed_mean_positive": all(
            value > 0.0 for value in per_base_seed_mean_delta.values()
        ),
        "positive_seed_repeats": int((delta_a0 > 0).sum())
        >= int(gate["minimum_positive_seed_repeats"]),
        "nonnegative_split_cells": int((split_delta >= 0).sum())
        >= int(gate["minimum_nonnegative_split_cells"]),
        "worst_split_cell": float(split_delta.min())
        >= float(gate["minimum_worst_split_cell_delta"]),
        "pooled_cross_entropy_vs_A0": pooled_metrics[PIPELINES[2]][
            "cross_entropy"
        ]
        <= pooled_metrics[PIPELINES[0]]["cross_entropy"],
        "pooled_brier_vs_A0": pooled_metrics[PIPELINES[2]]["brier"]
        <= pooled_metrics[PIPELINES[0]]["brier"],
        "per_base_seed_senior_recall": all(
            value >= float(gate["minimum_per_base_seed_senior_recall_delta"])
            for value in per_base_seed_senior_delta.values()
        ),
        "mean_B1_minus_A1_noninferior": float(delta_a1.mean())
        >= float(gate["minimum_mean_seed_repeat_B1_minus_A1"]),
        "pooled_cross_entropy_vs_A1": pooled_metrics[PIPELINES[2]][
            "cross_entropy"
        ]
        <= pooled_metrics[PIPELINES[1]]["cross_entropy"],
    }
    pipeline_means = {
        pipeline: float(seed_repeats[f"{pipeline}_macro_f1"].mean())
        for pipeline in PIPELINES
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
            "mean_B1_minus_A0": float(delta_a0.mean()),
            "mean_B1_minus_A1": float(delta_a1.mean()),
            "B1_minus_A0_sample_sd": float(delta_a0.std(ddof=1)),
            "B1_minus_A0_median": float(delta_a0.median()),
            "B1_minus_A0_positive": int((delta_a0 > 0).sum()),
            "B1_minus_A0_tied": int((delta_a0 == 0).sum()),
            "B1_minus_A0_negative": int((delta_a0 < 0).sum()),
            "B1_minus_A0_worst": float(delta_a0.min()),
            "B1_minus_A0_best": float(delta_a0.max()),
        },
        "fold_level": {
            "mean_B1_minus_A0": float(folds["B1_minus_A0_macro_f1"].mean()),
            "mean_B1_minus_A1": float(folds["B1_minus_A1_macro_f1"].mean()),
            "B1_minus_A0_positive": int((folds["B1_minus_A0_macro_f1"] > 0).sum()),
            "B1_minus_A0_tied": int((folds["B1_minus_A0_macro_f1"] == 0).sum()),
            "B1_minus_A0_negative": int((folds["B1_minus_A0_macro_f1"] < 0).sum()),
        },
        "split_cell_level": {
            "nonnegative": int((split_delta >= 0).sum()),
            "positive": int((split_delta > 0).sum()),
            "tied": int((split_delta == 0).sum()),
            "negative": int((split_delta < 0).sum()),
            "worst": float(split_delta.min()),
            "best": float(split_delta.max()),
        },
        "per_base_seed_mean_B1_minus_A0": per_base_seed_mean_delta,
        "per_base_seed_senior_recall_B1_minus_A0": per_base_seed_senior_delta,
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
        raise RuntimeError("IDEA-070 expected exactly 792 calls from 111 cats")
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
            raise RuntimeError("Existing IDEA-070 run manifest differs")
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
                trio = []
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
                    trio.append(fit)
                common_epochs = min(len(item["audit"]["history"]) for item in trio)
                for epoch in range(common_epochs):
                    reference = trio[0]["audit"]["history"][epoch]["train_audit"]
                    for candidate in trio[1:]:
                        current = candidate["audit"]["history"][epoch]["train_audit"]
                        if (
                            reference["cat_order_sha256"]
                            != current["cat_order_sha256"]
                            or reference["call_coverage_sha256"]
                            != current["call_coverage_sha256"]
                        ):
                            raise RuntimeError("IDEA-070 paired batch order differs")
    summary = aggregate(completed, protocol)
    write_json(run_root / "initial_evaluation_summary.json", summary)
    write_json(
        run_root / "run_summary.json",
        {
            "status": "complete",
            "completed_fits": len(completed),
            "expected_fits": int(protocol["model"]["total_fits"]),
            "gate_passed": summary["gate_passed"],
        },
    )
    return summary


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

