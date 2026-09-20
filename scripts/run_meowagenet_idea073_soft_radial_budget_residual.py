"""Run IDEA-073 soft radial-budget wide age-residual screening."""

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
import run_meowagenet_idea071_bounded_dual_path_fusion as idea071  # noqa: E402
import run_meowagenet_idea072_C1_seed_confirmation as idea072  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea073_soft_radial_budget_residual_v1.json"
)
PIPELINES = (
    "A0_ast_only",
    "U1_wide_unbounded_additive",
    "C1_bounded_wide_additive",
    "S1_soft_radial_budget",
)
BASE_SEEDS = (4025, 5118, 9821)
CAP = 0.25
NORM_EPSILON = 1.0e-8
HIDDEN_UNITS = 60
HIDDEN_DIMENSION = 128
EXPECTED_PARAMETERS = {
    "A0_ast_only": 99_075,
    "U1_wide_unbounded_additive": 108_143,
    "C1_bounded_wide_additive": 108_143,
    "S1_soft_radial_budget": 108_143,
}
_IDEA068_BUILD_MODEL = idea068.build_model
_IDEA068_PREDICT = idea068.predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-subdir", default="meowagenet_idea073_soft_radial_budget_residual_v1"
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


class SoftRadialBudgetClassifier(
    idea071.PerturbationAuditMixin, idea068.AgeResidualClassifier
):
    """Direction-preserving residual with a smooth AST-relative L2 budget."""

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
            age_hidden_units=HIDDEN_UNITS,
        )
        self.pipeline = "S1_soft_radial_budget"
        self.reset_perturbation_audit()

    def reset_perturbation_audit(self) -> None:
        super().reset_perturbation_audit()
        self._raw_to_budget: list[float] = []
        self._residual_to_budget: list[float] = []
        self._direction_cosines: list[float] = []
        self._budget_violations: list[float] = []

    def _record_radial_audit(
        self,
        raw: torch.Tensor,
        residual: torch.Tensor,
        budget: torch.Tensor,
    ) -> None:
        if self.training:
            return
        with torch.no_grad():
            raw32 = raw.float()
            residual32 = residual.float()
            raw_norm = torch.linalg.vector_norm(raw32, dim=1)
            residual_norm = torch.linalg.vector_norm(residual32, dim=1)
            budget_flat = budget.float().reshape(-1).clamp_min(1.0e-12)
            self._raw_to_budget.extend(
                (raw_norm / budget_flat).detach().cpu().tolist()
            )
            self._residual_to_budget.extend(
                (residual_norm / budget_flat).detach().cpu().tolist()
            )
            denominator = (raw_norm * residual_norm).clamp_min(1.0e-12)
            cosine = (raw32 * residual32).sum(dim=1) / denominator
            valid = raw_norm > 1.0e-12
            self._direction_cosines.extend(
                cosine[valid].detach().cpu().tolist()
            )
            self._budget_violations.extend(
                torch.clamp(residual_norm - budget_flat, min=0.0)
                .detach()
                .cpu()
                .tolist()
            )

    @staticmethod
    def _summary(values: list[float], prefix: str) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        if len(array) == 0:
            return {
                f"{prefix}_calls": 0,
                f"{prefix}_mean": None,
                f"{prefix}_min": None,
                f"{prefix}_max": None,
            }
        return {
            f"{prefix}_calls": int(len(array)),
            f"{prefix}_mean": float(array.mean()),
            f"{prefix}_min": float(array.min()),
            f"{prefix}_max": float(array.max()),
        }

    def radial_audit(self) -> dict[str, Any]:
        residual_ratios = np.asarray(self._residual_to_budget, dtype=np.float64)
        violations = np.asarray(self._budget_violations, dtype=np.float64)
        return {
            **self._summary(self._raw_to_budget, "validation_raw_to_budget"),
            **self._summary(
                self._residual_to_budget, "validation_residual_to_budget"
            ),
            **self._summary(
                self._direction_cosines, "validation_direction_cosine"
            ),
            "validation_near_saturation_fraction": (
                float(np.mean(residual_ratios >= 0.9))
                if len(residual_ratios)
                else None
            ),
            "validation_budget_violation_calls": (
                int(np.sum(violations > 1.0e-6)) if len(violations) else 0
            ),
            "validation_max_budget_violation": (
                float(violations.max()) if len(violations) else None
            ),
        }

    def forward(
        self, ast_embeddings: torch.Tensor, age_features: torch.Tensor
    ) -> torch.Tensor:
        ast = (ast_embeddings - self.ast_mean) / self.ast_scale
        hidden = self.relu(self.ast_linear(ast))
        if self.age_hidden is None or self.age_output is None:
            raise RuntimeError("IDEA-073 S1 age branch is missing")
        context = torch.nn.functional.gelu(
            self.age_hidden(idea071.standardized_age(self, age_features))
        )
        raw = self.age_output(context)
        # Keep norms and epsilon in FP32 even when the surrounding fit uses AMP.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden32 = hidden.float()
            raw32 = raw.float()
            budget = (
                CAP
                * torch.sqrt(
                    hidden32.square().sum(dim=1, keepdim=True)
                    + HIDDEN_DIMENSION * NORM_EPSILON
                ).detach()
            )
            raw_norm = torch.sqrt(
                raw32.square().sum(dim=1, keepdim=True) + NORM_EPSILON
            )
            residual32 = raw32 * budget / (budget + raw_norm)
        # Casting a nearly saturated vector to FP16 can round its L2 norm a few
        # ulps above the FP32 budget.  A one-epsilon downward guard preserves
        # the theoretical inequality in the tensor that is actually added.
        dtype_guard = 1.0 - torch.finfo(hidden.dtype).eps
        residual = (residual32 * dtype_guard).to(hidden.dtype)
        self.record_perturbation(hidden, residual)
        self._record_radial_audit(raw, residual, budget)
        hidden = hidden + residual
        hidden = self.batch_norm(hidden)
        hidden = self.dropout(hidden)
        return self.output(hidden)

    def audit(self) -> dict[str, Any]:
        result = super().audit()
        result.update(
            {
                "fusion": "soft_radial_AST_relative_budget",
                "cap": CAP,
                "norm_epsilon": NORM_EPSILON,
                "budget_dimension": HIDDEN_DIMENSION,
                **self.perturbation_audit(),
                **self.radial_audit(),
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
        return idea072.WideUnboundedAdditiveClassifier(**common)
    if pipeline == PIPELINES[2]:
        return idea071.BoundedWideAdditiveClassifier(**common)
    if pipeline == PIPELINES[3]:
        return SoftRadialBudgetClassifier(**common)
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
    logits: dict[str, np.ndarray] = {}
    parameters: dict[str, int] = {}
    states: dict[str, dict[str, torch.Tensor]] = {}
    for pipeline in PIPELINES:
        idea068.idea051.reference.historical.set_seed(seed)
        model = build_model(pipeline, protocol, store, features, train_indices).eval()
        parameters[pipeline] = sum(parameter.numel() for parameter in model.parameters())
        if pipeline != PIPELINES[0]:
            states[pipeline] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        with torch.no_grad():
            logits[pipeline] = model(
                torch.from_numpy(store.frozen_embeddings[probe_indices]),
                torch.from_numpy(features[probe_indices]),
            ).cpu().numpy()
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"IDEA-073 parameter audit mismatch: {parameters}")
    reference_state = states[PIPELINES[1]]
    for pipeline in PIPELINES[2:]:
        if reference_state.keys() != states[pipeline].keys() or not all(
            torch.equal(reference_state[key], states[pipeline][key])
            for key in reference_state
        ):
            raise RuntimeError("IDEA-073 wide branches differ at initialization")
    differences = {
        pipeline: float(np.max(np.abs(logits[pipeline] - logits[PIPELINES[0]])))
        for pipeline in PIPELINES[1:]
    }
    if any(value != 0.0 for value in differences.values()):
        raise RuntimeError("IDEA-073 pipelines are not identical at initialization")
    return differences


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea073-soft-radial-budget-v1":
        raise RuntimeError("Unexpected IDEA-073 protocol")
    if protocol.get("status") != "locked_before_initial_evaluation":
        raise RuntimeError("IDEA-073 protocol is not locked")
    model = protocol["model"]
    if tuple(model["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-073 pipeline matrix changed")
    if tuple(model["base_seeds"]) != BASE_SEEDS:
        raise RuntimeError("IDEA-073 base-seed bank changed")
    if model["repeats"] != [0, 1, 2] or model["folds"] != [0, 1, 2, 3]:
        raise RuntimeError("IDEA-073 split scope changed")
    if model.get("outer_test_predictions") is not False:
        raise RuntimeError("IDEA-073 must not access outer-test predictions")
    if float(model["cap"]) != CAP or float(model["norm_epsilon"]) != NORM_EPSILON:
        raise RuntimeError("IDEA-073 radial budget changed")
    if int(model["wide_hidden_units"]) != HIDDEN_UNITS:
        raise RuntimeError("IDEA-073 width changed")
    if int(model["hidden_dimension"]) != HIDDEN_DIMENSION:
        raise RuntimeError("IDEA-073 hidden dimension changed")
    if model["trainable_parameters"] != EXPECTED_PARAMETERS:
        raise RuntimeError("IDEA-073 parameter lock changed")
    expected_fits = len(PIPELINES) * len(BASE_SEEDS) * 3 * 4
    if expected_fits != 144 or int(model["total_fits"]) != expected_fits:
        raise RuntimeError("IDEA-073 fit budget is inconsistent")
    full_seeds = {
        base + 10_000 * repeat + 100 * fold
        for base in BASE_SEEDS
        for repeat in range(3)
        for fold in range(4)
    }
    if len(full_seeds) != 36:
        raise RuntimeError("IDEA-073 derived full seeds are not unique")
    source_protocol = read_json(idea068.PROTOCOL_PATH)
    if protocol["fixed_training"] != source_protocol["fixed_training"]:
        raise RuntimeError("IDEA-073 changed the locked IDEA-068 training recipe")
    if protocol["determinism"] != source_protocol["determinism"]:
        raise RuntimeError("IDEA-073 changed the locked IDEA-068 determinism recipe")
    dependencies = protocol["dependencies"]
    checks = {
        REPO_ROOT / dependencies["idea_path"]: dependencies["idea_sha256"],
        idea068.PROTOCOL_PATH: dependencies["idea068_protocol_sha256"],
        Path(idea068.__file__).resolve(): dependencies["idea068_runner_sha256"],
        idea071.PROTOCOL_PATH: dependencies["idea071_protocol_sha256"],
        Path(idea071.__file__).resolve(): dependencies["idea071_runner_sha256"],
        idea072.PROTOCOL_PATH: dependencies["idea072_protocol_sha256"],
        Path(idea072.__file__).resolve(): dependencies["idea072_runner_sha256"],
        REPO_ROOT / protocol["data"]["roles_path"]: protocol["data"]["roles_sha256"],
        REPO_ROOT / protocol["data"]["frozen_embedding_path"]: protocol["data"]["frozen_embedding_sha256"],
        REPO_ROOT / protocol["data"]["fbank_path"]: protocol["data"]["fbank_sha256"],
        REPO_ROOT / protocol["data"]["feature_path"]: protocol["data"]["feature_sha256"],
        REPO_ROOT / protocol["data"]["feature_summary_path"]: protocol["data"]["feature_summary_sha256"],
        Path(__file__).resolve(): dependencies["runner_sha256"],
    }
    for path, expected_sha in checks.items():
        if not path.is_file() or idea068.sha256(path) != expected_sha:
            raise RuntimeError(f"IDEA-073 dependency checksum mismatch: {path}")


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
            raise RuntimeError(f"IDEA-073 resume identity mismatch for {key}")
    if any(value != 0.0 for value in fit["initial_logit_differences"].values()):
        raise RuntimeError("IDEA-073 resume initial-logit audit mismatch")
    for prefix in ("validation_animal", "validation_call"):
        path = REPO_ROOT / fit[f"{prefix}_predictions"]
        if not path.is_file() or idea068.sha256(path) != fit[f"{prefix}_sha256"]:
            raise RuntimeError(f"IDEA-073 resume prediction hash mismatch: {path}")
    if pipeline == PIPELINES[3]:
        audit = fit["audit"]["model"]
        if audit["validation_budget_violation_calls"] != 0:
            raise RuntimeError("IDEA-073 S1 exceeded its radial budget")


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


COMPARISONS = (
    ("S1_minus_A0", PIPELINES[3], PIPELINES[0]),
    ("S1_minus_U1", PIPELINES[3], PIPELINES[1]),
    ("S1_minus_C1", PIPELINES[3], PIPELINES[2]),
    ("U1_minus_A0", PIPELINES[1], PIPELINES[0]),
    ("C1_minus_A0", PIPELINES[2], PIPELINES[0]),
    ("C1_minus_U1", PIPELINES[2], PIPELINES[1]),
)


def add_pipeline_metrics(row: dict[str, Any], bundles: dict[str, Any]) -> None:
    for pipeline in PIPELINES:
        row[f"{pipeline}_macro_f1"] = bundles[pipeline]["metrics"]["macro_f1"]
        row[f"{pipeline}_cross_entropy"] = bundles[pipeline]["cross_entropy"]
        row[f"{pipeline}_brier"] = bundles[pipeline]["brier"]
        row[f"{pipeline}_senior_recall"] = bundles[pipeline]["metrics"][
            "per_class"
        ]["senior"]["recall"]
    for name, left, right in COMPARISONS:
        row[f"{name}_macro_f1"] = (
            row[f"{left}_macro_f1"] - row[f"{right}_macro_f1"]
        )


def aggregate(fits: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    model = protocol["model"]
    by_key = {
        (fit["pipeline"], fit["base_seed"], fit["repeat"], fit["fold"]): fit
        for fit in fits
    }
    fold_results: list[dict[str, Any]] = []
    seed_repeat_results: list[dict[str, Any]] = []
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
    comparison_columns = [f"{name}_macro_f1" for name, _, _ in COMPARISONS]
    split_cells = (
        folds.groupby(["repeat", "fold"], as_index=False)[comparison_columns]
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
    target_columns = comparison_columns[:3]
    per_seed_deltas: dict[str, dict[str, float]] = {}
    per_seed_senior: dict[str, float] = {}
    for seed in model["base_seeds"]:
        selected_rows = seed_repeats[seed_repeats["base_seed"] == seed]
        per_seed_deltas[str(seed)] = {
            name: float(selected_rows[name].mean()) for name in target_columns
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
        per_seed_senior[str(seed)] = float(
            senior_recalls[PIPELINES[3]] - senior_recalls[PIPELINES[0]]
        )

    delta_a0 = seed_repeats["S1_minus_A0_macro_f1"]
    delta_u1 = seed_repeats["S1_minus_U1_macro_f1"]
    delta_c1 = seed_repeats["S1_minus_C1_macro_f1"]
    split_delta = split_cells["S1_minus_A0_macro_f1"]
    mean_cross_entropy = {
        pipeline: float(seed_repeats[f"{pipeline}_cross_entropy"].mean())
        for pipeline in PIPELINES
    }
    mean_brier = {
        pipeline: float(seed_repeats[f"{pipeline}_brier"].mean())
        for pipeline in PIPELINES
    }
    gate = protocol["gate"]
    conditions = {
        "mean_S1_minus_A0_macro_f1": float(delta_a0.mean())
        >= float(gate["minimum_mean_seed_repeat_S1_minus_A0"]),
        "every_base_seed_mean_S1_minus_A0_positive": all(
            values["S1_minus_A0_macro_f1"] > 0.0
            for values in per_seed_deltas.values()
        ),
        "positive_S1_minus_A0_seed_repeats": int((delta_a0 > 0).sum())
        >= int(gate["minimum_positive_seed_repeats_S1_minus_A0"]),
        "nonnegative_S1_minus_A0_split_cells": int((split_delta >= 0).sum())
        >= int(gate["minimum_nonnegative_split_cells_S1_minus_A0"]),
        "worst_S1_minus_A0_split_cell": float(split_delta.min())
        >= float(gate["minimum_worst_split_cell_delta_S1_minus_A0"]),
        "mean_cross_entropy_vs_A0": mean_cross_entropy[PIPELINES[3]]
        <= mean_cross_entropy[PIPELINES[0]],
        "mean_brier_vs_A0": mean_brier[PIPELINES[3]]
        <= mean_brier[PIPELINES[0]],
        "per_base_seed_senior_recall_vs_A0": all(
            value >= float(gate["minimum_per_base_seed_senior_recall_delta_vs_A0"])
            for value in per_seed_senior.values()
        ),
        "mean_S1_minus_U1_noninferior": float(delta_u1.mean())
        >= float(gate["minimum_mean_seed_repeat_S1_minus_U1"]),
        "nonnegative_S1_minus_U1_seed_repeats": int((delta_u1 >= 0).sum())
        >= int(gate["minimum_nonnegative_seed_repeats_S1_minus_U1"]),
        "mean_cross_entropy_vs_U1": mean_cross_entropy[PIPELINES[3]]
        <= mean_cross_entropy[PIPELINES[1]],
        "mean_brier_vs_U1": mean_brier[PIPELINES[3]]
        <= mean_brier[PIPELINES[1]],
        "mean_S1_minus_C1_positive": float(delta_c1.mean()) > 0.0,
        "base_seed_S1_minus_C1_nonnegative": sum(
            values["S1_minus_C1_macro_f1"] >= 0.0
            for values in per_seed_deltas.values()
        )
        >= int(gate["minimum_nonnegative_base_seeds_S1_minus_C1"]),
        "nonnegative_S1_minus_C1_seed_repeats": int((delta_c1 >= 0).sum())
        >= int(gate["minimum_nonnegative_seed_repeats_S1_minus_C1"]),
        "mean_cross_entropy_vs_C1_tolerance": mean_cross_entropy[PIPELINES[3]]
        <= mean_cross_entropy[PIPELINES[2]]
        + float(gate["maximum_mean_cross_entropy_increase_vs_C1"]),
        "mean_brier_vs_C1_tolerance": mean_brier[PIPELINES[3]]
        <= mean_brier[PIPELINES[2]]
        + float(gate["maximum_mean_brier_increase_vs_C1"]),
    }
    basic_names = (
        "mean_S1_minus_A0_macro_f1",
        "every_base_seed_mean_S1_minus_A0_positive",
        "positive_S1_minus_A0_seed_repeats",
        "nonnegative_S1_minus_A0_split_cells",
        "worst_S1_minus_A0_split_cell",
        "mean_cross_entropy_vs_A0",
        "mean_brier_vs_A0",
        "per_base_seed_senior_recall_vs_A0",
    )
    u1_names = (
        "mean_S1_minus_U1_noninferior",
        "nonnegative_S1_minus_U1_seed_repeats",
        "mean_cross_entropy_vs_U1",
        "mean_brier_vs_U1",
    )
    c1_names = (
        "mean_S1_minus_C1_positive",
        "base_seed_S1_minus_C1_nonnegative",
        "nonnegative_S1_minus_C1_seed_repeats",
        "mean_cross_entropy_vs_C1_tolerance",
        "mean_brier_vs_C1_tolerance",
    )
    pipeline_means = {
        pipeline: float(seed_repeats[f"{pipeline}_macro_f1"].mean())
        for pipeline in PIPELINES
    }
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
            "pipeline_mean_cross_entropy": mean_cross_entropy,
            "pipeline_mean_brier": mean_brier,
            "comparisons": comparisons,
        },
        "split_cell_level": {
            "S1_minus_A0_nonnegative": int((split_delta >= 0).sum()),
            "S1_minus_A0_positive": int((split_delta > 0).sum()),
            "S1_minus_A0_tied": int((split_delta == 0).sum()),
            "S1_minus_A0_negative": int((split_delta < 0).sum()),
            "S1_minus_A0_worst": float(split_delta.min()),
            "S1_minus_A0_best": float(split_delta.max()),
        },
        "per_base_seed_mean_deltas": per_seed_deltas,
        "per_base_seed_senior_recall_deltas": per_seed_senior,
        "pooled_validation": pooled_metrics,
        "gate_conditions": conditions,
        "basic_effectiveness_passed": bool(
            all(conditions[name] for name in basic_names)
        ),
        "U1_preservation_passed": bool(all(conditions[name] for name in u1_names)),
        "C1_calibration_passed": bool(all(conditions[name] for name in c1_names)),
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
        raise RuntimeError("IDEA-073 expected exactly 792 calls from 111 cats")
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
            raise RuntimeError("Existing IDEA-073 run manifest differs")
    else:
        write_json(manifest_path, manifest)
    completed: list[dict[str, Any]] = []
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
                quartet: list[dict[str, Any]] = []
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
                        if (
                            audit["model"]["trainable_parameters"]
                            != EXPECTED_PARAMETERS[pipeline]
                        ):
                            raise RuntimeError("IDEA-073 trained parameter audit mismatch")
                        if pipeline == PIPELINES[3]:
                            if audit["model"]["validation_budget_violation_calls"] != 0:
                                raise RuntimeError("IDEA-073 S1 exceeded its radial budget")
                            if (
                                audit["model"]["validation_raw_to_budget_calls"]
                                != len(indices["validation"])
                            ):
                                raise RuntimeError("IDEA-073 S1 audit call count mismatch")
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
                    quartet.append(fit)
                common_epochs = min(len(item["audit"]["history"]) for item in quartet)
                for epoch in range(common_epochs):
                    reference = quartet[0]["audit"]["history"][epoch]["train_audit"]
                    for candidate in quartet[1:]:
                        current = candidate["audit"]["history"][epoch]["train_audit"]
                        if (
                            reference["cat_order_sha256"] != current["cat_order_sha256"]
                            or reference["call_coverage_sha256"]
                            != current["call_coverage_sha256"]
                        ):
                            raise RuntimeError("IDEA-073 paired batch order differs")
    summary = aggregate(completed, protocol)
    write_json(run_root / "initial_evaluation_summary.json", summary)
    write_json(
        run_root / "run_summary.json",
        {
            "status": "complete",
            "completed_fits": len(completed),
            "expected_fits": int(protocol["model"]["total_fits"]),
            "basic_effectiveness_passed": summary["basic_effectiveness_passed"],
            "U1_preservation_passed": summary["U1_preservation_passed"],
            "C1_calibration_passed": summary["C1_calibration_passed"],
            "gate_passed": summary["gate_passed"],
        },
    )
    return summary


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
