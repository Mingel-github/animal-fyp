"""Run IDEA-084 grouped dual-branch acoustic residual experiment."""

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
import run_meowagenet_idea071_bounded_dual_path_fusion as idea071  # noqa: E402
import run_meowagenet_idea082_age_acoustic_group_ablation as idea082  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea084_grouped_dual_branch_acoustic_residual_v1.json"
)
PIPELINES = (
    "A0_ast_only",
    "C1_bounded_wide_additive",
    "P1_grouped_dual_branch",
    "R1_hash_random_dual_branch",
)
P1_GROUPS = (tuple(range(15)), tuple(range(15, 20)))
R1_GROUPS = (
    (0, 1, 2, 4, 5, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19),
    (3, 6, 7, 12, 16),
)
BASE_SEEDS = (3583, 5080, 9355)
BRANCH_WEIGHTS = (0.5, 0.5)
BRANCH_HIDDEN_UNITS = 32
EXPECTED_PARAMETERS = {
    "A0_ast_only": 99_075,
    "C1_bounded_wide_additive": 108_143,
    "P1_grouped_dual_branch": 108_227,
    "R1_hash_random_dual_branch": 108_227,
}
COMPARISONS = {
    "P1_minus_C1": ("P1_grouped_dual_branch", "C1_bounded_wide_additive"),
    "P1_minus_A0": ("P1_grouped_dual_branch", "A0_ast_only"),
    "P1_minus_R1": (
        "P1_grouped_dual_branch",
        "R1_hash_random_dual_branch",
    ),
}
DESCRIPTIVE_COMPARISONS = {
    "C1_minus_A0_descriptive": (
        "C1_bounded_wide_additive",
        "A0_ast_only",
    )
}
_IDEA068_BUILD_MODEL = idea068.build_model
_IDEA068_PREDICT = idea068.predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "run"), required=True)
    parser.add_argument(
        "--output-subdir",
        default="meowagenet_idea084_grouped_dual_branch_acoustic_residual_v1",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--director-authorized", action="store_true")
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Stop after this many complete 4-pipeline cells without aggregating; used for the first-cell audit.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def fixed_random_groups(material: str) -> tuple[tuple[int, ...], tuple[int, ...], list[int]]:
    ranking = sorted(
        range(20),
        key=lambda index: hashlib.sha256(
            f"{material}|feature_index={index}".encode("utf-8")
        ).digest(),
    )
    return tuple(sorted(ranking[:15])), tuple(sorted(ranking[15:])), ranking


def group_indices_for_pipeline(pipeline: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if pipeline == PIPELINES[2]:
        return P1_GROUPS
    if pipeline == PIPELINES[3]:
        return R1_GROUPS
    raise ValueError(f"Pipeline has no dual groups: {pipeline}")


class GroupedDualBranchClassifier(idea071.PerturbationAuditMixin, torch.nn.Module):
    """Two fixed acoustic branches sharing one total RMS-relative residual budget."""

    def __init__(
        self,
        pipeline: str,
        ast_mean: np.ndarray,
        ast_scale: np.ndarray,
        age_train: np.ndarray,
        dropout: float,
        group_a_indices: tuple[int, ...],
        group_b_indices: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.group_a_indices = tuple(group_a_indices)
        self.group_b_indices = tuple(group_b_indices)
        safe_ast_scale = np.where(ast_scale > 1.0e-12, ast_scale, 1.0).astype(
            np.float32
        )
        self.register_buffer("ast_mean", torch.from_numpy(ast_mean.astype(np.float32)))
        self.register_buffer("ast_scale", torch.from_numpy(safe_ast_scale))
        # Keep the common AST module creation order identical to IDEA-068/C1.
        self.ast_linear = torch.nn.Linear(768, 128)
        self.relu = torch.nn.ReLU()
        self.batch_norm = torch.nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01)
        self.dropout = torch.nn.Dropout(dropout)
        self.output = torch.nn.Linear(128, 3)

        a_median, a_mean, a_scale = idea082.stable_age_statistics(
            age_train[:, self.group_a_indices]
        )
        b_median, b_mean, b_scale = idea082.stable_age_statistics(
            age_train[:, self.group_b_indices]
        )
        self.register_buffer("group_a_median", torch.from_numpy(a_median))
        self.register_buffer("group_a_mean", torch.from_numpy(a_mean))
        self.register_buffer("group_a_scale", torch.from_numpy(a_scale))
        self.register_buffer("group_b_median", torch.from_numpy(b_median))
        self.register_buffer("group_b_mean", torch.from_numpy(b_mean))
        self.register_buffer("group_b_scale", torch.from_numpy(b_scale))

        self.group_a_hidden = torch.nn.Linear(len(self.group_a_indices), BRANCH_HIDDEN_UNITS)
        self.group_a_output = torch.nn.Linear(BRANCH_HIDDEN_UNITS, 128)
        self.group_b_hidden = torch.nn.Linear(len(self.group_b_indices), BRANCH_HIDDEN_UNITS)
        self.group_b_output = torch.nn.Linear(BRANCH_HIDDEN_UNITS, 128)
        torch.nn.init.zeros_(self.group_a_output.weight)
        torch.nn.init.zeros_(self.group_a_output.bias)
        torch.nn.init.zeros_(self.group_b_output.weight)
        torch.nn.init.zeros_(self.group_b_output.bias)
        self.reset_perturbation_audit()

    @staticmethod
    def _standardize(
        values: torch.Tensor,
        median: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        imputed = torch.where(torch.isfinite(values), values, median)
        return (imputed - mean) / scale

    def branch_inputs(self, age_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = age_features[:, self.group_a_indices]
        b = age_features[:, self.group_b_indices]
        return (
            self._standardize(a, self.group_a_median, self.group_a_mean, self.group_a_scale),
            self._standardize(b, self.group_b_median, self.group_b_mean, self.group_b_scale),
        )

    def branch_outputs(self, age_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = self.branch_inputs(age_features)
        a_context = torch.nn.functional.gelu(self.group_a_hidden(a))
        b_context = torch.nn.functional.gelu(self.group_b_hidden(b))
        return self.group_a_output(a_context), self.group_b_output(b_context)

    def forward(
        self, ast_embeddings: torch.Tensor, age_features: torch.Tensor
    ) -> torch.Tensor:
        ast = (ast_embeddings - self.ast_mean) / self.ast_scale
        hidden = self.relu(self.ast_linear(ast))
        a_output, b_output = self.branch_outputs(age_features)
        bounded_mix = BRANCH_WEIGHTS[0] * torch.tanh(a_output) + BRANCH_WEIGHTS[
            1
        ] * torch.tanh(b_output)
        residual = idea071.CAP * idea071.hidden_rms(hidden) * bounded_mix
        self.record_perturbation(hidden, residual)
        hidden = hidden + residual
        hidden = self.batch_norm(hidden)
        hidden = self.dropout(hidden)
        return self.output(hidden)

    def audit(self) -> dict[str, Any]:
        return {
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in self.parameters())
            ),
            "fusion": "equal_weight_grouped_dual_branch_bounded_rms_relative_additive",
            "cap": idea071.CAP,
            "rms_epsilon": idea071.RMS_EPSILON,
            "branch_weights": list(BRANCH_WEIGHTS),
            "branch_hidden_units": BRANCH_HIDDEN_UNITS,
            "group_a_indices": list(self.group_a_indices),
            "group_b_indices": list(self.group_b_indices),
            **self.perturbation_audit(),
        }


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
        return idea071.BoundedWideAdditiveClassifier(**common)
    if pipeline in PIPELINES[2:]:
        group_a, group_b = group_indices_for_pipeline(pipeline)
        return GroupedDualBranchClassifier(
            pipeline=pipeline,
            group_a_indices=group_a,
            group_b_indices=group_b,
            **common,
        )
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


def load_features(
    protocol: dict[str, Any], call_ids: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    return idea082.load_features(protocol, call_ids)


def resolve_run_root(output_subdir: str) -> Path:
    return idea082.resolve_run_root(output_subdir)


def role_cell_indices(
    store: Any, roles: pd.DataFrame, repeat: int, fold: int
) -> dict[str, np.ndarray]:
    return idea082.role_cell_indices(store, roles, repeat, fold)


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea084-grouped-dual-branch-acoustic-residual-v1":
        raise RuntimeError("Unexpected IDEA-084 protocol")
    if protocol.get("status") != "locked_for_cpu_preflight_before_initial_evaluation":
        raise RuntimeError("IDEA-084 protocol is not result-blind locked")
    model = protocol["model"]
    if tuple(model["pipelines"]) != PIPELINES or tuple(model["base_seeds"]) != BASE_SEEDS:
        raise RuntimeError("IDEA-084 pipeline or seed lock changed")
    if model["repeats"] != [0, 1, 2] or model["folds"] != [0, 1, 2, 3]:
        raise RuntimeError("IDEA-084 split scope changed")
    if int(model["fits_per_pipeline"]) != 36 or int(model["total_fits"]) != 144:
        raise RuntimeError("IDEA-084 fit budget changed")
    if float(model["cap"]) != idea071.CAP or float(model["rms_epsilon"]) != idea071.RMS_EPSILON:
        raise RuntimeError("IDEA-084 C1 bound changed")
    if tuple(float(value) for value in model["branch_weights"]) != BRANCH_WEIGHTS:
        raise RuntimeError("IDEA-084 branch weights changed")
    if int(model["dual_branch_hidden_units_each"]) != BRANCH_HIDDEN_UNITS:
        raise RuntimeError("IDEA-084 branch width changed")
    if int(model["dual_branch_age_parameters"]) != 9_152:
        raise RuntimeError("IDEA-084 dual branch parameter budget changed")
    if int(model["c1_age_branch_parameters"]) != 9_068:
        raise RuntimeError("IDEA-084 C1 parameter budget changed")
    expected_from_protocol = {
        PIPELINES[0]: int(model["a0_trainable_parameters"]),
        PIPELINES[1]: int(model["c1_total_trainable_parameters"]),
        PIPELINES[2]: int(model["dual_branch_total_trainable_parameters"]),
        PIPELINES[3]: int(model["dual_branch_total_trainable_parameters"]),
    }
    if expected_from_protocol != EXPECTED_PARAMETERS:
        raise RuntimeError("IDEA-084 trainable parameter lock changed")

    groups = protocol["feature_groups"]
    p1 = (
        tuple(groups["P1_source_related"]["indices"]),
        tuple(groups["P1_spectral_energy"]["indices"]),
    )
    r1 = (
        tuple(groups["R1_hash_group_A"]["indices"]),
        tuple(groups["R1_hash_group_B"]["indices"]),
    )
    if p1 != P1_GROUPS or r1 != R1_GROUPS:
        raise RuntimeError("IDEA-084 group membership changed")
    for pair in (p1, r1):
        if set(pair[0]) & set(pair[1]) or set(pair[0]) | set(pair[1]) != set(range(20)):
            raise RuntimeError("IDEA-084 group partition is not exclusive/exhaustive")
        if tuple(map(len, pair)) != (15, 5):
            raise RuntimeError("IDEA-084 group dimensions changed")
    if p1 == r1 or p1 == (r1[1], r1[0]):
        raise RuntimeError("IDEA-084 random grouping equals the proposed grouping")
    for group in r1:
        if not (set(group) & set(range(15))) or not (set(group) & set(range(15, 20))):
            raise RuntimeError("IDEA-084 random group does not mix both acoustic families")
    derivation = groups["hash_derivation"]
    material = derivation["material"]
    if hashlib.sha256(material.encode("utf-8")).hexdigest() != derivation["material_sha256"]:
        raise RuntimeError("IDEA-084 random grouping material digest changed")
    generated_a, generated_b, ranking = fixed_random_groups(material)
    if (generated_a, generated_b) != R1_GROUPS or ranking != derivation["ranked_indices"]:
        raise RuntimeError("IDEA-084 random grouping cannot be reconstructed")

    seed_digest = hashlib.sha256(model["seed_derivation_text"].encode("utf-8")).hexdigest()
    if seed_digest != model["seed_derivation_sha256"]:
        raise RuntimeError("IDEA-084 seed material digest changed")
    candidates = [
        int.from_bytes(bytes.fromhex(seed_digest)[offset : offset + 4], "big") % 10_000
        for offset in range(0, 32, 4)
    ]
    if candidates != model["candidate_sequence"] or tuple(candidates[:3]) != BASE_SEEDS:
        raise RuntimeError("IDEA-084 seed derivation changed")
    current_full = {
        base + 10_000 * repeat + 100 * fold
        for base in BASE_SEEDS
        for repeat in range(3)
        for fold in range(4)
    }
    if len(current_full) != 36:
        raise RuntimeError("IDEA-084 full seeds are not unique")
    prior_full = {
        int(base) + 10_000 * repeat + 100 * fold
        for base in model["known_prior_base_seeds"]
        for repeat in range(3)
        for fold in range(5)
    }
    if current_full & prior_full:
        raise RuntimeError("IDEA-084 full seed collides with a prior experiment")

    source_protocol = read_json(idea068.PROTOCOL_PATH)
    shared_training_keys = (
        "ast_head",
        "age_hidden_units",
        "dropout",
        "optimizer",
        "learning_rate",
        "optimizer_epsilon",
        "gradient_clip",
        "maximum_epochs",
        "early_stopping_patience",
        "cat_batch_size",
        "loss",
        "checkpoint_selection",
        "post_build_seed_offset",
    )
    for key in shared_training_keys:
        if protocol["fixed_training"][key] != source_protocol["fixed_training"][key]:
            raise RuntimeError(f"IDEA-084 changed locked training field: {key}")
    if protocol["determinism"] != source_protocol["determinism"]:
        raise RuntimeError("IDEA-084 changed locked determinism")

    dependencies = protocol["dependencies"]
    checks = {
        idea068.PROTOCOL_PATH: dependencies["idea068_protocol_sha256"],
        Path(idea068.__file__).resolve(): dependencies["idea068_runner_sha256"],
        idea071.PROTOCOL_PATH: dependencies["idea071_protocol_sha256"],
        Path(idea071.__file__).resolve(): dependencies["idea071_runner_sha256"],
        REPO_ROOT / "configs/protocol/meowagenet_idea076_C1_final_seed_confirmation_v1.json": dependencies["idea076_protocol_sha256"],
        REPO_ROOT / "scripts/run_meowagenet_idea076_C1_final_seed_confirmation.py": dependencies["idea076_runner_sha256"],
        REPO_ROOT / "metadata/experiments/meowagenet_idea076_C1_final_seed_confirmation_v1_results.json": dependencies["idea076_results_sha256"],
        idea082.PROTOCOL_PATH: dependencies["idea082_protocol_sha256"],
        Path(idea082.__file__).resolve(): dependencies["idea082_runner_sha256"],
        REPO_ROOT / "metadata/experiments/meowagenet_idea082_age_acoustic_group_ablation_v1_results.json": dependencies["idea082_results_sha256"],
        REPO_ROOT / protocol["data"]["roles_path"]: protocol["data"]["roles_sha256"],
        REPO_ROOT / protocol["data"]["frozen_embedding_path"]: protocol["data"]["frozen_embedding_sha256"],
        REPO_ROOT / protocol["data"]["feature_path"]: protocol["data"]["feature_sha256"],
        REPO_ROOT / protocol["data"]["feature_summary_path"]: protocol["data"]["feature_summary_sha256"],
    }
    optional_checks = {
        Path(__file__).resolve(): dependencies["runner_sha256"],
        REPO_ROOT / dependencies["tests_path"]: dependencies["tests_sha256"],
    }
    checks.update(
        {
            path: expected
            for path, expected in optional_checks.items()
            if not str(expected).startswith("PENDING_")
        }
    )
    for path, expected in checks.items():
        if not path.is_file() or idea068.sha256(path) != expected:
            raise RuntimeError(f"IDEA-084 dependency checksum mismatch: {path}")


COMMON_STATE_KEYS = (
    "ast_mean",
    "ast_scale",
    "ast_linear.weight",
    "ast_linear.bias",
    "batch_norm.weight",
    "batch_norm.bias",
    "batch_norm.running_mean",
    "batch_norm.running_var",
    "batch_norm.num_batches_tracked",
    "output.weight",
    "output.bias",
)


def common_state_equal(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return all(torch.equal(left_state[key], right_state[key]) for key in COMMON_STATE_KEYS)


def trainable_state_equal(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    left_params = dict(left.named_parameters())
    right_params = dict(right.named_parameters())
    return left_params.keys() == right_params.keys() and all(
        torch.equal(left_params[key], right_params[key]) for key in left_params
    )


def initial_model_audit(
    protocol: dict[str, Any],
    store: Any,
    features: np.ndarray,
    train_indices: np.ndarray,
    probe_indices: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    models: dict[str, torch.nn.Module] = {}
    logits: dict[str, np.ndarray] = {}
    parameters: dict[str, int] = {}
    for pipeline in PIPELINES:
        idea068.idea051.reference.historical.set_seed(seed)
        model = build_model(pipeline, protocol, store, features, train_indices).eval()
        models[pipeline] = model
        parameters[pipeline] = int(sum(parameter.numel() for parameter in model.parameters()))
        with torch.no_grad():
            logits[pipeline] = model(
                torch.from_numpy(store.frozen_embeddings[probe_indices]),
                torch.from_numpy(features[probe_indices]),
            ).cpu().numpy()
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"IDEA-084 parameter audit mismatch: {parameters}")
    differences = {
        pipeline: float(np.max(np.abs(logits[pipeline] - logits[PIPELINES[0]])))
        for pipeline in PIPELINES[1:]
    }
    if any(value != 0.0 for value in differences.values()):
        raise RuntimeError("IDEA-084 pipelines differ at zero-residual initialization")
    common_equal = {
        pipeline: common_state_equal(models[PIPELINES[0]], models[pipeline])
        for pipeline in PIPELINES[1:]
    }
    if not all(common_equal.values()):
        raise RuntimeError("IDEA-084 common AST head state differs")
    p1_r1_trainable_equal = trainable_state_equal(models[PIPELINES[2]], models[PIPELINES[3]])
    if not p1_r1_trainable_equal:
        raise RuntimeError("IDEA-084 P1/R1 trainable initial states differ")
    zero_outputs = {
        PIPELINES[1]: bool(
            torch.count_nonzero(models[PIPELINES[1]].age_output.weight) == 0
            and torch.count_nonzero(models[PIPELINES[1]].age_output.bias) == 0
        ),
        PIPELINES[2]: bool(
            torch.count_nonzero(models[PIPELINES[2]].group_a_output.weight) == 0
            and torch.count_nonzero(models[PIPELINES[2]].group_a_output.bias) == 0
            and torch.count_nonzero(models[PIPELINES[2]].group_b_output.weight) == 0
            and torch.count_nonzero(models[PIPELINES[2]].group_b_output.bias) == 0
        ),
        PIPELINES[3]: bool(
            torch.count_nonzero(models[PIPELINES[3]].group_a_output.weight) == 0
            and torch.count_nonzero(models[PIPELINES[3]].group_a_output.bias) == 0
            and torch.count_nonzero(models[PIPELINES[3]].group_b_output.weight) == 0
            and torch.count_nonzero(models[PIPELINES[3]].group_b_output.bias) == 0
        ),
    }
    if not all(zero_outputs.values()):
        raise RuntimeError("IDEA-084 residual output layers are not zero initialized")
    return {
        "parameters": parameters,
        "max_logit_difference_vs_A0": differences,
        "common_AST_state_equal_to_A0": common_equal,
        "P1_R1_trainable_state_equal": p1_r1_trainable_equal,
        "zero_initialized_residual_outputs": zero_outputs,
    }


def gradient_audit(
    model: GroupedDualBranchClassifier, features: np.ndarray
) -> dict[str, float]:
    age = torch.from_numpy(features)
    model.zero_grad(set_to_none=True)
    a_output, b_output = model.branch_outputs(age)
    (a_output.sum() + b_output.sum()).backward()
    zero_a_output_grad = float(model.group_a_output.bias.grad.norm())
    zero_b_output_grad = float(model.group_b_output.bias.grad.norm())
    zero_a_hidden_grad = float(model.group_a_hidden.weight.grad.norm())
    zero_b_hidden_grad = float(model.group_b_hidden.weight.grad.norm())
    if min(zero_a_output_grad, zero_b_output_grad) <= 0.0:
        raise RuntimeError("IDEA-084 zero-output layer is not gradient reachable")
    if zero_a_hidden_grad != 0.0 or zero_b_hidden_grad != 0.0:
        raise RuntimeError("IDEA-084 hidden branch unexpectedly receives zero-init gradient")
    with torch.no_grad():
        model.group_a_output.weight.fill_(0.01)
        model.group_b_output.weight.fill_(0.01)
    model.zero_grad(set_to_none=True)
    a_output, b_output = model.branch_outputs(age)
    (a_output.sum() + b_output.sum()).backward()
    probe_a_hidden_grad = float(model.group_a_hidden.weight.grad.norm())
    probe_b_hidden_grad = float(model.group_b_hidden.weight.grad.norm())
    if min(probe_a_hidden_grad, probe_b_hidden_grad) <= 0.0:
        raise RuntimeError("IDEA-084 a hidden branch is not gradient reachable")
    return {
        "zero_init_group_a_output_bias_grad_norm": zero_a_output_grad,
        "zero_init_group_b_output_bias_grad_norm": zero_b_output_grad,
        "zero_init_group_a_hidden_weight_grad_norm": zero_a_hidden_grad,
        "zero_init_group_b_hidden_weight_grad_norm": zero_b_hidden_grad,
        "nonzero_output_probe_group_a_hidden_weight_grad_norm": probe_a_hidden_grad,
        "nonzero_output_probe_group_b_hidden_weight_grad_norm": probe_b_hidden_grad,
    }


def cap_audit(
    model: GroupedDualBranchClassifier,
    store: Any,
    features: np.ndarray,
    probe_indices: np.ndarray,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        for layer in (model.group_a_output, model.group_b_output):
            layer.weight.fill_(100.0)
            layer.bias.fill_(100.0)
    model.reset_perturbation_audit()
    with torch.no_grad():
        model(
            torch.from_numpy(store.frozen_embeddings[probe_indices]),
            torch.from_numpy(features[probe_indices]),
        )
    audit = model.perturbation_audit()
    if audit["validation_relative_perturbation_max"] > idea071.CAP + 1.0e-5:
        raise RuntimeError("IDEA-084 total dual-branch perturbation cap failed")
    return audit


def preprocessing_audit(
    model: GroupedDualBranchClassifier,
    features: np.ndarray,
    train_indices: np.ndarray,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for prefix, indices in (
        ("group_a", model.group_a_indices),
        ("group_b", model.group_b_indices),
    ):
        expected = idea082.stable_age_statistics(features[train_indices][:, indices])
        actual = (
            getattr(model, f"{prefix}_median").cpu().numpy(),
            getattr(model, f"{prefix}_mean").cpu().numpy(),
            getattr(model, f"{prefix}_scale").cpu().numpy(),
        )
        checks[prefix] = all(np.array_equal(left, right) for left, right in zip(expected, actual))
    if not all(checks.values()):
        raise RuntimeError("IDEA-084 preprocessing is not train-role local")
    return {
        "training_role_statistics_equal_recomputed": checks,
        "validation_rows_used_for_statistics": False,
        "test_rows_used_for_statistics": False,
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cpu":
        raise RuntimeError("IDEA-084 preflight is CPU-only")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before IDEA-084 CPU preflight")
    idea068.configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    store = idea068.idea051.reference.historical.idea019.load_feature_store()
    if len(store.call_ids) != 792 or len(np.unique(store.cat_ids.astype(str))) != 111:
        raise RuntimeError("IDEA-084 expected 792 calls from 111 cats")
    features, feature_summary = load_features(protocol, store.call_ids)
    roles = pd.read_csv(
        REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str}
    )
    if set(roles["role"].astype(str)) != {"train", "validation", "test"}:
        raise RuntimeError("IDEA-084 role vocabulary changed")
    role_cells = 0
    first_indices: dict[str, np.ndarray] | None = None
    for repeat in protocol["model"]["repeats"]:
        for fold in protocol["model"]["folds"]:
            cell = roles[(roles["repeat"] == repeat) & (roles["outer_fold"] == fold)]
            cats = {
                role: set(cell[cell["role"] == role]["cat_id"].astype(str))
                for role in ("train", "validation", "test")
            }
            if any(
                cats[left] & cats[right]
                for left, right in (
                    ("train", "validation"),
                    ("train", "test"),
                    ("validation", "test"),
                )
            ):
                raise RuntimeError("IDEA-084 cat leakage across roles")
            indices = role_cell_indices(store, roles, repeat, fold)
            if np.intersect1d(indices["train"], indices["validation"]).size:
                raise RuntimeError("IDEA-084 call leakage across train/validation")
            if first_indices is None:
                first_indices = indices
            role_cells += 1
    if first_indices is None:
        raise RuntimeError("IDEA-084 found no role cells")
    probe_indices = first_indices["validation"][: min(32, len(first_indices["validation"]))]
    initialization = initial_model_audit(
        protocol,
        store,
        features,
        first_indices["train"],
        probe_indices,
        BASE_SEEDS[0],
    )
    gradients: dict[str, Any] = {}
    caps: dict[str, Any] = {}
    preprocessing: dict[str, Any] = {}
    for pipeline in PIPELINES[2:]:
        idea068.idea051.reference.historical.set_seed(BASE_SEEDS[0])
        model = build_model(pipeline, protocol, store, features, first_indices["train"])
        if not isinstance(model, GroupedDualBranchClassifier):
            raise RuntimeError("IDEA-084 dual model type changed")
        preprocessing[pipeline] = preprocessing_audit(
            model, features, first_indices["train"]
        )
        gradients[pipeline] = gradient_audit(model, features[probe_indices])
        idea068.idea051.reference.historical.set_seed(BASE_SEEDS[0])
        cap_model = build_model(
            pipeline, protocol, store, features, first_indices["train"]
        )
        caps[pipeline] = cap_audit(cap_model, store, features, probe_indices)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized during IDEA-084 CPU preflight")
    result = {
        "status": "GO",
        "scope": "CPU preflight only; formal GPU remains unauthorized",
        "protocol_sha256": idea068.sha256(PROTOCOL_PATH),
        "runner_sha256": idea068.sha256(Path(__file__).resolve()),
        "feature_sha256": feature_summary["feature_sha256"],
        "calls": 792,
        "cats": 111,
        "role_cells": role_cells,
        "base_seeds": list(BASE_SEEDS),
        "unique_full_seeds": 36,
        "pipelines": list(PIPELINES),
        "expected_fits": 144,
        "feature_groups": {
            "P1": [list(group) for group in P1_GROUPS],
            "R1": [list(group) for group in R1_GROUPS],
        },
        "outer_test_predictions_or_metrics_accessed": False,
        "cuda_initialized": False,
        "device": "cpu",
        "initialization": initialization,
        "preprocessing": preprocessing,
        "gradient_reachability": gradients,
        "cap_saturation_probes": caps,
    }
    output_path = resolve_run_root(args.output_subdir) / "cpu_preflight.json"
    write_json(output_path, result)
    return result


def require_matching_cpu_preflight(
    protocol: dict[str, Any], output_subdir: str
) -> tuple[dict[str, Any], Path]:
    preflight_path = resolve_run_root(output_subdir) / "cpu_preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("IDEA-084 formal run requires a completed CPU preflight GO")
    preflight_result = read_json(preflight_path)
    expected = {
        "status": "GO",
        "protocol_sha256": idea068.sha256(PROTOCOL_PATH),
        "runner_sha256": idea068.sha256(Path(__file__).resolve()),
        "expected_fits": 144,
        "outer_test_predictions_or_metrics_accessed": False,
        "cuda_initialized": False,
        "device": "cpu",
    }
    for key, value in expected.items():
        if preflight_result.get(key) != value:
            raise RuntimeError(f"IDEA-084 CPU preflight identity mismatch: {key}")
    if preflight_result.get("pipelines") != list(PIPELINES):
        raise RuntimeError("IDEA-084 CPU preflight pipeline matrix mismatch")
    if preflight_result.get("base_seeds") != list(BASE_SEEDS):
        raise RuntimeError("IDEA-084 CPU preflight seed bank mismatch")
    if protocol["execution_gate"].get("gpu_authorized") is not False:
        raise RuntimeError("IDEA-084 protocol GPU authorization field must remain false pre-launch")
    return preflight_result, preflight_path


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
            raise RuntimeError(f"IDEA-084 resume identity mismatch: {key}")
    for prefix in ("validation_animal", "validation_call"):
        path = REPO_ROOT / fit[f"{prefix}_predictions"]
        if not path.is_file() or idea068.sha256(path) != fit[f"{prefix}_sha256"]:
            raise RuntimeError(f"IDEA-084 resume prediction hash mismatch: {path}")


def load_animals(fit: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(
        REPO_ROOT / fit["validation_animal_predictions"], dtype={"cat_id": str}
    )


def metric_bundle(frame: pd.DataFrame) -> dict[str, Any]:
    return idea082.metric_bundle(frame)


def contrast_summary(values: pd.Series) -> dict[str, Any]:
    return idea082.contrast_summary(values)


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
    for name, (candidate, comparator) in {
        **COMPARISONS,
        **DESCRIPTIVE_COMPARISONS,
    }.items():
        row[f"{name}_macro_f1"] = (
            row[f"{candidate}_macro_f1"] - row[f"{comparator}_macro_f1"]
        )
        row[f"{name}_balanced_accuracy"] = (
            row[f"{candidate}_balanced_accuracy"]
            - row[f"{comparator}_balanced_accuracy"]
        )
        row[f"{name}_cross_entropy_gain"] = (
            row[f"{comparator}_cross_entropy"]
            - row[f"{candidate}_cross_entropy"]
        )
        row[f"{name}_brier_gain"] = (
            row[f"{comparator}_brier"] - row[f"{candidate}_brier"]
        )
        row[f"{name}_senior_recall"] = (
            row[f"{candidate}_senior_recall"]
            - row[f"{comparator}_senior_recall"]
        )


def paired_error_transitions(
    candidate: pd.DataFrame, comparator: pd.DataFrame
) -> dict[str, int]:
    keys = [
        key
        for key in ("base_seed", "repeat", "fold", "cat_id")
        if key in candidate.columns and key in comparator.columns
    ]
    if "cat_id" not in keys or "fold" not in keys:
        raise RuntimeError("IDEA-084 paired correction keys are incomplete")
    left = candidate[keys + ["true_label", "predicted_label"]].rename(
        columns={
            "true_label": "candidate_true",
            "predicted_label": "candidate_prediction",
        }
    )
    right = comparator[keys + ["true_label", "predicted_label"]].rename(
        columns={
            "true_label": "comparator_true",
            "predicted_label": "comparator_prediction",
        }
    )
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError("IDEA-084 paired correction rows are not one-to-one")
    if not np.array_equal(
        merged["candidate_true"].to_numpy(), merged["comparator_true"].to_numpy()
    ):
        raise RuntimeError("IDEA-084 paired correction labels differ")
    candidate_correct = merged["candidate_prediction"] == merged["candidate_true"]
    comparator_correct = merged["comparator_prediction"] == merged["comparator_true"]
    corrected = int((candidate_correct & ~comparator_correct).sum())
    introduced = int((~candidate_correct & comparator_correct).sum())
    unchanged_correct = int((candidate_correct & comparator_correct).sum())
    unchanged_wrong = int((~candidate_correct & ~comparator_correct).sum())
    return {
        "paired_occurrences": int(len(merged)),
        "corrected_errors": corrected,
        "introduced_errors": introduced,
        "net_corrections": corrected - introduced,
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
    }


def aggregate(fits: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    fold_results: list[dict[str, Any]] = []
    seed_repeat_results: list[dict[str, Any]] = []
    pooled_all: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
    for base_seed in BASE_SEEDS:
        for repeat in protocol["model"]["repeats"]:
            seed_repeat_frames = {pipeline: [] for pipeline in PIPELINES}
            for fold in protocol["model"]["folds"]:
                bundles: dict[str, Any] = {}
                for pipeline in PIPELINES:
                    fit = next(
                        item
                        for item in fits
                        if item["pipeline"] == pipeline
                        and item["base_seed"] == base_seed
                        and item["repeat"] == repeat
                        and item["fold"] == fold
                    )
                    animals = load_animals(fit)
                    bundles[pipeline] = metric_bundle(animals)
                    tagged = animals.copy()
                    tagged["base_seed"] = base_seed
                    tagged["repeat"] = repeat
                    tagged["fold"] = fold
                    seed_repeat_frames[pipeline].append(tagged)
                    pooled_all[pipeline].append(tagged)
                row: dict[str, Any] = {
                    "base_seed": int(base_seed),
                    "repeat": int(repeat),
                    "fold": int(fold),
                }
                add_pipeline_metrics(row, bundles)
                fold_results.append(row)
            pooled = {
                pipeline: pd.concat(parts, ignore_index=True)
                for pipeline, parts in seed_repeat_frames.items()
            }
            bundles = {pipeline: metric_bundle(frame) for pipeline, frame in pooled.items()}
            row = {"base_seed": int(base_seed), "repeat": int(repeat)}
            add_pipeline_metrics(row, bundles)
            for name, (candidate, comparator) in COMPARISONS.items():
                transition = paired_error_transitions(
                    pooled[candidate], pooled[comparator]
                )
                for key, value in transition.items():
                    row[f"{name}_{key}"] = value
            seed_repeat_results.append(row)

    folds = pd.DataFrame(fold_results)
    seed_repeats = pd.DataFrame(seed_repeat_results)
    contrast_columns = [f"{name}_macro_f1" for name in COMPARISONS]
    split_cells = (
        folds.groupby(["repeat", "fold"], as_index=False)[contrast_columns]
        .mean()
        .sort_values(["repeat", "fold"])
        .reset_index(drop=True)
    )
    pooled_frames = {
        pipeline: pd.concat(parts, ignore_index=True)
        for pipeline, parts in pooled_all.items()
    }
    pipeline_means = {
        metric: {
            pipeline: float(seed_repeats[f"{pipeline}_{metric}"].mean())
            for pipeline in PIPELINES
        }
        for metric in ("macro_f1", "balanced_accuracy", "cross_entropy", "brier")
    }
    gate = protocol["gate"]
    comparison_results: dict[str, Any] = {}
    for name, (candidate, comparator) in COMPARISONS.items():
        column = f"{name}_macro_f1"
        values = seed_repeats[column]
        split_values = split_cells[column]
        per_seed_means = {
            str(seed): float(
                seed_repeats[seed_repeats["base_seed"] == seed][column].mean()
            )
            for seed in BASE_SEEDS
        }
        per_seed_senior: dict[str, float] = {}
        for base_seed in BASE_SEEDS:
            selected_candidate = pooled_frames[candidate][
                pooled_frames[candidate]["base_seed"] == base_seed
            ]
            selected_comparator = pooled_frames[comparator][
                pooled_frames[comparator]["base_seed"] == base_seed
            ]
            per_seed_senior[str(base_seed)] = float(
                idea068.idea051.animal_metrics(selected_candidate)["per_class"]["senior"][
                    "recall"
                ]
                - idea068.idea051.animal_metrics(selected_comparator)["per_class"][
                    "senior"
                ]["recall"]
            )
        conditions = {
            "mean_macro_f1_delta": float(values.mean())
            >= float(gate["minimum_mean_seed_repeat_macro_f1_delta"]),
            "positive_base_seed_means": sum(value > 0.0 for value in per_seed_means.values())
            >= int(gate["minimum_positive_base_seed_means"]),
            "positive_seed_repeats": int((values > 0).sum())
            >= int(gate["minimum_positive_seed_repeats"]),
            "nonnegative_split_cells": int((split_values >= 0).sum())
            >= int(gate["minimum_nonnegative_split_cells"]),
            "worst_split_cell": float(split_values.min())
            >= float(gate["minimum_worst_split_cell_delta"]),
            "mean_cross_entropy_nonworse": pipeline_means["cross_entropy"][candidate]
            <= pipeline_means["cross_entropy"][comparator],
            "mean_brier_nonworse": pipeline_means["brier"][candidate]
            <= pipeline_means["brier"][comparator],
            "mean_balanced_accuracy_nonworse": pipeline_means["balanced_accuracy"][
                candidate
            ]
            >= pipeline_means["balanced_accuracy"][comparator],
            "per_base_seed_senior_recall_safety": all(
                value >= float(gate["minimum_per_base_seed_senior_recall_delta"])
                for value in per_seed_senior.values()
            ),
        }
        correction_columns = (
            "paired_occurrences",
            "corrected_errors",
            "introduced_errors",
            "net_corrections",
            "unchanged_correct",
            "unchanged_wrong",
        )
        correction_profile = {
            key: {
                "mean_per_seed_repeat": float(seed_repeats[f"{name}_{key}"].mean()),
                "total_descriptive_repeated_occurrences": int(
                    seed_repeats[f"{name}_{key}"].sum()
                ),
            }
            for key in correction_columns
        }
        correction_profile["net_correction_positive_tied_negative"] = {
            "positive": int((seed_repeats[f"{name}_net_corrections"] > 0).sum()),
            "tied": int((seed_repeats[f"{name}_net_corrections"] == 0).sum()),
            "negative": int((seed_repeats[f"{name}_net_corrections"] < 0).sum()),
        }
        comparison_results[name] = {
            "candidate": candidate,
            "comparator": comparator,
            "macro_f1": contrast_summary(values),
            "per_base_seed_mean_delta": per_seed_means,
            "split_cell_nonnegative": int((split_values >= 0).sum()),
            "split_cell_worst": float(split_values.min()),
            "mean_balanced_accuracy_delta": pipeline_means["balanced_accuracy"][candidate]
            - pipeline_means["balanced_accuracy"][comparator],
            "mean_cross_entropy_gain": pipeline_means["cross_entropy"][comparator]
            - pipeline_means["cross_entropy"][candidate],
            "mean_brier_gain": pipeline_means["brier"][comparator]
            - pipeline_means["brier"][candidate],
            "per_base_seed_senior_recall_delta": per_seed_senior,
            "error_correction_profile": correction_profile,
            "conditions": conditions,
            "gate_passed": bool(all(conditions.values())),
            "interpretation_boundary": (
                "One fixed random grouping controls capacity but cannot prove complete causality."
                if name == "P1_minus_R1"
                else "Same 111 cats; not external confirmation."
            ),
        }

    descriptive_results: dict[str, Any] = {}
    for name, (candidate, comparator) in DESCRIPTIVE_COMPARISONS.items():
        descriptive_results[name] = {
            "candidate": candidate,
            "comparator": comparator,
            "macro_f1": contrast_summary(seed_repeats[f"{name}_macro_f1"]),
            "mean_balanced_accuracy_delta": pipeline_means["balanced_accuracy"][candidate]
            - pipeline_means["balanced_accuracy"][comparator],
            "mean_cross_entropy_gain": pipeline_means["cross_entropy"][comparator]
            - pipeline_means["cross_entropy"][candidate],
            "mean_brier_gain": pipeline_means["brier"][comparator]
            - pipeline_means["brier"][candidate],
            "status": "contemporaneous_context_only_not_a_reopened_confirmation_gate",
        }
    pooled_metrics = {
        pipeline: {
            "animal_occurrences": int(len(frame)),
            **metric_bundle(frame),
        }
        for pipeline, frame in pooled_frames.items()
    }
    return {
        "status": "complete",
        "outer_test_accessed": False,
        "fits": len(fits),
        "paired_fold_comparisons": len(fold_results),
        "seed_repeat_estimates": len(seed_repeat_results),
        "split_cell_estimates": int(len(split_cells)),
        "independence_note": "Fold deltas and repeated animal occurrences reuse the same 111 cats and are descriptive, not independent samples.",
        "no_single_global_pass_flag": True,
        "pipeline_seed_repeat_means": pipeline_means,
        "comparison_results": comparison_results,
        "descriptive_results": descriptive_results,
        "fold_results": fold_results,
        "seed_repeat_results": seed_repeat_results,
        "split_cell_results": split_cells.to_dict(orient="records"),
        "pooled_validation": pooled_metrics,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.director_authorized:
        raise RuntimeError(
            "IDEA-084 formal run is blocked until the research director explicitly authorizes it; pass --director-authorized only after that decision."
        )
    idea068.configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    max_cells = getattr(args, "max_cells", None)
    if max_cells is not None and max_cells <= 0:
        raise ValueError("--max-cells must be positive")
    cpu_preflight, cpu_preflight_path = require_matching_cpu_preflight(
        protocol, args.output_subdir
    )
    run_root = resolve_run_root(args.output_subdir)
    run_root.mkdir(parents=True, exist_ok=True)
    store = idea068.idea051.reference.historical.idea019.load_feature_store()
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
        "cpu_preflight_sha256": idea068.sha256(cpu_preflight_path),
        "cpu_preflight_status": cpu_preflight["status"],
        "director_authorized": True,
        "outer_test_accessed": False,
        "pipelines": list(PIPELINES),
        "model": protocol["model"],
        "feature_groups": protocol["feature_groups"],
        "comparisons": protocol["comparisons"],
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
            raise RuntimeError("Existing IDEA-084 run manifest differs")
    else:
        write_json(manifest_path, manifest)

    completed: list[dict[str, Any]] = []
    processed_cells = 0
    for base_seed in BASE_SEEDS:
        for repeat in protocol["model"]["repeats"]:
            for fold in protocol["model"]["folds"]:
                indices = role_cell_indices(store, roles, repeat, fold)
                full_seed = idea068.idea051.reference.historical.full_seed(
                    int(base_seed), repeat, fold
                )
                initialization = initial_model_audit(
                    protocol,
                    store,
                    features,
                    indices["train"],
                    indices["validation"][: min(32, len(indices["validation"]))],
                    full_seed,
                )
                cell_fits: list[dict[str, Any]] = []
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
                            f"=== {pipeline} base_seed={base_seed} repeat={repeat} fold={fold} full_seed={full_seed} ===",
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
                        if audit["model"]["trainable_parameters"] != EXPECTED_PARAMETERS[pipeline]:
                            raise RuntimeError("IDEA-084 trained parameter audit mismatch")
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
                            "initialization": initialization,
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
                    cell_fits.append(fit)
                common_epochs = min(len(item["audit"]["history"]) for item in cell_fits)
                for epoch in range(common_epochs):
                    reference = cell_fits[0]["audit"]["history"][epoch]["train_audit"]
                    for candidate in cell_fits[1:]:
                        current = candidate["audit"]["history"][epoch]["train_audit"]
                        if (
                            reference["cat_order_sha256"] != current["cat_order_sha256"]
                            or reference["call_coverage_sha256"]
                            != current["call_coverage_sha256"]
                        ):
                            raise RuntimeError("IDEA-084 paired batch order differs")

                processed_cells += 1
                if max_cells is not None and processed_cells >= max_cells:
                    partial = {
                        "status": "partial_first_cell_audit_required",
                        "completed_cells_this_invocation": processed_cells,
                        "completed_fits_visible_this_invocation": len(completed),
                        "expected_total_fits": 144,
                        "aggregation_generated": False,
                        "resume_required": True,
                        "outer_test_accessed": False,
                        "protocol_sha256": idea068.sha256(PROTOCOL_PATH),
                        "runner_sha256": idea068.sha256(Path(__file__).resolve()),
                        "cpu_preflight_sha256": idea068.sha256(cpu_preflight_path),
                    }
                    partial_path = run_root / "partial_run_summary.json"
                    write_json(partial_path, partial)
                    return partial

    summary = aggregate(completed, protocol)
    summary_path = run_root / "initial_evaluation_summary.json"
    summary_bytes = canonical_json_bytes(summary)
    if summary_path.exists():
        if not args.resume:
            raise FileExistsError(summary_path)
        if read_json(summary_path) != summary or summary_path.read_bytes() != summary_bytes:
            raise RuntimeError("IDEA-084 resumed aggregate differs")
    else:
        write_json(summary_path, summary)
    compact = {
        "status": "complete",
        "completed_fits": len(completed),
        "expected_fits": 144,
        "comparison_gate_passed": {
            name: value["gate_passed"]
            for name, value in summary["comparison_results"].items()
        },
        "no_single_global_pass_flag": True,
        "outer_test_accessed": False,
    }
    compact_path = run_root / "run_summary.json"
    if compact_path.exists():
        if read_json(compact_path) != compact or compact_path.read_bytes() != canonical_json_bytes(compact):
            raise RuntimeError("IDEA-084 resumed compact summary differs")
    else:
        write_json(compact_path, compact)
    return summary


def main() -> None:
    args = parse_args()
    result = preflight(args) if args.stage == "preflight" else run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
