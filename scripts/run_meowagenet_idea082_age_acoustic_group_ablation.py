"""Run IDEA-082 preregistered acoustic-mechanism group ablations."""

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


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea082_age_acoustic_group_ablation_v1.json"
)
PIPELINES = (
    "A0_ast_only",
    "G1_f0_real",
    "G1_f0_shuffled",
    "G2_stability_real",
    "G2_stability_shuffled",
    "G3_spectral_energy_real",
    "G3_spectral_energy_shuffled",
    "G12_f0_stability_real",
    "G12_f0_stability_shuffled",
)
GROUPS = (
    "G1_f0",
    "G2_stability",
    "G3_spectral_energy",
    "G12_f0_stability",
)
GROUP_INDICES = {
    "G1_f0": (0, 1, 2, 3, 6),
    "G2_stability": (4, 5, 7, 8, 9, 10, 11, 12, 13, 14),
    "G3_spectral_energy": (15, 16, 17, 18, 19),
    "G12_f0_stability": tuple(range(15)),
}
GROUP_WIDTHS = {
    "G1_f0": 67,
    "G2_stability": 64,
    "G3_spectral_energy": 67,
    "G12_f0_stability": 62,
}
BASE_SEEDS = (8694, 5378, 5945)
EXPECTED_PARAMETERS = {
    "A0_ast_only": 99_075,
    "G1_f0_real": 108_181,
    "G1_f0_shuffled": 108_181,
    "G2_stability_real": 108_099,
    "G2_stability_shuffled": 108_099,
    "G3_spectral_energy_real": 108_181,
    "G3_spectral_energy_shuffled": 108_181,
    "G12_f0_stability_real": 108_131,
    "G12_f0_stability_shuffled": 108_131,
}
_IDEA068_BUILD_MODEL = idea068.build_model
_IDEA068_PREDICT = idea068.predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "run"), required=True)
    parser.add_argument(
        "--output-subdir",
        default="meowagenet_idea082_age_acoustic_group_ablation_v1",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--director-authorized", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def pipeline_group(pipeline: str) -> str:
    if pipeline == PIPELINES[0]:
        raise ValueError("A0 has no acoustic group")
    for group in GROUPS:
        if pipeline.startswith(group):
            return group
    raise ValueError(pipeline)


def is_shuffled(pipeline: str) -> bool:
    return pipeline.endswith("_shuffled")


def stable_age_statistics(age_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Permutation-invariant train-only imputation and scaling statistics."""

    with np.errstate(all="ignore"):
        median = np.nanmedian(age_train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    imputed = np.where(np.isfinite(age_train), age_train, median[None, :]).astype(
        np.float32
    )
    # Sorting each column makes real/shuffled controls bit-identical despite
    # floating-point reduction order while retaining exactly the same estimator.
    ordered = np.sort(imputed, axis=0)
    mean = ordered.mean(axis=0).astype(np.float32)
    scale = ordered.std(axis=0).astype(np.float32)
    scale = np.where(scale > 1.0e-8, scale, 1.0).astype(np.float32)
    return median, mean, scale


class GroupedC1Classifier(idea071.PerturbationAuditMixin, torch.nn.Module):
    """C1 with a preregistered feature subset and parameter-budgeted width."""

    def __init__(
        self,
        pipeline: str,
        ast_mean: np.ndarray,
        ast_scale: np.ndarray,
        age_train: np.ndarray,
        dropout: float,
        hidden_units: int,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        safe_ast_scale = np.where(ast_scale > 1.0e-12, ast_scale, 1.0).astype(
            np.float32
        )
        self.register_buffer("ast_mean", torch.from_numpy(ast_mean.astype(np.float32)))
        self.register_buffer("ast_scale", torch.from_numpy(safe_ast_scale))
        # Module creation order matches IDEA-068 so the shared AST head is paired.
        self.ast_linear = torch.nn.Linear(768, 128)
        self.relu = torch.nn.ReLU()
        self.batch_norm = torch.nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01)
        self.dropout = torch.nn.Dropout(dropout)
        self.output = torch.nn.Linear(128, 3)
        median, mean, scale = stable_age_statistics(age_train)
        self.register_buffer("age_median", torch.from_numpy(median))
        self.register_buffer("age_mean", torch.from_numpy(mean))
        self.register_buffer("age_scale", torch.from_numpy(scale))
        self.age_hidden = torch.nn.Linear(age_train.shape[1], hidden_units)
        self.age_output = torch.nn.Linear(hidden_units, 128)
        torch.nn.init.zeros_(self.age_output.weight)
        torch.nn.init.zeros_(self.age_output.bias)
        self.reset_perturbation_audit()

    def standardized_age(self, age_features: torch.Tensor) -> torch.Tensor:
        imputed = torch.where(
            torch.isfinite(age_features), age_features, self.age_median
        )
        return (imputed - self.age_mean) / self.age_scale

    def forward(
        self, ast_embeddings: torch.Tensor, age_features: torch.Tensor
    ) -> torch.Tensor:
        ast = (ast_embeddings - self.ast_mean) / self.ast_scale
        hidden = self.relu(self.ast_linear(ast))
        context = torch.nn.functional.gelu(
            self.age_hidden(self.standardized_age(age_features))
        )
        residual = (
            idea071.CAP
            * idea071.hidden_rms(hidden)
            * torch.tanh(self.age_output(context))
        )
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
            "fusion": "bounded_rms_relative_additive",
            "cap": idea071.CAP,
            "rms_epsilon": idea071.RMS_EPSILON,
            "input_dimensions": int(self.age_hidden.in_features),
            "hidden_units": int(self.age_hidden.out_features),
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
    group = pipeline_group(pipeline)
    embeddings = store.frozen_embeddings[train_indices]
    return GroupedC1Classifier(
        pipeline=pipeline,
        ast_mean=embeddings.mean(axis=0),
        ast_scale=embeddings.std(axis=0),
        age_train=age_features[train_indices],
        dropout=float(protocol["fixed_training"]["dropout"]),
        hidden_units=GROUP_WIDTHS[group],
    )


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


def role_derangement(
    indices: np.ndarray, repeat: int, fold: int, role: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    destinations = np.sort(np.asarray(indices, dtype=np.int64))
    if len(destinations) < 2:
        raise RuntimeError("IDEA-082 cannot derange a role with fewer than two calls")
    material = f"IDEA-082-shuffle-v1|repeat={repeat}|fold={fold}|role={role}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    order = destinations[rng.permutation(len(destinations))]
    source = np.roll(order, 1)
    sorting = np.argsort(order)
    destination_sorted = order[sorting]
    source_sorted = source[sorting]
    fixed_points = int(np.sum(destination_sorted == source_sorted))
    if fixed_points:
        raise RuntimeError("IDEA-082 derangement has a fixed point")
    pairs = np.column_stack([destination_sorted, source_sorted]).astype("<i8")
    return destination_sorted, source_sorted, {
        "role": role,
        "role_size": int(len(destinations)),
        "seed_material_sha256": hashlib.sha256(
            material.encode("utf-8")
        ).hexdigest(),
        "mapping_sha256": hashlib.sha256(pairs.tobytes()).hexdigest(),
        "fixed_points": fixed_points,
    }


def row_multiset_sha(values: np.ndarray) -> str:
    rows = sorted(np.ascontiguousarray(row).tobytes() for row in values)
    return hashlib.sha256(b"".join(rows)).hexdigest()


def prepare_pipeline_features(
    pipeline: str,
    full_features: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    repeat: int,
    fold: int,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    if pipeline == PIPELINES[0]:
        return full_features, None
    group = pipeline_group(pipeline)
    selected = full_features[:, GROUP_INDICES[group]].astype(np.float32, copy=True)
    if not is_shuffled(pipeline):
        return selected, None
    shuffled = np.full(selected.shape, np.nan, dtype=np.float32)
    audits: dict[str, Any] = {}
    for role, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
    ):
        destinations, sources, audit = role_derangement(indices, repeat, fold, role)
        shuffled[destinations] = selected[sources]
        real_sha = row_multiset_sha(selected[destinations])
        shuffled_sha = row_multiset_sha(shuffled[destinations])
        audit["real_row_multiset_sha256"] = real_sha
        audit["shuffled_row_multiset_sha256"] = shuffled_sha
        audit["multiset_equal"] = real_sha == shuffled_sha
        if not audit["multiset_equal"]:
            raise RuntimeError("IDEA-082 shuffled feature multiset changed")
        audits[role] = audit
    real_stats = stable_age_statistics(selected[train_indices])
    shuffled_stats = stable_age_statistics(shuffled[train_indices])
    stats_equal = all(
        np.array_equal(real, permuted)
        for real, permuted in zip(real_stats, shuffled_stats)
    )
    if not stats_equal:
        raise RuntimeError("IDEA-082 real/shuffled training statistics differ")
    return shuffled, {
        "group": group,
        "roles": audits,
        "training_statistics_equal": stats_equal,
        "test_rows_materialized": False,
    }


def load_features(protocol: dict[str, Any], call_ids: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    summary_path = REPO_ROOT / protocol["data"]["feature_summary_path"]
    summary = read_json(summary_path)
    required = {
        "status": "complete",
        "label_information_used": False,
        "calls": 792,
        "features": 20,
        "source_hashes_verified": 792,
        "fully_finite_calls": 772,
        "feature_sha256": protocol["data"]["feature_sha256"],
        "protocol_sha256": protocol["dependencies"]["idea068_protocol_sha256"],
        "runner_sha256": protocol["dependencies"]["idea068_runner_sha256"],
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"IDEA-082 feature audit changed: {key}")
    feature_path = REPO_ROOT / protocol["data"]["feature_path"]
    loaded = np.load(feature_path)
    if tuple(loaded["feature_names"].astype(str)) != idea068.FEATURE_NAMES:
        raise RuntimeError("IDEA-082 feature schema changed")
    if not np.array_equal(loaded["call_ids"].astype(str), call_ids.astype(str)):
        raise RuntimeError("IDEA-082 feature and AST call order differs")
    features = loaded["features"].astype(np.float32)
    if features.shape != (792, 20):
        raise RuntimeError("IDEA-082 feature matrix shape changed")
    return features, summary


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea082-age-acoustic-group-ablation-v1":
        raise RuntimeError("Unexpected IDEA-082 protocol")
    if protocol.get("status") != "locked_for_cpu_preflight_before_initial_evaluation":
        raise RuntimeError("IDEA-082 protocol is not pre-result locked")
    model = protocol["model"]
    if tuple(model["pipelines"]) != PIPELINES or tuple(model["base_seeds"]) != BASE_SEEDS:
        raise RuntimeError("IDEA-082 pipeline or seed lock changed")
    if model["repeats"] != [0, 1, 2] or model["folds"] != [0, 1, 2, 3]:
        raise RuntimeError("IDEA-082 split scope changed")
    if int(model["total_fits"]) != 324 or int(model["fits_per_pipeline"]) != 36:
        raise RuntimeError("IDEA-082 fit budget changed")
    if float(model["cap"]) != idea071.CAP or float(model["rms_epsilon"]) != idea071.RMS_EPSILON:
        raise RuntimeError("IDEA-082 C1 bound changed")
    if model.get("full_C1_rerun") is not False:
        raise RuntimeError("IDEA-082 must not rerun full C1")
    groups = protocol["feature_groups"]
    for group in GROUPS:
        entry = groups[group]
        indices = tuple(entry["indices"])
        if indices != GROUP_INDICES[group]:
            raise RuntimeError(f"IDEA-082 group indices changed: {group}")
        expected_names = tuple(idea068.FEATURE_NAMES[index] for index in indices)
        if tuple(entry["feature_names"]) != expected_names:
            raise RuntimeError(f"IDEA-082 group names changed: {group}")
        if int(entry["hidden_units"]) != GROUP_WIDTHS[group]:
            raise RuntimeError(f"IDEA-082 group width changed: {group}")
        age_parameters = GROUP_WIDTHS[group] * (len(indices) + 129) + 128
        if int(entry["age_branch_parameters"]) != age_parameters:
            raise RuntimeError(f"IDEA-082 parameter formula changed: {group}")
    primary = [set(GROUP_INDICES[group]) for group in GROUPS[:3]]
    if any(primary[left] & primary[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("IDEA-082 primary groups overlap")
    if set().union(*primary) != set(range(20)):
        raise RuntimeError("IDEA-082 primary groups do not exhaust 20 features")
    if set(GROUP_INDICES[GROUPS[3]]) != primary[0] | primary[1]:
        raise RuntimeError("IDEA-082 G12 is not exactly G1 union G2")
    expected_from_groups = {PIPELINES[0]: 99_075}
    for pipeline in PIPELINES[1:]:
        expected_from_groups[pipeline] = int(
            groups[pipeline_group(pipeline)]["total_trainable_parameters"]
        )
    if expected_from_groups != EXPECTED_PARAMETERS:
        raise RuntimeError("IDEA-082 trainable parameter lock changed")
    digest = hashlib.sha256(model["seed_derivation_text"].encode("utf-8")).hexdigest()
    if digest != model["seed_derivation_sha256"]:
        raise RuntimeError("IDEA-082 seed derivation digest changed")
    candidates = [
        int.from_bytes(bytes.fromhex(digest)[offset : offset + 4], "big") % 10_000
        for offset in range(0, 32, 4)
    ]
    if candidates != model["candidate_sequence"]:
        raise RuntimeError("IDEA-082 seed candidate sequence changed")
    current_full = {
        base + 10_000 * repeat + 100 * fold
        for base in BASE_SEEDS
        for repeat in range(3)
        for fold in range(4)
    }
    if len(current_full) != 36:
        raise RuntimeError("IDEA-082 full seeds are not unique")
    prior_full = {
        int(base) + 10_000 * repeat + 100 * fold
        for base in model["known_prior_base_seeds"]
        for repeat in range(3)
        for fold in range(5)
    }
    if current_full & prior_full:
        raise RuntimeError("IDEA-082 full seed collides with prior experiments")
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
            raise RuntimeError(f"IDEA-082 changed locked training field: {key}")
    if protocol["determinism"] != source_protocol["determinism"]:
        raise RuntimeError("IDEA-082 changed locked determinism")
    dependencies = protocol["dependencies"]
    checks = {
        REPO_ROOT / dependencies["independent_grouping_review_path"]: dependencies[
            "independent_grouping_review_sha256"
        ],
        idea068.PROTOCOL_PATH: dependencies["idea068_protocol_sha256"],
        Path(idea068.__file__).resolve(): dependencies["idea068_runner_sha256"],
        idea071.PROTOCOL_PATH: dependencies["idea071_protocol_sha256"],
        Path(idea071.__file__).resolve(): dependencies["idea071_runner_sha256"],
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
            raise RuntimeError(f"IDEA-082 dependency checksum mismatch: {path}")


def resolve_run_root(output_subdir: str) -> Path:
    run_root = (idea068.idea051.RUNS_ROOT / output_subdir).resolve()
    if idea068.idea051.RUNS_ROOT.resolve() not in run_root.parents:
        raise ValueError("--output-subdir must stay below runs")
    return run_root


def role_cell_indices(store: Any, roles: pd.DataFrame, repeat: int, fold: int) -> dict[str, np.ndarray]:
    return idea068.idea051.reference.historical.fold_indices(
        store, roles, repeat, fold, include_test=False
    )


def common_state_equal(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    keys = ("ast_mean", "ast_scale", "ast_linear.weight", "ast_linear.bias", "batch_norm.weight", "batch_norm.bias", "batch_norm.running_mean", "batch_norm.running_var", "batch_norm.num_batches_tracked", "output.weight", "output.bias")
    left_state = left.state_dict()
    right_state = right.state_dict()
    return all(torch.equal(left_state[key], right_state[key]) for key in keys)


def initial_model_audit(
    protocol: dict[str, Any],
    store: Any,
    feature_matrices: dict[str, np.ndarray],
    train_indices: np.ndarray,
    probe_indices: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    models: dict[str, torch.nn.Module] = {}
    logits: dict[str, np.ndarray] = {}
    parameters: dict[str, int] = {}
    for pipeline in PIPELINES:
        idea068.idea051.reference.historical.set_seed(seed)
        model = build_model(
            pipeline, protocol, store, feature_matrices[pipeline], train_indices
        ).eval()
        models[pipeline] = model
        parameters[pipeline] = int(sum(p.numel() for p in model.parameters()))
        with torch.no_grad():
            logits[pipeline] = model(
                torch.from_numpy(store.frozen_embeddings[probe_indices]),
                torch.from_numpy(feature_matrices[pipeline][probe_indices]),
            ).cpu().numpy()
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"IDEA-082 parameter audit mismatch: {parameters}")
    differences = {
        pipeline: float(np.max(np.abs(logits[pipeline] - logits[PIPELINES[0]])))
        for pipeline in PIPELINES[1:]
    }
    if any(value != 0.0 for value in differences.values()):
        raise RuntimeError("IDEA-082 pipelines differ at initialization")
    common_equal = {
        pipeline: common_state_equal(models[PIPELINES[0]], models[pipeline])
        for pipeline in PIPELINES[1:]
    }
    if not all(common_equal.values()):
        raise RuntimeError("IDEA-082 shared AST heads differ at initialization")
    pair_equal = {}
    for group in GROUPS:
        real = models[f"{group}_real"].state_dict()
        shuffled = models[f"{group}_shuffled"].state_dict()
        pair_equal[group] = real.keys() == shuffled.keys() and all(
            torch.equal(real[key], shuffled[key]) for key in real
        )
    if not all(pair_equal.values()):
        raise RuntimeError("IDEA-082 real/shuffled initial states differ")
    return {
        "parameters": parameters,
        "max_logit_difference_vs_A0": differences,
        "common_AST_state_equal_to_A0": common_equal,
        "real_shuffled_full_state_equal": pair_equal,
    }


def gradient_audit(model: GroupedC1Classifier, features: np.ndarray) -> dict[str, float]:
    age = torch.from_numpy(features)
    model.zero_grad(set_to_none=True)
    context = torch.nn.functional.gelu(
        model.age_hidden(model.standardized_age(age))
    )
    model.age_output(context).sum().backward()
    zero_output_grad = float(model.age_output.bias.grad.norm())
    zero_hidden_grad = float(model.age_hidden.weight.grad.norm())
    if zero_output_grad <= 0.0 or zero_hidden_grad != 0.0:
        raise RuntimeError("IDEA-082 zero-output gradient audit failed")
    with torch.no_grad():
        model.age_output.weight.fill_(0.01)
    model.zero_grad(set_to_none=True)
    context = torch.nn.functional.gelu(
        model.age_hidden(model.standardized_age(age))
    )
    model.age_output(context).sum().backward()
    probe_hidden_grad = float(model.age_hidden.weight.grad.norm())
    if probe_hidden_grad <= 0.0:
        raise RuntimeError("IDEA-082 hidden branch is not gradient reachable")
    return {
        "zero_init_age_output_bias_grad_norm": zero_output_grad,
        "zero_init_age_hidden_weight_grad_norm": zero_hidden_grad,
        "nonzero_output_probe_age_hidden_weight_grad_norm": probe_hidden_grad,
    }


def cap_audit(
    model: GroupedC1Classifier,
    store: Any,
    features: np.ndarray,
    probe_indices: np.ndarray,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        model.age_output.weight.fill_(100.0)
        model.age_output.bias.fill_(100.0)
    model.reset_perturbation_audit()
    with torch.no_grad():
        model(
            torch.from_numpy(store.frozen_embeddings[probe_indices]),
            torch.from_numpy(features[probe_indices]),
        )
    audit = model.perturbation_audit()
    if audit["validation_relative_perturbation_max"] > idea071.CAP + 1.0e-5:
        raise RuntimeError("IDEA-082 C1 perturbation cap failed")
    return audit


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cpu":
        raise RuntimeError("IDEA-082 preflight is CPU-only")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before IDEA-082 CPU preflight")
    idea068.configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    store = idea068.idea051.reference.historical.idea019.load_feature_store()
    if len(store.call_ids) != 792 or len(np.unique(store.cat_ids.astype(str))) != 111:
        raise RuntimeError("IDEA-082 expected 792 calls from 111 cats")
    features, feature_summary = load_features(protocol, store.call_ids)
    roles = pd.read_csv(
        REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str}
    )
    if set(roles["role"].astype(str)) != {"train", "validation", "test"}:
        raise RuntimeError("IDEA-082 role vocabulary changed")
    role_cells = 0
    shuffle_cells: list[dict[str, Any]] = []
    first: tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None = None
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
                raise RuntimeError("IDEA-082 cat leakage across roles")
            indices = role_cell_indices(store, roles, repeat, fold)
            if np.intersect1d(indices["train"], indices["validation"]).size:
                raise RuntimeError("IDEA-082 call leakage across train/validation")
            matrices: dict[str, np.ndarray] = {PIPELINES[0]: features}
            cell_audits: dict[str, Any] = {}
            for pipeline in PIPELINES[1:]:
                matrix, audit = prepare_pipeline_features(
                    pipeline,
                    features,
                    indices["train"],
                    indices["validation"],
                    repeat,
                    fold,
                )
                matrices[pipeline] = matrix
                if audit is not None:
                    cell_audits[pipeline_group(pipeline)] = audit
            mapping_hashes = {
                role: {
                    audit["roles"][role]["mapping_sha256"]
                    for audit in cell_audits.values()
                }
                for role in ("train", "validation")
            }
            if any(len(values) != 1 for values in mapping_hashes.values()):
                raise RuntimeError("IDEA-082 groups do not share the same shuffle mapping")
            shuffle_cells.append(
                {
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "mapping_sha256": {
                        role: next(iter(values)) for role, values in mapping_hashes.items()
                    },
                    "fixed_points": {
                        role: next(iter(cell_audits.values()))["roles"][role][
                            "fixed_points"
                        ]
                        for role in ("train", "validation")
                    },
                    "all_group_multisets_equal": all(
                        all(role_audit["multiset_equal"] for role_audit in audit["roles"].values())
                        for audit in cell_audits.values()
                    ),
                    "all_group_training_statistics_equal": all(
                        audit["training_statistics_equal"] for audit in cell_audits.values()
                    ),
                }
            )
            if first is None:
                first = (indices, matrices)
            role_cells += 1
    if first is None:
        raise RuntimeError("IDEA-082 found no role cells")
    indices, matrices = first
    probe_indices = indices["validation"][: min(32, len(indices["validation"]))]
    initialization = initial_model_audit(
        protocol, store, matrices, indices["train"], probe_indices, BASE_SEEDS[0]
    )
    gradients: dict[str, Any] = {}
    caps: dict[str, Any] = {}
    for group in GROUPS:
        pipeline = f"{group}_real"
        idea068.idea051.reference.historical.set_seed(BASE_SEEDS[0])
        model = build_model(
            pipeline, protocol, store, matrices[pipeline], indices["train"]
        )
        if not isinstance(model, GroupedC1Classifier):
            raise RuntimeError("IDEA-082 grouped model type changed")
        gradients[group] = gradient_audit(model, matrices[pipeline][probe_indices])
        idea068.idea051.reference.historical.set_seed(BASE_SEEDS[0])
        cap_model = build_model(
            pipeline, protocol, store, matrices[pipeline], indices["train"]
        )
        caps[group] = cap_audit(cap_model, store, matrices[pipeline], probe_indices)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized during IDEA-082 CPU preflight")
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
        "expected_fits": 324,
        "outer_test_predictions_or_metrics_accessed": False,
        "cuda_initialized": False,
        "device": "cpu",
        "initialization": initialization,
        "shuffle_cells": shuffle_cells,
        "gradient_reachability": gradients,
        "cap_saturation_probes": caps,
    }
    output_path = resolve_run_root(args.output_subdir) / "cpu_preflight.json"
    write_json(output_path, result)
    return result


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
            raise RuntimeError(f"IDEA-082 resume identity mismatch: {key}")
    for prefix in ("validation_animal", "validation_call"):
        path = REPO_ROOT / fit[f"{prefix}_predictions"]
        if not path.is_file() or idea068.sha256(path) != fit[f"{prefix}_sha256"]:
            raise RuntimeError(f"IDEA-082 resume prediction hash mismatch: {path}")


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
    for group in GROUPS:
        real = f"{group}_real"
        shuffled = f"{group}_shuffled"
        row[f"{group}_information_macro_f1"] = (
            row[f"{real}_macro_f1"] - row[f"{shuffled}_macro_f1"]
        )
        row[f"{group}_utility_macro_f1"] = (
            row[f"{real}_macro_f1"] - row[f"{PIPELINES[0]}_macro_f1"]
        )
    row["G12_real_minus_G1_real_macro_f1"] = (
        row["G12_f0_stability_real_macro_f1"] - row["G1_f0_real_macro_f1"]
    )
    row["G12_real_minus_G2_real_macro_f1"] = (
        row["G12_f0_stability_real_macro_f1"] - row["G2_stability_real_macro_f1"]
    )
    row["G12_redundancy_interaction_macro_f1"] = (
        row["G12_f0_stability_real_macro_f1"]
        - row["G1_f0_real_macro_f1"]
        - row["G2_stability_real_macro_f1"]
        + row["A0_ast_only_macro_f1"]
    )


def contrast_summary(values: pd.Series) -> dict[str, Any]:
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "median": float(values.median()),
        "positive": int((values > 0).sum()),
        "tied": int((values == 0).sum()),
        "negative": int((values < 0).sum()),
        "worst": float(values.min()),
        "best": float(values.max()),
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
            seed_repeat_results.append(row)
    folds = pd.DataFrame(fold_results)
    seed_repeats = pd.DataFrame(seed_repeat_results)
    contrast_columns = [
        f"{group}_{kind}_macro_f1"
        for group in GROUPS
        for kind in ("information", "utility")
    ]
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
    group_results: dict[str, Any] = {}
    for group in GROUPS:
        real = f"{group}_real"
        shuffled = f"{group}_shuffled"
        per_seed_senior: dict[str, float] = {}
        for base_seed in BASE_SEEDS:
            selected = {
                pipeline: frame[frame["base_seed"] == base_seed]
                for pipeline, frame in pooled_frames.items()
            }
            per_seed_senior[str(base_seed)] = float(
                idea068.idea051.animal_metrics(selected[real])["per_class"]["senior"]["recall"]
                - idea068.idea051.animal_metrics(selected[PIPELINES[0]])["per_class"]["senior"]["recall"]
            )
        contrasts: dict[str, Any] = {}
        for kind, comparator in (("information", shuffled), ("utility", PIPELINES[0])):
            column = f"{group}_{kind}_macro_f1"
            values = seed_repeats[column]
            split_values = split_cells[column]
            per_seed_means = {
                str(seed): float(
                    seed_repeats[seed_repeats["base_seed"] == seed][column].mean()
                )
                for seed in BASE_SEEDS
            }
            conditions = {
                "mean_macro_f1_delta": float(values.mean())
                >= float(gate["minimum_mean_seed_repeat_macro_f1_delta"]),
                "positive_base_seed_means": sum(v > 0.0 for v in per_seed_means.values())
                >= int(gate["minimum_positive_base_seed_means"]),
                "positive_seed_repeats": int((values > 0).sum())
                >= int(gate["minimum_positive_seed_repeats"]),
                "nonnegative_split_cells": int((split_values >= 0).sum())
                >= int(gate["minimum_nonnegative_split_cells"]),
                "worst_split_cell": float(split_values.min())
                >= float(gate["minimum_worst_split_cell_delta"]),
                "mean_cross_entropy_nonworse": pipeline_means["cross_entropy"][real]
                <= pipeline_means["cross_entropy"][comparator],
                "mean_brier_nonworse": pipeline_means["brier"][real]
                <= pipeline_means["brier"][comparator],
                "mean_balanced_accuracy_nonworse": pipeline_means["balanced_accuracy"][real]
                >= pipeline_means["balanced_accuracy"][comparator],
            }
            if kind == "utility":
                conditions["per_base_seed_senior_recall_safety"] = all(
                    value
                    >= float(gate["utility_minimum_per_base_seed_senior_recall_delta"])
                    for value in per_seed_senior.values()
                )
            contrasts[kind] = {
                "comparator": comparator,
                "summary": contrast_summary(values),
                "per_base_seed_mean_delta": per_seed_means,
                "split_cell_nonnegative": int((split_values >= 0).sum()),
                "split_cell_worst": float(split_values.min()),
                "conditions": conditions,
                "gate_passed": bool(all(conditions.values())),
            }
        group_results[group] = {
            "contrasts": contrasts,
            "per_base_seed_senior_recall_delta_vs_A0": per_seed_senior,
            "source_supported": bool(
                contrasts["information"]["gate_passed"]
                and contrasts["utility"]["gate_passed"]
            ),
        }
    combo_columns = (
        "G12_real_minus_G1_real_macro_f1",
        "G12_real_minus_G2_real_macro_f1",
        "G12_redundancy_interaction_macro_f1",
    )
    combo_descriptive = {
        column: contrast_summary(seed_repeats[column]) for column in combo_columns
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
        "independence_note": "Fold deltas and repeated animal occurrences are descriptive, not independent samples.",
        "pipeline_seed_repeat_means": pipeline_means,
        "group_results": group_results,
        "combo_descriptive_only": combo_descriptive,
        "any_source_supported": any(
            value["source_supported"] for value in group_results.values()
        ),
        "fold_results": fold_results,
        "seed_repeat_results": seed_repeat_results,
        "split_cell_results": split_cells.to_dict(orient="records"),
        "pooled_validation": pooled_metrics,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.director_authorized:
        raise RuntimeError(
            "IDEA-082 formal run is blocked until the research director explicitly authorizes it; pass --director-authorized only after that decision."
        )
    idea068.configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
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
        "director_authorized": True,
        "outer_test_accessed": False,
        "pipelines": list(PIPELINES),
        "model": protocol["model"],
        "feature_groups": protocol["feature_groups"],
        "shuffle_control": protocol["shuffle_control"],
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
            raise RuntimeError("Existing IDEA-082 run manifest differs")
    else:
        write_json(manifest_path, manifest)
    completed: list[dict[str, Any]] = []
    for base_seed in BASE_SEEDS:
        for repeat in protocol["model"]["repeats"]:
            for fold in protocol["model"]["folds"]:
                indices = role_cell_indices(store, roles, repeat, fold)
                full_seed = idea068.idea051.reference.historical.full_seed(
                    int(base_seed), repeat, fold
                )
                matrices: dict[str, np.ndarray] = {PIPELINES[0]: features}
                shuffle_audits: dict[str, Any] = {}
                for pipeline in PIPELINES[1:]:
                    matrix, audit = prepare_pipeline_features(
                        pipeline,
                        features,
                        indices["train"],
                        indices["validation"],
                        repeat,
                        fold,
                    )
                    matrices[pipeline] = matrix
                    if audit is not None:
                        shuffle_audits[pipeline] = audit
                initialization = initial_model_audit(
                    protocol,
                    store,
                    matrices,
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
                            matrices[pipeline],
                            indices["train"],
                            indices["validation"],
                            device,
                            full_seed,
                        )
                        if audit["model"]["trainable_parameters"] != EXPECTED_PARAMETERS[pipeline]:
                            raise RuntimeError("IDEA-082 trained parameter audit mismatch")
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
                            "shuffle_audit": shuffle_audits.get(pipeline),
                            "train_calls": int(len(indices["train"])),
                            "validation_calls": int(len(indices["validation"])),
                            "validation_cats": int(animals["cat_id"].nunique()),
                            "validation_animal_predictions": animal_path.relative_to(REPO_ROOT).as_posix(),
                            "validation_animal_sha256": idea068.sha256(animal_path),
                            "validation_call_predictions": call_path.relative_to(REPO_ROOT).as_posix(),
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
                            raise RuntimeError("IDEA-082 paired batch order differs")
    summary = aggregate(completed, protocol)
    summary_path = run_root / "initial_evaluation_summary.json"
    summary_bytes = canonical_json_bytes(summary)
    if summary_path.exists():
        if not args.resume:
            raise FileExistsError(summary_path)
        if read_json(summary_path) != summary or summary_path.read_bytes() != summary_bytes:
            raise RuntimeError("IDEA-082 resumed aggregate differs")
    else:
        write_json(summary_path, summary)
    compact = {
        "status": "complete",
        "completed_fits": len(completed),
        "expected_fits": 324,
        "source_supported": {
            group: value["source_supported"]
            for group, value in summary["group_results"].items()
        },
        "any_source_supported": summary["any_source_supported"],
        "outer_test_accessed": False,
    }
    compact_path = run_root / "run_summary.json"
    if compact_path.exists():
        if read_json(compact_path) != compact or compact_path.read_bytes() != canonical_json_bytes(compact):
            raise RuntimeError("IDEA-082 resumed compact summary differs")
    else:
        write_json(compact_path, compact)
    return summary


def main() -> None:
    args = parse_args()
    result = preflight(args) if args.stage == "preflight" else run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
