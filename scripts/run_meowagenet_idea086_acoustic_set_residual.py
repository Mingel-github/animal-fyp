"""Run IDEA-086 permutation-invariant acoustic set residual screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_idea068_age_sensitive_ast as idea068  # noqa: E402
import run_meowagenet_idea071_bounded_dual_path_fusion as idea071  # noqa: E402
import run_meowagenet_idea084_grouped_dual_branch_acoustic_residual as idea084  # noqa: E402
import run_meowagenet_idea085_acoustic_temporal_residual as idea085  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT / "configs" / "protocol" / "meowagenet_idea086_acoustic_set_residual_v1.json"
)
PIPELINE = "SET1_acoustic_set_residual"
REFERENCE_PIPELINES = idea085.PIPELINES
ALL_PIPELINES = (PIPELINE, *REFERENCE_PIPELINES)
BASE_SEEDS = idea085.BASE_SEEDS
EXPECTED_PARAMETERS = 108_323
EXPECTED_BRANCH_PARAMETERS = 9_248
COMPARISONS = {
    "SET1_minus_A0": (PIPELINE, REFERENCE_PIPELINES[0]),
    "SET1_minus_C1": (PIPELINE, REFERENCE_PIPELINES[1]),
    "SET1_minus_T1": (PIPELINE, REFERENCE_PIPELINES[2]),
    "SET1_minus_J1": (PIPELINE, REFERENCE_PIPELINES[3]),
}
GATED_COMPARISONS = ("SET1_minus_A0", "SET1_minus_C1")
CLASS_NAMES = ("kitten", "adult", "senior")
PROBABILITY_COLUMNS = tuple(idea068.PROBABILITY_COLUMNS)
_ACTIVE_TRAJECTORIES: idea085.TemporalRaggedCache | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "run"), required=True)
    parser.add_argument(
        "--output-subdir", default="meowagenet_idea086_acoustic_set_residual_v1"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--director-authorized", action="store_true")
    parser.add_argument("--max-cells", type=int, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def sha256(path: Path) -> str:
    return idea068.sha256(path)


def set_active_trajectories(store: idea085.TemporalRaggedCache) -> None:
    global _ACTIVE_TRAJECTORIES
    _ACTIVE_TRAJECTORIES = store
    idea085.set_active_trajectories(store)


class SetResidualClassifier(idea071.PerturbationAuditMixin, torch.nn.Module):
    def __init__(
        self,
        ast_mean: np.ndarray,
        ast_scale: np.ndarray,
        trajectories: idea085.TemporalRaggedCache,
        train_indices: np.ndarray,
        audit_embeddings: np.ndarray,
        audit_indices: np.ndarray,
        dropout: float,
        permutation_atol: float,
        permutation_rtol: float,
    ) -> None:
        super().__init__()
        self.trajectories = trajectories
        self.permutation_atol = permutation_atol
        self.permutation_rtol = permutation_rtol
        safe_ast_scale = np.where(ast_scale > 1.0e-12, ast_scale, 1.0).astype(
            np.float32
        )
        self.register_buffer("ast_mean", torch.from_numpy(ast_mean.astype(np.float32)))
        self.register_buffer("ast_scale", torch.from_numpy(safe_ast_scale))
        # Identical construction order for the common frozen-AST head.
        self.ast_linear = torch.nn.Linear(768, 128)
        self.relu = torch.nn.ReLU()
        self.batch_norm = torch.nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01)
        self.dropout = torch.nn.Dropout(dropout)
        self.output = torch.nn.Linear(128, 3)

        median, mean, scale = trajectories.training_statistics(train_indices)
        self.register_buffer("set_median", torch.from_numpy(median))
        self.register_buffer("set_mean", torch.from_numpy(mean))
        self.register_buffer("set_scale", torch.from_numpy(scale))
        self.frame_linear1 = torch.nn.Linear(12, 48)
        self.frame_linear2 = torch.nn.Linear(48, 48)
        self.set_projection = torch.nn.Linear(48, 128)
        torch.nn.init.zeros_(self.set_projection.weight)
        torch.nn.init.zeros_(self.set_projection.bias)
        self._audit_indices = tuple(int(value) for value in audit_indices.tolist())
        self._audit_embeddings = np.asarray(audit_embeddings, dtype=np.float32).copy()
        self.reset_perturbation_audit()

    @staticmethod
    def _lookup_indices(packed_features: torch.Tensor) -> list[int]:
        if packed_features.ndim != 2 or packed_features.shape[1] < 21:
            raise ValueError("IDEA-086 packed features are missing lookup column")
        lookup = packed_features[:, 20]
        rounded = torch.round(lookup).to(dtype=torch.long)
        if not torch.equal(lookup, rounded.to(dtype=lookup.dtype)):
            raise ValueError("IDEA-086 call lookup is not an exact integer")
        return [int(value) for value in rounded.detach().cpu().tolist()]

    def _ordered_raw(self, call_index: int, order: str) -> np.ndarray:
        raw = self.trajectories.sequence(call_index, shuffled=False)
        if order == "native":
            return raw
        if order == "reverse":
            return np.ascontiguousarray(raw[::-1])
        if order == "idea085_j1":
            return self.trajectories.sequence(call_index, shuffled=True)
        raise ValueError(order)

    def _prepared_sequences(
        self, packed_features: torch.Tensor, order: str = "native"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences: list[torch.Tensor] = []
        lengths: list[int] = []
        for call_index in self._lookup_indices(packed_features):
            raw = torch.from_numpy(
                np.ascontiguousarray(self._ordered_raw(call_index, order))
            ).to(device=packed_features.device, dtype=torch.float32)
            finite = torch.isfinite(raw)
            imputed = torch.where(finite, raw, self.set_median)
            standardized = (imputed - self.set_mean) / self.set_scale
            # Finite indicators stay on the same row as all six raw channels.
            model_input = torch.cat(
                [standardized, finite.to(dtype=standardized.dtype)], dim=1
            )
            sequences.append(model_input)
            lengths.append(len(model_input))
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=0.0
        )
        length_tensor = torch.tensor(lengths, device=padded.device, dtype=torch.long)
        positions = torch.arange(padded.shape[1], device=padded.device)[None, :]
        return padded, positions < length_tensor[:, None]

    def set_context(
        self, packed_features: torch.Tensor, order: str = "native"
    ) -> torch.Tensor:
        padded, mask = self._prepared_sequences(packed_features, order)
        encoded = torch.nn.functional.gelu(self.frame_linear1(padded))
        encoded = torch.nn.functional.gelu(self.frame_linear2(encoded))
        mask_float = mask[:, :, None].to(dtype=encoded.dtype)
        denominator = mask_float.sum(dim=1).clamp_min(1.0)
        return (encoded * mask_float).sum(dim=1) / denominator

    def set_output(
        self, packed_features: torch.Tensor, order: str = "native"
    ) -> torch.Tensor:
        return self.set_projection(self.set_context(packed_features, order))

    def _forward_order(
        self,
        ast_embeddings: torch.Tensor,
        packed_features: torch.Tensor,
        order: str,
        record: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ast = (ast_embeddings - self.ast_mean) / self.ast_scale
        hidden = self.relu(self.ast_linear(ast))
        residual = (
            idea071.CAP
            * idea071.hidden_rms(hidden)
            * torch.tanh(self.set_output(packed_features, order))
        )
        if record:
            self.record_perturbation(hidden, residual)
        logits = self.output(self.dropout(self.batch_norm(hidden + residual)))
        return logits, residual

    def forward(
        self, ast_embeddings: torch.Tensor, packed_features: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_order(ast_embeddings, packed_features, "native", True)[0]

    def trained_permutation_probe(self) -> dict[str, Any]:
        device = self.ast_mean.device
        packed = torch.zeros((len(self._audit_indices), 21), device=device)
        packed[:, 20] = torch.tensor(
            self._audit_indices, device=device, dtype=packed.dtype
        )
        embeddings = torch.from_numpy(self._audit_embeddings).to(device=device)
        return permutation_invariance_audit(
            self, embeddings, packed, override_projection=False
        )

    def audit(self) -> dict[str, Any]:
        projection_norm = float(self.set_projection.weight.detach().norm().cpu())
        result = {
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in self.parameters())
            ),
            "set_branch_parameters": EXPECTED_BRANCH_PARAMETERS,
            "fusion": "bounded_rms_relative_permutation_invariant_acoustic_set_residual",
            "cap": idea071.CAP,
            "raw_channels": 6,
            "finite_indicators": 6,
            "frame_hidden_units": 48,
            "time_convolution": False,
            "position_encoding": False,
            "attention": False,
            "variance_pooling": False,
            "call_lookup_used_only_for_ragged_retrieval": True,
            "time_or_identity_feature_entered_encoder": False,
            "projection_weight_norm": projection_norm,
            **self.perturbation_audit(),
        }
        if projection_norm > 0.0:
            result["trained_checkpoint_permutation_invariance"] = (
                self.trained_permutation_probe()
            )
        return result


def audit_probe_indices(
    trajectories: idea085.TemporalRaggedCache, indices: np.ndarray
) -> np.ndarray:
    candidates = np.asarray(indices, dtype=np.int64)
    lengths = trajectories.lengths[candidates]
    selected = [int(candidates[int(np.argmin(lengths))])]
    longest = int(candidates[int(np.argmax(lengths))])
    if longest not in selected:
        selected.append(longest)
    middle = int(candidates[len(candidates) // 2])
    if middle not in selected:
        selected.append(middle)
    return np.asarray(selected, dtype=np.int64)


def build_model(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    packed_features: np.ndarray,
    train_indices: np.ndarray,
) -> torch.nn.Module:
    if pipeline != PIPELINE:
        raise ValueError(pipeline)
    if _ACTIVE_TRAJECTORIES is None:
        raise RuntimeError("IDEA-086 trajectory cache is not active")
    embeddings = store.frozen_embeddings[train_indices]
    probes = audit_probe_indices(_ACTIVE_TRAJECTORIES, train_indices)
    return SetResidualClassifier(
        ast_mean=embeddings.mean(axis=0),
        ast_scale=embeddings.std(axis=0),
        trajectories=_ACTIVE_TRAJECTORIES,
        train_indices=train_indices,
        audit_embeddings=store.frozen_embeddings[probes],
        audit_indices=probes,
        dropout=float(protocol["fixed_training"]["dropout"]),
        permutation_atol=float(protocol["determinism"]["permutation_invariance_atol"]),
        permutation_rtol=float(protocol["determinism"]["permutation_invariance_rtol"]),
    )


def load_features(
    protocol: dict[str, Any], call_ids: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    packed, summary = idea085.load_features(protocol, call_ids)
    if idea085._ACTIVE_TRAJECTORIES is None:
        raise RuntimeError("IDEA-086 failed to activate trajectory cache")
    set_active_trajectories(idea085._ACTIVE_TRAJECTORIES)
    return packed, summary


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
    idea068.predict = idea084.predict_with_perturbation_reset
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


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea086-acoustic-set-residual-v1":
        raise RuntimeError("Unexpected IDEA-086 protocol")
    if protocol.get("status") != "locked_for_cpu_preflight_before_initial_evaluation":
        raise RuntimeError("IDEA-086 protocol is not result-blind locked")
    data = protocol["data"]
    for key in (
        "roles_sha256",
        "frozen_embedding_sha256",
        "summary_feature_sha256",
        "trajectory_sha256",
    ):
        path_key = key.replace("_sha256", "_path")
        path = REPO_ROOT / data[path_key]
        if not path.is_file() or sha256(path) != data[key]:
            raise RuntimeError(f"IDEA-086 locked data checksum changed: {key}")
    reuse = protocol["reuse"]
    for stem in (
        "idea085_protocol", "idea085_runner", "idea085_tests", "idea085_summary",
        "idea085_run_manifest", "idea085_independent_results_audit",
    ):
        path = REPO_ROOT / reuse[f"{stem}_path"]
        if not path.is_file() or sha256(path) != reuse[f"{stem}_sha256"]:
            raise RuntimeError(f"IDEA-086 read-only dependency changed: {stem}")
    old_protocol = read_json(REPO_ROOT / reuse["idea085_protocol_path"])
    # Execute the full parent verification, including its 068/071/082/084 chain.
    idea085.verify_protocol(old_protocol)
    model = protocol["model"]
    if model["pipeline"] != PIPELINE or tuple(model["reference_pipelines"]) != REFERENCE_PIPELINES:
        raise RuntimeError("IDEA-086 pipeline identities changed")
    if tuple(model["base_seeds"]) != BASE_SEEDS or model["repeats"] != [0, 1, 2] or model["folds"] != [0, 1, 2, 3]:
        raise RuntimeError("IDEA-086 paired matrix changed")
    if int(model["new_fits"]) != 36 or int(model["reused_reference_fits"]) != 144:
        raise RuntimeError("IDEA-086 fit budget changed")
    if int(model["set_branch_parameters"]) != EXPECTED_BRANCH_PARAMETERS or int(model["set_total_trainable_parameters"]) != EXPECTED_PARAMETERS:
        raise RuntimeError("IDEA-086 parameter lock changed")
    if float(model["cap"]) != idea071.CAP or float(model["rms_epsilon"]) != idea071.RMS_EPSILON:
        raise RuntimeError("IDEA-086 residual cap changed")
    acoustic = protocol["acoustic_set"]
    prohibited = ("time_convolution", "position_encoding", "attention", "variance_pooling")
    if any(acoustic[key] is not False for key in prohibited):
        raise RuntimeError("IDEA-086 prohibited temporal/set operator enabled")
    if protocol["classification_gate"]["gated_comparisons"] != list(GATED_COMPARISONS):
        raise RuntimeError("IDEA-086 gate scope changed")
    shared_training = (
        "ast_head", "age_hidden_units", "dropout", "optimizer", "learning_rate",
        "optimizer_epsilon", "gradient_clip", "maximum_epochs",
        "early_stopping_patience", "cat_batch_size", "loss",
        "checkpoint_selection", "temporal_preprocessing",
        "post_build_seed_offset", "paired_batch_order",
    )
    for key in shared_training:
        if protocol["fixed_training"][key] != old_protocol["fixed_training"][key]:
            raise RuntimeError(f"IDEA-086 changed locked IDEA-085 training field: {key}")
    for key, value in old_protocol["determinism"].items():
        if protocol["determinism"].get(key) != value:
            raise RuntimeError(f"IDEA-086 changed locked determinism field: {key}")
    dependencies = protocol["dependencies"]
    self_paths = {
        "idea086_runner_sha256": Path(__file__).resolve(),
        "idea086_tests_sha256": REPO_ROOT / "tests" / "test_idea086_acoustic_set_residual.py",
    }
    for key, path in self_paths.items():
        if dependencies.get(key) != sha256(path):
            raise RuntimeError(f"IDEA-086 self checksum changed: {key}")
    if protocol["execution_gate"].get("gpu_authorized") is not False or protocol["execution_gate"].get("outer_test_accessed") is not False:
        raise RuntimeError("IDEA-086 execution boundary changed")


def old_fit_path(pipeline: str, base_seed: int, repeat: int, fold: int) -> Path:
    return (
        REPO_ROOT
        / "runs"
        / "meowagenet_idea085_acoustic_temporal_residual_v1"
        / "fits"
        / pipeline
        / f"base_seed_{base_seed}"
        / f"repeat_{repeat}"
        / f"fold_{fold}"
        / "fit_summary.json"
    )


def validate_reused_fit(
    fit: dict[str, Any], pipeline: str, base_seed: int, repeat: int, fold: int
) -> None:
    full_seed = idea068.idea051.reference.historical.full_seed(base_seed, repeat, fold)
    idea085.validate_completed_fit(fit, pipeline, base_seed, full_seed, repeat, fold)
    if fit.get("outer_test_accessed") is not False:
        raise RuntimeError("IDEA-086 reused fit touched outer test")


def validate_prediction_probabilities(frame: pd.DataFrame, identity: str) -> None:
    required = {"true_label", *PROBABILITY_COLUMNS}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"{identity}: prediction columns incomplete")
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError(f"{identity}: non-finite probability")
    if np.any(probabilities < -1.0e-7) or np.any(probabilities > 1.0 + 1.0e-7):
        raise RuntimeError(f"{identity}: probability outside [0,1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0):
        raise RuntimeError(f"{identity}: probabilities do not sum to one")
    labels = frame["true_label"].to_numpy(dtype=np.int64)
    if np.any(labels < 0) or np.any(labels > 2):
        raise RuntimeError(f"{identity}: label outside locked classes")
    if "predicted_label" in frame.columns and not np.array_equal(
        frame["predicted_label"].to_numpy(dtype=np.int64), probabilities.argmax(axis=1)
    ):
        raise RuntimeError(f"{identity}: predicted label is not probability argmax")


def reconstruct_animals_from_calls(calls: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cat_id, group in calls.groupby("cat_id", sort=True):
        labels = group["true_label"].to_numpy(dtype=np.int64)
        if len(np.unique(labels)) != 1:
            raise RuntimeError(f"IDEA-086 inconsistent call labels for cat {cat_id}")
        probabilities = group[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64).mean(axis=0)
        rows.append(
            {
                "cat_id": str(cat_id), "true_label": int(labels[0]),
                "call_count": int(len(group)),
                **{column: float(probabilities[index]) for index, column in enumerate(PROBABILITY_COLUMNS)},
                "predicted_label": int(probabilities.argmax()),
            }
        )
    return pd.DataFrame(rows).sort_values("cat_id").reset_index(drop=True)


def compare_animal_reconstruction(
    saved: pd.DataFrame, reconstructed: pd.DataFrame, identity: str
) -> None:
    saved = saved.sort_values("cat_id").reset_index(drop=True)
    for column in ("cat_id", "true_label", "call_count", "predicted_label"):
        if column not in saved or not np.array_equal(
            saved[column].to_numpy(), reconstructed[column].to_numpy()
        ):
            raise RuntimeError(f"{identity}: animal reconstruction differs in {column}")
    if not np.allclose(
        saved[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64),
        reconstructed[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64),
        atol=1.0e-12, rtol=0.0,
    ):
        raise RuntimeError(f"{identity}: reconstructed animal probabilities differ")


def director_protected_baseline(protocol: dict[str, Any]) -> dict[str, Any]:
    run_root = REPO_ROOT / protocol["reuse"]["idea085_run_root"]
    protected = [path for path in run_root.rglob("*") if path.is_file()]
    protected.extend(
        REPO_ROOT / relative
        for relative in (
            "configs/protocol/meowagenet_idea085_acoustic_temporal_residual_v1.json",
            "scripts/run_meowagenet_idea085_acoustic_temporal_residual.py",
            "tests/test_idea085_acoustic_temporal_residual.py",
            "reports/70_IDEA-085_acoustic_temporal_residual_results.md",
            "metadata/experiments/meowagenet_idea085_acoustic_temporal_residual_v1_results.json",
            "reports/58_Formal_Research_Summary_v3.md",
            "metadata/experiments/meowagenet_formal_research_summary_v3.json",
        )
    )
    if any(not path.is_file() for path in protected):
        raise RuntimeError("IDEA-086 director protected baseline file is missing")
    # Match the director's Windows Sort-Object FullName order exactly, then freeze
    # the resulting relative-path/hash lines into the reuse manifest.
    ordered = sorted(protected, key=lambda path: str(path.resolve()).lower())
    entries = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256(path),
        }
        for path in ordered
    ]
    lines = "\n".join(f"{item['path']}:{item['sha256']}" for item in entries)
    digest = hashlib.sha256(lines.encode("utf-8")).hexdigest()
    expected_count = int(protocol["reuse"]["director_readonly_baseline_files"])
    expected_digest = protocol["reuse"]["director_readonly_baseline_sha256"]
    if len(entries) != expected_count or digest != expected_digest:
        raise RuntimeError(
            f"IDEA-086 director protected baseline changed: count={len(entries)} digest={digest}"
        )
    return {"count": len(entries), "sha256": digest, "entries": entries}


def collect_reused_fits(
    protocol: dict[str, Any], store: Any, roles: pd.DataFrame, verify_rows: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evidence_path = REPO_ROOT / protocol["reuse"]["idea085_independent_results_audit_path"]
    evidence = read_json(evidence_path)["integrity"]["fit_evidence"]
    entries: list[dict[str, str]] = []
    fits: list[dict[str, Any]] = []
    batch_history_checks = 0
    animal_reconstructions = 0
    for base_seed in BASE_SEEDS:
        for repeat in protocol["model"]["repeats"]:
            for fold in protocol["model"]["folds"]:
                indices = idea084.role_cell_indices(store, roles, repeat, fold)
                expected_call_ids = set(store.call_ids[indices["validation"]].astype(str))
                expected_cats = set(store.cat_ids[indices["validation"]].astype(str))
                for pipeline in REFERENCE_PIPELINES:
                    path = old_fit_path(pipeline, base_seed, repeat, fold)
                    if not path.is_file():
                        raise RuntimeError(f"IDEA-086 missing reused fit: {path}")
                    fit = read_json(path)
                    validate_reused_fit(fit, pipeline, base_seed, repeat, fold)
                    evidence_key = path.relative_to(REPO_ROOT).as_posix()
                    if evidence_key not in evidence:
                        raise RuntimeError(f"IDEA-086 fit missing from frozen audit evidence: {evidence_key}")
                    frozen = evidence[evidence_key]
                    if sha256(path) != frozen["fit_summary_sha256"]:
                        raise RuntimeError(f"IDEA-086 reused fit differs from frozen audit: {evidence_key}")
                    fits.append(fit)
                    entries.append({"path": path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256(path)})
                    for prefix in ("validation_animal", "validation_call"):
                        prediction_path = REPO_ROOT / fit[f"{prefix}_predictions"]
                        frozen_key = f"{prefix}_sha256"
                        if sha256(prediction_path) != frozen[frozen_key]:
                            raise RuntimeError(f"IDEA-086 reused prediction differs from frozen audit: {prediction_path}")
                        entries.append({"path": prediction_path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256(prediction_path)})
                    if verify_rows:
                        calls = pd.read_csv(REPO_ROOT / fit["validation_call_predictions"], dtype={"call_id": str, "cat_id": str})
                        animals = pd.read_csv(REPO_ROOT / fit["validation_animal_predictions"], dtype={"cat_id": str})
                        validate_prediction_probabilities(calls, f"{evidence_key}/calls")
                        validate_prediction_probabilities(animals, f"{evidence_key}/animals")
                        if calls["call_id"].duplicated().any() or animals["cat_id"].duplicated().any():
                            raise RuntimeError("IDEA-086 reused predictions contain duplicate identities")
                        if set(calls["call_id"].astype(str)) != expected_call_ids or set(animals["cat_id"].astype(str)) != expected_cats:
                            raise RuntimeError("IDEA-086 reused prediction role identity mismatch")
                        lookup = {str(call): index for index, call in enumerate(store.call_ids.astype(str))}
                        expected_labels = np.asarray([store.labels[lookup[str(call)]] for call in calls["call_id"]], dtype=np.int64)
                        if not np.array_equal(expected_labels, calls["true_label"].to_numpy(dtype=np.int64)):
                            raise RuntimeError("IDEA-086 reused call labels changed")
                        reconstructed = reconstruct_animals_from_calls(calls)
                        compare_animal_reconstruction(animals, reconstructed, evidence_key)
                        animal_reconstructions += 1
                        history = fit["audit"]["history"]
                        if not history:
                            raise RuntimeError("IDEA-086 reused fit has empty history")
                        batch_history_checks += 1
    entries.sort(key=lambda item: item["path"])
    if len(fits) != 144 or len(entries) != 432:
        raise RuntimeError("IDEA-086 reused artifact matrix incomplete")
    manifest_core = {
        "source_experiment": "meowagenet-idea085-acoustic-temporal-residual-v1",
        "fit_summaries": 144,
        "prediction_files": 288,
        "entries": entries,
    }
    content_sha = hashlib.sha256(canonical_json_bytes(manifest_core)).hexdigest()
    manifest = {
        **manifest_core,
        "content_sha256": content_sha,
        "previous_second_resume_snapshot_sha256": protocol["reuse"]["previous_second_resume_snapshot_sha256"],
        "frozen_per_file_baseline_path": protocol["reuse"]["idea085_independent_results_audit_path"],
        "frozen_per_file_baseline_sha256": protocol["reuse"]["idea085_independent_results_audit_sha256"],
        "all_432_files_match_frozen_independent_audit": True,
        "all_individual_prediction_hashes_match_fit_summaries": True,
        "role_and_label_rows_checked": bool(verify_rows),
        "batch_histories_present": batch_history_checks,
        "animal_predictions_reconstructed_from_calls": animal_reconstructions,
        "outer_test_accessed": False,
    }
    protected = director_protected_baseline(protocol)
    manifest["director_protected_baseline_files"] = protected["count"]
    manifest["director_protected_baseline_sha256"] = protected["sha256"]
    manifest["director_protected_entries"] = protected["entries"]
    return fits, manifest


def batch_order_audit(
    protocol: dict[str, Any], store: Any, roles: pd.DataFrame
) -> dict[str, Any]:
    checks = 0
    for base_seed in BASE_SEEDS:
        for repeat in protocol["model"]["repeats"]:
            for fold in protocol["model"]["folds"]:
                indices = idea084.role_cell_indices(store, roles, repeat, fold)
                dataset = idea068.idea051.CatSetDataset(store, indices["train"])
                loader = idea068.idea051.build_set_loader(
                    dataset, int(protocol["fixed_training"]["cat_batch_size"]), True,
                    idea068.idea051.reference.historical.full_seed(base_seed, repeat, fold),
                )
                cats: list[str] = []
                calls: list[int] = []
                for batch in loader:
                    cats.extend(str(value) for value in batch["cat_ids"])
                    calls.extend(int(value) for value in batch["call_indices"].numpy())
                expected_cat = hashlib.sha256("\n".join(cats).encode("utf-8")).hexdigest()
                expected_calls = hashlib.sha256(np.sort(np.asarray(calls, dtype="<i8")).tobytes()).hexdigest()
                for pipeline in REFERENCE_PIPELINES:
                    fit = read_json(old_fit_path(pipeline, base_seed, repeat, fold))
                    first = fit["audit"]["history"][0]["train_audit"]
                    if first["cat_order_sha256"] != expected_cat or first["call_coverage_sha256"] != expected_calls:
                        raise RuntimeError("IDEA-086 post-build batch compatibility failed")
                    checks += 1
    return {"old_fit_first_epochs_checked": checks, "same_seed_cat_order_and_call_coverage": True}


def initial_compatibility_audit(
    protocol: dict[str, Any], store: Any, features: np.ndarray, train_indices: np.ndarray,
    probe_indices: np.ndarray, seed: int,
) -> dict[str, Any]:
    old_protocol = read_json(REPO_ROOT / protocol["reuse"]["idea085_protocol_path"])
    idea068.idea051.reference.historical.set_seed(seed)
    a0 = idea085.build_model(REFERENCE_PIPELINES[0], old_protocol, store, features, train_indices).eval()
    idea068.idea051.reference.historical.set_seed(seed)
    t1 = idea085.build_model(REFERENCE_PIPELINES[2], old_protocol, store, features, train_indices).eval()
    idea068.idea051.reference.historical.set_seed(seed)
    set1 = build_model(PIPELINE, protocol, store, features, train_indices).eval()
    if not isinstance(set1, SetResidualClassifier) or not isinstance(t1, idea085.TemporalResidualClassifier):
        raise RuntimeError("IDEA-086 compatibility model type changed")
    params = int(sum(parameter.numel() for parameter in set1.parameters()))
    if params != EXPECTED_PARAMETERS:
        raise RuntimeError(f"IDEA-086 parameter mismatch: {params}")
    common_a0 = idea084.common_state_equal(a0, set1)
    common_t1 = idea084.common_state_equal(t1, set1)
    if not common_a0 or not common_t1:
        raise RuntimeError("IDEA-086 common AST head initial state differs")
    with torch.no_grad():
        emb = torch.from_numpy(store.frozen_embeddings[probe_indices])
        packed = torch.from_numpy(features[probe_indices])
        a0_logits = a0(emb, packed)
        set_logits = set1(emb, packed)
    max_difference = float(torch.max(torch.abs(a0_logits - set_logits)))
    if max_difference != 0.0:
        raise RuntimeError("IDEA-086 zero-init logits differ from A0")
    stats_equal = all(
        torch.equal(getattr(t1, old), getattr(set1, new))
        for old, new in (
            ("temporal_median", "set_median"),
            ("temporal_mean", "set_mean"),
            ("temporal_scale", "set_scale"),
        )
    )
    if not stats_equal:
        raise RuntimeError("IDEA-086 input preprocessing differs from IDEA-085")
    offset = int(protocol["fixed_training"]["post_build_seed_offset"])
    idea068.idea051.reference.historical.set_seed(seed + offset)
    left_rng = torch.rand(16)
    idea068.idea051.reference.historical.set_seed(seed)
    _ = build_model(PIPELINE, protocol, store, features, train_indices)
    idea068.idea051.reference.historical.set_seed(seed + offset)
    right_rng = torch.rand(16)
    if not torch.equal(left_rng, right_rng):
        raise RuntimeError("IDEA-086 post-build RNG reset differs")
    return {
        "parameters": params,
        "branch_parameters": EXPECTED_BRANCH_PARAMETERS,
        "common_AST_state_equal_to_A0": common_a0,
        "common_AST_state_equal_to_T1": common_t1,
        "zero_init_max_logit_difference_vs_A0": max_difference,
        "projection_zero_initialized": bool(
            torch.count_nonzero(set1.set_projection.weight) == 0
            and torch.count_nonzero(set1.set_projection.bias) == 0
        ),
        "train_only_preprocessing_equal_to_IDEA085_T1": stats_equal,
        "post_build_rng_reset_probe_equal": True,
        "C1_branch_stacked": False,
    }


def _max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).detach().cpu())


def permutation_invariance_audit(
    model: SetResidualClassifier,
    embeddings: torch.Tensor,
    packed: torch.Tensor,
    override_projection: bool,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    saved_weight = model.set_projection.weight.detach().clone()
    saved_bias = model.set_projection.bias.detach().clone()
    original_projection_norm = float(saved_weight.norm().cpu())
    if override_projection:
        with torch.no_grad():
            values = torch.linspace(
                -0.03, 0.03, model.set_projection.weight.numel(),
                device=model.set_projection.weight.device,
                dtype=model.set_projection.weight.dtype,
            )
            model.set_projection.weight.copy_(values.reshape_as(model.set_projection.weight))
            model.set_projection.bias.copy_(
                torch.linspace(-0.01, 0.01, 128, device=model.set_projection.bias.device)
            )
    active_projection_norm = float(model.set_projection.weight.detach().norm().cpu())
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for order in ("native", "reverse", "idea085_j1"):
            context = model.set_context(packed, order)
            logits, residual = model._forward_order(embeddings, packed, order, False)
            outputs[order] = {"context": context, "residual": residual, "logits": logits}
    differences: dict[str, float] = {}
    passed = True
    for order in ("reverse", "idea085_j1"):
        for name in ("context", "residual", "logits"):
            key = f"{order}_{name}_max_abs_difference"
            differences[key] = _max_difference(outputs["native"][name], outputs[order][name])
            passed = passed and torch.allclose(
                outputs["native"][name], outputs[order][name],
                atol=model.permutation_atol, rtol=model.permutation_rtol,
            )
    context_norm = float(outputs["native"]["context"].norm().detach().cpu())
    residual_norm = float(outputs["native"]["residual"].norm().detach().cpu())
    if context_norm <= 0.0 or active_projection_norm <= 0.0 or residual_norm <= 0.0:
        passed = False
    if override_projection:
        with torch.no_grad():
            model.set_projection.weight.copy_(saved_weight)
            model.set_projection.bias.copy_(saved_bias)
    if was_training:
        model.train()
    if not passed:
        raise RuntimeError(f"IDEA-086 permutation invariance failed: {differences}")
    lengths = [int(model.trajectories.lengths[index]) for index in model._lookup_indices(packed)]
    return {
        "passed": True,
        "atol": model.permutation_atol,
        "rtol": model.permutation_rtol,
        "orders": ["native", "reverse", "idea085_j1"],
        "mixed_lengths": len(set(lengths)) > 1,
        "lengths": lengths,
        "nonzero_frame_encoder_norm": float(model.frame_linear1.weight.detach().norm().cpu()),
        "projection_overridden_nonzero": override_projection,
        "original_projection_weight_norm": original_projection_norm,
        "active_probe_projection_weight_norm": active_projection_norm,
        "native_context_norm": context_norm,
        "native_residual_norm": residual_norm,
        **differences,
    }


def gradient_audit(model: SetResidualClassifier, features: np.ndarray) -> dict[str, float]:
    packed = torch.from_numpy(features)
    model.zero_grad(set_to_none=True)
    model.set_output(packed).sum().backward()
    bias_grad = float(model.set_projection.bias.grad.norm())
    hidden1_zero = float(model.frame_linear1.weight.grad.norm())
    hidden2_zero = float(model.frame_linear2.weight.grad.norm())
    if bias_grad <= 0.0 or hidden1_zero != 0.0 or hidden2_zero != 0.0:
        raise RuntimeError("IDEA-086 zero-init gradient contract failed")
    with torch.no_grad():
        model.set_projection.weight.fill_(0.01)
    model.zero_grad(set_to_none=True)
    model.set_output(packed).sum().backward()
    hidden1_probe = float(model.frame_linear1.weight.grad.norm())
    hidden2_probe = float(model.frame_linear2.weight.grad.norm())
    if min(hidden1_probe, hidden2_probe) <= 0.0:
        raise RuntimeError("IDEA-086 frame encoder is not gradient reachable")
    return {
        "zero_init_projection_bias_grad_norm": bias_grad,
        "zero_init_frame_linear1_weight_grad_norm": hidden1_zero,
        "zero_init_frame_linear2_weight_grad_norm": hidden2_zero,
        "nonzero_projection_frame_linear1_weight_grad_norm": hidden1_probe,
        "nonzero_projection_frame_linear2_weight_grad_norm": hidden2_probe,
    }


def cap_audit(
    model: SetResidualClassifier, store: Any, features: np.ndarray, probe_indices: np.ndarray
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        model.set_projection.weight.fill_(100.0)
        model.set_projection.bias.fill_(100.0)
    model.reset_perturbation_audit()
    with torch.no_grad():
        model(
            torch.from_numpy(store.frozen_embeddings[probe_indices]),
            torch.from_numpy(features[probe_indices]),
        )
    audit = model.perturbation_audit()
    if audit["validation_relative_perturbation_max"] > idea071.CAP + 1.0e-5:
        raise RuntimeError("IDEA-086 residual cap failed")
    return audit


def padding_audit(model: SetResidualClassifier, features: np.ndarray) -> dict[str, Any]:
    shortest = int(np.argmin(model.trajectories.lengths))
    longest = int(np.argmax(model.trajectories.lengths))
    alone = torch.from_numpy(features[[shortest]])
    mixed = torch.from_numpy(features[[shortest, longest]])
    model.eval()
    with torch.no_grad():
        first = model.set_context(alone)[0]
        batched = model.set_context(mixed)[0]
    difference = _max_difference(first, batched)
    if difference > 1.0e-6:
        raise RuntimeError("IDEA-086 padding changed pooled context")
    return {
        "short_call_frames": int(model.trajectories.lengths[shortest]),
        "long_call_frames": int(model.trajectories.lengths[longest]),
        "eval_context_max_difference": difference,
        "padding_excluded": True,
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cpu" or torch.cuda.is_initialized():
        raise RuntimeError("IDEA-086 preflight is CPU-only and must not initialize CUDA")
    idea068.configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    store = idea068.idea051.reference.historical.idea019.load_feature_store()
    features, feature_summary = load_features(protocol, store.call_ids)
    roles = pd.read_csv(REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    reused, reuse_manifest = collect_reused_fits(protocol, store, roles, verify_rows=True)
    run_root = idea084.resolve_run_root(args.output_subdir)
    reuse_manifest_path = run_root / protocol["outputs"]["reuse_manifest"]
    write_json(reuse_manifest_path, reuse_manifest)
    first = idea084.role_cell_indices(store, roles, 0, 0)
    probes = first["validation"][: min(16, len(first["validation"]))]
    compatibility = initial_compatibility_audit(
        protocol, store, features, first["train"], probes, BASE_SEEDS[0]
    )
    idea068.idea051.reference.historical.set_seed(BASE_SEEDS[0])
    model = build_model(PIPELINE, protocol, store, features, first["train"])
    if not isinstance(model, SetResidualClassifier):
        raise RuntimeError("IDEA-086 model type changed")
    mixed_indices = audit_probe_indices(model.trajectories, first["validation"])
    packed = torch.from_numpy(features[mixed_indices])
    embeddings = torch.from_numpy(store.frozen_embeddings[mixed_indices])
    permutation = permutation_invariance_audit(
        model, embeddings, packed, override_projection=True
    )
    idea068.idea051.reference.historical.set_seed(BASE_SEEDS[0])
    gradient_model = build_model(PIPELINE, protocol, store, features, first["train"])
    gradient = gradient_audit(gradient_model, features[mixed_indices])
    idea068.idea051.reference.historical.set_seed(BASE_SEEDS[0])
    cap_model = build_model(PIPELINE, protocol, store, features, first["train"])
    cap = cap_audit(cap_model, store, features, probes)
    idea068.idea051.reference.historical.set_seed(BASE_SEEDS[0])
    padding_model = build_model(PIPELINE, protocol, store, features, first["train"])
    padding = padding_audit(padding_model, features)
    batch_order = batch_order_audit(protocol, store, roles)
    if len(reused) != 144 or torch.cuda.is_initialized():
        raise RuntimeError("IDEA-086 preflight boundary failed")
    result = {
        "status": "GO",
        "scope": "CPU preflight only; GPU remains unauthorized",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "reuse_manifest_sha256": sha256(reuse_manifest_path),
        "reuse_manifest_content_sha256": reuse_manifest["content_sha256"],
        "calls": 792,
        "cats": 111,
        "new_candidate_fits": 36,
        "reused_reference_fits": 144,
        "reused_prediction_files": 288,
        "base_seeds": list(BASE_SEEDS),
        "outer_test_predictions_or_metrics_accessed": False,
        "cuda_initialized": False,
        "device": "cpu",
        "trajectory_quality": feature_summary,
        "reuse_integrity": {key: value for key, value in reuse_manifest.items() if key != "entries"},
        "initial_compatibility": compatibility,
        "batch_order_compatibility": batch_order,
        "nontrivial_permutation_invariance": permutation,
        "gradient_reachability": gradient,
        "cap_saturation_probe": cap,
        "padding_invariance": padding,
        "lookup_column_entered_encoder": False,
        "prohibited_modules_present": False,
    }
    write_json(run_root / protocol["outputs"]["cpu_preflight"], result)
    return result


def require_matching_cpu_preflight(
    protocol: dict[str, Any], output_subdir: str
) -> tuple[dict[str, Any], Path, Path]:
    run_root = idea084.resolve_run_root(output_subdir)
    preflight_path = run_root / protocol["outputs"]["cpu_preflight"]
    reuse_path = run_root / protocol["outputs"]["reuse_manifest"]
    if not preflight_path.is_file() or not reuse_path.is_file():
        raise RuntimeError("IDEA-086 formal run requires CPU preflight and reuse manifest")
    result = read_json(preflight_path)
    expected = {
        "status": "GO",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "reuse_manifest_sha256": sha256(reuse_path),
        "new_candidate_fits": 36,
        "reused_reference_fits": 144,
        "outer_test_predictions_or_metrics_accessed": False,
        "cuda_initialized": False,
        "device": "cpu",
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"IDEA-086 CPU preflight mismatch: {key}")
    if protocol["execution_gate"].get("gpu_authorized") is not False:
        raise RuntimeError("IDEA-086 protocol GPU flag must remain false")
    return result, preflight_path, reuse_path


def validate_new_fit(
    fit: dict[str, Any], base_seed: int, full_seed: int, repeat: int, fold: int
) -> None:
    expected = {
        "status": "complete", "pipeline": PIPELINE, "base_seed": base_seed,
        "full_seed": full_seed, "repeat": repeat, "fold": fold,
        "outer_test_accessed": False,
    }
    for key, value in expected.items():
        if fit.get(key) != value:
            raise RuntimeError(f"IDEA-086 resume identity mismatch: {key}")
    for prefix in ("validation_animal", "validation_call"):
        path = REPO_ROOT / fit[f"{prefix}_predictions"]
        if not path.is_file() or sha256(path) != fit[f"{prefix}_sha256"]:
            raise RuntimeError(f"IDEA-086 resume prediction hash mismatch: {path}")


def load_animals(fit: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / fit["validation_animal_predictions"], dtype={"cat_id": str})


def metric_bundle(frame: pd.DataFrame) -> dict[str, Any]:
    return idea085.metric_bundle(frame)


def add_metrics(row: dict[str, Any], bundles: dict[str, Any]) -> None:
    for pipeline in ALL_PIPELINES:
        metrics = bundles[pipeline]["metrics"]
        row[f"{pipeline}_plain_accuracy"] = metrics["plain_accuracy"]
        row[f"{pipeline}_macro_f1"] = metrics["macro_f1"]
        row[f"{pipeline}_balanced_accuracy"] = metrics["balanced_accuracy"]
        row[f"{pipeline}_cross_entropy"] = bundles[pipeline]["cross_entropy"]
        row[f"{pipeline}_brier"] = bundles[pipeline]["brier"]
        for class_name in CLASS_NAMES:
            row[f"{pipeline}_{class_name}_recall"] = metrics["per_class"][class_name]["recall"]
    for name, (candidate, comparator) in COMPARISONS.items():
        for metric in ("plain_accuracy", "macro_f1", "balanced_accuracy"):
            row[f"{name}_{metric}"] = row[f"{candidate}_{metric}"] - row[f"{comparator}_{metric}"]
        row[f"{name}_cross_entropy_gain"] = row[f"{comparator}_cross_entropy"] - row[f"{candidate}_cross_entropy"]
        row[f"{name}_brier_gain"] = row[f"{comparator}_brier"] - row[f"{candidate}_brier"]
        for class_name in CLASS_NAMES:
            row[f"{name}_{class_name}_recall"] = row[f"{candidate}_{class_name}_recall"] - row[f"{comparator}_{class_name}_recall"]


def aggregate(
    new_fits: list[dict[str, Any]], reused_fits: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    all_fits = [*new_fits, *reused_fits]
    fold_rows: list[dict[str, Any]] = []
    seed_repeat_rows: list[dict[str, Any]] = []
    pooled_all: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in ALL_PIPELINES}
    for base_seed in BASE_SEEDS:
        for repeat in protocol["model"]["repeats"]:
            seed_frames = {pipeline: [] for pipeline in ALL_PIPELINES}
            for fold in protocol["model"]["folds"]:
                bundles: dict[str, Any] = {}
                for pipeline in ALL_PIPELINES:
                    fit = next(
                        item for item in all_fits
                        if item["pipeline"] == pipeline and item["base_seed"] == base_seed
                        and item["repeat"] == repeat and item["fold"] == fold
                    )
                    animals = load_animals(fit)
                    bundles[pipeline] = metric_bundle(animals)
                    tagged = animals.copy()
                    tagged["base_seed"] = base_seed
                    tagged["repeat"] = repeat
                    tagged["fold"] = fold
                    seed_frames[pipeline].append(tagged)
                    pooled_all[pipeline].append(tagged)
                row: dict[str, Any] = {"base_seed": base_seed, "repeat": repeat, "fold": fold}
                add_metrics(row, bundles)
                fold_rows.append(row)
            pooled = {pipeline: pd.concat(parts, ignore_index=True) for pipeline, parts in seed_frames.items()}
            row = {"base_seed": base_seed, "repeat": repeat}
            add_metrics(row, {pipeline: metric_bundle(frame) for pipeline, frame in pooled.items()})
            for name, (candidate, comparator) in COMPARISONS.items():
                for key, value in idea084.paired_error_transitions(pooled[candidate], pooled[comparator]).items():
                    row[f"{name}_{key}"] = value
            seed_repeat_rows.append(row)
    folds = pd.DataFrame(fold_rows)
    seed_repeats = pd.DataFrame(seed_repeat_rows)
    contrast_columns = [f"{name}_macro_f1" for name in COMPARISONS]
    split_cells = folds.groupby(["repeat", "fold"], as_index=False)[contrast_columns].mean()
    metrics = (
        "plain_accuracy", "macro_f1", "balanced_accuracy", "cross_entropy", "brier",
        "kitten_recall", "adult_recall", "senior_recall",
    )
    means = {
        metric: {pipeline: float(seed_repeats[f"{pipeline}_{metric}"].mean()) for pipeline in ALL_PIPELINES}
        for metric in metrics
    }
    gate = protocol["classification_gate"]
    comparison_results: dict[str, Any] = {}
    for name, (candidate, comparator) in COMPARISONS.items():
        values = seed_repeats[f"{name}_macro_f1"]
        splits = split_cells[f"{name}_macro_f1"]
        per_seed = {
            str(seed): float(seed_repeats[seed_repeats["base_seed"] == seed][f"{name}_macro_f1"].mean())
            for seed in BASE_SEEDS
        }
        conditions = {
            "mean_macro_f1_delta": float(values.mean()) >= float(gate["minimum_mean_seed_repeat_macro_f1_delta"]),
            "positive_base_seed_means": sum(value > 0 for value in per_seed.values()) >= int(gate["minimum_positive_base_seed_means"]),
            "positive_seed_repeats": int((values > 0).sum()) >= int(gate["minimum_positive_seed_repeats"]),
            "nonnegative_split_cells": int((splits >= 0).sum()) >= int(gate["minimum_nonnegative_split_cells"]),
            "worst_split_cell": float(splits.min()) >= float(gate["minimum_worst_split_cell_delta"]),
        }
        gated = name in GATED_COMPARISONS
        correction_keys = ("paired_occurrences", "corrected_errors", "introduced_errors", "net_corrections", "unchanged_correct", "unchanged_wrong")
        comparison_results[name] = {
            "candidate": candidate,
            "comparator": comparator,
            "macro_f1": idea084.contrast_summary(values),
            "per_base_seed_mean_delta": per_seed,
            "positive_base_seed_means": sum(value > 0 for value in per_seed.values()),
            "split_cell_nonnegative": int((splits >= 0).sum()),
            "split_cell_worst": float(splits.min()),
            "classification_gate_applicable": gated,
            "classification_conditions": conditions if gated else None,
            "classification_gate_passed": bool(all(conditions.values())) if gated else None,
            "auxiliary_profile": {
                "mean_plain_accuracy_delta": means["plain_accuracy"][candidate] - means["plain_accuracy"][comparator],
                "mean_balanced_accuracy_delta": means["balanced_accuracy"][candidate] - means["balanced_accuracy"][comparator],
                "mean_cross_entropy_gain": means["cross_entropy"][comparator] - means["cross_entropy"][candidate],
                "mean_brier_gain": means["brier"][comparator] - means["brier"][candidate],
                "mean_class_recall_delta": {
                    class_name: means[f"{class_name}_recall"][candidate] - means[f"{class_name}_recall"][comparator]
                    for class_name in CLASS_NAMES
                },
                "not_classification_gate_conditions": True,
            },
            "error_correction_profile": {
                key: {
                    "mean_per_seed_repeat": float(seed_repeats[f"{name}_{key}"].mean()),
                    "total_descriptive_repeated_occurrences": int(seed_repeats[f"{name}_{key}"].sum()),
                }
                for key in correction_keys
            },
            "interpretation_boundary": "Exploratory paired comparison on reused IDEA-085 validation predictions; not independent confirmation or causal proof.",
        }
        comparison_results[name]["error_correction_profile"]["net_correction_positive_tied_negative"] = {
            "positive": int((seed_repeats[f"{name}_net_corrections"] > 0).sum()),
            "tied": int((seed_repeats[f"{name}_net_corrections"] == 0).sum()),
            "negative": int((seed_repeats[f"{name}_net_corrections"] < 0).sum()),
        }
    pooled_frames = {pipeline: pd.concat(parts, ignore_index=True) for pipeline, parts in pooled_all.items()}
    occurrence_accounting = {}
    for pipeline, frame in pooled_frames.items():
        counts = frame.groupby("cat_id").size().to_numpy(dtype=np.int64)
        occurrence_accounting[pipeline] = {
            "repeated_animal_occurrences": int(len(frame)),
            "unique_cats": int(frame["cat_id"].nunique()),
            "occurrences_per_cat_min": int(counts.min()),
            "occurrences_per_cat_median": float(np.median(counts)),
            "occurrences_per_cat_max": int(counts.max()),
            "note": "Repeated validation occurrences are descriptive and are not independent cats.",
        }
    return {
        "status": "complete",
        "outer_test_accessed": False,
        "new_candidate_fits": len(new_fits),
        "reused_reference_fits": len(reused_fits),
        "paired_fold_comparisons": len(fold_rows),
        "seed_repeat_estimates": len(seed_repeat_rows),
        "split_cell_estimates": len(split_cells),
        "no_single_global_gate": True,
        "exploratory_not_independent_confirmation": True,
        "pipeline_seed_repeat_means": means,
        "comparison_results": comparison_results,
        "fold_results": fold_rows,
        "seed_repeat_results": seed_repeat_rows,
        "split_cell_results": split_cells.to_dict(orient="records"),
        "pooled_validation": {
            pipeline: {"animal_occurrences": len(frame), **metric_bundle(frame)}
            for pipeline, frame in pooled_frames.items()
        },
        "animal_occurrence_accounting": occurrence_accounting,
    }


def compare_new_batch_history_to_reused(
    fit: dict[str, Any], base_seed: int, repeat: int, fold: int
) -> None:
    history = fit["audit"]["history"]
    for pipeline in REFERENCE_PIPELINES:
        old = read_json(old_fit_path(pipeline, base_seed, repeat, fold))["audit"]["history"]
        for epoch in range(min(len(history), len(old))):
            current = history[epoch]["train_audit"]
            reference = old[epoch]["train_audit"]
            if current["cat_order_sha256"] != reference["cat_order_sha256"] or current["call_coverage_sha256"] != reference["call_coverage_sha256"]:
                raise RuntimeError("IDEA-086 trained batch order differs from reused IDEA-085 fit")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.director_authorized:
        raise RuntimeError("IDEA-086 formal run is blocked until the research director explicitly authorizes it")
    idea068.configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    if args.max_cells is not None and args.max_cells <= 0:
        raise ValueError("--max-cells must be positive")
    preflight_result, preflight_path, reuse_path = require_matching_cpu_preflight(protocol, args.output_subdir)
    run_root = idea084.resolve_run_root(args.output_subdir)
    store = idea068.idea051.reference.historical.idea019.load_feature_store()
    features, feature_summary = load_features(protocol, store.call_ids)
    roles = pd.read_csv(REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    reused_fits, current_reuse_manifest = collect_reused_fits(protocol, store, roles, verify_rows=False)
    if current_reuse_manifest["content_sha256"] != read_json(reuse_path)["content_sha256"]:
        raise RuntimeError("IDEA-086 read-only reused artifacts changed after preflight")
    device = idea068.idea051.reference.historical.idea019.resolve_device(args.device)
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "source_feature_sha256": feature_summary["feature_sha256"],
        "cpu_preflight_sha256": sha256(preflight_path),
        "reuse_manifest_sha256": sha256(reuse_path),
        "reuse_manifest_content_sha256": current_reuse_manifest["content_sha256"],
        "director_authorized": True,
        "outer_test_accessed": False,
        "new_pipeline": PIPELINE,
        "reference_pipelines": list(REFERENCE_PIPELINES),
        "new_candidate_fits": 36,
        "reused_reference_fits": 144,
        "model": protocol["model"],
        "comparisons": protocol["comparisons"],
        "determinism": protocol["determinism"],
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": str(device), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.exists():
        if not args.resume or read_json(manifest_path) != manifest:
            raise RuntimeError("Existing IDEA-086 run manifest differs")
    else:
        write_json(manifest_path, manifest)
    completed: list[dict[str, Any]] = []
    processed = 0
    for base_seed in BASE_SEEDS:
        for repeat in protocol["model"]["repeats"]:
            for fold in protocol["model"]["folds"]:
                indices = idea084.role_cell_indices(store, roles, repeat, fold)
                full_seed = idea068.idea051.reference.historical.full_seed(base_seed, repeat, fold)
                initialization = initial_compatibility_audit(
                    protocol, store, features, indices["train"],
                    indices["validation"][: min(32, len(indices["validation"]))], full_seed,
                )
                output_dir = run_root / "fits" / PIPELINE / f"base_seed_{base_seed}" / f"repeat_{repeat}" / f"fold_{fold}"
                summary_path = output_dir / "fit_summary.json"
                if summary_path.exists():
                    if not args.resume:
                        raise FileExistsError(summary_path)
                    fit = read_json(summary_path)
                    validate_new_fit(fit, base_seed, full_seed, repeat, fold)
                else:
                    print(f"=== {PIPELINE} base_seed={base_seed} repeat={repeat} fold={fold} full_seed={full_seed} ===", flush=True)
                    audit, animals, calls = fit_inner(
                        PIPELINE, protocol, store, features, indices["train"],
                        indices["validation"], device, full_seed,
                    )
                    if audit["model"]["trainable_parameters"] != EXPECTED_PARAMETERS:
                        raise RuntimeError("IDEA-086 trained parameter audit mismatch")
                    trained_probe = audit["model"].get("trained_checkpoint_permutation_invariance")
                    if not trained_probe or not trained_probe.get("passed") or audit["model"]["projection_weight_norm"] <= 0.0:
                        raise RuntimeError("IDEA-086 trained checkpoint permutation audit failed")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    animal_path = output_dir / "validation_animal_predictions.csv"
                    call_path = output_dir / "validation_call_predictions.csv"
                    animals.to_csv(animal_path, index=False)
                    calls.to_csv(call_path, index=False)
                    fit = {
                        "status": "complete", "pipeline": PIPELINE,
                        "base_seed": int(base_seed), "full_seed": int(full_seed),
                        "repeat": int(repeat), "fold": int(fold),
                        "outer_test_accessed": False,
                        "initialization": initialization,
                        "train_calls": int(len(indices["train"])),
                        "validation_calls": int(len(indices["validation"])),
                        "validation_cats": int(animals["cat_id"].nunique()),
                        "validation_animal_predictions": animal_path.relative_to(REPO_ROOT).as_posix(),
                        "validation_animal_sha256": sha256(animal_path),
                        "validation_call_predictions": call_path.relative_to(REPO_ROOT).as_posix(),
                        "validation_call_sha256": sha256(call_path),
                        "audit": audit,
                    }
                    compare_new_batch_history_to_reused(fit, base_seed, repeat, fold)
                    write_json(summary_path, fit)
                completed.append(fit)
                processed += 1
                if args.max_cells is not None and processed >= args.max_cells:
                    partial = {
                        "status": "partial_first_fit_audit_required",
                        "completed_new_fits_visible": len(completed),
                        "expected_new_fits": 36,
                        "reused_reference_fits": 144,
                        "aggregation_generated": False,
                        "resume_required": True,
                        "outer_test_accessed": False,
                        "first_fit_trained_permutation_invariance": completed[0]["audit"]["model"]["trained_checkpoint_permutation_invariance"],
                        "protocol_sha256": sha256(PROTOCOL_PATH),
                        "runner_sha256": sha256(Path(__file__).resolve()),
                        "cpu_preflight_sha256": sha256(preflight_path),
                    }
                    write_json(run_root / "partial_run_summary.json", partial)
                    return partial
    summary = aggregate(completed, reused_fits, protocol)
    summary_path = run_root / "initial_evaluation_summary.json"
    if summary_path.exists():
        if not args.resume or read_json(summary_path) != summary or summary_path.read_bytes() != canonical_json_bytes(summary):
            raise RuntimeError("IDEA-086 resumed aggregate differs")
    else:
        write_json(summary_path, summary)
    compact = {
        "status": "complete", "completed_new_fits": len(completed),
        "expected_new_fits": 36, "reused_reference_fits": 144,
        "classification_gate_passed": {
            name: summary["comparison_results"][name]["classification_gate_passed"]
            for name in GATED_COMPARISONS
        },
        "auxiliary_comparisons_have_no_gate": ["SET1_minus_T1", "SET1_minus_J1"],
        "no_single_global_gate": True, "outer_test_accessed": False,
    }
    compact_path = run_root / "run_summary.json"
    if compact_path.exists():
        if read_json(compact_path) != compact or compact_path.read_bytes() != canonical_json_bytes(compact):
            raise RuntimeError("IDEA-086 resumed compact summary differs")
    else:
        write_json(compact_path, compact)
    return summary


def main() -> None:
    args = parse_args()
    result = preflight(args) if args.stage == "preflight" else run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
