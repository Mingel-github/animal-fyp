"""Run IDEA-057 structured local time-frequency patch validation.

The runner has four explicit stages:

* ``prepare`` caches a frozen 3 x 3 relative AST patch grid for every call;
* ``select`` compares two bounded local readers using inner roles only;
* ``smoke`` audits the selected R0/M1/C1 implementation and writes the lock;
* ``evaluate`` is the only stage that requests outer-test indices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from torch import nn
from transformers import ASTModel


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for local_root in (SCRIPTS_ROOT, SRC_ROOT):
    if str(local_root) not in sys.path:
        sys.path.insert(0, str(local_root))

import extract_ast_embeddings as ast_extraction  # noqa: E402
import extract_ast_temporal_tokens as temporal_extraction  # noqa: E402
import run_meowagenet_ast_internal_diagnosis as diagnosis  # noqa: E402
import run_meowagenet_idea052_ast_local_residual as base  # noqa: E402


reference = base.reference
PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea057_structured_local_patch_v1.json"
)
PLAN_PATH = REPO_ROOT / "plan" / "IDEA-057_structured_local_patch_branch.md"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
LOCKED_PROTOCOL_PATH = REPO_ROOT / "configs" / "protocol" / "meowagenet_locked_v1.json"
FBANK_PATH = (
    REPO_ROOT / "runs" / "ast_locked_v1" / "gpu_rerun_2026-08-26" / "ast_fbank_128.npz"
)
GLOBAL_EMBEDDING_PATH = (
    REPO_ROOT
    / "runs"
    / "ast_locked_v1"
    / "gpu_rerun_2026-08-26"
    / "ast_standard_call_embeddings.npz"
)
DIAGNOSTIC_FEATURE_PATH = (
    REPO_ROOT
    / "runs"
    / "meowagenet_ast_internal_diagnosis_v1"
    / "features"
    / "ast_internal_diagnostic_features.npz"
)
DIAGNOSTIC_REPORT_PATH = REPO_ROOT / "reports" / "30_AST_internal_diagnosis_results.md"
REFERENCE_RUNNER_PATH = SCRIPTS_ROOT / "run_meowagenet_idea052_ast_local_residual.py"
RUNS_ROOT = REPO_ROOT / "runs"
DEFAULT_OUTPUT_SUBDIR = "meowagenet_idea057_structured_local_patch_v1"
FEATURE_NAME = "ast_final_3x3_grid_embeddings.npz"
PIPELINES = (
    "R0_tuned_frozen_ast",
    "M1_position_aware_local_patch",
    "C1_position_removed_control",
)
LOCAL_PIPELINES = PIPELINES[1:]
LABEL_NAMES = base.LABEL_NAMES
PROBABILITY_COLUMNS = base.PROBABILITY_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("prepare", "select", "smoke", "evaluate"), required=True
    )
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
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


def git_blob_object_id(revision: str, path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{revision}:{repo_relative(path)}"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def worktree_blob_object_id(path: Path) -> str:
    return subprocess.check_output(
        ["git", "hash-object", repo_relative(path)], cwd=REPO_ROOT, text=True
    ).strip()


def feature_path(run_root: Path) -> Path:
    return run_root / "features" / FEATURE_NAME


def selection_path(run_root: Path) -> Path:
    return run_root / "selection" / "selection_record.json"


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol["protocol_id"] != "meowagenet-idea057-structured-local-patch-v1":
        raise RuntimeError("Unexpected IDEA-057 protocol")
    if tuple(protocol["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-057 pipeline matrix changed")
    selection = protocol["inner_selection"]
    expected_candidate_fits = (
        len(selection["candidate_recipes"])
        * len(selection["repeats"])
        * len(selection["outer_folds"])
    )
    if expected_candidate_fits != int(selection["candidate_fits"]):
        raise RuntimeError("IDEA-057 inner-selection fit budget is inconsistent")
    evaluation = protocol["initial_evaluation"]
    expected_outer_fits = (
        len(PIPELINES)
        * len(evaluation["base_seeds"])
        * len(evaluation["repeats"])
        * len(evaluation["outer_folds"])
    )
    if expected_outer_fits != int(evaluation["total_outer_fits"]):
        raise RuntimeError("IDEA-057 outer fit budget is inconsistent")
    fixed = protocol["fixed_training"]
    if int(fixed["call_micro_batch_size"]) * int(
        fixed["gradient_accumulation_steps"]
    ) != int(fixed["accumulation_window_calls"]):
        raise RuntimeError("IDEA-057 accumulation window is inconsistent")
    dependencies = protocol["dependencies"]
    checks = {
        PLAN_PATH: protocol["idea"]["sha256"],
        ROLES_PATH: protocol["splits"]["roles_sha256"],
        LOCKED_PROTOCOL_PATH: dependencies["locked_protocol_sha256"],
        FBANK_PATH: dependencies["fbank_sha256"],
        GLOBAL_EMBEDDING_PATH: dependencies["global_embedding_sha256"],
        DIAGNOSTIC_FEATURE_PATH: dependencies["diagnostic_feature_sha256"],
        DIAGNOSTIC_REPORT_PATH: dependencies["diagnostic_report_sha256"],
        REFERENCE_RUNNER_PATH: dependencies["reference_runner_sha256"],
    }
    for path, expected in checks.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"IDEA-057 dependency checksum mismatch: {path}")


def prepare_features(
    run_root: Path, protocol: dict[str, Any], device: torch.device, resume: bool
) -> None:
    output_path = feature_path(run_root)
    summary_path = output_path.parent / "summary.json"
    if output_path.is_file() and summary_path.is_file():
        if not resume:
            raise FileExistsError(output_path)
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return
    locked = read_json(LOCKED_PROTOCOL_PATH)
    loaded = np.load(FBANK_PATH)
    frozen = np.load(GLOBAL_EMBEDDING_PATH)
    diagnostics = np.load(DIAGNOSTIC_FEATURE_PATH)
    features = loaded["features"].astype(np.float32)
    segment_call_indices = loaded["segment_call_indices"].astype(np.int64)
    segment_counts = loaded["segment_counts"].astype(np.int64)
    call_ids = loaded["call_ids"].astype(str)
    if not np.array_equal(call_ids, frozen["call_ids"].astype(str)):
        raise RuntimeError("Fbank and final-embedding call order differs")
    if not np.array_equal(call_ids, diagnostics["call_ids"].astype(str)):
        raise RuntimeError("Fbank and diagnostic call order differs")
    frame_counts = temporal_extraction.valid_fbank_frames(features).astype(np.int64)
    ast_config = locked["ast"]
    standard = ast_config["variants"]["ast_standard"]
    model = ASTModel.from_pretrained(
        ast_config["checkpoint"],
        revision=ast_config["revision"],
        cache_dir=REPO_ROOT / "data" / "models" / "huggingface",
        use_safetensors=True,
    )
    geometry = ast_extraction.adapt_geometry(
        model,
        max_length=int(ast_config["max_length_frames"]),
        frequency_stride=int(standard["frequency_stride"]),
        time_stride=int(standard["time_stride"]),
    )
    model.eval().to(device)
    frequency_positions, time_positions = geometry["target_grid"]
    patch_size = int(model.config.patch_size)
    valid_time_mask, _ = temporal_extraction.temporal_patch_mask(
        frame_counts,
        int(time_positions),
        patch_size,
        int(standard["time_stride"]),
        float(protocol["feature_preparation"]["minimum_real_patch_overlap_fraction"]),
    )
    call_count = len(call_ids)
    hidden_size = int(model.config.hidden_size)
    grid_sums = np.zeros((call_count, 3, 3, hidden_size), dtype=np.float64)
    grid_counts = np.zeros((call_count, 3, 3), dtype=np.int64)
    reconstructed = np.zeros((call_count, hidden_size), dtype=np.float64)
    frequency_groups = diagnosis.relative_groups(
        np.arange(int(frequency_positions), dtype=np.int64)
    )
    batch_size = 32
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            output = model(input_values=torch.from_numpy(features[start:stop]).to(device))
            pooled = output.pooler_output.float().cpu().numpy()
            patches = (
                output.last_hidden_state[:, 2:, :]
                .reshape(
                    stop - start,
                    int(frequency_positions),
                    int(time_positions),
                    hidden_size,
                )
                .float()
                .cpu()
                .numpy()
            )
            np.add.at(reconstructed, segment_call_indices[start:stop], pooled)
            for local_segment, segment_index in enumerate(range(start, stop)):
                call_index = int(segment_call_indices[segment_index])
                valid_times = np.flatnonzero(valid_time_mask[segment_index])
                time_groups = diagnosis.relative_groups(valid_times)
                for frequency_index, frequency_group in enumerate(frequency_groups):
                    for time_index, time_group in enumerate(time_groups):
                        value = patches[
                            local_segment,
                            frequency_group[:, None],
                            time_group[None, :],
                            :,
                        ].mean(axis=(0, 1))
                        grid_sums[call_index, frequency_index, time_index] += value
                        grid_counts[call_index, frequency_index, time_index] += 1
    if np.any(grid_counts == 0):
        raise RuntimeError("At least one call has an empty relative grid cell")
    grid = (grid_sums / grid_counts[..., None]).astype(np.float32)
    reconstructed = (reconstructed / segment_counts[:, None]).astype(np.float32)
    maximum_difference = float(
        np.abs(reconstructed - frozen["embeddings"].astype(np.float32)).max()
    )
    if maximum_difference > 5.0e-4:
        raise RuntimeError("Reconstructed AST embedding differs from locked embedding")
    if grid.shape != (792, 3, 3, 768) or not np.isfinite(grid).all():
        raise RuntimeError("Prepared IDEA-057 grid has an invalid shape or value")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        grid_embeddings=grid,
        call_ids=call_ids,
        cat_ids=loaded["cat_ids"].astype(str),
        labels=loaded["labels"].astype(np.int8),
        durations=loaded["durations"].astype(np.float32),
        segment_counts=segment_counts.astype(np.int16),
        valid_frame_fraction=diagnostics["valid_frame_fraction"].astype(np.float32),
        mean_abs_fbank=diagnostics["mean_abs_fbank"].astype(np.float32),
        fbank_std=diagnostics["fbank_std"].astype(np.float32),
    )
    summary = {
        "status": "complete",
        "stage": "idea057_feature_preparation",
        "outer_test_accessed": False,
        "protocol_id": protocol["protocol_id"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "calls": call_count,
        "cats": int(len(np.unique(loaded["cat_ids"].astype(str)))),
        "segments": int(len(features)),
        "grid_shape": list(grid.shape),
        "grid_order": "frequency_low_to_high_then_time_early_to_late",
        "geometry": geometry,
        "valid_time_token_range": [
            int(valid_time_mask.sum(axis=1).min()),
            int(valid_time_mask.sum(axis=1).max()),
        ],
        "reconstructed_final_embedding_max_abs_difference": maximum_difference,
        "inference_seconds": float(time.perf_counter() - started),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "feature_path": repo_relative(output_path),
        "feature_sha256": sha256(output_path),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


@dataclass(frozen=True)
class GridStore:
    global_embeddings: np.ndarray
    temporal_tokens: np.ndarray
    call_token_indices: tuple[np.ndarray, ...]
    call_ids: np.ndarray
    cat_ids: np.ndarray
    labels: np.ndarray
    durations: np.ndarray
    valid_frame_fraction: np.ndarray
    mean_abs_fbank: np.ndarray
    fbank_std: np.ndarray


def load_store(run_root: Path) -> GridStore:
    grid_data = np.load(feature_path(run_root))
    global_data = np.load(GLOBAL_EMBEDDING_PATH)
    for field in ("call_ids", "cat_ids", "labels"):
        if not np.array_equal(
            global_data[field].astype(str), grid_data[field].astype(str)
        ):
            raise RuntimeError(f"Global/grid {field} order differs")
    grid = grid_data["grid_embeddings"].astype(np.float32)
    if grid.shape != (792, 3, 3, 768):
        raise RuntimeError("Unexpected IDEA-057 feature shape")
    flattened = grid.reshape(792 * 9, 768)
    call_token_indices = tuple(
        np.arange(index * 9, (index + 1) * 9, dtype=np.int64)
        for index in range(792)
    )
    return GridStore(
        global_embeddings=global_data["embeddings"].astype(np.float32),
        temporal_tokens=flattened,
        call_token_indices=call_token_indices,
        call_ids=global_data["call_ids"].astype(str),
        cat_ids=global_data["cat_ids"].astype(str),
        labels=global_data["labels"].astype(np.int64),
        durations=global_data["durations"].astype(np.float32),
        valid_frame_fraction=grid_data["valid_frame_fraction"].astype(np.float32),
        mean_abs_fbank=grid_data["mean_abs_fbank"].astype(np.float32),
        fbank_std=grid_data["fbank_std"].astype(np.float32),
    )


def active_recipe(protocol: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    recipe_id = protocol.get("_active_recipe_id")
    if recipe_id is None:
        raise RuntimeError("No active IDEA-057 local-reader recipe")
    recipes = protocol["inner_selection"]["candidate_recipes"]
    if recipe_id not in recipes:
        raise RuntimeError(f"Unknown IDEA-057 recipe: {recipe_id}")
    return str(recipe_id), recipes[recipe_id]


class StructuredLocalPatchClassifier(nn.Module):
    def __init__(
        self,
        global_mean: np.ndarray,
        global_scale: np.ndarray,
        local_mean: np.ndarray,
        local_scale: np.ndarray,
        dropout: float,
        mode: str,
        recipe_id: str,
        recipe: dict[str, Any],
    ) -> None:
        super().__init__()
        if mode not in ("none", "positioned", "position_removed"):
            raise ValueError(mode)
        self.mode = mode
        self.recipe_id = recipe_id
        self.register_buffer("global_mean", torch.from_numpy(global_mean.astype(np.float32)))
        self.register_buffer("global_scale", torch.from_numpy(global_scale.astype(np.float32)))
        self.register_buffer("token_mean", torch.from_numpy(local_mean.astype(np.float32)))
        self.register_buffer("token_scale", torch.from_numpy(local_scale.astype(np.float32)))
        self.projection = nn.Linear(768, 128)
        self.batch_norm = nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, 3)
        if recipe_id == "middle_frequency_strip":
            selected_cells = (3, 4, 5)
        elif recipe_id == "full_3x3_grid":
            selected_cells = tuple(range(9))
        else:
            raise ValueError(recipe_id)
        self.selected_cells = selected_cells
        if mode == "none":
            self.register_parameter("residual_gate", None)
            self.local_projection = None
            self.local_readout = None
        else:
            local_dimension = int(recipe["local_projection_dimension"])
            # Keep the subsequent dropout/random-stream state matched to R0.
            # The local weights remain deterministic from the fit seed while their
            # initialization does not advance the caller's global CPU RNG.
            with torch.random.fork_rng(devices=[]):
                self.local_projection = nn.Linear(768, local_dimension)
                self.local_readout = nn.Linear(
                    local_dimension * len(selected_cells), 128
                )
            self.residual_gate = nn.Parameter(torch.zeros(128))

    def hidden_and_local(
        self,
        global_embeddings: torch.Tensor,
        temporal_tokens: torch.Tensor,
        token_to_call: torch.Tensor,
        call_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        standardized_global = (global_embeddings - self.global_mean) / self.global_scale
        global_hidden = torch.relu(self.projection(standardized_global))
        if self.mode == "none":
            return global_hidden, torch.zeros_like(global_hidden)
        groups = [
            temporal_tokens[token_to_call == index] for index in range(call_count)
        ]
        if any(len(group) != 9 for group in groups):
            raise RuntimeError("Every IDEA-057 call must provide nine grid cells")
        grid = torch.stack(groups)
        standardized = (grid - self.token_mean) / self.token_scale
        if self.mode == "position_removed":
            standardized = standardized.mean(dim=1, keepdim=True).expand(-1, 9, -1)
        selected = standardized[:, self.selected_cells, :]
        projected = torch.relu(self.local_projection(selected))
        local_hidden = torch.relu(self.local_readout(projected.flatten(start_dim=1)))
        gate = torch.tanh(self.residual_gate)[None, :]
        return global_hidden + gate * local_hidden, local_hidden

    def forward(
        self,
        global_embeddings: torch.Tensor,
        temporal_tokens: torch.Tensor,
        token_to_call: torch.Tensor,
        call_count: int,
    ) -> torch.Tensor:
        hidden, _ = self.hidden_and_local(
            global_embeddings, temporal_tokens, token_to_call, call_count
        )
        return self.classifier(self.dropout(self.batch_norm(hidden)))

    def gate_summary(self) -> dict[str, float] | None:
        if self.residual_gate is None:
            return None
        values = torch.tanh(self.residual_gate.detach()).float().cpu().numpy()
        return {
            "mean_abs_gate": float(np.abs(values).mean()),
            "max_abs_gate": float(np.abs(values).max()),
            "rms_gate": float(np.sqrt(np.mean(values**2))),
            "positive_fraction": float(np.mean(values > 0)),
            "active_fraction_above_1e_3": float(np.mean(np.abs(values) > 1.0e-3)),
        }


def build_model(
    pipeline: str,
    protocol: dict[str, Any],
    store: GridStore,
    train_indices: np.ndarray,
) -> StructuredLocalPatchClassifier:
    recipe_id, recipe = active_recipe(protocol)
    global_values = store.global_embeddings[train_indices]
    local_indices = np.concatenate(
        [store.call_token_indices[int(index)] for index in train_indices]
    )
    local_values = store.temporal_tokens[local_indices]
    modes = {
        PIPELINES[0]: "none",
        PIPELINES[1]: "positioned",
        PIPELINES[2]: "position_removed",
    }
    return StructuredLocalPatchClassifier(
        global_values.mean(axis=0),
        base.safe_scale(global_values),
        local_values.mean(axis=0),
        base.safe_scale(local_values),
        float(protocol["fixed_training"]["dropout"]),
        modes[pipeline],
        recipe_id,
        recipe,
    )


def predict_calls(
    model: StructuredLocalPatchClassifier,
    loader: torch.utils.data.DataLoader,
    store: GridStore,
    device: torch.device,
) -> tuple[float, pd.DataFrame]:
    model.eval()
    rows: list[dict[str, Any]] = []
    total_loss = 0.0
    with torch.no_grad():
        for cpu_batch in loader:
            batch = base.move_batch(cpu_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(
                    batch["global_embeddings"],
                    batch["temporal_tokens"],
                    batch["token_to_call"],
                    len(batch["labels"]),
                )
                total_loss += float(
                    torch.nn.functional.cross_entropy(
                        logits, batch["labels"], reduction="sum"
                    )
                )
                probabilities = torch.softmax(logits, dim=1).float().cpu().numpy()
            indices = cpu_batch["call_indices"].numpy().astype(np.int64)
            labels = cpu_batch["labels"].numpy().astype(np.int64)
            for local_index, call_index in enumerate(indices):
                rows.append(
                    {
                        "call_index": int(call_index),
                        "call_id": str(store.call_ids[call_index]),
                        "cat_id": str(store.cat_ids[call_index]),
                        "true_label": int(labels[local_index]),
                        "duration": float(store.durations[call_index]),
                        "temporal_token_count": 9,
                        "valid_frame_fraction": float(
                            store.valid_frame_fraction[call_index]
                        ),
                        "mean_abs_fbank": float(store.mean_abs_fbank[call_index]),
                        "fbank_std": float(store.fbank_std[call_index]),
                        **{
                            column: float(probabilities[local_index, class_index])
                            for class_index, column in enumerate(PROBABILITY_COLUMNS)
                        },
                        "predicted_label": int(probabilities[local_index].argmax()),
                    }
                )
    calls = pd.DataFrame(rows).sort_values("call_index").reset_index(drop=True)
    return total_loss / len(calls), calls


def calls_to_animals(calls: pd.DataFrame) -> pd.DataFrame:
    animals = base.previous.calls_to_animals(calls)
    metadata = (
        calls.groupby("cat_id", sort=True)
        .agg(
            mean_call_duration=("duration", "mean"),
            mean_valid_frame_fraction=("valid_frame_fraction", "mean"),
            mean_abs_fbank=("mean_abs_fbank", "mean"),
            mean_fbank_std=("fbank_std", "mean"),
        )
        .reset_index()
    )
    return animals.merge(metadata, on="cat_id", validate="one_to_one")


def configure_base_runtime() -> None:
    base.PIPELINES = PIPELINES
    base.RESIDUAL_PIPELINES = LOCAL_PIPELINES
    base.PROTOCOL_PATH = PROTOCOL_PATH
    base.PLAN_PATH = PLAN_PATH
    base.ROLES_PATH = ROLES_PATH
    base.GLOBAL_EMBEDDING_PATH = GLOBAL_EMBEDDING_PATH
    base.TEMPORAL_TOKEN_PATH = Path("unused-by-idea057")
    base.build_model = build_model
    base.predict_calls = predict_calls
    base.calls_to_animals = calls_to_animals


def run_selection(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: GridStore,
    device: torch.device,
    resume: bool,
) -> None:
    record_path = selection_path(run_root)
    if record_path.is_file() and not resume:
        raise FileExistsError(record_path)
    settings = protocol["inner_selection"]
    candidate_summaries = []
    for recipe_id in sorted(settings["candidate_recipes"]):
        protocol["_active_recipe_id"] = recipe_id
        rows = []
        for repeat in settings["repeats"]:
            for outer_fold in settings["outer_folds"]:
                output_dir = (
                    run_root
                    / "selection"
                    / "fits"
                    / recipe_id
                    / f"repeat_{repeat}"
                    / f"fold_{outer_fold}"
                )
                fit_path = output_dir / "fit_summary.json"
                if fit_path.is_file():
                    if not resume:
                        raise FileExistsError(fit_path)
                    fit = read_json(fit_path)
                    rows.append(fit)
                    continue
                indices = reference.historical.fold_indices(
                    store, roles, int(repeat), int(outer_fold), include_test=False
                )
                seed = reference.historical.full_seed(
                    int(settings["base_seed"]), int(repeat), int(outer_fold)
                )
                print(
                    f"IDEA057 SELECT recipe={recipe_id} repeat={repeat} "
                    f"fold={outer_fold} seed={seed}",
                    flush=True,
                )
                best_epoch, inner, _, _ = base.fit_inner(
                    PIPELINES[1],
                    protocol,
                    store,
                    indices["train"],
                    indices["validation"],
                    device,
                    seed,
                )
                fit = {
                    "status": "complete",
                    "stage": "idea057_inner_only_selection",
                    "outer_test_accessed": False,
                    "recipe_id": recipe_id,
                    "repeat": int(repeat),
                    "outer_fold": int(outer_fold),
                    "full_seed": int(seed),
                    "selected_epoch": int(best_epoch),
                    "inner": inner,
                }
                write_json(fit_path, fit)
                rows.append(fit)
        macro_f1 = [
            row["inner"]["best_validation_animal_metrics"]["macro_f1"]
            for row in rows
        ]
        cross_entropy = [
            row["inner"]["best_validation_animal_cross_entropy"] for row in rows
        ]
        candidate_summaries.append(
            {
                "recipe_id": recipe_id,
                "inner_splits": len(rows),
                "mean_validation_animal_macro_f1": float(np.mean(macro_f1)),
                "sample_sd_validation_animal_macro_f1": float(
                    np.std(macro_f1, ddof=1)
                ),
                "mean_validation_animal_cross_entropy": float(
                    np.mean(cross_entropy)
                ),
                "trainable_parameters": int(
                    rows[0]["inner"]["parameters"]["trainable"]
                ),
                "selected_epochs": [int(row["selected_epoch"]) for row in rows],
            }
        )
    ranked = sorted(
        candidate_summaries,
        key=lambda row: (
            -round(row["mean_validation_animal_macro_f1"], 6),
            row["mean_validation_animal_cross_entropy"],
            row["trainable_parameters"],
            row["recipe_id"],
        ),
    )
    selected = ranked[0]
    record = {
        "schema_version": "1.0",
        "status": "selected_from_inner_only_roles",
        "protocol_id": protocol["protocol_id"],
        "outer_test_accessed": False,
        "selection_rule": settings["selection_rule"],
        "candidate_summaries": candidate_summaries,
        "ranking": [row["recipe_id"] for row in ranked],
        "selected_recipe_id": selected["recipe_id"],
        "selected_recipe": settings["candidate_recipes"][selected["recipe_id"]],
        "selected_mean_validation_animal_macro_f1": selected[
            "mean_validation_animal_macro_f1"
        ],
        "selected_mean_validation_animal_cross_entropy": selected[
            "mean_validation_animal_cross_entropy"
        ],
        "feature_sha256": sha256(feature_path(run_root)),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
    }
    write_json(record_path, record)
    write_json(
        record_path.parent / "run_summary.json",
        {
            "status": "complete",
            "outer_test_accessed": False,
            "completed_candidate_fits": int(settings["candidate_fits"]),
            "selected_recipe_id": record["selected_recipe_id"],
            "selection_record_sha256": sha256(record_path),
        },
    )
    print(json.dumps(record, indent=2), flush=True)


def initialization_audit(
    protocol: dict[str, Any],
    store: GridStore,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    models = []
    for pipeline in PIPELINES:
        reference.historical.set_seed(seed)
        models.append(build_model(pipeline, protocol, store, train_indices).to(device).eval())
    states = [model.state_dict() for model in models]
    global_keys = (
        "global_mean",
        "global_scale",
        "projection.weight",
        "projection.bias",
        "batch_norm.weight",
        "batch_norm.bias",
        "batch_norm.running_mean",
        "batch_norm.running_var",
        "classifier.weight",
        "classifier.bias",
    )
    global_equal = all(
        torch.equal(states[0][key], states[index][key])
        for key in global_keys
        for index in (1, 2)
    )
    local_keys = tuple(key for key in states[1] if key not in global_keys)
    local_equal = all(torch.equal(states[1][key], states[2][key]) for key in local_keys)
    loader = base.build_loader(store, validation_indices[:8], 8, False, seed)
    batch = base.move_batch(next(iter(loader)), device)
    with torch.no_grad():
        logits = [
            model(
                batch["global_embeddings"],
                batch["temporal_tokens"],
                batch["token_to_call"],
                len(batch["labels"]),
            ).float()
            for model in models
        ]
    max_differences = [
        float((logits[index] - logits[0]).abs().max()) for index in (1, 2)
    ]
    parameters = [reference.historical.idea019.trainable_counts(model) for model in models]
    result = {
        "shared_global_initial_state_equal": bool(global_equal),
        "M1_C1_local_initial_state_equal": bool(local_equal),
        "M1_initial_max_logit_difference_from_R0": max_differences[0],
        "C1_initial_max_logit_difference_from_R0": max_differences[1],
        "zero_initialized_local_gates": bool(
            torch.count_nonzero(models[1].residual_gate) == 0
            and torch.count_nonzero(models[2].residual_gate) == 0
        ),
        "parameters": {
            pipeline: counts for pipeline, counts in zip(PIPELINES, parameters)
        },
        "M1_C1_trainable_parameter_match": bool(
            parameters[1]["trainable"] == parameters[2]["trainable"]
        ),
    }
    if not all(
        (
            result["shared_global_initial_state_equal"],
            result["M1_C1_local_initial_state_equal"],
            result["zero_initialized_local_gates"],
            result["M1_C1_trainable_parameter_match"],
            max(max_differences) == 0.0,
        )
    ):
        raise RuntimeError("IDEA-057 initialization or parameter-match audit failed")
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def environment_lock(
    run_root: Path, protocol: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "git_revision": git_revision(),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "idea_sha256": sha256(PLAN_PATH),
        "roles_sha256": sha256(ROLES_PATH),
        "global_embedding_sha256": sha256(GLOBAL_EMBEDDING_PATH),
        "grid_feature_sha256": sha256(feature_path(run_root)),
        "selection_record_sha256": sha256(selection_path(run_root)),
    }


def run_smoke(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: GridStore,
    device: torch.device,
    resume: bool,
) -> None:
    record_path = selection_path(run_root)
    if not record_path.is_file():
        raise FileNotFoundError("Run IDEA-057 inner-only selection before smoke")
    selection = read_json(record_path)
    protocol["_active_recipe_id"] = selection["selected_recipe_id"]
    smoke_root = run_root / "smoke"
    summary_path = smoke_root / "summary.json"
    if summary_path.is_file() and not resume:
        raise FileExistsError(summary_path)
    settings = protocol["smoke"]
    indices = reference.historical.fold_indices(
        store,
        roles,
        int(settings["repeat"]),
        int(settings["outer_fold"]),
        include_test=False,
    )
    seed = reference.historical.full_seed(
        int(settings["base_seed"]), int(settings["repeat"]), int(settings["outer_fold"])
    )
    initial = initialization_audit(
        protocol, store, indices["train"], indices["validation"], device, seed
    )
    fits = []
    for pipeline in PIPELINES:
        output_dir = smoke_root / "fits" / pipeline
        checkpoint_path = output_dir / "best_checkpoint.pt"
        best_epoch, inner, state, validation_calls = base.fit_inner(
            pipeline,
            protocol,
            store,
            indices["train"],
            indices["validation"],
            device,
            seed,
            max_epochs_override=int(settings["epochs"]),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": state, "best_epoch": best_epoch}, checkpoint_path)
        reload_difference = base.reload_probability_difference(
            pipeline,
            protocol,
            store,
            indices["train"],
            indices["validation"],
            checkpoint_path,
            validation_calls,
            device,
            seed,
        )
        fit = {
            "status": "complete",
            "stage": "idea057_inner_only_smoke",
            "outer_test_accessed": False,
            "pipeline": pipeline,
            "selected_recipe_id": selection["selected_recipe_id"],
            "base_seed": int(settings["base_seed"]),
            "repeat": int(settings["repeat"]),
            "outer_fold": int(settings["outer_fold"]),
            "full_seed": int(seed),
            "selected_epoch": int(best_epoch),
            "inner": inner,
            "checkpoint_path": repo_relative(checkpoint_path),
            "checkpoint_reload_max_probability_difference": reload_difference,
        }
        write_json(output_dir / "fit_summary.json", fit)
        fits.append(fit)
    if any(fit["checkpoint_reload_max_probability_difference"] != 0.0 for fit in fits):
        raise RuntimeError("IDEA-057 checkpoint reload audit failed")
    environment_path = run_root / "environment_lock.json"
    write_json(environment_path, environment_lock(run_root, protocol, device))
    code_commit = git_revision()
    if code_commit is None:
        raise RuntimeError("A Git commit is required before IDEA-057 smoke lock")
    lock = {
        "schema_version": "1.0",
        "status": "locked_for_idea057_initial_evaluation",
        "outer_test_accessed": False,
        "protocol_id": protocol["protocol_id"],
        "code_commit": code_commit,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "idea_sha256": sha256(PLAN_PATH),
        "roles_sha256": sha256(ROLES_PATH),
        "global_embedding_sha256": sha256(GLOBAL_EMBEDDING_PATH),
        "grid_feature_sha256": sha256(feature_path(run_root)),
        "selection_record_sha256": sha256(record_path),
        "environment_lock_sha256": sha256(environment_path),
        "selected_recipe_id": selection["selected_recipe_id"],
        "selected_recipe": selection["selected_recipe"],
        "initial_evaluation": protocol["initial_evaluation"],
        "fixed_training": protocol["fixed_training"],
        "initialization_audit": initial,
        "smoke_fit_sha256": {
            pipeline: sha256(smoke_root / "fits" / pipeline / "fit_summary.json")
            for pipeline in PIPELINES
        },
    }
    lock_path = run_root / "execution_lock.json"
    write_json(lock_path, lock)
    summary = {
        "status": "complete",
        "outer_test_accessed": False,
        "excluded_from_evaluation_summary": True,
        "selected_recipe_id": selection["selected_recipe_id"],
        "completed_fits": len(fits),
        "calls": int(len(store.call_ids)),
        "cats": int(len(np.unique(store.cat_ids))),
        "grid_cells_per_call": 9,
        "initialization_audit": initial,
        "checkpoint_reload_passed": True,
        "execution_lock": repo_relative(lock_path),
        "execution_lock_sha256": sha256(lock_path),
        "environment_lock": repo_relative(environment_path),
        "environment_lock_sha256": sha256(environment_path),
        "pipelines": {
            fit["pipeline"]: {
                "validation_animal_macro_f1": fit["inner"][
                    "best_validation_animal_metrics"
                ]["macro_f1"],
                "validation_animal_cross_entropy": fit["inner"][
                    "best_validation_animal_cross_entropy"
                ],
                "local_gate": fit["inner"]["best_residual_gate"],
                "checkpoint_reload_max_probability_difference": fit[
                    "checkpoint_reload_max_probability_difference"
                ],
            }
            for fit in fits
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


def verify_execution_lock(
    run_root: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    lock_path = run_root / "execution_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("Run IDEA-057 smoke before outer evaluation")
    lock = read_json(lock_path)
    if lock["status"] != "locked_for_idea057_initial_evaluation":
        raise RuntimeError("IDEA-057 execution lock status is invalid")
    checks = {
        PROTOCOL_PATH: lock["protocol_sha256"],
        Path(__file__).resolve(): lock["runner_sha256"],
        PLAN_PATH: lock["idea_sha256"],
        ROLES_PATH: lock["roles_sha256"],
        GLOBAL_EMBEDDING_PATH: lock["global_embedding_sha256"],
        feature_path(run_root): lock["grid_feature_sha256"],
        selection_path(run_root): lock["selection_record_sha256"],
        run_root / "environment_lock.json": lock["environment_lock_sha256"],
    }
    for path, expected in checks.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"IDEA-057 execution-lock file changed: {path}")
    for path in (PROTOCOL_PATH, Path(__file__).resolve(), PLAN_PATH):
        if git_blob_object_id(lock["code_commit"], path) != worktree_blob_object_id(path):
            raise RuntimeError(f"IDEA-057 locked commit blob differs: {path}")
    if lock["initial_evaluation"] != protocol["initial_evaluation"]:
        raise RuntimeError("IDEA-057 evaluation matrix differs from lock")
    selection = read_json(selection_path(run_root))
    if lock["selected_recipe_id"] != selection["selected_recipe_id"]:
        raise RuntimeError("IDEA-057 selected recipe differs from lock")
    return lock


def add_idea057_gate(
    summary: dict[str, Any], evaluation_root: Path, protocol: dict[str, Any]
) -> None:
    r0, m1, c1 = PIPELINES
    m1_r0 = summary["paired_summary"][f"{m1}_minus_{r0}"]
    c1_m1 = summary["paired_summary"][f"{c1}_minus_{m1}"]
    m1_c1_mean = -float(c1_m1["mean_macro_f1_difference"])
    m1_c1_positive = sum(
        row["macro_f1_difference"] < 0
        for row in summary["paired"][f"{c1}_minus_{m1}"]
    )
    ce_by_pipeline: dict[str, list[float]] = {pipeline: [] for pipeline in PIPELINES}
    oof_animals: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
    for repeat in protocol["initial_evaluation"]["repeats"]:
        for pipeline in PIPELINES:
            animals = pd.read_csv(
                evaluation_root / "oof" / pipeline / f"repeat_{repeat}_animals.csv",
                dtype={"cat_id": str},
            )
            ce_by_pipeline[pipeline].append(base.previous.animal_cross_entropy(animals))
            oof_animals[pipeline].append(animals)
    mean_ce = {pipeline: float(np.mean(values)) for pipeline, values in ce_by_pipeline.items()}
    aggregate = summary["aggregate"]
    gate = protocol["seed_expansion_gate"]
    class_recall_drops = {
        label: float(
            aggregate[r0]["mean_per_class"][label]["recall"]
            - aggregate[m1]["mean_per_class"][label]["recall"]
        )
        for label in LABEL_NAMES
    }
    checks = {
        "mean_macro_f1_gain_over_R0": bool(
            m1_r0["mean_macro_f1_difference"]
            >= float(gate["minimum_mean_macro_f1_gain_over_R0"])
        ),
        "positive_repeats_over_R0": bool(
            m1_r0["positive_repeats"]
            >= int(gate["minimum_positive_repeats_over_R0"])
        ),
        "mean_macro_f1_gain_over_C1": bool(
            m1_c1_mean >= float(gate["minimum_mean_macro_f1_gain_over_C1"])
        ),
        "positive_repeats_over_C1": bool(
            m1_c1_positive >= int(gate["minimum_positive_repeats_over_C1"])
        ),
        "balanced_accuracy_cost": bool(
            m1_r0["mean_balanced_accuracy_difference"]
            >= -float(gate["maximum_tolerated_mean_balanced_accuracy_drop"])
        ),
        "qwk_cost": bool(
            m1_r0["mean_qwk_difference"]
            >= -float(gate["maximum_tolerated_mean_qwk_drop"])
        ),
        "animal_ce_cost": bool(
            mean_ce[m1] - mean_ce[r0]
            <= float(gate["maximum_tolerated_mean_animal_ce_increase"])
        ),
        "class_recall_cost": bool(
            max(class_recall_drops.values())
            <= float(gate["maximum_tolerated_class_recall_drop"])
        ),
    }
    summary["animal_cross_entropy"] = {
        "complete_oof_by_pipeline": ce_by_pipeline,
        "mean_by_pipeline": mean_ce,
        "M1_minus_R0": mean_ce[m1] - mean_ce[r0],
    }
    summary["idea057_seed_expansion_gate"] = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "M1_minus_R0_mean_macro_f1": m1_r0["mean_macro_f1_difference"],
        "M1_minus_R0_positive_repeats": m1_r0["positive_repeats"],
        "M1_minus_C1_mean_macro_f1": m1_c1_mean,
        "M1_minus_C1_positive_repeats": int(m1_c1_positive),
        "class_recall_drop_from_R0": class_recall_drops,
        "action": gate["action_when_met"]
        if all(checks.values())
        else gate["action_when_not_met"],
    }
    boundaries = {
        "mean_call_duration": [0.58, 0.82],
        "mean_valid_frame_fraction": [0.44, 0.63],
        "mean_abs_fbank": [0.375, 0.493],
        "mean_fbank_std": [0.377, 0.457],
    }
    summary["nuisance_subgroups"] = {
        pipeline: {
            column: base.subgroup_metrics(
                pd.concat(frames, ignore_index=True), column, cuts
            )
            for column, cuts in boundaries.items()
        }
        for pipeline, frames in oof_animals.items()
    }


def run_evaluation(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: GridStore,
    device: torch.device,
    resume: bool,
) -> None:
    lock = verify_execution_lock(run_root, protocol)
    protocol["_active_recipe_id"] = lock["selected_recipe_id"]
    evaluation_root = run_root / "evaluation"
    summary_path = evaluation_root / "summary.json"
    if summary_path.is_file() and not resume:
        raise FileExistsError(summary_path)
    evaluation_root.mkdir(parents=True, exist_ok=True)
    write_json(
        evaluation_root / "run_manifest.json",
        {
            "status": "running",
            "stage": "idea057_initial_evaluation",
            "outer_test_accessed": True,
            "code_commit": lock["code_commit"],
            "protocol_sha256": lock["protocol_sha256"],
            "runner_sha256": lock["runner_sha256"],
            "execution_lock_sha256": sha256(run_root / "execution_lock.json"),
            "environment_lock_sha256": sha256(run_root / "environment_lock.json"),
            "selected_recipe_id": lock["selected_recipe_id"],
            "device": str(device),
        },
    )
    completed = []
    order_audits = []
    evaluation = protocol["initial_evaluation"]
    for base_seed in evaluation["base_seeds"]:
        for repeat in evaluation["repeats"]:
            for outer_fold in evaluation["outer_folds"]:
                indices = reference.historical.fold_indices(
                    store, roles, int(repeat), int(outer_fold), include_test=True
                )
                outer_train = np.concatenate((indices["train"], indices["validation"]))
                seed = reference.historical.full_seed(
                    int(base_seed), int(repeat), int(outer_fold)
                )
                fit_group = []
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
                        fit = read_json(fit_path)
                        completed.append(fit)
                        fit_group.append(fit)
                        continue
                    print(
                        f"IDEA057 EVAL {pipeline} base_seed={base_seed} "
                        f"repeat={repeat} fold={outer_fold} seed={seed}",
                        flush=True,
                    )
                    best_epoch, inner, _, _ = base.fit_inner(
                        pipeline,
                        protocol,
                        store,
                        indices["train"],
                        indices["validation"],
                        device,
                        seed,
                    )
                    animals, calls, outer = base.fit_outer_and_predict(
                        pipeline,
                        protocol,
                        store,
                        outer_train,
                        indices["test"],
                        best_epoch,
                        device,
                        seed,
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    animal_path = output_dir / "outer_test_animal_predictions.csv"
                    call_path = output_dir / "outer_test_call_predictions.csv"
                    animals.to_csv(animal_path, index=False)
                    calls.to_csv(call_path, index=False)
                    fit = {
                        "status": "complete",
                        "stage": "idea057_initial_evaluation",
                        "outer_test_accessed": True,
                        "pipeline": pipeline,
                        "selected_recipe_id": lock["selected_recipe_id"],
                        "base_seed": int(base_seed),
                        "repeat": int(repeat),
                        "outer_fold": int(outer_fold),
                        "full_seed": int(seed),
                        "selected_epoch": int(best_epoch),
                        "inner": inner,
                        "outer": outer,
                        "animal_prediction_path": repo_relative(animal_path),
                        "call_prediction_path": repo_relative(call_path),
                    }
                    write_json(fit_path, fit)
                    completed.append(fit)
                    fit_group.append(fit)
                order_audits.append(
                    {
                        "base_seed": int(base_seed),
                        "repeat": int(repeat),
                        "outer_fold": int(outer_fold),
                        **base.assert_batch_orders(fit_group),
                    }
                )
    summary = base.aggregate_evaluation(evaluation_root, protocol)
    add_idea057_gate(summary, evaluation_root, protocol)
    summary["batch_order_audit"] = {
        "fold_groups": len(order_audits),
        "all_common_epoch_hashes_match": True,
        "details": order_audits,
    }
    inventory = base.previous.raw_prediction_inventory(evaluation_root)
    inventory_path = evaluation_root / "raw_prediction_inventory.json"
    write_json(inventory_path, inventory)
    summary["raw_prediction_inventory"] = {
        "path": repo_relative(inventory_path),
        "inventory_sha256": sha256(inventory_path),
        "files": inventory["files"],
        "bytes": inventory["bytes"],
        "aggregate_sha256": inventory["aggregate_sha256"],
    }
    summary["selected_recipe_id"] = lock["selected_recipe_id"]
    summary["selection_record_sha256"] = lock["selection_record_sha256"]
    summary["code_commit"] = lock["code_commit"]
    summary["execution_lock_sha256"] = sha256(run_root / "execution_lock.json")
    summary["environment_lock_sha256"] = sha256(run_root / "environment_lock.json")
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
            "raw_prediction_inventory_sha256": sha256(inventory_path),
            "raw_prediction_aggregate_sha256": inventory["aggregate_sha256"],
        },
    )
    result_path = REPO_ROOT / protocol["outputs"]["result_metadata"]
    write_json(result_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = (RUNS_ROOT / args.output_subdir).resolve()
    if RUNS_ROOT.resolve() not in run_root.parents:
        raise ValueError("--output-subdir must stay below runs")
    run_root.mkdir(parents=True, exist_ok=True)
    device = reference.historical.idea019.resolve_device(args.device)
    print(
        f"IDEA-057 stage={args.stage}; device={device}; "
        f"device_name={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}",
        flush=True,
    )
    if args.stage == "prepare":
        prepare_features(run_root, protocol, device, args.resume)
        return
    configure_base_runtime()
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    store = load_store(run_root)
    if args.stage == "select":
        run_selection(run_root, protocol, roles, store, device, args.resume)
    elif args.stage == "smoke":
        run_smoke(run_root, protocol, roles, store, device, args.resume)
    else:
        run_evaluation(run_root, protocol, roles, store, device, args.resume)


if __name__ == "__main__":
    main()
