"""Run the seed-17 Cat-balanced Multi-Layer AST exploratory matrix.

The runner has three explicit stages:

* ``diagnose`` summarizes already-known model errors and inner-only layer probes;
* ``smoke`` trains A0--A3 on repeat 0 / fold 0 without requesting outer-test calls,
  then writes a hash-bound execution lock;
* ``evaluate`` verifies that lock and produces three complete seed-17 OOF runs.

Historical formal, HPO, and backbone-screening artifacts remain read-only inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for local_root in (SCRIPTS_ROOT, SRC_ROOT):
    if str(local_root) not in sys.path:
        sys.path.insert(0, str(local_root))

import run_idea019_peft_placement as idea019  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_ast_accuracy_enhancement_v1.json"
)
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
RUNS_ROOT = REPO_ROOT / "runs"
PIPELINES = (
    "A0_final_class_balanced",
    "A1_final_cat_balanced",
    "A2_scalar_fusion_class_balanced",
    "A3_scalar_fusion_cat_balanced",
)
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
LABEL_NAMES = ("kitten", "adult", "senior")
HPO_ROOT = RUNS_ROOT / "meowagenet_ast_hpo_v1"
FORMAL_ROOT = RUNS_ROOT / "meowagenet_formal_v2_1_core"
PANNS_ROOT = RUNS_ROOT / "idea049_panns_cnn14_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("diagnose", "smoke", "evaluate"), required=True)
    parser.add_argument(
        "--output-subdir", default="meowagenet_ast_accuracy_enhancement_v1"
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


def full_seed(base_seed: int, repeat: int, outer_fold: int) -> int:
    return int(base_seed + 10_000 * repeat + 100 * outer_fold)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol["protocol_id"] != "meowagenet-ast-accuracy-enhancement-v1":
        raise RuntimeError("Unexpected AST accuracy-enhancement protocol")
    if tuple(protocol["pipelines"]) != PIPELINES:
        raise RuntimeError("Protocol A0-A3 pipeline order changed")
    if sha256(ROLES_PATH) != protocol["splits"]["roles_sha256"]:
        raise RuntimeError("Formal nested-role checksum mismatch")
    dependencies = protocol["dependencies"]
    keyed_paths = {
        "fbank": idea019.FEATURE_PATH,
        "final_embedding": idea019.FROZEN_EMBEDDING_PATH,
        "layer_embedding": idea019.LAYER_EMBEDDING_PATH,
        "ast_hpo_summary": HPO_ROOT / "evaluation" / "summary.json",
        "panns_summary": PANNS_ROOT / "initial" / "initial_summary.json",
    }
    for key, path in keyed_paths.items():
        expected = dependencies[f"{key}_sha256"]
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Dependency checksum mismatch: {path}")
    manifest_path = REPO_ROOT / protocol["dataset"]["manifest_path"]
    if sha256(manifest_path) != protocol["dataset"]["manifest_sha256"]:
        raise RuntimeError("Dataset manifest checksum mismatch")
    evaluation = protocol["exploratory_evaluation"]
    expected_fits = len(PIPELINES) * len(evaluation["repeats"]) * len(
        evaluation["outer_folds"]
    )
    if expected_fits != int(evaluation["total_fits"]):
        raise RuntimeError("Evaluation fit budget is inconsistent")


def stage_manifest(stage: str, protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "stage": stage,
        "outer_test_accessed": stage == "evaluate",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "roles_sha256": sha256(ROLES_PATH),
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


def fold_indices(
    store: Any,
    roles: pd.DataFrame,
    repeat: int,
    outer_fold: int,
    include_test: bool,
) -> dict[str, np.ndarray]:
    fold_roles = roles[
        (roles["repeat"] == repeat) & (roles["outer_fold"] == outer_fold)
    ]
    mapping = dict(zip(fold_roles["cat_id"], fold_roles["role"], strict=True))
    if len(mapping) != 111:
        raise RuntimeError("Each fold must assign all 111 cats")
    call_roles = np.asarray([mapping[cat_id] for cat_id in store.cat_ids])
    selected = {
        "train": np.flatnonzero(call_roles == "train"),
        "validation": np.flatnonzero(call_roles == "validation"),
    }
    if include_test:
        selected["test"] = np.flatnonzero(call_roles == "test")
    train_cats = set(store.cat_ids[selected["train"]])
    validation_cats = set(store.cat_ids[selected["validation"]])
    if train_cats & validation_cats:
        raise RuntimeError("Cat-ID leakage between train and validation roles")
    if include_test:
        test_cats = set(store.cat_ids[selected["test"]])
        if (train_cats | validation_cats) & test_cats:
            raise RuntimeError("Cat-ID leakage into outer test role")
    return selected


class ScalarFusionClassifier(nn.Module):
    """Learn one softmax-normalized scalar for each frozen AST layer."""

    def __init__(self, head: nn.Module, layer_count: int = 12) -> None:
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(layer_count, dtype=torch.float32))
        self.head = head

    def forward(
        self, instances: torch.Tensor, instance_to_call: torch.Tensor, call_count: int
    ) -> torch.Tensor:
        del instance_to_call, call_count
        if instances.ndim != 3 or instances.shape[1:] != (12, 768):
            raise RuntimeError(
                f"Expected frozen layer embeddings with shape (batch, 12, 768), got {tuple(instances.shape)}"
            )
        weights = torch.softmax(self.layer_logits, dim=0)
        fused = (instances * weights[None, :, None]).sum(dim=1)
        return self.head(fused)

    def layer_weights(self) -> list[float]:
        return (
            torch.softmax(self.layer_logits.detach().float(), dim=0).cpu().numpy().tolist()
        )


def is_cat_balanced(pipeline: str) -> bool:
    return pipeline in ("A1_final_cat_balanced", "A3_scalar_fusion_cat_balanced")


def uses_scalar_fusion(pipeline: str) -> bool:
    return pipeline in (
        "A2_scalar_fusion_class_balanced",
        "A3_scalar_fusion_cat_balanced",
    )


def cat_balanced_call_weights(
    labels: np.ndarray, cat_ids: np.ndarray, train_indices: np.ndarray
) -> np.ndarray:
    """Return global lookup weights with equal class and within-class cat totals."""

    labels = np.asarray(labels, dtype=np.int64)
    cat_ids = np.asarray(cat_ids, dtype=str)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    selected_labels = labels[train_indices]
    selected_cats = cat_ids[train_indices]
    n_calls = len(train_indices)
    lookup = np.zeros(len(labels), dtype=np.float32)
    for class_index in range(3):
        class_mask = selected_labels == class_index
        class_cats = np.unique(selected_cats[class_mask])
        if len(class_cats) == 0:
            raise RuntimeError(f"Training role is missing class {class_index}")
        for cat_id in class_cats:
            local_mask = class_mask & (selected_cats == cat_id)
            call_indices = train_indices[local_mask]
            per_call = n_calls / (3.0 * len(class_cats) * len(call_indices))
            lookup[call_indices] = per_call
    if not np.isclose(lookup[train_indices].mean(), 1.0, atol=1.0e-6):
        raise RuntimeError("Cat-balanced call weights do not have mean one")
    return lookup


def build_pipeline_model(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    layer_store: Any,
    train_indices: np.ndarray,
) -> tuple[nn.Module, Any]:
    dropout = float(protocol["fixed_training"]["dropout"])
    if uses_scalar_fusion(pipeline):
        initial_fused = layer_store.embeddings[train_indices].mean(axis=1)
        head = idea019.ClassificationHead(
            mean=initial_fused.mean(axis=0),
            scale=initial_fused.std(axis=0),
            dropout=dropout,
        )
        model = ScalarFusionClassifier(head, layer_count=layer_store.embeddings.shape[1])
        active_store = replace(store, frozen_embeddings=layer_store.embeddings)
        return model, active_store
    embeddings = store.frozen_embeddings[train_indices]
    head = idea019.ClassificationHead(
        mean=embeddings.mean(axis=0),
        scale=embeddings.std(axis=0),
        dropout=dropout,
    )
    return idea019.FrozenClassifier(head), store


def train_one_epoch_cat_balanced(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    call_weight_lookup: torch.Tensor,
    device: torch.device,
    accumulation_steps: int,
    gradient_clip: float,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    weighted_loss_total = 0.0
    weight_total = 0.0
    for step, cpu_batch in enumerate(loader):
        batch = idea019.move_batch(cpu_batch, device)
        labels = batch["labels"]
        weights = call_weight_lookup[batch["call_indices"]]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(
                batch["instances"], batch["instance_to_call"], len(labels)
            )
            per_call_loss = torch_functional.cross_entropy(
                logits, labels, reduction="none"
            )
            raw_loss = (per_call_loss * weights).sum() / weights.sum()
            loss = raw_loss / accumulation_steps
        scaler.scale(loss).backward()
        should_step = (step + 1) % accumulation_steps == 0 or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        weighted_loss_total += float((per_call_loss.detach() * weights).sum())
        weight_total += float(weights.sum())
    return weighted_loss_total / weight_total


def current_layer_weights(model: nn.Module) -> list[float] | None:
    if isinstance(model, ScalarFusionClassifier):
        return model.layer_weights()
    return None


def training_values(protocol: dict[str, Any]) -> dict[str, Any]:
    fixed = protocol["fixed_training"]
    return {
        "learning_rate": float(fixed["head_and_fusion_learning_rate"]),
        "batch_size": int(fixed["micro_batch_size"]),
        "accumulation_steps": int(fixed["gradient_accumulation_steps"]),
        "max_epochs": int(fixed["maximum_epochs"]),
        "patience": int(fixed["early_stopping_patience"]),
        "gradient_clip": float(fixed["gradient_clip"]),
        "optimizer_epsilon": float(fixed["optimizer_epsilon"]),
    }


def run_training_epoch(
    pipeline: str,
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    cat_weights: torch.Tensor | None,
    device: torch.device,
    values: dict[str, Any],
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    if is_cat_balanced(pipeline):
        if cat_weights is None:
            raise RuntimeError("Cat-balanced pipeline has no call-weight lookup")
        return train_one_epoch_cat_balanced(
            model,
            loader,
            optimizer,
            cat_weights,
            device,
            values["accumulation_steps"],
            values["gradient_clip"],
            scaler,
        )
    return idea019.train_one_epoch(
        model,
        loader,
        optimizer,
        class_weights,
        device,
        values["accumulation_steps"],
        values["gradient_clip"],
        scaler,
    )


def fit_inner(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    layer_store: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[int, dict[str, Any]]:
    set_seed(seed)
    values = training_values(protocol)
    model, active_store = build_pipeline_model(
        pipeline, protocol, store, layer_store, train_indices
    )
    model = model.to(device)
    counts = idea019.trainable_counts(model)
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=values["learning_rate"],
        eps=values["optimizer_epsilon"],
    )
    class_weights = idea019.class_weights(store.labels[train_indices]).to(device)
    cat_weights = None
    if is_cat_balanced(pipeline):
        cat_weights = torch.from_numpy(
            cat_balanced_call_weights(store.labels, store.cat_ids, train_indices)
        ).to(device)
    train_loader = idea019.build_loader(
        active_store, train_indices, "frozen", values["batch_size"], True, seed
    )
    validation_loader = idea019.build_loader(
        active_store,
        validation_indices,
        "frozen",
        values["batch_size"],
        False,
        seed,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_loss = float("inf")
    best_epoch = 1
    best_metrics: dict[str, Any] = {}
    best_layer_weights = None
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, values["max_epochs"] + 1):
        train_loss = run_training_epoch(
            pipeline,
            model,
            train_loader,
            optimizer,
            class_weights,
            cat_weights,
            device,
            values,
            scaler,
        )
        validation_loss, validation_frame = idea019.predict_calls(
            model, validation_loader, active_store, device
        )
        validation_metrics, _ = idea019.evaluate_frame(validation_frame)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_balanced_accuracy": validation_metrics[
                    "balanced_accuracy"
                ],
                "validation_qwk": validation_metrics[
                    "quadratic_weighted_kappa"
                ],
                "layer_weights": current_layer_weights(model),
            }
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_metrics = validation_metrics
            best_layer_weights = current_layer_weights(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"{pipeline} inner epoch={epoch} train={train_loss:.4f} "
            f"val={validation_loss:.4f} val_F1={validation_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if epochs_without_improvement >= values["patience"]:
            break
    audit = {
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "best_validation_loss": best_loss,
        "best_validation_animal_metrics": best_metrics,
        "best_layer_weights": best_layer_weights,
        "history": history,
        "train_seconds": time.perf_counter() - started,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "parameters": counts,
        "loss_balance": "cat_and_class" if is_cat_balanced(pipeline) else "class",
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, audit


def fit_outer_and_predict(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    layer_store: Any,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    epochs: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    set_seed(seed)
    values = training_values(protocol)
    model, active_store = build_pipeline_model(
        pipeline, protocol, store, layer_store, train_indices
    )
    model = model.to(device)
    counts = idea019.trainable_counts(model)
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=values["learning_rate"],
        eps=values["optimizer_epsilon"],
    )
    class_weights = idea019.class_weights(store.labels[train_indices]).to(device)
    cat_weights = None
    if is_cat_balanced(pipeline):
        cat_weights = torch.from_numpy(
            cat_balanced_call_weights(store.labels, store.cat_ids, train_indices)
        ).to(device)
    train_loader = idea019.build_loader(
        active_store, train_indices, "frozen", values["batch_size"], True, seed
    )
    test_loader = idea019.build_loader(
        active_store, test_indices, "frozen", values["batch_size"], False, seed
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    training_losses = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        train_loss = run_training_epoch(
            pipeline,
            model,
            train_loader,
            optimizer,
            class_weights,
            cat_weights,
            device,
            values,
            scaler,
        )
        training_losses.append(train_loss)
        print(
            f"{pipeline} outer epoch={epoch}/{epochs} train={train_loss:.4f}",
            flush=True,
        )
    test_loss, test_frame = idea019.predict_calls(
        model, test_loader, active_store, device
    )
    test_metrics, _ = idea019.evaluate_frame(test_frame)
    audit = {
        "epochs": epochs,
        "training_losses": training_losses,
        "test_loss": test_loss,
        "test_animal_metrics": test_metrics,
        "final_layer_weights": current_layer_weights(model),
        "train_and_predict_seconds": time.perf_counter() - started,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "parameters": counts,
        "loss_balance": "cat_and_class" if is_cat_balanced(pipeline) else "class",
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_frame, audit


def safe_spearman(first: pd.Series, second: pd.Series) -> dict[str, float | None]:
    result = spearmanr(first.to_numpy(dtype=float), second.to_numpy(dtype=float))
    statistic = float(result.statistic)
    pvalue = float(result.pvalue)
    return {
        "rho": statistic if np.isfinite(statistic) else None,
        "pvalue": pvalue if np.isfinite(pvalue) else None,
    }


def read_animal_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"cat_id": str})
    if "predicted_label" not in frame:
        frame["predicted_label"] = frame[list(PROBABILITY_COLUMNS)].to_numpy().argmax(
            axis=1
        )
    return frame


def error_overlap(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    left = reference[["cat_id", "true_label", "predicted_label"]].rename(
        columns={"predicted_label": "reference_prediction"}
    )
    right = candidate[["cat_id", "true_label", "predicted_label"]].rename(
        columns={"predicted_label": "candidate_prediction"}
    )
    merged = left.merge(right, on=["cat_id", "true_label"], validate="one_to_one")
    if len(merged) != 111:
        raise RuntimeError("Historical error-overlap comparison must contain 111 cats")
    reference_correct = merged["reference_prediction"] == merged["true_label"]
    candidate_correct = merged["candidate_prediction"] == merged["true_label"]
    return {
        "both_correct": int((reference_correct & candidate_correct).sum()),
        "reference_only_correct": int((reference_correct & ~candidate_correct).sum()),
        "candidate_only_correct": int((~reference_correct & candidate_correct).sum()),
        "both_wrong": int((~reference_correct & ~candidate_correct).sum()),
        "reference_only_cat_ids": merged.loc[
            reference_correct & ~candidate_correct, "cat_id"
        ].tolist(),
        "candidate_only_cat_ids": merged.loc[
            ~reference_correct & candidate_correct, "cat_id"
        ].tolist(),
    }


def run_diagnostics(
    run_root: Path, protocol: dict[str, Any], store: Any, device: torch.device
) -> None:
    output_root = run_root / "diagnostics"
    summary_path = output_root / "summary.json"
    if summary_path.is_file():
        raise FileExistsError(summary_path)
    write_json(output_root / "run_manifest.json", stage_manifest("diagnose", protocol, device))
    fbank = np.load(idea019.FEATURE_PATH)
    durations = fbank["durations"].astype(np.float64)
    segment_counts = fbank["segment_counts"].astype(np.int64)
    segment_seconds = float(protocol["diagnostics"]["fixed_segment_seconds"])
    effective_ratio = np.minimum(
        durations / (segment_counts * segment_seconds), 1.0
    )
    call_frame = pd.DataFrame(
        {
            "cat_id": store.cat_ids,
            "duration_seconds": durations,
            "effective_frame_ratio": effective_ratio,
            "padding_ratio": 1.0 - effective_ratio,
        }
    )
    cat_characteristics = (
        call_frame.groupby("cat_id", as_index=False)
        .agg(
            call_count=("cat_id", "size"),
            mean_duration_seconds=("duration_seconds", "mean"),
            maximum_duration_seconds=("duration_seconds", "max"),
            mean_effective_frame_ratio=("effective_frame_ratio", "mean"),
            mean_padding_ratio=("padding_ratio", "mean"),
        )
        .sort_values("cat_id")
    )
    historical_rows = []
    overlaps: dict[str, Any] = {}
    for repeat in range(3):
        ast = read_animal_predictions(
            HPO_ROOT
            / "evaluation"
            / "oof"
            / "ast_head_only"
            / f"repeat_{repeat}_animals.csv"
        )
        ast = ast.merge(cat_characteristics, on="cat_id", validate="one_to_one")
        ast["repeat"] = repeat
        ast["correct"] = ast["predicted_label"] == ast["true_label"]
        ast["confidence"] = ast[list(PROBABILITY_COLUMNS)].max(axis=1)
        historical_rows.append(ast)
        vggish = read_animal_predictions(
            FORMAL_ROOT
            / "oof"
            / "vggish_mlp"
            / f"repeat_{repeat}_base_seed_17_animal_predictions.csv"
        )
        panns = read_animal_predictions(
            PANNS_ROOT
            / "initial"
            / "oof"
            / f"repeat_{repeat}_base_seed_17_animal_predictions.csv"
        )
        overlaps[f"repeat_{repeat}"] = {
            "tuned_ast_vs_vggish": error_overlap(ast, vggish),
            "tuned_ast_vs_panns_cnn14": error_overlap(ast, panns),
        }
    historical = pd.concat(historical_rows, ignore_index=True)
    per_cat = (
        historical.groupby("cat_id", as_index=False)
        .agg(
            true_label=("true_label", "first"),
            correct_repeats=("correct", "sum"),
            repeat_accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
            call_count=("call_count", "first"),
            mean_duration_seconds=("mean_duration_seconds", "first"),
            maximum_duration_seconds=("maximum_duration_seconds", "first"),
            mean_effective_frame_ratio=("mean_effective_frame_ratio", "first"),
            mean_padding_ratio=("mean_padding_ratio", "first"),
        )
        .sort_values("cat_id")
    )
    per_cat["age_group"] = per_cat["true_label"].map(dict(enumerate(LABEL_NAMES)))
    per_cat["call_count_bucket"] = pd.cut(
        per_cat["call_count"],
        bins=[0, 1, 5, 10, 20, np.inf],
        labels=["1", "2-5", "6-10", "11-20", "21+"],
    ).astype(str)
    per_cat.to_csv(output_root / "tuned_ast_animal_diagnostics.csv", index=False)
    bucket_summary = (
        per_cat.groupby("call_count_bucket", observed=True)
        .agg(
            cats=("cat_id", "size"),
            mean_repeat_accuracy=("repeat_accuracy", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            mean_padding_ratio=("mean_padding_ratio", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    bucket_order = {
        label: index
        for index, label in enumerate(("1", "2-5", "6-10", "11-20", "21+"))
    }
    bucket_summary.sort(key=lambda row: bucket_order[row["call_count_bucket"]])
    probe_paths = sorted((HPO_ROOT / "probes").glob("repeat_*_fold_*.json"))
    if len(probe_paths) != 12:
        raise RuntimeError("Expected 12 existing inner-only AST layer probes")
    probes = [read_json(path) for path in probe_paths]
    layer_probe_summary = []
    for layer_index in range(12):
        selected_count = sum(
            layer_index in probe["selected_layers_zero_based"] for probe in probes
        )
        layer_probe_summary.append(
            {
                "layer_one_based": layer_index + 1,
                "mean_inner_cv_macro_f1": float(
                    np.mean(
                        [probe["layer_mean_macro_f1"][layer_index] for probe in probes]
                    )
                ),
                "selected_in_top_two_folds": int(selected_count),
            }
        )
    layer_probe_summary.sort(
        key=lambda row: (-row["mean_inner_cv_macro_f1"], row["layer_one_based"])
    )
    summary = {
        "status": "complete",
        "scope": {
            "calls": int(len(store.call_ids)),
            "cats": int(len(per_cat)),
            "historical_tuned_ast_repeats": 3,
        },
        "call_count_buckets": bucket_summary,
        "correlations": {
            "call_count_vs_repeat_accuracy": safe_spearman(
                per_cat["call_count"], per_cat["repeat_accuracy"]
            ),
            "call_count_vs_confidence": safe_spearman(
                per_cat["call_count"], per_cat["mean_confidence"]
            ),
            "mean_duration_vs_repeat_accuracy": safe_spearman(
                per_cat["mean_duration_seconds"], per_cat["repeat_accuracy"]
            ),
            "mean_padding_ratio_vs_repeat_accuracy": safe_spearman(
                per_cat["mean_padding_ratio"], per_cat["repeat_accuracy"]
            ),
        },
        "persistent_errors": per_cat.loc[
            per_cat["correct_repeats"] == 0,
            [
                "cat_id",
                "age_group",
                "call_count",
                "mean_duration_seconds",
                "mean_padding_ratio",
            ],
        ].to_dict(orient="records"),
        "error_overlap": overlaps,
        "inner_only_layer_probe_ranking": layer_probe_summary,
        "artifacts": {
            "animal_table": repo_relative(
                output_root / "tuned_ast_animal_diagnostics.csv"
            ),
            "historical_outer_use": "descriptive diagnosis only",
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


def verify_or_write_execution_lock(
    run_root: Path,
    protocol: dict[str, Any],
    smoke_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    lock_path = run_root / "execution_lock.json"
    expected_runner_hash = sha256(Path(__file__))
    expected_protocol_hash = sha256(PROTOCOL_PATH)
    if lock_path.is_file():
        lock = read_json(lock_path)
        if (
            lock["runner_sha256"] != expected_runner_hash
            or lock["protocol_sha256"] != expected_protocol_hash
        ):
            raise RuntimeError("Existing execution lock does not match runner/protocol")
        return lock
    reference_path = (
        HPO_ROOT
        / "evaluation"
        / "fits"
        / "ast_head_only"
        / "repeat_0"
        / "fold_0"
        / "base_seed_17"
        / "fit_summary.json"
    )
    reference = read_json(reference_path)["inner"]
    a0 = next(
        summary for summary in smoke_summaries if summary["pipeline"] == PIPELINES[0]
    )["inner"]
    metric_difference = float(
        a0["best_validation_animal_metrics"]["macro_f1"]
        - reference["best_validation_animal_metrics"]["macro_f1"]
    )
    loss_difference = float(
        a0["best_validation_loss"] - reference["best_validation_loss"]
    )
    if a0["best_epoch"] != reference["best_epoch"]:
        raise RuntimeError("A0 smoke best epoch does not reproduce tuned AST head-only")
    if abs(metric_difference) > 1.0e-9 or abs(loss_difference) > 1.0e-7:
        raise RuntimeError("A0 smoke metrics do not reproduce tuned AST head-only")
    lock = {
        "schema_version": "1.0",
        "status": "locked_for_seed17_exploratory_evaluation",
        "outer_test_accessed": False,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": expected_protocol_hash,
        "runner_sha256": expected_runner_hash,
        "roles_sha256": sha256(ROLES_PATH),
        "pipelines": list(PIPELINES),
        "training": protocol["fixed_training"],
        "scalar_fusion": protocol["scalar_fusion"],
        "evaluation": protocol["exploratory_evaluation"],
        "smoke_fit_sha256": {
            summary["pipeline"]: sha256(Path(summary["summary_path_absolute"]))
            for summary in smoke_summaries
        },
        "a0_reproduction": {
            "reference": repo_relative(reference_path),
            "best_epoch": int(a0["best_epoch"]),
            "macro_f1_difference": metric_difference,
            "validation_loss_difference": loss_difference,
        },
    }
    write_json(lock_path, lock)
    return lock


def run_smoke(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    layer_store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    smoke_root = run_root / "smoke"
    lock_path = run_root / "execution_lock.json"
    if lock_path.is_file() and not resume:
        raise FileExistsError("Execution lock already exists; pass --resume to verify")
    write_json(smoke_root / "run_manifest.json", stage_manifest("smoke", protocol, device))
    smoke = protocol["smoke"]
    repeat = int(smoke["repeat"])
    outer_fold = int(smoke["outer_fold"])
    base_seed = int(smoke["base_seed"])
    seed = full_seed(base_seed, repeat, outer_fold)
    indices = fold_indices(store, roles, repeat, outer_fold, include_test=False)
    completed = []
    for pipeline in PIPELINES:
        output_dir = smoke_root / "fits" / pipeline
        summary_path = output_dir / "fit_summary.json"
        if summary_path.is_file():
            if not resume:
                raise FileExistsError(summary_path)
            summary = read_json(summary_path)
        else:
            print(f"SMOKE {pipeline} repeat={repeat} fold={outer_fold} seed={seed}", flush=True)
            best_epoch, inner = fit_inner(
                pipeline,
                protocol,
                store,
                layer_store,
                indices["train"],
                indices["validation"],
                device,
                seed,
            )
            summary = {
                "status": "complete",
                "stage": "inner_only_smoke",
                "outer_test_accessed": False,
                "pipeline": pipeline,
                "repeat": repeat,
                "outer_fold": outer_fold,
                "base_seed": base_seed,
                "full_seed": seed,
                "best_epoch": best_epoch,
                "train_calls": int(len(indices["train"])),
                "validation_calls": int(len(indices["validation"])),
                "inner": inner,
            }
            write_json(summary_path, summary)
        summary["summary_path_absolute"] = str(summary_path.resolve())
        completed.append(summary)
    lock = verify_or_write_execution_lock(run_root, protocol, completed)
    smoke_summary = {
        "status": "complete",
        "outer_test_accessed": False,
        "completed_inner_fits": len(completed),
        "execution_lock": repo_relative(lock_path),
        "execution_lock_sha256": sha256(lock_path),
        "a0_reproduction": lock["a0_reproduction"],
        "pipelines": {
            summary["pipeline"]: {
                "best_epoch": summary["best_epoch"],
                "best_validation_loss": summary["inner"]["best_validation_loss"],
                "best_validation_animal_metrics": summary["inner"][
                    "best_validation_animal_metrics"
                ],
                "best_layer_weights": summary["inner"]["best_layer_weights"],
            }
            for summary in completed
        },
    }
    write_json(smoke_root / "summary.json", smoke_summary)
    print(json.dumps(smoke_summary, indent=2), flush=True)


def verify_execution_lock(run_root: Path) -> dict[str, Any]:
    lock_path = run_root / "execution_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("Run the inner-only smoke stage before evaluation")
    lock = read_json(lock_path)
    if lock["status"] != "locked_for_seed17_exploratory_evaluation":
        raise RuntimeError("Execution lock has unexpected status")
    if lock["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise RuntimeError("Protocol changed after execution lock")
    if lock["runner_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("Runner changed after execution lock")
    return lock


def aggregate_evaluation(
    evaluation_root: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    evaluation = protocol["exploratory_evaluation"]
    base_seed = int(evaluation["base_seed"])
    metrics_by_pipeline: dict[str, list[dict[str, Any]]] = {}
    animals_by_pipeline_repeat: dict[tuple[str, int], pd.DataFrame] = {}
    for pipeline in PIPELINES:
        rows = []
        for repeat in evaluation["repeats"]:
            parts = []
            for outer_fold in evaluation["outer_folds"]:
                prediction_path = (
                    evaluation_root
                    / "fits"
                    / pipeline
                    / f"repeat_{repeat}"
                    / f"fold_{outer_fold}"
                    / f"base_seed_{base_seed}"
                    / "outer_test_call_predictions.csv"
                )
                parts.append(pd.read_csv(prediction_path, dtype={"cat_id": str}))
            calls = pd.concat(parts, ignore_index=True).sort_values("call_index")
            if len(calls) != 792 or calls["call_index"].nunique() != 792:
                raise RuntimeError("Complete OOF must cover 792 calls exactly once")
            metrics, animals = idea019.evaluate_frame(calls)
            if len(animals) != 111 or animals["cat_id"].nunique() != 111:
                raise RuntimeError("Complete OOF must cover 111 cats exactly once")
            animals.insert(0, "base_seed", base_seed)
            animals.insert(0, "repeat", repeat)
            animals.insert(0, "pipeline", pipeline)
            output = evaluation_root / "oof" / pipeline / f"repeat_{repeat}_animals.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            animals.to_csv(output, index=False)
            animals_by_pipeline_repeat[(pipeline, int(repeat))] = animals
            rows.append({"repeat": int(repeat), **metrics})
        metrics_by_pipeline[pipeline] = rows
    aggregate = {}
    for pipeline, rows in metrics_by_pipeline.items():
        macro_f1 = [float(row["macro_f1"]) for row in rows]
        aggregate[pipeline] = {
            "macro_f1_mean": float(np.mean(macro_f1)),
            "macro_f1_sample_sd": float(np.std(macro_f1, ddof=1)),
            "balanced_accuracy_mean": float(
                np.mean([row["balanced_accuracy"] for row in rows])
            ),
            "qwk_mean": float(
                np.mean([row["quadratic_weighted_kappa"] for row in rows])
            ),
        }
    pairwise = {}
    reference = PIPELINES[0]
    for candidate in PIPELINES[1:]:
        differences = [
            float(
                metrics_by_pipeline[candidate][index]["macro_f1"]
                - metrics_by_pipeline[reference][index]["macro_f1"]
            )
            for index in range(len(evaluation["repeats"]))
        ]
        pairwise[f"{candidate}_minus_{reference}"] = {
            "macro_f1_differences": differences,
            "mean_macro_f1_difference": float(np.mean(differences)),
            "positive_repeats": int(sum(value > 0 for value in differences)),
        }
    changed_rows = []
    for candidate in PIPELINES[1:]:
        for repeat in evaluation["repeats"]:
            left = animals_by_pipeline_repeat[(reference, int(repeat))][
                ["cat_id", "true_label", "predicted_label"]
            ].rename(columns={"predicted_label": "a0_prediction"})
            right = animals_by_pipeline_repeat[(candidate, int(repeat))][
                ["cat_id", "true_label", "predicted_label"]
            ].rename(columns={"predicted_label": "candidate_prediction"})
            merged = left.merge(right, on=["cat_id", "true_label"], validate="one_to_one")
            changed = merged[merged["a0_prediction"] != merged["candidate_prediction"]].copy()
            changed.insert(0, "repeat", repeat)
            changed.insert(0, "candidate", candidate)
            changed["a0_correct"] = changed["a0_prediction"] == changed["true_label"]
            changed["candidate_correct"] = (
                changed["candidate_prediction"] == changed["true_label"]
            )
            changed_rows.append(changed)
    changed_frame = pd.concat(changed_rows, ignore_index=True)
    changed_path = evaluation_root / "paired_prediction_changes_vs_A0.csv"
    changed_frame.to_csv(changed_path, index=False)
    layer_weight_rows = []
    for pipeline in PIPELINES:
        if not uses_scalar_fusion(pipeline):
            continue
        for repeat in evaluation["repeats"]:
            for outer_fold in evaluation["outer_folds"]:
                fit_path = (
                    evaluation_root
                    / "fits"
                    / pipeline
                    / f"repeat_{repeat}"
                    / f"fold_{outer_fold}"
                    / f"base_seed_{base_seed}"
                    / "fit_summary.json"
                )
                fit = read_json(fit_path)
                weights = fit["outer"]["final_layer_weights"]
                for layer_index, weight in enumerate(weights, start=1):
                    layer_weight_rows.append(
                        {
                            "pipeline": pipeline,
                            "repeat": int(repeat),
                            "outer_fold": int(outer_fold),
                            "layer_one_based": layer_index,
                            "weight": float(weight),
                        }
                    )
    layer_weight_frame = pd.DataFrame(layer_weight_rows)
    layer_weight_path = evaluation_root / "scalar_fusion_layer_weights.csv"
    layer_weight_frame.to_csv(layer_weight_path, index=False)
    layer_weight_summary = (
        layer_weight_frame.groupby(["pipeline", "layer_one_based"], as_index=False)
        .agg(mean_weight=("weight", "mean"), sample_sd=("weight", "std"))
        .to_dict(orient="records")
    )
    ranking = sorted(
        PIPELINES,
        key=lambda pipeline: (-aggregate[pipeline]["macro_f1_mean"], pipeline),
    )
    expansion_threshold = 0.01
    expansion_candidates = []
    for candidate in PIPELINES[1:]:
        comparison = pairwise[f"{candidate}_minus_{reference}"]
        if (
            comparison["mean_macro_f1_difference"] >= expansion_threshold
            and comparison["positive_repeats"] >= 2
        ):
            expansion_candidates.append(candidate)
    historical_anchor = read_json(HPO_ROOT / "evaluation" / "summary.json")
    historical_a0 = historical_anchor["pipelines"]["ast_head_only"]
    historical_differences = [
        float(metrics_by_pipeline[reference][index]["macro_f1"] - row["macro_f1"])
        for index, row in enumerate(historical_a0)
    ]
    return {
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "completed_fits": int(evaluation["total_fits"]),
        "complete_oof": metrics_by_pipeline,
        "aggregate": aggregate,
        "ranking_by_mean_macro_f1": ranking,
        "paired_vs_A0": pairwise,
        "factorial_interaction_macro_f1": {
            "per_repeat": [
                float(
                    (
                        metrics_by_pipeline[PIPELINES[3]][index]["macro_f1"]
                        - metrics_by_pipeline[PIPELINES[2]][index]["macro_f1"]
                    )
                    - (
                        metrics_by_pipeline[PIPELINES[1]][index]["macro_f1"]
                        - metrics_by_pipeline[PIPELINES[0]][index]["macro_f1"]
                    )
                )
                for index in range(3)
            ]
        },
        "a0_vs_historical_tuned_ast": {
            "macro_f1_differences": historical_differences,
            "mean_macro_f1_difference": float(np.mean(historical_differences)),
        },
        "scalar_fusion_layer_weight_summary": layer_weight_summary,
        "stage_decision": {
            "condition": protocol["stage_decision"]["expansion_condition"],
            "expansion_candidates": expansion_candidates,
            "leading_pipeline": ranking[0],
        },
        "artifacts": {
            "paired_prediction_changes": repo_relative(changed_path),
            "scalar_fusion_layer_weights": repo_relative(layer_weight_path),
        },
    }


def run_evaluation(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    layer_store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    lock = verify_execution_lock(run_root)
    evaluation_root = run_root / "evaluation"
    summary_path = evaluation_root / "summary.json"
    if summary_path.is_file():
        if not resume:
            raise FileExistsError("Evaluation summary exists; pass --resume to verify")
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
    evaluation = protocol["exploratory_evaluation"]
    base_seed = int(evaluation["base_seed"])
    completed = []
    for repeat in evaluation["repeats"]:
        for outer_fold in evaluation["outer_folds"]:
            indices = fold_indices(
                store, roles, int(repeat), int(outer_fold), include_test=True
            )
            outer_train = np.concatenate((indices["train"], indices["validation"]))
            seed = full_seed(base_seed, int(repeat), int(outer_fold))
            for pipeline in PIPELINES:
                output_dir = (
                    evaluation_root
                    / "fits"
                    / pipeline
                    / f"repeat_{repeat}"
                    / f"fold_{outer_fold}"
                    / f"base_seed_{base_seed}"
                )
                fit_path = output_dir / "fit_summary.json"
                if fit_path.is_file():
                    if not resume:
                        raise FileExistsError(fit_path)
                    completed.append(read_json(fit_path))
                    continue
                print(
                    f"EVAL {pipeline} repeat={repeat} fold={outer_fold} seed={seed}",
                    flush=True,
                )
                best_epoch, inner = fit_inner(
                    pipeline,
                    protocol,
                    store,
                    layer_store,
                    indices["train"],
                    indices["validation"],
                    device,
                    seed,
                )
                test_frame, outer = fit_outer_and_predict(
                    pipeline,
                    protocol,
                    store,
                    layer_store,
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
                    "stage": "locked_seed17_exploratory_evaluation",
                    "outer_test_accessed": True,
                    "pipeline": pipeline,
                    "repeat": int(repeat),
                    "outer_fold": int(outer_fold),
                    "base_seed": base_seed,
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
            "expected_fits": int(evaluation["total_fits"]),
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
    store = idea019.load_feature_store()
    layer_store = idea019.load_layer_store(store)
    device = idea019.resolve_device(args.device)
    print(
        f"AST enhancement stage={args.stage}; device={device}; "
        f"device_name={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}",
        flush=True,
    )
    if args.stage == "diagnose":
        run_diagnostics(run_root, protocol, store, device)
    elif args.stage == "smoke":
        run_smoke(
            run_root,
            protocol,
            roles,
            store,
            layer_store,
            device,
            args.resume,
        )
    else:
        run_evaluation(
            run_root,
            protocol,
            roles,
            store,
            layer_store,
            device,
            args.resume,
        )


if __name__ == "__main__":
    main()
