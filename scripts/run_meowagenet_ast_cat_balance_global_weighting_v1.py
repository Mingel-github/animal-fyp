"""Retest frozen AST with strict global class and cat-and-class weighting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Sampler


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for local_root in (SCRIPTS_ROOT, SRC_ROOT):
    if str(local_root) not in sys.path:
        sys.path.insert(0, str(local_root))

import run_ast_finetuning as ast_base  # noqa: E402
import run_meowagenet_ast_accuracy_enhancement_v1 as historical  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_ast_cat_balance_global_weighting_v1.json"
)
PLAN_PATH = REPO_ROOT / "plan" / "AST_cat_balance_global_weighting_retest.md"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
RUNS_ROOT = REPO_ROOT / "runs"
PIPELINES = (
    "C0_global_class_balanced",
    "C1_global_cat_and_class_balanced",
)
LABEL_NAMES = ("kitten", "adult", "senior")
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "evaluate"), required=True)
    parser.add_argument(
        "--output-subdir",
        default="meowagenet_ast_cat_balance_global_weighting_v1",
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


def git_blob_sha256(revision: str, path: Path) -> str:
    content = subprocess.check_output(
        ["git", "show", f"{revision}:{repo_relative(path)}"], cwd=REPO_ROOT
    )
    return hashlib.sha256(content).hexdigest()


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol["protocol_id"] != "meowagenet-ast-cat-balance-global-weighting-v1":
        raise RuntimeError("Unexpected global-weighting protocol")
    if tuple(protocol["pipelines"]) != PIPELINES:
        raise RuntimeError("Global-weighting pipeline pair changed")
    if sha256(PLAN_PATH) != protocol["historical_scope"]["plan_sha256"]:
        raise RuntimeError("Global-weighting plan checksum mismatch")
    if sha256(ROLES_PATH) != protocol["splits"]["roles_sha256"]:
        raise RuntimeError("Formal nested-role checksum mismatch")
    fixed = protocol["fixed_training"]
    if int(fixed["micro_batch_size"]) * int(
        fixed["gradient_accumulation_steps"]
    ) != int(fixed["accumulation_window_calls"]):
        raise RuntimeError("Accumulation window does not equal micro-batch times steps")
    dependencies = protocol["dependencies"]
    paths = {
        "accuracy_runner": SCRIPTS_ROOT
        / "run_meowagenet_ast_accuracy_enhancement_v1.py",
        "ast_base_runner": SCRIPTS_ROOT / "run_ast_finetuning.py",
        "fbank": historical.idea019.FEATURE_PATH,
        "final_embedding": historical.idea019.FROZEN_EMBEDDING_PATH,
    }
    for key, path in paths.items():
        if not path.is_file() or sha256(path) != dependencies[f"{key}_sha256"]:
            raise RuntimeError(f"Global-weighting dependency checksum mismatch: {path}")
    evaluation = protocol["evaluation"]
    expected_fits = (
        len(PIPELINES)
        * len(evaluation["base_seeds"])
        * len(evaluation["repeats"])
        * len(evaluation["outer_folds"])
    )
    if expected_fits != int(evaluation["total_outer_fits"]):
        raise RuntimeError("Global-weighting fit budget is inconsistent")


class DeterministicNoSingletonBatchSampler(Sampler[list[int]]):
    """Shuffle deterministically while preserving every call and avoiding size one."""

    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        if size < 2:
            raise ValueError("Training split must contain at least two calls")
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))

    def __iter__(self) -> Iterator[list[int]]:
        order = torch.randperm(self.size, generator=self.generator).tolist()
        batches = [
            order[start : start + self.batch_size]
            for start in range(0, self.size, self.batch_size)
        ]
        if len(batches) > 1 and len(batches[-1]) == 1:
            donor = batches[-2].pop()
            batches[-1].insert(0, donor)
        if any(len(batch) == 1 for batch in batches):
            raise RuntimeError("Singleton training batch remains after redistribution")
        yield from batches

    def __len__(self) -> int:
        return math.ceil(self.size / self.batch_size)


def build_train_loader(
    store: Any, indices: np.ndarray, batch_size: int, seed: int
) -> DataLoader:
    dataset = ast_base.CallDataset(store, indices, "frozen")
    sampler = DeterministicNoSingletonBatchSampler(len(indices), batch_size, seed)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=ast_base.collate_calls,
    )


def global_class_balanced_call_weights(
    labels: np.ndarray, train_indices: np.ndarray
) -> np.ndarray:
    train_labels = labels[train_indices]
    counts = np.bincount(train_labels, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("A training split is missing an age class")
    lookup = np.zeros(len(labels), dtype=np.float32)
    for class_index in range(3):
        class_calls = train_indices[train_labels == class_index]
        lookup[class_calls] = len(train_indices) / (3.0 * len(class_calls))
    if not np.isclose(lookup[train_indices].mean(), 1.0, atol=1.0e-6):
        raise RuntimeError("Global class weights do not have mean one")
    return lookup


def global_cat_and_class_balanced_call_weights(
    labels: np.ndarray, cat_ids: np.ndarray, train_indices: np.ndarray
) -> np.ndarray:
    lookup = np.zeros(len(labels), dtype=np.float32)
    for class_index in range(3):
        class_calls = train_indices[labels[train_indices] == class_index]
        class_cats = np.unique(cat_ids[class_calls])
        if not len(class_cats):
            raise RuntimeError("A training split is missing an age class")
        for cat_id in class_cats:
            cat_calls = class_calls[cat_ids[class_calls] == cat_id]
            lookup[cat_calls] = len(train_indices) / (
                3.0 * len(class_cats) * len(cat_calls)
            )
    if not np.isclose(lookup[train_indices].mean(), 1.0, atol=1.0e-6):
        raise RuntimeError("Global cat-and-class weights do not have mean one")
    return lookup


def pipeline_call_weights(
    pipeline: str, store: Any, train_indices: np.ndarray
) -> np.ndarray:
    if pipeline == PIPELINES[0]:
        return global_class_balanced_call_weights(store.labels, train_indices)
    if pipeline == PIPELINES[1]:
        return global_cat_and_class_balanced_call_weights(
            store.labels, store.cat_ids, train_indices
        )
    raise ValueError(f"Unknown pipeline: {pipeline}")


def target_weight_audit(
    lookup: np.ndarray, store: Any, train_indices: np.ndarray
) -> dict[str, Any]:
    class_totals = {
        LABEL_NAMES[class_index]: float(
            lookup[train_indices[store.labels[train_indices] == class_index]].sum()
        )
        for class_index in range(3)
    }
    cat_totals = {
        str(cat_id): float(lookup[train_indices[store.cat_ids[train_indices] == cat_id]].sum())
        for cat_id in sorted(np.unique(store.cat_ids[train_indices]).tolist())
    }
    return {
        "calls": int(len(train_indices)),
        "weight_total": float(lookup[train_indices].sum()),
        "weight_mean": float(lookup[train_indices].mean()),
        "weight_min": float(lookup[train_indices].min()),
        "weight_max": float(lookup[train_indices].max()),
        "weight_by_class": class_totals,
        "weight_by_cat": cat_totals,
    }


def global_weighted_micro_loss(
    per_call_loss: torch.Tensor,
    call_weights: torch.Tensor,
    accumulation_window_calls: int,
) -> torch.Tensor:
    return (per_call_loss * call_weights).sum() / float(accumulation_window_calls)


def effective_coefficient_audit(
    processed_indices: list[int],
    batch_sizes: list[int],
    lookup: np.ndarray,
    store: Any,
    accumulation_window_calls: int,
    accumulation_steps: int,
) -> dict[str, Any]:
    indices = np.asarray(processed_indices, dtype=np.int64)
    if len(indices) != len(np.unique(indices)):
        raise RuntimeError("An epoch processed a training call more than once")
    coefficients = lookup[indices].astype(np.float64) / float(accumulation_window_calls)
    class_totals = {
        LABEL_NAMES[class_index]: float(
            coefficients[store.labels[indices] == class_index].sum()
        )
        for class_index in range(3)
    }
    cat_totals = {
        str(cat_id): float(coefficients[store.cat_ids[indices] == cat_id].sum())
        for cat_id in sorted(np.unique(store.cat_ids[indices]).tolist())
    }
    final_group = batch_sizes[-accumulation_steps:]
    return {
        "processed_calls": int(len(indices)),
        "unique_processed_calls": int(len(np.unique(indices))),
        "batch_count": int(len(batch_sizes)),
        "batch_sizes": [int(value) for value in batch_sizes],
        "optimizer_steps": int(math.ceil(len(batch_sizes) / accumulation_steps)),
        "final_window_calls": int(sum(final_group)),
        "final_window_is_partial": bool(
            len(final_group) < accumulation_steps
            or sum(final_group) != accumulation_window_calls
        ),
        "effective_coefficient_total": float(coefficients.sum()),
        "effective_coefficient_by_class": class_totals,
        "effective_coefficient_by_cat": cat_totals,
        "batch_order_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
    }


def build_model(protocol: dict[str, Any], store: Any, train_indices: np.ndarray) -> torch.nn.Module:
    embeddings = store.frozen_embeddings[train_indices]
    head = historical.idea019.ClassificationHead(
        mean=embeddings.mean(axis=0),
        scale=embeddings.std(axis=0),
        dropout=float(protocol["fixed_training"]["dropout"]),
    )
    return historical.idea019.FrozenClassifier(head)


def training_values(protocol: dict[str, Any]) -> dict[str, Any]:
    fixed = protocol["fixed_training"]
    return {
        "learning_rate": float(fixed["head_learning_rate"]),
        "optimizer_epsilon": float(fixed["optimizer_epsilon"]),
        "batch_size": int(fixed["micro_batch_size"]),
        "accumulation_steps": int(fixed["gradient_accumulation_steps"]),
        "accumulation_window_calls": int(fixed["accumulation_window_calls"]),
        "gradient_clip": float(fixed["gradient_clip"]),
        "max_epochs": int(fixed["maximum_epochs"]),
        "patience": int(fixed["early_stopping_patience"]),
    }


def train_one_epoch_global_weighted(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    call_weight_lookup: torch.Tensor,
    call_weight_lookup_numpy: np.ndarray,
    store: Any,
    device: torch.device,
    values: dict[str, Any],
    scaler: torch.cuda.amp.GradScaler,
) -> tuple[float, dict[str, Any]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    weighted_loss_total = 0.0
    weight_total = 0.0
    processed_indices: list[int] = []
    batch_sizes: list[int] = []
    for step, cpu_batch in enumerate(loader):
        cpu_call_indices = cpu_batch["call_indices"].numpy().astype(np.int64).tolist()
        processed_indices.extend(cpu_call_indices)
        batch_sizes.append(len(cpu_call_indices))
        batch = historical.idea019.move_batch(cpu_batch, device)
        labels = batch["labels"]
        weights = call_weight_lookup[batch["call_indices"]]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(batch["instances"], batch["instance_to_call"], len(labels))
            per_call_loss = torch.nn.functional.cross_entropy(
                logits, labels, reduction="none"
            )
            loss = global_weighted_micro_loss(
                per_call_loss, weights, values["accumulation_window_calls"]
            )
        scaler.scale(loss).backward()
        should_step = (
            (step + 1) % values["accumulation_steps"] == 0
            or step + 1 == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), values["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        weighted_loss_total += float((per_call_loss.detach() * weights).sum())
        weight_total += float(weights.sum())
    audit = effective_coefficient_audit(
        processed_indices,
        batch_sizes,
        call_weight_lookup_numpy,
        store,
        values["accumulation_window_calls"],
        values["accumulation_steps"],
    )
    return weighted_loss_total / weight_total, audit


def fit_inner(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[int, dict[str, Any]]:
    historical.set_seed(seed)
    values = training_values(protocol)
    model = build_model(protocol, store, train_indices).to(device)
    counts = historical.idea019.trainable_counts(model)
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=values["learning_rate"],
        eps=values["optimizer_epsilon"],
    )
    lookup_numpy = pipeline_call_weights(pipeline, store, train_indices)
    lookup = torch.from_numpy(lookup_numpy).to(device)
    train_loader = build_train_loader(store, train_indices, values["batch_size"], seed)
    validation_loader = historical.idea019.build_loader(
        store, validation_indices, "frozen", values["batch_size"], False, seed
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_loss = float("inf")
    best_epoch = 1
    best_metrics: dict[str, Any] = {}
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, values["max_epochs"] + 1):
        train_loss, coefficient_audit = train_one_epoch_global_weighted(
            model,
            train_loader,
            optimizer,
            lookup,
            lookup_numpy,
            store,
            device,
            values,
            scaler,
        )
        if coefficient_audit["processed_calls"] != len(train_indices):
            raise RuntimeError("Inner epoch did not process every training call")
        validation_loss, validation_frame = historical.idea019.predict_calls(
            model, validation_loader, store, device
        )
        validation_metrics, _ = historical.idea019.evaluate_frame(validation_frame)
        history.append(
            {
                "epoch": epoch,
                "train_global_weighted_loss": train_loss,
                "validation_unweighted_call_loss": validation_loss,
                "validation_animal_macro_f1": validation_metrics["macro_f1"],
                "validation_animal_balanced_accuracy": validation_metrics[
                    "balanced_accuracy"
                ],
                "validation_animal_qwk": validation_metrics[
                    "quadratic_weighted_kappa"
                ],
                "effective_coefficients": coefficient_audit,
            }
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_metrics = validation_metrics
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
        "history": history,
        "target_weight_audit": target_weight_audit(lookup_numpy, store, train_indices),
        "train_seconds": time.perf_counter() - started,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "parameters": counts,
        "loss_implementation": "fixed_global_denominator_32",
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, audit


def fit_outer_and_predict(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    epochs: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    historical.set_seed(seed)
    values = training_values(protocol)
    model = build_model(protocol, store, train_indices).to(device)
    counts = historical.idea019.trainable_counts(model)
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=values["learning_rate"],
        eps=values["optimizer_epsilon"],
    )
    lookup_numpy = pipeline_call_weights(pipeline, store, train_indices)
    lookup = torch.from_numpy(lookup_numpy).to(device)
    train_loader = build_train_loader(store, train_indices, values["batch_size"], seed)
    test_loader = historical.idea019.build_loader(
        store, test_indices, "frozen", values["batch_size"], False, seed
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        train_loss, coefficient_audit = train_one_epoch_global_weighted(
            model,
            train_loader,
            optimizer,
            lookup,
            lookup_numpy,
            store,
            device,
            values,
            scaler,
        )
        if coefficient_audit["processed_calls"] != len(train_indices):
            raise RuntimeError("Outer epoch did not process every training call")
        history.append(
            {
                "epoch": epoch,
                "train_global_weighted_loss": train_loss,
                "effective_coefficients": coefficient_audit,
            }
        )
        print(
            f"{pipeline} outer epoch={epoch}/{epochs} train={train_loss:.4f}",
            flush=True,
        )
    test_loss, test_frame = historical.idea019.predict_calls(
        model, test_loader, store, device
    )
    test_metrics, _ = historical.idea019.evaluate_frame(test_frame)
    audit = {
        "epochs": epochs,
        "history": history,
        "test_unweighted_call_loss": test_loss,
        "test_animal_metrics": test_metrics,
        "target_weight_audit": target_weight_audit(lookup_numpy, store, train_indices),
        "train_and_predict_seconds": time.perf_counter() - started,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "parameters": counts,
        "loss_implementation": "fixed_global_denominator_32",
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_frame, audit


def assert_paired_batch_orders(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, int]:
    compared = {"inner_epochs": 0, "outer_epochs": 0}
    for section in ("inner", "outer"):
        first_history = first[section]["history"]
        second_history = second[section]["history"]
        common = min(len(first_history), len(second_history))
        for index in range(common):
            first_hash = first_history[index]["effective_coefficients"][
                "batch_order_sha256"
            ]
            second_hash = second_history[index]["effective_coefficients"][
                "batch_order_sha256"
            ]
            if first_hash != second_hash:
                raise RuntimeError(f"Paired {section} batch orders differ at epoch {index + 1}")
        compared[f"{section}_epochs"] = common
    return compared


def environment_lock(device: torch.device) -> dict[str, Any]:
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
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "git_revision": git_revision(),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "plan_sha256": sha256(PLAN_PATH),
        "roles_sha256": sha256(ROLES_PATH),
    }


def run_smoke(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    smoke_root = run_root / "smoke"
    lock_path = run_root / "execution_lock.json"
    if lock_path.is_file() and not resume:
        raise FileExistsError("Global-weighting execution lock exists; pass --resume")
    smoke = protocol["smoke"]
    indices = historical.fold_indices(
        store,
        roles,
        int(smoke["repeat"]),
        int(smoke["outer_fold"]),
        include_test=False,
    )
    seed = historical.full_seed(
        int(smoke["base_seed"]), int(smoke["repeat"]), int(smoke["outer_fold"])
    )
    values = training_values(protocol)
    fits: list[dict[str, Any]] = []
    for pipeline in PIPELINES:
        fit_path = smoke_root / "fits" / pipeline / "fit_summary.json"
        if fit_path.is_file():
            if not resume:
                raise FileExistsError(fit_path)
            fits.append(read_json(fit_path))
            continue
        historical.set_seed(seed)
        model = build_model(protocol, store, indices["train"]).to(device)
        optimizer = torch.optim.Adamax(
            model.parameters(),
            lr=values["learning_rate"],
            eps=values["optimizer_epsilon"],
        )
        lookup_numpy = pipeline_call_weights(pipeline, store, indices["train"])
        lookup = torch.from_numpy(lookup_numpy).to(device)
        loader = build_train_loader(
            store, indices["train"], values["batch_size"], seed
        )
        validation_loader = historical.idea019.build_loader(
            store,
            indices["validation"],
            "frozen",
            values["batch_size"],
            False,
            seed,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        history = []
        for epoch in range(1, int(smoke["epochs"]) + 1):
            train_loss, coefficient_audit = train_one_epoch_global_weighted(
                model,
                loader,
                optimizer,
                lookup,
                lookup_numpy,
                store,
                device,
                values,
                scaler,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_global_weighted_loss": train_loss,
                    "effective_coefficients": coefficient_audit,
                }
            )
        validation_loss, before_frame = historical.idea019.predict_calls(
            model, validation_loader, store, device
        )
        metrics, animals = historical.idea019.evaluate_frame(before_frame)
        checkpoint_path = smoke_root / "fits" / pipeline / "checkpoint.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "epochs": smoke["epochs"]}, checkpoint_path)
        reloaded = build_model(protocol, store, indices["train"]).to(device)
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        reloaded.load_state_dict(payload["model"])
        _, after_frame = historical.idea019.predict_calls(
            reloaded, validation_loader, store, device
        )
        maximum_difference = float(
            np.max(
                np.abs(
                    before_frame[list(PROBABILITY_COLUMNS)].to_numpy()
                    - after_frame[list(PROBABILITY_COLUMNS)].to_numpy()
                )
            )
        )
        if maximum_difference > 1.0e-7:
            raise RuntimeError("Smoke checkpoint reload changed validation probabilities")
        prediction_path = smoke_root / "fits" / pipeline / "validation_predictions.csv"
        before_frame.to_csv(prediction_path, index=False)
        fit = {
            "status": "complete",
            "stage": "inner_only_checkpoint_reload_smoke",
            "outer_test_accessed": False,
            "excluded_from_formal_summary": True,
            "pipeline": pipeline,
            "base_seed": int(smoke["base_seed"]),
            "repeat": int(smoke["repeat"]),
            "outer_fold": int(smoke["outer_fold"]),
            "full_seed": seed,
            "epochs": int(smoke["epochs"]),
            "history": history,
            "validation_unweighted_call_loss": validation_loss,
            "validation_animal_metrics": metrics,
            "validation_animals": int(len(animals)),
            "checkpoint_path": repo_relative(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_reload_max_probability_difference": maximum_difference,
            "target_weight_audit": target_weight_audit(
                lookup_numpy, store, indices["train"]
            ),
        }
        write_json(fit_path, fit)
        fits.append(fit)
        del model, reloaded, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pair_audit = {
        "inner_epochs": 0,
        "outer_epochs": 0,
    }
    first_history = fits[0]["history"]
    second_history = fits[1]["history"]
    for index in range(min(len(first_history), len(second_history))):
        if (
            first_history[index]["effective_coefficients"]["batch_order_sha256"]
            != second_history[index]["effective_coefficients"]["batch_order_sha256"]
        ):
            raise RuntimeError("Smoke paired batch order mismatch")
        pair_audit["inner_epochs"] += 1
    environment_path = run_root / "environment_lock.json"
    if not environment_path.is_file():
        write_json(environment_path, environment_lock(device))
    if lock_path.is_file():
        lock = read_json(lock_path)
        if lock["protocol_sha256"] != sha256(PROTOCOL_PATH) or lock[
            "runner_sha256"
        ] != sha256(Path(__file__)):
            raise RuntimeError("Global-weighting execution-lock hash mismatch")
    else:
        lock = {
            "schema_version": "1.0",
            "status": "locked_for_global_weighting_evaluation",
            "outer_test_accessed": False,
            "protocol_id": protocol["protocol_id"],
            "code_commit": git_revision(),
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "runner_sha256": sha256(Path(__file__)),
            "plan_sha256": sha256(PLAN_PATH),
            "roles_sha256": sha256(ROLES_PATH),
            "environment_lock_sha256": sha256(environment_path),
            "pipelines": list(PIPELINES),
            "evaluation": protocol["evaluation"],
            "fixed_training": protocol["fixed_training"],
            "smoke_fit_sha256": {
                fit["pipeline"]: sha256(
                    smoke_root / "fits" / fit["pipeline"] / "fit_summary.json"
                )
                for fit in fits
            },
        }
        write_json(lock_path, lock)
    summary = {
        "status": "complete",
        "outer_test_accessed": False,
        "excluded_from_formal_summary": True,
        "completed_fits": len(fits),
        "paired_batch_order_audit": pair_audit,
        "checkpoint_reload_passed": all(
            fit["checkpoint_reload_max_probability_difference"] <= 1.0e-7
            for fit in fits
        ),
        "execution_lock": repo_relative(lock_path),
        "execution_lock_sha256": sha256(lock_path),
        "environment_lock": repo_relative(environment_path),
        "environment_lock_sha256": sha256(environment_path),
        "pipelines": {
            fit["pipeline"]: {
                "validation_animal_macro_f1": fit["validation_animal_metrics"][
                    "macro_f1"
                ],
                "validation_unweighted_call_loss": fit[
                    "validation_unweighted_call_loss"
                ],
                "checkpoint_reload_max_probability_difference": fit[
                    "checkpoint_reload_max_probability_difference"
                ],
            }
            for fit in fits
        },
    }
    write_json(smoke_root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def verify_execution_lock(run_root: Path) -> dict[str, Any]:
    lock_path = run_root / "execution_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("Run global-weighting smoke before evaluation")
    lock = read_json(lock_path)
    if lock["status"] != "locked_for_global_weighting_evaluation":
        raise RuntimeError("Unexpected global-weighting execution-lock status")
    checks = {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "plan_sha256": sha256(PLAN_PATH),
        "roles_sha256": sha256(ROLES_PATH),
        "environment_lock_sha256": sha256(run_root / "environment_lock.json"),
    }
    for key, value in checks.items():
        if lock[key] != value:
            raise RuntimeError(f"Global-weighting execution-lock mismatch: {key}")
    if git_blob_sha256(lock["code_commit"], Path(__file__)) != lock["runner_sha256"]:
        raise RuntimeError("Locked code commit does not contain the locked runner")
    if git_blob_sha256(lock["code_commit"], PROTOCOL_PATH) != lock["protocol_sha256"]:
        raise RuntimeError("Locked code commit does not contain the locked protocol")
    return lock


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    macro_f1 = [float(row["macro_f1"]) for row in rows]
    return {
        "macro_f1_mean": float(np.mean(macro_f1)),
        "macro_f1_sample_sd": float(np.std(macro_f1, ddof=1)),
        "balanced_accuracy_mean": float(
            np.mean([row["balanced_accuracy"] for row in rows])
        ),
        "qwk_mean": float(
            np.mean([row["quadratic_weighted_kappa"] for row in rows])
        ),
        "plain_accuracy_mean": float(
            np.mean(
                [
                    np.trace(np.asarray(row["confusion_matrix"])) / int(row["n"])
                    for row in rows
                ]
            )
        ),
        "mean_per_class_recall": {
            label: float(np.mean([row["per_class_recall"][label] for row in rows]))
            for label in LABEL_NAMES
        },
        "macro_f1_range": [float(min(macro_f1)), float(max(macro_f1))],
    }


def paired_cat_bootstrap(
    animals_by_key: dict[tuple[str, int, int], pd.DataFrame],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    keys = [
        (int(base_seed), int(repeat))
        for base_seed in protocol["evaluation"]["base_seeds"]
        for repeat in protocol["evaluation"]["repeats"]
    ]
    reference = animals_by_key[(PIPELINES[0], *keys[0])].sort_values("cat_id")
    cat_ids = reference["cat_id"].astype(str).to_numpy()
    labels = reference["true_label"].to_numpy(dtype=np.int64)
    predictions: dict[str, list[np.ndarray]] = {pipeline: [] for pipeline in PIPELINES}
    for pipeline in PIPELINES:
        for base_seed, repeat in keys:
            frame = animals_by_key[(pipeline, base_seed, repeat)].sort_values("cat_id")
            if frame["cat_id"].astype(str).tolist() != cat_ids.tolist():
                raise RuntimeError("Bootstrap cat order mismatch")
            if not np.array_equal(frame["true_label"].to_numpy(), labels):
                raise RuntimeError("Bootstrap label mismatch")
            predictions[pipeline].append(frame["predicted_label"].to_numpy(dtype=np.int64))
    bootstrap = protocol["bootstrap"]
    rng = np.random.default_rng(int(bootstrap["seed"]))
    deltas = np.empty(int(bootstrap["iterations"]), dtype=np.float64)
    for iteration in range(len(deltas)):
        sampled = rng.integers(0, len(cat_ids), size=len(cat_ids))
        pair_deltas = []
        for pair_index in range(len(keys)):
            c0 = f1_score(
                labels[sampled],
                predictions[PIPELINES[0]][pair_index][sampled],
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
            c1 = f1_score(
                labels[sampled],
                predictions[PIPELINES[1]][pair_index][sampled],
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
            pair_deltas.append(c1 - c0)
        deltas[iteration] = float(np.mean(pair_deltas))
    lower, upper = np.quantile(deltas, bootstrap["interval"])
    return {
        "method": bootstrap["method"],
        "resampling_unit": "cat_id",
        "cats_per_resample": int(len(cat_ids)),
        "paired_oof_comparisons_per_resample": int(len(keys)),
        "iterations": int(len(deltas)),
        "seed": int(bootstrap["seed"]),
        "interval_quantiles": bootstrap["interval"],
        "macro_f1_difference_interval": [float(lower), float(upper)],
        "bootstrap_mean_difference": float(np.mean(deltas)),
        "fraction_above_zero": float(np.mean(deltas > 0)),
    }


def safe_spearman(first: np.ndarray, second: np.ndarray) -> dict[str, float | None]:
    result = spearmanr(first, second)
    statistic = float(result.statistic)
    pvalue = float(result.pvalue)
    return {
        "rho": statistic if np.isfinite(statistic) else None,
        "pvalue": pvalue if np.isfinite(pvalue) else None,
    }


def call_count_association(
    animals_by_key: dict[tuple[str, int, int], pd.DataFrame], store: Any
) -> dict[str, Any]:
    call_counts = pd.Series(store.cat_ids).value_counts().to_dict()
    output = {}
    for pipeline in PIPELINES:
        parts = []
        for (candidate, base_seed, repeat), frame in animals_by_key.items():
            if candidate != pipeline:
                continue
            current = frame.copy()
            current["base_seed"] = base_seed
            current["repeat"] = repeat
            current["confidence"] = current[list(PROBABILITY_COLUMNS)].max(axis=1)
            current["correct"] = (
                current["predicted_label"] == current["true_label"]
            ).astype(float)
            parts.append(current)
        stacked = pd.concat(parts, ignore_index=True)
        per_cat = stacked.groupby("cat_id", as_index=False).agg(
            mean_confidence=("confidence", "mean"),
            repeat_accuracy=("correct", "mean"),
        )
        per_cat["call_count"] = per_cat["cat_id"].map(call_counts).astype(int)
        output[pipeline] = {
            "cats": int(len(per_cat)),
            "call_count_vs_mean_confidence": safe_spearman(
                per_cat["call_count"].to_numpy(),
                per_cat["mean_confidence"].to_numpy(),
            ),
            "call_count_vs_repeat_accuracy": safe_spearman(
                per_cat["call_count"].to_numpy(),
                per_cat["repeat_accuracy"].to_numpy(),
            ),
        }
    return output


def raw_prediction_inventory(evaluation_root: Path) -> dict[str, Any]:
    paths = sorted(
        list((evaluation_root / "fits").rglob("outer_test_call_predictions.csv"))
        + list((evaluation_root / "oof").rglob("*_animals.csv"))
        + [evaluation_root / "paired_prediction_changes_C1_vs_C0.csv"]
    )
    entries = [
        {
            "path": repo_relative(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        }
        for path in paths
    ]
    canonical = "".join(
        f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n" for entry in entries
    ).encode("utf-8")
    return {
        "schema_version": "1.0",
        "files": len(entries),
        "bytes": int(sum(entry["bytes"] for entry in entries)),
        "aggregate_sha256": hashlib.sha256(canonical).hexdigest(),
        "entries": entries,
    }


def aggregate_evaluation(
    evaluation_root: Path, protocol: dict[str, Any], store: Any
) -> dict[str, Any]:
    evaluation = protocol["evaluation"]
    metrics_by_pipeline: dict[str, list[dict[str, Any]]] = {
        pipeline: [] for pipeline in PIPELINES
    }
    animals_by_key: dict[tuple[str, int, int], pd.DataFrame] = {}
    for base_seed in evaluation["base_seeds"]:
        for repeat in evaluation["repeats"]:
            for pipeline in PIPELINES:
                parts = []
                for outer_fold in evaluation["outer_folds"]:
                    path = (
                        evaluation_root
                        / "fits"
                        / pipeline
                        / f"base_seed_{base_seed}"
                        / f"repeat_{repeat}"
                        / f"fold_{outer_fold}"
                        / "outer_test_call_predictions.csv"
                    )
                    parts.append(pd.read_csv(path, dtype={"cat_id": str}))
                calls = pd.concat(parts, ignore_index=True).sort_values("call_index")
                if len(calls) != 792 or calls["call_index"].nunique() != 792:
                    raise RuntimeError("Complete OOF must cover all 792 calls")
                metrics, animals = historical.idea019.evaluate_frame(calls)
                if len(animals) != 111 or animals["cat_id"].nunique() != 111:
                    raise RuntimeError("Complete OOF must cover all 111 cats")
                row = {"base_seed": int(base_seed), "repeat": int(repeat), **metrics}
                metrics_by_pipeline[pipeline].append(row)
                output = (
                    evaluation_root
                    / "oof"
                    / pipeline
                    / f"base_seed_{base_seed}_repeat_{repeat}_animals.csv"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                animals.to_csv(output, index=False)
                animals_by_key[(pipeline, int(base_seed), int(repeat))] = animals
    paired = []
    changes = []
    for base_seed in evaluation["base_seeds"]:
        for repeat in evaluation["repeats"]:
            c0_metric = next(
                row
                for row in metrics_by_pipeline[PIPELINES[0]]
                if row["base_seed"] == base_seed and row["repeat"] == repeat
            )
            c1_metric = next(
                row
                for row in metrics_by_pipeline[PIPELINES[1]]
                if row["base_seed"] == base_seed and row["repeat"] == repeat
            )
            left = animals_by_key[(PIPELINES[0], int(base_seed), int(repeat))][
                ["cat_id", "true_label", "predicted_label"]
            ].rename(columns={"predicted_label": "c0_prediction"})
            right = animals_by_key[(PIPELINES[1], int(base_seed), int(repeat))][
                ["cat_id", "true_label", "predicted_label"]
            ].rename(columns={"predicted_label": "c1_prediction"})
            merged = left.merge(right, on=["cat_id", "true_label"], validate="one_to_one")
            merged["c0_correct"] = merged["c0_prediction"] == merged["true_label"]
            merged["c1_correct"] = merged["c1_prediction"] == merged["true_label"]
            changed = merged[merged["c0_prediction"] != merged["c1_prediction"]].copy()
            changed.insert(0, "repeat", repeat)
            changed.insert(0, "base_seed", base_seed)
            changes.append(changed)
            paired.append(
                {
                    "base_seed": int(base_seed),
                    "repeat": int(repeat),
                    "C0_macro_f1": float(c0_metric["macro_f1"]),
                    "C1_macro_f1": float(c1_metric["macro_f1"]),
                    "C1_minus_C0_macro_f1": float(
                        c1_metric["macro_f1"] - c0_metric["macro_f1"]
                    ),
                    "changed_animals": int(len(changed)),
                    "gained_correct_animals": int(
                        ((~changed["c0_correct"]) & changed["c1_correct"]).sum()
                    ),
                    "lost_correct_animals": int(
                        (changed["c0_correct"] & (~changed["c1_correct"])).sum()
                    ),
                }
            )
    change_path = evaluation_root / "paired_prediction_changes_C1_vs_C0.csv"
    pd.concat(changes, ignore_index=True).to_csv(change_path, index=False)
    aggregate = {
        pipeline: mean_metrics(rows) for pipeline, rows in metrics_by_pipeline.items()
    }
    differences = [float(row["C1_minus_C0_macro_f1"]) for row in paired]
    per_seed = {}
    for base_seed in evaluation["base_seeds"]:
        values = [
            row["C1_minus_C0_macro_f1"]
            for row in paired
            if row["base_seed"] == base_seed
        ]
        per_seed[str(base_seed)] = {
            "paired_macro_f1_differences": values,
            "mean_difference": float(np.mean(values)),
            "positive_repeats": int(sum(value > 0 for value in values)),
        }
    bootstrap = paired_cat_bootstrap(animals_by_key, protocol)
    mean_difference = float(np.mean(differences))
    positive_comparisons = int(sum(value > 0 for value in differences))
    positive_seed_means = int(
        sum(value["mean_difference"] > 0 for value in per_seed.values())
    )
    lower = float(bootstrap["macro_f1_difference_interval"][0])
    stable = (
        mean_difference >= 0.010
        and positive_seed_means >= 2
        and positive_comparisons >= 6
        and lower > 0
    )
    no_improvement = mean_difference <= 0 and positive_seed_means <= 1
    interpretation = (
        "stable_improvement"
        if stable
        else "no_improvement_evidence"
        if no_improvement
        else "small_or_mixed"
    )
    return {
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "completed_outer_fits": int(evaluation["total_outer_fits"]),
        "complete_oof": metrics_by_pipeline,
        "aggregate": aggregate,
        "paired_C1_vs_C0": paired,
        "paired_summary": {
            "macro_f1_differences": differences,
            "mean_C1_minus_C0_macro_f1": mean_difference,
            "positive_comparisons": positive_comparisons,
            "per_seed": per_seed,
        },
        "paired_cat_hierarchical_bootstrap": bootstrap,
        "call_count_association": call_count_association(animals_by_key, store),
        "stage_decision": {
            "rule_category": interpretation,
            "stable_improvement": stable,
            "no_improvement_evidence": no_improvement,
            "positive_base_seed_means": positive_seed_means,
            "rules": protocol["interpretation_rules"],
        },
        "artifacts": {
            "paired_prediction_changes": repo_relative(change_path),
        },
    }


def run_evaluation(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    lock = verify_execution_lock(run_root)
    evaluation_root = run_root / "evaluation"
    summary_path = evaluation_root / "summary.json"
    if summary_path.is_file():
        if not resume:
            raise FileExistsError("Global-weighting evaluation exists; pass --resume")
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return
    write_json(
        evaluation_root / "run_manifest.json",
        {
            "schema_version": "1.0",
            "protocol_id": protocol["protocol_id"],
            "stage": "evaluate",
            "outer_test_accessed": True,
            "code_commit": lock["code_commit"],
            "protocol_sha256": lock["protocol_sha256"],
            "runner_sha256": lock["runner_sha256"],
            "execution_lock_sha256": sha256(run_root / "execution_lock.json"),
            "environment_lock_sha256": sha256(run_root / "environment_lock.json"),
            "device": str(device),
        },
    )
    evaluation = protocol["evaluation"]
    completed = []
    pair_audits = []
    for base_seed in evaluation["base_seeds"]:
        for repeat in evaluation["repeats"]:
            for outer_fold in evaluation["outer_folds"]:
                indices = historical.fold_indices(
                    store, roles, int(repeat), int(outer_fold), include_test=True
                )
                outer_train = np.concatenate((indices["train"], indices["validation"]))
                seed = historical.full_seed(
                    int(base_seed), int(repeat), int(outer_fold)
                )
                pair = []
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
                        pair.append(fit)
                        completed.append(fit)
                        continue
                    print(
                        f"GLOBAL EVAL {pipeline} base_seed={base_seed} "
                        f"repeat={repeat} fold={outer_fold} seed={seed}",
                        flush=True,
                    )
                    best_epoch, inner = fit_inner(
                        pipeline,
                        protocol,
                        store,
                        indices["train"],
                        indices["validation"],
                        device,
                        seed,
                    )
                    test_frame, outer = fit_outer_and_predict(
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
                    prediction_path = output_dir / "outer_test_call_predictions.csv"
                    test_frame.to_csv(prediction_path, index=False)
                    fit = {
                        "status": "complete",
                        "stage": "locked_global_weighting_evaluation",
                        "outer_test_accessed": True,
                        "pipeline": pipeline,
                        "base_seed": int(base_seed),
                        "repeat": int(repeat),
                        "outer_fold": int(outer_fold),
                        "full_seed": seed,
                        "selected_epoch": best_epoch,
                        "inner": inner,
                        "outer": outer,
                        "prediction_path": repo_relative(prediction_path),
                    }
                    write_json(fit_path, fit)
                    pair.append(fit)
                    completed.append(fit)
                compared = assert_paired_batch_orders(pair[0], pair[1])
                pair_audits.append(
                    {
                        "base_seed": int(base_seed),
                        "repeat": int(repeat),
                        "outer_fold": int(outer_fold),
                        **compared,
                    }
                )
    summary = aggregate_evaluation(evaluation_root, protocol, store)
    summary["paired_batch_order_audit"] = {
        "pairs": len(pair_audits),
        "all_common_epoch_hashes_match": True,
        "details": pair_audits,
    }
    inventory = raw_prediction_inventory(evaluation_root)
    inventory_path = evaluation_root / "raw_prediction_inventory.json"
    write_json(inventory_path, inventory)
    summary["raw_prediction_inventory"] = {
        "path": repo_relative(inventory_path),
        "inventory_sha256": sha256(inventory_path),
        "files": inventory["files"],
        "bytes": inventory["bytes"],
        "aggregate_sha256": inventory["aggregate_sha256"],
    }
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
    store = historical.idea019.load_feature_store()
    device = historical.idea019.resolve_device(args.device)
    print(
        f"AST global weighting stage={args.stage}; device={device}; "
        f"device_name={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}",
        flush=True,
    )
    if args.stage == "smoke":
        run_smoke(run_root, protocol, roles, store, device, args.resume)
    else:
        run_evaluation(run_root, protocol, roles, store, device, args.resume)


if __name__ == "__main__":
    main()
