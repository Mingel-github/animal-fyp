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
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for search_root in (SCRIPTS_ROOT, SRC_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_ast_finetuning as ast_base  # noqa: E402
import run_meowagenet_ast_accuracy_enhancement_v1 as split_utils  # noqa: E402
import run_meowagenet_ast_cat_balance_global_weighting_v1 as weighting  # noqa: E402
import run_meowagenet_idea051_cat_set as evaluation  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea058_constrained_top_block_v1.json"
)
PLAN_PATH = REPO_ROOT / "plan" / "IDEA-058_constrained_top_block_adaptation.md"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
RUNS_ROOT = REPO_ROOT / "runs"
DEFAULT_OUTPUT_SUBDIR = "meowagenet_idea058_constrained_top_block_v1"
PIPELINES = (
    "R0_tuned_frozen_ast",
    "M1_top_block_adaptation",
    "C1_bottom_block_control",
)
ONLINE_PIPELINES = PIPELINES[1:]
LABEL_NAMES = evaluation.LABEL_NAMES
PROBABILITY_COLUMNS = evaluation.PROBABILITY_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked IDEA-058 experiment")
    parser.add_argument(
        "--stage", choices=("select", "smoke", "evaluate", "all"), default="all"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
    relative = repo_relative(path)
    return subprocess.check_output(
        ["git", "rev-parse", f"{revision}:{relative}"], cwd=REPO_ROOT, text=True
    ).strip()


def worktree_blob_object_id(path: Path) -> str:
    return subprocess.check_output(
        ["git", "hash-object", str(path)], cwd=REPO_ROOT, text=True
    ).strip()


def selection_path(run_root: Path) -> Path:
    return run_root / "selection" / "selection_record.json"


def dependency_paths(protocol: dict[str, Any]) -> dict[str, Path]:
    dependencies = protocol["dependencies"]
    return {
        key.removesuffix("_path"): REPO_ROOT / value
        for key, value in dependencies.items()
        if key.endswith("_path")
    }


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol["protocol_id"] != "meowagenet-idea058-constrained-top-block-v1":
        raise RuntimeError("Unexpected IDEA-058 protocol")
    selection = protocol["inner_selection"]
    candidates = selection["candidate_recipes"]
    expected_selection_fits = (
        len(candidates)
        * len(selection["repeats"])
        * len(selection["outer_folds"])
    )
    if expected_selection_fits != int(selection["candidate_fits"]):
        raise RuntimeError("IDEA-058 candidate-fit budget is inconsistent")
    for recipe_id, recipe in candidates.items():
        if int(recipe["block_count"]) not in (1, 2):
            raise RuntimeError(f"Invalid block count for {recipe_id}")
        if float(recipe["encoder_learning_rate"]) not in (3.0e-6, 1.0e-5):
            raise RuntimeError(f"Invalid encoder learning rate for {recipe_id}")
    initial = protocol["initial_evaluation"]
    expected_outer_fits = (
        len(PIPELINES)
        * len(initial["base_seeds"])
        * len(initial["repeats"])
        * len(initial["outer_folds"])
    )
    if expected_outer_fits != int(initial["total_outer_fits"]):
        raise RuntimeError("IDEA-058 outer-fit budget is inconsistent")
    fixed = protocol["fixed_training"]
    if int(fixed["micro_batch_size"]) * int(
        fixed["gradient_accumulation_steps"]
    ) != int(fixed["accumulation_window_calls"]):
        raise RuntimeError("IDEA-058 accumulation window is inconsistent")
    checks = {
        PLAN_PATH: protocol["idea"]["sha256"],
        ROLES_PATH: protocol["splits"]["roles_sha256"],
    }
    for key, path in dependency_paths(protocol).items():
        checks[path] = protocol["dependencies"][f"{key}_sha256"]
    for path, expected in checks.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"IDEA-058 dependency checksum mismatch: {path}")


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def active_recipe(protocol: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    recipe_id = protocol.get("_active_recipe_id")
    if recipe_id is None:
        raise RuntimeError("IDEA-058 active recipe has not been selected")
    recipes = protocol["inner_selection"]["candidate_recipes"]
    if recipe_id not in recipes:
        raise RuntimeError(f"Unknown IDEA-058 recipe: {recipe_id}")
    return str(recipe_id), recipes[str(recipe_id)]


class LocatedBlockASTClassifier(nn.Module):
    def __init__(
        self,
        locked_protocol: dict[str, Any],
        pipeline: str,
        recipe: dict[str, Any],
        update_final_layernorm: bool,
        head: nn.Module,
    ) -> None:
        super().__init__()
        if pipeline not in ONLINE_PIPELINES:
            raise ValueError(pipeline)
        ast_config = locked_protocol["ast"]
        # Loading a frozen checkpoint is deterministic and should not advance the
        # classifier/dropout RNG stream used by the matched pipelines.
        with torch.random.fork_rng(devices=[]):
            self.ast = ast_base.ASTModel.from_pretrained(
                ast_config["checkpoint"],
                revision=ast_config["revision"],
                cache_dir=ast_base.HF_CACHE,
                use_safetensors=True,
            )
            self.geometry_audit = ast_base.adapt_standard_geometry(
                self.ast, locked_protocol
            )
        for parameter in self.ast.parameters():
            parameter.requires_grad = False
        layer_count = len(self.ast.encoder.layer)
        block_count = int(recipe["block_count"])
        if pipeline == PIPELINES[1]:
            block_indices = list(range(layer_count - block_count, layer_count))
        else:
            block_indices = list(range(block_count))
        for index in block_indices:
            for parameter in self.ast.encoder.layer[index].parameters():
                parameter.requires_grad = True
        if update_final_layernorm:
            for parameter in self.ast.layernorm.parameters():
                parameter.requires_grad = True
        self.pipeline = pipeline
        self.block_indices = block_indices
        self.update_final_layernorm = bool(update_final_layernorm)
        self.head = head

    def forward(
        self, instances: torch.Tensor, instance_to_call: torch.Tensor, call_count: int
    ) -> torch.Tensor:
        segment_embeddings = self.ast(input_values=instances).pooler_output
        call_embeddings = torch.zeros(
            (call_count, segment_embeddings.shape[-1]),
            dtype=segment_embeddings.dtype,
            device=segment_embeddings.device,
        )
        call_embeddings.index_add_(0, instance_to_call, segment_embeddings)
        counts = torch.bincount(instance_to_call, minlength=call_count).to(
            segment_embeddings.dtype
        )
        call_embeddings = call_embeddings / counts[:, None]
        return self.head(call_embeddings)


def build_model(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
) -> nn.Module:
    embeddings = store.frozen_embeddings[train_indices]
    head = ast_base.ClassificationHead(
        mean=embeddings.mean(axis=0),
        scale=embeddings.std(axis=0),
        dropout=float(protocol["fixed_training"]["dropout"]),
    )
    if pipeline == PIPELINES[0]:
        return ast_base.FrozenClassifier(head)
    _, recipe = active_recipe(protocol)
    locked = read_json(dependency_paths(protocol)["locked_protocol"])
    constraints = protocol["inner_selection"]["fixed_candidate_constraints"]
    return LocatedBlockASTClassifier(
        locked,
        pipeline,
        recipe,
        bool(constraints["update_final_layernorm"]),
        head,
    )


def parameter_audit(model: nn.Module) -> dict[str, Any]:
    trainable = {
        name: int(parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    encoder = {name: count for name, count in trainable.items() if name.startswith("ast.")}
    head = {name: count for name, count in trainable.items() if name.startswith("head.")}
    return {
        "trainable": int(sum(trainable.values())),
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_encoder": int(sum(encoder.values())),
        "trainable_head": int(sum(head.values())),
        "trainable_parameter_names": sorted(trainable),
        "trainable_encoder_parameter_names": sorted(encoder),
        "block_indices_zero_based": list(getattr(model, "block_indices", [])),
        "update_final_layernorm": bool(
            getattr(model, "update_final_layernorm", False)
        ),
    }


def training_values(protocol: dict[str, Any]) -> dict[str, Any]:
    fixed = protocol["fixed_training"]
    constraints = protocol["inner_selection"]["fixed_candidate_constraints"]
    _, recipe = active_recipe(protocol)
    return {
        "head_learning_rate": float(fixed["head_learning_rate"]),
        "encoder_learning_rate": float(recipe["encoder_learning_rate"]),
        "encoder_weight_decay": float(constraints["encoder_weight_decay"]),
        "head_weight_decay": float(constraints["head_weight_decay"]),
        "optimizer_epsilon": float(fixed["optimizer_epsilon"]),
        "batch_size": int(fixed["micro_batch_size"]),
        "evaluation_batch_size": int(fixed["evaluation_batch_size"]),
        "accumulation_steps": int(fixed["gradient_accumulation_steps"]),
        "accumulation_window_calls": int(fixed["accumulation_window_calls"]),
        "gradient_clip": float(constraints["gradient_clip"]),
        "max_epochs": int(fixed["maximum_epochs"]),
        "patience": int(fixed["early_stopping_patience"]),
    }


def build_loader(
    store: Any,
    indices: np.ndarray,
    pipeline: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    mode = "frozen" if pipeline == PIPELINES[0] else "online"
    dataset = ast_base.CallDataset(store, indices, mode)
    if shuffle:
        sampler = weighting.DeterministicNoSingletonBatchSampler(
            len(indices), batch_size, seed
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=ast_base.collate_calls,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ast_base.collate_calls,
    )


def make_optimizer(
    model: nn.Module, pipeline: str, values: dict[str, Any]
) -> torch.optim.Optimizer:
    if pipeline == PIPELINES[0]:
        return torch.optim.Adamax(
            model.parameters(),
            lr=values["head_learning_rate"],
            eps=values["optimizer_epsilon"],
            weight_decay=values["head_weight_decay"],
        )
    encoder_parameters = [
        parameter for parameter in model.ast.parameters() if parameter.requires_grad
    ]
    return torch.optim.Adamax(
        [
            {
                "params": encoder_parameters,
                "lr": values["encoder_learning_rate"],
                "weight_decay": values["encoder_weight_decay"],
            },
            {
                "params": model.head.parameters(),
                "lr": values["head_learning_rate"],
                "weight_decay": values["head_weight_decay"],
            },
        ],
        eps=values["optimizer_epsilon"],
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    weight_lookup: torch.Tensor,
    weight_lookup_numpy: np.ndarray,
    store: Any,
    device: torch.device,
    values: dict[str, Any],
    scaler: torch.cuda.amp.GradScaler,
) -> tuple[float, dict[str, Any]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    weighted_total = 0.0
    weight_total = 0.0
    processed_indices: list[int] = []
    batch_sizes: list[int] = []
    for step, cpu_batch in enumerate(loader):
        indices = cpu_batch["call_indices"].numpy().astype(np.int64).tolist()
        processed_indices.extend(indices)
        batch_sizes.append(len(indices))
        batch = ast_base.move_batch(cpu_batch, device)
        weights = weight_lookup[batch["call_indices"]]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(
                batch["instances"], batch["instance_to_call"], len(batch["labels"])
            )
            per_call_loss = torch.nn.functional.cross_entropy(
                logits, batch["labels"], reduction="none"
            )
            loss = weighting.global_weighted_micro_loss(
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
        weighted_total += float((per_call_loss.detach() * weights).sum())
        weight_total += float(weights.sum())
    audit = weighting.effective_coefficient_audit(
        processed_indices,
        batch_sizes,
        weight_lookup_numpy,
        store,
        values["accumulation_window_calls"],
        values["accumulation_steps"],
    )
    return weighted_total / weight_total, audit


def portable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            state[name] = parameter.detach().cpu().clone()
    for name, buffer in model.named_buffers():
        if name.startswith("head."):
            state[name] = buffer.detach().cpu().clone()
    return state


def load_portable_state_dict(
    model: nn.Module, state: dict[str, torch.Tensor]
) -> None:
    targets: dict[str, torch.Tensor] = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    missing = sorted(set(state) - set(targets))
    if missing:
        raise RuntimeError(f"Portable checkpoint has unknown tensors: {missing}")
    with torch.no_grad():
        for name, value in state.items():
            targets[name].copy_(value.to(targets[name].device))


def fit_inner(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
    max_epochs_override: int | None = None,
) -> tuple[int, dict[str, Any], dict[str, torch.Tensor], pd.DataFrame]:
    split_utils.set_seed(seed)
    values = training_values(protocol)
    maximum_epochs = int(max_epochs_override or values["max_epochs"])
    model = build_model(pipeline, protocol, store, train_indices).to(device)
    parameters = parameter_audit(model)
    optimizer = make_optimizer(model, pipeline, values)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    weights_numpy = weighting.global_class_balanced_call_weights(
        store.labels, train_indices
    )
    weights = torch.from_numpy(weights_numpy).to(device)
    train_loader = build_loader(
        store, train_indices, pipeline, values["batch_size"], True, seed
    )
    validation_loader = build_loader(
        store,
        validation_indices,
        pipeline,
        values["evaluation_batch_size"],
        False,
        seed,
    )
    best_loss = float("inf")
    best_epoch = 1
    best_metrics: dict[str, Any] = {}
    best_state = portable_state_dict(model)
    history = []
    epochs_without_improvement = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, maximum_epochs + 1):
        train_loss, train_audit = train_one_epoch(
            model,
            train_loader,
            optimizer,
            weights,
            weights_numpy,
            store,
            device,
            values,
            scaler,
        )
        call_loss, validation_calls = ast_base.predict_calls(
            model, validation_loader, store, device
        )
        validation_animals = evaluation.calls_to_animals(validation_calls)
        animal_loss = evaluation.animal_cross_entropy(validation_animals)
        metrics = evaluation.animal_metrics(validation_animals)
        history.append(
            {
                "epoch": int(epoch),
                "train_weighted_loss": float(train_loss),
                "validation_call_cross_entropy": float(call_loss),
                "validation_animal_cross_entropy": float(animal_loss),
                "validation_animal_macro_f1": metrics["macro_f1"],
                "validation_animal_balanced_accuracy": metrics["balanced_accuracy"],
                "validation_animal_qwk": metrics["quadratic_weighted_kappa"],
                "validation_plain_accuracy": metrics["plain_accuracy"],
                "train_unit_audit": train_audit,
            }
        )
        if animal_loss < best_loss - 1.0e-6:
            best_loss = animal_loss
            best_epoch = epoch
            best_metrics = metrics
            best_state = portable_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"IDEA058 {pipeline} inner epoch={epoch} train={train_loss:.4f} "
            f"animal_CE={animal_loss:.4f} animal_F1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if max_epochs_override is None and epochs_without_improvement >= values["patience"]:
            break
    load_portable_state_dict(model, best_state)
    _, best_calls = ast_base.predict_calls(model, validation_loader, store, device)
    audit = {
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(len(history)),
        "best_validation_animal_cross_entropy": float(best_loss),
        "best_validation_animal_metrics": best_metrics,
        "history": history,
        "target_weight_audit": weighting.target_weight_audit(
            weights_numpy, store, train_indices
        ),
        "train_seconds": float(time.perf_counter() - started),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "parameters": parameters,
        "checkpoint_selection": "minimum_unweighted_animal_cross_entropy",
    }
    del model, optimizer, weights
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, audit, best_state, best_calls


def fit_outer_and_predict(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    epochs: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    split_utils.set_seed(seed)
    values = training_values(protocol)
    model = build_model(pipeline, protocol, store, train_indices).to(device)
    parameters = parameter_audit(model)
    optimizer = make_optimizer(model, pipeline, values)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    weights_numpy = weighting.global_class_balanced_call_weights(
        store.labels, train_indices
    )
    weights = torch.from_numpy(weights_numpy).to(device)
    train_loader = build_loader(
        store, train_indices, pipeline, values["batch_size"], True, seed
    )
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(epochs) + 1):
        train_loss, train_audit = train_one_epoch(
            model,
            train_loader,
            optimizer,
            weights,
            weights_numpy,
            store,
            device,
            values,
            scaler,
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_weighted_loss": float(train_loss),
                "train_unit_audit": train_audit,
            }
        )
        print(
            f"IDEA058 {pipeline} outer epoch={epoch}/{epochs} train={train_loss:.4f}",
            flush=True,
        )
    test_loader = build_loader(
        store,
        test_indices,
        pipeline,
        values["evaluation_batch_size"],
        False,
        seed,
    )
    call_loss, calls = ast_base.predict_calls(model, test_loader, store, device)
    animals = evaluation.calls_to_animals(calls)
    audit = {
        "epochs": int(epochs),
        "history": history,
        "test_call_cross_entropy": float(call_loss),
        "test_animal_cross_entropy": evaluation.animal_cross_entropy(animals),
        "test_animal_metrics": evaluation.animal_metrics(animals),
        "target_weight_audit": weighting.target_weight_audit(
            weights_numpy, store, train_indices
        ),
        "train_and_predict_seconds": float(time.perf_counter() - started),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "parameters": parameters,
        "checkpoint_selection": "inner_minimum_unweighted_animal_cross_entropy",
    }
    del model, optimizer, weights
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return animals, calls, audit


def summarize_selection_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row["best_validation_animal_metrics"] for row in rows]
    macro_f1 = np.asarray([row["macro_f1"] for row in metrics], dtype=float)
    animal_ce = np.asarray(
        [row["best_validation_animal_cross_entropy"] for row in rows], dtype=float
    )
    return {
        "fits": int(len(rows)),
        "mean_animal_macro_f1": float(macro_f1.mean()),
        "sample_sd_animal_macro_f1": float(macro_f1.std(ddof=1)),
        "range_animal_macro_f1": [float(macro_f1.min()), float(macro_f1.max())],
        "mean_animal_cross_entropy": float(animal_ce.mean()),
        "mean_balanced_accuracy": float(
            np.mean([row["balanced_accuracy"] for row in metrics])
        ),
        "mean_qwk": float(
            np.mean([row["quadratic_weighted_kappa"] for row in metrics])
        ),
        "mean_plain_accuracy": float(
            np.mean([row["plain_accuracy"] for row in metrics])
        ),
        "mean_selected_epoch": float(
            np.mean([row["best_epoch"] for row in rows])
        ),
        "mean_train_seconds": float(
            np.mean([row["train_seconds"] for row in rows])
        ),
        "maximum_peak_vram_bytes": int(
            max(row["peak_vram_bytes"] for row in rows)
        ),
        "trainable_parameters": int(rows[0]["parameters"]["trainable"]),
        "trainable_encoder_parameters": int(
            rows[0]["parameters"]["trainable_encoder"]
        ),
        "block_indices_zero_based": rows[0]["parameters"][
            "block_indices_zero_based"
        ],
    }


def run_selection(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    output_path = selection_path(run_root)
    if output_path.is_file() and not resume:
        raise FileExistsError(output_path)
    settings = protocol["inner_selection"]
    rows_by_recipe: dict[str, list[dict[str, Any]]] = {
        recipe_id: [] for recipe_id in settings["candidate_recipes"]
    }
    for recipe_id in settings["candidate_recipes"]:
        protocol["_active_recipe_id"] = recipe_id
        for repeat in settings["repeats"]:
            for outer_fold in settings["outer_folds"]:
                fit_path = (
                    run_root
                    / "selection"
                    / "fits"
                    / recipe_id
                    / f"repeat_{repeat}"
                    / f"fold_{outer_fold}.json"
                )
                if fit_path.is_file():
                    if not resume:
                        raise FileExistsError(fit_path)
                    fit = read_json(fit_path)
                    rows_by_recipe[recipe_id].append(fit)
                    continue
                indices = split_utils.fold_indices(
                    store, roles, int(repeat), int(outer_fold), include_test=False
                )
                seed = split_utils.full_seed(
                    int(settings["base_seed"]), int(repeat), int(outer_fold)
                )
                print(
                    f"IDEA058 SELECT recipe={recipe_id} repeat={repeat} "
                    f"fold={outer_fold} seed={seed}",
                    flush=True,
                )
                _, audit, state, _ = fit_inner(
                    PIPELINES[1],
                    protocol,
                    store,
                    indices["train"],
                    indices["validation"],
                    device,
                    seed,
                )
                del state
                fit = {
                    "status": "complete",
                    "stage": "idea058_inner_recipe_selection",
                    "outer_test_accessed": False,
                    "recipe_id": recipe_id,
                    "recipe": settings["candidate_recipes"][recipe_id],
                    "repeat": int(repeat),
                    "outer_fold": int(outer_fold),
                    "full_seed": int(seed),
                    **audit,
                }
                write_json(fit_path, fit)
                rows_by_recipe[recipe_id].append(fit)
    summaries = {
        recipe_id: summarize_selection_rows(rows)
        for recipe_id, rows in rows_by_recipe.items()
    }
    ranked = sorted(
        summaries,
        key=lambda recipe_id: (
            -round(summaries[recipe_id]["mean_animal_macro_f1"], 6),
            summaries[recipe_id]["mean_animal_cross_entropy"],
            summaries[recipe_id]["trainable_parameters"],
            recipe_id,
        ),
    )
    selected = ranked[0]
    record = {
        "status": "complete",
        "stage": "idea058_inner_recipe_selection",
        "outer_test_accessed": False,
        "protocol_id": protocol["protocol_id"],
        "candidate_fits_completed": int(sum(len(rows) for rows in rows_by_recipe.values())),
        "candidate_summaries": summaries,
        "ranking": ranked,
        "selection_rule": settings["selection_rule"],
        "selected_recipe_id": selected,
        "selected_recipe": settings["candidate_recipes"][selected],
    }
    write_json(output_path, record)
    print(json.dumps(record, indent=2), flush=True)


def initialization_audit(
    protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    models: list[nn.Module] = []
    rng_states = []
    for pipeline in PIPELINES:
        split_utils.set_seed(seed)
        model = build_model(pipeline, protocol, store, train_indices).eval()
        models.append(model)
        rng_states.append(torch.get_rng_state().clone())
    head_states = [model.head.state_dict() for model in models]
    head_equal = all(
        torch.equal(head_states[0][key], head_states[index][key])
        for key in head_states[0]
        for index in (1, 2)
    )
    rng_equal = torch.equal(rng_states[0], rng_states[1]) and torch.equal(
        rng_states[0], rng_states[2]
    )
    m1_state = models[1].ast.state_dict()
    c1_state = models[2].ast.state_dict()
    ast_equal = all(torch.equal(m1_state[key], c1_state[key]) for key in m1_state)
    parameters = {pipeline: parameter_audit(model) for pipeline, model in zip(PIPELINES, models)}
    parameter_match = (
        parameters[PIPELINES[1]]["trainable"]
        == parameters[PIPELINES[2]]["trainable"]
        and parameters[PIPELINES[1]]["trainable_encoder"]
        == parameters[PIPELINES[2]]["trainable_encoder"]
    )
    loader = build_loader(
        store, validation_indices[:8], PIPELINES[1], 8, False, seed
    )
    cpu_batch = next(iter(loader))
    logits = []
    for model in models[1:]:
        model = model.to(device)
        batch = ast_base.move_batch(cpu_batch, device)
        with torch.inference_mode():
            logits.append(
                model(
                    batch["instances"],
                    batch["instance_to_call"],
                    len(batch["labels"]),
                ).float().cpu()
            )
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    maximum_logit_difference = float((logits[0] - logits[1]).abs().max())
    result = {
        "common_head_initial_state_equal": bool(head_equal),
        "post_build_training_rng_state_equal": bool(rng_equal),
        "M1_C1_pretrained_ast_state_equal": bool(ast_equal),
        "M1_C1_trainable_parameter_count_equal": bool(parameter_match),
        "M1_C1_initial_max_logit_difference": maximum_logit_difference,
        "parameters": parameters,
    }
    del models
    if not all((head_equal, rng_equal, ast_equal, parameter_match)):
        raise RuntimeError("IDEA-058 matched-initialization audit failed")
    if maximum_logit_difference != 0.0:
        raise RuntimeError("IDEA-058 M1/C1 initial logits differ")
    return result


def reload_probability_difference(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    checkpoint_path: Path,
    before: pd.DataFrame,
    device: torch.device,
    seed: int,
) -> float:
    split_utils.set_seed(seed)
    model = build_model(pipeline, protocol, store, train_indices).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_portable_state_dict(model, checkpoint["state_dict"])
    loader = build_loader(
        store,
        validation_indices,
        pipeline,
        training_values(protocol)["evaluation_batch_size"],
        False,
        seed,
    )
    _, after = ast_base.predict_calls(model, loader, store, device)
    left = before.sort_values("call_index")[list(PROBABILITY_COLUMNS)].to_numpy()
    right = after.sort_values("call_index")[list(PROBABILITY_COLUMNS)].to_numpy()
    difference = float(np.abs(left - right).max())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return difference


def environment_lock(protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    dependencies = dependency_paths(protocol)
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
        "fbank_sha256": sha256(dependencies["fbank"]),
        "global_embedding_sha256": sha256(dependencies["global_embedding"]),
        "ast_checkpoint": protocol["fixed_training"]["checkpoint"],
        "ast_checkpoint_revision": protocol["fixed_training"]["revision"],
    }


def run_smoke(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
    device: torch.device,
    resume: bool,
) -> None:
    output_path = run_root / "smoke" / "summary.json"
    if output_path.is_file() and not resume:
        raise FileExistsError(output_path)
    selected = read_json(selection_path(run_root))
    protocol["_active_recipe_id"] = selected["selected_recipe_id"]
    settings = protocol["smoke"]
    indices = split_utils.fold_indices(
        store,
        roles,
        int(settings["repeat"]),
        int(settings["outer_fold"]),
        include_test=False,
    )
    seed = split_utils.full_seed(
        int(settings["base_seed"]),
        int(settings["repeat"]),
        int(settings["outer_fold"]),
    )
    initial = initialization_audit(
        protocol,
        store,
        indices["train"],
        indices["validation"],
        device,
        seed,
    )
    fits = []
    for pipeline in PIPELINES:
        best_epoch, audit, state, before = fit_inner(
            pipeline,
            protocol,
            store,
            indices["train"],
            indices["validation"],
            device,
            seed,
            max_epochs_override=int(settings["epochs"]),
        )
        checkpoint_path = run_root / "smoke" / f"{pipeline}_checkpoint.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "pipeline": pipeline,
                "selected_recipe_id": selected["selected_recipe_id"],
                "best_epoch": int(best_epoch),
                "state_dict": state,
            },
            checkpoint_path,
        )
        difference = reload_probability_difference(
            pipeline,
            protocol,
            store,
            indices["train"],
            indices["validation"],
            checkpoint_path,
            before,
            device,
            seed,
        )
        if difference > 1.0e-6:
            raise RuntimeError(f"IDEA-058 checkpoint reload mismatch for {pipeline}")
        fits.append(
            {
                "pipeline": pipeline,
                "best_epoch": int(best_epoch),
                "checkpoint_path": repo_relative(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "reload_probability_max_abs_difference": difference,
                "audit": audit,
            }
        )
        del state
    environment_path = run_root / "environment_lock.json"
    write_json(environment_path, environment_lock(protocol, device))
    summary = {
        "status": "complete",
        "stage": "idea058_inner_only_smoke",
        "outer_test_accessed": False,
        "selected_recipe_id": selected["selected_recipe_id"],
        "repeat": int(settings["repeat"]),
        "outer_fold": int(settings["outer_fold"]),
        "epochs": int(settings["epochs"]),
        "initialization_audit": initial,
        "fits": fits,
        "environment_lock_path": repo_relative(environment_path),
        "environment_lock_sha256": sha256(environment_path),
    }
    write_json(output_path, summary)
    revision = git_revision()
    if revision is None:
        raise RuntimeError("IDEA-058 execution lock requires a Git revision")
    lock = {
        "schema_version": "1.0",
        "status": "locked_for_idea058_initial_evaluation",
        "code_commit": revision,
        "selected_recipe_id": selected["selected_recipe_id"],
        "selected_recipe": selected["selected_recipe"],
        "selection_record_sha256": sha256(selection_path(run_root)),
        "smoke_summary_sha256": sha256(output_path),
        "environment_lock_sha256": sha256(environment_path),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "idea_sha256": sha256(PLAN_PATH),
        "roles_sha256": sha256(ROLES_PATH),
        "initial_evaluation": protocol["initial_evaluation"],
        "git_blobs": {
            repo_relative(path): git_blob_object_id(revision, path)
            for path in (PROTOCOL_PATH, Path(__file__).resolve(), PLAN_PATH)
        },
    }
    write_json(run_root / "execution_lock.json", lock)
    print(json.dumps(summary, indent=2), flush=True)


def verify_execution_lock(
    run_root: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    lock_path = run_root / "execution_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("Run IDEA-058 smoke before outer evaluation")
    lock = read_json(lock_path)
    if lock["status"] != "locked_for_idea058_initial_evaluation":
        raise RuntimeError("IDEA-058 execution lock status is invalid")
    checks = {
        PROTOCOL_PATH: lock["protocol_sha256"],
        Path(__file__).resolve(): lock["runner_sha256"],
        PLAN_PATH: lock["idea_sha256"],
        ROLES_PATH: lock["roles_sha256"],
        selection_path(run_root): lock["selection_record_sha256"],
        run_root / "smoke" / "summary.json": lock["smoke_summary_sha256"],
        run_root / "environment_lock.json": lock["environment_lock_sha256"],
    }
    for path, expected in checks.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"IDEA-058 execution-lock file changed: {path}")
    for path in (PROTOCOL_PATH, Path(__file__).resolve(), PLAN_PATH):
        if git_blob_object_id(lock["code_commit"], path) != worktree_blob_object_id(path):
            raise RuntimeError(f"IDEA-058 locked commit blob differs: {path}")
    if lock["initial_evaluation"] != protocol["initial_evaluation"]:
        raise RuntimeError("IDEA-058 evaluation matrix differs from lock")
    selected = read_json(selection_path(run_root))
    if lock["selected_recipe_id"] != selected["selected_recipe_id"]:
        raise RuntimeError("IDEA-058 selected recipe differs from lock")
    return lock


def assert_batch_orders(fits: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = []
    for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
        left = fits[left_index]
        right = fits[right_index]
        comparison = {"left": left["pipeline"], "right": right["pipeline"]}
        for phase in ("inner", "outer"):
            left_history = left[phase]["history"]
            right_history = right[phase]["history"]
            common = min(len(left_history), len(right_history))
            for epoch in range(common):
                left_hash = left_history[epoch]["train_unit_audit"]["batch_order_sha256"]
                right_hash = right_history[epoch]["train_unit_audit"]["batch_order_sha256"]
                if left_hash != right_hash:
                    raise RuntimeError(
                        f"IDEA-058 batch order differs for {phase} epoch {epoch + 1}"
                    )
            comparison[f"{phase}_common_epochs"] = int(common)
        comparisons.append(comparison)
    return {"all_common_epoch_hashes_match": True, "comparisons": comparisons}


def paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([row["macro_f1_difference"] for row in rows], dtype=float)
    return {
        "mean_macro_f1_difference": float(values.mean()),
        "sample_sd_macro_f1_difference": float(values.std(ddof=1)),
        "macro_f1_differences_by_repeat": values.tolist(),
        "positive_repeats": int(np.sum(values > 0)),
        "tied_repeats": int(np.sum(np.isclose(values, 0.0, atol=1.0e-12))),
        "negative_repeats": int(np.sum(values < 0)),
        "mean_balanced_accuracy_difference": float(
            np.mean([row["balanced_accuracy_difference"] for row in rows])
        ),
        "mean_qwk_difference": float(
            np.mean([row["qwk_difference"] for row in rows])
        ),
        "mean_plain_accuracy_difference": float(
            np.mean([row["plain_accuracy_difference"] for row in rows])
        ),
        "mean_changed_animals": float(np.mean([row["changed_animals"] for row in rows])),
        "mean_gained_correct_animals": float(
            np.mean([row["gained_correct_animals"] for row in rows])
        ),
        "mean_lost_correct_animals": float(
            np.mean([row["lost_correct_animals"] for row in rows])
        ),
    }


def aggregate_evaluation(
    evaluation_root: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    settings = protocol["initial_evaluation"]
    base_seed = int(settings["base_seeds"][0])
    metrics_by_pipeline: dict[str, list[dict[str, Any]]] = {
        pipeline: [] for pipeline in PIPELINES
    }
    animals_by_key: dict[tuple[str, int], pd.DataFrame] = {}
    selected_epochs: dict[str, list[int]] = {pipeline: [] for pipeline in PIPELINES}
    fit_audits: dict[str, list[dict[str, Any]]] = {pipeline: [] for pipeline in PIPELINES}
    for repeat in settings["repeats"]:
        for pipeline in PIPELINES:
            call_frames = []
            for outer_fold in settings["outer_folds"]:
                fit_root = (
                    evaluation_root
                    / "fits"
                    / pipeline
                    / f"base_seed_{base_seed}"
                    / f"repeat_{repeat}"
                    / f"fold_{outer_fold}"
                )
                call_frames.append(
                    pd.read_csv(
                        fit_root / "outer_test_call_predictions.csv",
                        dtype={"cat_id": str},
                    )
                )
                fit = read_json(fit_root / "fit_summary.json")
                selected_epochs[pipeline].append(int(fit["selected_epoch"]))
                fit_audits[pipeline].append(fit)
            calls = (
                pd.concat(call_frames, ignore_index=True)
                .sort_values("call_index")
                .reset_index(drop=True)
            )
            if len(calls) != 792 or calls["call_index"].nunique() != 792:
                raise RuntimeError("Complete IDEA-058 OOF must contain 792 calls")
            animals = evaluation.calls_to_animals(calls)
            if len(animals) != 111 or animals["cat_id"].nunique() != 111:
                raise RuntimeError("Complete IDEA-058 OOF must contain 111 cats")
            metrics = evaluation.animal_metrics(animals)
            metrics_by_pipeline[pipeline].append(
                {
                    "base_seed": base_seed,
                    "repeat": int(repeat),
                    "animal_cross_entropy": evaluation.animal_cross_entropy(animals),
                    **metrics,
                }
            )
            animals_by_key[(pipeline, int(repeat))] = animals
            output_root = evaluation_root / "oof" / pipeline
            output_root.mkdir(parents=True, exist_ok=True)
            calls.to_csv(output_root / f"repeat_{repeat}_calls.csv", index=False)
            animals.to_csv(output_root / f"repeat_{repeat}_animals.csv", index=False)
    contrasts = (
        (PIPELINES[0], PIPELINES[1]),
        (PIPELINES[0], PIPELINES[2]),
        (PIPELINES[1], PIPELINES[2]),
    )
    paired: dict[str, list[dict[str, Any]]] = {}
    bootstrap = {}
    change_frames = []
    for left_pipeline, right_pipeline in contrasts:
        name = f"{right_pipeline}_minus_{left_pipeline}"
        rows = []
        for repeat in settings["repeats"]:
            left_metrics = metrics_by_pipeline[left_pipeline][int(repeat)]
            right_metrics = metrics_by_pipeline[right_pipeline][int(repeat)]
            left = animals_by_key[(left_pipeline, int(repeat))]
            right = animals_by_key[(right_pipeline, int(repeat))]
            merged = left.merge(
                right,
                on=["cat_id", "true_label", "call_count"],
                suffixes=("_left", "_right"),
            )
            rows.append(
                {
                    "repeat": int(repeat),
                    "left_macro_f1": left_metrics["macro_f1"],
                    "right_macro_f1": right_metrics["macro_f1"],
                    "macro_f1_difference": right_metrics["macro_f1"]
                    - left_metrics["macro_f1"],
                    "balanced_accuracy_difference": right_metrics["balanced_accuracy"]
                    - left_metrics["balanced_accuracy"],
                    "qwk_difference": right_metrics["quadratic_weighted_kappa"]
                    - left_metrics["quadratic_weighted_kappa"],
                    "plain_accuracy_difference": right_metrics["plain_accuracy"]
                    - left_metrics["plain_accuracy"],
                    "animal_cross_entropy_difference": right_metrics["animal_cross_entropy"]
                    - left_metrics["animal_cross_entropy"],
                    "changed_animals": int(
                        (merged["predicted_label_left"] != merged["predicted_label_right"]).sum()
                    ),
                    "gained_correct_animals": int(
                        (
                            (merged["predicted_label_right"] == merged["true_label"])
                            & (merged["predicted_label_left"] != merged["true_label"])
                        ).sum()
                    ),
                    "lost_correct_animals": int(
                        (
                            (merged["predicted_label_right"] != merged["true_label"])
                            & (merged["predicted_label_left"] == merged["true_label"])
                        ).sum()
                    ),
                }
            )
            changed = merged[
                merged["predicted_label_left"] != merged["predicted_label_right"]
            ].copy()
            changed.insert(0, "contrast", name)
            changed.insert(1, "repeat", int(repeat))
            change_frames.append(changed)
        paired[name] = rows
        bootstrap[name] = evaluation.paired_cat_bootstrap(
            animals_by_key, left_pipeline, right_pipeline, protocol
        )
    pd.concat(change_frames, ignore_index=True).to_csv(
        evaluation_root / "paired_prediction_changes.csv", index=False
    )
    aggregate = {
        pipeline: evaluation.aggregate_metrics(rows)
        for pipeline, rows in metrics_by_pipeline.items()
    }
    ce = {
        pipeline: {
            "by_repeat": [float(row["animal_cross_entropy"]) for row in rows],
            "mean": float(np.mean([row["animal_cross_entropy"] for row in rows])),
            "sample_sd": float(
                np.std([row["animal_cross_entropy"] for row in rows], ddof=1)
            ),
        }
        for pipeline, rows in metrics_by_pipeline.items()
    }
    training = {}
    for pipeline, fits in fit_audits.items():
        parameters = fits[0]["outer"]["parameters"]
        training[pipeline] = {
            "selected_epochs": selected_epochs[pipeline],
            "mean_selected_epoch": float(np.mean(selected_epochs[pipeline])),
            "selected_epoch_range": [
                int(min(selected_epochs[pipeline])),
                int(max(selected_epochs[pipeline])),
            ],
            "mean_inner_train_seconds": float(
                np.mean([fit["inner"]["train_seconds"] for fit in fits])
            ),
            "mean_outer_train_and_predict_seconds": float(
                np.mean([fit["outer"]["train_and_predict_seconds"] for fit in fits])
            ),
            "maximum_peak_vram_bytes": int(
                max(
                    max(fit["inner"]["peak_vram_bytes"], fit["outer"]["peak_vram_bytes"])
                    for fit in fits
                )
            ),
            "parameters": parameters,
        }
    summary = {
        "status": "complete",
        "stage": "idea058_initial_evaluation",
        "outer_test_accessed": True,
        "protocol_id": protocol["protocol_id"],
        "primary_unit": "animal",
        "primary_metric": "macro_f1",
        "metrics_by_repeat": metrics_by_pipeline,
        "aggregate": aggregate,
        "animal_cross_entropy": ce,
        "paired": paired,
        "paired_summary": {name: paired_summary(rows) for name, rows in paired.items()},
        "paired_cat_bootstrap": bootstrap,
        "training_and_parameters": training,
        "complete_oof": {
            "calls_per_pipeline_repeat": 792,
            "animals_per_pipeline_repeat": 111,
            "evaluations": int(len(PIPELINES) * len(settings["repeats"])),
        },
    }
    r0, m1, c1 = PIPELINES
    m1_r0 = summary["paired_summary"][f"{m1}_minus_{r0}"]
    c1_m1 = summary["paired_summary"][f"{c1}_minus_{m1}"]
    m1_c1_mean = -float(c1_m1["mean_macro_f1_difference"])
    m1_c1_positive = sum(
        row["macro_f1_difference"] < 0 for row in paired[f"{c1}_minus_{m1}"]
    )
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
            m1_r0["positive_repeats"] >= int(gate["minimum_positive_repeats_over_R0"])
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
            ce[m1]["mean"] - ce[r0]["mean"]
            <= float(gate["maximum_tolerated_mean_animal_ce_increase"])
        ),
        "class_recall_cost": bool(
            max(class_recall_drops.values())
            <= float(gate["maximum_tolerated_class_recall_drop"])
        ),
    }
    summary["idea058_seed_expansion_gate"] = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "M1_minus_R0_mean_macro_f1": m1_r0["mean_macro_f1_difference"],
        "M1_minus_R0_positive_repeats": m1_r0["positive_repeats"],
        "M1_minus_C1_mean_macro_f1": m1_c1_mean,
        "M1_minus_C1_positive_repeats": int(m1_c1_positive),
        "M1_minus_R0_mean_animal_cross_entropy": ce[m1]["mean"] - ce[r0]["mean"],
        "class_recall_drop_from_R0": class_recall_drops,
        "action": gate["action_when_met"] if all(checks.values()) else gate["action_when_not_met"],
    }
    return summary


def run_evaluation(
    run_root: Path,
    protocol: dict[str, Any],
    roles: pd.DataFrame,
    store: Any,
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
            "stage": "idea058_initial_evaluation",
            "outer_test_accessed": True,
            "code_commit": lock["code_commit"],
            "selected_recipe_id": lock["selected_recipe_id"],
            "execution_lock_sha256": sha256(run_root / "execution_lock.json"),
            "device": str(device),
        },
    )
    completed = []
    order_audits = []
    settings = protocol["initial_evaluation"]
    for base_seed in settings["base_seeds"]:
        for repeat in settings["repeats"]:
            for outer_fold in settings["outer_folds"]:
                indices = split_utils.fold_indices(
                    store, roles, int(repeat), int(outer_fold), include_test=True
                )
                # Preserve the locked reference order: current outer-training
                # calls first, followed by current validation calls.  The
                # deterministic sampler permutes dataset positions, so sorting
                # these indices would change the seed-to-call mapping.
                outer_train = np.concatenate(
                    (indices["train"], indices["validation"])
                )
                seed = split_utils.full_seed(
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
                        f"IDEA058 EVAL {pipeline} base_seed={base_seed} "
                        f"repeat={repeat} fold={outer_fold} seed={seed}",
                        flush=True,
                    )
                    best_epoch, inner, state, _ = fit_inner(
                        pipeline,
                        protocol,
                        store,
                        indices["train"],
                        indices["validation"],
                        device,
                        seed,
                    )
                    del state
                    animals, calls, outer = fit_outer_and_predict(
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
                        "stage": "idea058_initial_evaluation",
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
                        **assert_batch_orders(fit_group),
                    }
                )
    summary = aggregate_evaluation(evaluation_root, protocol)
    inventory = evaluation.raw_prediction_inventory(evaluation_root)
    inventory_path = evaluation_root / "raw_prediction_inventory.json"
    write_json(inventory_path, inventory)
    summary.update(
        {
            "selected_recipe_id": lock["selected_recipe_id"],
            "selection_record_sha256": lock["selection_record_sha256"],
            "code_commit": lock["code_commit"],
            "execution_lock_sha256": sha256(run_root / "execution_lock.json"),
            "environment_lock_sha256": sha256(run_root / "environment_lock.json"),
            "runner_sha256": lock["runner_sha256"],
            "protocol_sha256": lock["protocol_sha256"],
            "batch_order_audit": {
                "fold_groups": len(order_audits),
                "all_common_epoch_hashes_match": True,
                "details": order_audits,
            },
            "raw_prediction_inventory": {
                "path": repo_relative(inventory_path),
                "inventory_sha256": sha256(inventory_path),
                "files": inventory["files"],
                "bytes": inventory["bytes"],
                "aggregate_sha256": inventory["aggregate_sha256"],
            },
        }
    )
    write_json(summary_path, summary)
    write_json(
        evaluation_root / "run_summary.json",
        {
            "status": "complete",
            "completed_fits": len(completed),
            "expected_fits": int(settings["total_outer_fits"]),
            "summary_path": repo_relative(summary_path),
            "summary_sha256": sha256(summary_path),
            "raw_prediction_inventory_sha256": sha256(inventory_path),
            "raw_prediction_aggregate_sha256": inventory["aggregate_sha256"],
        },
    )
    write_json(REPO_ROOT / protocol["outputs"]["result_metadata"], summary)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    device = resolve_device(args.device)
    run_root = RUNS_ROOT / args.output_subdir
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    store = ast_base.load_feature_store()
    if len(store.call_ids) != int(protocol["dataset"]["calls"]):
        raise RuntimeError("IDEA-058 call count differs from protocol")
    if len(np.unique(store.cat_ids)) != int(protocol["dataset"]["cats"]):
        raise RuntimeError("IDEA-058 cat count differs from protocol")
    if args.stage in ("select", "all"):
        run_selection(run_root, protocol, roles, store, device, args.resume)
    if args.stage in ("smoke", "all"):
        run_smoke(run_root, protocol, roles, store, device, args.resume)
    if args.stage in ("evaluate", "all"):
        run_evaluation(run_root, protocol, roles, store, device, args.resume)


if __name__ == "__main__":
    main()
