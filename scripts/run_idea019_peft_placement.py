"""Run IDEA-019 matched-budget AST adapter placement diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import ASTModel


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for local_root in (SCRIPTS_ROOT, SRC_ROOT):
    if str(local_root) not in sys.path:
        sys.path.insert(0, str(local_root))

from animal_fyp.evaluation import animal_level_metrics, categorical_metrics  # noqa: E402
from run_ast_finetuning import (  # noqa: E402
    CONFIG_PATH,
    FEATURE_PATH,
    FROZEN_EMBEDDING_PATH,
    HF_CACHE,
    ClassificationHead,
    FeatureStore,
    FrozenClassifier,
    adapt_standard_geometry,
    build_loader,
    call_indices_for_roles,
    class_weights,
    evaluate_frame,
    load_feature_store,
    move_batch,
    predict_calls,
    train_one_epoch,
    trainable_counts,
)


ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_nested_roles_v1.csv"
LAYER_EMBEDDING_PATH = (
    REPO_ROOT
    / "runs"
    / "ast_layer_embeddings_v2_float32"
    / "ast_standard_layer_call_embeddings.npz"
)
PRIOR_FINETUNING_SUMMARY = REPO_ROOT / "runs" / "ast_finetuning_v1" / "summary.json"
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
FIXED_PLACEMENTS = {
    "adapter_early": (0, 1),
    "adapter_middle": (5, 6),
    "adapter_late": (10, 11),
    "adapter_distributed": (2, 9),
    # np.random.default_rng(20260826).choice(12, size=2, replace=False)
    "adapter_random": (1, 9),
}
MODES = (
    "head_only",
    "adapter_early",
    "adapter_middle",
    "adapter_late",
    "adapter_distributed",
    "adapter_random",
    "adapter_probe_guided",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--folds", default="0,1,2,3")
    parser.add_argument("--output-subdir", default="idea019_peft_placement_v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--adapter-width", type=int, default=32)
    parser.add_argument("--adapter-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--head-learning-rate", type=float, default=3.109800273709165e-3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--probe-cv-folds", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class LayerStore:
    embeddings: np.ndarray
    call_ids: np.ndarray
    cat_ids: np.ndarray
    labels: np.ndarray


def load_layer_store(store: FeatureStore) -> LayerStore:
    loaded = np.load(LAYER_EMBEDDING_PATH)
    call_ids = loaded["call_ids"].astype(str)
    cat_ids = loaded["cat_ids"].astype(str)
    labels = loaded["labels"].astype(np.int64)
    embeddings = loaded["embeddings"].astype(np.float32)
    if embeddings.shape != (792, 12, 768):
        raise RuntimeError(f"Unexpected layer embedding shape: {embeddings.shape}")
    if not (
        np.array_equal(call_ids, store.call_ids)
        and np.array_equal(cat_ids, store.cat_ids)
        and np.array_equal(labels, store.labels)
    ):
        raise RuntimeError("Layer embedding metadata does not match the fbank store")
    return LayerStore(embeddings, call_ids, cat_ids, labels)


class ResidualBottleneckAdapter(nn.Module):
    def __init__(self, hidden_size: int, width: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, width)
        self.activation = nn.GELU()
        self.up = nn.Linear(width, hidden_size)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.up(self.activation(self.down(hidden_states)))


class ASTLayerWithAdapter(nn.Module):
    def __init__(self, base_layer: nn.Module, hidden_size: int, width: int) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.adapter = ResidualBottleneckAdapter(hidden_size, width)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self.base_layer(hidden_states, head_mask, output_attentions)
        return (self.adapter(outputs[0]),) + outputs[1:]


class AdapterASTClassifier(nn.Module):
    def __init__(
        self,
        protocol: dict[str, object],
        placement: tuple[int, int],
        adapter_width: int,
        head: ClassificationHead,
    ) -> None:
        super().__init__()
        ast_config = protocol["ast"]
        self.ast = ASTModel.from_pretrained(
            ast_config["checkpoint"],
            revision=ast_config["revision"],
            cache_dir=HF_CACHE,
            use_safetensors=True,
        )
        self.geometry_audit = adapt_standard_geometry(self.ast, protocol)
        for parameter in self.ast.parameters():
            parameter.requires_grad = False
        hidden_size = int(self.ast.config.hidden_size)
        if len(placement) != 2 or len(set(placement)) != 2:
            raise ValueError(f"Placement must contain two distinct layers: {placement}")
        for layer_index in placement:
            if layer_index not in range(len(self.ast.encoder.layer)):
                raise ValueError(f"AST layer index out of range: {layer_index}")
            self.ast.encoder.layer[layer_index] = ASTLayerWithAdapter(
                self.ast.encoder.layer[layer_index], hidden_size, adapter_width
            )
        self.placement = tuple(sorted(placement))
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
        return self.head(call_embeddings / counts[:, None])


def build_model(
    mode: str,
    placement: tuple[int, int] | None,
    protocol: dict[str, object],
    train_indices: np.ndarray,
    store: FeatureStore,
    adapter_width: int,
) -> nn.Module:
    frozen_embeddings = store.frozen_embeddings[train_indices]
    head = ClassificationHead(
        mean=frozen_embeddings.mean(axis=0),
        scale=frozen_embeddings.std(axis=0),
        dropout=float(protocol["classifier_head"]["dropout"]),
    )
    if mode == "head_only":
        return FrozenClassifier(head)
    if placement is None:
        raise ValueError(f"Adapter mode {mode} requires a placement")
    return AdapterASTClassifier(protocol, placement, adapter_width, head)


def make_optimizer(
    model: nn.Module,
    mode: str,
    adapter_learning_rate: float,
    head_learning_rate: float,
) -> torch.optim.Optimizer:
    if mode == "head_only":
        return torch.optim.Adamax(model.parameters(), lr=head_learning_rate, eps=1.0e-7)
    adapter_parameters = [
        parameter for parameter in model.ast.parameters() if parameter.requires_grad
    ]
    if not adapter_parameters:
        raise RuntimeError("Adapter mode has no trainable AST parameters")
    return torch.optim.Adamax(
        [
            {"params": adapter_parameters, "lr": adapter_learning_rate},
            {"params": model.head.parameters(), "lr": head_learning_rate},
        ],
        eps=1.0e-7,
    )


def probe_layer_utilities(
    layer_store: LayerStore,
    call_indices: np.ndarray,
    cv_folds: int,
    seed: int,
) -> dict[str, object]:
    selected_cat_ids = np.unique(layer_store.cat_ids[call_indices])
    label_by_cat = {
        cat_id: int(layer_store.labels[np.flatnonzero(layer_store.cat_ids == cat_id)[0]])
        for cat_id in selected_cat_ids
    }
    cat_labels = np.asarray([label_by_cat[cat_id] for cat_id in selected_cat_ids])
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    layer_fold_scores: list[list[float]] = [[] for _ in range(12)]
    for cv_fold, (train_cat_rows, validation_cat_rows) in enumerate(
        splitter.split(selected_cat_ids, cat_labels)
    ):
        train_cats = selected_cat_ids[train_cat_rows]
        validation_cats = selected_cat_ids[validation_cat_rows]
        train_calls = call_indices[np.isin(layer_store.cat_ids[call_indices], train_cats)]
        validation_calls = call_indices[
            np.isin(layer_store.cat_ids[call_indices], validation_cats)
        ]
        for layer_index in range(12):
            scaler = StandardScaler().fit(layer_store.embeddings[train_calls, layer_index])
            classifier = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                multi_class="auto",
                random_state=seed + cv_fold,
            )
            classifier.fit(
                scaler.transform(layer_store.embeddings[train_calls, layer_index]),
                layer_store.labels[train_calls],
            )
            probabilities = classifier.predict_proba(
                scaler.transform(layer_store.embeddings[validation_calls, layer_index])
            )
            metrics, _, _, _ = animal_level_metrics(
                probabilities,
                layer_store.cat_ids[validation_calls],
                layer_store.labels[validation_calls],
            )
            layer_fold_scores[layer_index].append(float(metrics["macro_f1"]))
    layer_mean_scores = [float(np.mean(scores)) for scores in layer_fold_scores]
    ranking = sorted(range(12), key=lambda layer: (-layer_mean_scores[layer], layer))
    return {
        "probe_train_calls": int(len(call_indices)),
        "probe_train_cats": int(len(selected_cat_ids)),
        "cv_folds": cv_folds,
        "seed": seed,
        "layer_fold_macro_f1": layer_fold_scores,
        "layer_mean_macro_f1": layer_mean_scores,
        "ranking_zero_based": ranking,
        "ranking_one_based": [layer + 1 for layer in ranking],
        "selected_layers_zero_based": sorted(ranking[:2]),
        "selected_layers_one_based": sorted(layer + 1 for layer in ranking[:2]),
    }


def placement_for_mode(
    mode: str, probe_result: dict[str, object]
) -> tuple[int, int] | None:
    if mode == "head_only":
        return None
    if mode == "adapter_probe_guided":
        selected = probe_result["selected_layers_zero_based"]
        return int(selected[0]), int(selected[1])
    return FIXED_PLACEMENTS[mode]


def placement_probe_score(
    placement: tuple[int, int] | None, probe_result: dict[str, object]
) -> float | None:
    if placement is None:
        return None
    layer_scores = probe_result["layer_mean_macro_f1"]
    return float(np.mean([layer_scores[layer] for layer in placement]))


def fit_inner(
    mode: str,
    placement: tuple[int, int] | None,
    protocol: dict[str, object],
    store: FeatureStore,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[int, dict[str, object]]:
    set_seed(seed)
    model = build_model(
        mode, placement, protocol, train_indices, store, args.adapter_width
    ).to(device)
    counts = trainable_counts(model)
    adapter_parameters = (
        sum(p.numel() for p in model.ast.parameters() if p.requires_grad)
        if mode != "head_only"
        else 0
    )
    optimizer = make_optimizer(
        model, mode, args.adapter_learning_rate, args.head_learning_rate
    )
    weights = class_weights(store.labels[train_indices]).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loader_mode = "frozen" if mode == "head_only" else "adapter"
    train_loader = build_loader(
        store, train_indices, loader_mode, args.batch_size, True, seed
    )
    validation_loader = build_loader(
        store, validation_indices, loader_mode, args.batch_size, False, seed
    )
    best_loss = float("inf")
    best_epoch = 1
    best_metrics: dict[str, object] = {}
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            weights,
            device,
            args.accumulation_steps,
            args.gradient_clip,
            scaler,
        )
        validation_loss, validation_frame = predict_calls(
            model, validation_loader, store, device
        )
        validation_metrics, _ = evaluate_frame(validation_frame)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_qwk": validation_metrics["quadratic_weighted_kappa"],
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
            f"{mode} epoch {epoch}: train={train_loss:.4f}, "
            f"val={validation_loss:.4f}, val_F1={validation_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            break
    audit = {
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "best_validation_loss": best_loss,
        "best_validation_animal_metrics": best_metrics,
        "history": history,
        "train_seconds": time.perf_counter() - started,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "parameters": counts,
        "trainable_adapter_parameters": adapter_parameters,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, audit


def fit_outer_and_predict(
    mode: str,
    placement: tuple[int, int] | None,
    protocol: dict[str, object],
    store: FeatureStore,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    set_seed(seed)
    model = build_model(
        mode, placement, protocol, train_indices, store, args.adapter_width
    ).to(device)
    counts = trainable_counts(model)
    adapter_parameters = (
        sum(p.numel() for p in model.ast.parameters() if p.requires_grad)
        if mode != "head_only"
        else 0
    )
    optimizer = make_optimizer(
        model, mode, args.adapter_learning_rate, args.head_learning_rate
    )
    weights = class_weights(store.labels[train_indices]).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loader_mode = "frozen" if mode == "head_only" else "adapter"
    train_loader = build_loader(
        store, train_indices, loader_mode, args.batch_size, True, seed
    )
    test_loader = build_loader(store, test_indices, loader_mode, args.batch_size, False, seed)
    training_losses = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            weights,
            device,
            args.accumulation_steps,
            args.gradient_clip,
            scaler,
        )
        training_losses.append(train_loss)
        print(
            f"{mode} outer retrain {epoch}/{epochs}: train={train_loss:.4f}",
            flush=True,
        )
    test_loss, test_frame = predict_calls(model, test_loader, store, device)
    test_metrics, _ = evaluate_frame(test_frame)
    audit = {
        "epochs": epochs,
        "training_losses": training_losses,
        "test_loss": test_loss,
        "test_animal_metrics": test_metrics,
        "train_and_predict_seconds": time.perf_counter() - started,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "parameters": counts,
        "trainable_adapter_parameters": adapter_parameters,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_frame, audit


def bootstrap_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    metrics = categorical_metrics(labels, probabilities)
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "qwk": float(metrics["quadratic_weighted_kappa"]),
    }


def paired_stratified_bootstrap(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    repeats: int,
    seed: int = 20260826,
) -> dict[str, object]:
    merged = reference.merge(
        candidate, on=["cat_id", "true_label"], suffixes=("_ref", "_cand")
    )
    expected_cats = reference["cat_id"].nunique()
    if (
        len(merged) != expected_cats
        or merged["cat_id"].nunique() != expected_cats
        or candidate["cat_id"].nunique() != expected_cats
    ):
        raise RuntimeError("Paired bootstrap did not match the current run's cats one-to-one")
    labels_all = merged["true_label"].to_numpy()
    reference_probabilities = merged[
        [f"{column}_ref" for column in PROBABILITY_COLUMNS]
    ].to_numpy()
    candidate_probabilities = merged[
        [f"{column}_cand" for column in PROBABILITY_COLUMNS]
    ].to_numpy()
    reference_metrics = bootstrap_metrics(labels_all, reference_probabilities)
    candidate_metrics = bootstrap_metrics(labels_all, candidate_probabilities)
    observed = {
        metric: candidate_metrics[metric] - reference_metrics[metric]
        for metric in reference_metrics
    }
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels_all == label) for label in range(3)]
    differences = {metric: np.empty(repeats) for metric in observed}
    for repeat in range(repeats):
        sampled = np.concatenate(
            [rng.choice(indices, len(indices), replace=True) for indices in class_indices]
        )
        labels = labels_all[sampled]
        ref = bootstrap_metrics(labels, reference_probabilities[sampled])
        cand = bootstrap_metrics(labels, candidate_probabilities[sampled])
        for metric in differences:
            differences[metric][repeat] = cand[metric] - ref[metric]
    return {
        "repeats": repeats,
        "seed": seed,
        "observed_candidate_minus_reference": observed,
        "intervals": {
            metric: {
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "bootstrap_mean": float(values.mean()),
            }
            for metric, values in differences.items()
        },
    }


def safe_spearman(values_a: list[float], values_b: list[float]) -> dict[str, object]:
    if (
        len(values_a) < 2
        or np.ptp(np.asarray(values_a, dtype=np.float64)) == 0.0
        or np.ptp(np.asarray(values_b, dtype=np.float64)) == 0.0
    ):
        return {
            "spearman_rho": None,
            "two_sided_pvalue": None,
            "reason": "correlation undefined because at least one variable is constant",
        }
    statistic, pvalue = spearmanr(values_a, values_b)
    return {
        "spearman_rho": float(statistic),
        "two_sided_pvalue": float(pvalue),
    }


def main() -> None:
    args = parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown_modes = set(modes) - set(MODES)
    if unknown_modes:
        raise ValueError(f"Unknown modes: {sorted(unknown_modes)}")
    folds = [int(fold.strip()) for fold in args.folds.split(",") if fold.strip()]
    if not folds or any(fold not in range(4) for fold in folds):
        raise ValueError("Folds must be selected from 0,1,2,3")
    if args.batch_size < 2 or args.adapter_width < 1 or args.probe_cv_folds < 2:
        raise ValueError("Invalid batch size, adapter width, or probe CV folds")
    output_dir = REPO_ROOT / "runs" / args.output_subdir
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output_dir}")
    output_dir.mkdir(parents=True)

    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    store = load_feature_store()
    layer_store = load_layer_store(store)
    device = resolve_device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"IDEA-019 device: {device} ({device_name})", flush=True)

    probe_results: dict[str, object] = {}
    probe_rows = []
    for outer_fold in folds:
        indices = call_indices_for_roles(store, roles, outer_fold)
        probe_result = probe_layer_utilities(
            layer_store,
            indices["train"],
            args.probe_cv_folds,
            seed=19000 + outer_fold,
        )
        probe_results[str(outer_fold)] = probe_result
        selected = set(probe_result["selected_layers_zero_based"])
        for layer_index, score in enumerate(probe_result["layer_mean_macro_f1"]):
            probe_rows.append(
                {
                    "outer_fold": outer_fold,
                    "layer_zero_based": layer_index,
                    "layer_one_based": layer_index + 1,
                    "mean_cv_animal_macro_f1": score,
                    "rank": probe_result["ranking_zero_based"].index(layer_index) + 1,
                    "probe_guided_selected": layer_index in selected,
                }
            )
        print(
            f"Fold {outer_fold} probe-selected layers: "
            f"{probe_result['selected_layers_one_based']}",
            flush=True,
        )
    pd.DataFrame(probe_rows).to_csv(output_dir / "layer_probe_results.csv", index=False)

    results: dict[str, object] = {}
    animal_frames: dict[str, pd.DataFrame] = {}
    for mode in modes:
        mode_fold_results = []
        mode_test_frames = []
        for outer_fold in folds:
            indices = call_indices_for_roles(store, roles, outer_fold)
            probe_result = probe_results[str(outer_fold)]
            placement = placement_for_mode(mode, probe_result)
            seed = 42 + outer_fold
            print(
                f"=== {mode} outer fold {outer_fold}; "
                f"layers={None if placement is None else [x + 1 for x in placement]} ===",
                flush=True,
            )
            best_epoch, inner_audit = fit_inner(
                mode,
                placement,
                protocol,
                store,
                indices["train"],
                indices["validation"],
                args,
                device,
                seed,
            )
            outer_train_indices = np.concatenate(
                (indices["train"], indices["validation"])
            )
            test_frame, outer_audit = fit_outer_and_predict(
                mode,
                placement,
                protocol,
                store,
                outer_train_indices,
                indices["test"],
                best_epoch,
                args,
                device,
                seed,
            )
            test_frame.insert(0, "outer_fold", outer_fold)
            mode_test_frames.append(test_frame)
            mode_fold_results.append(
                {
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "placement_layers_zero_based": (
                        list(placement) if placement is not None else []
                    ),
                    "placement_layers_one_based": (
                        [layer + 1 for layer in placement] if placement is not None else []
                    ),
                    "placement_probe_score": placement_probe_score(
                        placement, probe_result
                    ),
                    "inner_train_calls": int(len(indices["train"])),
                    "inner_validation_calls": int(len(indices["validation"])),
                    "outer_test_calls": int(len(indices["test"])),
                    "inner": inner_audit,
                    "outer": outer_audit,
                }
            )
            print(
                f"{mode} fold {outer_fold}: best_epoch={best_epoch}, "
                f"outer_F1={outer_audit['test_animal_metrics']['macro_f1']:.4f}",
                flush=True,
            )
        all_test_calls = pd.concat(mode_test_frames, ignore_index=True).sort_values(
            "call_index"
        )
        overall_metrics, animal_frame = evaluate_frame(all_test_calls)
        all_test_calls.to_csv(output_dir / f"{mode}_call_predictions.csv", index=False)
        animal_frame.to_csv(output_dir / f"{mode}_animal_predictions.csv", index=False)
        results[mode] = {
            "folds": mode_fold_results,
            "overall_animal_metrics": overall_metrics,
        }
        animal_frames[mode] = animal_frame

    contrasts = {}
    contrast_pairs = [
        ("adapter_probe_guided", "adapter_late"),
        ("adapter_probe_guided", "adapter_random"),
    ]
    contrast_pairs.extend(
        (mode, "head_only") for mode in modes if mode.startswith("adapter_")
    )
    for candidate, reference in contrast_pairs:
        if candidate not in results or reference not in results:
            continue
        key = f"{candidate}_minus_{reference}"
        if key not in contrasts:
            contrasts[key] = paired_stratified_bootstrap(
                animal_frames[reference],
                animal_frames[candidate],
                repeats=args.bootstrap_repeats,
            )

    correlation_rows = []
    fold_correlations = {}
    adapter_modes = [mode for mode in modes if mode.startswith("adapter_")]
    if "head_only" in results:
        for outer_fold in folds:
            head_fold_result = next(
                fold_result
                for fold_result in results["head_only"]["folds"]
                if fold_result["outer_fold"] == outer_fold
            )
            head_f1 = head_fold_result["outer"]["test_animal_metrics"]["macro_f1"]
            scores = []
            gains = []
            for mode in adapter_modes:
                fold_result = next(
                    candidate_fold
                    for candidate_fold in results[mode]["folds"]
                    if candidate_fold["outer_fold"] == outer_fold
                )
                score = fold_result["placement_probe_score"]
                gain = (
                    fold_result["outer"]["test_animal_metrics"]["macro_f1"]
                    - head_f1
                )
                scores.append(score)
                gains.append(gain)
                correlation_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "mode": mode,
                        "placement_probe_score": score,
                        "outer_macro_f1_gain_vs_head_only": gain,
                    }
                )
            fold_correlations[str(outer_fold)] = {
                **safe_spearman(scores, gains),
                "placements": len(scores),
            }
    pd.DataFrame(correlation_rows).to_csv(
        output_dir / "probe_placement_correspondence.csv", index=False
    )
    if correlation_rows:
        correspondence = pd.DataFrame(correlation_rows)
        pooled_correlation = {
            **safe_spearman(
                correspondence["placement_probe_score"].tolist(),
                correspondence["outer_macro_f1_gain_vs_head_only"].tolist(),
            ),
            "placement_fold_observations": int(len(correspondence)),
        }
    else:
        pooled_correlation = None

    prior_summary = json.loads(PRIOR_FINETUNING_SUMMARY.read_text(encoding="utf-8"))
    context_controls = {
        mode: prior_summary["results"][mode]["overall_animal_metrics"]
        for mode in ("frozen", "last2", "full")
    }
    summary = {
        "experiment": "idea019-ast-adapter-placement-v1",
        "protocol_id": protocol["protocol_id"],
        "hashes": {
            "protocol_config_sha256": sha256(CONFIG_PATH),
            "roles_sha256": sha256(ROLES_PATH),
            "feature_sha256": sha256(FEATURE_PATH),
            "frozen_embedding_sha256": sha256(FROZEN_EMBEDDING_PATH),
            "layer_embedding_sha256": sha256(LAYER_EMBEDDING_PATH),
            "prior_finetuning_summary_sha256": sha256(PRIOR_FINETUNING_SUMMARY),
        },
        "execution": {
            "device": str(device),
            "device_name": device_name,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "data_scope": {
            "calls": int(len(store.call_ids)),
            "cats": int(len(np.unique(store.cat_ids))),
            "segments": int(len(store.features)),
            "encoder_layers": 12,
        },
        "training_config": {
            "modes": modes,
            "folds": folds,
            "peft_family": "block-output residual bottleneck adapter",
            "adapters_per_model": 2,
            "adapter_width": args.adapter_width,
            "fixed_placements_zero_based": {
                mode: list(layers) for mode, layers in FIXED_PLACEMENTS.items()
            },
            "fixed_placements_one_based": {
                mode: [layer + 1 for layer in layers]
                for mode, layers in FIXED_PLACEMENTS.items()
            },
            "random_placement_seed": 20260826,
            "probe_rule": "top two layers by mean animal macro F1 from 3-fold cat-level CV using only each outer fold's inner-train calls",
            "probe_tie_break": "shallower layer index",
            "batch_size": args.batch_size,
            "accumulation_steps": args.accumulation_steps,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "adapter_learning_rate": args.adapter_learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "gradient_clip": args.gradient_clip,
            "optimizer": "Adamax",
            "mixed_precision": device.type == "cuda",
            "primary_metric": "animal-level macro F1",
        },
        "probe_results": probe_results,
        "results": results,
        "contrasts": contrasts,
        "probe_placement_correspondence": {
            "by_outer_fold": fold_correlations,
            "pooled_exploratory": pooled_correlation,
        },
        "context_controls_from_ast_finetuning_v1": context_controls,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
