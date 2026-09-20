"""Run the IDEA-074 C1 x Hybrid Call+Cat 2x2 inner-only factorial."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_idea066_hybrid_call_cat_ast as idea066  # noqa: E402
import run_meowagenet_idea068_age_sensitive_ast as idea068  # noqa: E402
import run_meowagenet_idea069_age_residual_seed_replication as idea069  # noqa: E402
import run_meowagenet_idea071_bounded_dual_path_fusion as idea071  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea074_C1_hybrid_factorial_v1.json"
)
PIPELINES = (
    "A0_C_call_only",
    "A0_H_hybrid",
    "C1_C_call_only",
    "C1_H_hybrid",
)
BASE_SEEDS = (6346, 7243, 9617)
EXPECTED_PARAMETERS = {
    "A0_C_call_only": 99_075,
    "A0_H_hybrid": 99_075,
    "C1_C_call_only": 108_143,
    "C1_H_hybrid": 108_143,
}
HYBRID_CALL_WEIGHT = 0.5
HYBRID_CAT_WEIGHT = 0.5
AGE_RESIDUAL_HIDDEN = 60
CAP = 0.25
RMS_EPSILON = 1.0e-8
PROBABILITY_COLUMNS = idea068.PROBABILITY_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-subdir", default="meowagenet_idea074_C1_hybrid_factorial_v1"
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


def is_c1(pipeline: str) -> bool:
    return pipeline in PIPELINES[2:]


def is_hybrid(pipeline: str) -> bool:
    return pipeline in (PIPELINES[1], PIPELINES[3])


def build_model(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    age_features: np.ndarray,
    train_indices: np.ndarray,
) -> torch.nn.Module:
    embeddings = store.frozen_embeddings[train_indices]
    common = {
        "ast_mean": embeddings.mean(axis=0),
        "ast_scale": embeddings.std(axis=0),
        "age_train": age_features[train_indices],
        "dropout": float(protocol["fixed_training"]["dropout"]),
    }
    if pipeline in PIPELINES[:2]:
        return idea068.AgeResidualClassifier(
            pipeline="A0_ast_only",
            **common,
            age_hidden_units=AGE_RESIDUAL_HIDDEN,
        )
    if pipeline in PIPELINES[2:]:
        return idea071.BoundedWideAdditiveClassifier(**common)
    raise ValueError(pipeline)


def initialization_audit(
    protocol: dict[str, Any],
    store: Any,
    features: np.ndarray,
    train_indices: np.ndarray,
    probe_indices: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    logits: dict[str, np.ndarray] = {}
    states: dict[str, dict[str, torch.Tensor]] = {}
    parameters: dict[str, int] = {}
    for pipeline in PIPELINES:
        idea068.idea051.reference.historical.set_seed(seed)
        model = build_model(
            pipeline, protocol, store, features, train_indices
        ).eval()
        parameters[pipeline] = sum(
            parameter.numel() for parameter in model.parameters()
        )
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
        raise RuntimeError(f"IDEA-074 parameter audit mismatch: {parameters}")

    paired_state_dict_equal: dict[str, bool] = {}
    for name, left, right in (
        ("A0_call_vs_hybrid", PIPELINES[0], PIPELINES[1]),
        ("C1_call_vs_hybrid", PIPELINES[2], PIPELINES[3]),
    ):
        equal = states[left].keys() == states[right].keys() and all(
            torch.equal(states[left][key], states[right][key])
            for key in states[left]
        )
        paired_state_dict_equal[name] = bool(equal)
    if not all(paired_state_dict_equal.values()):
        raise RuntimeError("IDEA-074 loss-paired state dicts differ")

    shared_ast_keys = tuple(states[PIPELINES[0]].keys())
    shared_ast_state_equal = all(
        all(torch.equal(states[PIPELINES[0]][key], states[pipeline][key]) for key in shared_ast_keys)
        for pipeline in PIPELINES[1:]
    )
    if not shared_ast_state_equal:
        raise RuntimeError("IDEA-074 shared AST trunks differ at initialization")

    differences = {
        pipeline: float(np.max(np.abs(logits[pipeline] - logits[PIPELINES[0]])))
        for pipeline in PIPELINES[1:]
    }
    if any(value != 0.0 for value in differences.values()):
        raise RuntimeError("IDEA-074 pipelines are not identical at initialization")
    return {
        "trainable_parameters": parameters,
        "max_logit_difference_vs_A0_C": differences,
        "paired_state_dict_equal": paired_state_dict_equal,
        "shared_AST_state_equal": bool(shared_ast_state_equal),
    }


def train_one_epoch(
    pipeline: str,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    store: Any,
    age_features: np.ndarray,
    call_class_weights: torch.Tensor,
    cat_class_weights: torch.Tensor,
    device: torch.device,
    gradient_clip: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    model.train()
    total_values: list[float] = []
    call_values: list[float] = []
    cat_values: list[float] = []
    processed_cats: list[str] = []
    processed_calls: list[int] = []
    batch_sizes: list[int] = []
    for cpu_batch in loader:
        batch = idea068.idea051.move_set_batch(cpu_batch, device)
        call_indices = cpu_batch["call_indices"].numpy().astype(np.int64)
        call_labels = torch.from_numpy(store.labels[call_indices]).to(device)
        age = torch.from_numpy(age_features[call_indices]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            call_logits = model(batch["embeddings"], age)
            per_call = torch.nn.functional.cross_entropy(
                call_logits, call_labels, reduction="none"
            )
            call_weights = call_class_weights[call_labels]
            call_loss = (per_call * call_weights).sum() / call_weights.sum()
            cat_probs = idea066.cat_probabilities(
                torch.softmax(call_logits, dim=1),
                batch["instance_to_cat"],
                len(batch["labels"]),
            )
            true_probs = cat_probs[
                torch.arange(len(batch["labels"]), device=device), batch["labels"]
            ]
            per_cat = -torch.log(torch.clamp(true_probs, min=1.0e-7))
            cat_weights = cat_class_weights[batch["labels"]]
            cat_loss = (per_cat * cat_weights).sum() / cat_weights.sum()
            loss = (
                HYBRID_CALL_WEIGHT * call_loss + HYBRID_CAT_WEIGHT * cat_loss
                if is_hybrid(pipeline)
                else call_loss
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        total_values.append(float(loss.detach()))
        call_values.append(float(call_loss.detach()))
        cat_values.append(float(cat_loss.detach()))
        processed_cats.extend(str(value) for value in cpu_batch["cat_ids"])
        processed_calls.extend(int(value) for value in call_indices)
        batch_sizes.append(len(cpu_batch["cat_ids"]))
    if len(processed_cats) != len(set(processed_cats)):
        raise RuntimeError("An IDEA-074 epoch repeated a cat")
    if len(processed_calls) != len(set(processed_calls)):
        raise RuntimeError("An IDEA-074 epoch repeated a call")
    return (
        {
            "total": float(np.mean(total_values)),
            "call": float(np.mean(call_values)),
            "cat": float(np.mean(cat_values)),
        },
        {
            "cats": len(processed_cats),
            "calls": len(processed_calls),
            "batch_sizes_cats": batch_sizes,
            "cat_order_sha256": idea068.hashlib.sha256(
                "\n".join(processed_cats).encode("utf-8")
            ).hexdigest(),
            "call_coverage_sha256": idea068.hashlib.sha256(
                np.sort(np.asarray(processed_calls, dtype="<i8")).tobytes()
            ).hexdigest(),
        },
    )


def predict(
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
    return idea068.predict(
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
    idea068.idea051.reference.historical.set_seed(seed)
    fixed = protocol["fixed_training"]
    model = build_model(pipeline, protocol, store, features, train_indices).to(device)
    training_seed = seed + int(fixed["post_build_seed_offset"])
    idea068.idea051.reference.historical.set_seed(training_seed)
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=float(fixed["learning_rate"]),
        eps=float(fixed["optimizer_epsilon"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    dataset = idea068.idea051.CatSetDataset(store, train_indices)
    loader = idea068.idea051.build_set_loader(
        dataset, int(fixed["cat_batch_size"]), True, seed
    )
    call_weights = torch.from_numpy(
        idea068.class_weights(store.labels[train_indices])
    ).to(device)
    cat_weights = torch.from_numpy(idea068.class_weights(dataset.labels)).to(device)
    best_loss = float("inf")
    best_epoch = 1
    best_state = idea068.idea051.cpu_state_dict(model)
    best_animals: pd.DataFrame | None = None
    best_calls: pd.DataFrame | None = None
    history: list[dict[str, Any]] = []
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, int(fixed["maximum_epochs"]) + 1):
        losses, train_audit = train_one_epoch(
            pipeline,
            model,
            loader,
            optimizer,
            scaler,
            store,
            features,
            call_weights,
            cat_weights,
            device,
            float(fixed["gradient_clip"]),
        )
        animals, calls = predict(
            model,
            store,
            features,
            validation_indices,
            int(fixed["cat_batch_size"]) * 2,
            device,
            seed,
        )
        validation_loss = idea068.idea051.animal_cross_entropy(animals)
        metrics = idea068.idea051.animal_metrics(animals)
        history.append(
            {
                "epoch": epoch,
                "train_loss": losses,
                "train_audit": train_audit,
                "validation_animal_cross_entropy": validation_loss,
                "validation_animal_brier": idea068.brier(animals),
                "validation_animal_metrics": metrics,
            }
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = idea068.idea051.cpu_state_dict(model)
            best_animals = animals.copy()
            best_calls = calls.copy()
            stale = 0
        else:
            stale += 1
        print(
            f"{pipeline} epoch={epoch} total={losses['total']:.4f} "
            f"call={losses['call']:.4f} cat={losses['cat']:.4f} "
            f"val_CE={validation_loss:.4f} val_F1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if stale >= int(fixed["early_stopping_patience"]):
            break
    if best_animals is None or best_calls is None:
        raise RuntimeError("No IDEA-074 checkpoint was selected")
    model.load_state_dict(best_state)
    reload_animals, _ = predict(
        model,
        store,
        features,
        validation_indices,
        int(fixed["cat_batch_size"]) * 2,
        device,
        seed,
    )
    reload_difference = float(
        np.abs(
            reload_animals[list(PROBABILITY_COLUMNS)].to_numpy()
            - best_animals[list(PROBABILITY_COLUMNS)].to_numpy()
        ).max()
    )
    audit = {
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "base_seed": seed,
        "training_seed": training_seed,
        "loss": {
            "kind": "hybrid" if is_hybrid(pipeline) else "call_only",
            "call_weight": HYBRID_CALL_WEIGHT if is_hybrid(pipeline) else 1.0,
            "cat_weight": HYBRID_CAT_WEIGHT if is_hybrid(pipeline) else 0.0,
        },
        "best_validation_animal_cross_entropy": best_loss,
        "best_validation_animal_brier": idea068.brier(best_animals),
        "best_validation_animal_metrics": idea068.idea051.animal_metrics(
            best_animals
        ),
        "checkpoint_reload_max_probability_difference": reload_difference,
        "model": model.audit(),
        "train_seconds": float(time.perf_counter() - started),
        "history": history,
        "outer_test_accessed": False,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return audit, best_animals, best_calls


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea074-C1-hybrid-factorial-v1":
        raise RuntimeError("Unexpected IDEA-074 protocol")
    if protocol.get("status") != "locked_before_initial_evaluation":
        raise RuntimeError("IDEA-074 protocol is not locked")
    model = protocol["model"]
    if tuple(model["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-074 pipeline matrix changed")
    if tuple(model["base_seeds"]) != BASE_SEEDS:
        raise RuntimeError("IDEA-074 base-seed bank changed")
    if model["repeats"] != [0, 1, 2] or model["folds"] != [0, 1, 2, 3]:
        raise RuntimeError("IDEA-074 split scope changed")
    if model.get("outer_test_predictions") is not False:
        raise RuntimeError("IDEA-074 must not access outer-test predictions")
    constants = (
        int(model["age_residual_hidden"]) == AGE_RESIDUAL_HIDDEN
        and float(model["relative_cap"]) == CAP
        and float(model["rms_epsilon"]) == RMS_EPSILON
        and float(model["hybrid_call_weight"]) == HYBRID_CALL_WEIGHT
        and float(model["hybrid_cat_weight"]) == HYBRID_CAT_WEIGHT
    )
    if not constants:
        raise RuntimeError("IDEA-074 frozen model or loss constants changed")
    if model["trainable_parameters"] != EXPECTED_PARAMETERS:
        raise RuntimeError("IDEA-074 parameter lock changed")
    expected_fits = len(PIPELINES) * len(BASE_SEEDS) * 3 * 4
    if expected_fits != 144 or int(model["total_fits"]) != expected_fits:
        raise RuntimeError("IDEA-074 fit budget is inconsistent")
    full_seeds = {
        base + 10_000 * repeat + 100 * fold
        for base in BASE_SEEDS
        for repeat in range(3)
        for fold in range(4)
    }
    if len(full_seeds) != 36:
        raise RuntimeError("IDEA-074 derived full seeds are not unique")
    source_protocol = read_json(idea068.PROTOCOL_PATH)
    if protocol["fixed_training"] != source_protocol["fixed_training"]:
        raise RuntimeError("IDEA-074 changed the locked IDEA-068 training recipe")
    if protocol["determinism"] != source_protocol["determinism"]:
        raise RuntimeError("IDEA-074 changed the locked IDEA-068 determinism recipe")
    dependencies = protocol["dependencies"]
    checks = {
        REPO_ROOT / dependencies["idea_path"]: dependencies["idea_sha256"],
        idea066.PROTOCOL_PATH: dependencies["idea066_protocol_sha256"],
        Path(idea066.__file__).resolve(): dependencies["idea066_runner_sha256"],
        idea068.PROTOCOL_PATH: dependencies["idea068_protocol_sha256"],
        Path(idea068.__file__).resolve(): dependencies["idea068_runner_sha256"],
        idea071.PROTOCOL_PATH: dependencies["idea071_protocol_sha256"],
        Path(idea071.__file__).resolve(): dependencies["idea071_runner_sha256"],
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
            raise RuntimeError(f"IDEA-074 dependency checksum mismatch: {path}")


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
            raise RuntimeError(f"IDEA-074 resume identity mismatch for {key}")
    initialization = fit["initialization_audit"]
    if any(
        value != 0.0
        for value in initialization["max_logit_difference_vs_A0_C"].values()
    ) or not all(initialization["paired_state_dict_equal"].values()) or not initialization[
        "shared_AST_state_equal"
    ]:
        raise RuntimeError("IDEA-074 resume initialization audit mismatch")
    if fit["audit"]["checkpoint_reload_max_probability_difference"] != 0.0:
        raise RuntimeError("IDEA-074 resume checkpoint reload mismatch")
    for prefix in ("validation_animal", "validation_call"):
        path = REPO_ROOT / fit[f"{prefix}_predictions"]
        if not path.is_file() or idea068.sha256(path) != fit[f"{prefix}_sha256"]:
            raise RuntimeError(f"IDEA-074 resume prediction hash mismatch: {path}")


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
    ("C1C_minus_A0C", PIPELINES[2], PIPELINES[0]),
    ("A0H_minus_A0C", PIPELINES[1], PIPELINES[0]),
    ("C1H_minus_C1C", PIPELINES[3], PIPELINES[2]),
    ("C1H_minus_A0C", PIPELINES[3], PIPELINES[0]),
    ("C1H_minus_A0H", PIPELINES[3], PIPELINES[1]),
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
        # Probability-quality effects are improvement scores: positive is better.
        row[f"{name}_cross_entropy_improvement"] = (
            row[f"{right}_cross_entropy"] - row[f"{left}_cross_entropy"]
        )
        row[f"{name}_brier_improvement"] = (
            row[f"{right}_brier"] - row[f"{left}_brier"]
        )
        row[f"{name}_senior_recall"] = (
            row[f"{left}_senior_recall"] - row[f"{right}_senior_recall"]
        )
    row["interaction_macro_f1"] = (
        row["C1H_minus_C1C_macro_f1"] - row["A0H_minus_A0C_macro_f1"]
    )
    row["interaction_cross_entropy_improvement"] = (
        row["C1H_minus_C1C_cross_entropy_improvement"]
        - row["A0H_minus_A0C_cross_entropy_improvement"]
    )
    row["interaction_brier_improvement"] = (
        row["C1H_minus_C1C_brier_improvement"]
        - row["A0H_minus_A0C_brier_improvement"]
    )
    row["interaction_senior_recall"] = (
        row["C1H_minus_C1C_senior_recall"]
        - row["A0H_minus_A0C_senior_recall"]
    )


def comparison_summary(values: pd.Series) -> dict[str, Any]:
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
    by_key = {
        (fit["pipeline"], fit["base_seed"], fit["repeat"], fit["fold"]): fit
        for fit in fits
    }
    pooled: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
    per_seed_pipeline: dict[tuple[int, str], list[pd.DataFrame]] = {}
    fold_rows: list[dict[str, Any]] = []
    for base_seed in protocol["model"]["base_seeds"]:
        for repeat in protocol["model"]["repeats"]:
            for fold in protocol["model"]["folds"]:
                bundles: dict[str, Any] = {}
                for pipeline in PIPELINES:
                    frame = load_animals(by_key[(pipeline, base_seed, repeat, fold)])
                    tagged = frame.copy()
                    tagged["base_seed"] = base_seed
                    tagged["repeat"] = repeat
                    tagged["fold"] = fold
                    pooled[pipeline].append(tagged)
                    per_seed_pipeline.setdefault((base_seed, pipeline), []).append(
                        tagged
                    )
                    bundles[pipeline] = metric_bundle(frame)
                row: dict[str, Any] = {
                    "base_seed": int(base_seed),
                    "repeat": int(repeat),
                    "fold": int(fold),
                }
                add_pipeline_metrics(row, bundles)
                fold_rows.append(row)

    seed_repeat_rows: list[dict[str, Any]] = []
    for base_seed in protocol["model"]["base_seeds"]:
        for repeat in protocol["model"]["repeats"]:
            bundles = {}
            for pipeline in PIPELINES:
                parts = [
                    frame
                    for frame in pooled[pipeline]
                    if int(frame["base_seed"].iloc[0]) == int(base_seed)
                    and int(frame["repeat"].iloc[0]) == int(repeat)
                ]
                bundles[pipeline] = metric_bundle(pd.concat(parts, ignore_index=True))
            row = {"base_seed": int(base_seed), "repeat": int(repeat)}
            add_pipeline_metrics(row, bundles)
            seed_repeat_rows.append(row)
    seed_repeats = pd.DataFrame(seed_repeat_rows)

    fold_frame = pd.DataFrame(fold_rows)
    metric_columns = [
        column
        for column in fold_frame.columns
        if column not in ("base_seed", "repeat", "fold")
    ]
    split_cells = (
        fold_frame.groupby(["repeat", "fold"], as_index=False)[metric_columns]
        .mean()
        .sort_values(["repeat", "fold"])
        .reset_index(drop=True)
    )

    pooled_frames = {
        pipeline: pd.concat(parts, ignore_index=True)
        for pipeline, parts in pooled.items()
    }
    pooled_metrics = {
        pipeline: {
            "animal_occurrences": int(len(frame)),
            **metric_bundle(frame),
        }
        for pipeline, frame in pooled_frames.items()
    }

    per_seed_deltas: dict[str, dict[str, float]] = {}
    per_seed_senior: dict[str, dict[str, float]] = {}
    for base_seed in protocol["model"]["base_seeds"]:
        subset = seed_repeats[seed_repeats["base_seed"] == base_seed]
        per_seed_deltas[str(base_seed)] = {
            "C1C_minus_A0C": float(subset["C1C_minus_A0C_macro_f1"].mean()),
            "C1H_minus_C1C": float(subset["C1H_minus_C1C_macro_f1"].mean()),
            "C1H_minus_A0C": float(subset["C1H_minus_A0C_macro_f1"].mean()),
            "C1H_minus_A0H": float(subset["C1H_minus_A0H_macro_f1"].mean()),
            "interaction": float(subset["interaction_macro_f1"].mean()),
        }
        bundles = {
            pipeline: metric_bundle(
                pd.concat(per_seed_pipeline[(base_seed, pipeline)], ignore_index=True)
            )
            for pipeline in PIPELINES
        }
        per_seed_senior[str(base_seed)] = {
            "C1C_minus_A0C": float(
                bundles[PIPELINES[2]]["metrics"]["per_class"]["senior"]["recall"]
                - bundles[PIPELINES[0]]["metrics"]["per_class"]["senior"]["recall"]
            ),
            "C1H_minus_C1C": float(
                bundles[PIPELINES[3]]["metrics"]["per_class"]["senior"]["recall"]
                - bundles[PIPELINES[2]]["metrics"]["per_class"]["senior"]["recall"]
            ),
            "C1H_minus_A0H": float(
                bundles[PIPELINES[3]]["metrics"]["per_class"]["senior"]["recall"]
                - bundles[PIPELINES[1]]["metrics"]["per_class"]["senior"]["recall"]
            ),
        }

    means = {
        pipeline: {
            "macro_f1": float(seed_repeats[f"{pipeline}_macro_f1"].mean()),
            "cross_entropy": float(
                seed_repeats[f"{pipeline}_cross_entropy"].mean()
            ),
            "brier": float(seed_repeats[f"{pipeline}_brier"].mean()),
        }
        for pipeline in PIPELINES
    }
    comparison_columns = [
        f"{name}_{metric}"
        for name, _, _ in COMPARISONS
        for metric in (
            "macro_f1",
            "cross_entropy_improvement",
            "brier_improvement",
            "senior_recall",
        )
    ] + [
        "interaction_macro_f1",
        "interaction_cross_entropy_improvement",
        "interaction_brier_improvement",
        "interaction_senior_recall",
    ]
    comparisons = {
        column: comparison_summary(seed_repeats[column])
        for column in comparison_columns
    }

    gate = protocol["gate"]
    c1_gate = gate["C1_replication"]
    hybrid_gate = gate["hybrid_adoption"]
    synergy_gate = gate["synergy"]
    c1_delta = seed_repeats["C1C_minus_A0C_macro_f1"]
    c1_split = split_cells["C1C_minus_A0C_macro_f1"]
    hybrid_delta = seed_repeats["C1H_minus_C1C_macro_f1"]
    hybrid_split = split_cells["C1H_minus_C1C_macro_f1"]
    interaction = seed_repeats["interaction_macro_f1"]
    interaction_split = split_cells["interaction_macro_f1"]

    c1_conditions = {
        "mean_macro_f1_gain": float(c1_delta.mean())
        >= float(c1_gate["minimum_mean_macro_f1_gain"]),
        "positive_base_seeds": sum(
            values["C1C_minus_A0C"] > 0 for values in per_seed_deltas.values()
        )
        >= int(c1_gate["minimum_positive_base_seeds"]),
        "positive_seed_repeats": int((c1_delta > 0).sum())
        >= int(c1_gate["minimum_positive_seed_repeats"]),
        "nonnegative_split_cells": int((c1_split >= 0).sum())
        >= int(c1_gate["minimum_nonnegative_split_cells"]),
        "worst_split_cell": float(c1_split.min())
        >= float(c1_gate["minimum_worst_split_cell_delta"]),
        "cross_entropy_nonworse": means[PIPELINES[2]]["cross_entropy"]
        <= means[PIPELINES[0]]["cross_entropy"],
        "brier_nonworse": means[PIPELINES[2]]["brier"]
        <= means[PIPELINES[0]]["brier"],
        "per_base_seed_senior_recall": all(
            values["C1C_minus_A0C"]
            >= float(c1_gate["minimum_per_base_seed_senior_recall_delta"])
            for values in per_seed_senior.values()
        ),
    }
    hybrid_conditions = {
        "mean_macro_f1_gain": float(hybrid_delta.mean())
        >= float(hybrid_gate["minimum_mean_macro_f1_gain"]),
        "positive_base_seeds": sum(
            values["C1H_minus_C1C"] > 0 for values in per_seed_deltas.values()
        )
        >= int(hybrid_gate["minimum_positive_base_seeds"]),
        "positive_seed_repeats": int((hybrid_delta > 0).sum())
        >= int(hybrid_gate["minimum_positive_seed_repeats"]),
        "nonnegative_split_cells": int((hybrid_split >= 0).sum())
        >= int(hybrid_gate["minimum_nonnegative_split_cells"]),
        "worst_split_cell": float(hybrid_split.min())
        >= float(hybrid_gate["minimum_worst_split_cell_delta"]),
        "cross_entropy_nonworse": means[PIPELINES[3]]["cross_entropy"]
        <= means[PIPELINES[2]]["cross_entropy"],
        "brier_nonworse": means[PIPELINES[3]]["brier"]
        <= means[PIPELINES[2]]["brier"],
        "per_base_seed_senior_recall": all(
            values["C1H_minus_C1C"]
            >= float(hybrid_gate["minimum_per_base_seed_senior_recall_delta"])
            for values in per_seed_senior.values()
        ),
        "representation_under_hybrid_mean_macro_f1_gain": float(
            seed_repeats["C1H_minus_A0H_macro_f1"].mean()
        )
        >= float(
            hybrid_gate["minimum_representation_under_hybrid_mean_macro_f1_gain"]
        ),
        "representation_under_hybrid_positive_base_seeds": sum(
            values["C1H_minus_A0H"] > 0 for values in per_seed_deltas.values()
        )
        >= int(
            hybrid_gate["minimum_representation_under_hybrid_positive_base_seeds"]
        ),
        "representation_under_hybrid_cross_entropy_nonworse": means[PIPELINES[3]][
            "cross_entropy"
        ]
        <= means[PIPELINES[1]]["cross_entropy"],
        "representation_under_hybrid_brier_nonworse": means[PIPELINES[3]][
            "brier"
        ]
        <= means[PIPELINES[1]]["brier"],
        "representation_under_hybrid_per_base_seed_senior_recall": all(
            values["C1H_minus_A0H"]
            >= float(
                hybrid_gate[
                    "minimum_representation_under_hybrid_per_base_seed_senior_recall_delta"
                ]
            )
            for values in per_seed_senior.values()
        ),
    }
    synergy_conditions = {
        "mean_interaction_positive": float(interaction.mean())
        > float(synergy_gate["minimum_mean_interaction_macro_f1"]),
        "nonnegative_base_seeds": sum(
            values["interaction"] >= 0 for values in per_seed_deltas.values()
        )
        >= int(synergy_gate["minimum_nonnegative_base_seeds"]),
        "nonnegative_seed_repeats": int((interaction >= 0).sum())
        >= int(synergy_gate["minimum_nonnegative_seed_repeats"]),
        "nonnegative_split_cells": int((interaction_split >= 0).sum())
        >= int(synergy_gate["minimum_nonnegative_split_cells"]),
        "worst_split_cell": float(interaction_split.min())
        >= float(synergy_gate["minimum_worst_split_cell_interaction"]),
    }
    return {
        "status": "complete",
        "outer_test_accessed": False,
        "fits": len(fits),
        "paired_fold_comparisons": len(fold_rows),
        "seed_repeat_estimates": len(seed_repeat_rows),
        "split_cell_estimates": int(len(split_cells)),
        "independence_note": (
            "Fold deltas and pooled animal occurrences repeat animals across "
            "seeds/repeats and are descriptive, not independent samples."
        ),
        "fold_results": fold_rows,
        "seed_repeat_results": seed_repeat_rows,
        "split_cell_results": split_cells.to_dict(orient="records"),
        "seed_repeat_level": {
            "pipeline_means": means,
            "comparisons": comparisons,
        },
        "per_base_seed_mean_deltas": per_seed_deltas,
        "per_base_seed_senior_recall_deltas": per_seed_senior,
        "pooled_validation": pooled_metrics,
        "gate_conditions": {
            "C1_replication": c1_conditions,
            "hybrid_adoption": hybrid_conditions,
            "synergy": synergy_conditions,
        },
        "C1_replication_passed": bool(all(c1_conditions.values())),
        "hybrid_adoption_passed": bool(all(hybrid_conditions.values())),
        "synergy_passed": bool(all(synergy_conditions.values())),
        "gate_passed": bool(
            all(c1_conditions.values())
            and all(hybrid_conditions.values())
            and all(synergy_conditions.values())
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    idea068.configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = resolve_run_root(args.output_subdir)
    run_root.mkdir(parents=True, exist_ok=True)
    store = idea068.idea051.reference.historical.idea019.load_feature_store()
    if len(store.call_ids) != 792 or len(np.unique(store.cat_ids.astype(str))) != 111:
        raise RuntimeError("IDEA-074 expected exactly 792 calls from 111 cats")
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
            raise RuntimeError("Existing IDEA-074 run manifest differs")
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
                initialization = initialization_audit(
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
                        if audit["model"]["trainable_parameters"] != EXPECTED_PARAMETERS[
                            pipeline
                        ]:
                            raise RuntimeError("IDEA-074 trained parameter audit mismatch")
                        expected_calls = set(int(value) for value in indices["validation"])
                        if set(int(value) for value in calls["call_index"]) != expected_calls:
                            raise RuntimeError("IDEA-074 prediction role coverage mismatch")
                        if audit["checkpoint_reload_max_probability_difference"] != 0.0:
                            raise RuntimeError("IDEA-074 checkpoint reload mismatch")
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
                            "initialization_audit": initialization,
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
                            reference["cat_order_sha256"]
                            != current["cat_order_sha256"]
                            or reference["call_coverage_sha256"]
                            != current["call_coverage_sha256"]
                        ):
                            raise RuntimeError("IDEA-074 paired batch order differs")
    summary = aggregate(completed, protocol)
    write_json(run_root / "initial_evaluation_summary.json", summary)
    write_json(
        run_root / "run_summary.json",
        {
            "status": "complete",
            "completed_fits": len(completed),
            "expected_fits": int(protocol["model"]["total_fits"]),
            "C1_replication_passed": summary["C1_replication_passed"],
            "hybrid_adoption_passed": summary["hybrid_adoption_passed"],
            "synergy_passed": summary["synergy_passed"],
            "gate_passed": summary["gate_passed"],
        },
    )
    return summary


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
