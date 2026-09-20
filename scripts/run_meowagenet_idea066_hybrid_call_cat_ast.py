"""Run the IDEA-066 inner-only hybrid call+cat AST screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_idea051_cat_set as idea051  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea066_hybrid_call_cat_ast_v1.json"
)
PIPELINES = ("H0_matched_call_only", "H1_hybrid_call_cat")
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-subdir", default="meowagenet_idea066_hybrid_call_cat_ast_v1"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea066-hybrid-call-cat-ast-v1":
        raise RuntimeError("Unexpected IDEA-066 protocol")
    if protocol.get("status") != "locked_before_inner_screen":
        raise RuntimeError("IDEA-066 protocol is not locked")
    if tuple(protocol["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-066 pipeline matrix changed")
    screen = protocol["screen"]
    expected = len(PIPELINES) * len(screen["repeats"]) * len(
        screen["outer_folds_used_as_inner_role_definitions"]
    )
    if expected != int(screen["total_fits"]):
        raise RuntimeError("IDEA-066 fit budget is inconsistent")
    dependencies = protocol["dependencies"]
    checks = {
        REPO_ROOT / protocol["data"]["roles_path"]: protocol["data"][
            "roles_sha256"
        ],
        REPO_ROOT / dependencies["idea_path"]: dependencies["idea_sha256"],
        REPO_ROOT / dependencies["idea051_runner_path"]: dependencies[
            "idea051_runner_sha256"
        ],
        REPO_ROOT / dependencies["frozen_embedding_path"]: dependencies[
            "frozen_embedding_sha256"
        ],
        Path(__file__).resolve(): dependencies["runner_sha256"],
    }
    for path, expected_sha in checks.items():
        if not path.is_file() or sha256(path) != expected_sha:
            raise RuntimeError(f"IDEA-066 dependency checksum mismatch: {path}")


def build_model(
    protocol: dict[str, Any], store: Any, train_indices: np.ndarray
) -> torch.nn.Module:
    embeddings = store.frozen_embeddings[train_indices]
    head = idea051.reference.historical.idea019.ClassificationHead(
        mean=embeddings.mean(axis=0),
        scale=embeddings.std(axis=0),
        dropout=float(protocol["fixed_training"]["dropout"]),
    )
    return idea051.reference.historical.idea019.FrozenClassifier(head)


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("A training role is missing a class")
    return (len(labels) / (3.0 * counts)).astype(np.float32)


def cat_probabilities(
    call_probabilities: torch.Tensor,
    instance_to_cat: torch.Tensor,
    cat_count: int,
) -> torch.Tensor:
    result = torch.zeros(
        (cat_count, call_probabilities.shape[1]),
        dtype=call_probabilities.dtype,
        device=call_probabilities.device,
    )
    result.index_add_(0, instance_to_cat, call_probabilities)
    counts = torch.bincount(instance_to_cat, minlength=cat_count).to(
        call_probabilities.dtype
    )
    return result / counts[:, None]


def train_one_epoch(
    pipeline: str,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    store: Any,
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
        batch = idea051.move_set_batch(cpu_batch, device)
        call_indices_numpy = cpu_batch["call_indices"].numpy().astype(np.int64)
        call_labels = torch.from_numpy(store.labels[call_indices_numpy]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            call_logits = model(
                batch["embeddings"], batch["instance_to_cat"], len(call_labels)
            )
            per_call = torch.nn.functional.cross_entropy(
                call_logits, call_labels, reduction="none"
            )
            call_weights = call_class_weights[call_labels]
            call_loss = (per_call * call_weights).sum() / call_weights.sum()
            call_probs = torch.softmax(call_logits, dim=1)
            animal_probs = cat_probabilities(
                call_probs, batch["instance_to_cat"], len(batch["labels"])
            )
            true_probs = animal_probs[
                torch.arange(len(batch["labels"]), device=device), batch["labels"]
            ]
            per_cat = -torch.log(torch.clamp(true_probs, min=1.0e-7))
            cat_weights = cat_class_weights[batch["labels"]]
            cat_loss = (per_cat * cat_weights).sum() / cat_weights.sum()
            if pipeline == PIPELINES[0]:
                loss = call_loss
            elif pipeline == PIPELINES[1]:
                loss = 0.5 * call_loss + 0.5 * cat_loss
            else:
                raise ValueError(pipeline)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        total_values.append(float(loss.detach()))
        call_values.append(float(call_loss.detach()))
        cat_values.append(float(cat_loss.detach()))
        processed_cats.extend(str(value) for value in cpu_batch["cat_ids"])
        processed_calls.extend(int(value) for value in call_indices_numpy)
        batch_sizes.append(len(cpu_batch["cat_ids"]))
    if len(processed_cats) != len(set(processed_cats)):
        raise RuntimeError("A training epoch repeated a cat")
    if len(processed_calls) != len(set(processed_calls)):
        raise RuntimeError("A training epoch repeated a call")
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
            "cat_order_sha256": hashlib.sha256(
                "\n".join(processed_cats).encode("utf-8")
            ).hexdigest(),
            "call_coverage_sha256": hashlib.sha256(
                np.sort(np.asarray(processed_calls, dtype="<i8")).tobytes()
            ).hexdigest(),
        },
    )


def predict(
    model: torch.nn.Module,
    store: Any,
    indices: np.ndarray,
    cat_batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = idea051.CatSetDataset(store, indices)
    loader = idea051.build_set_loader(dataset, cat_batch_size, False, seed)
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = idea051.move_set_batch(cpu_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(
                    batch["embeddings"],
                    batch["instance_to_cat"],
                    len(cpu_batch["call_indices"]),
                )
                probabilities = torch.softmax(logits, dim=1).float().cpu().numpy()
            call_indices = cpu_batch["call_indices"].numpy().astype(np.int64)
            for local_index, call_index in enumerate(call_indices):
                rows.append(
                    {
                        "call_index": int(call_index),
                        "call_id": str(store.call_ids[call_index]),
                        "cat_id": str(store.cat_ids[call_index]),
                        "true_label": int(store.labels[call_index]),
                        **{
                            column: float(probabilities[local_index, class_index])
                            for class_index, column in enumerate(PROBABILITY_COLUMNS)
                        },
                    }
                )
    calls = pd.DataFrame(rows).sort_values("call_index").reset_index(drop=True)
    animals = idea051.calls_to_animals(calls)
    return animals, calls


def brier(animals: pd.DataFrame) -> float:
    probabilities = animals[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    labels = animals["true_label"].to_numpy(dtype=np.int64)
    targets = np.eye(3, dtype=float)[labels]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def fit_inner(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    idea051.reference.historical.set_seed(seed)
    fixed = protocol["fixed_training"]
    model = build_model(protocol, store, train_indices).to(device)
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=float(fixed["learning_rate"]),
        eps=float(fixed["optimizer_epsilon"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    dataset = idea051.CatSetDataset(store, train_indices)
    loader = idea051.build_set_loader(
        dataset, int(fixed["cat_batch_size"]), True, seed
    )
    call_weights = torch.from_numpy(
        class_weights(store.labels[train_indices])
    ).to(device)
    cat_weights = torch.from_numpy(class_weights(dataset.labels)).to(device)
    best_loss = float("inf")
    best_epoch = 1
    best_state = idea051.cpu_state_dict(model)
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
            call_weights,
            cat_weights,
            device,
            float(fixed["gradient_clip"]),
        )
        animals, calls = predict(
            model,
            store,
            validation_indices,
            int(fixed["cat_batch_size"]) * 2,
            device,
            seed,
        )
        validation_loss = idea051.animal_cross_entropy(animals)
        metrics = idea051.animal_metrics(animals)
        history.append(
            {
                "epoch": epoch,
                "train_loss": losses,
                "train_audit": train_audit,
                "validation_animal_cross_entropy": validation_loss,
                "validation_animal_brier": brier(animals),
                "validation_animal_metrics": metrics,
            }
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = idea051.cpu_state_dict(model)
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
        raise RuntimeError("No IDEA-066 checkpoint was selected")
    model.load_state_dict(best_state)
    reload_animals, _ = predict(
        model,
        store,
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
        "best_validation_animal_cross_entropy": best_loss,
        "best_validation_animal_brier": brier(best_animals),
        "best_validation_animal_metrics": idea051.animal_metrics(best_animals),
        "checkpoint_reload_max_probability_difference": reload_difference,
        "train_seconds": float(time.perf_counter() - started),
        "history": history,
        "outer_test_accessed": False,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return audit, best_animals, best_calls


def aggregate(fits: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    by_key = {
        (fit["pipeline"], fit["repeat"], fit["outer_fold"]): fit for fit in fits
    }
    fold_rows = []
    pooled: dict[str, list[pd.DataFrame]] = {pipeline: [] for pipeline in PIPELINES}
    for repeat in protocol["screen"]["repeats"]:
        for outer_fold in protocol["screen"][
            "outer_folds_used_as_inner_role_definitions"
        ]:
            controls = by_key[(PIPELINES[0], repeat, outer_fold)]
            hybrids = by_key[(PIPELINES[1], repeat, outer_fold)]
            left = pd.read_csv(
                REPO_ROOT / controls["validation_animal_predictions"],
                dtype={"cat_id": str},
            )
            right = pd.read_csv(
                REPO_ROOT / hybrids["validation_animal_predictions"],
                dtype={"cat_id": str},
            )
            for pipeline, frame in ((PIPELINES[0], left), (PIPELINES[1], right)):
                tagged = frame.copy()
                tagged["repeat"] = repeat
                tagged["outer_fold"] = outer_fold
                pooled[pipeline].append(tagged)
            lm = idea051.animal_metrics(left)
            rm = idea051.animal_metrics(right)
            fold_rows.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "control_macro_f1": lm["macro_f1"],
                    "hybrid_macro_f1": rm["macro_f1"],
                    "delta_macro_f1": rm["macro_f1"] - lm["macro_f1"],
                    "control_cross_entropy": idea051.animal_cross_entropy(left),
                    "hybrid_cross_entropy": idea051.animal_cross_entropy(right),
                    "control_brier": brier(left),
                    "hybrid_brier": brier(right),
                    "control_senior_recall": lm["per_class"]["senior"]["recall"],
                    "hybrid_senior_recall": rm["per_class"]["senior"]["recall"],
                }
            )
    folds = pd.DataFrame(fold_rows)
    pooled_frames = {
        pipeline: pd.concat(parts, ignore_index=True) for pipeline, parts in pooled.items()
    }
    pooled_metrics = {
        pipeline: {
            "animal_occurrences": int(len(frame)),
            "metrics": idea051.animal_metrics(frame),
            "cross_entropy": idea051.animal_cross_entropy(frame),
            "brier": brier(frame),
        }
        for pipeline, frame in pooled_frames.items()
    }
    per_repeat_senior_delta = {}
    for repeat in protocol["screen"]["repeats"]:
        metrics = {}
        for pipeline, frame in pooled_frames.items():
            metrics[pipeline] = idea051.animal_metrics(frame[frame["repeat"] == repeat])
        per_repeat_senior_delta[str(repeat)] = (
            metrics[PIPELINES[1]]["per_class"]["senior"]["recall"]
            - metrics[PIPELINES[0]]["per_class"]["senior"]["recall"]
        )
    gate = protocol["gate"]
    conditions = {
        "mean_fold_macro_f1_gain": float(folds["delta_macro_f1"].mean())
        >= float(gate["minimum_mean_fold_macro_f1_gain"]),
        "positive_folds": int((folds["delta_macro_f1"] > 0).sum())
        >= int(gate["minimum_positive_folds"]),
        "pooled_cross_entropy_nonworse": pooled_metrics[PIPELINES[1]][
            "cross_entropy"
        ]
        <= pooled_metrics[PIPELINES[0]]["cross_entropy"],
        "pooled_brier_nonworse": pooled_metrics[PIPELINES[1]]["brier"]
        <= pooled_metrics[PIPELINES[0]]["brier"],
        "worst_fold_macro_f1": float(folds["delta_macro_f1"].min())
        >= float(gate["minimum_worst_fold_macro_f1_delta"]),
        "per_repeat_senior_recall": all(
            delta >= float(gate["minimum_per_repeat_senior_recall_delta"])
            for delta in per_repeat_senior_delta.values()
        ),
    }
    return {
        "status": "complete",
        "outer_test_accessed": False,
        "fits": len(fits),
        "fold_results": fold_rows,
        "mean_fold_macro_f1": {
            PIPELINES[0]: float(folds["control_macro_f1"].mean()),
            PIPELINES[1]: float(folds["hybrid_macro_f1"].mean()),
            "delta": float(folds["delta_macro_f1"].mean()),
        },
        "positive_folds": int((folds["delta_macro_f1"] > 0).sum()),
        "worst_fold_macro_f1_delta": float(folds["delta_macro_f1"].min()),
        "pooled_validation": pooled_metrics,
        "per_repeat_senior_recall_delta": per_repeat_senior_delta,
        "gate_conditions": conditions,
        "gate_passed": bool(all(conditions.values())),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = (idea051.RUNS_ROOT / args.output_subdir).resolve()
    if idea051.RUNS_ROOT.resolve() not in run_root.parents:
        raise ValueError("--output-subdir must stay below runs")
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "run_manifest.json"
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "outer_test_accessed": False,
        "pipelines": list(PIPELINES),
        "screen": protocol["screen"],
    }
    if manifest_path.exists():
        if not args.resume or read_json(manifest_path) != manifest:
            raise RuntimeError("Existing IDEA-066 run manifest differs")
    else:
        write_json(manifest_path, manifest)
    roles = pd.read_csv(
        REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str}
    )
    store = idea051.reference.historical.idea019.load_feature_store()
    if len(store.call_ids) != 792 or len(np.unique(store.cat_ids)) != 111:
        raise RuntimeError("IDEA-066 analysis view changed")
    device = idea051.reference.historical.idea019.resolve_device(args.device)
    completed: list[dict[str, Any]] = []
    screen = protocol["screen"]
    for repeat in screen["repeats"]:
        for outer_fold in screen["outer_folds_used_as_inner_role_definitions"]:
            indices = idea051.reference.historical.fold_indices(
                store, roles, repeat, outer_fold, include_test=False
            )
            seed = idea051.reference.historical.full_seed(
                int(screen["base_seed"]), repeat, outer_fold
            )
            pair: list[dict[str, Any]] = []
            for pipeline in PIPELINES:
                output_dir = (
                    run_root
                    / "fits"
                    / pipeline
                    / f"repeat_{repeat}"
                    / f"fold_{outer_fold}"
                )
                summary_path = output_dir / "fit_summary.json"
                if summary_path.exists():
                    if not args.resume:
                        raise FileExistsError(summary_path)
                    fit = read_json(summary_path)
                    completed.append(fit)
                    pair.append(fit)
                    continue
                print(
                    f"=== {pipeline} repeat={repeat} fold={outer_fold} seed={seed} ===",
                    flush=True,
                )
                audit, animals, calls = fit_inner(
                    pipeline,
                    protocol,
                    store,
                    indices["train"],
                    indices["validation"],
                    device,
                    seed,
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                animal_path = output_dir / "validation_animal_predictions.csv"
                call_path = output_dir / "validation_call_predictions.csv"
                animals.to_csv(animal_path, index=False)
                calls.to_csv(call_path, index=False)
                fit = {
                    "status": "complete",
                    "pipeline": pipeline,
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "base_seed": int(screen["base_seed"]),
                    "full_seed": seed,
                    "outer_test_accessed": False,
                    "train_calls": int(len(indices["train"])),
                    "validation_calls": int(len(indices["validation"])),
                    "validation_cats": int(animals["cat_id"].nunique()),
                    "validation_animal_predictions": animal_path.relative_to(
                        REPO_ROOT
                    ).as_posix(),
                    "validation_call_predictions": call_path.relative_to(
                        REPO_ROOT
                    ).as_posix(),
                    "audit": audit,
                }
                write_json(summary_path, fit)
                completed.append(fit)
                pair.append(fit)
            common = min(
                len(pair[0]["audit"]["history"]),
                len(pair[1]["audit"]["history"]),
            )
            for epoch in range(common):
                left = pair[0]["audit"]["history"][epoch]["train_audit"]
                right = pair[1]["audit"]["history"][epoch]["train_audit"]
                if left["cat_order_sha256"] != right["cat_order_sha256"]:
                    raise RuntimeError("IDEA-066 paired cat order differs")
                if left["call_coverage_sha256"] != right["call_coverage_sha256"]:
                    raise RuntimeError("IDEA-066 paired call coverage differs")
    summary = aggregate(completed, protocol)
    write_json(run_root / "inner_screen_summary.json", summary)
    write_json(
        run_root / "run_summary.json",
        {
            "status": "complete",
            "completed_fits": len(completed),
            "expected_fits": int(screen["total_fits"]),
            "gate_passed": summary["gate_passed"],
        },
    )
    return summary


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
