"""Run IDEA-056 matched call-level versus animal-level checkpoint selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import f1_score


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_ast_cat_balance_global_weighting_v1 as reference  # noqa: E402
import run_meowagenet_idea051_cat_set as evaluation  # noqa: E402
import run_meowagenet_idea055_checkpoint_calibration as shared  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea056_checkpoint_selection_confirmation_v1.json"
)
PLAN_PATH = (
    REPO_ROOT
    / "plan"
    / "IDEA-056_animal_level_checkpoint_selection_confirmation.md"
)
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
RUN_ROOT = (
    REPO_ROOT / "runs" / "meowagenet_idea056_checkpoint_selection_confirmation_v1"
)
PIPELINES = (
    "C0_call_level_CE_selection",
    "C1_animal_level_CE_selection",
)
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "evaluate"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def git_blob_sha256(revision: str, path: Path) -> str:
    content = subprocess.check_output(
        ["git", "show", f"{revision}:{repo_relative(path)}"], cwd=REPO_ROOT
    )
    return hashlib.sha256(content).hexdigest()


def resolve_device(requested: str) -> torch.device:
    return shared.resolve_device(requested)


def verify_protocol(protocol: dict[str, Any]) -> None:
    if (
        protocol["protocol_id"]
        != "meowagenet-idea056-checkpoint-selection-confirmation-v1"
    ):
        raise RuntimeError("Unexpected IDEA-056 protocol")
    settings = protocol["confirmation_evaluation"]
    if tuple(settings["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-056 pipeline pair changed")
    trajectories = (
        len(settings["base_seeds"])
        * len(settings["repeats"])
        * len(settings["outer_folds"])
    )
    if trajectories != int(settings["shared_training_trajectories"]):
        raise RuntimeError("IDEA-056 trajectory budget is inconsistent")
    if trajectories * len(PIPELINES) != int(settings["pipeline_fold_predictions"]):
        raise RuntimeError("IDEA-056 prediction budget is inconsistent")
    if len(settings["base_seeds"]) * len(settings["repeats"]) != int(
        settings["paired_complete_oof_comparisons"]
    ):
        raise RuntimeError("IDEA-056 paired comparison budget is inconsistent")
    if set(settings["base_seeds"]) != {43, 101}:
        raise RuntimeError("IDEA-056 primary confirmation seeds changed")
    fixed = protocol["fixed_training"]
    if int(fixed["micro_batch_size"]) * int(
        fixed["gradient_accumulation_steps"]
    ) != int(fixed["accumulation_window_calls"]):
        raise RuntimeError("IDEA-056 accumulation window is inconsistent")
    dependencies = protocol["dependencies"]
    checks = {
        PLAN_PATH: protocol["idea"]["sha256"],
        ROLES_PATH: protocol["splits"]["roles_sha256"],
        REPO_ROOT / dependencies["global_weighting_runner"]: dependencies[
            "global_weighting_runner_sha256"
        ],
        REPO_ROOT / dependencies["evaluation_runner"]: dependencies[
            "evaluation_runner_sha256"
        ],
        REPO_ROOT / dependencies["checkpoint_runner"]: dependencies[
            "checkpoint_runner_sha256"
        ],
        REPO_ROOT / dependencies["frozen_embedding_path"]: dependencies[
            "frozen_embedding_sha256"
        ],
    }
    for path, expected in checks.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"IDEA-056 dependency checksum mismatch: {path}")


def load_inputs() -> tuple[Any, pd.DataFrame]:
    store = reference.ast_base.load_feature_store()
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    if len(store.call_ids) != 792:
        raise RuntimeError("IDEA-056 expects 792 calls")
    if len(np.unique(store.cat_ids)) != 111 or roles["cat_id"].nunique() != 111:
        raise RuntimeError("IDEA-056 expects 111 cats")
    return store, roles


def unweighted_call_cross_entropy(calls: pd.DataFrame) -> float:
    probabilities = calls[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    labels = calls["true_label"].to_numpy(dtype=np.int64)
    true_probabilities = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(true_probabilities, 1.0e-12, 1.0)).mean())


def earliest_minimum_epoch(history: list[dict[str, Any]], field: str) -> int:
    if not history:
        raise ValueError("Checkpoint history is empty")
    return int(min(history, key=lambda row: (float(row[field]), int(row["epoch"])))["epoch"])


def dual_patience_exhausted(
    call_without_improvement: int,
    animal_without_improvement: int,
    patience: int,
) -> bool:
    return bool(
        int(call_without_improvement) >= int(patience)
        and int(animal_without_improvement) >= int(patience)
    )


def probability_sum_error(frame: pd.DataFrame) -> float:
    values = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float).sum(axis=1)
    return float(np.abs(values - 1.0).max())


def fit_inner_trajectory(
    protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
    max_epochs_override: int | None = None,
) -> dict[str, Any]:
    reference.historical.set_seed(seed)
    values = reference.training_values(protocol)
    model = reference.build_model(protocol, store, train_indices).to(device)
    parameters = shared.trainable_parameter_audit(model)
    initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=values["learning_rate"],
        eps=values["optimizer_epsilon"],
    )
    weights_numpy = reference.global_class_balanced_call_weights(
        store.labels, train_indices
    )
    weights = torch.from_numpy(weights_numpy).to(device)
    train_loader = reference.build_train_loader(
        store, train_indices, values["batch_size"], seed
    )
    validation_loader = shared.prediction_loader(
        store, validation_indices, protocol, seed
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    maximum_epochs = max_epochs_override or values["max_epochs"]
    patience = int(values["patience"])
    best_call = float("inf")
    best_animal = float("inf")
    call_without_improvement = 0
    animal_without_improvement = 0
    history: list[dict[str, Any]] = []
    last_calls: pd.DataFrame | None = None
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(maximum_epochs) + 1):
        train_loss, train_audit = reference.train_one_epoch_global_weighted(
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
        calls = shared.predict_calls(model, validation_loader, store, device)
        animals = evaluation.calls_to_animals(calls)
        call_loss = unweighted_call_cross_entropy(calls)
        animal_loss = evaluation.animal_cross_entropy(animals)
        metrics = evaluation.animal_metrics(animals)
        history.append(
            {
                "epoch": int(epoch),
                "train_weighted_loss": float(train_loss),
                "validation_call_cross_entropy": float(call_loss),
                "validation_animal_cross_entropy": float(animal_loss),
                "validation_animal_macro_f1": metrics["macro_f1"],
                "validation_animal_balanced_accuracy": metrics[
                    "balanced_accuracy"
                ],
                "validation_animal_qwk": metrics["quadratic_weighted_kappa"],
                "train_unit_audit": train_audit,
            }
        )
        if call_loss < best_call - 1.0e-12:
            best_call = call_loss
            call_without_improvement = 0
        else:
            call_without_improvement += 1
        if animal_loss < best_animal - 1.0e-12:
            best_animal = animal_loss
            animal_without_improvement = 0
        else:
            animal_without_improvement += 1
        last_calls = calls
        print(
            f"IDEA056 inner epoch={epoch} train={train_loss:.4f} "
            f"call_CE={call_loss:.4f} animal_CE={animal_loss:.4f} "
            f"animal_F1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if max_epochs_override is None and dual_patience_exhausted(
            call_without_improvement, animal_without_improvement, patience
        ):
            break
    selected_call_epoch = earliest_minimum_epoch(
        history, "validation_call_cross_entropy"
    )
    selected_animal_epoch = earliest_minimum_epoch(
        history, "validation_animal_cross_entropy"
    )
    updates = {
        name: float((parameter.detach().cpu() - initial[name]).abs().max())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if last_calls is None:
        raise RuntimeError("IDEA-056 inner trajectory produced no validation output")
    audit = {
        "stopped_epoch": int(len(history)),
        "selected_call_epoch": selected_call_epoch,
        "selected_animal_epoch": selected_animal_epoch,
        "selected_epoch_difference": int(selected_animal_epoch - selected_call_epoch),
        "best_validation_call_cross_entropy": float(
            history[selected_call_epoch - 1]["validation_call_cross_entropy"]
        ),
        "best_validation_animal_cross_entropy": float(
            history[selected_animal_epoch - 1]["validation_animal_cross_entropy"]
        ),
        "final_call_without_improvement": int(call_without_improvement),
        "final_animal_without_improvement": int(animal_without_improvement),
        "history": history,
        "last_validation_probability_sum_max_error": probability_sum_error(
            last_calls
        ),
        "target_weight_audit": reference.target_weight_audit(
            weights_numpy, store, train_indices
        ),
        "parameters": parameters,
        "updated_trainable_parameter_tensors": int(
            sum(value > 0.0 for value in updates.values())
        ),
        "maximum_trainable_parameter_updates": updates,
        "train_seconds": float(time.perf_counter() - started),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return audit


def fit_outer_trajectory(
    protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    selected_call_epoch: int,
    selected_animal_epoch: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    reference.historical.set_seed(seed)
    values = reference.training_values(protocol)
    model = reference.build_model(protocol, store, train_indices).to(device)
    parameters = shared.trainable_parameter_audit(model)
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=values["learning_rate"],
        eps=values["optimizer_epsilon"],
    )
    weights_numpy = reference.global_class_balanced_call_weights(
        store.labels, train_indices
    )
    weights = torch.from_numpy(weights_numpy).to(device)
    train_loader = reference.build_train_loader(
        store, train_indices, values["batch_size"], seed
    )
    test_loader = shared.prediction_loader(store, test_indices, protocol, seed)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    checkpoints = {
        PIPELINES[0]: int(selected_call_epoch),
        PIPELINES[1]: int(selected_animal_epoch),
    }
    requested_epochs = set(checkpoints.values())
    max_epoch = max(requested_epochs)
    calls_by_epoch: dict[int, pd.DataFrame] = {}
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, max_epoch + 1):
        train_loss, train_audit = reference.train_one_epoch_global_weighted(
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
        if epoch in requested_epochs:
            calls_by_epoch[epoch] = shared.predict_calls(
                model, test_loader, store, device
            )
        print(
            f"IDEA056 outer epoch={epoch}/{max_epoch} train={train_loss:.4f}",
            flush=True,
        )
    calls = {
        pipeline: calls_by_epoch[epoch].copy()
        for pipeline, epoch in checkpoints.items()
    }
    animals = {
        pipeline: evaluation.calls_to_animals(frame)
        for pipeline, frame in calls.items()
    }
    audit = {
        "maximum_epoch": int(max_epoch),
        "checkpoint_epochs": checkpoints,
        "shared_checkpoint": bool(selected_call_epoch == selected_animal_epoch),
        "history": history,
        "test_metrics": {
            pipeline: evaluation.animal_metrics(frame)
            for pipeline, frame in animals.items()
        },
        "test_animal_cross_entropy": {
            pipeline: evaluation.animal_cross_entropy(frame)
            for pipeline, frame in animals.items()
        },
        "probability_sum_max_error": {
            pipeline: probability_sum_error(frame)
            for pipeline, frame in animals.items()
        },
        "target_weight_audit": reference.target_weight_audit(
            weights_numpy, store, train_indices
        ),
        "parameters": parameters,
        "train_and_predict_seconds": float(time.perf_counter() - started),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    outputs = {
        **animals,
        **{f"calls::{pipeline}": frame for pipeline, frame in calls.items()},
    }
    return outputs, audit


def environment_lock(protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
    }


def smoke(
    protocol: dict[str, Any],
    store: Any,
    roles: pd.DataFrame,
    device: torch.device,
    resume: bool,
) -> None:
    output = RUN_ROOT / "smoke" / "summary.json"
    if output.is_file():
        if not resume:
            raise FileExistsError(output)
        previous = read_json(output)
        if previous.get("status") == "passed":
            print(output.read_text(encoding="utf-8"), flush=True)
            return
    settings = protocol["smoke"]
    seed = reference.historical.full_seed(
        int(settings["base_seed"]),
        int(settings["repeat"]),
        int(settings["outer_fold"]),
    )
    indices = reference.historical.fold_indices(
        store,
        roles,
        int(settings["repeat"]),
        int(settings["outer_fold"]),
        include_test=False,
    )
    inner = fit_inner_trajectory(
        protocol,
        store,
        indices["train"],
        indices["validation"],
        device,
        seed,
        max_epochs_override=int(settings["epochs"]),
    )
    tie_history = [
        {"epoch": 1, "metric": 0.5},
        {"epoch": 2, "metric": 0.5},
    ]
    earliest_tie_epoch = earliest_minimum_epoch(tie_history, "metric")
    status = "passed"
    if (
        inner["parameters"]["trainable_parameters"] != 99075
        or inner["updated_trainable_parameter_tensors"]
        != inner["parameters"]["trainable_parameter_tensors"]
        or len(inner["history"]) != int(settings["epochs"])
        or any(
            "validation_call_cross_entropy" not in row
            or "validation_animal_cross_entropy" not in row
            for row in inner["history"]
        )
        or earliest_tie_epoch != 1
        or not dual_patience_exhausted(8, 8, 8)
        or dual_patience_exhausted(8, 7, 8)
        or inner["last_validation_probability_sum_max_error"] > 1.0e-6
    ):
        status = "failed"
    result = {
        "status": status,
        "stage": "inner_only_smoke",
        "outer_test_accessed": False,
        "base_seed": int(settings["base_seed"]),
        "repeat": int(settings["repeat"]),
        "outer_fold": int(settings["outer_fold"]),
        "full_seed": int(seed),
        "calls": int(len(store.call_ids)),
        "cats": int(len(np.unique(store.cat_ids))),
        "inner_train_calls": int(len(indices["train"])),
        "inner_validation_calls": int(len(indices["validation"])),
        "selected_call_epoch": inner["selected_call_epoch"],
        "selected_animal_epoch": inner["selected_animal_epoch"],
        "earliest_tie_epoch_test": earliest_tie_epoch,
        "dual_patience_test": {
            "both_eight": dual_patience_exhausted(8, 8, 8),
            "call_eight_animal_seven": dual_patience_exhausted(8, 7, 8),
        },
        "parameters": inner["parameters"],
        "updated_trainable_parameter_tensors": inner[
            "updated_trainable_parameter_tensors"
        ],
        "inner_audit": inner,
    }
    write_json(output, result)
    if status != "passed":
        raise RuntimeError("IDEA-056 smoke failed")
    environment_path = RUN_ROOT / "environment_lock.json"
    write_json(environment_path, environment_lock(protocol, device))
    revision = git_revision()
    if revision is None:
        raise RuntimeError("IDEA-056 requires a committed source revision")
    for path in (PLAN_PATH, PROTOCOL_PATH, Path(__file__)):
        if git_blob_sha256(revision, path) != sha256(path):
            raise RuntimeError(f"IDEA-056 source differs from commit {revision}: {path}")
    lock = {
        "schema_version": "1.0",
        "status": "locked_after_inner_only_smoke",
        "protocol_id": protocol["protocol_id"],
        "outer_test_accessed": False,
        "code_commit": revision,
        "plan_sha256": sha256(PLAN_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "roles_sha256": sha256(ROLES_PATH),
        "smoke_summary_sha256": sha256(output),
        "environment_lock_sha256": sha256(environment_path),
        "confirmation_evaluation": protocol["confirmation_evaluation"],
        "fixed_training": protocol["fixed_training"],
        "shared_training": protocol["shared_training"],
        "primary_confirmation_rule": protocol["primary_confirmation_rule"],
    }
    write_json(RUN_ROOT / "execution_lock.json", lock)
    print(json.dumps(result, indent=2), flush=True)


def verify_execution_lock(protocol: dict[str, Any]) -> dict[str, Any]:
    path = RUN_ROOT / "execution_lock.json"
    if not path.is_file():
        raise FileNotFoundError("Run IDEA-056 smoke before evaluation")
    lock = read_json(path)
    if lock["status"] != "locked_after_inner_only_smoke":
        raise RuntimeError("IDEA-056 execution lock is incomplete")
    checks = {
        PLAN_PATH: lock["plan_sha256"],
        PROTOCOL_PATH: lock["protocol_sha256"],
        Path(__file__): lock["runner_sha256"],
        ROLES_PATH: lock["roles_sha256"],
        RUN_ROOT / "smoke" / "summary.json": lock["smoke_summary_sha256"],
        RUN_ROOT / "environment_lock.json": lock["environment_lock_sha256"],
    }
    for source, expected in checks.items():
        if sha256(source) != expected:
            raise RuntimeError(f"IDEA-056 execution-lock mismatch: {source}")
    if lock["confirmation_evaluation"] != protocol["confirmation_evaluation"]:
        raise RuntimeError("IDEA-056 evaluation matrix changed after smoke")
    return lock


def paired_row(
    call_frame: pd.DataFrame,
    animal_frame: pd.DataFrame,
    base_seed: int,
    repeat: int,
) -> dict[str, Any]:
    left = call_frame.sort_values("cat_id").reset_index(drop=True)
    right = animal_frame.sort_values("cat_id").reset_index(drop=True)
    if not np.array_equal(left["cat_id"].to_numpy(), right["cat_id"].to_numpy()):
        raise RuntimeError("IDEA-056 paired cat order differs")
    left_metrics = evaluation.animal_metrics(left)
    right_metrics = evaluation.animal_metrics(right)
    left_correct = left["predicted_label"].to_numpy() == left["true_label"].to_numpy()
    right_correct = right["predicted_label"].to_numpy() == right["true_label"].to_numpy()
    changed = left["predicted_label"].to_numpy() != right["predicted_label"].to_numpy()
    return {
        "base_seed": int(base_seed),
        "repeat": int(repeat),
        "C0_macro_f1": left_metrics["macro_f1"],
        "C1_macro_f1": right_metrics["macro_f1"],
        "macro_f1_difference": right_metrics["macro_f1"] - left_metrics["macro_f1"],
        "balanced_accuracy_difference": right_metrics["balanced_accuracy"]
        - left_metrics["balanced_accuracy"],
        "qwk_difference": right_metrics["quadratic_weighted_kappa"]
        - left_metrics["quadratic_weighted_kappa"],
        "plain_accuracy_difference": right_metrics["plain_accuracy"]
        - left_metrics["plain_accuracy"],
        "animal_cross_entropy_difference": evaluation.animal_cross_entropy(right)
        - evaluation.animal_cross_entropy(left),
        "changed_animals": int(changed.sum()),
        "gained_correct_animals": int((~left_correct & right_correct).sum()),
        "lost_correct_animals": int((left_correct & ~right_correct).sum()),
    }


def paired_cat_bootstrap(
    animals_by_key: dict[tuple[str, int, int], pd.DataFrame],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    settings = protocol["confirmation_evaluation"]
    pairs = [
        (int(base_seed), int(repeat))
        for base_seed in settings["base_seeds"]
        for repeat in settings["repeats"]
    ]
    reference_frame = animals_by_key[(PIPELINES[0], *pairs[0])].sort_values("cat_id")
    labels = reference_frame["true_label"].to_numpy(dtype=np.int64)
    predictions = []
    for base_seed, repeat in pairs:
        left = animals_by_key[(PIPELINES[0], base_seed, repeat)].sort_values("cat_id")
        right = animals_by_key[(PIPELINES[1], base_seed, repeat)].sort_values("cat_id")
        if not np.array_equal(left["cat_id"].to_numpy(), right["cat_id"].to_numpy()):
            raise RuntimeError("IDEA-056 bootstrap cat order differs")
        predictions.append(
            (
                left["predicted_label"].to_numpy(dtype=np.int64),
                right["predicted_label"].to_numpy(dtype=np.int64),
            )
        )
    bootstrap = protocol["bootstrap"]
    generator = np.random.default_rng(int(bootstrap["seed"]))
    differences = []
    for _ in range(int(bootstrap["iterations"])):
        sampled = generator.integers(0, len(labels), size=len(labels))
        pair_differences = []
        for left, right in predictions:
            left_score = f1_score(
                labels[sampled],
                left[sampled],
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
            right_score = f1_score(
                labels[sampled],
                right[sampled],
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
            pair_differences.append(float(right_score - left_score))
        differences.append(float(np.mean(pair_differences)))
    values = np.asarray(differences, dtype=float)
    return {
        "left": PIPELINES[0],
        "right": PIPELINES[1],
        "resampling_unit": "cat_id",
        "paired_complete_oof_comparisons": int(len(pairs)),
        "iterations": int(bootstrap["iterations"]),
        "seed": int(bootstrap["seed"]),
        "mean_difference": float(values.mean()),
        "interval": [
            float(np.quantile(values, float(bootstrap["interval"][0]))),
            float(np.quantile(values, float(bootstrap["interval"][1]))),
        ],
        "fraction_above_zero": float(np.mean(values > 0.0)),
    }


def aggregate_results(protocol: dict[str, Any]) -> dict[str, Any]:
    root = RUN_ROOT / "evaluation"
    settings = protocol["confirmation_evaluation"]
    metrics_by_pipeline = {pipeline: [] for pipeline in PIPELINES}
    animals_by_key: dict[tuple[str, int, int], pd.DataFrame] = {}
    fit_summaries = []
    for base_seed in settings["base_seeds"]:
        for repeat in settings["repeats"]:
            for pipeline in PIPELINES:
                frames = []
                for fold in settings["outer_folds"]:
                    fit_root = (
                        root
                        / "fits"
                        / f"base_seed_{base_seed}"
                        / f"repeat_{repeat}"
                        / f"fold_{fold}"
                    )
                    frames.append(
                        pd.read_csv(
                            fit_root / f"outer_test_animals__{pipeline}.csv",
                            dtype={"cat_id": str},
                        )
                    )
                    if pipeline == PIPELINES[0]:
                        fit_summaries.append(read_json(fit_root / "fit_summary.json"))
                animals = pd.concat(frames, ignore_index=True).sort_values("cat_id")
                if len(animals) != 111 or animals["cat_id"].nunique() != 111:
                    raise RuntimeError("IDEA-056 complete OOF must contain 111 cats")
                animals = animals.reset_index(drop=True)
                key = (pipeline, int(base_seed), int(repeat))
                animals_by_key[key] = animals
                metrics = evaluation.animal_metrics(animals)
                metrics["animal_cross_entropy"] = evaluation.animal_cross_entropy(
                    animals
                )
                metrics["base_seed"] = int(base_seed)
                metrics["repeat"] = int(repeat)
                metrics_by_pipeline[pipeline].append(metrics)
                output = (
                    root
                    / "complete_oof"
                    / f"{pipeline}_base_seed_{base_seed}_repeat_{repeat}_animals.csv"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                animals.to_csv(output, index=False)
    aggregate = {
        pipeline: evaluation.aggregate_metrics(rows)
        for pipeline, rows in metrics_by_pipeline.items()
    }
    for pipeline, rows in metrics_by_pipeline.items():
        ce_values = np.asarray(
            [row["animal_cross_entropy"] for row in rows], dtype=float
        )
        aggregate[pipeline]["animal_cross_entropy_mean"] = float(ce_values.mean())
        aggregate[pipeline]["animal_cross_entropy_sample_sd"] = float(
            ce_values.std(ddof=1)
        )
    paired = []
    changes = []
    for base_seed in settings["base_seeds"]:
        for repeat in settings["repeats"]:
            left = animals_by_key[(PIPELINES[0], int(base_seed), int(repeat))]
            right = animals_by_key[(PIPELINES[1], int(base_seed), int(repeat))]
            paired.append(paired_row(left, right, int(base_seed), int(repeat)))
            merged = left.merge(
                right,
                on=["cat_id", "true_label", "call_count"],
                suffixes=("_C0", "_C1"),
            )
            merged.insert(0, "base_seed", int(base_seed))
            merged.insert(1, "repeat", int(repeat))
            changes.append(merged)
    pd.concat(changes, ignore_index=True).to_csv(
        root / "paired_prediction_changes.csv", index=False
    )
    differences = [row["macro_f1_difference"] for row in paired]
    paired_summary = {
        "macro_f1_differences": differences,
        "mean_macro_f1_difference": float(np.mean(differences)),
        "positive_pairs": int(sum(value > 0.0 for value in differences)),
        "comparison_count": int(len(differences)),
        "mean_balanced_accuracy_difference": float(
            np.mean([row["balanced_accuracy_difference"] for row in paired])
        ),
        "mean_qwk_difference": float(
            np.mean([row["qwk_difference"] for row in paired])
        ),
        "mean_plain_accuracy_difference": float(
            np.mean([row["plain_accuracy_difference"] for row in paired])
        ),
        "mean_animal_cross_entropy_difference": float(
            np.mean([row["animal_cross_entropy_difference"] for row in paired])
        ),
        "changed_animals": int(sum(row["changed_animals"] for row in paired)),
        "gained_correct_animals": int(
            sum(row["gained_correct_animals"] for row in paired)
        ),
        "lost_correct_animals": int(
            sum(row["lost_correct_animals"] for row in paired)
        ),
    }
    per_seed = {}
    for base_seed in settings["base_seeds"]:
        rows = [row for row in paired if row["base_seed"] == int(base_seed)]
        per_seed[str(base_seed)] = {
            "C0_macro_f1_mean": float(np.mean([row["C0_macro_f1"] for row in rows])),
            "C1_macro_f1_mean": float(np.mean([row["C1_macro_f1"] for row in rows])),
            "C1_minus_C0_mean": float(
                np.mean([row["macro_f1_difference"] for row in rows])
            ),
            "positive_pairs": int(
                sum(row["macro_f1_difference"] > 0.0 for row in rows)
            ),
        }
    bootstrap = paired_cat_bootstrap(animals_by_key, protocol)
    support = []
    for label, key in (
        ("balanced_accuracy", "mean_balanced_accuracy_difference"),
        ("QWK", "mean_qwk_difference"),
        ("plain_accuracy", "mean_plain_accuracy_difference"),
    ):
        if paired_summary[key] > 0.0:
            support.append({"metric": label, "difference": paired_summary[key]})
    c0_values = [row["macro_f1"] for row in metrics_by_pipeline[PIPELINES[0]]]
    c1_values = [row["macro_f1"] for row in metrics_by_pipeline[PIPELINES[1]]]
    if min(c1_values) > min(c0_values):
        support.append(
            {
                "metric": "worst_repeat_macro_f1",
                "difference": float(min(c1_values) - min(c0_values)),
            }
        )
    if aggregate[PIPELINES[1]]["macro_f1_sample_sd"] <= aggregate[PIPELINES[0]][
        "macro_f1_sample_sd"
    ]:
        support.append(
            {
                "metric": "macro_f1_sample_sd_reduction",
                "difference": float(
                    aggregate[PIPELINES[0]]["macro_f1_sample_sd"]
                    - aggregate[PIPELINES[1]]["macro_f1_sample_sd"]
                ),
            }
        )
    if bootstrap["mean_difference"] > 0.0:
        support.append(
            {
                "metric": "paired_bootstrap_center",
                "difference": bootstrap["mean_difference"],
            }
        )
    rule = protocol["primary_confirmation_rule"]
    mean_gain = paired_summary["mean_macro_f1_difference"]
    positive_pairs = paired_summary["positive_pairs"]
    passed = bool(
        mean_gain >= float(rule["minimum_mean_macro_f1_gain"])
        and positive_pairs >= int(rule["minimum_positive_pairs"])
        and bool(support)
    )
    if passed:
        category = "confirmed"
        next_reference = "animal_level_CE_checkpoint_selection"
    elif mean_gain > 0.0:
        category = "mixed_positive_below_confirmation_threshold"
        next_reference = "team_decision_required"
    else:
        category = "nonpositive_confirmation_result"
        next_reference = "call_level_CE_checkpoint_selection"
    selection_summary = {
        "selected_call_epochs": [
            int(fit["inner"]["selected_call_epoch"]) for fit in fit_summaries
        ],
        "selected_animal_epochs": [
            int(fit["inner"]["selected_animal_epoch"]) for fit in fit_summaries
        ],
        "epoch_differences": [
            int(fit["inner"]["selected_epoch_difference"])
            for fit in fit_summaries
        ],
        "same_epoch_folds": int(
            sum(
                fit["inner"]["selected_call_epoch"]
                == fit["inner"]["selected_animal_epoch"]
                for fit in fit_summaries
            )
        ),
    }
    inventory = evaluation.raw_prediction_inventory(root)
    inventory_path = root / "raw_prediction_inventory.json"
    write_json(inventory_path, inventory)
    return {
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "completed_shared_training_trajectories": int(len(fit_summaries)),
        "pipeline_fold_predictions": int(settings["pipeline_fold_predictions"]),
        "complete_oof": metrics_by_pipeline,
        "aggregate": aggregate,
        "paired": paired,
        "paired_summary": paired_summary,
        "per_seed_summary": per_seed,
        "paired_cat_bootstrap": bootstrap,
        "selection_summary": selection_summary,
        "primary_confirmation": {
            "passed": passed,
            "category": category,
            "mean_gain_threshold": float(rule["minimum_mean_macro_f1_gain"]),
            "positive_pair_threshold": int(rule["minimum_positive_pairs"]),
            "supporting_signals": support,
            "next_reference": next_reference,
            "seed17_excluded": True,
        },
        "historical_seed17": protocol["historical_seed17"],
        "raw_prediction_inventory": {
            "path": repo_relative(inventory_path),
            "sha256": sha256(inventory_path),
            "files": inventory["files"],
            "bytes": inventory["bytes"],
            "aggregate_sha256": inventory["aggregate_sha256"],
        },
    }


def evaluate(
    protocol: dict[str, Any],
    store: Any,
    roles: pd.DataFrame,
    device: torch.device,
    resume: bool,
) -> None:
    lock = verify_execution_lock(protocol)
    settings = protocol["confirmation_evaluation"]
    root = RUN_ROOT / "evaluation"
    run_manifest = root / "run_manifest.json"
    write_json(
        run_manifest,
        {
            "status": "running",
            "stage": "idea056_confirmation_evaluation",
            "outer_test_accessed": True,
            "code_commit": lock["code_commit"],
            "protocol_sha256": lock["protocol_sha256"],
            "runner_sha256": lock["runner_sha256"],
            "device": str(device),
        },
    )
    completed = 0
    for base_seed in settings["base_seeds"]:
        for repeat in settings["repeats"]:
            for fold in settings["outer_folds"]:
                output = (
                    root
                    / "fits"
                    / f"base_seed_{base_seed}"
                    / f"repeat_{repeat}"
                    / f"fold_{fold}"
                )
                fit_path = output / "fit_summary.json"
                required = [
                    output / f"outer_test_animals__{pipeline}.csv"
                    for pipeline in PIPELINES
                ]
                if resume and fit_path.is_file() and all(
                    path.is_file() for path in required
                ):
                    completed += 1
                    continue
                indices = reference.historical.fold_indices(
                    store, roles, int(repeat), int(fold), include_test=True
                )
                seed = reference.historical.full_seed(
                    int(base_seed), int(repeat), int(fold)
                )
                print(
                    f"IDEA056 base_seed={base_seed} repeat={repeat} "
                    f"fold={fold} full_seed={seed}",
                    flush=True,
                )
                inner = fit_inner_trajectory(
                    protocol,
                    store,
                    indices["train"],
                    indices["validation"],
                    device,
                    seed,
                )
                outer_train = np.concatenate(
                    (indices["train"], indices["validation"])
                )
                outputs, outer = fit_outer_trajectory(
                    protocol,
                    store,
                    outer_train,
                    indices["test"],
                    int(inner["selected_call_epoch"]),
                    int(inner["selected_animal_epoch"]),
                    device,
                    seed,
                )
                output.mkdir(parents=True, exist_ok=True)
                for pipeline in PIPELINES:
                    outputs[pipeline].to_csv(
                        output / f"outer_test_animals__{pipeline}.csv", index=False
                    )
                    outputs[f"calls::{pipeline}"].to_csv(
                        output / f"outer_test_calls__{pipeline}.csv", index=False
                    )
                fit = {
                    "status": "complete",
                    "stage": "idea056_confirmation_evaluation",
                    "outer_test_accessed": True,
                    "base_seed": int(base_seed),
                    "repeat": int(repeat),
                    "outer_fold": int(fold),
                    "full_seed": int(seed),
                    "inner_train_calls": int(len(indices["train"])),
                    "inner_validation_calls": int(len(indices["validation"])),
                    "outer_test_calls": int(len(indices["test"])),
                    "inner": inner,
                    "outer": outer,
                }
                write_json(fit_path, fit)
                completed += 1
                print(
                    "IDEA056 "
                    + " ".join(
                        f"{pipeline}={outer['test_metrics'][pipeline]['macro_f1']:.4f}"
                        for pipeline in PIPELINES
                    ),
                    flush=True,
                )
    summary = aggregate_results(protocol)
    summary.update(
        {
            "code_commit": lock["code_commit"],
            "execution_lock_sha256": sha256(RUN_ROOT / "execution_lock.json"),
            "environment_lock_sha256": sha256(RUN_ROOT / "environment_lock.json"),
            "runner_sha256": sha256(Path(__file__)),
            "protocol_sha256": sha256(PROTOCOL_PATH),
        }
    )
    summary_path = root / "summary.json"
    write_json(summary_path, summary)
    write_json(
        root / "run_summary.json",
        {
            "status": "complete",
            "completed_shared_training_trajectories": completed,
            "expected_shared_training_trajectories": int(
                settings["shared_training_trajectories"]
            ),
            "pipeline_fold_predictions": int(settings["pipeline_fold_predictions"]),
            "complete_oof_evaluations": int(settings["complete_oof_evaluations"]),
            "summary_path": repo_relative(summary_path),
            "summary_sha256": sha256(summary_path),
            "raw_prediction_inventory_sha256": summary[
                "raw_prediction_inventory"
            ]["sha256"],
            "raw_prediction_aggregate_sha256": summary[
                "raw_prediction_inventory"
            ]["aggregate_sha256"],
        },
    )
    write_json(
        run_manifest,
        {
            "status": "complete",
            "stage": "idea056_confirmation_evaluation",
            "outer_test_accessed": True,
            "code_commit": lock["code_commit"],
            "protocol_sha256": lock["protocol_sha256"],
            "runner_sha256": lock["runner_sha256"],
            "device": str(device),
            "completed_shared_training_trajectories": completed,
            "pipeline_fold_predictions": int(settings["pipeline_fold_predictions"]),
            "summary_path": repo_relative(summary_path),
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    device = resolve_device(args.device)
    store, roles = load_inputs()
    print(f"IDEA-056 stage={args.stage}; device={device}", flush=True)
    if args.stage == "smoke":
        smoke(protocol, store, roles, device, args.resume)
    else:
        evaluate(protocol, store, roles, device, args.resume)


if __name__ == "__main__":
    main()
