"""Run IDEA-055 checkpoint prediction averaging and class-bias calibration."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_ast_cat_balance_global_weighting_v1 as reference  # noqa: E402
import run_meowagenet_idea051_cat_set as evaluation  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea055_checkpoint_calibration_v1.json"
)
PLAN_PATH = (
    REPO_ROOT
    / "plan"
    / "IDEA-055_checkpoint_ensemble_and_class_bias_calibration.md"
)
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
RUN_ROOT = REPO_ROOT / "runs" / "meowagenet_idea055_checkpoint_calibration_v1"
PIPELINES = (
    "A0_single_selected_checkpoint",
    "P1_tail3_checkpoint_ensemble",
    "P2_class_bias_calibration",
    "P3_tail3_plus_class_bias",
)
CANDIDATES = PIPELINES[1:]
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
LABEL_NAMES = ("kitten", "adult", "senior")


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
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol["protocol_id"] != "meowagenet-idea055-checkpoint-calibration-v1":
        raise RuntimeError("Unexpected IDEA-055 protocol")
    settings = protocol["initial_evaluation"]
    if tuple(settings["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-055 pipeline matrix changed")
    expected_trajectories = (
        len(settings["base_seeds"])
        * len(settings["repeats"])
        * len(settings["outer_folds"])
    )
    if expected_trajectories != int(settings["shared_training_trajectories"]):
        raise RuntimeError("IDEA-055 trajectory budget is inconsistent")
    if len(PIPELINES) * expected_trajectories != int(
        settings["pipeline_fold_predictions"]
    ):
        raise RuntimeError("IDEA-055 prediction budget is inconsistent")
    fixed = protocol["fixed_training"]
    if int(fixed["micro_batch_size"]) * int(
        fixed["gradient_accumulation_steps"]
    ) != int(fixed["accumulation_window_calls"]):
        raise RuntimeError("IDEA-055 accumulation window is inconsistent")
    bias = protocol["class_bias_calibration"]
    if len(bias["kitten_bias_grid"]) * len(bias["senior_bias_grid"]) != int(
        bias["grid_points"]
    ):
        raise RuntimeError("IDEA-055 bias-grid size is inconsistent")
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
        REPO_ROOT / dependencies["frozen_embedding_path"]: dependencies[
            "frozen_embedding_sha256"
        ],
    }
    for path, expected in checks.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"IDEA-055 dependency checksum mismatch: {path}")


def load_inputs() -> tuple[Any, pd.DataFrame]:
    store = reference.ast_base.load_feature_store()
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    if len(store.call_ids) != 792:
        raise RuntimeError("IDEA-055 expects 792 calls")
    if len(np.unique(store.cat_ids)) != 111 or roles["cat_id"].nunique() != 111:
        raise RuntimeError("IDEA-055 expects 111 cats")
    return store, roles


def trainable_parameter_audit(model: torch.nn.Module) -> dict[str, Any]:
    trainable = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
        }
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "model_total_parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(row["numel"] for row in trainable)),
        "trainable_parameter_tensors": int(len(trainable)),
        "trainable": trainable,
    }


def prediction_loader(
    store: Any,
    indices: np.ndarray,
    protocol: dict[str, Any],
    seed: int,
) -> Any:
    return reference.historical.idea019.build_loader(
        store,
        indices,
        "frozen",
        int(protocol["fixed_training"]["evaluation_batch_size"]),
        False,
        seed,
    )


def predict_calls(
    model: torch.nn.Module,
    loader: Any,
    store: Any,
    device: torch.device,
) -> pd.DataFrame:
    _, frame = reference.historical.idea019.predict_calls(
        model, loader, store, device
    )
    return frame.sort_values("call_index").reset_index(drop=True)


def average_probability_frames(
    frames: list[pd.DataFrame], key: str
) -> pd.DataFrame:
    if not frames:
        raise ValueError("At least one probability frame is required")
    aligned = [frame.sort_values(key).reset_index(drop=True) for frame in frames]
    keys = aligned[0][key].to_numpy()
    for frame in aligned[1:]:
        if not np.array_equal(keys, frame[key].to_numpy()):
            raise RuntimeError("Checkpoint probability-frame order differs")
    result = aligned[0].copy()
    probabilities = np.mean(
        [frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float) for frame in aligned],
        axis=0,
    )
    result.loc[:, list(PROBABILITY_COLUMNS)] = probabilities
    result["predicted_label"] = probabilities.argmax(axis=1)
    return result


def apply_class_bias(animals: pd.DataFrame, bias: list[float]) -> pd.DataFrame:
    if len(bias) != 3 or not np.isclose(float(bias[1]), 0.0):
        raise ValueError("IDEA-055 requires three biases with adult fixed to zero")
    result = animals.copy()
    if np.allclose(np.asarray(bias, dtype=float), 0.0, atol=0.0):
        return result
    probabilities = np.clip(
        result[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float), 1.0e-12, 1.0
    )
    logits = np.log(probabilities) + np.asarray(bias, dtype=float)[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    result.loc[:, list(PROBABILITY_COLUMNS)] = calibrated
    result["predicted_label"] = calibrated.argmax(axis=1)
    return result


def select_class_bias(
    animals: pd.DataFrame, protocol: dict[str, Any]
) -> tuple[list[float], list[dict[str, Any]]]:
    settings = protocol["class_bias_calibration"]
    rows = []
    best_key: tuple[Any, ...] | None = None
    best_bias: list[float] | None = None
    for kitten in settings["kitten_bias_grid"]:
        for senior in settings["senior_bias_grid"]:
            bias = [float(kitten), 0.0, float(senior)]
            calibrated = apply_class_bias(animals, bias)
            metrics = evaluation.animal_metrics(calibrated)
            cross_entropy = evaluation.animal_cross_entropy(calibrated)
            row = {
                "bias": bias,
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"],
                "plain_accuracy": metrics["plain_accuracy"],
                "animal_cross_entropy": cross_entropy,
                "l1_bias": abs(float(kitten)) + abs(float(senior)),
            }
            rows.append(row)
            key = (
                -row["macro_f1"],
                -row["balanced_accuracy"],
                row["animal_cross_entropy"],
                row["l1_bias"],
                float(kitten),
                float(senior),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_bias = bias
    if best_bias is None:
        raise RuntimeError("IDEA-055 bias selection produced no candidate")
    return best_bias, rows


def tail_epochs(selected_epoch: int, maximum_checkpoints: int) -> list[int]:
    first = max(1, int(selected_epoch) - int(maximum_checkpoints) + 1)
    return list(range(first, int(selected_epoch) + 1))


def probability_sum_error(frame: pd.DataFrame) -> float:
    sums = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float).sum(axis=1)
    return float(np.abs(sums - 1.0).max())


def fit_inner_trajectory(
    protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
    max_epochs_override: int | None = None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    reference.historical.set_seed(seed)
    values = reference.training_values(protocol)
    model = reference.build_model(protocol, store, train_indices).to(device)
    parameters = trainable_parameter_audit(model)
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
    validation_loader = prediction_loader(store, validation_indices, protocol, seed)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    maximum_epochs = max_epochs_override or values["max_epochs"]
    best_loss = float("inf")
    best_epoch = 1
    without_improvement = 0
    history = []
    call_frames: dict[int, pd.DataFrame] = {}
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, maximum_epochs + 1):
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
        calls = predict_calls(model, validation_loader, store, device)
        call_frames[epoch] = calls
        animals = evaluation.calls_to_animals(calls)
        validation_loss = evaluation.animal_cross_entropy(animals)
        metrics = evaluation.animal_metrics(animals)
        history.append(
            {
                "epoch": int(epoch),
                "train_weighted_loss": train_loss,
                "validation_animal_cross_entropy": validation_loss,
                "validation_animal_macro_f1": metrics["macro_f1"],
                "validation_animal_balanced_accuracy": metrics[
                    "balanced_accuracy"
                ],
                "validation_animal_qwk": metrics["quadratic_weighted_kappa"],
                "train_unit_audit": train_audit,
            }
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            without_improvement = 0
        else:
            without_improvement += 1
        print(
            f"IDEA055 inner epoch={epoch} train={train_loss:.4f} "
            f"animal_val={validation_loss:.4f} val_F1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if max_epochs_override is None and without_improvement >= values["patience"]:
            break
    selected_calls = call_frames[best_epoch]
    selected_animals = evaluation.calls_to_animals(selected_calls)
    epochs = tail_epochs(
        best_epoch, int(protocol["checkpoint_ensemble"]["maximum_checkpoints"])
    )
    ensemble_calls = average_probability_frames(
        [call_frames[epoch] for epoch in epochs], "call_index"
    )
    ensemble_animals = evaluation.calls_to_animals(ensemble_calls)
    single_bias, single_curve = select_class_bias(selected_animals, protocol)
    ensemble_bias, ensemble_curve = select_class_bias(ensemble_animals, protocol)
    updates = {
        name: float((parameter.detach().cpu() - initial[name]).abs().max())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    outputs = {
        "single_animals": selected_animals,
        "ensemble_animals": ensemble_animals,
        "single_calibrated_animals": apply_class_bias(selected_animals, single_bias),
        "ensemble_calibrated_animals": apply_class_bias(
            ensemble_animals, ensemble_bias
        ),
    }
    audit = {
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(len(history)),
        "best_validation_animal_cross_entropy": float(best_loss),
        "history": history,
        "tail_epochs": epochs,
        "tail_checkpoint_count": int(len(epochs)),
        "single_bias": single_bias,
        "ensemble_bias": ensemble_bias,
        "single_bias_curve": single_curve,
        "ensemble_bias_curve": ensemble_curve,
        "validation_metrics": {
            PIPELINES[0]: evaluation.animal_metrics(outputs["single_animals"]),
            PIPELINES[1]: evaluation.animal_metrics(outputs["ensemble_animals"]),
            PIPELINES[2]: evaluation.animal_metrics(
                outputs["single_calibrated_animals"]
            ),
            PIPELINES[3]: evaluation.animal_metrics(
                outputs["ensemble_calibrated_animals"]
            ),
        },
        "validation_animal_cross_entropy": {
            PIPELINES[0]: evaluation.animal_cross_entropy(outputs["single_animals"]),
            PIPELINES[1]: evaluation.animal_cross_entropy(
                outputs["ensemble_animals"]
            ),
            PIPELINES[2]: evaluation.animal_cross_entropy(
                outputs["single_calibrated_animals"]
            ),
            PIPELINES[3]: evaluation.animal_cross_entropy(
                outputs["ensemble_calibrated_animals"]
            ),
        },
        "probability_sum_max_error": {
            key: probability_sum_error(frame) for key, frame in outputs.items()
        },
        "target_weight_audit": reference.target_weight_audit(
            weights_numpy, store, train_indices
        ),
        "parameters": parameters,
        "updated_trainable_parameter_tensors": int(
            sum(value > 0.0 for value in updates.values())
        ),
        "maximum_trainable_parameter_updates": updates,
        "train_seconds": time.perf_counter() - started,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return audit, outputs


def fit_outer_trajectory(
    protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    selected_epoch: int,
    single_bias: list[float],
    ensemble_bias: list[float],
    device: torch.device,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[int, pd.DataFrame], dict[str, Any]]:
    reference.historical.set_seed(seed)
    values = reference.training_values(protocol)
    model = reference.build_model(protocol, store, train_indices).to(device)
    parameters = trainable_parameter_audit(model)
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
    test_loader = prediction_loader(store, test_indices, protocol, seed)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    epochs = tail_epochs(
        selected_epoch, int(protocol["checkpoint_ensemble"]["maximum_checkpoints"])
    )
    history = []
    checkpoint_calls: dict[int, pd.DataFrame] = {}
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(selected_epoch) + 1):
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
                "train_weighted_loss": train_loss,
                "train_unit_audit": train_audit,
            }
        )
        if epoch in epochs:
            checkpoint_calls[epoch] = predict_calls(model, test_loader, store, device)
        print(
            f"IDEA055 outer epoch={epoch}/{selected_epoch} train={train_loss:.4f}",
            flush=True,
        )
    single_calls = checkpoint_calls[int(selected_epoch)]
    ensemble_calls = average_probability_frames(
        [checkpoint_calls[epoch] for epoch in epochs], "call_index"
    )
    single_animals = evaluation.calls_to_animals(single_calls)
    ensemble_animals = evaluation.calls_to_animals(ensemble_calls)
    animals = {
        PIPELINES[0]: single_animals,
        PIPELINES[1]: ensemble_animals,
        PIPELINES[2]: apply_class_bias(single_animals, single_bias),
        PIPELINES[3]: apply_class_bias(ensemble_animals, ensemble_bias),
    }
    audit = {
        "epochs": int(selected_epoch),
        "history": history,
        "tail_epochs": epochs,
        "tail_checkpoint_count": int(len(epochs)),
        "single_bias": single_bias,
        "ensemble_bias": ensemble_bias,
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
        "train_and_predict_seconds": time.perf_counter() - started,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    calls = {
        PIPELINES[0]: single_calls,
        PIPELINES[1]: ensemble_calls,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {**animals, **{f"calls::{key}": value for key, value in calls.items()}}, checkpoint_calls, audit


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
    inner, outputs = fit_inner_trajectory(
        protocol,
        store,
        indices["train"],
        indices["validation"],
        device,
        seed,
        max_epochs_override=int(settings["epochs"]),
    )
    zero_single = apply_class_bias(outputs["single_animals"], [0.0, 0.0, 0.0])
    zero_ensemble = apply_class_bias(
        outputs["ensemble_animals"], [0.0, 0.0, 0.0]
    )
    zero_error = max(
        float(
            np.abs(
                zero_single[list(PROBABILITY_COLUMNS)].to_numpy()
                - outputs["single_animals"][list(PROBABILITY_COLUMNS)].to_numpy()
            ).max()
        ),
        float(
            np.abs(
                zero_ensemble[list(PROBABILITY_COLUMNS)].to_numpy()
                - outputs["ensemble_animals"][list(PROBABILITY_COLUMNS)].to_numpy()
            ).max()
        ),
    )
    status = "passed"
    if (
        inner["parameters"]["trainable_parameters"] != 99075
        or inner["updated_trainable_parameter_tensors"]
        != inner["parameters"]["trainable_parameter_tensors"]
        or len(inner["single_bias_curve"]) != 25
        or len(inner["ensemble_bias_curve"]) != 25
        or max(inner["probability_sum_max_error"].values()) > 1.0e-6
        or zero_error != 0.0
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
        "selected_epoch": inner["best_epoch"],
        "tail_epochs": inner["tail_epochs"],
        "single_bias": inner["single_bias"],
        "ensemble_bias": inner["ensemble_bias"],
        "single_bias_grid_points": len(inner["single_bias_curve"]),
        "ensemble_bias_grid_points": len(inner["ensemble_bias_curve"]),
        "zero_bias_max_probability_difference": zero_error,
        "probability_sum_max_error": inner["probability_sum_max_error"],
        "parameters": inner["parameters"],
        "updated_trainable_parameter_tensors": inner[
            "updated_trainable_parameter_tensors"
        ],
        "validation_metrics": inner["validation_metrics"],
        "inner_audit": inner,
    }
    write_json(output, result)
    if status != "passed":
        raise RuntimeError("IDEA-055 smoke failed")
    environment_path = RUN_ROOT / "environment_lock.json"
    write_json(environment_path, environment_lock(protocol, device))
    revision = git_revision()
    if revision is None:
        raise RuntimeError("IDEA-055 requires a committed source revision")
    for path in (PLAN_PATH, PROTOCOL_PATH, Path(__file__)):
        if git_blob_sha256(revision, path) != sha256(path):
            raise RuntimeError(f"IDEA-055 source differs from commit {revision}: {path}")
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
        "initial_evaluation": protocol["initial_evaluation"],
        "fixed_training": protocol["fixed_training"],
        "checkpoint_ensemble": protocol["checkpoint_ensemble"],
        "class_bias_calibration": protocol["class_bias_calibration"],
    }
    write_json(RUN_ROOT / "execution_lock.json", lock)
    print(json.dumps(result, indent=2), flush=True)


def verify_execution_lock(protocol: dict[str, Any]) -> dict[str, Any]:
    path = RUN_ROOT / "execution_lock.json"
    if not path.is_file():
        raise FileNotFoundError("Run IDEA-055 smoke before evaluation")
    lock = read_json(path)
    if lock["status"] != "locked_after_inner_only_smoke":
        raise RuntimeError("IDEA-055 execution lock is incomplete")
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
            raise RuntimeError(f"IDEA-055 execution-lock mismatch: {source}")
    if lock["initial_evaluation"] != protocol["initial_evaluation"]:
        raise RuntimeError("IDEA-055 evaluation matrix changed after smoke")
    return lock


def paired_row(
    reference_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    repeat: int,
) -> dict[str, Any]:
    left = reference_frame.sort_values("cat_id").reset_index(drop=True)
    right = candidate_frame.sort_values("cat_id").reset_index(drop=True)
    if not np.array_equal(left["cat_id"].to_numpy(), right["cat_id"].to_numpy()):
        raise RuntimeError("IDEA-055 paired cat order differs")
    left_metrics = evaluation.animal_metrics(left)
    right_metrics = evaluation.animal_metrics(right)
    left_correct = left["predicted_label"].to_numpy() == left["true_label"].to_numpy()
    right_correct = (
        right["predicted_label"].to_numpy() == right["true_label"].to_numpy()
    )
    changed = left["predicted_label"].to_numpy() != right["predicted_label"].to_numpy()
    return {
        "repeat": int(repeat),
        "reference_macro_f1": left_metrics["macro_f1"],
        "candidate_macro_f1": right_metrics["macro_f1"],
        "macro_f1_difference": right_metrics["macro_f1"] - left_metrics["macro_f1"],
        "balanced_accuracy_difference": right_metrics["balanced_accuracy"]
        - left_metrics["balanced_accuracy"],
        "qwk_difference": right_metrics["quadratic_weighted_kappa"]
        - left_metrics["quadratic_weighted_kappa"],
        "plain_accuracy_difference": right_metrics["plain_accuracy"]
        - left_metrics["plain_accuracy"],
        "changed_animals": int(changed.sum()),
        "gained_correct_animals": int((~left_correct & right_correct).sum()),
        "lost_correct_animals": int((left_correct & ~right_correct).sum()),
    }


def aggregate_results(protocol: dict[str, Any]) -> dict[str, Any]:
    root = RUN_ROOT / "evaluation"
    settings = protocol["initial_evaluation"]
    metrics_by_pipeline = {pipeline: [] for pipeline in PIPELINES}
    animals_by_key: dict[tuple[str, int], pd.DataFrame] = {}
    fit_summaries = []
    for repeat in settings["repeats"]:
        for pipeline in PIPELINES:
            frames = []
            for fold in settings["outer_folds"]:
                fit_root = (
                    root
                    / "fits"
                    / f"base_seed_{settings['base_seeds'][0]}"
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
                raise RuntimeError("IDEA-055 complete OOF must contain 111 cats")
            animals = animals.reset_index(drop=True)
            animals_by_key[(pipeline, int(repeat))] = animals
            metrics = evaluation.animal_metrics(animals)
            metrics["animal_cross_entropy"] = evaluation.animal_cross_entropy(animals)
            metrics["base_seed"] = int(settings["base_seeds"][0])
            metrics["repeat"] = int(repeat)
            metrics_by_pipeline[pipeline].append(metrics)
            output = root / "complete_oof" / f"{pipeline}_repeat_{repeat}_animals.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            animals.to_csv(output, index=False)
    aggregate = {
        pipeline: evaluation.aggregate_metrics(rows)
        for pipeline, rows in metrics_by_pipeline.items()
    }
    for pipeline, rows in metrics_by_pipeline.items():
        values = np.asarray([row["animal_cross_entropy"] for row in rows], dtype=float)
        aggregate[pipeline]["animal_cross_entropy_mean"] = float(values.mean())
        aggregate[pipeline]["animal_cross_entropy_sample_sd"] = float(
            values.std(ddof=1)
        )
    contrasts = [
        (PIPELINES[0], PIPELINES[1]),
        (PIPELINES[0], PIPELINES[2]),
        (PIPELINES[0], PIPELINES[3]),
        (PIPELINES[1], PIPELINES[3]),
        (PIPELINES[2], PIPELINES[3]),
    ]
    paired = {}
    paired_summary = {}
    bootstraps = {}
    changes = []
    gates = {}
    gate = protocol["seed_expansion_gate"]
    for left_pipeline, right_pipeline in contrasts:
        name = f"{right_pipeline}_minus_{left_pipeline}"
        rows = [
            paired_row(
                animals_by_key[(left_pipeline, int(repeat))],
                animals_by_key[(right_pipeline, int(repeat))],
                int(repeat),
            )
            for repeat in settings["repeats"]
        ]
        paired[name] = rows
        differences = [row["macro_f1_difference"] for row in rows]
        paired_summary[name] = {
            "macro_f1_differences": differences,
            "mean_macro_f1_difference": float(np.mean(differences)),
            "positive_repeats": int(sum(value > 0.0 for value in differences)),
            "mean_balanced_accuracy_difference": float(
                np.mean([row["balanced_accuracy_difference"] for row in rows])
            ),
            "mean_qwk_difference": float(
                np.mean([row["qwk_difference"] for row in rows])
            ),
            "mean_plain_accuracy_difference": float(
                np.mean([row["plain_accuracy_difference"] for row in rows])
            ),
        }
        bootstraps[name] = evaluation.paired_cat_bootstrap(
            animals_by_key, left_pipeline, right_pipeline, protocol
        )
        for repeat in settings["repeats"]:
            left = animals_by_key[(left_pipeline, int(repeat))]
            right = animals_by_key[(right_pipeline, int(repeat))]
            merged = left.merge(
                right,
                on=["cat_id", "true_label", "call_count"],
                suffixes=("_left", "_right"),
            )
            merged.insert(0, "contrast", name)
            merged.insert(1, "repeat", int(repeat))
            changes.append(merged)
        if left_pipeline == PIPELINES[0]:
            candidate = right_pipeline
            summary = paired_summary[name]
            supporting = []
            for metric, key in (
                ("balanced_accuracy", "mean_balanced_accuracy_difference"),
                ("QWK", "mean_qwk_difference"),
                ("plain_accuracy", "mean_plain_accuracy_difference"),
            ):
                if summary[key] > 0.0:
                    supporting.append({"metric": metric, "difference": summary[key]})
            reference_values = [
                row["macro_f1"] for row in metrics_by_pipeline[PIPELINES[0]]
            ]
            candidate_values = [
                row["macro_f1"] for row in metrics_by_pipeline[candidate]
            ]
            if min(candidate_values) > min(reference_values):
                supporting.append(
                    {
                        "metric": "worst_repeat_macro_f1",
                        "difference": float(
                            min(candidate_values) - min(reference_values)
                        ),
                    }
                )
            if aggregate[candidate]["macro_f1_sample_sd"] < aggregate[PIPELINES[0]][
                "macro_f1_sample_sd"
            ]:
                supporting.append(
                    {
                        "metric": "macro_f1_sample_sd_reduction",
                        "difference": float(
                            aggregate[PIPELINES[0]]["macro_f1_sample_sd"]
                            - aggregate[candidate]["macro_f1_sample_sd"]
                        ),
                    }
                )
            gain = summary["mean_macro_f1_difference"]
            positive = summary["positive_repeats"]
            gates[candidate] = {
                "passed": bool(
                    gain >= float(gate["minimum_mean_macro_f1_gain"])
                    and positive >= int(gate["minimum_positive_repeats"])
                    and bool(supporting)
                ),
                "strong_gain": bool(gain >= float(gate["strong_gain"])),
                "supporting_signals": supporting,
            }
    pd.concat(changes, ignore_index=True).to_csv(
        root / "paired_prediction_changes.csv", index=False
    )
    selection = {
        "selected_epochs": [int(fit["selected_epoch"]) for fit in fit_summaries],
        "tail_checkpoint_counts": [
            int(fit["inner"]["tail_checkpoint_count"]) for fit in fit_summaries
        ],
        "single_biases": [fit["inner"]["single_bias"] for fit in fit_summaries],
        "ensemble_biases": [
            fit["inner"]["ensemble_bias"] for fit in fit_summaries
        ],
    }
    inventory = evaluation.raw_prediction_inventory(root)
    inventory_path = root / "raw_prediction_inventory.json"
    write_json(inventory_path, inventory)
    return {
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "completed_training_trajectories": int(len(fit_summaries)),
        "pipeline_fold_predictions": int(settings["pipeline_fold_predictions"]),
        "complete_oof": metrics_by_pipeline,
        "aggregate": aggregate,
        "paired": paired,
        "paired_summary": paired_summary,
        "paired_cat_bootstrap": bootstraps,
        "selection_summary": selection,
        "seed_expansion_gates": gates,
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
    settings = protocol["initial_evaluation"]
    root = RUN_ROOT / "evaluation"
    run_manifest = root / "run_manifest.json"
    write_json(
        run_manifest,
        {
            "status": "running",
            "stage": "idea055_initial_evaluation",
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
                if resume and fit_path.is_file() and all(path.is_file() for path in required):
                    completed += 1
                    continue
                indices = reference.historical.fold_indices(
                    store, roles, int(repeat), int(fold), include_test=True
                )
                seed = reference.historical.full_seed(
                    int(base_seed), int(repeat), int(fold)
                )
                print(
                    f"IDEA055 repeat={repeat} fold={fold} seed={seed}", flush=True
                )
                inner, _ = fit_inner_trajectory(
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
                outputs, checkpoint_calls, outer = fit_outer_trajectory(
                    protocol,
                    store,
                    outer_train,
                    indices["test"],
                    int(inner["best_epoch"]),
                    inner["single_bias"],
                    inner["ensemble_bias"],
                    device,
                    seed,
                )
                output.mkdir(parents=True, exist_ok=True)
                for pipeline in PIPELINES:
                    outputs[pipeline].to_csv(
                        output / f"outer_test_animals__{pipeline}.csv", index=False
                    )
                for pipeline in PIPELINES[:2]:
                    outputs[f"calls::{pipeline}"].to_csv(
                        output / f"outer_test_calls__{pipeline}.csv", index=False
                    )
                for epoch, frame in checkpoint_calls.items():
                    frame.to_csv(
                        output / f"outer_test_checkpoint_epoch_{epoch}_calls.csv",
                        index=False,
                    )
                fit = {
                    "status": "complete",
                    "stage": "idea055_initial_evaluation",
                    "outer_test_accessed": True,
                    "base_seed": int(base_seed),
                    "repeat": int(repeat),
                    "outer_fold": int(fold),
                    "full_seed": int(seed),
                    "inner_train_calls": int(len(indices["train"])),
                    "inner_validation_calls": int(len(indices["validation"])),
                    "outer_test_calls": int(len(indices["test"])),
                    "selected_epoch": int(inner["best_epoch"]),
                    "inner": inner,
                    "outer": outer,
                }
                write_json(fit_path, fit)
                completed += 1
                print(
                    "IDEA055 "
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
            "completed_training_trajectories": completed,
            "expected_training_trajectories": int(
                settings["shared_training_trajectories"]
            ),
            "pipeline_fold_predictions": int(settings["pipeline_fold_predictions"]),
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
            "stage": "idea055_initial_evaluation",
            "outer_test_accessed": True,
            "code_commit": lock["code_commit"],
            "protocol_sha256": lock["protocol_sha256"],
            "runner_sha256": lock["runner_sha256"],
            "device": str(device),
            "completed_training_trajectories": completed,
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
    print(f"IDEA-055 stage={args.stage}; device={device}", flush=True)
    if args.stage == "smoke":
        smoke(protocol, store, roles, device, args.resume)
    else:
        evaluate(protocol, store, roles, device, args.resume)


if __name__ == "__main__":
    main()
