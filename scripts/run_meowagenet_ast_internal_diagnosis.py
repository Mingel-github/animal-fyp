"""Run the inner-only AST internal diagnosis before selecting one new module."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
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
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import ASTModel


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import extract_ast_embeddings as ast_extraction  # noqa: E402
import extract_ast_temporal_tokens as temporal_extraction  # noqa: E402
import run_ast_finetuning as ast_base  # noqa: E402
import run_meowagenet_ast_accuracy_enhancement_v1 as split_utils  # noqa: E402
import run_meowagenet_ast_cat_balance_global_weighting_v1 as weighting  # noqa: E402
import run_meowagenet_idea051_cat_set as evaluation  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_ast_internal_diagnosis_v1.json"
)
PLAN_PATH = REPO_ROOT / "plan" / "AST_internal_diagnosis_and_single_module_plan.md"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
RUN_ROOT = REPO_ROOT / "runs" / "meowagenet_ast_internal_diagnosis_v1"
FEATURE_PATH = RUN_ROOT / "features" / "ast_internal_diagnostic_features.npz"
FEATURE_SUMMARY_PATH = RUN_ROOT / "features" / "summary.json"
LOCK_PATH = RUN_ROOT / "execution_lock.json"
REGION_NAMES = (
    "time_early",
    "time_middle",
    "time_late",
    "frequency_low",
    "frequency_middle",
    "frequency_high",
)
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
LABEL_NAMES = ("kitten", "adult", "senior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "smoke", "diagnose"), required=True
    )
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


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def git_blob_sha256(revision: str, path: Path) -> str:
    content = subprocess.check_output(
        ["git", "show", f"{revision}:{repo_relative(path)}"], cwd=REPO_ROOT
    )
    return hashlib.sha256(content).hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def dependency_paths(protocol: dict[str, Any]) -> dict[str, Path]:
    dependencies = protocol["dependencies"]
    return {
        "locked_protocol": REPO_ROOT / dependencies["locked_protocol_path"],
        "fbank": REPO_ROOT / dependencies["fbank_path"],
        "final_embedding": REPO_ROOT / dependencies["final_embedding_path"],
        "layer_embedding": REPO_ROOT / dependencies["layer_embedding_path"],
        "ast_runner": REPO_ROOT / dependencies["ast_runner_path"],
        "global_weighting_runner": REPO_ROOT
        / dependencies["global_weighting_runner_path"],
        "evaluation_runner": REPO_ROOT / dependencies["evaluation_runner_path"],
    }


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol["protocol_id"] != "meowagenet-ast-internal-diagnosis-v1":
        raise RuntimeError("Unexpected AST internal-diagnosis protocol")
    if protocol["stage_boundary"]["outer_test_accessed"] is not False:
        raise RuntimeError("The diagnostic protocol must keep outer test inaccessible")
    if tuple(protocol["feature_preparation"]["regional_representations"]) != REGION_NAMES:
        raise RuntimeError("Regional representation order changed")
    splits = protocol["splits"]
    if int(splits["diagnostic_split_count"]) != len(splits["repeats"]) * len(
        splits["outer_folds"]
    ):
        raise RuntimeError("Diagnostic split budget is inconsistent")
    if tuple(splits["roles_used"]) != ("train", "validation"):
        raise RuntimeError("Diagnostic roles changed")
    fixed = protocol["fixed_training"]
    if int(fixed["micro_batch_size"]) * int(
        fixed["gradient_accumulation_steps"]
    ) != int(fixed["accumulation_window_calls"]):
        raise RuntimeError("Gradient-accumulation window is inconsistent")
    checks = {
        PLAN_PATH: protocol["plan"]["sha256"],
        ROLES_PATH: splits["roles_sha256"],
    }
    dependencies = protocol["dependencies"]
    for key, path in dependency_paths(protocol).items():
        checks[path] = dependencies[f"{key}_sha256"]
    for path, expected in checks.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Diagnostic dependency checksum mismatch: {path}")


def relative_groups(values: np.ndarray, group_count: int = 3) -> list[np.ndarray]:
    values = np.asarray(values, dtype=np.int64)
    if not len(values):
        raise ValueError("Cannot divide an empty position sequence")
    groups = [part.astype(np.int64) for part in np.array_split(values, group_count)]
    targets = np.linspace(0, len(values) - 1, group_count)
    for index, group in enumerate(groups):
        if not len(group):
            groups[index] = np.asarray(
                [values[int(round(float(targets[index])))]], dtype=np.int64
            )
    return groups


def mean_replace_region(
    segment: np.ndarray, valid_frames: int, region_index: int
) -> np.ndarray:
    output = segment.copy()
    valid = output[:valid_frames]
    if region_index < 3:
        frame_group = relative_groups(np.arange(valid_frames))[region_index]
        replacement = valid.mean(axis=0, keepdims=True)
        output[frame_group, :] = replacement
    else:
        frequency_group = relative_groups(np.arange(output.shape[1]))[region_index - 3]
        replacement = valid.mean(axis=1, keepdims=True)
        valid[:, frequency_group] = replacement
        output[:valid_frames] = valid
    return output


def prepare_features(
    protocol: dict[str, Any], device: torch.device, resume: bool
) -> None:
    if FEATURE_PATH.is_file() and FEATURE_SUMMARY_PATH.is_file():
        if not resume:
            raise FileExistsError(FEATURE_PATH)
        print(FEATURE_SUMMARY_PATH.read_text(encoding="utf-8"), flush=True)
        return
    paths = dependency_paths(protocol)
    locked = read_json(paths["locked_protocol"])
    loaded = np.load(paths["fbank"])
    frozen = np.load(paths["final_embedding"])
    features = loaded["features"].astype(np.float32)
    segment_call_indices = loaded["segment_call_indices"].astype(np.int64)
    segment_counts = loaded["segment_counts"].astype(np.int64)
    call_ids = loaded["call_ids"].astype(str)
    if not np.array_equal(call_ids, frozen["call_ids"].astype(str)):
        raise RuntimeError("Fbank and final-embedding call order differs")
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
    valid_time_mask, real_overlap = temporal_extraction.temporal_patch_mask(
        frame_counts,
        int(time_positions),
        patch_size,
        int(standard["time_stride"]),
        0.5,
    )
    call_count = len(call_ids)
    hidden_size = int(model.config.hidden_size)
    reconstructed = np.zeros((call_count, hidden_size), dtype=np.float64)
    regional_sums = np.zeros(
        (call_count, len(REGION_NAMES), hidden_size), dtype=np.float64
    )
    regional_counts = np.zeros((call_count, len(REGION_NAMES)), dtype=np.int64)
    batch_size = 32
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            batch = torch.from_numpy(features[start:stop]).to(device)
            output = model(input_values=batch)
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
                time_groups = relative_groups(valid_times)
                frequency_groups = relative_groups(
                    np.arange(int(frequency_positions), dtype=np.int64)
                )
                for region_index, time_group in enumerate(time_groups):
                    value = patches[local_segment, :, time_group, :].mean(axis=(0, 1))
                    regional_sums[call_index, region_index] += value
                    regional_counts[call_index, region_index] += 1
                for local_frequency, frequency_group in enumerate(frequency_groups):
                    value = patches[
                        local_segment,
                        frequency_group[:, None],
                        valid_times[None, :],
                        :,
                    ].mean(axis=(0, 1))
                    region_index = local_frequency + 3
                    regional_sums[call_index, region_index] += value
                    regional_counts[call_index, region_index] += 1
    reconstructed = (reconstructed / segment_counts[:, None]).astype(np.float32)
    if np.any(regional_counts == 0):
        raise RuntimeError("At least one call has an empty regional representation")
    regional = (regional_sums / regional_counts[:, :, None]).astype(np.float32)

    masked = np.zeros(
        (call_count, len(REGION_NAMES), hidden_size), dtype=np.float32
    )
    for region_index, region_name in enumerate(REGION_NAMES):
        call_sums = np.zeros((call_count, hidden_size), dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(features), batch_size):
                stop = min(start + batch_size, len(features))
                replaced = np.stack(
                    [
                        mean_replace_region(
                            features[segment_index],
                            int(frame_counts[segment_index]),
                            region_index,
                        )
                        for segment_index in range(start, stop)
                    ]
                )
                pooled = (
                    model(input_values=torch.from_numpy(replaced).to(device))
                    .pooler_output.float()
                    .cpu()
                    .numpy()
                )
                np.add.at(call_sums, segment_call_indices[start:stop], pooled)
        masked[:, region_index] = (
            call_sums / segment_counts[:, None]
        ).astype(np.float32)
        print(f"Prepared masked region {region_name}", flush=True)

    valid_fraction = np.zeros(call_count, dtype=np.float64)
    mean_abs_fbank = np.zeros(call_count, dtype=np.float64)
    fbank_std = np.zeros(call_count, dtype=np.float64)
    for call_index in range(call_count):
        segment_indices = np.flatnonzero(segment_call_indices == call_index)
        fractions = []
        mean_abs_values = []
        standard_deviations = []
        for segment_index in segment_indices:
            count = int(frame_counts[segment_index])
            real = features[segment_index, :count]
            fractions.append(count / features.shape[1])
            mean_abs_values.append(float(np.abs(real).mean()))
            standard_deviations.append(float(real.std()))
        valid_fraction[call_index] = float(np.mean(fractions))
        mean_abs_fbank[call_index] = float(np.mean(mean_abs_values))
        fbank_std[call_index] = float(np.mean(standard_deviations))
    maximum_difference = float(
        np.abs(reconstructed - frozen["embeddings"].astype(np.float32)).max()
    )
    if maximum_difference > 5.0e-4:
        raise RuntimeError("Reconstructed AST embedding differs from locked embedding")
    if not np.isfinite(regional).all() or not np.isfinite(masked).all():
        raise RuntimeError("Prepared diagnostic embeddings contain non-finite values")
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FEATURE_PATH,
        regional_embeddings=regional,
        masked_embeddings=masked,
        region_names=np.asarray(REGION_NAMES),
        call_ids=call_ids,
        cat_ids=loaded["cat_ids"].astype(str),
        labels=loaded["labels"].astype(np.int8),
        durations=loaded["durations"].astype(np.float32),
        segment_counts=segment_counts.astype(np.int16),
        valid_frame_fraction=valid_fraction.astype(np.float32),
        mean_abs_fbank=mean_abs_fbank.astype(np.float32),
        fbank_std=fbank_std.astype(np.float32),
    )
    summary = {
        "status": "complete",
        "stage": "feature_preparation",
        "outer_test_accessed": False,
        "protocol_id": protocol["protocol_id"],
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "calls": call_count,
        "cats": int(len(np.unique(loaded["cat_ids"].astype(str)))),
        "segments": int(len(features)),
        "region_names": list(REGION_NAMES),
        "regional_shape": list(regional.shape),
        "masked_shape": list(masked.shape),
        "geometry": geometry,
        "valid_time_token_range": [
            int(valid_time_mask.sum(axis=1).min()),
            int(valid_time_mask.sum(axis=1).max()),
        ],
        "real_overlap_shape": list(real_overlap.shape),
        "reconstructed_final_embedding_max_abs_difference": maximum_difference,
        "inference_seconds": float(time.perf_counter() - started),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "feature_path": repo_relative(FEATURE_PATH),
        "feature_sha256": sha256(FEATURE_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
    }
    write_json(FEATURE_SUMMARY_PATH, summary)
    print(json.dumps(summary, indent=2), flush=True)


def load_diagnostic_inputs(
    protocol: dict[str, Any],
) -> tuple[Any, pd.DataFrame, np.lib.npyio.NpzFile, np.lib.npyio.NpzFile]:
    if not FEATURE_PATH.is_file() or not FEATURE_SUMMARY_PATH.is_file():
        raise FileNotFoundError("Run --stage prepare before diagnostic stages")
    store = ast_base.load_feature_store()
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    layers = np.load(dependency_paths(protocol)["layer_embedding"])
    diagnostics = np.load(FEATURE_PATH)
    for loaded in (layers, diagnostics):
        if not np.array_equal(store.call_ids, loaded["call_ids"].astype(str)):
            raise RuntimeError("Diagnostic feature call order differs")
        if not np.array_equal(store.labels, loaded["labels"].astype(np.int64)):
            raise RuntimeError("Diagnostic feature labels differ")
    if layers["embeddings"].shape != (792, 12, 768):
        raise RuntimeError("Unexpected layer embedding shape")
    if diagnostics["regional_embeddings"].shape != (792, 6, 768):
        raise RuntimeError("Unexpected regional embedding shape")
    if diagnostics["masked_embeddings"].shape != (792, 6, 768):
        raise RuntimeError("Unexpected masked embedding shape")
    if roles["cat_id"].nunique() != 111:
        raise RuntimeError("Diagnostic roles must cover 111 cats")
    return store, roles, layers, diagnostics


def training_sample_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("Probe training role is missing an age class")
    return (len(labels) / (3.0 * counts[labels])).astype(np.float64)


def fit_probe_model(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    seed: int,
    protocol: dict[str, Any],
) -> tuple[StandardScaler, LogisticRegression]:
    settings = protocol["shared_probe"]
    scaler = StandardScaler()
    train_values = scaler.fit_transform(embeddings[train_indices])
    classifier = LogisticRegression(
        C=float(settings["C"]),
        solver=str(settings["solver"]),
        max_iter=int(settings["maximum_iterations"]),
        random_state=int(seed),
    )
    classifier.fit(
        train_values,
        labels[train_indices],
        sample_weight=training_sample_weights(labels[train_indices]),
    )
    return scaler, classifier


def predict_probe(
    embeddings: np.ndarray,
    store: Any,
    indices: np.ndarray,
    scaler: StandardScaler,
    classifier: LogisticRegression,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    probabilities = classifier.predict_proba(scaler.transform(embeddings[indices]))
    if probabilities.shape != (len(indices), 3):
        raise RuntimeError("Probe probability shape changed")
    calls = pd.DataFrame(
        {
            "call_index": indices.astype(np.int64),
            "call_id": store.call_ids[indices],
            "cat_id": store.cat_ids[indices],
            "true_label": store.labels[indices],
            **{
                column: probabilities[:, class_index]
                for class_index, column in enumerate(PROBABILITY_COLUMNS)
            },
        }
    )
    animals = evaluation.calls_to_animals(calls)
    metrics = evaluation.animal_metrics(animals)
    metrics["animal_cross_entropy"] = evaluation.animal_cross_entropy(animals)
    predictions = probabilities.argmax(axis=1)
    metrics["call_macro_f1"] = float(
        f1_score(
            store.labels[indices],
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )
    metrics["call_accuracy"] = float(
        accuracy_score(store.labels[indices], predictions)
    )
    metrics["probability_sum_max_error"] = float(
        np.abs(probabilities.sum(axis=1) - 1.0).max()
    )
    metrics["probe_iterations"] = [int(value) for value in classifier.n_iter_]
    return metrics, calls, animals


def probe_representation(
    embeddings: np.ndarray,
    store: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    seed: int,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, StandardScaler, LogisticRegression]:
    scaler, classifier = fit_probe_model(
        embeddings, store.labels, train_indices, seed, protocol
    )
    metrics, calls, animals = predict_probe(
        embeddings, store, validation_indices, scaler, classifier
    )
    return metrics, calls, animals, scaler, classifier


def animal_prediction_rows(animals: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "cat_id": str(row.cat_id),
            "true_label": int(row.true_label),
            "predicted_label": int(row.predicted_label),
        }
        for row in animals.itertuples(index=False)
    ]


def safe_spearman(first: np.ndarray, second: np.ndarray) -> dict[str, float | None]:
    result = spearmanr(np.asarray(first, dtype=float), np.asarray(second, dtype=float))
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    return {
        "rho": rho if np.isfinite(rho) else None,
        "pvalue": pvalue if np.isfinite(pvalue) else None,
    }


def nuisance_probe_results(
    embeddings: np.ndarray,
    diagnostics: np.lib.npyio.NpzFile,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> dict[str, Any]:
    scaler = StandardScaler()
    train_values = scaler.fit_transform(embeddings[train_indices])
    validation_values = scaler.transform(embeddings[validation_indices])
    targets = {
        "log_duration": np.log1p(diagnostics["durations"].astype(float)),
        "valid_frame_fraction": diagnostics["valid_frame_fraction"].astype(float),
        "mean_abs_fbank": diagnostics["mean_abs_fbank"].astype(float),
        "fbank_std": diagnostics["fbank_std"].astype(float),
    }
    output = {}
    for name, target in targets.items():
        model = Ridge(alpha=10.0)
        model.fit(train_values, target[train_indices])
        predicted = model.predict(validation_values)
        output[name] = {
            "validation_r2": float(
                r2_score(target[validation_indices], predicted)
            ),
            "prediction_vs_observed_spearman": safe_spearman(
                predicted, target[validation_indices]
            ),
        }
    return output


def domain_train_loader(
    store: Any, indices: np.ndarray, mode: str, batch_size: int, seed: int
) -> DataLoader:
    dataset = ast_base.CallDataset(store, indices, mode)
    sampler = weighting.DeterministicNoSingletonBatchSampler(
        len(indices), batch_size, seed
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=ast_base.collate_calls,
    )


def fit_domain_trajectory(
    mode: str,
    protocol: dict[str, Any],
    store: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
    maximum_epochs_override: int | None = None,
) -> dict[str, Any]:
    if mode not in ("frozen", "last2"):
        raise ValueError(mode)
    split_utils.set_seed(seed)
    locked = read_json(dependency_paths(protocol)["locked_protocol"])
    model = ast_base.build_model(mode, locked, train_indices, store).to(device)
    counts = ast_base.trainable_counts(model)
    fixed = protocol["fixed_training"]
    optimizer = ast_base.make_optimizer(
        model,
        mode,
        float(protocol["diagnostic_axes"]["pretraining_domain"]["encoder_learning_rate"]),
        float(fixed["head_learning_rate"]),
    )
    weights_numpy = weighting.global_class_balanced_call_weights(
        store.labels, train_indices
    )
    weights = torch.from_numpy(weights_numpy).to(device)
    train_loader = domain_train_loader(
        store, train_indices, mode, int(fixed["micro_batch_size"]), seed
    )
    validation_loader = ast_base.build_loader(
        store,
        validation_indices,
        mode,
        int(fixed["evaluation_batch_size"]),
        False,
        seed,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    maximum_epochs = int(maximum_epochs_override or fixed["maximum_epochs"])
    patience = int(fixed["early_stopping_patience"])
    best_loss = float("inf")
    best_epoch = 1
    best_metrics: dict[str, Any] | None = None
    without_improvement = 0
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        processed: list[int] = []
        for step, cpu_batch in enumerate(train_loader):
            cpu_indices = cpu_batch["call_indices"].numpy().astype(np.int64)
            processed.extend(cpu_indices.tolist())
            batch = ast_base.move_batch(cpu_batch, device)
            batch_weights = weights[batch["call_indices"]]
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
                    per_call_loss,
                    batch_weights,
                    int(fixed["accumulation_window_calls"]),
                )
            scaler.scale(loss).backward()
            should_step = (
                (step + 1) % int(fixed["gradient_accumulation_steps"]) == 0
                or step + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(fixed["gradient_clip"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            weighted_loss_sum += float(
                (per_call_loss.detach() * batch_weights).sum()
            )
            weight_sum += float(batch_weights.sum())
        if len(processed) != len(train_indices) or len(set(processed)) != len(
            train_indices
        ):
            raise RuntimeError("Domain trajectory did not process each train call once")
        call_loss, calls = ast_base.predict_calls(
            model, validation_loader, store, device
        )
        animals = evaluation.calls_to_animals(calls)
        animal_loss = evaluation.animal_cross_entropy(animals)
        metrics = evaluation.animal_metrics(animals)
        row = {
            "epoch": int(epoch),
            "train_weighted_loss": float(weighted_loss_sum / weight_sum),
            "validation_call_cross_entropy": float(call_loss),
            "validation_animal_cross_entropy": float(animal_loss),
            "validation_animal_macro_f1": metrics["macro_f1"],
            "validation_animal_balanced_accuracy": metrics["balanced_accuracy"],
            "validation_animal_qwk": metrics["quadratic_weighted_kappa"],
            "validation_plain_accuracy": metrics["plain_accuracy"],
        }
        history.append(row)
        if animal_loss < best_loss - 1.0e-12:
            best_loss = animal_loss
            best_epoch = epoch
            best_metrics = metrics
            without_improvement = 0
        else:
            without_improvement += 1
        print(
            f"diagnostic domain {mode} epoch={epoch} train={row['train_weighted_loss']:.4f} "
            f"animal_CE={animal_loss:.4f} animal_F1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if maximum_epochs_override is None and without_improvement >= patience:
            break
    if best_metrics is None:
        raise RuntimeError("Domain trajectory produced no selected checkpoint")
    result = {
        "mode": mode,
        "selected_epoch": int(best_epoch),
        "stopped_epoch": int(len(history)),
        "best_validation_animal_cross_entropy": float(best_loss),
        "selected_validation_metrics": best_metrics,
        "history": history,
        "parameters": counts,
        "train_seconds": float(time.perf_counter() - started),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "loss_implementation": "fixed_global_class_weights_and_denominator_32",
        "outer_test_accessed": False,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def mask_sensitivity(
    baseline_calls: pd.DataFrame,
    masked_calls: pd.DataFrame,
    baseline_metrics: dict[str, Any],
    masked_metrics: dict[str, Any],
    diagnostics: np.lib.npyio.NpzFile,
) -> dict[str, Any]:
    left = baseline_calls.sort_values("call_index").reset_index(drop=True)
    right = masked_calls.sort_values("call_index").reset_index(drop=True)
    if not np.array_equal(left["call_index"].to_numpy(), right["call_index"].to_numpy()):
        raise RuntimeError("Masked and baseline call order differs")
    labels = left["true_label"].to_numpy(dtype=np.int64)
    left_probabilities = left[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    right_probabilities = right[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    drop = left_probabilities[np.arange(len(labels)), labels] - right_probabilities[
        np.arange(len(labels)), labels
    ]
    indices = left["call_index"].to_numpy(dtype=np.int64)
    return {
        "animal_macro_f1_drop": float(
            baseline_metrics["macro_f1"] - masked_metrics["macro_f1"]
        ),
        "animal_balanced_accuracy_drop": float(
            baseline_metrics["balanced_accuracy"]
            - masked_metrics["balanced_accuracy"]
        ),
        "mean_true_class_probability_drop": float(drop.mean()),
        "median_true_class_probability_drop": float(np.median(drop)),
        "positive_call_fraction": float(np.mean(drop > 0.0)),
        "associations": {
            "duration": safe_spearman(drop, diagnostics["durations"][indices]),
            "valid_frame_fraction": safe_spearman(
                drop, diagnostics["valid_frame_fraction"][indices]
            ),
            "mean_abs_fbank": safe_spearman(
                drop, diagnostics["mean_abs_fbank"][indices]
            ),
            "fbank_std": safe_spearman(drop, diagnostics["fbank_std"][indices]),
        },
    }


def run_split_diagnosis(
    protocol: dict[str, Any],
    store: Any,
    roles: pd.DataFrame,
    layers: np.lib.npyio.NpzFile,
    diagnostics: np.lib.npyio.NpzFile,
    repeat: int,
    fold: int,
    device: torch.device,
    layer_indices: list[int] | None = None,
    region_indices: list[int] | None = None,
    domain_epochs_override: int | None = None,
) -> dict[str, Any]:
    indices = split_utils.fold_indices(
        store, roles, repeat, fold, include_test=False
    )
    seed = split_utils.full_seed(
        int(protocol["diagnosis"]["base_seed"]), repeat, fold
    )
    selected_layers = layer_indices or list(range(12))
    selected_regions = region_indices or list(range(len(REGION_NAMES)))
    layer_rows = []
    for layer_index in selected_layers:
        metrics, _, animals, _, _ = probe_representation(
            layers["embeddings"][:, layer_index].astype(np.float32),
            store,
            indices["train"],
            indices["validation"],
            seed,
            protocol,
        )
        layer_rows.append(
            {
                "layer": int(layer_index + 1),
                "metrics": metrics,
                "animal_predictions": animal_prediction_rows(animals),
            }
        )
    region_rows = []
    for region_index in selected_regions:
        metrics, _, animals, _, _ = probe_representation(
            diagnostics["regional_embeddings"][:, region_index].astype(np.float32),
            store,
            indices["train"],
            indices["validation"],
            seed,
            protocol,
        )
        region_rows.append(
            {
                "region": REGION_NAMES[region_index],
                "metrics": metrics,
                "animal_predictions": animal_prediction_rows(animals),
            }
        )
    global_metrics, global_calls, global_animals, scaler, classifier = probe_representation(
        store.frozen_embeddings,
        store,
        indices["train"],
        indices["validation"],
        seed,
        protocol,
    )
    mask_rows = []
    for region_index in selected_regions:
        masked_metrics, masked_calls, masked_animals = predict_probe(
            diagnostics["masked_embeddings"][:, region_index].astype(np.float32),
            store,
            indices["validation"],
            scaler,
            classifier,
        )
        mask_rows.append(
            {
                "region": REGION_NAMES[region_index],
                "metrics": masked_metrics,
                "sensitivity": mask_sensitivity(
                    global_calls,
                    masked_calls,
                    global_metrics,
                    masked_metrics,
                    diagnostics,
                ),
                "animal_predictions": animal_prediction_rows(masked_animals),
            }
        )
    domain = {
        mode: fit_domain_trajectory(
            mode,
            protocol,
            store,
            indices["train"],
            indices["validation"],
            device,
            seed,
            maximum_epochs_override=domain_epochs_override,
        )
        for mode in ("frozen", "last2")
    }
    return {
        "status": "complete",
        "outer_test_accessed": False,
        "repeat": int(repeat),
        "outer_fold": int(fold),
        "full_seed": int(seed),
        "inner_train_calls": int(len(indices["train"])),
        "inner_validation_calls": int(len(indices["validation"])),
        "inner_train_cats": int(len(np.unique(store.cat_ids[indices["train"]]))),
        "inner_validation_cats": int(
            len(np.unique(store.cat_ids[indices["validation"]]))
        ),
        "layer_probes": layer_rows,
        "regional_probes": region_rows,
        "global_probe": {
            "metrics": global_metrics,
            "animal_predictions": animal_prediction_rows(global_animals),
        },
        "masked_sensitivity": mask_rows,
        "nuisance_probes": nuisance_probe_results(
            store.frozen_embeddings,
            diagnostics,
            indices["train"],
            indices["validation"],
        ),
        "domain_adaptation": domain,
    }


def environment_record(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
    }


def create_execution_lock(protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    revision = git_revision()
    lock = {
        "schema_version": "1.0",
        "status": "locked_after_inner_only_smoke",
        "protocol_id": protocol["protocol_id"],
        "outer_test_accessed": False,
        "code_commit": revision,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "plan_sha256": sha256(PLAN_PATH),
        "roles_sha256": sha256(ROLES_PATH),
        "prepared_features_sha256": sha256(FEATURE_PATH),
        "feature_summary_sha256": sha256(FEATURE_SUMMARY_PATH),
        "environment": environment_record(device),
    }
    if git_blob_sha256(revision, PROTOCOL_PATH) != lock["protocol_sha256"]:
        raise RuntimeError("Committed diagnostic protocol differs from executed file")
    if git_blob_sha256(revision, Path(__file__)) != lock["runner_sha256"]:
        raise RuntimeError("Committed diagnostic runner differs from executed file")
    write_json(LOCK_PATH, lock)
    return lock


def verify_execution_lock(protocol: dict[str, Any]) -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise FileNotFoundError("Run inner-only smoke before full diagnosis")
    lock = read_json(LOCK_PATH)
    checks = {
        PROTOCOL_PATH: lock["protocol_sha256"],
        Path(__file__): lock["runner_sha256"],
        PLAN_PATH: lock["plan_sha256"],
        ROLES_PATH: lock["roles_sha256"],
        FEATURE_PATH: lock["prepared_features_sha256"],
        FEATURE_SUMMARY_PATH: lock["feature_summary_sha256"],
    }
    for path, expected in checks.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Execution-lock mismatch: {path}")
    if lock["protocol_id"] != protocol["protocol_id"]:
        raise RuntimeError("Execution lock belongs to another protocol")
    return lock


def run_smoke(
    protocol: dict[str, Any], device: torch.device, resume: bool
) -> None:
    output = RUN_ROOT / "smoke" / "summary.json"
    if output.is_file():
        if not resume:
            raise FileExistsError(output)
        print(output.read_text(encoding="utf-8"), flush=True)
        return
    store, roles, layers, diagnostics = load_diagnostic_inputs(protocol)
    settings = protocol["smoke"]
    result = run_split_diagnosis(
        protocol,
        store,
        roles,
        layers,
        diagnostics,
        int(settings["repeat"]),
        int(settings["outer_fold"]),
        device,
        layer_indices=[int(value) - 1 for value in settings["layer_indices"]],
        region_indices=[int(value) for value in settings["region_indices"]],
        domain_epochs_override=int(settings["domain_epochs"]),
    )
    probability_errors = [
        row["metrics"]["probability_sum_max_error"]
        for key in ("layer_probes", "regional_probes", "masked_sensitivity")
        for row in result[key]
    ] + [result["global_probe"]["metrics"]["probability_sum_max_error"]]
    domain_histories = result["domain_adaptation"]
    passed = bool(
        result["outer_test_accessed"] is False
        and result["inner_train_cats"] > 0
        and result["inner_validation_cats"] > 0
        and max(probability_errors) <= 1.0e-6
        and all(
            len(domain_histories[mode]["history"]) == int(settings["domain_epochs"])
            for mode in ("frozen", "last2")
        )
        and domain_histories["frozen"]["parameters"]["trainable"] == 99075
        and domain_histories["last2"]["parameters"]["trainable"] == 14276355
    )
    summary = {
        "status": "passed" if passed else "failed",
        "stage": "inner_only_smoke",
        "outer_test_accessed": False,
        "split": {
            "repeat": result["repeat"],
            "outer_fold": result["outer_fold"],
            "inner_train_calls": result["inner_train_calls"],
            "inner_validation_calls": result["inner_validation_calls"],
            "inner_train_cats": result["inner_train_cats"],
            "inner_validation_cats": result["inner_validation_cats"],
        },
        "layer_probe_count": len(result["layer_probes"]),
        "regional_probe_count": len(result["regional_probes"]),
        "masked_evaluation_count": len(result["masked_sensitivity"]),
        "maximum_probability_sum_error": max(probability_errors),
        "domain_parameters": {
            mode: domain_histories[mode]["parameters"]
            for mode in ("frozen", "last2")
        },
        "domain_epochs": {
            mode: len(domain_histories[mode]["history"])
            for mode in ("frozen", "last2")
        },
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "prepared_features_sha256": sha256(FEATURE_PATH),
    }
    write_json(output, summary)
    if not passed:
        raise RuntimeError("AST internal-diagnosis smoke failed")
    lock = create_execution_lock(protocol, device)
    print(json.dumps({"smoke": summary, "execution_lock": lock}, indent=2), flush=True)


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in (
            "macro_f1",
            "balanced_accuracy",
            "quadratic_weighted_kappa",
            "plain_accuracy",
            "animal_cross_entropy",
        )
    }


def prediction_complementarity(
    candidate_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]
) -> dict[str, int]:
    candidate = {
        (int(row["repeat"]), int(row["outer_fold"])): row
        for row in candidate_rows
    }
    reference = {
        (int(row["repeat"]), int(row["outer_fold"])): row
        for row in reference_rows
    }
    totals = {
        "both_correct": 0,
        "reference_only_correct": 0,
        "candidate_only_correct": 0,
        "both_wrong": 0,
    }
    for key, candidate_row in candidate.items():
        reference_row = reference[key]
        left = {
            row["cat_id"]: row for row in reference_row["animal_predictions"]
        }
        right = {
            row["cat_id"]: row for row in candidate_row["animal_predictions"]
        }
        if set(left) != set(right):
            raise RuntimeError("Complementarity cat sets differ")
        for cat_id in left:
            truth = int(left[cat_id]["true_label"])
            left_correct = int(left[cat_id]["predicted_label"]) == truth
            right_correct = int(right[cat_id]["predicted_label"]) == truth
            if left_correct and right_correct:
                totals["both_correct"] += 1
            elif left_correct:
                totals["reference_only_correct"] += 1
            elif right_correct:
                totals["candidate_only_correct"] += 1
            else:
                totals["both_wrong"] += 1
    return totals


def aggregate_diagnosis(
    protocol: dict[str, Any], split_results: list[dict[str, Any]], lock: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    layer_flat = []
    region_flat = []
    global_flat = []
    mask_flat = []
    domain_flat = []
    nuisance_flat = []
    for split in split_results:
        context = {"repeat": split["repeat"], "outer_fold": split["outer_fold"]}
        global_flat.append(
            {
                **context,
                "metrics": split["global_probe"]["metrics"],
                "animal_predictions": split["global_probe"]["animal_predictions"],
            }
        )
        for row in split["layer_probes"]:
            layer_flat.append({**context, **row})
        for row in split["regional_probes"]:
            region_flat.append({**context, **row})
        for row in split["masked_sensitivity"]:
            mask_flat.append({**context, **row})
        for mode, row in split["domain_adaptation"].items():
            domain_flat.append({**context, "mode": mode, **row})
        nuisance_flat.append({**context, **split["nuisance_probes"]})

    layer_aggregate = []
    for layer in range(1, 13):
        rows = [row for row in layer_flat if row["layer"] == layer]
        layer_aggregate.append(
            {
                "layer": layer,
                **mean_metrics([row["metrics"] for row in rows]),
                "macro_f1_sample_sd": float(
                    np.std([row["metrics"]["macro_f1"] for row in rows], ddof=1)
                ),
            }
        )
    reference_layer = next(row for row in layer_aggregate if row["layer"] == 12)
    best_intermediate = max(
        [row for row in layer_aggregate if row["layer"] < 12],
        key=lambda row: (row["macro_f1"], row["layer"]),
    )
    paired_layer_rows = []
    for split in split_results:
        candidate = next(
            row
            for row in split["layer_probes"]
            if row["layer"] == best_intermediate["layer"]
        )
        reference = next(
            row for row in split["layer_probes"] if row["layer"] == 12
        )
        paired_layer_rows.append(
            {
                "repeat": split["repeat"],
                "outer_fold": split["outer_fold"],
                "macro_f1_difference": float(
                    candidate["metrics"]["macro_f1"]
                    - reference["metrics"]["macro_f1"]
                ),
            }
        )
    layer_rule = protocol["diagnostic_axes"]["intermediate_layer"]["support_rule"]
    layer_gain = float(
        best_intermediate["macro_f1"] - reference_layer["macro_f1"]
    )
    layer_positive = int(
        sum(row["macro_f1_difference"] > 0.0 for row in paired_layer_rows)
    )
    layer_candidate_rows = [
        row for row in layer_flat if row["layer"] == best_intermediate["layer"]
    ]
    layer_reference_rows = [row for row in layer_flat if row["layer"] == 12]
    layer_summary = {
        "aggregate_by_layer": layer_aggregate,
        "reference_layer": 12,
        "best_intermediate_layer": int(best_intermediate["layer"]),
        "mean_macro_f1_gain": layer_gain,
        "positive_splits": layer_positive,
        "paired": paired_layer_rows,
        "complementarity": prediction_complementarity(
            layer_candidate_rows, layer_reference_rows
        ),
        "support_rule_passed": bool(
            layer_gain
            >= float(layer_rule["minimum_mean_macro_f1_gain_over_layer12"])
            and layer_positive >= int(layer_rule["minimum_positive_splits"])
        ),
    }

    global_aggregate = mean_metrics([row["metrics"] for row in global_flat])
    region_aggregate = []
    for region in REGION_NAMES:
        rows = [row for row in region_flat if row["region"] == region]
        region_aggregate.append(
            {
                "region": region,
                **mean_metrics([row["metrics"] for row in rows]),
                "macro_f1_sample_sd": float(
                    np.std([row["metrics"]["macro_f1"] for row in rows], ddof=1)
                ),
            }
        )
    best_region = max(
        region_aggregate, key=lambda row: (row["macro_f1"], row["region"])
    )
    region_candidate_rows = [
        row for row in region_flat if row["region"] == best_region["region"]
    ]
    region_complementarity = prediction_complementarity(
        region_candidate_rows, global_flat
    )
    mask_aggregate = []
    for region in REGION_NAMES:
        rows = [row for row in mask_flat if row["region"] == region]
        mask_aggregate.append(
            {
                "region": region,
                "mean_animal_macro_f1_drop": float(
                    np.mean(
                        [row["sensitivity"]["animal_macro_f1_drop"] for row in rows]
                    )
                ),
                "mean_true_class_probability_drop": float(
                    np.mean(
                        [
                            row["sensitivity"]["mean_true_class_probability_drop"]
                            for row in rows
                        ]
                    )
                ),
                "positive_true_probability_splits": int(
                    sum(
                        row["sensitivity"]["mean_true_class_probability_drop"] > 0.0
                        for row in rows
                    )
                ),
                "mean_association_rho": {
                    nuisance: float(
                        np.mean(
                            [
                                row["sensitivity"]["associations"][nuisance]["rho"]
                                for row in rows
                                if row["sensitivity"]["associations"][nuisance]["rho"]
                                is not None
                            ]
                        )
                    )
                    for nuisance in (
                        "duration",
                        "valid_frame_fraction",
                        "mean_abs_fbank",
                        "fbank_std",
                    )
                },
            }
        )
    strongest_mask = max(
        mask_aggregate,
        key=lambda row: (
            row["mean_true_class_probability_drop"],
            row["mean_animal_macro_f1_drop"],
        ),
    )
    local_rule = protocol["diagnostic_axes"]["local_patch"]["support_rule"]
    region_gap = float(best_region["macro_f1"] - global_aggregate["macro_f1"])
    local_summary = {
        "global_probe": global_aggregate,
        "regional_probes": region_aggregate,
        "best_region": best_region["region"],
        "best_region_macro_f1_gap_vs_global": region_gap,
        "best_region_complementarity": region_complementarity,
        "mask_sensitivity": mask_aggregate,
        "strongest_mask_region": strongest_mask["region"],
        "support_rule_passed": bool(
            strongest_mask["mean_true_class_probability_drop"]
            >= float(local_rule["minimum_mask_true_class_probability_drop"])
            and strongest_mask["positive_true_probability_splits"]
            >= int(local_rule["minimum_positive_mask_splits"])
            and region_gap
            >= -float(local_rule["maximum_region_macro_f1_gap_below_global"])
            and region_complementarity["candidate_only_correct"]
            >= int(local_rule["minimum_candidate_only_correct_events"])
        ),
    }

    domain_by_mode = {}
    for mode in ("frozen", "last2"):
        rows = [row for row in domain_flat if row["mode"] == mode]
        metrics = [
            {
                **row["selected_validation_metrics"],
                "animal_cross_entropy": row["best_validation_animal_cross_entropy"],
            }
            for row in rows
        ]
        domain_by_mode[mode] = {
            **mean_metrics(metrics),
            "macro_f1_sample_sd": float(
                np.std([metric["macro_f1"] for metric in metrics], ddof=1)
            ),
            "mean_selected_epoch": float(
                np.mean([row["selected_epoch"] for row in rows])
            ),
            "trainable_parameters": int(rows[0]["parameters"]["trainable"]),
            "summed_train_seconds": float(
                sum(row["train_seconds"] for row in rows)
            ),
            "peak_vram_bytes": int(max(row["peak_vram_bytes"] for row in rows)),
        }
    domain_pairs = []
    for repeat in protocol["splits"]["repeats"]:
        for fold in protocol["splits"]["outer_folds"]:
            frozen = next(
                row
                for row in domain_flat
                if row["mode"] == "frozen"
                and row["repeat"] == repeat
                and row["outer_fold"] == fold
            )
            last2 = next(
                row
                for row in domain_flat
                if row["mode"] == "last2"
                and row["repeat"] == repeat
                and row["outer_fold"] == fold
            )
            domain_pairs.append(
                {
                    "repeat": int(repeat),
                    "outer_fold": int(fold),
                    "macro_f1_difference": float(
                        last2["selected_validation_metrics"]["macro_f1"]
                        - frozen["selected_validation_metrics"]["macro_f1"]
                    ),
                    "animal_cross_entropy_difference": float(
                        last2["best_validation_animal_cross_entropy"]
                        - frozen["best_validation_animal_cross_entropy"]
                    ),
                }
            )
    domain_gain = float(
        np.mean([row["macro_f1_difference"] for row in domain_pairs])
    )
    domain_ce = float(
        np.mean([row["animal_cross_entropy_difference"] for row in domain_pairs])
    )
    domain_positive = int(
        sum(row["macro_f1_difference"] > 0.0 for row in domain_pairs)
    )
    domain_rule = protocol["diagnostic_axes"]["pretraining_domain"]["support_rule"]
    nuisance_aggregate = {
        target: {
            "mean_validation_r2": float(
                np.mean([row[target]["validation_r2"] for row in nuisance_flat])
            ),
            "mean_prediction_vs_observed_spearman_rho": float(
                np.mean(
                    [
                        row[target]["prediction_vs_observed_spearman"]["rho"]
                        for row in nuisance_flat
                        if row[target]["prediction_vs_observed_spearman"]["rho"]
                        is not None
                    ]
                )
            ),
        }
        for target in (
            "log_duration",
            "valid_frame_fraction",
            "mean_abs_fbank",
            "fbank_std",
        )
    }
    domain_summary = {
        "aggregate_by_mode": domain_by_mode,
        "paired": domain_pairs,
        "last2_minus_frozen_mean_macro_f1": domain_gain,
        "last2_minus_frozen_mean_animal_cross_entropy": domain_ce,
        "positive_splits": domain_positive,
        "nuisance_linear_probe": nuisance_aggregate,
        "support_rule_passed": bool(
            domain_gain >= float(domain_rule["minimum_mean_macro_f1_gain"])
            and domain_positive >= int(domain_rule["minimum_positive_splits"])
            and domain_ce < 0.0
        ),
    }

    axis_support = {
        "local_patch": local_summary["support_rule_passed"],
        "intermediate_layer": layer_summary["support_rule_passed"],
        "pretraining_domain": domain_summary["support_rule_passed"],
    }
    supported = [axis for axis, value in axis_support.items() if value]
    decision = {
        "status": "awaiting_team_selection",
        "outer_test_accessed": False,
        "supported_axes": supported,
        "support_rule_audit": axis_support,
        "provisional_readout": (
            f"one supported axis: {supported[0]}"
            if len(supported) == 1
            else (
                "multiple supported axes require mechanism and feasibility review"
                if supported
                else "no axis passed its complete diagnostic support rule"
            )
        ),
        "required_human_record": "select one direction and state why the other two are deferred",
        "next_core_comparison": [
            "R0_original_AST_reference",
            "M1_single_diagnostic_driven_module",
            "C1_parameter_matched_control",
        ],
    }
    summary = {
        "status": "complete",
        "stage": "inner_only_ast_internal_diagnosis",
        "protocol_id": protocol["protocol_id"],
        "outer_test_accessed": False,
        "diagnostic_splits": len(split_results),
        "inner_validation_cat_evaluations": int(
            sum(split["inner_validation_cats"] for split in split_results)
        ),
        "axes": {
            "intermediate_layer": layer_summary,
            "local_patch": local_summary,
            "pretraining_domain": domain_summary,
        },
        "decision_record": decision,
        "code_commit": lock["code_commit"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "prepared_features_sha256": sha256(FEATURE_PATH),
        "execution_lock_sha256": sha256(LOCK_PATH),
    }
    return summary, decision


def run_diagnosis(
    protocol: dict[str, Any], device: torch.device, resume: bool
) -> None:
    lock = verify_execution_lock(protocol)
    store, roles, layers, diagnostics = load_diagnostic_inputs(protocol)
    root = RUN_ROOT / "diagnosis"
    manifest_path = root / "run_manifest.json"
    write_json(
        manifest_path,
        {
            "status": "running",
            "stage": "inner_only_ast_internal_diagnosis",
            "outer_test_accessed": False,
            "code_commit": lock["code_commit"],
            "protocol_sha256": lock["protocol_sha256"],
            "runner_sha256": lock["runner_sha256"],
            "device": str(device),
        },
    )
    split_results = []
    for repeat in protocol["diagnosis"]["repeats"]:
        for fold in protocol["diagnosis"]["outer_folds"]:
            output = root / "splits" / f"repeat_{repeat}" / f"fold_{fold}.json"
            if resume and output.is_file():
                result = read_json(output)
                if result.get("status") == "complete":
                    split_results.append(result)
                    continue
            print(f"AST diagnosis repeat={repeat} fold={fold}", flush=True)
            result = run_split_diagnosis(
                protocol,
                store,
                roles,
                layers,
                diagnostics,
                int(repeat),
                int(fold),
                device,
            )
            write_json(output, result)
            split_results.append(result)
    summary, decision = aggregate_diagnosis(protocol, split_results, lock)
    summary_path = root / "summary.json"
    decision_path = root / "decision_record_draft.json"
    write_json(summary_path, summary)
    write_json(decision_path, decision)
    write_json(
        root / "run_summary.json",
        {
            "status": "complete",
            "outer_test_accessed": False,
            "diagnostic_splits": len(split_results),
            "layer_probe_fits": int(protocol["diagnosis"]["layer_probe_fits"]),
            "regional_probe_fits": int(protocol["diagnosis"]["regional_probe_fits"]),
            "masked_evaluations": int(protocol["diagnosis"]["masked_evaluations"]),
            "domain_training_trajectories": int(
                protocol["diagnosis"]["domain_training_trajectories"]
            ),
            "summary_path": repo_relative(summary_path),
            "summary_sha256": sha256(summary_path),
            "decision_record_path": repo_relative(decision_path),
            "decision_record_sha256": sha256(decision_path),
        },
    )
    write_json(
        manifest_path,
        {
            "status": "complete",
            "stage": "inner_only_ast_internal_diagnosis",
            "outer_test_accessed": False,
            "code_commit": lock["code_commit"],
            "protocol_sha256": lock["protocol_sha256"],
            "runner_sha256": lock["runner_sha256"],
            "device": str(device),
            "diagnostic_splits": len(split_results),
            "summary_path": repo_relative(summary_path),
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    device = resolve_device(args.device)
    print(f"AST internal diagnosis stage={args.stage}; device={device}", flush=True)
    if args.stage == "prepare":
        prepare_features(protocol, device, args.resume)
    elif args.stage == "smoke":
        run_smoke(protocol, device, args.resume)
    else:
        run_diagnosis(protocol, device, args.resume)


if __name__ == "__main__":
    main()
