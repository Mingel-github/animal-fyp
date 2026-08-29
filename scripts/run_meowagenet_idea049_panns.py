"""Run the PANNs CNN14 candidate in the IDEA-049 MeowAgeNet screening protocol.

This runner owns PANNs feature extraction and candidate-specific audit logs. It
reuses the frozen-feature head training functions, formal split bank, and
animal-level evaluation already exercised by the SSAST IDEA-049 runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import time
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_functional
import torchaudio

import run_meowagenet_idea049 as engine


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT / "configs" / "protocol" / "meowagenet_idea049_backbone_screening_v1.json"
)
RECIPE_PATH = (
    REPO_ROOT / "configs" / "experiment" / "idea049" / "panns_cnn14_frozen_v1.json"
)
DATA_MANIFEST_PATH = REPO_ROOT / "metadata" / "datasets" / "meowagenet" / "data_manifest.csv"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
FORMAL_AST_OOF_ROOT = (
    REPO_ROOT / "runs" / "meowagenet_formal_v2_1_core" / "oof" / "ast_head_only"
)
RUNS_ROOT = REPO_ROOT / "runs"
CHECKPOINT_ROOT = REPO_ROOT / "data" / "models" / "panns"
ENGINE_PATH = REPO_ROOT / "scripts" / "run_meowagenet_idea049.py"
PIPELINE_ID = "panns_cnn14_audioset_frozen_mlp"
MINIMUM_AUDIO_SAMPLES = 9_920


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "smoke", "initial"), required=True)
    parser.add_argument("--output-subdir", default="idea049_panns_cnn14_v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--extraction-batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
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
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def verify_inputs(protocol: dict[str, Any], recipe: dict[str, Any]) -> None:
    expected = {
        DATA_MANIFEST_PATH: protocol["dataset"]["manifest_sha256"],
        ROLES_PATH: protocol["splits"]["roles_sha256"],
    }
    for path, digest in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != digest:
            raise RuntimeError(f"Input checksum mismatch: {path}")
    if recipe["protocol_id"] != protocol["protocol_id"]:
        raise RuntimeError("Recipe and protocol IDs differ")
    if recipe["pipeline_id"] != PIPELINE_ID:
        raise RuntimeError("Unsupported PANNs IDEA-049 pipeline")
    if sha256(ENGINE_PATH) != "419ac7815f7b24151b3d3c9536e76858cd53a7954f147cc6acc776951bcd07f0":
        raise RuntimeError("Shared frozen-feature training engine changed")


def checkpoint_path(recipe: dict[str, Any]) -> Path:
    model = recipe["model"]
    path = CHECKPOINT_ROOT / model["checkpoint_file"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Download the audited PANNs checkpoint before extraction: {path}"
        )
    if path.stat().st_size != int(model["checkpoint_bytes"]):
        raise RuntimeError("PANNs checkpoint size mismatch")
    if sha256(path) != model["checkpoint_sha256"]:
        raise RuntimeError("PANNs checkpoint SHA-256 mismatch")
    return path


def feature_paths(run_root: Path) -> tuple[Path, Path]:
    root = run_root / "features"
    return root / "panns_cnn14_call_embeddings.npz", root / "feature_manifest.json"


def import_panns_cnn14() -> type[torch.nn.Module]:
    """Load the wheel's model module without its unrelated label-download side effect."""
    package_spec = importlib.util.find_spec("panns_inference")
    if package_spec is None or not package_spec.submodule_search_locations:
        raise RuntimeError("panns_inference package is unavailable")
    package_root = Path(next(iter(package_spec.submodule_search_locations)))
    package = types.ModuleType("panns_inference")
    package.__path__ = [str(package_root)]
    package.__package__ = "panns_inference"
    sys.modules["panns_inference"] = package
    module_spec = importlib.util.spec_from_file_location(
        "panns_inference.models", package_root / "models.py"
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("Unable to load panns_inference.models")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module.Cnn14


def load_panns_model(recipe: dict[str, Any], device: torch.device) -> torch.nn.Module:
    if importlib.metadata.version("panns-inference") != recipe["model"]["runtime_package_version"]:
        raise RuntimeError("panns-inference package version differs from the recipe")
    if importlib.metadata.version("torchlibrosa") != recipe["model"]["torchlibrosa_version"]:
        raise RuntimeError("torchlibrosa package version differs from the recipe")
    cnn14 = import_panns_cnn14()
    model = cnn14(
        sample_rate=32_000,
        window_size=1_024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14_000,
        classes_num=527,
    )
    checkpoint = torch.load(checkpoint_path(recipe), map_location="cpu")
    if sorted(checkpoint) != ["iteration", "model"]:
        raise RuntimeError("Unexpected PANNs checkpoint structure")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    model.requires_grad_(False)
    if model.fc1.in_features != 2_048 or model.fc1.out_features != 2_048:
        raise RuntimeError("Unexpected PANNs CNN14 embedding layer")
    if model.fc_audioset.out_features != 527:
        raise RuntimeError("Unexpected PANNs AudioSet output layer")
    return model


def prepare_embeddings(
    run_root: Path,
    protocol: dict[str, Any],
    recipe: dict[str, Any],
    device: torch.device,
    batch_size: int,
    resume: bool,
) -> Path:
    if batch_size != 1:
        raise ValueError("PANNs variable-duration extraction is locked to batch size 1")
    feature_path, manifest_path = feature_paths(run_root)
    if feature_path.is_file():
        if not resume:
            raise FileExistsError(f"Embedding cache exists; pass --resume to verify: {feature_path}")
        loaded = np.load(feature_path)
        if loaded["embeddings"].shape != (792, 2048):
            raise RuntimeError("Existing PANNs embedding cache has the wrong shape")
        if not manifest_path.is_file() or sha256(feature_path) != read_json(manifest_path)["feature_sha256"]:
            raise RuntimeError("Existing PANNs feature cache failed checksum verification")
        print(f"verified existing embedding cache: {feature_path}", flush=True)
        return feature_path

    checkpoint = checkpoint_path(recipe)
    rows = engine.load_analysis_manifest()
    model = load_panns_model(recipe, device)
    embeddings: list[np.ndarray] = []
    sample_counts: list[int] = []
    source_sample_rates: set[int] = set()
    padded_calls = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Empty filters detected.*", category=UserWarning)
        for call_index, row in enumerate(rows.itertuples(index=False), start=1):
            waveform, sample_rate = torchaudio.load(row.absolute_path)
            source_sample_rates.add(int(sample_rate))
            waveform = waveform.mean(dim=0)
            if int(sample_rate) != 32_000:
                waveform = torchaudio.functional.resample(waveform, int(sample_rate), 32_000)
            samples_before_padding = int(waveform.numel())
            sample_counts.append(samples_before_padding)
            if samples_before_padding < MINIMUM_AUDIO_SAMPLES:
                waveform = torch_functional.pad(
                    waveform, (0, MINIMUM_AUDIO_SAMPLES - samples_before_padding)
                )
                padded_calls += 1
            waveform = waveform.unsqueeze(0).to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                output = model(waveform, None)
                embedding = output["embedding"]
            if tuple(embedding.shape) != (1, 2048) or not torch.isfinite(embedding).all():
                raise RuntimeError(f"Invalid PANNs embedding at call {call_index - 1}")
            embeddings.append(embedding.float().cpu().numpy())
            if call_index % 25 == 0 or call_index == len(rows):
                print(f"PANNs embeddings {call_index}/792", flush=True)

    embedding_array = np.concatenate(embeddings).astype(np.float32)
    if embedding_array.shape != (792, 2048) or not np.isfinite(embedding_array).all():
        raise RuntimeError("Invalid PANNs embedding output")
    if padded_calls != int(recipe["preprocessing"]["clips_below_310ms"]):
        raise RuntimeError(f"Short-call padding count differs: {padded_calls}")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_path,
        embeddings=embedding_array,
        call_ids=rows["filename"].to_numpy(dtype=str),
        cat_ids=rows["analysis_cat_id"].to_numpy(dtype=str),
        labels=rows["label_index"].to_numpy(dtype=np.int64),
        durations=rows["duration_seconds"].to_numpy(dtype=np.float32),
        samples_at_32khz=np.asarray(sample_counts, dtype=np.int64),
        padded_to_310ms=np.asarray(
            [count < MINIMUM_AUDIO_SAMPLES for count in sample_counts], dtype=np.bool_
        ),
        source_paths=rows["local_relpath"].to_numpy(dtype=str),
        runtime_package_version=np.asarray([recipe["model"]["runtime_package_version"]]),
    )
    durations = rows["duration_seconds"].to_numpy(dtype=np.float64)
    elapsed = time.perf_counter() - started
    manifest = {
        "status": "complete",
        "pipeline": recipe["pipeline_id"],
        "feature_path": repo_relative(feature_path),
        "feature_sha256": sha256(feature_path),
        "feature_bytes": feature_path.stat().st_size,
        "shape": list(embedding_array.shape),
        "calls": 792,
        "cats": 111,
        "finite": True,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "official_repository": recipe["model"]["official_repository"],
        "runtime_package": f"panns-inference=={importlib.metadata.version('panns-inference')}",
        "torchlibrosa": importlib.metadata.version("torchlibrosa"),
        "pooling": recipe["model"]["pooling"],
        "source_sample_rates_hz": sorted(source_sample_rates),
        "short_calls_padded_to_310ms": padded_calls,
        "native_duration_seconds": {
            "minimum": float(durations.min()),
            "median": float(np.median(durations)),
            "p95": float(np.quantile(durations, 0.95)),
            "maximum": float(durations.max()),
        },
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "extraction_batch_size": 1,
        "elapsed_seconds": elapsed,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "backbone_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "recipe_sha256": sha256(RECIPE_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "training_engine_sha256": sha256(ENGINE_PATH),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return feature_path


def load_feature_store(run_root: Path, recipe: dict[str, Any]) -> engine.FeatureStore:
    feature_path, manifest_path = feature_paths(run_root)
    if not feature_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Run --stage prepare before smoke or initial screening")
    manifest = read_json(manifest_path)
    if sha256(feature_path) != manifest["feature_sha256"]:
        raise RuntimeError("PANNs feature-cache checksum mismatch")
    loaded = np.load(feature_path)
    store = engine.FeatureStore(
        embeddings=loaded["embeddings"].astype(np.float32),
        call_ids=loaded["call_ids"].astype(str),
        cat_ids=loaded["cat_ids"].astype(str),
        labels=loaded["labels"].astype(np.int64),
    )
    if store.embeddings.shape != (792, int(recipe["model"]["hidden_size"])):
        raise RuntimeError(f"Unexpected feature shape: {store.embeddings.shape}")
    if len(np.unique(store.cat_ids)) != 111 or not np.isfinite(store.embeddings).all():
        raise RuntimeError("Invalid PANNs feature store")
    return store


def stage_manifest(
    stage: str,
    protocol: dict[str, Any],
    recipe: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "recipe_id": recipe["recipe_id"],
        "pipeline": recipe["pipeline_id"],
        "stage": stage,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "recipe_sha256": sha256(RECIPE_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "training_engine_sha256": sha256(ENGINE_PATH),
        "git_revision_at_start": git_revision(),
        "roles_sha256": sha256(ROLES_PATH),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "panns_inference": importlib.metadata.version("panns-inference"),
        "torchlibrosa": importlib.metadata.version("torchlibrosa"),
    }


def run_smoke(
    run_root: Path,
    store: engine.FeatureStore,
    protocol: dict[str, Any],
    recipe: dict[str, Any],
    device: torch.device,
    resume: bool,
) -> None:
    smoke_root = run_root / "smoke"
    summary_path = smoke_root / "smoke_summary.json"
    if summary_path.exists():
        if not resume:
            raise FileExistsError(f"Smoke result exists; pass --resume to inspect: {summary_path}")
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return
    smoke = protocol["stages"]["smoke"]
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    indices = engine.indices_for_roles(
        store, roles, int(smoke["repeat"]), int(smoke["outer_fold"])
    )
    seed = engine.full_model_seed(
        int(smoke["base_seed"]), int(smoke["repeat"]), int(smoke["outer_fold"])
    )
    best_epoch, audit = engine.fit_inner(
        store,
        indices["train"],
        indices["validation"],
        protocol,
        device,
        seed,
        int(smoke["maximum_epochs"]),
    )
    result = {
        "status": "complete",
        "outer_test_accessed": False,
        "pipeline": recipe["pipeline_id"],
        "repeat": int(smoke["repeat"]),
        "outer_fold": int(smoke["outer_fold"]),
        "base_seed": int(smoke["base_seed"]),
        "full_seed": seed,
        "train_calls": int(len(indices["train"])),
        "validation_calls": int(len(indices["validation"])),
        "best_epoch": best_epoch,
        "audit": audit,
    }
    write_json(smoke_root / "run_manifest.json", stage_manifest("smoke", protocol, recipe, device))
    write_json(summary_path, result)
    print(json.dumps(result, indent=2), flush=True)


def fit_directory(initial_root: Path, repeat: int, outer_fold: int, base_seed: int) -> Path:
    return initial_root / "fits" / f"repeat_{repeat}" / f"fold_{outer_fold}" / f"base_seed_{base_seed}"


def aggregate_initial(
    initial_root: Path,
    protocol: dict[str, Any],
    pipeline_id: str,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    stage = protocol["stages"]["initial"]
    base_seed = int(stage["base_seeds"][0])
    candidate_sets: dict[int, pd.DataFrame] = {}
    oof_metrics: dict[str, Any] = {}
    for repeat in stage["repeats"]:
        parts = []
        for outer_fold in stage["outer_folds"]:
            path = fit_directory(initial_root, repeat, outer_fold, base_seed) / "outer_test_call_predictions.csv"
            if not path.is_file():
                raise RuntimeError(f"Missing PANNs prediction file: {path}")
            parts.append(pd.read_csv(path, dtype={"cat_id": str}))
        calls = pd.concat(parts, ignore_index=True)
        metrics, animals = engine.evaluate_frame(calls)
        if len(calls) != 792 or calls["call_index"].nunique() != 792:
            raise RuntimeError("PANNs complete OOF must contain 792 unique calls")
        if len(animals) != 111 or animals["cat_id"].nunique() != 111:
            raise RuntimeError("PANNs complete OOF must contain 111 cats")
        animals.insert(0, "base_seed", base_seed)
        animals.insert(0, "repeat", repeat)
        animals.insert(0, "pipeline", pipeline_id)
        output_path = initial_root / "oof" / f"repeat_{repeat}_base_seed_{base_seed}_animal_predictions.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        animals.to_csv(output_path, index=False)
        candidate_sets[int(repeat)] = animals
        oof_metrics[f"repeat_{repeat}_base_seed_{base_seed}"] = metrics

    values = [metrics["macro_f1"] for metrics in oof_metrics.values()]
    reference_sets = {
        int(repeat): pd.read_csv(
            FORMAL_AST_OOF_ROOT / f"repeat_{repeat}_base_seed_{base_seed}_animal_predictions.csv",
            dtype={"cat_id": str},
        )
        for repeat in stage["repeats"]
    }
    summary = {
        "status": "complete_for_initial_screening",
        "pipeline": pipeline_id,
        "base_seeds": [base_seed],
        "repeats": list(stage["repeats"]),
        "complete_oof_evaluations": len(values),
        "oof_metrics": oof_metrics,
        "aggregate": {
            "macro_f1_mean": float(np.mean(values)),
            "macro_f1_sample_sd": float(np.std(values, ddof=1)),
            "macro_f1_min": float(np.min(values)),
            "macro_f1_max": float(np.max(values)),
        },
        "paired_vs_ast_head_only": engine.paired_bootstrap(
            candidate_sets, reference_sets, bootstrap_repeats
        ),
    }
    write_json(initial_root / "initial_summary.json", summary)
    return summary


def run_initial(
    run_root: Path,
    store: engine.FeatureStore,
    protocol: dict[str, Any],
    recipe: dict[str, Any],
    device: torch.device,
    bootstrap_repeats: int,
    resume: bool,
) -> None:
    initial_root = run_root / "initial"
    initial_root.mkdir(parents=True, exist_ok=True)
    manifest_path = initial_root / "run_manifest.json"
    manifest = stage_manifest("initial", protocol, recipe, device)
    manifest.update(protocol["stages"]["initial"])
    if manifest_path.exists():
        existing = read_json(manifest_path)
        checked = (
            "protocol_sha256",
            "recipe_sha256",
            "runner_sha256",
            "training_engine_sha256",
            "pipeline",
        )
        if any(existing.get(key) != manifest.get(key) for key in checked):
            raise RuntimeError("PANNs initial-screening resume manifest differs")
        if not resume:
            raise FileExistsError("PANNs initial run exists; pass --resume after review")
    else:
        write_json(manifest_path, manifest)

    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    stage = protocol["stages"]["initial"]
    completed = []
    for repeat in stage["repeats"]:
        for outer_fold in stage["outer_folds"]:
            for base_seed in stage["base_seeds"]:
                output_dir = fit_directory(initial_root, repeat, outer_fold, base_seed)
                summary_path = output_dir / "fit_summary.json"
                if summary_path.exists():
                    if not resume:
                        raise FileExistsError(summary_path)
                    completed.append(read_json(summary_path))
                    print(f"resume: {summary_path}", flush=True)
                    continue
                output_dir.mkdir(parents=True, exist_ok=True)
                indices = engine.indices_for_roles(store, roles, repeat, outer_fold)
                seed = engine.full_model_seed(base_seed, repeat, outer_fold)
                print(
                    f"=== PANNs repeat={repeat} fold={outer_fold} base_seed={base_seed} "
                    f"full_seed={seed} ===",
                    flush=True,
                )
                best_epoch, inner_audit = engine.fit_inner(
                    store,
                    indices["train"],
                    indices["validation"],
                    protocol,
                    device,
                    seed,
                    int(protocol["shared_head"]["maximum_epochs"]),
                )
                outer_train = np.concatenate((indices["train"], indices["validation"]))
                test_frame, outer_audit = engine.fit_outer(
                    store,
                    outer_train,
                    indices["test"],
                    protocol,
                    device,
                    seed,
                    best_epoch,
                )
                test_frame.insert(0, "full_seed", seed)
                test_frame.insert(0, "base_seed", base_seed)
                test_frame.insert(0, "outer_fold", outer_fold)
                test_frame.insert(0, "repeat", repeat)
                test_frame.insert(0, "pipeline", recipe["pipeline_id"])
                prediction_path = output_dir / "outer_test_call_predictions.csv"
                test_frame.to_csv(prediction_path, index=False)
                fit_summary = {
                    "status": "complete",
                    "stage": "initial",
                    "pipeline": recipe["pipeline_id"],
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "base_seed": base_seed,
                    "full_seed": seed,
                    "train_calls": int(len(indices["train"])),
                    "validation_calls": int(len(indices["validation"])),
                    "test_calls": int(len(indices["test"])),
                    "selected_epoch": best_epoch,
                    "inner": inner_audit,
                    "outer": outer_audit,
                    "prediction_path": repo_relative(prediction_path),
                }
                write_json(summary_path, fit_summary)
                completed.append(fit_summary)

    initial_summary = aggregate_initial(
        initial_root, protocol, recipe["pipeline_id"], bootstrap_repeats
    )
    run_summary = {
        "status": "complete",
        "stage": "initial",
        "completed_fits": len(completed),
        "expected_fits": int(stage["fold_level_fits"]),
        "complete_oof_evaluations": initial_summary["complete_oof_evaluations"],
        "initial_summary": repo_relative(initial_root / "initial_summary.json"),
        "fits": completed,
    }
    write_json(initial_root / "run_summary.json", run_summary)
    print(json.dumps(initial_summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_PATH)
    recipe = read_json(RECIPE_PATH)
    verify_inputs(protocol, recipe)
    device = engine.resolve_device(args.device)
    run_root = (RUNS_ROOT / args.output_subdir).resolve()
    if RUNS_ROOT.resolve() not in run_root.parents:
        raise ValueError("--output-subdir must stay below the repository runs directory")
    run_root.mkdir(parents=True, exist_ok=True)
    print(
        f"IDEA-049 PANNs CNN14 stage={args.stage}; device={device}; "
        f"device_name={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}",
        flush=True,
    )
    if args.stage == "prepare":
        prepare_embeddings(
            run_root,
            protocol,
            recipe,
            device,
            args.extraction_batch_size,
            args.resume,
        )
        return
    store = load_feature_store(run_root, recipe)
    if args.stage == "smoke":
        run_smoke(run_root, store, protocol, recipe, device, args.resume)
        return
    run_initial(
        run_root,
        store,
        protocol,
        recipe,
        device,
        args.bootstrap_repeats,
        args.resume,
    )


if __name__ == "__main__":
    main()
