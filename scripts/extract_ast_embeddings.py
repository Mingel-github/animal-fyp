"""Extract locked frozen-AST call embeddings for the MeowAgeNet pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as torch_functional
from scipy.signal import resample_poly
from transformers import ASTFeatureExtractor, ASTModel
from transformers.utils import SAFE_WEIGHTS_NAME
from transformers.utils.hub import cached_file


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "protocol" / "meowagenet_locked_v1.json"
MANIFEST_PATH = REPO_ROOT / "metadata" / "datasets" / "meowagenet" / "data_manifest.csv"
RUN_ROOT = REPO_ROOT / "runs" / "ast_locked_v1"
HF_CACHE = REPO_ROOT / "data" / "models" / "huggingface"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        default="ast_standard,ast_time_fine,ast_frequency_fine",
        help="Comma-separated AST variants from the locked protocol.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device. 'auto' selects CUDA when available.",
    )
    parser.add_argument("--limit-calls", type=int, default=None)
    parser.add_argument("--output-subdir", default="full")
    parser.add_argument("--reuse-features", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this PyTorch environment cannot access it. "
            "Check that a CUDA-enabled PyTorch wheel is installed."
        )
    return torch.device(requested)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_audio(path: Path, target_rate: int) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if sample_rate != target_rate:
        divisor = math.gcd(sample_rate, target_rate)
        waveform = resample_poly(waveform, target_rate // divisor, sample_rate // divisor).astype(
            np.float32
        )
    return waveform


def segment_waveform(waveform: np.ndarray, window_samples: int, hop_samples: int) -> list[np.ndarray]:
    if len(waveform) <= window_samples:
        return [waveform]
    starts = list(range(0, len(waveform) - window_samples + 1, hop_samples))
    final_start = len(waveform) - window_samples
    if starts[-1] != final_start:
        starts.append(final_start)
    return [waveform[start : start + window_samples] for start in starts]


def prepare_features(
    manifest: pd.DataFrame,
    feature_extractor: ASTFeatureExtractor,
    sample_rate: int,
    segment_seconds: float,
    hop_seconds: float,
) -> dict[str, np.ndarray]:
    window_samples = round(sample_rate * segment_seconds)
    hop_samples = round(sample_rate * hop_seconds)
    features: list[np.ndarray] = []
    segment_call_indices: list[int] = []
    segment_counts: list[int] = []
    call_ids: list[str] = []
    cat_ids: list[str] = []
    labels: list[int] = []
    durations: list[float] = []
    source_paths: list[str] = []
    label_to_id = {"kitten": 0, "adult": 1, "senior": 2}

    start_time = time.perf_counter()
    for call_index, row in enumerate(manifest.itertuples(index=False)):
        audio_path = REPO_ROOT / row.local_relpath
        waveform = load_audio(audio_path, sample_rate)
        segments = segment_waveform(waveform, window_samples, hop_samples)
        batch = feature_extractor(segments, sampling_rate=sample_rate, return_tensors="np")[
            "input_values"
        ].astype(np.float32)
        features.extend(batch)
        segment_call_indices.extend([call_index] * len(segments))
        segment_counts.append(len(segments))
        call_ids.append(row.filename)
        cat_ids.append(row.analysis_cat_id)
        labels.append(label_to_id[row.age_group_filename])
        durations.append(float(row.duration_seconds))
        source_paths.append(row.source_path)
        if (call_index + 1) % 100 == 0 or call_index + 1 == len(manifest):
            print(f"Prepared fbank features for {call_index + 1}/{len(manifest)} calls", flush=True)

    return {
        "features": np.stack(features),
        "segment_call_indices": np.asarray(segment_call_indices, dtype=np.int32),
        "segment_counts": np.asarray(segment_counts, dtype=np.int16),
        "call_ids": np.asarray(call_ids),
        "cat_ids": np.asarray(cat_ids),
        "labels": np.asarray(labels, dtype=np.int8),
        "durations": np.asarray(durations, dtype=np.float32),
        "source_paths": np.asarray(source_paths),
        "feature_seconds": np.asarray([time.perf_counter() - start_time], dtype=np.float64),
    }


def adapt_geometry(
    model: ASTModel,
    max_length: int,
    frequency_stride: int,
    time_stride: int,
) -> dict[str, object]:
    config = model.config
    embeddings = model.embeddings
    patch_size = int(config.patch_size)
    old_frequency = (config.num_mel_bins - patch_size) // config.frequency_stride + 1
    old_time = (config.max_length - patch_size) // config.time_stride + 1
    new_frequency = (config.num_mel_bins - patch_size) // frequency_stride + 1
    new_time = (max_length - patch_size) // time_stride + 1

    projection_weight = embeddings.patch_embeddings.projection.weight.detach().clone()
    projection_bias = embeddings.patch_embeddings.projection.bias.detach().clone()
    special_positions = embeddings.position_embeddings[:, :2]
    patch_positions = embeddings.position_embeddings[:, 2:]
    hidden_size = patch_positions.shape[-1]
    patch_positions = patch_positions.reshape(1, old_frequency, old_time, hidden_size)
    patch_positions = patch_positions.permute(0, 3, 1, 2)
    patch_positions = torch_functional.interpolate(
        patch_positions,
        size=(new_frequency, new_time),
        mode="bilinear",
        align_corners=False,
    )
    patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(
        1, new_frequency * new_time, hidden_size
    )
    embeddings.position_embeddings = torch.nn.Parameter(
        torch.cat((special_positions, patch_positions), dim=1)
    )
    embeddings.patch_embeddings.projection.stride = (frequency_stride, time_stride)
    config.max_length = max_length
    config.frequency_stride = frequency_stride
    config.time_stride = time_stride
    embeddings.config = config

    projection_reused = torch.equal(
        projection_weight, embeddings.patch_embeddings.projection.weight.detach()
    ) and torch.equal(projection_bias, embeddings.patch_embeddings.projection.bias.detach())
    if not projection_reused:
        raise RuntimeError("Patch projection weights changed during stride adaptation")
    for parameter in model.parameters():
        parameter.requires_grad = False
    return {
        "source_grid": [old_frequency, old_time],
        "target_grid": [new_frequency, new_time],
        "patch_tokens": new_frequency * new_time,
        "total_tokens": new_frequency * new_time + 2,
        "patch_projection_weights_reused": projection_reused,
        "position_interpolation": "bilinear",
        "trainable_encoder_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_encoder_parameters": sum(p.numel() for p in model.parameters()),
    }


def extract_variant(
    variant_name: str,
    variant: dict[str, int],
    protocol: dict[str, object],
    feature_data: dict[str, np.ndarray],
    batch_size: int,
    output_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    ast_config = protocol["ast"]
    print(f"Loading {variant_name} from {ast_config['checkpoint']}...", flush=True)
    model = ASTModel.from_pretrained(
        ast_config["checkpoint"],
        revision=ast_config["revision"],
        cache_dir=HF_CACHE,
        use_safetensors=True,
    )
    geometry_audit = adapt_geometry(
        model,
        max_length=ast_config["max_length_frames"],
        frequency_stride=variant["frequency_stride"],
        time_stride=variant["time_stride"],
    )
    model.to(device)
    model.eval()
    features = feature_data["features"]
    segment_embeddings: list[np.ndarray] = []
    start_time = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            batch = torch.from_numpy(features[start:stop]).to(device)
            pooled = model(input_values=batch).pooler_output
            segment_embeddings.append(pooled.cpu().numpy().astype(np.float32))
            if stop % 100 < batch_size or stop == len(features):
                print(f"{variant_name}: embedded {stop}/{len(features)} segments", flush=True)
    segment_embeddings_array = np.concatenate(segment_embeddings, axis=0)
    call_count = len(feature_data["call_ids"])
    call_embeddings = np.zeros((call_count, segment_embeddings_array.shape[1]), dtype=np.float64)
    np.add.at(call_embeddings, feature_data["segment_call_indices"], segment_embeddings_array)
    call_embeddings /= feature_data["segment_counts"][:, None]
    call_embeddings = call_embeddings.astype(np.float32)
    output_path = output_dir / f"{variant_name}_call_embeddings.npz"
    np.savez_compressed(
        output_path,
        embeddings=call_embeddings,
        call_ids=feature_data["call_ids"],
        cat_ids=feature_data["cat_ids"],
        labels=feature_data["labels"],
        durations=feature_data["durations"],
        segment_counts=feature_data["segment_counts"],
        source_paths=feature_data["source_paths"],
    )
    elapsed = time.perf_counter() - start_time
    del model, segment_embeddings_array, call_embeddings
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "variant": variant_name,
        "frequency_stride": variant["frequency_stride"],
        "time_stride": variant["time_stride"],
        "calls": call_count,
        "segments": int(len(features)),
        "embedding_dimensions": 768,
        "batch_size": batch_size,
        "inference_seconds": elapsed,
        "device": str(device),
        "output": str(output_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "geometry_audit": geometry_audit,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(6, os.cpu_count() or 1))
        torch.set_num_interop_threads(1)
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"AST execution device: {device} ({device_name})", flush=True)
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    ast_config = protocol["ast"]
    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = set(variants) - set(ast_config["variants"])
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")

    manifest = pd.read_csv(MANIFEST_PATH)
    include = manifest["analysis_include"].astype(str).str.lower() == "true"
    manifest = manifest[include].sort_values("source_path").reset_index(drop=True)
    if args.limit_calls is not None:
        manifest = manifest.iloc[: args.limit_calls].copy()
    elif len(manifest) != protocol["dataset"]["expected_calls"]:
        raise ValueError(f"Expected {protocol['dataset']['expected_calls']} calls, found {len(manifest)}")

    output_dir = RUN_ROOT / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "ast_fbank_128.npz"
    feature_extractor = ASTFeatureExtractor.from_pretrained(
        ast_config["checkpoint"],
        revision=ast_config["revision"],
        cache_dir=HF_CACHE,
    )
    feature_extractor.max_length = ast_config["max_length_frames"]
    feature_extractor.num_mel_bins = ast_config["num_mel_bins"]
    if args.reuse_features and feature_path.exists():
        loaded = np.load(feature_path)
        feature_data = {key: loaded[key] for key in loaded.files}
    else:
        feature_data = prepare_features(
            manifest,
            feature_extractor,
            ast_config["sample_rate_hz"],
            ast_config["segment_seconds"],
            ast_config["segment_hop_seconds"],
        )
        np.savez_compressed(feature_path, **feature_data)

    model_file = Path(
        cached_file(
            ast_config["checkpoint"],
            SAFE_WEIGHTS_NAME,
            revision=ast_config["revision"],
            cache_dir=HF_CACHE,
        )
    )
    variant_audits = []
    for variant_name in variants:
        variant_audits.append(
            extract_variant(
                variant_name,
                ast_config["variants"][variant_name],
                protocol,
                feature_data,
                args.batch_size,
                output_dir,
                device,
            )
        )
    audit = {
        "protocol_id": protocol["protocol_id"],
        "checkpoint": ast_config["checkpoint"],
        "revision": ast_config["revision"],
        "model_safetensors_sha256": sha256(model_file),
        "model_safetensors_bytes": model_file.stat().st_size,
        "execution": {
            "device": str(device),
            "device_name": device_name,
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
        },
        "feature_extractor": {
            "sampling_rate": feature_extractor.sampling_rate,
            "max_length": feature_extractor.max_length,
            "num_mel_bins": feature_extractor.num_mel_bins,
            "mean": feature_extractor.mean,
            "std": feature_extractor.std,
        },
        "calls": int(len(feature_data["call_ids"])),
        "segments": int(len(feature_data["features"])),
        "segment_count_range": [
            int(feature_data["segment_counts"].min()),
            int(feature_data["segment_counts"].max()),
        ],
        "feature_seconds": float(feature_data["feature_seconds"][0]),
        "variants": variant_audits,
    }
    audit_path = output_dir / "ast_embedding_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
