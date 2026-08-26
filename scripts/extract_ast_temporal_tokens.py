"""Extract frozen standard-AST temporal token sequences for IDEA-013."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
from transformers import ASTModel


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from extract_ast_embeddings import adapt_geometry  # noqa: E402


CONFIG_PATH = REPO_ROOT / "configs" / "protocol" / "meowagenet_locked_v1.json"
FBANK_PATH = (
    REPO_ROOT
    / "runs"
    / "ast_locked_v1"
    / "gpu_rerun_2026-08-26"
    / "ast_fbank_128.npz"
)
HF_CACHE = REPO_ROOT / "data" / "models" / "huggingface"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-subdir", default="ast_temporal_tokens_v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--minimum-real-overlap-fraction",
        type=float,
        default=0.5,
        help="Minimum real-fbank overlap required to retain an AST time patch.",
    )
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


def valid_fbank_frames(features: np.ndarray) -> np.ndarray:
    # ASTFeatureExtractor pads missing frames with one constant normalized value.
    # Real log-mel frames vary over frequency, including quiet frames.
    frame_standard_deviation = features.std(axis=2)
    counts = (frame_standard_deviation > 1.0e-7).sum(axis=1).astype(np.int16)
    if np.any(counts < 1):
        raise RuntimeError("At least one segment has no detectable real fbank frame")
    return counts


def temporal_patch_mask(
    frame_counts: np.ndarray,
    time_positions: int,
    patch_size: int,
    time_stride: int,
    minimum_overlap_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(time_positions, dtype=np.int16) * time_stride
    real_overlap = np.clip(frame_counts[:, None] - starts[None, :], 0, patch_size)
    threshold = patch_size * minimum_overlap_fraction
    mask = real_overlap >= threshold
    for row_index in np.flatnonzero(mask.sum(axis=1) == 0):
        mask[row_index, int(real_overlap[row_index].argmax())] = True
    return mask, real_overlap.astype(np.int16)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.minimum_real_overlap_fraction <= 1.0:
        raise ValueError("minimum-real-overlap-fraction must be in (0, 1]")
    output_dir = REPO_ROOT / "runs" / args.output_subdir
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    ast_config = protocol["ast"]
    standard = ast_config["variants"]["ast_standard"]
    fbank = np.load(FBANK_PATH)
    features = fbank["features"].astype(np.float32)
    segment_call_indices = fbank["segment_call_indices"].astype(np.int64)
    frame_counts = valid_fbank_frames(features)

    device = resolve_device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Temporal-token extraction device: {device} ({device_name})", flush=True)
    model = ASTModel.from_pretrained(
        ast_config["checkpoint"],
        revision=ast_config["revision"],
        cache_dir=HF_CACHE,
        use_safetensors=True,
    )
    geometry = adapt_geometry(
        model,
        max_length=int(ast_config["max_length_frames"]),
        frequency_stride=int(standard["frequency_stride"]),
        time_stride=int(standard["time_stride"]),
    )
    model.to(device)
    model.eval()
    frequency_positions, time_positions = geometry["target_grid"]
    patch_size = int(model.config.patch_size)
    mask, real_overlap = temporal_patch_mask(
        frame_counts,
        time_positions,
        patch_size,
        int(standard["time_stride"]),
        args.minimum_real_overlap_fraction,
    )

    token_batches: list[np.ndarray] = []
    token_segment_indices: list[np.ndarray] = []
    token_local_time_indices: list[np.ndarray] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for start in range(0, len(features), args.batch_size):
            stop = min(start + args.batch_size, len(features))
            batch = torch.from_numpy(features[start:stop]).to(device)
            hidden = model(input_values=batch).last_hidden_state[:, 2:, :]
            expected_tokens = frequency_positions * time_positions
            if hidden.shape[1] != expected_tokens:
                raise RuntimeError(
                    f"Unexpected patch-token count {hidden.shape[1]} != {expected_tokens}"
                )
            temporal = hidden.reshape(
                stop - start, frequency_positions, time_positions, hidden.shape[-1]
            ).mean(dim=1)
            temporal = temporal.float().cpu().numpy()
            for local_segment, segment_index in enumerate(range(start, stop)):
                selected_times = np.flatnonzero(mask[segment_index]).astype(np.int16)
                token_batches.append(temporal[local_segment, selected_times].astype(np.float32))
                token_segment_indices.append(
                    np.full(len(selected_times), segment_index, dtype=np.int32)
                )
                token_local_time_indices.append(selected_times)
            if stop % 100 < args.batch_size or stop == len(features):
                print(f"Embedded {stop}/{len(features)} segments", flush=True)
    elapsed = time.perf_counter() - started
    temporal_tokens = np.concatenate(token_batches, axis=0)
    token_segment_indices_array = np.concatenate(token_segment_indices)
    token_local_time_indices_array = np.concatenate(token_local_time_indices)
    token_call_indices = segment_call_indices[token_segment_indices_array]
    call_token_counts = np.bincount(
        token_call_indices, minlength=len(fbank["call_ids"])
    ).astype(np.int16)
    if np.any(call_token_counts < 1):
        raise RuntimeError("At least one call has no retained temporal token")

    output_path = output_dir / "ast_standard_temporal_tokens.npz"
    np.savez_compressed(
        output_path,
        temporal_tokens=temporal_tokens,
        token_call_indices=token_call_indices.astype(np.int32),
        token_segment_indices=token_segment_indices_array,
        token_local_time_indices=token_local_time_indices_array,
        segment_call_indices=segment_call_indices.astype(np.int32),
        segment_valid_fbank_frames=frame_counts,
        segment_valid_time_tokens=mask.sum(axis=1).astype(np.int16),
        segment_time_patch_real_overlap=real_overlap,
        call_token_counts=call_token_counts,
        call_ids=fbank["call_ids"],
        cat_ids=fbank["cat_ids"],
        labels=fbank["labels"],
        durations=fbank["durations"],
        segment_counts=fbank["segment_counts"],
        source_paths=fbank["source_paths"],
    )
    summary = {
        "experiment": "ast-standard-temporal-token-extraction-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_config_sha256": sha256(CONFIG_PATH),
        "fbank_sha256": sha256(FBANK_PATH),
        "checkpoint": ast_config["checkpoint"],
        "revision": ast_config["revision"],
        "device": str(device),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "batch_size": args.batch_size,
        "calls": int(len(fbank["call_ids"])),
        "segments": int(len(features)),
        "temporal_tokens": int(len(temporal_tokens)),
        "embedding_dimensions": int(temporal_tokens.shape[1]),
        "call_token_count_range": [int(call_token_counts.min()), int(call_token_counts.max())],
        "call_token_count_quantiles": {
            str(quantile): float(np.quantile(call_token_counts, quantile))
            for quantile in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "mask": {
            "real_frame_detection": "fbank row standard deviation > 1e-7",
            "patch_size_frames": patch_size,
            "time_stride_frames": int(standard["time_stride"]),
            "minimum_real_overlap_fraction": args.minimum_real_overlap_fraction,
            "minimum_one_token_per_segment": True,
        },
        "geometry_audit": geometry,
        "inference_seconds": elapsed,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "output": str(output_path.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    summary_path = output_dir / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
