"""Extract per-layer frozen standard-AST call representations for IDEA-019."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
from transformers import ASTModel


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "protocol" / "meowagenet_locked_v1.json"
FEATURE_PATH = (
    REPO_ROOT
    / "runs"
    / "ast_locked_v1"
    / "gpu_rerun_2026-08-26"
    / "ast_fbank_128.npz"
)
FROZEN_EMBEDDING_PATH = (
    REPO_ROOT
    / "runs"
    / "ast_locked_v1"
    / "gpu_rerun_2026-08-26"
    / "ast_standard_call_embeddings.npz"
)
HF_CACHE = REPO_ROOT / "data" / "models" / "huggingface"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-subdir", default="ast_layer_embeddings_v2_float32")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
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


def adapt_standard_geometry(model: ASTModel, protocol: dict[str, object]) -> dict[str, object]:
    ast_config = protocol["ast"]
    variant = ast_config["variants"]["ast_standard"]
    config = model.config
    embeddings = model.embeddings
    patch_size = int(config.patch_size)
    old_frequency = (config.num_mel_bins - patch_size) // config.frequency_stride + 1
    old_time = (config.max_length - patch_size) // config.time_stride + 1
    new_frequency = (
        config.num_mel_bins - patch_size
    ) // int(variant["frequency_stride"]) + 1
    new_time = (
        int(ast_config["max_length_frames"]) - patch_size
    ) // int(variant["time_stride"]) + 1
    projection_weight = embeddings.patch_embeddings.projection.weight.detach().clone()
    projection_bias = embeddings.patch_embeddings.projection.bias.detach().clone()
    special_positions = embeddings.position_embeddings[:, :2]
    patch_positions = embeddings.position_embeddings[:, 2:]
    hidden_size = patch_positions.shape[-1]
    patch_positions = patch_positions.reshape(
        1, old_frequency, old_time, hidden_size
    ).permute(0, 3, 1, 2)
    patch_positions = torch.nn.functional.interpolate(
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
    embeddings.patch_embeddings.projection.stride = (
        int(variant["frequency_stride"]),
        int(variant["time_stride"]),
    )
    config.max_length = int(ast_config["max_length_frames"])
    config.frequency_stride = int(variant["frequency_stride"])
    config.time_stride = int(variant["time_stride"])
    embeddings.config = config
    if not (
        torch.equal(projection_weight, embeddings.patch_embeddings.projection.weight)
        and torch.equal(projection_bias, embeddings.patch_embeddings.projection.bias)
    ):
        raise RuntimeError("Patch projection changed during geometry adaptation")
    return {
        "source_grid": [old_frequency, old_time],
        "target_grid": [new_frequency, new_time],
        "patch_tokens": new_frequency * new_time,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    output_dir = REPO_ROOT / "runs" / args.output_subdir
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output_dir}")
    output_dir.mkdir(parents=True)

    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    loaded = np.load(FEATURE_PATH)
    frozen = np.load(FROZEN_EMBEDDING_PATH)
    call_ids = loaded["call_ids"].astype(str)
    if not np.array_equal(call_ids, frozen["call_ids"].astype(str)):
        raise RuntimeError("Fbank and frozen embedding call order differs")
    features = loaded["features"].astype(np.float32)
    segment_call_indices = loaded["segment_call_indices"].astype(np.int64)
    segment_counts = loaded["segment_counts"].astype(np.int64)
    device = resolve_device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    ast_config = protocol["ast"]
    model = ASTModel.from_pretrained(
        ast_config["checkpoint"],
        revision=ast_config["revision"],
        cache_dir=HF_CACHE,
        use_safetensors=True,
    )
    geometry = adapt_standard_geometry(model, protocol)
    model.eval().to(device)
    layer_count = len(model.encoder.layer)
    hidden_size = int(model.config.hidden_size)
    call_sums = np.zeros((len(call_ids), layer_count, hidden_size), dtype=np.float64)
    started = time.perf_counter()
    print(f"Layer extraction device: {device} ({device_name})", flush=True)
    with torch.no_grad():
        for start in range(0, len(features), args.batch_size):
            stop = min(start + args.batch_size, len(features))
            batch = torch.from_numpy(features[start:stop]).to(device)
            output = model(input_values=batch, output_hidden_states=True)
            per_layer = torch.stack(
                [
                    model.layernorm(hidden_state)[:, :2].mean(dim=1)
                    for hidden_state in output.hidden_states[1:]
                ],
                dim=1,
            )
            per_layer_np = per_layer.float().cpu().numpy()
            np.add.at(call_sums, segment_call_indices[start:stop], per_layer_np)
            print(f"Embedded segments {stop}/{len(features)}", flush=True)
    call_embeddings = (call_sums / segment_counts[:, None, None]).astype(np.float32)
    final_layer_difference = np.abs(
        call_embeddings[:, -1] - frozen["embeddings"].astype(np.float32)
    )
    output_path = output_dir / "ast_standard_layer_call_embeddings.npz"
    np.savez_compressed(
        output_path,
        embeddings=call_embeddings,
        call_ids=call_ids,
        cat_ids=loaded["cat_ids"].astype(str),
        labels=loaded["labels"].astype(np.int8),
        layer_indices_zero_based=np.arange(layer_count, dtype=np.int8),
        layer_indices_one_based=np.arange(1, layer_count + 1, dtype=np.int8),
    )
    summary = {
        "artifact": "ast-standard-layer-call-embeddings-v2-float32",
        "protocol_id": protocol["protocol_id"],
        "protocol_config_sha256": sha256(CONFIG_PATH),
        "feature_sha256": sha256(FEATURE_PATH),
        "frozen_embedding_sha256": sha256(FROZEN_EMBEDDING_PATH),
        "execution": {
            "device": str(device),
            "device_name": device_name,
            "torch_version": torch.__version__,
            "model_precision": "float32",
            "seconds": time.perf_counter() - started,
        },
        "geometry": geometry,
        "scope": {
            "segments": int(len(features)),
            "calls": int(len(call_ids)),
            "cats": int(len(np.unique(loaded["cat_ids"].astype(str)))),
            "layers": layer_count,
            "hidden_size": hidden_size,
        },
        "layer_representation": "final LayerNorm applied to each block output, then mean of CLS and distillation tokens, then mean across call segments",
        "final_layer_vs_locked_pooler_output": {
            "mean_absolute_difference": float(final_layer_difference.mean()),
            "maximum_absolute_difference": float(final_layer_difference.max()),
        },
        "output": str(output_path.relative_to(REPO_ROOT)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
