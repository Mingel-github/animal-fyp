"""Extract the label-free AST block-11 token cache preregistered for IDEA-078."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
from transformers import ASTModel
from transformers.utils import SAFE_WEIGHTS_NAME
from transformers.utils.hub import cached_file


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea078_ast_prelast_special_token_age_injection_v1.json"
)
GEOMETRY_RUNNER_PATH = REPO_ROOT / "scripts" / "extract_ast_layer_embeddings.py"
HF_CACHE = REPO_ROOT / "data" / "models" / "huggingface"

SPEC = importlib.util.spec_from_file_location("idea078_geometry", GEOMETRY_RUNNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load locked AST geometry helper")
geometry_helper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = geometry_helper
SPEC.loader.exec_module(geometry_helper)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "extract"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def string_array_sha256(value: np.ndarray) -> str:
    payload = ("\n".join(value.astype(str).tolist()) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea078-ast-prelast-special-token-age-injection-v1":
        raise RuntimeError("Unexpected IDEA-078 protocol")
    cache = protocol["cache"]
    if cache["prelast_block_output_one_based"] != 11:
        raise RuntimeError("IDEA-078 cache layer changed")
    if cache["last_block_input_one_based"] != 12:
        raise RuntimeError("IDEA-078 tail layer changed")
    if cache["shape"] != [843, 146, 768] or cache["dtype"] != "float32":
        raise RuntimeError("IDEA-078 cache geometry changed")
    if cache["contains_labels"] or cache["contains_roles"] or cache["contains_cat_ids"]:
        raise RuntimeError("IDEA-078 cache must remain label-free")
    dependencies = protocol["dependencies"]
    checks = {
        "plan_sha256": REPO_ROOT / dependencies["plan_path"],
        "cache_schema_sha256": REPO_ROOT / dependencies["cache_schema_path"],
        "fbank_sha256": REPO_ROOT / protocol["data"]["fbank_path"],
        "locked_embedding_sha256": REPO_ROOT / protocol["data"]["frozen_embedding_path"],
        "data_manifest_sha256": REPO_ROOT / protocol["data"]["data_manifest_path"],
        "audio_checksums_sha256": REPO_ROOT / protocol["data"]["audio_checksums_path"],
        "roles_sha256": REPO_ROOT / protocol["data"]["roles_path"],
        "geometry_runner_sha256": GEOMETRY_RUNNER_PATH,
        "cache_extractor_sha256": Path(__file__).resolve(),
    }
    for field, path in checks.items():
        if sha256(path) != dependencies[field]:
            raise RuntimeError(f"IDEA-078 dependency hash changed: {field}")


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def load_source(protocol: dict[str, Any]) -> dict[str, np.ndarray]:
    path = REPO_ROOT / protocol["data"]["fbank_path"]
    with np.load(path) as loaded:
        required = (
            "features",
            "segment_call_indices",
            "segment_counts",
            "call_ids",
            "source_paths",
        )
        if any(key not in loaded.files for key in required):
            raise RuntimeError("IDEA-078 fbank source is incomplete")
        source = {key: loaded[key].copy() for key in required}
    if source["features"].shape != (843, 128, 128):
        raise RuntimeError("IDEA-078 fbank segment geometry changed")
    if source["segment_call_indices"].shape != (843,):
        raise RuntimeError("IDEA-078 segment mapping changed")
    if source["segment_counts"].shape != (792,) or int(source["segment_counts"].sum()) != 843:
        raise RuntimeError("IDEA-078 segment counts changed")
    if source["call_ids"].shape != (792,):
        raise RuntimeError("IDEA-078 call ids changed")
    if source["source_paths"].shape != (792,):
        raise RuntimeError("IDEA-078 source paths changed")
    observed_counts = np.bincount(
        source["segment_call_indices"].astype(np.int64), minlength=792
    )
    if not np.array_equal(observed_counts, source["segment_counts"].astype(np.int64)):
        raise RuntimeError("IDEA-078 843-to-792 segment mapping is not closed")
    if int(source["segment_call_indices"].min()) != 0 or int(source["segment_call_indices"].max()) != 791:
        raise RuntimeError("IDEA-078 segment mapping does not cover all calls")
    return source


def output_paths(protocol: dict[str, Any]) -> tuple[Path, Path, Path]:
    root = REPO_ROOT / protocol["cache"]["output_root"]
    token_path = root / protocol["cache"]["token_filename"]
    index_path = root / protocol["cache"]["index_filename"]
    summary_path = root / protocol["cache"]["summary_filename"]
    return token_path, index_path, summary_path


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    source = load_source(protocol)
    token_path, index_path, summary_path = output_paths(protocol)
    expected_bytes = int(np.prod(protocol["cache"]["shape"]) * 4)
    free_bytes = shutil.disk_usage(token_path.parent.parent).free
    if free_bytes < expected_bytes * 2:
        raise RuntimeError("IDEA-078 has insufficient disk headroom for atomic cache extraction")
    if any(path.exists() for path in (token_path, index_path, summary_path)):
        state = "existing_cache_requires_read_only_validation"
    else:
        state = "ready_for_label_free_extraction"
    return {
        "status": "GO",
        "read_only": True,
        "stage": "cache_preflight",
        "state": state,
        "requested_device": args.device,
        "gpu_used": False,
        "segments": int(len(source["features"])),
        "calls": int(len(source["call_ids"])),
        "segment_count_range": [
            int(source["segment_counts"].min()),
            int(source["segment_counts"].max()),
        ],
        "planned_shape": protocol["cache"]["shape"],
        "planned_dtype": protocol["cache"]["dtype"],
        "planned_raw_data_bytes": expected_bytes,
        "planned_raw_data_mib": expected_bytes / (1024**2),
        "free_disk_bytes": free_bytes,
        "contains_labels": False,
        "contains_roles": False,
        "contains_cat_ids": False,
        "segment_call_mapping_closed": True,
        "segment_call_indices_sha256": array_sha256(
            source["segment_call_indices"].astype(np.int32)
        ),
        "segment_counts_sha256": array_sha256(source["segment_counts"].astype(np.int16)),
        "call_ids_sha256": string_array_sha256(source["call_ids"]),
        "source_paths_sha256": string_array_sha256(source["source_paths"]),
    }


def extract(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    source = load_source(protocol)
    token_path, index_path, summary_path = output_paths(protocol)
    if any(path.exists() for path in (token_path, index_path, summary_path)):
        raise FileExistsError("IDEA-078 refuses to overwrite an existing cache artifact")
    token_path.parent.mkdir(parents=True, exist_ok=False)
    temporary_token = token_path.with_suffix(token_path.suffix + ".tmp")
    temporary_index = index_path.with_suffix(index_path.suffix + ".tmp.npz")
    try:
        device = resolve_device(args.device)
        ast = protocol["ast"]
        model = ASTModel.from_pretrained(
            ast["checkpoint"],
            revision=ast["revision"],
            cache_dir=HF_CACHE,
            use_safetensors=True,
        )
        geometry = geometry_helper.adapt_standard_geometry(model, read_json(REPO_ROOT / protocol["data"]["locked_ast_protocol_path"]))
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval().to(device)
        if len(model.encoder.layer) != 12:
            raise RuntimeError("IDEA-078 expected 12 AST blocks")
        shape = tuple(protocol["cache"]["shape"])
        output = np.lib.format.open_memmap(
            temporary_token,
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
        features = source["features"].astype(np.float32, copy=False)
        started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(features), args.batch_size):
                stop = min(start + args.batch_size, len(features))
                batch = torch.from_numpy(features[start:stop]).to(device)
                hidden = model.embeddings(batch)
                head_mask = model.get_head_mask(None, model.config.num_hidden_layers)
                for layer_index in range(11):
                    layer_head_mask = head_mask[layer_index] if head_mask is not None else None
                    hidden = model.encoder.layer[layer_index](hidden, layer_head_mask, False)[0]
                output[start:stop] = hidden.float().cpu().numpy()
                print(f"IDEA-078 cached pre-last tokens {stop}/{len(features)}", flush=True)
        output.flush()
        del output
        np.savez_compressed(
            temporary_index,
            segment_call_indices=source["segment_call_indices"].astype(np.int32),
            segment_counts=source["segment_counts"].astype(np.int16),
            call_ids=source["call_ids"].astype(str),
            source_paths=source["source_paths"].astype(str),
            prelast_block_output_one_based=np.asarray([11], dtype=np.int8),
            last_block_input_one_based=np.asarray([12], dtype=np.int8),
            special_token_indices=np.asarray([0, 1], dtype=np.int8),
            patch_token_start=np.asarray([2], dtype=np.int8),
        )
        cached_tokens = np.load(temporary_token, mmap_mode="r")
        reconstructed_sum = np.zeros((792, 768), dtype=np.float64)
        with torch.inference_mode():
            head_mask = model.get_head_mask(None, model.config.num_hidden_layers)
            last_head_mask = head_mask[11] if head_mask is not None else None
            for start in range(0, len(cached_tokens), args.batch_size):
                stop = min(start + args.batch_size, len(cached_tokens))
                hidden = torch.from_numpy(
                    np.asarray(cached_tokens[start:stop], dtype=np.float32)
                ).to(device)
                hidden = model.encoder.layer[11](hidden, last_head_mask, False)[0]
                hidden = model.layernorm(hidden)
                pooled = ((hidden[:, 0] + hidden[:, 1]) / 2.0).float().cpu().numpy()
                np.add.at(
                    reconstructed_sum,
                    source["segment_call_indices"][start:stop].astype(np.int64),
                    pooled.astype(np.float64),
                )
        reconstructed = (
            reconstructed_sum / source["segment_counts"].astype(np.float64)[:, None]
        ).astype(np.float32)
        with np.load(REPO_ROOT / protocol["data"]["frozen_embedding_path"]) as locked:
            locked_final = locked["embeddings"].astype(np.float32)
            locked_call_ids = locked["call_ids"].astype(str)
        if not np.array_equal(source["call_ids"].astype(str), locked_call_ids):
            raise RuntimeError("IDEA-078 reconstruction call order changed")
        reconstruction_error = np.abs(reconstructed - locked_final)
        reconstruction_mean_absolute = float(reconstruction_error.mean())
        reconstruction_absolute_maximum = float(reconstruction_error.max())
        tolerance = protocol["preflight"]["full_cache_vs_locked_final_tolerance"]
        if (
            reconstruction_mean_absolute > float(tolerance["mean_absolute_maximum"])
            or reconstruction_absolute_maximum > float(tolerance["absolute_maximum"])
        ):
            raise RuntimeError("IDEA-078 cache does not reconstruct the locked AST pooler")
        temporary_token.replace(token_path)
        temporary_index.replace(index_path)
        model_file = Path(
            cached_file(
                ast["checkpoint"],
                SAFE_WEIGHTS_NAME,
                revision=ast["revision"],
                cache_dir=HF_CACHE,
            )
        )
        summary = {
            "status": "complete",
            "artifact": "idea078-label-free-ast-prelast-token-cache-v1",
            "cache_schema_path": protocol["dependencies"]["cache_schema_path"],
            "cache_schema_sha256": protocol["dependencies"]["cache_schema_sha256"],
            "label_information_used": False,
            "roles_used": False,
            "cat_ids_stored": False,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "extractor_sha256": sha256(Path(__file__).resolve()),
            "source_fbank_path": protocol["data"]["fbank_path"],
            "source_fbank_sha256": sha256(REPO_ROOT / protocol["data"]["fbank_path"]),
            "source_data_manifest_path": protocol["data"]["data_manifest_path"],
            "source_data_manifest_sha256": sha256(
                REPO_ROOT / protocol["data"]["data_manifest_path"]
            ),
            "source_audio_checksums_path": protocol["data"]["audio_checksums_path"],
            "source_audio_checksums_sha256": sha256(
                REPO_ROOT / protocol["data"]["audio_checksums_path"]
            ),
            "downstream_roles_path": protocol["data"]["roles_path"],
            "downstream_roles_sha256": sha256(
                REPO_ROOT / protocol["data"]["roles_path"]
            ),
            "locked_final_embedding_path": protocol["data"]["frozen_embedding_path"],
            "locked_final_embedding_sha256": sha256(
                REPO_ROOT / protocol["data"]["frozen_embedding_path"]
            ),
            "model_safetensors_sha256": sha256(model_file),
            "checkpoint": ast["checkpoint"],
            "revision": ast["revision"],
            "geometry": geometry,
            "segments": 843,
            "calls": 792,
            "segment_to_call_mapping": {
                "minimum_call_index": 0,
                "maximum_call_index": 791,
                "bincount_matches_segment_counts": True,
                "segment_count_range": [
                    int(source["segment_counts"].min()),
                    int(source["segment_counts"].max()),
                ],
                "segment_call_indices_sha256": array_sha256(
                    source["segment_call_indices"].astype(np.int32)
                ),
                "segment_counts_sha256": array_sha256(
                    source["segment_counts"].astype(np.int16)
                ),
                "call_ids_sha256": string_array_sha256(source["call_ids"]),
                "source_paths_sha256": string_array_sha256(source["source_paths"]),
            },
            "shape": list(shape),
            "dtype": "float32",
            "prelast_block_output_one_based": 11,
            "last_block_input_one_based": 12,
            "special_token_indices": [0, 1],
            "patch_token_indices": [2, 145],
            "a0_reconstruction": {
                "reference": protocol["data"]["frozen_embedding_path"],
                "mean_absolute_error": reconstruction_mean_absolute,
                "maximum_absolute_error": reconstruction_absolute_maximum,
                "tolerance": tolerance,
                "passed": True,
            },
            "window_seconds": float(protocol["ast"]["segment_seconds"]),
            "hop_seconds": float(protocol["ast"]["segment_hop_seconds"]),
            "token_path": token_path.relative_to(REPO_ROOT).as_posix(),
            "token_sha256": sha256(token_path),
            "token_bytes": token_path.stat().st_size,
            "index_path": index_path.relative_to(REPO_ROOT).as_posix(),
            "index_sha256": sha256(index_path),
            "execution": {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "batch_size": args.batch_size,
                "seconds": time.perf_counter() - started,
            },
        }
        write_json(summary_path, summary)
        return summary
    except Exception:
        for path in (temporary_token, temporary_index):
            if path.exists():
                path.unlink()
        raise


def main() -> None:
    args = parse_args()
    result = preflight(args) if args.stage == "preflight" else extract(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
