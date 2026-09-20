"""Read-only verifier for the IDEA-078 shared pre-last-token cache."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_meowagenet_idea078_ast_prelast_special_token_age_injection.py"
SPEC = importlib.util.spec_from_file_location("idea078_cache_verifier_runner", RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load IDEA-078 runner")
idea078 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = idea078
SPEC.loader.exec_module(idea078)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(device: str, temporary: bool) -> dict[str, Any]:
    protocol = idea078.read_json(idea078.PROTOCOL_PATH)
    idea078.verify_protocol(protocol)
    cache_root = ROOT / protocol["cache"]["output_root"]
    if temporary:
        token_path = cache_root / "prelast_tokens.float32.npy.tmp"
        index_path = cache_root / "prelast_token_index.npz.tmp.npz"
    else:
        token_path = cache_root / protocol["cache"]["token_filename"]
        index_path = cache_root / protocol["cache"]["index_filename"]
    tokens = np.load(token_path, mmap_mode="r")
    with np.load(index_path, allow_pickle=False) as index:
        keys = list(index.files)
        mapping = index["segment_call_indices"].astype(np.int64)
        counts = index["segment_counts"].astype(np.int64)
        call_ids = index["call_ids"].astype(str)
        source_paths = index["source_paths"].astype(str)
    with np.load(ROOT / protocol["data"]["fbank_path"], allow_pickle=False) as fbank:
        mapping_checks = {
            "segment_call_indices": bool(
                np.array_equal(mapping, fbank["segment_call_indices"].astype(np.int64))
            ),
            "segment_counts": bool(
                np.array_equal(counts, fbank["segment_counts"].astype(np.int64))
            ),
            "call_ids": bool(np.array_equal(call_ids, fbank["call_ids"].astype(str))),
            "source_paths": bool(
                np.array_equal(source_paths, fbank["source_paths"].astype(str))
            ),
        }
    with np.load(ROOT / protocol["data"]["frozen_embedding_path"]) as frozen:
        locked = frozen["embeddings"].astype(np.float32)
    target = torch.device(device)
    model = idea078.load_ast_model(protocol, target)
    tail = idea078.take_tail(model).to(target)
    sums = np.zeros((792, 768), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, 843, 16):
            stop = min(start + 16, 843)
            batch = torch.from_numpy(
                np.array(tokens[start:stop], dtype=np.float32, copy=True)
            ).to(target)
            pooled = tail(batch).float().cpu().numpy()
            np.add.at(sums, mapping[start:stop], pooled.astype(np.float64))
    reconstructed = (sums / counts.astype(np.float64)[:, None]).astype(np.float32)
    difference = np.abs(reconstructed - locked)
    tolerance = protocol["preflight"]["full_cache_vs_locked_final_tolerance"]
    result = {
        "status": "PASS",
        "read_only": True,
        "temporary_artifacts": temporary,
        "device": str(target),
        "token_path": token_path.relative_to(ROOT).as_posix(),
        "token_sha256": sha256(token_path),
        "token_bytes": token_path.stat().st_size,
        "token_shape": list(tokens.shape),
        "token_dtype": str(tokens.dtype),
        "index_path": index_path.relative_to(ROOT).as_posix(),
        "index_sha256": sha256(index_path),
        "index_bytes": index_path.stat().st_size,
        "index_keys": keys,
        "mapping_checks": mapping_checks,
        "mapping_closed": bool(
            np.array_equal(np.bincount(mapping, minlength=792), counts)
        ),
        "mean_absolute_error": float(difference.mean()),
        "maximum_absolute_error": float(difference.max()),
        "tolerance": tolerance,
    }
    if list(tokens.shape) != [843, 146, 768] or tokens.dtype != np.float32:
        raise RuntimeError("IDEA-078 token geometry changed")
    if not all(mapping_checks.values()) or not result["mapping_closed"]:
        raise RuntimeError("IDEA-078 mapping verification failed")
    if (
        result["mean_absolute_error"] > float(tolerance["mean_absolute_maximum"])
        or result["maximum_absolute_error"] > float(tolerance["absolute_maximum"])
    ):
        raise RuntimeError(f"IDEA-078 reconstruction failed: {result}")
    del tokens
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--temporary", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.device, args.temporary), indent=2), flush=True)


if __name__ == "__main__":
    main()
