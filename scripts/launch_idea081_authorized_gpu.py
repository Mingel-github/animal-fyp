"""Audited authorization shim for the immutable IDEA-081 formal runner.

The locked protocol intentionally records ``current_gpu_allowed=false``.  This
shim preserves the locked protocol and runner bytes, verifies the separately
recorded research-director authorization, then changes that one execution flag
only in the in-memory protocol object returned to the locked runner.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_idea081_ast_tail_convpass_c1_factorial.py"
PROTOCOL_PATH = ROOT / "configs" / "protocol" / "meowagenet_idea081_ast_tail_convpass_c1_factorial_v1.json"
AUTHORIZATION_PATH = ROOT / "runs" / "meowagenet_idea081_ast_tail_convpass_c1_factorial_v1" / "gpu_authorization.json"
LOCKED_PROTOCOL_SHA256 = "50d30df1ece31dacfa33bc33307f8c7992b2d2f0c5d9863b08214821c0dbccd4"
LOCKED_RUNNER_SHA256 = "def87e10852506c3ac26d1133a232594f92768d9d81912d47de1d47227f97903"
MINIMUM_FREE_GPU_BYTES = 5 * 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_locked_runner():
    spec = importlib.util.spec_from_file_location("idea081_locked_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import locked IDEA-081 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if sha256(PROTOCOL_PATH) != LOCKED_PROTOCOL_SHA256:
        raise RuntimeError("IDEA-081 locked protocol hash changed")
    if sha256(RUNNER_PATH) != LOCKED_RUNNER_SHA256:
        raise RuntimeError("IDEA-081 locked runner hash changed")
    authorization = read_json(AUTHORIZATION_PATH)
    expected = {
        "status": "AUTHORIZED_FOR_FORMAL_GPU_RUN",
        "source_thread_id": "01a0a8e4-831f-7872-9f7a-c04ca7ea0f02",
        "protocol_sha256": LOCKED_PROTOCOL_SHA256,
        "runner_sha256": LOCKED_RUNNER_SHA256,
        "base_seeds": [9217, 7339, 4211],
        "expected_fits": 144,
        "outer_test_predictions": False,
        "resume_required": True,
        "first_complete_fold_pause_required": True,
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise RuntimeError("IDEA-081 GPU authorization artifact is invalid")
    if authorization.get("launcher_sha256") != sha256(Path(__file__).resolve()):
        raise RuntimeError("IDEA-081 authorization launcher hash mismatch")
    protocol = read_json(PROTOCOL_PATH)
    if protocol["execution"].get("current_gpu_allowed") is not False:
        raise RuntimeError("Locked protocol was unexpectedly modified")
    if protocol["model"]["outer_test_predictions"] is not False:
        raise RuntimeError("IDEA-081 outer-test lock changed")
    if not torch.cuda.is_available():
        raise RuntimeError("IDEA-081 GPU authorization received but CUDA is unavailable")
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if free_bytes < MINIMUM_FREE_GPU_BYTES:
        raise RuntimeError(f"IDEA-081 requires >=5 GiB free GPU memory, got {free_bytes}")

    runner = load_locked_runner()
    original_read_json = runner.read_json

    def authorized_read_json(path: Path) -> dict:
        value = original_read_json(path)
        if Path(path).resolve() == runner.PROTOCOL_PATH.resolve():
            value = copy.deepcopy(value)
            value["execution"]["current_gpu_allowed"] = True
            value["execution"]["authorization_artifact"] = runner.repo_relative(
                AUTHORIZATION_PATH
            )
        return value

    runner.read_json = authorized_read_json
    run_args = argparse.Namespace(
        stage="run",
        output_subdir="meowagenet_idea081_ast_tail_convpass_c1_factorial_v1",
        device="cuda",
        resume=bool(args.resume),
    )
    runner.run(run_args)


if __name__ == "__main__":
    main()
