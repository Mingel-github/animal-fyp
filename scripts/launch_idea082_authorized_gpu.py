"""Authorization and first-cell pause shim for the locked IDEA-082 runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_meowagenet_idea082_age_acoustic_group_ablation.py"
PROTOCOL_PATH = ROOT / "configs" / "protocol" / "meowagenet_idea082_age_acoustic_group_ablation_v1.json"
TESTS_PATH = ROOT / "tests" / "test_idea082_age_acoustic_group_ablation.py"
RUN_ROOT = ROOT / "runs" / "meowagenet_idea082_age_acoustic_group_ablation_v1"
PREFLIGHT_PATH = RUN_ROOT / "cpu_preflight.json"
AUTHORIZATION_PATH = RUN_ROOT / "gpu_authorization.json"
PAUSE_PATH = RUN_ROOT / "first_complete_cell_pause.json"
LOCKED_PROTOCOL_SHA256 = "0b5e4a4c0be591925811a42769555cab7e5ac9ba8cd2781ac91d1d341e782022"
LOCKED_RUNNER_SHA256 = "2ce03190d3079cbfe34fbdd140288b2309dfa12ace54f8d21279e66fa7f2c415"
LOCKED_TESTS_SHA256 = "b91e4f02a28a98aa6aa7223c4dbc0f5a7a37b9f93f81c3e60f3f697e94fdb00a"
LOCKED_PREFLIGHT_SHA256 = "b0967358ed610ce41441cf3bb7bde2cbaa950d38727bb2359eeef7724d783952"
MINIMUM_FREE_GPU_BYTES = 5 * 1024**3


class FirstCellComplete(RuntimeError):
    """Internal sentinel raised before the second cell starts."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_runner():
    spec = importlib.util.spec_from_file_location("idea082_locked_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import locked IDEA-082 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--first-cell-only", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise RuntimeError("IDEA-082 formal GPU launch requires --resume")
    locked = {
        PROTOCOL_PATH: LOCKED_PROTOCOL_SHA256,
        RUNNER_PATH: LOCKED_RUNNER_SHA256,
        TESTS_PATH: LOCKED_TESTS_SHA256,
        PREFLIGHT_PATH: LOCKED_PREFLIGHT_SHA256,
    }
    for path, expected in locked.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"IDEA-082 locked artifact changed: {path}")
    authorization = read_json(AUTHORIZATION_PATH)
    expected_authorization = {
        "status": "AUTHORIZED_FOR_FORMAL_GPU_RUN",
        "source_thread_id": "01a0a8e4-831f-7872-9f7a-c04ca7ea0f02",
        "protocol_sha256": LOCKED_PROTOCOL_SHA256,
        "runner_sha256": LOCKED_RUNNER_SHA256,
        "tests_sha256": LOCKED_TESTS_SHA256,
        "cpu_preflight_sha256": LOCKED_PREFLIGHT_SHA256,
        "base_seeds": [8694, 5378, 5945],
        "pipelines": 9,
        "expected_fits": 324,
        "outer_test_predictions": False,
        "resume_required": True,
        "first_complete_cell_pause_required": True,
    }
    if any(
        authorization.get(key) != value
        for key, value in expected_authorization.items()
    ):
        raise RuntimeError("IDEA-082 GPU authorization artifact is invalid")
    if authorization.get("launcher_sha256") != sha256(Path(__file__).resolve()):
        raise RuntimeError("IDEA-082 authorization launcher hash mismatch")
    preflight = read_json(PREFLIGHT_PATH)
    if preflight.get("status") != "GO" or preflight.get(
        "outer_test_predictions_or_metrics_accessed"
    ) is not False:
        raise RuntimeError("IDEA-082 CPU preflight is not a clean GO")
    if not torch.cuda.is_available():
        raise RuntimeError("IDEA-082 authorization received but CUDA is unavailable")
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if free_bytes < MINIMUM_FREE_GPU_BYTES:
        raise RuntimeError(
            f"IDEA-082 requires >=5 GiB free GPU memory, got {free_bytes}"
        )
    runner = load_runner()
    if args.first_cell_only:
        original_initial_audit = runner.initial_model_audit
        calls = 0

        def pause_before_second_cell(*call_args, **call_kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise FirstCellComplete("First IDEA-082 cell completed; audit required")
            return original_initial_audit(*call_args, **call_kwargs)

        runner.initial_model_audit = pause_before_second_cell
    run_args = argparse.Namespace(
        stage="run",
        output_subdir="meowagenet_idea082_age_acoustic_group_ablation_v1",
        device="cuda",
        resume=True,
        director_authorized=True,
    )
    try:
        runner.run(run_args)
    except FirstCellComplete:
        write_json(
            PAUSE_PATH,
            {
                "status": "PAUSED_FOR_MANDATORY_FIRST_CELL_AUDIT",
                "complete_cell": {"base_seed": 8694, "repeat": 0, "fold": 0},
                "expected_complete_fits": 9,
                "outer_test_accessed": False,
                "protocol_sha256": LOCKED_PROTOCOL_SHA256,
                "runner_sha256": LOCKED_RUNNER_SHA256,
            },
        )
        print(PAUSE_PATH.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()

