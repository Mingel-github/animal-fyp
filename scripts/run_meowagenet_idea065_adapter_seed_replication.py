"""Run the deterministic IDEA-065 AST-adapter seed replication.

This is a post-outcome robustness experiment on the existing grouped folds.  It
reuses the frozen formal-v2.1 recipe but deliberately does not modify or claim
to extend the original formal execution lock.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import pandas as pd
import tensorflow as tf
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_formal_v2_1 as formal  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea065_adapter_seed_replication_v1.json"
)
PIPELINES = ("ast_head_only", "ast_probe_guided_adapter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-subdir",
        default="meowagenet_idea065_adapter_seed_replication_v1",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_determinism() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea065-adapter-seed-replication-v1":
        raise RuntimeError("Unexpected IDEA-065 protocol")
    if protocol.get("status") != "locked_before_idea065_execution":
        raise RuntimeError("IDEA-065 protocol is not execution-locked")
    model = protocol["model"]
    if tuple(model["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-065 pipeline matrix changed")
    if model["base_seeds"] != [151, 307, 509]:
        raise RuntimeError("IDEA-065 seed bank changed")
    if model["repeats"] != [0, 1, 2] or model["outer_folds"] != [0, 1, 2, 3]:
        raise RuntimeError("IDEA-065 split scope changed")
    expected_fits = (
        len(PIPELINES)
        * len(model["base_seeds"])
        * len(model["repeats"])
        * len(model["outer_folds"])
    )
    if expected_fits != int(model["fold_level_fits"]):
        raise RuntimeError("IDEA-065 fit budget is inconsistent")
    checks = {
        REPO_ROOT / protocol["data"]["roles_path"]: protocol["data"][
            "roles_sha256"
        ],
        REPO_ROOT / model["recipe_path"]: model["recipe_sha256"],
        REPO_ROOT / protocol["dependencies"]["formal_runner_path"]: protocol[
            "dependencies"
        ]["formal_runner_sha256"],
        Path(__file__).resolve(): protocol["dependencies"]["runner_sha256"],
    }
    for path, expected_sha in checks.items():
        if not path.is_file() or formal.sha256(path) != expected_sha:
            raise RuntimeError(f"IDEA-065 dependency checksum mismatch: {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    model_protocol = protocol["model"]
    recipe_path = REPO_ROOT / model_protocol["recipe_path"]
    recipe = formal.verify_recipe(recipe_path)
    locked_model_protocol = formal.read_json(formal.LOCKED_V1_CONFIG_PATH)
    roles = pd.read_csv(formal.ROLES_PATH, dtype={"cat_id": str})
    if roles["cat_id"].nunique() != int(protocol["data"]["cats"]):
        raise RuntimeError("IDEA-065 role table cat count changed")

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as error:
        raise RuntimeError("TensorFlow device state was initialized too early") from error
    tf.config.threading.set_intra_op_parallelism_threads(min(6, os.cpu_count() or 1))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    device = formal.resolve_device(args.device)

    run_root = (formal.RUNS_ROOT / args.output_subdir).resolve()
    if formal.RUNS_ROOT.resolve() not in run_root.parents:
        raise ValueError("--output-subdir must stay below runs")
    manifest = {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "scope": "post-outcome deterministic seed replication",
        "outer_test_accessed": True,
        "protocol_path": formal.relative_repo_path(PROTOCOL_PATH),
        "protocol_sha256": formal.sha256(PROTOCOL_PATH),
        "runner_path": formal.relative_repo_path(Path(__file__).resolve()),
        "runner_sha256": formal.sha256(Path(__file__).resolve()),
        "recipe_path": formal.relative_repo_path(recipe_path),
        "recipe_sha256": formal.sha256(recipe_path),
        "git_revision_at_start": formal.git_revision(),
        "pipelines": list(PIPELINES),
        "repeats": model_protocol["repeats"],
        "folds": model_protocol["outer_folds"],
        "base_seeds": model_protocol["base_seeds"],
        "determinism": protocol["determinism"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tensorflow": tf.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU",
        },
    }
    formal.prepare_run_root(run_root, manifest, args.resume)
    store = formal.idea019.load_feature_store()
    layer_store = formal.idea019.load_layer_store(store)
    completed: list[dict[str, Any]] = []

    for repeat in model_protocol["repeats"]:
        for outer_fold in model_protocol["outer_folds"]:
            fold_roles = roles[
                (roles["repeat"] == repeat) & (roles["outer_fold"] == outer_fold)
            ]
            probe_seeds = fold_roles["inner_seed"].unique()
            if len(probe_seeds) != 1:
                raise RuntimeError("Each repeat-fold must have one probe seed")
            probe_seed = int(probe_seeds[0])
            indices = formal.ast_indices(
                store, roles, repeat, outer_fold, include_test=True
            )
            probe_result = formal.load_or_compute_probe(
                run_root,
                layer_store,
                indices["train"],
                repeat,
                outer_fold,
                probe_seed,
            )
            print(
                f"repeat={repeat} fold={outer_fold} "
                f"probe_layers={probe_result['selected_layers_one_based']}",
                flush=True,
            )
            for base_seed in model_protocol["base_seeds"]:
                seed = formal.full_model_seed(base_seed, repeat, outer_fold)
                for pipeline in PIPELINES:
                    output_dir = formal.fit_directory(
                        run_root, pipeline, repeat, outer_fold, base_seed
                    )
                    summary_path = output_dir / "fit_summary.json"
                    if summary_path.exists():
                        if not args.resume:
                            raise FileExistsError(f"Fit already exists: {summary_path}")
                        completed.append(read_json(summary_path))
                        print(f"resume: {summary_path}", flush=True)
                        continue
                    output_dir.mkdir(parents=True, exist_ok=True)
                    print(
                        f"=== {pipeline} repeat={repeat} fold={outer_fold} "
                        f"base_seed={base_seed} full_seed={seed} ===",
                        flush=True,
                    )
                    audit, predictions = formal.fit_ast(
                        "formal",
                        pipeline,
                        recipe,
                        locked_model_protocol,
                        store,
                        indices,
                        probe_result if pipeline == "ast_probe_guided_adapter" else None,
                        device,
                        seed,
                        None,
                    )
                    predictions = formal.add_prediction_identity(
                        predictions,
                        pipeline,
                        repeat,
                        outer_fold,
                        base_seed,
                        seed,
                    )
                    prediction_path = output_dir / "outer_test_unit_predictions.csv"
                    predictions.to_csv(prediction_path, index=False)
                    fit_summary = {
                        "status": "complete",
                        "scope": "idea065_seed_replication",
                        "pipeline": pipeline,
                        "repeat": int(repeat),
                        "outer_fold": int(outer_fold),
                        "base_seed": int(base_seed),
                        "full_seed": int(seed),
                        "probe_seed": probe_seed
                        if pipeline == "ast_probe_guided_adapter"
                        else None,
                        "outer_test_predictions": formal.relative_repo_path(
                            prediction_path
                        ),
                        "audit": audit,
                    }
                    formal.write_json(summary_path, fit_summary)
                    completed.append(fit_summary)

    aggregate = formal.aggregate_formal_run(
        run_root,
        recipe,
        list(PIPELINES),
        model_protocol["repeats"],
        model_protocol["outer_folds"],
        model_protocol["base_seeds"],
    )
    summary = {
        "status": "complete",
        "completed_fits": len(completed),
        "expected_fits": int(model_protocol["fold_level_fits"]),
        "formal_summary": aggregate,
    }
    formal.write_json(run_root / "run_summary.json", summary)
    return summary


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
