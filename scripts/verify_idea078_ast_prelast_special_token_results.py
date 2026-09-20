"""CPU-only full artifact verifier for completed IDEA-078 results."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_meowagenet_idea078_ast_prelast_special_token_age_injection.py"
SPEC = importlib.util.spec_from_file_location("idea078_result_verifier_runner", RUNNER)
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    protocol = read_json(idea078.PROTOCOL_PATH)
    idea078.verify_protocol(protocol)
    run_root = ROOT / protocol["outputs"]["run_root"]
    run_manifest_path = run_root / "run_manifest.json"
    summary_path = run_root / "initial_evaluation_summary.json"
    compact_path = run_root / "run_summary.json"
    run_manifest = read_json(run_manifest_path)
    summary = read_json(summary_path)
    compact = read_json(compact_path)
    if run_manifest["protocol_sha256"] != sha256(idea078.PROTOCOL_PATH):
        raise RuntimeError("IDEA-078 run protocol hash mismatch")
    if run_manifest["runner_sha256"] != sha256(RUNNER):
        raise RuntimeError("IDEA-078 run runner hash mismatch")
    cache_manifest_path = ROOT / protocol["cache"]["manifest_path"]
    cache_manifest = read_json(cache_manifest_path)
    if sha256(cache_manifest_path) != protocol["cache"]["manifest_sha256"]:
        raise RuntimeError("IDEA-078 cache manifest hash mismatch")
    for field, name in (("token", "token_sha256"), ("index", "index_sha256")):
        path = ROOT / cache_manifest[f"{field}_path"]
        observed = sha256(path)
        if observed != cache_manifest[name] or observed != protocol["cache"][name]:
            raise RuntimeError(f"IDEA-078 cache {field} hash mismatch")

    roles = pd.read_csv(ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str})
    with np.load(ROOT / protocol["data"]["frozen_embedding_path"]) as frozen:
        call_ids = frozen["call_ids"].astype(str)
        cat_ids = frozen["cat_ids"].astype(str)
        labels = frozen["labels"].astype(np.int64)
    call_label = dict(zip(call_ids.tolist(), labels.tolist()))
    call_cat = dict(zip(call_ids.tolist(), cat_ids.tolist()))
    completed: list[dict[str, Any]] = []
    inventory: list[tuple[str, str]] = []
    expected_summary_paths: set[Path] = set()
    paired_batches_checked = 0
    validation_cells_checked = 0
    probability_rows_checked = 0
    for base_seed in idea078.BASE_SEEDS:
        for repeat in range(3):
            for fold in range(4):
                cell = roles[(roles["repeat"] == repeat) & (roles["outer_fold"] == fold)]
                validation_cats = set(
                    cell.loc[cell["role"] == "validation", "cat_id"].astype(str)
                )
                test_cats = set(cell.loc[cell["role"] == "test", "cat_id"].astype(str))
                expected_calls = {
                    call_id
                    for call_id, cat_id in zip(call_ids, cat_ids)
                    if cat_id in validation_cats
                }
                triplet: list[dict[str, Any]] = []
                for pipeline in idea078.PIPELINES:
                    fit_dir = (
                        run_root
                        / "fits"
                        / pipeline
                        / f"base_seed_{base_seed}"
                        / f"repeat_{repeat}"
                        / f"fold_{fold}"
                    )
                    fit_path = fit_dir / "fit_summary.json"
                    expected_summary_paths.add(fit_path.resolve())
                    fit = read_json(fit_path)
                    idea078.validate_completed_fit(
                        fit,
                        pipeline,
                        base_seed,
                        idea078.full_seed(base_seed, repeat, fold),
                        repeat,
                        fold,
                    )
                    if fit["outer_test_accessed"] is not False:
                        raise RuntimeError("IDEA-078 fit accessed outer test")
                    init = fit["initialization_audit"]
                    if (
                        init["trainable_parameters"] != idea078.EXPECTED_PARAMETERS
                        or init["shared_head_state_equal"] is not True
                        or init["P1_T1_injection_state_equal"] is not True
                        or any(value != 0.0 for value in init["max_initial_logit_differences_vs_A0"].values())
                        or len(set(init["initial_loss"].values())) != 1
                    ):
                        raise RuntimeError("IDEA-078 initialization audit changed")
                    audit = fit["audit"]
                    if (
                        audit["checkpoint_reload_max_probability_difference"] != 0.0
                        or audit["best_model"]["trainable_parameters"]
                        != idea078.EXPECTED_PARAMETERS[pipeline]
                        or audit["best_model"]["frozen_tail_parameters"] != 7_089_408
                        or audit["best_model"]["patch_tokens_modified"] is not False
                    ):
                        raise RuntimeError("IDEA-078 model/checkpoint audit changed")
                    injected = audit["best_model"]["injected_token_indices"]
                    if injected != ([0, 1] if pipeline == idea078.PIPELINES[2] else []):
                        raise RuntimeError("IDEA-078 injection token scope changed")
                    animal_path = ROOT / fit["validation_animal_predictions"]
                    call_path = ROOT / fit["validation_call_predictions"]
                    for path, field in (
                        (animal_path, "validation_animal_sha256"),
                        (call_path, "validation_call_sha256"),
                    ):
                        observed = sha256(path)
                        if observed != fit[field]:
                            raise RuntimeError(f"IDEA-078 prediction hash mismatch: {path}")
                        inventory.append((path.relative_to(ROOT).as_posix(), observed))
                    inventory.append((fit_path.relative_to(ROOT).as_posix(), sha256(fit_path)))
                    animals = pd.read_csv(animal_path, dtype={"cat_id": str})
                    calls = pd.read_csv(call_path, dtype={"cat_id": str, "call_id": str})
                    if (
                        set(animals["cat_id"]) != validation_cats
                        or set(calls["call_id"]) != expected_calls
                        or set(animals["cat_id"]) & test_cats
                        or set(calls["cat_id"]) & test_cats
                        or animals["cat_id"].duplicated().any()
                        or calls["call_id"].duplicated().any()
                    ):
                        raise RuntimeError("IDEA-078 validation-only prediction coverage failed")
                    if any(
                        int(row.true_label) != int(call_label[str(row.call_id)])
                        or str(row.cat_id) != call_cat[str(row.call_id)]
                        for row in calls.itertuples()
                    ):
                        raise RuntimeError("IDEA-078 call identity or labels changed")
                    probabilities = calls[list(idea078.PROBABILITY_COLUMNS)].to_numpy(float)
                    if (
                        not np.isfinite(probabilities).all()
                        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=2e-7)
                    ):
                        raise RuntimeError("IDEA-078 invalid call probabilities")
                    probability_rows_checked += len(calls)
                    triplet.append(fit)
                    completed.append(fit)
                common_epochs = min(len(item["audit"]["history"]) for item in triplet)
                for epoch in range(common_epochs):
                    reference = triplet[0]["audit"]["history"][epoch]["train_audit"]
                    for candidate in triplet[1:]:
                        current = candidate["audit"]["history"][epoch]["train_audit"]
                        for field in (
                            "cat_order_sha256",
                            "call_coverage_sha256",
                            "segment_coverage_sha256",
                        ):
                            if current[field] != reference[field]:
                                raise RuntimeError("IDEA-078 paired batch coverage changed")
                    paired_batches_checked += 1
                validation_cells_checked += 1
    observed_summary_paths = {
        path.resolve() for path in run_root.rglob("fit_summary.json")
    }
    if observed_summary_paths != expected_summary_paths or len(completed) != 108:
        raise RuntimeError("IDEA-078 fit artifact set changed")
    rebuilt = idea078.aggregate(completed, protocol)
    if summary != rebuilt or summary_path.read_bytes() != idea078.canonical_json_bytes(rebuilt):
        raise RuntimeError("IDEA-078 aggregate does not reproduce canonically")
    expected_compact = {
        "status": "complete",
        "completed_fits": 108,
        "expected_fits": 108,
        "outer_test_accessed": False,
        "main_gate_passed": rebuilt["main_gate_passed"],
        "mechanism_gate_interpretable": rebuilt["mechanism_gate_interpretable"],
        "mechanism_gate_passed": rebuilt["mechanism_gate_passed"],
        "gate_passed": rebuilt["gate_passed"],
    }
    if compact != expected_compact or compact_path.read_bytes() != idea078.canonical_json_bytes(expected_compact):
        raise RuntimeError("IDEA-078 compact summary changed")
    inventory.extend(
        (
            (path.relative_to(ROOT).as_posix(), sha256(path))
            for path in (run_manifest_path, summary_path, compact_path)
        )
    )
    inventory_payload = "".join(
        f"{path}\t{digest}\n" for path, digest in sorted(inventory)
    ).encode("utf-8")
    result = {
        "status": "PASS",
        "cpu_only": True,
        "outer_test_accessed": False,
        "completed_fits": len(completed),
        "fit_summaries_verified": 108,
        "prediction_files_verified": 216,
        "prediction_rows_checked": probability_rows_checked,
        "validation_cells_checked": validation_cells_checked,
        "paired_epoch_batch_audits_checked": paired_batches_checked,
        "initialization_audits_checked": 108,
        "checkpoint_reload_audits_checked": 108,
        "cache_manifest_sha256": sha256(cache_manifest_path),
        "cache_token_sha256": cache_manifest["token_sha256"],
        "cache_index_sha256": cache_manifest["index_sha256"],
        "run_manifest_sha256": sha256(run_manifest_path),
        "initial_evaluation_summary_sha256": sha256(summary_path),
        "run_summary_sha256": sha256(compact_path),
        "artifact_inventory_entries": len(inventory),
        "artifact_inventory_sha256": hashlib.sha256(inventory_payload).hexdigest(),
        "main_gate_passed": summary["main_gate_passed"],
        "mechanism_gate_interpretable": summary["mechanism_gate_interpretable"],
        "mechanism_gate_passed": summary["mechanism_gate_passed"],
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
