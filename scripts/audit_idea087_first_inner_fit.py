"""Independent technical audit for the first authorized IDEA-087 inner fit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "meowagenet_idea087_nested_hpo_v1"
FIT_PATH = (
    RUN_ROOT
    / "inner"
    / "A0_ast_only"
    / "q00"
    / "outer_0"
    / "inner_0"
    / "search_seed_1314160575"
    / "fit_summary.json"
)
OUTPUT_PATH = RUN_ROOT / "first_inner_fit_audit.json"
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
LOCKED = {
    "protocol_sha256": "ff74e49579afa5a6cad6d4e1ad14894817286924a84af78787dc00191ace2786",
    "runner_sha256": "6152100d5a2e53420aa4877674a63d4163b8cc7a26d157d227491b35e1d57fe7",
    "tests_sha256": "8ae5687f96abdb0fc1d587d2ab7b8e26cec582118dae34db94cba8b8da089ee1",
    "inner_roles_sha256": "70a2f95d65a9a1b584e6d5b45ee9eb450d2c535c3e8c44a4bc9ae0fdf7fd3c11",
    "cpu_preflight_sha256": "9f0b3c494679fbcc41deaafc7ea99a577a9990ff96906d618876b414779253be",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def validate_probabilities(frame: pd.DataFrame, identity: str) -> None:
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"nonfinite probabilities: {identity}")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise RuntimeError(f"out-of-range probabilities: {identity}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0):
        raise RuntimeError(f"unnormalized probabilities: {identity}")
    if "predicted_label" in frame and not np.array_equal(
        frame["predicted_label"].to_numpy(dtype=np.int64), probabilities.argmax(axis=1)
    ):
        raise RuntimeError(f"argmax mismatch: {identity}")


def main() -> None:
    summaries = list((RUN_ROOT / "inner").rglob("fit_summary.json"))
    if summaries != [FIT_PATH]:
        raise RuntimeError(f"expected exactly the first fit, found {len(summaries)}")
    if (RUN_ROOT / "outer").exists() or (RUN_ROOT / "selection" / "selection_lock.json").exists():
        raise RuntimeError("outer or selection artifacts exist during first-fit audit")
    fit = read_json(FIT_PATH)
    expected_identity = {
        "status": "complete",
        "stage": "inner",
        "pipeline": "A0_ast_only",
        "config_id": "q00",
        "outer_fold": 0,
        "inner_fold": 0,
        "search_base_seed": 1314160575,
        "full_seed": 1314160575,
        "outer_test_accessed": False,
    }
    for key, expected in expected_identity.items():
        if fit.get(key) != expected:
            raise RuntimeError(f"identity mismatch: {key}")
    for key in ("protocol_sha256", "runner_sha256", "tests_sha256"):
        if fit.get(key) != LOCKED[key]:
            raise RuntimeError(f"locked hash mismatch: {key}")
    if sha256(RUN_ROOT / "inner_roles.csv") != LOCKED["inner_roles_sha256"]:
        raise RuntimeError("inner roles changed")
    if sha256(RUN_ROOT / "cpu_preflight.json") != LOCKED["cpu_preflight_sha256"]:
        raise RuntimeError("CPU preflight changed")
    manifest = read_json(RUN_ROOT / "run_manifest.json")
    for key in ("protocol_sha256", "runner_sha256", "tests_sha256"):
        if manifest.get(key) != LOCKED[key]:
            raise RuntimeError(f"manifest hash mismatch: {key}")
    if manifest.get("outer_test_accessed_at_manifest_creation") is not False:
        raise RuntimeError("manifest outer access flag changed")
    if manifest["environment"]["device"] != "cuda" or not manifest["environment"]["gpu"]:
        raise RuntimeError("first fit did not use the authorized CUDA device")

    roles = pd.read_csv(RUN_ROOT / "inner_roles.csv", dtype={"cat_id": str})
    cell = roles[(roles["outer_fold"] == 0) & (roles["inner_fold"] == 0)]
    train_cats = set(cell[cell["role"] == "train"]["cat_id"].astype(str))
    validation_cats = set(cell[cell["role"] == "validation"]["cat_id"].astype(str))
    if train_cats & validation_cats or len(train_cats) != 55 or len(validation_cats) != 28:
        raise RuntimeError("first-fit cat roles changed")
    animal_path = ROOT / fit["prediction_animal_path"]
    call_path = ROOT / fit["prediction_call_path"]
    if sha256(animal_path) != fit["prediction_animal_sha256"] or sha256(call_path) != fit["prediction_call_sha256"]:
        raise RuntimeError("first-fit prediction hash mismatch")
    animals = pd.read_csv(animal_path, dtype={"cat_id": str})
    calls = pd.read_csv(call_path, dtype={"cat_id": str, "call_id": str})
    validate_probabilities(animals, "animals")
    validate_probabilities(calls, "calls")
    if set(animals["cat_id"].astype(str)) != validation_cats or set(calls["cat_id"].astype(str)) != validation_cats:
        raise RuntimeError("first-fit predictions contain the wrong cats")
    if set(calls["cat_id"].astype(str)) & train_cats:
        raise RuntimeError("training cats leaked into validation predictions")
    reconstructed_rows = []
    for cat_id, group in calls.groupby("cat_id", sort=True):
        labels = group["true_label"].unique()
        if len(labels) != 1:
            raise RuntimeError("inconsistent call labels")
        probabilities = group[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64).mean(axis=0)
        reconstructed_rows.append(
            {
                "cat_id": str(cat_id),
                "true_label": int(labels[0]),
                "call_count": int(len(group)),
                **{column: float(probabilities[index]) for index, column in enumerate(PROBABILITY_COLUMNS)},
                "predicted_label": int(probabilities.argmax()),
            }
        )
    reconstructed = pd.DataFrame(reconstructed_rows).sort_values("cat_id").reset_index(drop=True)
    saved = animals.sort_values("cat_id").reset_index(drop=True)
    for column in ("cat_id", "true_label", "call_count", "predicted_label"):
        if not np.array_equal(saved[column].to_numpy(), reconstructed[column].to_numpy()):
            raise RuntimeError(f"call-to-cat mismatch: {column}")
    if not np.allclose(
        saved[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64),
        reconstructed[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64),
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise RuntimeError("call-to-cat probability mismatch")

    audit = fit["audit"]
    history = audit["history"]
    if audit["best_epoch"] != 14 or audit["stopped_epoch"] != 22:
        raise RuntimeError("first-fit epoch identity changed")
    if audit["checkpoint_reload_applicable"] is not True or audit["best_state_reload_max_probability_difference"] != 0.0:
        raise RuntimeError("first-fit checkpoint reload failed")
    if audit["model"]["trainable_parameters"] != 99075:
        raise RuntimeError("first-fit parameter count changed")
    if [row["epoch"] for row in history] != list(range(1, 23)):
        raise RuntimeError("first-fit history is incomplete")
    validation_losses = [row["validation_animal_cross_entropy"] for row in history]
    if int(np.argmin(validation_losses)) + 1 != audit["best_epoch"]:
        raise RuntimeError("first-fit best epoch is not minimum validation CE")
    for row in history:
        train_audit = row["train_audit"]
        if train_audit["cats"] != 55 or train_audit["calls"] != 456:
            raise RuntimeError("first-fit epoch coverage changed")
        if sum(train_audit["batch_sizes_cats"]) != 55:
            raise RuntimeError("first-fit batch sizes do not cover training cats")
        if train_audit["call_coverage_sha256"] != fit["train_role"]["call_indices_sha256"]:
            raise RuntimeError("first-fit call coverage hash changed")

    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "fit_summary_path": FIT_PATH.relative_to(ROOT).as_posix(),
        "fit_summary_sha256": sha256(FIT_PATH),
        "prediction_animal_sha256": sha256(animal_path),
        "prediction_call_sha256": sha256(call_path),
        "identity": expected_identity,
        "device": manifest["environment"],
        "train_cats": 55,
        "train_calls": 456,
        "validation_cats": 28,
        "validation_calls": int(len(calls)),
        "best_epoch": 14,
        "stopped_epoch": 22,
        "checkpoint_reload_max_probability_difference": 0.0,
        "epoch_coverage_checked": 22,
        "call_to_cat_reconstruction": "PASS",
        "probability_and_argmax_checks": "PASS",
        "inner_fit_summaries_present": 1,
        "selection_lock_present": False,
        "outer_artifacts_present": False,
        "outer_test_accessed": False,
    }
    write_json(OUTPUT_PATH, result)
    print(json.dumps({**result, "audit_sha256": sha256(OUTPUT_PATH)}, indent=2))


if __name__ == "__main__":
    main()
