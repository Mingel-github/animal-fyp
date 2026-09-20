from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR_PATH = ROOT / "scripts" / "audit_idea082_protocol_preflight.py"


def load_auditor():
    spec = importlib.util.spec_from_file_location("idea082_independent_auditor", AUDITOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDITOR = load_auditor()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def make_candidate(tmp_path: Path):
    runner = tmp_path / "runner.py"
    runner.write_text(
        "\n".join(
            [
                "train_indices = []",
                "nanmedian = None",
                "age_train = None",
                "cat_id = None",
                "permutation = None",
                "outer_test_accessed = False",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.json"
    write_json(
        protocol,
        {
            "acoustic_features": {
                "feature_names": list(AUDITOR.EXPECTED_FEATURES),
                "feature_cache_sha256": AUDITOR.EXPECTED_HASHES["feature_cache"],
                "groups": {
                    group: {"indices": list(indices)}
                    for group, indices in AUDITOR.EXPECTED_GROUPS.items()
                },
            },
            "data": {"roles_sha256": AUDITOR.EXPECTED_HASHES["roles"]},
            "screen": {"outer_test_predictions": False},
        },
    )
    preflight = tmp_path / "preflight.json"
    write_json(
        preflight,
        {
            "status": "GO_FOR_GPU",
            "protocol_sha256": sha256(protocol),
            "runner_sha256": sha256(runner),
            "feature_cache_sha256": AUDITOR.EXPECTED_HASHES["feature_cache"],
            "roles_sha256": AUDITOR.EXPECTED_HASHES["roles"],
            "outer_test_accessed": False,
            "training_preprocessing_only": True,
            "cat_roles_disjoint": True,
            "common_initialization_equal": True,
            "same_batch_order": True,
            "shuffle_within_role": True,
            "shuffle_deterministic": True,
            "shuffle_cross_role": False,
            "shuffle_train_permutation_sha256": "1" * 64,
            "shuffle_validation_permutation_sha256": "2" * 64,
            "paired_seed_repeat_fold_cells": True,
            "animal_level_metrics": True,
        },
    )
    return protocol, preflight, runner


def test_reference_partition_is_exact_5_10_5_and_g12_native_order_union():
    assert tuple(map(len, (AUDITOR.EXPECTED_GROUPS["G1"], AUDITOR.EXPECTED_GROUPS["G2"], AUDITOR.EXPECTED_GROUPS["G3"]))) == (5, 10, 5)
    assert AUDITOR.EXPECTED_GROUPS["G12"] == tuple(range(15))
    assert set(AUDITOR.EXPECTED_GROUPS["G12"]) == set(AUDITOR.EXPECTED_GROUPS["G1"]) | set(AUDITOR.EXPECTED_GROUPS["G2"])
    combined = AUDITOR.EXPECTED_GROUPS["G1"] + AUDITOR.EXPECTED_GROUPS["G2"] + AUDITOR.EXPECTED_GROUPS["G3"]
    assert len(combined) == len(set(combined)) == 20
    assert set(combined) == set(range(20))


def test_complete_final_evidence_passes():
    protocol = ROOT / "configs" / "protocol" / "meowagenet_idea082_age_acoustic_group_ablation_v1.json"
    preflight = ROOT / "runs" / "meowagenet_idea082_age_acoustic_group_ablation_v1" / "cpu_preflight.json"
    runner = ROOT / "scripts" / "run_meowagenet_idea082_age_acoustic_group_ablation.py"
    result = AUDITOR.audit(protocol, preflight, runner)
    assert result["status"] == "GO", result["failed_checks"]
    assert result["checks_passed"] == result["checks_total"]
    assert result["gpu_used"] is False
    assert result["outer_test_accessed"] is False


def test_wrong_group_member_is_no_go(tmp_path: Path):
    protocol, preflight, runner = make_candidate(tmp_path)
    value = json.loads(protocol.read_text(encoding="utf-8"))
    value["acoustic_features"]["groups"]["G1"]["indices"] = [0, 1, 2, 3, 4]
    write_json(protocol, value)
    preflight_value = json.loads(preflight.read_text(encoding="utf-8"))
    preflight_value["protocol_sha256"] = sha256(protocol)
    write_json(preflight, preflight_value)
    result = AUDITOR.audit(protocol, preflight, runner)
    assert result["status"] == "NO_GO"
    assert "protocol_g1_indices" in result["failed_checks"]


def test_cross_role_shuffle_without_explicit_rejection_is_no_go(tmp_path: Path):
    protocol, preflight, runner = make_candidate(tmp_path)
    value = json.loads(preflight.read_text(encoding="utf-8"))
    value["shuffle_cross_role"] = True
    write_json(preflight, value)
    result = AUDITOR.audit(protocol, preflight, runner)
    assert result["status"] == "NO_GO"
    assert "shuffled_control_no_cross_role" in result["failed_checks"]


def test_missing_role_permutation_hash_is_no_go(tmp_path: Path):
    protocol, preflight, runner = make_candidate(tmp_path)
    value = json.loads(preflight.read_text(encoding="utf-8"))
    del value["shuffle_validation_permutation_sha256"]
    write_json(preflight, value)
    result = AUDITOR.audit(protocol, preflight, runner)
    assert result["status"] == "NO_GO"
    assert "shuffled_control_permutation_hashes" in result["failed_checks"]
