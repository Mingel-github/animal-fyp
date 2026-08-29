from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_meowagenet_idea049.py"
PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea049_backbone_screening_v1.json"
)
RECIPE_PATH = (
    REPO_ROOT
    / "configs"
    / "experiment"
    / "idea049"
    / "ssast_base_patch400_frozen_v1.json"
)
RESULT_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_idea049_ssast_initial_v1_results.json"
)
RUN_ROOT = REPO_ROOT / "runs" / "idea049_ssast_base_patch400_v1"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_idea049_runner_is_independent_and_exposes_locked_stages() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert {"load_feature_store", "aggregate_initial", "main"} <= function_names
    assert 'choices=("prepare", "smoke", "initial")' in source
    assert "animal_fyp.evaluation" in source
    assert "run_meowagenet_formal_v2_1" not in source


def test_idea049_protocol_locks_initial_scope_and_shared_head() -> None:
    protocol = load_json(PROTOCOL_PATH)
    assert protocol["relationship_to_formal_v2_1"].startswith("separate exploratory")
    assert protocol["splits"]["initial_repeats"] == [0, 1, 2]
    assert protocol["splits"]["outer_folds"] == [0, 1, 2, 3]
    assert protocol["splits"]["initial_base_seeds"] == [17]
    assert protocol["stages"]["initial"]["fold_level_fits"] == 12
    assert protocol["shared_head"]["structure"] == "embedding_dimension -> 128 -> 3"
    assert protocol["stages"]["smoke"]["outer_test_access"] is False

    recipe = load_json(RECIPE_PATH)
    assert recipe["pipeline_id"] == "ssast_hf_base_patch400_frozen_mlp"
    assert recipe["model"]["backbone"] == "frozen"
    assert recipe["model"]["hidden_size"] == 768
    assert recipe["preprocessing"]["sampling_rate_hz"] == 16000
    assert recipe["preprocessing"]["num_mel_bins"] == 128


def test_idea049_result_record_matches_completed_initial_screening() -> None:
    result = load_json(RESULT_PATH)
    assert result["status"] == "complete_for_initial_screening"
    assert result["candidate_status"] == "screened_not_better"
    assert result["scope"]["completed_fits"] == 12
    assert result["scope"]["complete_oof_evaluations"] == 3
    assert result["scope"]["calls"] == 792
    assert result["scope"]["cats"] == 111
    assert result["embedding_cache"]["shape"] == [792, 768]
    assert result["smoke"]["outer_test_accessed"] is False

    oof = result["complete_oof"]
    assert [row["repeat"] for row in oof] == [0, 1, 2]
    assert all(row["n_animals"] == 111 for row in oof)
    assert result["aggregate"]["macro_f1_mean"] == pytest.approx(
        0.6291588859015511
    )
    paired = result["paired_vs_ast_head_only"]
    assert paired["mean_difference"] == pytest.approx(-0.10452458547541876)
    assert paired["positive_complete_oof_evaluations"] == 0
    assert all(delta < 0 for delta in paired["candidate_minus_ast_macro_f1"])
    audit = result["integrity_audit"]
    assert audit["failed_fits"] == 0
    assert audit["rows_per_repeat"] == [792, 792, 792]
    assert audit["unique_cats_per_repeat"] == [111, 111, 111]
    assert audit["cat_id_partition_overlap"] is False
    assert audit["complete_oof_audit_passed"] is True
    assert result["expansion_decision"]["seeds_43_101_executed"] is False


def test_idea049_versioned_provenance_hashes_match() -> None:
    result = load_json(RESULT_PATH)
    paths = {
        "protocol": PROTOCOL_PATH,
        "recipe": RECIPE_PATH,
        "checkpoint_card": (
            REPO_ROOT / "metadata" / "models" / "idea049" / "ssast_base_patch400.json"
        ),
        "runner": RUNNER_PATH,
        "feature_manifest": RUN_ROOT / "features" / "feature_manifest.json",
        "smoke_summary": RUN_ROOT / "smoke" / "smoke_summary.json",
        "initial_summary": RUN_ROOT / "initial" / "initial_summary.json",
        "run_summary": RUN_ROOT / "initial" / "run_summary.json",
        "run_manifest": RUN_ROOT / "initial" / "run_manifest.json",
    }
    for name, path in paths.items():
        assert path.is_file(), name
        assert sha256(path) == result["provenance_sha256"][name]


def test_idea049_compact_json_audit_contains_all_twelve_fits() -> None:
    json_files = list(RUN_ROOT.rglob("*.json"))
    fit_summaries = list((RUN_ROOT / "initial" / "fits").rglob("fit_summary.json"))
    assert len(json_files) == 18
    assert len(fit_summaries) == 12

    observed = set()
    for path in fit_summaries:
        fit = load_json(path)
        assert fit["status"] == "complete"
        assert fit["base_seed"] == 17
        assert fit["inner"]["parameters"]["trainable"] == 99075
        assert fit["outer"]["parameters"]["trainable"] == 99075
        observed.add((fit["repeat"], fit["outer_fold"]))
    assert observed == {(repeat, fold) for repeat in range(3) for fold in range(4)}

    run_summary = load_json(RUN_ROOT / "initial" / "run_summary.json")
    assert run_summary["status"] == "complete"
    assert run_summary["completed_fits"] == 12
    assert run_summary["expected_fits"] == 12
    assert run_summary["complete_oof_evaluations"] == 3
