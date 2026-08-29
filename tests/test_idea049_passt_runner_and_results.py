from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_meowagenet_idea049_passt.py"
ENGINE_PATH = REPO_ROOT / "scripts" / "run_meowagenet_idea049.py"
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
    / "passt_s_ap476_frozen_v1.json"
)
CARD_PATH = REPO_ROOT / "metadata" / "models" / "idea049" / "passt_s_ap476.json"
RESULT_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_idea049_passt_initial_v1_results.json"
)
ENVIRONMENT_PATH = REPO_ROOT / "environment" / "meowagenet-repro.yml"
RUN_ROOT = REPO_ROOT / "runs" / "idea049_passt_s_ap476_v1"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_passt_runner_is_independent_and_reuses_frozen_feature_engine() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert {
        "load_passt_model",
        "prepare_embeddings",
        "load_feature_store",
        "run_smoke",
        "aggregate_initial",
        "run_initial",
        "main",
    } <= function_names
    assert 'choices=("prepare", "smoke", "initial")' in source
    assert "import run_meowagenet_idea049 as engine" in source
    assert "run_meowagenet_formal_v2_1" not in source
    assert "MINIMUM_AUDIO_SAMPLES = 5_120" in source
    assert '"outer_test_accessed": False' in source


def test_passt_recipe_and_card_lock_official_runtime_identity() -> None:
    recipe = load_json(RECIPE_PATH)
    card = load_json(CARD_PATH)
    assert recipe["pipeline_id"] == "passt_s_ap476_frozen_mlp"
    assert recipe["model"]["runtime_arch"] == "passt_s_swa_p16_128_ap476"
    assert recipe["model"]["runtime_package_version"] == "0.0.26"
    assert recipe["model"]["checkpoint_bytes"] == 344_665_734
    assert recipe["model"]["checkpoint_sha256"] == (
        "302903fa8c4aee817b11dc982da0b29aaf8d11a3e722420476d0a12c9db70c2c"
    )
    assert recipe["model"]["wrapper_mode"] == "embed_only"
    assert recipe["model"]["hidden_size"] == 768
    assert recipe["preprocessing"]["sampling_rate_hz"] == 32_000
    assert recipe["preprocessing"]["clips_below_160ms"] == 11
    assert recipe["preprocessing"]["extraction_batch_size"] == 1
    assert card["official_code"]["license"] == "Apache-2.0"
    assert card["output"]["audioset_logits_excluded"] is True
    assert card["local_forward_audit"]["output_shape"] == [1, 768]


def test_passt_result_record_matches_completed_initial_screening() -> None:
    result = load_json(RESULT_PATH)
    assert result["status"] == "complete_for_initial_screening"
    assert result["candidate_status"] == "screened_not_better"
    assert result["scope"]["completed_fits"] == 12
    assert result["scope"]["complete_oof_evaluations"] == 3
    assert result["embedding_cache"]["shape"] == [792, 768]
    assert result["embedding_cache"]["short_calls_padded_to_160ms"] == 11
    assert result["smoke"]["outer_test_accessed"] is False

    oof = result["complete_oof"]
    assert [row["repeat"] for row in oof] == [0, 1, 2]
    assert all(row["n_animals"] == 111 for row in oof)
    assert result["aggregate"]["macro_f1_mean"] == pytest.approx(
        0.6559525667680349
    )
    assert result["aggregate"]["macro_f1_sample_sd"] == pytest.approx(
        0.021695796686076674
    )
    ast = result["contrasts"]["passt_minus_ast_head_only_macro_f1"]
    assert ast["mean_difference"] == pytest.approx(-0.0777309046089349)
    assert ast["positive_complete_oof_evaluations"] == 0
    assert all(delta < 0 for delta in ast["paired_differences"])
    ssast = result["contrasts"]["passt_minus_ssast_macro_f1"]
    assert ssast["mean_difference"] == pytest.approx(0.026793680866483816)
    assert result["expansion_decision"]["seeds_43_101_executed"] is False


def test_passt_versioned_provenance_hashes_match() -> None:
    result = load_json(RESULT_PATH)
    paths = {
        "protocol": PROTOCOL_PATH,
        "recipe": RECIPE_PATH,
        "checkpoint_card": CARD_PATH,
        "runner": RUNNER_PATH,
        "training_engine": ENGINE_PATH,
        "environment_recipe": ENVIRONMENT_PATH,
        "feature_manifest": RUN_ROOT / "features" / "feature_manifest.json",
        "smoke_summary": RUN_ROOT / "smoke" / "smoke_summary.json",
        "initial_summary": RUN_ROOT / "initial" / "initial_summary.json",
        "run_summary": RUN_ROOT / "initial" / "run_summary.json",
        "run_manifest": RUN_ROOT / "initial" / "run_manifest.json",
    }
    for name, path in paths.items():
        assert path.is_file(), name
        assert sha256(path) == result["provenance_sha256"][name]


def test_passt_compact_json_audit_contains_all_twelve_fits() -> None:
    result = load_json(RESULT_PATH)
    json_files = list(RUN_ROOT.rglob("*.json"))
    fit_summaries = list((RUN_ROOT / "initial" / "fits").rglob("fit_summary.json"))
    assert len(json_files) == result["versioned_json_audit"]["files"] == 18
    assert sum(path.stat().st_size for path in json_files) == (
        result["versioned_json_audit"]["bytes"]
    )
    assert len(fit_summaries) == 12

    observed = set()
    for path in fit_summaries:
        fit = load_json(path)
        assert fit["status"] == "complete"
        assert fit["base_seed"] == 17
        assert fit["inner"]["parameters"]["trainable"] == 99_075
        assert fit["outer"]["parameters"]["trainable"] == 99_075
        observed.add((fit["repeat"], fit["outer_fold"]))
    assert observed == {(repeat, fold) for repeat in range(3) for fold in range(4)}

    audit = result["integrity_audit"]
    assert audit["failed_fits"] == 0
    assert audit["rows_per_repeat"] == [792, 792, 792]
    assert audit["unique_cats_per_repeat"] == [111, 111, 111]
    assert audit["cat_id_partition_overlap"] is False
    assert audit["complete_oof_audit_passed"] is True
