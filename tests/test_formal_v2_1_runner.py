from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_meowagenet_formal_v2_1.py"
RECIPE_PATH = (
    REPO_ROOT
    / "configs"
    / "experiment"
    / "meowagenet_formal_v2_1_probe_guided_candidate_v1.json"
)
LOCK_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_formal_v2_1_execution_lock.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_recipe() -> dict[str, object]:
    return json.loads(RECIPE_PATH.read_text(encoding="utf-8"))


def test_formal_runner_is_valid_python_and_has_both_scopes() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "verify_execution_lock" in function_names
    assert "fit_vggish" in function_names
    assert "fit_ast" in function_names
    assert "hierarchical_paired_bootstrap" in function_names
    assert 'choices=("inner-only", "formal")' in source
    assert "Formal scope requires --execution-lock" in source
    assert 'include_test=args.scope == "formal"' in source


def test_candidate_recipe_resolves_pooling_and_core_pipeline_roles() -> None:
    recipe = load_recipe()
    assert recipe["status"] == "candidate_recipe_before_execution_lock"
    assert recipe["core_pipelines"] == [
        "vggish_mlp",
        "ast_head_only",
        "ast_probe_guided_adapter",
    ]
    pooling = recipe["ast_shared"]["segment_to_call_pooling"]
    assert "embeddings within each call before the classifier head" in pooling
    assert "or probabilities" not in pooling
    assert recipe["analysis"]["hierarchical_bootstrap_repeats"] == 10000


def test_candidate_recipe_versioned_dependency_hashes_match() -> None:
    recipe = load_recipe()
    for entry in recipe["versioned_dependencies"]:
        path = REPO_ROOT / entry["path"]
        assert path.is_file()
        assert sha256(path) == entry["sha256"]


def test_completed_execution_lock_matches_runner_and_recipe() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["status"] == "locked_before_formal_outcomes"
    assert lock["formal_outcomes_accessed_before_lock"] is False
    assert lock["selected_repeat_indices"] == [0, 1, 2]
    assert lock["model_seeds"] == [17, 43, 101]
    assert lock["enabled_optional_modules"] == []
    assert lock["formal_core"]["fold_level_fits"] == 108
    assert sha256(RUNNER_PATH) == lock["runner"]["sha256"]
    assert sha256(RECIPE_PATH) == lock["selected_primary_adapter"]["recipe_sha256"]
    assert len(lock["runner"]["git_revision"]) == 40

    def contains_null(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, dict):
            return any(contains_null(child) for child in value.values())
        if isinstance(value, list):
            return any(contains_null(child) for child in value)
        return False

    assert not contains_null(lock)
