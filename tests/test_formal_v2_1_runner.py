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
