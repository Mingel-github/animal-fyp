from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "protocol" / "meowagenet_formal_v2_1.json"
LOCK_TEMPLATE_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_formal_v2_1_execution_lock_template.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_h_is_explained_and_mapped_to_idea_ids() -> None:
    config = load_config()
    assert config["terminology"]["H"].startswith("Hypothesis")
    assert "IDEA-048" in config["terminology"]["H048"]
    assert "IDEA-019" in config["terminology"]["H019"]


def test_parent_and_split_bank_hashes_match() -> None:
    config = load_config()
    parent = REPO_ROOT / config["parent_protocol"]["path"]
    summary = REPO_ROOT / config["split_bank"]["summary_path"]
    outer = REPO_ROOT / config["split_bank"]["outer_path"]
    roles = REPO_ROOT / config["split_bank"]["roles_path"]
    assert sha256(parent) == config["parent_protocol"]["sha256"]
    assert sha256(summary) == config["split_bank"]["summary_sha256"]
    assert sha256(outer) == config["split_bank"]["outer_sha256"]
    assert sha256(roles) == config["split_bank"]["roles_sha256"]


def test_core_is_small_and_extensions_are_optional() -> None:
    config = load_config()
    core = config["formal_core"]
    assert len(core["required_pipelines"]) == 3
    assert core["minimum_repeat_indices"] == [0, 1, 2]
    assert core["optional_extension_repeat_indices"] == [3, 4]
    assert core["minimum_fold_level_fits"] == 108
    assert len(config["optional_modules"]) == 3


def test_practical_delta_is_not_an_automatic_gate() -> None:
    config = load_config()
    assert config["frozen_evidence_core"]["practical_reference_delta_macro_f1"] == 0.03
    assert "not a mechanical pass/fail" in config["frozen_evidence_core"][
        "practical_delta_role"
    ]
    joined = " ".join(config["analysis"]["interpretation"])
    assert "does not automatically accept or reject" in joined


def test_execution_lock_is_still_a_template() -> None:
    lock = json.loads(LOCK_TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert lock["status"] == "template_not_locked"
    assert lock["formal_outcomes_accessed_before_lock"] is False
    assert lock["selected_repeat_indices"] == [0, 1, 2]
