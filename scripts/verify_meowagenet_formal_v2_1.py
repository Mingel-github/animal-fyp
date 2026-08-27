"""Verify the formal-v2.1 amendment and create its immutable audit record."""

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
REPORT_PATH = REPO_ROOT / "reports" / "10_formal_protocol_v2_1_amendment.md"
RECORD_PATH = (
    REPO_ROOT / "metadata" / "experiments" / "meowagenet_formal_v2_1_amendment.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_path(value: str) -> Path:
    return REPO_ROOT / Path(value)


def write_or_verify(path: Path, content: str) -> None:
    encoded = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(
                f"Amendment record differs: {path}. Preserve it and create a later version."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def check_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    lock_template = json.loads(LOCK_TEMPLATE_PATH.read_text(encoding="utf-8"))
    if config["protocol_id"] != "meowagenet-formal-v2.1":
        raise ValueError("Unexpected protocol ID")
    if config["status"] != "core_frozen_execution_scope_pending":
        raise ValueError("Unexpected amendment status")
    if config["prior_formal_outcomes_accessed"] is not False:
        raise ValueError("The amendment must precede formal-v2.1 outcomes")
    if not config["terminology"]["H"].startswith("Hypothesis"):
        raise ValueError("H must be defined as Hypothesis at first use")
    if lock_template["status"] != "template_not_locked":
        raise ValueError("Execution-lock template must not masquerade as a completed lock")

    parent = config["parent_protocol"]
    check_hash(repo_path(parent["path"]), parent["sha256"])
    split_bank = config["split_bank"]
    check_hash(repo_path(split_bank["summary_path"]), split_bank["summary_sha256"])
    check_hash(repo_path(split_bank["outer_path"]), split_bank["outer_sha256"])
    check_hash(repo_path(split_bank["roles_path"]), split_bank["roles_sha256"])

    minimum = set(config["formal_core"]["minimum_repeat_indices"])
    available = set(config["split_bank"]["available_repeat_indices"])
    if not minimum or not minimum <= available:
        raise ValueError("Minimum repeat indices must be a non-empty subset of split bank")
    if len(config["formal_core"]["required_pipelines"]) != 3:
        raise ValueError("Formal core must contain VGGish, head-only AST, and one adapter")
    if "IDEA-003 ordinal learning" not in config["excluded_routes"]:
        raise ValueError("IDEA-003 exclusion is missing")

    record = {
        "schema_version": "2.1",
        "decision_id": "meowagenet-formal-v2.1-amendment",
        "date": config["amendment_date"],
        "status": "core_frozen_execution_scope_pending",
        "reason": config["amendment_reason"],
        "formal_outcomes_accessed": False,
        "active_route": {
            "goal": "IDEA-048",
            "method": "IDEA-019 low-parameter AST adaptation",
            "idea003": "paused_and_excluded"
        },
        "artifacts": {
            "amended_protocol": {
                "path": str(CONFIG_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": sha256(CONFIG_PATH)
            },
            "execution_lock_template": {
                "path": str(LOCK_TEMPLATE_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": sha256(LOCK_TEMPLATE_PATH)
            },
            "report": {
                "path": str(REPORT_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": sha256(REPORT_PATH)
            },
            "verification_script": {
                "path": str(Path(__file__).relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": sha256(Path(__file__))
            },
            "parent_protocol": parent,
            "split_summary": {
                "path": split_bank["summary_path"],
                "sha256": split_bank["summary_sha256"]
            }
        },
        "minimum_core": {
            "pipelines": 3,
            "split_repeats": len(config["formal_core"]["minimum_repeat_indices"]),
            "outer_folds_per_repeat": config["split_bank"]["outer_folds_per_repeat"],
            "model_seeds": len(config["formal_core"]["model_seeds"]),
            "fold_level_fits": config["formal_core"]["minimum_fold_level_fits"]
        },
        "next_gate": "complete and freeze the execution lock before formal-v2.1 outcomes"
    }
    write_or_verify(RECORD_PATH, json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
