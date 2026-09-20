"""Verify the v3 evidence ledger from locked result artifacts without GPU use."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "metadata" / "experiments" / "meowagenet_v3_independent_evidence_ledger.json"
SOURCE_PATHS = {
    "pilot_report": "reports/07_IDEA-019_PEFT_placement_results.md",
    "formal_v2_1_report": "reports/12_formal_v2_1_core_results.md",
    "formal_v2_1_results": "metadata/experiments/meowagenet_formal_v2_1_core_results.json",
    "formal_v2_1_summary": "runs/meowagenet_formal_v2_1_core/formal_summary.json",
    "hpo_report": "reports/18_AST_head_and_adapter_hyperparameter_search.md",
    "adapter_deterministic_report": "reports/35_AST_adapter_deterministic_replication_results.md",
    "adapter_new_seed_summary": "runs/meowagenet_idea065_adapter_seed_replication_v1/formal_summary.json",
    "c1_pre_idea076_ledger": "metadata/experiments/meowagenet_C1_evidence_ledger_pre_IDEA076.json",
    "idea076_results": "metadata/experiments/meowagenet_idea076_C1_final_seed_confirmation_v1_results.json",
    "idea077_results": "metadata/experiments/meowagenet_idea077_ast_last4_layer_mix_v1_results.json",
    "idea078_results": "metadata/experiments/meowagenet_idea078_ast_prelast_special_token_age_injection_v1_results.json",
    "idea079_results": "metadata/experiments/meowagenet_idea079_spatial_patch_adapter_v1_results.json",
    "idea080_results": "metadata/experiments/meowagenet_idea080_ast_last_block_lora_c1_factorial_v1_results.json",
    "idea081_results": "metadata/experiments/meowagenet_idea081_ast_tail_convpass_c1_factorial_v1_results.json",
    "idea082_results": "metadata/experiments/meowagenet_idea082_age_acoustic_group_ablation_v1_results.json",
    "catmeows_dog_AST_report": "reports/37_IDEA-067_external_AST_representation_results.md",
    "dog_C1_results": "metadata/experiments/idea075_dog_C1_age_sensitive_AST_v1_results.json",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def sign_counts(values: list[float]) -> list[int]:
    return [sum(value > 0 for value in values), sum(value == 0 for value in values), sum(value < 0 for value in values)]


def verify() -> dict[str, Any]:
    ledger = read_json(LEDGER_PATH)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    observed_hashes = {name: sha256(ROOT / path) for name, path in SOURCE_PATHS.items()}
    check("source_hashes", observed_hashes == ledger["source_hashes"], observed_hashes)

    formal = read_json(ROOT / SOURCE_PATHS["formal_v2_1_results"])
    formal_ledger = ledger["probe_guided_adapter_evidence"]["formal_v2_1"]
    formal_values = {
        name: formal["pipeline_aggregate"][name]["macro_f1"]["mean"]
        for name in ("vggish_mlp", "ast_head_only", "ast_probe_guided_adapter")
    }
    check("formal_pipeline_means", formal_values == formal_ledger["macro_f1_mean"], formal_values)
    h048 = formal["primary_contrasts"]["H048_adapter_minus_vggish"]
    h019 = formal["primary_contrasts"]["H019_adapter_minus_ast_head_only"]
    check(
        "formal_contrasts",
        close(h048["mean_macro_f1_difference"], formal_ledger["adapter_minus_vggish"]["mean"])
        and close(h019["mean_macro_f1_difference"], formal_ledger["adapter_minus_head_only"]["mean"])
        and h048["positive_complete_oof_evaluations"] == 9
        and h019["positive_complete_oof_evaluations"] == 5,
        {"H048": h048, "H019": h019},
    )

    deterministic = read_json(ROOT / SOURCE_PATHS["adapter_new_seed_summary"])
    head_values = [row["macro_f1"] for row in deterministic["oof_metrics"]["ast_head_only"].values()]
    adapter_values = [row["macro_f1"] for row in deterministic["oof_metrics"]["ast_probe_guided_adapter"].values()]
    adapter_deltas = [candidate - control for candidate, control in zip(adapter_values, head_values)]
    new_seed = ledger["probe_guided_adapter_evidence"]["deterministic_new_seed_replication"]
    check(
        "deterministic_new_seed",
        close(sum(head_values) / 9, new_seed["macro_f1_mean"]["ast_head_only"], 5e-5)
        and close(sum(adapter_values) / 9, new_seed["macro_f1_mean"]["ast_probe_guided_adapter"], 5e-5)
        and sign_counts(adapter_deltas)[::2] == [5, 4],
        {"head_mean": sum(head_values) / 9, "adapter_mean": sum(adapter_values) / 9, "signs": sign_counts(adapter_deltas)},
    )

    pre = read_json(ROOT / SOURCE_PATHS["c1_pre_idea076_ledger"])
    idea076 = read_json(ROOT / SOURCE_PATHS["idea076_results"])
    summary080 = read_json(ROOT / "runs/meowagenet_idea080_ast_last_block_lora_c1_factorial_v1/initial_evaluation_summary.json")
    summary081 = read_json(ROOT / "runs/meowagenet_idea081_ast_tail_convpass_c1_factorial_v1/initial_evaluation_summary.json")
    c1_deltas = []
    for row in pre["rounds"]:
        c1_deltas.extend([row["C1_minus_A0_mean_macro_f1"]] * 9)
    deltas076 = [float(row["C1_minus_A0_macro_f1"]) for row in idea076["seed_repeat_level"]["C1_minus_A0_macro_f1"].get("values", [])]
    if not deltas076:
        mean076 = idea076["seed_repeat_level"]["C1_minus_A0_macro_f1"]["mean"]
        signs076 = idea076["seed_repeat_level"]["C1_minus_A0_macro_f1"]
        c1_sum = sum(row["C1_minus_A0_mean_macro_f1"] * 9 for row in pre["rounds"]) + mean076 * 18
        signs = [
            pre["combined_description"]["seed_repeat_signs_positive_tied_negative"][index]
            + [signs076["positive"], signs076["tied"], signs076["negative"]][index]
            for index in range(3)
        ]
    else:
        c1_sum = sum(c1_deltas) + sum(deltas076)
        signs = [pre["combined_description"]["seed_repeat_signs_positive_tied_negative"][index] + sign_counts(deltas076)[index] for index in range(3)]
    deltas080 = [float(row["C1_minus_A0_macro_f1"]) for row in summary080["seed_repeat_results"]]
    deltas081 = [
        float(row["C1_bounded_age_macro_f1"]) - float(row["A0_frozen_tail_macro_f1"])
        for row in summary081["seed_repeat_results"]
    ]
    combined = ledger["a0_c1_meowagenet_evidence"]["descriptive_combined"]
    mean54 = c1_sum / 54
    mean72 = (c1_sum + sum(deltas080) + sum(deltas081)) / 72
    signs72 = [signs[index] + sign_counts(deltas080)[index] + sign_counts(deltas081)[index] for index in range(3)]
    check(
        "C1_descriptive_combination",
        close(mean54, combined["idea071_through_idea076"]["C1_minus_A0_mean_macro_f1"])
        and close(mean72, combined["idea071_through_idea081_including_factorial_controls"]["C1_minus_A0_mean_macro_f1"])
        and signs72 == combined["idea071_through_idea081_including_factorial_controls"]["signs_positive_tied_negative"],
        {"mean54": mean54, "mean72": mean72, "signs72": signs72},
    )

    result_sources = {
        item["idea"]: item for item in ledger["idea077_to_idea082"]
    }
    source077 = read_json(ROOT / SOURCE_PATHS["idea077_results"])
    source078 = read_json(ROOT / SOURCE_PATHS["idea078_results"])
    source079 = read_json(ROOT / SOURCE_PATHS["idea079_results"])
    source080 = read_json(ROOT / SOURCE_PATHS["idea080_results"])
    source081 = read_json(ROOT / SOURCE_PATHS["idea081_results"])
    source082 = read_json(ROOT / SOURCE_PATHS["idea082_results"])
    metric_checks = {
        "IDEA-077": close(result_sources["IDEA-077"]["candidate_minus_A0"], source077["comparisons"]["L1_minus_A0_macro_f1"]["mean"]),
        "IDEA-078": close(result_sources["IDEA-078"]["candidate_minus_A0"], source078["comparisons"]["T1_minus_A0_macro_f1"]["mean"]),
        "IDEA-079": close(result_sources["IDEA-079"]["candidate_minus_A0"], source079["comparisons"]["S1_minus_A0_macro_f1"]["mean"]),
        "IDEA-080": close(result_sources["IDEA-080"]["candidate_minus_A0"], source080["comparisons"]["L1_minus_A0_macro_f1"]["mean"]),
        "IDEA-081": close(result_sources["IDEA-081"]["candidate_minus_A0"], source081["contrasts"]["V1_minus_A0"]["mean_macro_f1"]),
        "IDEA-082": close(result_sources["IDEA-082"]["G3_real_minus_A0"], source082["group_decisions"]["G3_spectral_energy"]["real_minus_A0_macro_f1"]),
    }
    check("IDEA077_082_metrics", all(metric_checks.values()), metric_checks)

    cat = read_json(ROOT / "runs/idea067_external_ast_representation_v1/catmeows/evaluation_summary.json")
    dog = read_json(ROOT / "runs/idea067_external_ast_representation_v1/canine/evaluation_summary.json")
    external = ledger["external_boundaries"]
    check(
        "external_boundaries",
        close(cat["macro_f1"]["ast_mean"], external["catmeows"]["macro_f1_mean"]["frozen_AST"], 5e-5)
        and close(dog["macro_f1"]["ast_mean"], external["dog_subset_AST"]["macro_f1_mean"]["frozen_AST"], 5e-5)
        and external["numeric_pooling_with_meowagenet_allowed"] is False,
        {"cat": cat["macro_f1"], "dog": dog["macro_f1"]},
    )

    failed = [row["name"] for row in checks if not row["passed"]]
    return {
        "audit_id": "v3-independent-evidence-ledger-verification-v1",
        "status": "PASS" if not failed else "FAIL",
        "ledger_path": LEDGER_PATH.relative_to(ROOT).as_posix(),
        "ledger_sha256": sha256(LEDGER_PATH),
        "gpu_used": False,
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_checks": failed,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify()
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
