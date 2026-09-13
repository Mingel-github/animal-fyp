"""Measure animal-level complementarity between the current AST and VGGish paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score


REPO_ROOT = Path(__file__).resolve().parents[1]
AST_ROOT = (
    REPO_ROOT
    / "runs"
    / "meowagenet_idea052_ast_local_residual_v1"
    / "evaluation"
    / "oof"
    / "R0_global_probability_mean"
)
VGGISH_ROOT = (
    REPO_ROOT
    / "runs"
    / "meowagenet_formal_v2_1_core"
    / "oof"
    / "vggish_mlp"
)
AST_SUMMARY_PATH = AST_ROOT.parents[1] / "summary.json"
VGGISH_SUMMARY_PATH = VGGISH_ROOT.parents[1] / "formal_summary.json"
OUTPUT_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_idea053_ast_vggish_complementarity_v1.json"
)
REPEATS = (0, 1, 2)
BASE_SEED = 17
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
LABEL_NAMES = ("kitten", "adult", "senior")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(labels, predictions, weights="quadratic")
        ),
        "plain_accuracy": float(accuracy_score(labels, predictions)),
    }


def load_pair(repeat: int) -> pd.DataFrame:
    ast_path = AST_ROOT / f"repeat_{repeat}_animals.csv"
    vggish_path = (
        VGGISH_ROOT / f"repeat_{repeat}_base_seed_{BASE_SEED}_animal_predictions.csv"
    )
    ast = pd.read_csv(ast_path, dtype={"cat_id": str}).sort_values("cat_id")
    vggish = pd.read_csv(vggish_path, dtype={"cat_id": str}).sort_values("cat_id")
    if len(ast) != 111 or len(vggish) != 111:
        raise RuntimeError("Each complete OOF source must contain 111 animals")
    if not np.array_equal(ast["cat_id"].to_numpy(), vggish["cat_id"].to_numpy()):
        raise RuntimeError("AST and VGGish animal order differs")
    if not np.array_equal(
        ast["true_label"].to_numpy(), vggish["true_label"].to_numpy()
    ):
        raise RuntimeError("AST and VGGish labels differ")
    frame = ast[["cat_id", "true_label", *PROBABILITY_COLUMNS]].copy()
    frame = frame.rename(
        columns={column: f"ast_{column}" for column in PROBABILITY_COLUMNS}
    )
    for column in PROBABILITY_COLUMNS:
        frame[f"vggish_{column}"] = vggish[column].to_numpy(dtype=float)
    frame["repeat"] = repeat
    return frame


def probability_matrix(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    return frame[[f"{prefix}_{column}" for column in PROBABILITY_COLUMNS]].to_numpy(
        dtype=np.float64
    )


def class_counts(
    labels: np.ndarray, ast_correct: np.ndarray, vggish_correct: np.ndarray
) -> dict[str, dict[str, int]]:
    output = {}
    for label, name in enumerate(LABEL_NAMES):
        mask = labels == label
        output[name] = {
            "support": int(mask.sum()),
            "both_correct": int((mask & ast_correct & vggish_correct).sum()),
            "ast_only_correct": int((mask & ast_correct & ~vggish_correct).sum()),
            "vggish_only_correct": int((mask & ~ast_correct & vggish_correct).sum()),
            "both_wrong": int((mask & ~ast_correct & ~vggish_correct).sum()),
        }
    return output


def repeat_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    labels = frame["true_label"].to_numpy(dtype=np.int64)
    ast_probabilities = probability_matrix(frame, "ast")
    vggish_probabilities = probability_matrix(frame, "vggish")
    ast_predictions = ast_probabilities.argmax(axis=1)
    vggish_predictions = vggish_probabilities.argmax(axis=1)
    ast_correct = ast_predictions == labels
    vggish_correct = vggish_predictions == labels
    disagreement = ast_predictions != vggish_predictions
    oracle_predictions = np.where(vggish_correct, labels, ast_predictions)
    confidence_predictions = np.where(
        ast_probabilities.max(axis=1) >= vggish_probabilities.max(axis=1),
        ast_predictions,
        vggish_predictions,
    )
    blend_rows = []
    for ast_weight in np.linspace(0.0, 1.0, 11):
        probabilities = (
            ast_weight * ast_probabilities
            + (1.0 - ast_weight) * vggish_probabilities
        )
        values = metrics(labels, probabilities.argmax(axis=1))
        blend_rows.append({"ast_weight": float(ast_weight), **values})
    epsilon = 1.0e-12
    midpoint = 0.5 * (ast_probabilities + vggish_probabilities)
    js_divergence = 0.5 * np.sum(
        ast_probabilities
        * np.log((ast_probabilities + epsilon) / (midpoint + epsilon)),
        axis=1,
    ) + 0.5 * np.sum(
        vggish_probabilities
        * np.log((vggish_probabilities + epsilon) / (midpoint + epsilon)),
        axis=1,
    )
    return {
        "repeat": int(frame["repeat"].iloc[0]),
        "animals": int(len(frame)),
        "ast": metrics(labels, ast_predictions),
        "vggish": metrics(labels, vggish_predictions),
        "joint_correctness": {
            "both_correct": int((ast_correct & vggish_correct).sum()),
            "ast_only_correct": int((ast_correct & ~vggish_correct).sum()),
            "vggish_only_correct": int((~ast_correct & vggish_correct).sum()),
            "both_wrong": int((~ast_correct & ~vggish_correct).sum()),
            "prediction_disagreement": int(disagreement.sum()),
            "both_wrong_with_different_predictions": int(
                ((~ast_correct & ~vggish_correct) & disagreement).sum()
            ),
        },
        "joint_correctness_by_class": class_counts(
            labels, ast_correct, vggish_correct
        ),
        "probability_relationship": {
            "flattened_pearson_correlation": float(
                np.corrcoef(ast_probabilities.ravel(), vggish_probabilities.ravel())[0, 1]
            ),
            "mean_jensen_shannon_divergence": float(js_divergence.mean()),
            "mean_absolute_probability_difference": float(
                np.abs(ast_probabilities - vggish_probabilities).mean()
            ),
        },
        "descriptive_controls": {
            "confidence_selection": metrics(labels, confidence_predictions),
            "oracle_either_correct_upper_bound": metrics(labels, oracle_predictions),
            "fixed_probability_blends": blend_rows,
        },
    }


def main() -> None:
    frames = [load_pair(repeat) for repeat in REPEATS]
    analyses = [repeat_analysis(frame) for frame in frames]
    joined = pd.concat(frames, ignore_index=True)
    labels = joined["true_label"].to_numpy(dtype=np.int64)
    ast_probabilities = probability_matrix(joined, "ast")
    vggish_probabilities = probability_matrix(joined, "vggish")
    ast_predictions = ast_probabilities.argmax(axis=1)
    vggish_predictions = vggish_probabilities.argmax(axis=1)
    ast_correct = ast_predictions == labels
    vggish_correct = vggish_predictions == labels
    blend_summary = []
    for ast_weight in np.linspace(0.0, 1.0, 11):
        repeat_values = []
        for frame in frames:
            repeat_labels = frame["true_label"].to_numpy(dtype=np.int64)
            blended = (
                ast_weight * probability_matrix(frame, "ast")
                + (1.0 - ast_weight) * probability_matrix(frame, "vggish")
            )
            repeat_values.append(
                metrics(repeat_labels, blended.argmax(axis=1))["macro_f1"]
            )
        blend_summary.append(
            {
                "ast_weight": float(ast_weight),
                "macro_f1_by_repeat": [float(value) for value in repeat_values],
                "macro_f1_mean": float(np.mean(repeat_values)),
                "positive_repeats_vs_ast": int(
                    sum(
                        value > analysis["ast"]["macro_f1"]
                        for value, analysis in zip(repeat_values, analyses, strict=True)
                    )
                ),
            }
        )
    ast_files = [AST_ROOT / f"repeat_{repeat}_animals.csv" for repeat in REPEATS]
    vggish_files = [
        VGGISH_ROOT
        / f"repeat_{repeat}_base_seed_{BASE_SEED}_animal_predictions.csv"
        for repeat in REPEATS
    ]
    output = {
        "schema_version": "1.0",
        "diagnostic_id": "meowagenet-idea053-ast-vggish-complementarity-v1",
        "status": "complete",
        "role": "post-evidence complementarity diagnosis; fixed blends are descriptive and do not define a final candidate",
        "source_alignment": {
            "ast": "IDEA-052 R0 tuned frozen-AST with animal-level checkpoint selection",
            "vggish": "formal-v2.1 VGGish+MLP base-seed-17 complete OOF",
            "repeats": list(REPEATS),
            "animals_per_repeat": 111,
            "paired_animal_evaluations": int(len(joined)),
            "same_cat_ids_and_labels": True,
            "same_nested_role_file": True,
            "recipe_difference": "The AST path uses the newer tuned R0 recipe; VGGish is the preserved formal-v2.1 baseline recipe.",
        },
        "sources": {
            "ast_summary_sha256": sha256(AST_SUMMARY_PATH),
            "vggish_summary_sha256": sha256(VGGISH_SUMMARY_PATH),
            "ast_oof_sha256": {path.name: sha256(path) for path in ast_files},
            "vggish_oof_sha256": {path.name: sha256(path) for path in vggish_files},
        },
        "repeat_analyses": analyses,
        "aggregate_joint_correctness": {
            "both_correct": int((ast_correct & vggish_correct).sum()),
            "ast_only_correct": int((ast_correct & ~vggish_correct).sum()),
            "vggish_only_correct": int((~ast_correct & vggish_correct).sum()),
            "both_wrong": int((~ast_correct & ~vggish_correct).sum()),
            "prediction_disagreement": int((ast_predictions != vggish_predictions).sum()),
            "oracle_either_correct_plain_accuracy": float(
                (ast_correct | vggish_correct).mean()
            ),
        },
        "aggregate_joint_correctness_by_class": class_counts(
            labels, ast_correct, vggish_correct
        ),
        "descriptive_fixed_blend_summary": blend_summary,
        "decision_rule": {
            "probability_fusion_signal": "At least one fixed interior AST weight improves mean macro F1 and improves at least two of three repeats relative to AST.",
            "next_if_present": "Build an independent nested runner that selects fusion weight from inner validation only.",
            "next_if_absent": "Skip direct probability fusion and examine representation fusion or constrained calibration.",
        },
    }
    interior = [row for row in blend_summary if 0.0 < row["ast_weight"] < 1.0]
    ast_mean = float(np.mean([analysis["ast"]["macro_f1"] for analysis in analyses]))
    best = max(interior, key=lambda row: row["macro_f1_mean"])
    signal = (
        best["macro_f1_mean"] > ast_mean
        and best["positive_repeats_vs_ast"] >= 2
    )
    output["diagnostic_decision"] = {
        "ast_macro_f1_mean": ast_mean,
        "best_descriptive_interior_blend": best,
        "mean_macro_f1_gain": float(best["macro_f1_mean"] - ast_mean),
        "probability_fusion_signal_present": bool(signal),
        "next_action": output["decision_rule"][
            "next_if_present" if signal else "next_if_absent"
        ],
    }
    write_json(OUTPUT_PATH, output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
