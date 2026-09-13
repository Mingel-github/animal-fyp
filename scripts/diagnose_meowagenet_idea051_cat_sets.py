"""Diagnose within-cat call disagreement for IDEA-051 set aggregation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    REPO_ROOT
    / "runs"
    / "meowagenet_ast_cat_balance_global_weighting_v1"
    / "evaluation"
    / "fits"
    / "C0_global_class_balanced"
)
SOURCE_SUMMARY = SOURCE_ROOT.parents[1] / "summary.json"
OUTPUT_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_idea051_cat_set_diagnostics_v1.json"
)
BASE_SEEDS = (17, 43, 101)
REPEATS = (0, 1, 2)
FOLDS = (0, 1, 2, 3)
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


def safe_spearman(left: pd.Series, right: pd.Series) -> dict[str, float]:
    result = spearmanr(left.to_numpy(dtype=float), right.to_numpy(dtype=float))
    return {"rho": float(result.statistic), "pvalue": float(result.pvalue)}


def bag_size_group(call_count: int) -> str:
    if call_count == 1:
        return "1"
    if call_count <= 4:
        return "2-4"
    if call_count <= 9:
        return "5-9"
    return "10+"


def cat_rows(calls: pd.DataFrame, base_seed: int, repeat: int) -> pd.DataFrame:
    rows = []
    for cat_id, group in calls.groupby("cat_id", sort=True):
        labels = group["true_label"].unique()
        if len(labels) != 1:
            raise RuntimeError(f"Cat {cat_id} has inconsistent labels")
        probabilities = group[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        mean_probabilities = probabilities.mean(axis=0)
        predicted_label = int(mean_probabilities.argmax())
        call_predictions = probabilities.argmax(axis=1)
        total_variation = 0.5 * np.abs(probabilities - mean_probabilities).sum(axis=1)
        true_label = int(labels[0])
        rows.append(
            {
                "base_seed": base_seed,
                "repeat": repeat,
                "cat_id": str(cat_id),
                "true_label": true_label,
                "call_count": int(len(group)),
                "bag_size_group": bag_size_group(int(len(group))),
                "animal_prediction": predicted_label,
                "correct": int(predicted_label == true_label),
                "animal_confidence": float(mean_probabilities.max()),
                "mean_true_class_probability": float(probabilities[:, true_label].mean()),
                "call_argmax_disagreement_rate": float(
                    np.mean(call_predictions != predicted_label)
                ),
                "within_cat_probability_total_variation": float(total_variation.mean()),
                "within_cat_probability_std": float(probabilities.std(axis=0).mean()),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 111 or result["cat_id"].nunique() != 111:
        raise RuntimeError("A complete OOF diagnostic must cover 111 cats")
    return result


def load_complete_oof(base_seed: int, repeat: int) -> pd.DataFrame:
    frames = []
    for fold in FOLDS:
        path = (
            SOURCE_ROOT
            / f"base_seed_{base_seed}"
            / f"repeat_{repeat}"
            / f"fold_{fold}"
            / "outer_test_call_predictions.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"cat_id": str})
        frame.insert(0, "outer_fold", fold)
        frames.append(frame)
    calls = pd.concat(frames, ignore_index=True)
    if len(calls) != 792 or calls["call_index"].nunique() != 792:
        raise RuntimeError("A complete OOF diagnostic must cover all 792 calls once")
    return calls.sort_values("call_index").reset_index(drop=True)


def subgroup_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("1", "2-4", "5-9", "10+"):
        group = frame[frame["bag_size_group"] == name]
        unique = group.drop_duplicates("cat_id")
        result[name] = {
            "cat_evaluations": int(len(group)),
            "unique_cats": int(group["cat_id"].nunique()),
            "unique_cats_by_class": {
                LABEL_NAMES[class_index]: int((unique["true_label"] == class_index).sum())
                for class_index in range(3)
            },
            "accuracy": float(accuracy_score(group["true_label"], group["animal_prediction"])),
            "balanced_accuracy": float(
                balanced_accuracy_score(group["true_label"], group["animal_prediction"])
            ),
            "macro_f1": float(
                f1_score(
                    group["true_label"],
                    group["animal_prediction"],
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                )
            ),
            "mean_disagreement_rate": float(group["call_argmax_disagreement_rate"].mean()),
            "mean_probability_total_variation": float(
                group["within_cat_probability_total_variation"].mean()
            ),
        }
    return result


def main() -> None:
    all_cats = []
    source_files = []
    for base_seed in BASE_SEEDS:
        for repeat in REPEATS:
            calls = load_complete_oof(base_seed, repeat)
            all_cats.append(cat_rows(calls, base_seed, repeat))
            for fold in FOLDS:
                source_files.append(
                    SOURCE_ROOT
                    / f"base_seed_{base_seed}"
                    / f"repeat_{repeat}"
                    / f"fold_{fold}"
                    / "outer_test_call_predictions.csv"
                )
    frame = pd.concat(all_cats, ignore_index=True)
    if len(frame) != 999:
        raise RuntimeError("Expected nine repeated evaluations of 111 cats")

    call_counts = frame.drop_duplicates("cat_id")["call_count"]
    by_cat = (
        frame.groupby("cat_id", sort=True)
        .agg(
            true_label=("true_label", "first"),
            call_count=("call_count", "first"),
            mean_disagreement_rate=("call_argmax_disagreement_rate", "mean"),
            mean_probability_total_variation=(
                "within_cat_probability_total_variation",
                "mean",
            ),
            repeat_accuracy=("correct", "mean"),
            mean_confidence=("animal_confidence", "mean"),
        )
        .reset_index()
    )
    top_disagreement = by_cat.sort_values(
        ["mean_disagreement_rate", "mean_probability_total_variation"],
        ascending=False,
    ).head(12)

    correct = frame[frame["correct"] == 1]
    incorrect = frame[frame["correct"] == 0]
    output = {
        "schema_version": "1.0",
        "diagnostic_id": "meowagenet-idea051-cat-set-diagnostics-v1",
        "status": "complete",
        "role": "post-evidence diagnostic for IDEA-051; no new model selection or outer-test fit",
        "source": {
            "pipeline": "C0_global_class_balanced",
            "base_seeds": list(BASE_SEEDS),
            "repeats": list(REPEATS),
            "outer_folds": list(FOLDS),
            "complete_oof_evaluations": 9,
            "call_rows": 792 * 9,
            "cat_evaluations": 111 * 9,
            "summary_path": str(SOURCE_SUMMARY.relative_to(REPO_ROOT)).replace("\\", "/"),
            "summary_sha256": sha256(SOURCE_SUMMARY),
            "prediction_files": len(source_files),
            "prediction_aggregate_sha256": hashlib.sha256(
                "".join(sha256(path) for path in sorted(source_files)).encode("ascii")
            ).hexdigest(),
        },
        "dataset_structure": {
            "calls": 792,
            "cats": 111,
            "call_count_min": int(call_counts.min()),
            "call_count_quartiles": [float(value) for value in call_counts.quantile([0.25, 0.5, 0.75])],
            "call_count_mean": float(call_counts.mean()),
            "call_count_max": int(call_counts.max()),
            "single_call_cats": int((call_counts == 1).sum()),
            "cats_with_at_least_10_calls": int((call_counts >= 10).sum()),
        },
        "within_cat_signal": {
            "cat_evaluations_with_any_argmax_disagreement": int(
                (frame["call_argmax_disagreement_rate"] > 0).sum()
            ),
            "fraction_with_any_argmax_disagreement": float(
                (frame["call_argmax_disagreement_rate"] > 0).mean()
            ),
            "cat_evaluations_with_at_least_25pct_disagreement": int(
                (frame["call_argmax_disagreement_rate"] >= 0.25).sum()
            ),
            "fraction_with_at_least_25pct_disagreement": float(
                (frame["call_argmax_disagreement_rate"] >= 0.25).mean()
            ),
            "mean_argmax_disagreement_rate": float(
                frame["call_argmax_disagreement_rate"].mean()
            ),
            "mean_probability_total_variation": float(
                frame["within_cat_probability_total_variation"].mean()
            ),
            "mean_probability_std": float(frame["within_cat_probability_std"].mean()),
            "correct_vs_incorrect": {
                "correct_cat_evaluations": int(len(correct)),
                "incorrect_cat_evaluations": int(len(incorrect)),
                "correct_mean_disagreement_rate": float(
                    correct["call_argmax_disagreement_rate"].mean()
                ),
                "incorrect_mean_disagreement_rate": float(
                    incorrect["call_argmax_disagreement_rate"].mean()
                ),
                "correct_mean_probability_total_variation": float(
                    correct["within_cat_probability_total_variation"].mean()
                ),
                "incorrect_mean_probability_total_variation": float(
                    incorrect["within_cat_probability_total_variation"].mean()
                ),
            },
        },
        "associations_across_999_cat_evaluations": {
            "call_count_vs_argmax_disagreement": safe_spearman(
                frame["call_count"], frame["call_argmax_disagreement_rate"]
            ),
            "call_count_vs_probability_total_variation": safe_spearman(
                frame["call_count"], frame["within_cat_probability_total_variation"]
            ),
            "disagreement_vs_correct": safe_spearman(
                frame["call_argmax_disagreement_rate"], frame["correct"]
            ),
            "probability_total_variation_vs_correct": safe_spearman(
                frame["within_cat_probability_total_variation"], frame["correct"]
            ),
        },
        "bag_size_groups": subgroup_metrics(frame),
        "highest_mean_disagreement_cats": [
            {
                "cat_id": str(row.cat_id),
                "label": LABEL_NAMES[int(row.true_label)],
                "call_count": int(row.call_count),
                "mean_disagreement_rate": float(row.mean_disagreement_rate),
                "mean_probability_total_variation": float(
                    row.mean_probability_total_variation
                ),
                "repeat_accuracy": float(row.repeat_accuracy),
            }
            for row in top_disagreement.itertuples(index=False)
        ],
        "interpretation_boundary": {
            "located_evidence": "Repeated complete-OOF predictions show whether calls from the same cat carry heterogeneous model evidence.",
            "inference": "Higher disagreement and dispersion among errors would support testing learned cat-level aggregation.",
            "limitation": "These completed OOF results motivate a post-evidence exploratory idea and do not constitute independent confirmation of a new model.",
        },
    }
    write_json(OUTPUT_PATH, output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
