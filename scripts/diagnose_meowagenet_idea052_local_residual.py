"""Diagnose frozen AST temporal-token residual information for IDEA-052."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_PATH = (
    REPO_ROOT
    / "runs"
    / "ast_locked_v1"
    / "gpu_rerun_2026-08-26"
    / "ast_standard_call_embeddings.npz"
)
TOKEN_PATH = (
    REPO_ROOT / "runs" / "ast_temporal_tokens_v1" / "ast_standard_temporal_tokens.npz"
)
TOKEN_SUMMARY_PATH = REPO_ROOT / "runs" / "ast_temporal_tokens_v1" / "extraction_summary.json"
SOURCE_ROOT = (
    REPO_ROOT
    / "runs"
    / "meowagenet_idea051_cat_set_v1"
    / "evaluation"
    / "fits"
    / "S0_call_probability_mean"
)
SOURCE_SUMMARY_PATH = SOURCE_ROOT.parents[1] / "summary.json"
IDEA013_RESULT_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_idea013_temporal_pooling_v2_results.json"
)
OUTPUT_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_idea052_local_residual_diagnostics_v1.json"
)
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


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def token_diagnostics() -> pd.DataFrame:
    global_store = np.load(GLOBAL_PATH)
    token_store = np.load(TOKEN_PATH)
    if not np.array_equal(global_store["call_ids"], token_store["call_ids"]):
        raise RuntimeError("Global embedding and temporal-token call order differs")
    if not np.array_equal(global_store["labels"], token_store["labels"]):
        raise RuntimeError("Global embedding and temporal-token labels differ")
    embeddings = global_store["embeddings"].astype(np.float64)
    tokens = token_store["temporal_tokens"].astype(np.float64)
    token_call_indices = token_store["token_call_indices"].astype(np.int64)
    rows = []
    dimensions = tokens.shape[1]
    for call_index in range(len(embeddings)):
        call_tokens = tokens[token_call_indices == call_index]
        if len(call_tokens) == 0:
            raise RuntimeError(f"Call {call_index} has no temporal token")
        token_mean = call_tokens.mean(axis=0)
        centered = call_tokens - token_mean
        token_rms_dispersion = float(np.sqrt(np.mean(centered**2)))
        peak_residual = call_tokens.max(axis=0) - token_mean
        peak_residual_rms = float(np.linalg.norm(peak_residual) / np.sqrt(dimensions))
        cosine_distances = [1.0 - cosine(token, token_mean) for token in call_tokens]
        rows.append(
            {
                "call_index": call_index,
                "call_id": str(global_store["call_ids"][call_index]),
                "cat_id": str(global_store["cat_ids"][call_index]),
                "true_label": int(global_store["labels"][call_index]),
                "duration": float(global_store["durations"][call_index]),
                "token_count": int(len(call_tokens)),
                "token_rms_dispersion": token_rms_dispersion,
                "peak_residual_rms": peak_residual_rms,
                "mean_token_cosine_distance": float(np.mean(cosine_distances)),
                "global_temporal_mean_cosine": cosine(
                    embeddings[call_index], token_mean
                ),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 792 or result["cat_id"].nunique() != 111:
        raise RuntimeError("Expected 792 calls from 111 cats")
    return result


def load_repeat_predictions(repeat: int) -> pd.DataFrame:
    calls = []
    animals = []
    for fold in FOLDS:
        root = SOURCE_ROOT / "base_seed_17" / f"repeat_{repeat}" / f"fold_{fold}"
        call_frame = pd.read_csv(
            root / "outer_test_call_predictions.csv", dtype={"cat_id": str}
        )
        animal_frame = pd.read_csv(
            root / "outer_test_animal_predictions.csv", dtype={"cat_id": str}
        )
        calls.append(call_frame)
        animals.append(
            animal_frame[["cat_id", "true_label", "predicted_label"]].rename(
                columns={"predicted_label": "animal_prediction"}
            )
        )
    call_oof = pd.concat(calls, ignore_index=True)
    animal_oof = pd.concat(animals, ignore_index=True)
    if len(call_oof) != 792 or call_oof["call_index"].nunique() != 792:
        raise RuntimeError("Expected one OOF prediction per call")
    if len(animal_oof) != 111 or animal_oof["cat_id"].nunique() != 111:
        raise RuntimeError("Expected one OOF prediction per cat")
    merged = call_oof.merge(
        animal_oof[["cat_id", "animal_prediction"]], on="cat_id", validate="many_to_one"
    )
    probabilities = merged[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    merged["call_prediction"] = probabilities.argmax(axis=1)
    merged["call_correct"] = (
        merged["call_prediction"].to_numpy() == merged["true_label"].to_numpy()
    ).astype(int)
    merged["animal_correct"] = (
        merged["animal_prediction"].to_numpy() == merged["true_label"].to_numpy()
    ).astype(int)
    merged["call_confidence"] = probabilities.max(axis=1)
    merged["true_class_probability"] = probabilities[
        np.arange(len(merged)), merged["true_label"].to_numpy(dtype=int)
    ]
    merged["repeat"] = repeat
    return merged


def mean_by_outcome(frame: pd.DataFrame, outcome: str, value: str) -> dict[str, float]:
    return {
        "correct": float(frame.loc[frame[outcome] == 1, value].mean()),
        "incorrect": float(frame.loc[frame[outcome] == 0, value].mean()),
    }


def main() -> None:
    diagnostics = token_diagnostics()
    predictions = pd.concat(
        [load_repeat_predictions(repeat) for repeat in REPEATS], ignore_index=True
    )
    frame = predictions.merge(
        diagnostics,
        on=["call_index", "call_id", "cat_id", "true_label"],
        validate="many_to_one",
    )
    if len(frame) != 792 * len(REPEATS):
        raise RuntimeError("Expected 2,376 repeated call evaluations")
    idea013 = json.loads(IDEA013_RESULT_PATH.read_text(encoding="utf-8"))
    source_files = sorted(SOURCE_ROOT.rglob("outer_test_*_predictions.csv"))
    output = {
        "schema_version": "1.0",
        "diagnostic_id": "meowagenet-idea052-local-residual-diagnostics-v1",
        "status": "complete",
        "role": "post-evidence diagnostic for IDEA-052; no model fit or method selection",
        "sources": {
            "global_embeddings": str(GLOBAL_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "global_embeddings_sha256": sha256(GLOBAL_PATH),
            "temporal_tokens": str(TOKEN_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "temporal_tokens_sha256": sha256(TOKEN_PATH),
            "temporal_extraction_summary_sha256": sha256(TOKEN_SUMMARY_PATH),
            "idea051_S0_summary_sha256": sha256(SOURCE_SUMMARY_PATH),
            "idea013_result_sha256": sha256(IDEA013_RESULT_PATH),
            "prediction_files": len(source_files),
            "prediction_aggregate_sha256": hashlib.sha256(
                "".join(sha256(path) for path in source_files).encode("ascii")
            ).hexdigest(),
        },
        "dataset": {
            "calls": 792,
            "cats": 111,
            "temporal_tokens": 5842,
            "embedding_dimensions": 768,
            "token_count_range": [
                int(diagnostics["token_count"].min()),
                int(diagnostics["token_count"].max()),
            ],
            "token_count_quantiles": {
                str(quantile): float(diagnostics["token_count"].quantile(quantile))
                for quantile in (0.25, 0.5, 0.75, 0.9)
            },
        },
        "local_variation": {
            "token_rms_dispersion": {
                "mean": float(diagnostics["token_rms_dispersion"].mean()),
                "quartiles": [
                    float(value)
                    for value in diagnostics["token_rms_dispersion"].quantile(
                        [0.25, 0.5, 0.75]
                    )
                ],
                "range": [
                    float(diagnostics["token_rms_dispersion"].min()),
                    float(diagnostics["token_rms_dispersion"].max()),
                ],
            },
            "peak_residual_rms": {
                "mean": float(diagnostics["peak_residual_rms"].mean()),
                "quartiles": [
                    float(value)
                    for value in diagnostics["peak_residual_rms"].quantile(
                        [0.25, 0.5, 0.75]
                    )
                ],
                "range": [
                    float(diagnostics["peak_residual_rms"].min()),
                    float(diagnostics["peak_residual_rms"].max()),
                ],
            },
            "mean_token_cosine_distance": float(
                diagnostics["mean_token_cosine_distance"].mean()
            ),
            "global_temporal_mean_cosine": {
                "mean": float(diagnostics["global_temporal_mean_cosine"].mean()),
                "range": [
                    float(diagnostics["global_temporal_mean_cosine"].min()),
                    float(diagnostics["global_temporal_mean_cosine"].max()),
                ],
            },
        },
        "associations_across_2376_call_evaluations": {
            "token_dispersion_vs_call_correct": safe_spearman(
                frame["token_rms_dispersion"], frame["call_correct"]
            ),
            "token_dispersion_vs_animal_correct": safe_spearman(
                frame["token_rms_dispersion"], frame["animal_correct"]
            ),
            "peak_residual_vs_call_correct": safe_spearman(
                frame["peak_residual_rms"], frame["call_correct"]
            ),
            "peak_residual_vs_animal_correct": safe_spearman(
                frame["peak_residual_rms"], frame["animal_correct"]
            ),
            "global_temporal_cosine_vs_call_correct": safe_spearman(
                frame["global_temporal_mean_cosine"], frame["call_correct"]
            ),
            "duration_vs_token_dispersion": safe_spearman(
                frame["duration"], frame["token_rms_dispersion"]
            ),
            "token_count_vs_token_dispersion": safe_spearman(
                frame["token_count"], frame["token_rms_dispersion"]
            ),
        },
        "associations_across_792_unique_calls": {
            "ordinal_age_label_vs_token_dispersion": safe_spearman(
                diagnostics["true_label"], diagnostics["token_rms_dispersion"]
            ),
            "ordinal_age_label_vs_peak_residual": safe_spearman(
                diagnostics["true_label"], diagnostics["peak_residual_rms"]
            ),
            "duration_vs_token_dispersion": safe_spearman(
                diagnostics["duration"], diagnostics["token_rms_dispersion"]
            ),
        },
        "outcome_contrasts": {
            value: {
                "by_call_correct": mean_by_outcome(frame, "call_correct", value),
                "by_animal_correct": mean_by_outcome(frame, "animal_correct", value),
            }
            for value in (
                "token_rms_dispersion",
                "peak_residual_rms",
                "global_temporal_mean_cosine",
            )
        },
        "class_means": {
            LABEL_NAMES[label]: {
                "calls": int((diagnostics["true_label"] == label).sum()),
                "token_rms_dispersion": float(
                    diagnostics.loc[
                        diagnostics["true_label"] == label, "token_rms_dispersion"
                    ].mean()
                ),
                "peak_residual_rms": float(
                    diagnostics.loc[
                        diagnostics["true_label"] == label, "peak_residual_rms"
                    ].mean()
                ),
            }
            for label in range(3)
        },
        "historical_local_only_control": {
            "source": "IDEA-013 frozen temporal pooling v2",
            "temporal_mean_macro_f1": idea013["results"]["mean"]["macro_f1"],
            "temporal_mean_capacity_macro_f1": idea013["results"]["mean_capacity"][
                "macro_f1"
            ],
            "temporal_attention_macro_f1": idea013["results"]["gated_attention"][
                "macro_f1"
            ],
            "interpretation": "Temporal tokens alone were weaker in the historical pilot; IDEA-052 therefore preserves the strong global embedding and adds zero-initialized residual paths.",
        },
        "interpretation_boundary": {
            "located_evidence": "Frozen temporal tokens retain measurable within-call variation and differ from the global pooler representation.",
            "inference": "A residual path can expose local mean-shift or transient salience while preserving the global reference at initialization.",
            "discriminating_requirement": "A useful local branch must outperform the matched global reference in paired complete OOF; non-zero gates alone only show that the branch was used.",
        },
    }
    write_json(OUTPUT_PATH, output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
