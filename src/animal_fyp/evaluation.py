"""Locked prediction aggregation and metrics for MeowAgeNet."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


LABELS = ("kitten", "adult", "senior")


def aggregate_animal_probabilities(
    probabilities: np.ndarray,
    cat_ids: Sequence[str],
    true_labels: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average prediction-unit probabilities within each cat ID.

    Returns sorted cat IDs, one true label per cat, and one probability vector
    per cat. A cat carrying conflicting class labels is rejected.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    cat_ids = np.asarray(cat_ids, dtype=str)
    true_labels = np.asarray(true_labels, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(LABELS):
        raise ValueError(f"Expected probabilities with shape (n, {len(LABELS)})")
    if not (len(probabilities) == len(cat_ids) == len(true_labels)):
        raise ValueError("Probabilities, cat IDs, and labels must have equal lengths")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities contain non-finite values")

    unique_cats = np.unique(cat_ids)
    animal_probabilities = np.empty((len(unique_cats), len(LABELS)), dtype=np.float64)
    animal_labels = np.empty(len(unique_cats), dtype=np.int64)
    for index, cat_id in enumerate(unique_cats):
        mask = cat_ids == cat_id
        labels = np.unique(true_labels[mask])
        if len(labels) != 1:
            raise ValueError(f"cat_id {cat_id} has conflicting age-group labels: {labels.tolist()}")
        animal_labels[index] = labels[0]
        animal_probabilities[index] = probabilities[mask].mean(axis=0)
    return unique_cats, animal_labels, animal_probabilities


def categorical_metrics(
    true_labels: Sequence[int],
    probabilities: np.ndarray,
) -> dict[str, object]:
    """Calculate the locked categorical metrics for one prediction table."""

    true_labels = np.asarray(true_labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted_labels = probabilities.argmax(axis=1)
    label_indices = np.arange(len(LABELS))
    recalls = recall_score(
        true_labels,
        predicted_labels,
        labels=label_indices,
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=label_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(true_labels, predicted_labels, labels=label_indices, weights="quadratic")
        ),
        "per_class_recall": {
            label: float(value) for label, value in zip(LABELS, recalls, strict=True)
        },
        "confusion_matrix": confusion_matrix(
            true_labels,
            predicted_labels,
            labels=label_indices,
        ).tolist(),
        "n": int(len(true_labels)),
    }


def animal_level_metrics(
    probabilities: np.ndarray,
    cat_ids: Sequence[str],
    true_labels: Sequence[int],
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate to animals and calculate the locked primary metrics."""

    animal_ids, animal_labels, animal_probabilities = aggregate_animal_probabilities(
        probabilities,
        cat_ids,
        true_labels,
    )
    return (
        categorical_metrics(animal_labels, animal_probabilities),
        animal_ids,
        animal_labels,
        animal_probabilities,
    )
