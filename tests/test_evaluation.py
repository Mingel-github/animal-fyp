from __future__ import annotations

import numpy as np
import pytest

from animal_fyp.evaluation import animal_level_metrics


def test_animal_aggregation_gives_each_cat_one_vote() -> None:
    probabilities = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
        ]
    )
    metrics, cat_ids, labels, animal_probabilities = animal_level_metrics(
        probabilities,
        ["cat-a", "cat-a", "cat-b"],
        [0, 0, 1],
    )
    assert cat_ids.tolist() == ["cat-a", "cat-b"]
    assert labels.tolist() == [0, 1]
    np.testing.assert_allclose(animal_probabilities[0], [0.8, 0.125, 0.075])
    assert metrics["n"] == 2


def test_conflicting_labels_within_cat_are_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting"):
        animal_level_metrics(
            np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]),
            ["same-cat", "same-cat"],
            [0, 1],
        )
