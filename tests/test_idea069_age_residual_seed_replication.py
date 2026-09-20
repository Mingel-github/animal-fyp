from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_idea069_age_residual_seed_replication as runner  # noqa: E402


def test_protocol_locks_three_outcome_independent_seeds_and_72_fits() -> None:
    protocol = json.loads(runner.PROTOCOL_PATH.read_text(encoding="utf-8"))
    runner.verify_protocol(protocol)
    assert tuple(protocol["model"]["base_seeds"]) == runner.BASE_SEEDS
    assert protocol["model"]["outer_test_predictions"] is False
    assert protocol["model"]["total_fits"] == 72
    full_seeds = {
        base + 10_000 * repeat + 100 * fold
        for base in runner.BASE_SEEDS
        for repeat in range(3)
        for fold in range(4)
    }
    assert len(full_seeds) == 36


def test_real_data_zero_initialized_pair_has_identical_logits() -> None:
    protocol = json.loads(runner.PROTOCOL_PATH.read_text(encoding="utf-8"))
    store = runner.idea068.idea051.reference.historical.idea019.load_feature_store()
    features, _ = runner.load_features(protocol, store.call_ids)
    roles = pd.read_csv(
        REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str}
    )
    indices = runner.idea068.idea051.reference.historical.fold_indices(
        store, roles, 0, 0, include_test=False
    )
    difference = runner.initial_logit_max_difference(
        protocol,
        store,
        features,
        indices["train"],
        indices["validation"][:32],
        runner.BASE_SEEDS[0],
    )
    assert difference == 0.0
    assert len(store.call_ids) == 792
    assert len(np.unique(store.cat_ids.astype(str))) == 111

