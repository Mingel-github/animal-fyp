"""Run the frozen VGGish/AST outer comparison with animal-level metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from animal_fyp.evaluation import (
    LABELS,
    animal_level_metrics,
    categorical_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "protocol" / "meowagenet_locked_v1.json"
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_nested_roles_v1.csv"
VGGISH_PATH = (
    REPO_ROOT
    / "data"
    / "meowagenet"
    / "official-3d02295bef15"
    / "embeddings"
    / "vggish_looped_embeddings.csv"
)
AST_RUN_ROOT = REPO_ROOT / "runs" / "ast_locked_v1"
RUNS_ROOT = REPO_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ast-subdir",
        default="full",
        help="Subdirectory under runs/ast_locked_v1 containing AST embeddings.",
    )
    parser.add_argument(
        "--output-subdir",
        default="locked_comparison_v1",
        help="Subdirectory under runs for comparison outputs.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_model(input_dimensions: int, config: dict[str, object], seed: int) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dimensions,)),
            tf.keras.layers.Dense(config["hidden_units"], activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(config["dropout"], seed=seed),
            tf.keras.layers.Dense(len(LABELS), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def balanced_class_weights(labels: np.ndarray) -> dict[int, float]:
    classes = np.arange(len(LABELS))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(label): float(weight) for label, weight in zip(classes, weights, strict=True)}


def load_representations(
    analysis_cat_ids: set[str], ast_root: Path
) -> dict[str, dict[str, np.ndarray]]:
    vggish = pd.read_csv(VGGISH_PATH)
    vggish = vggish[vggish["cat_id"].isin(analysis_cat_ids)].reset_index(drop=True)
    feature_columns = [str(index) for index in range(128)]
    target = vggish["target"].to_numpy(dtype=np.float64)
    labels = np.where(target < 0.5, 0, np.where(target < 10, 1, 2)).astype(np.int64)
    representations: dict[str, dict[str, np.ndarray]] = {
        "vggish_mlp": {
            "features": vggish[feature_columns].to_numpy(dtype=np.float32),
            "cat_ids": vggish["cat_id"].astype(str).to_numpy(),
            "labels": labels,
            "unit_ids": np.asarray([f"vggish-row-{index:04d}" for index in range(len(vggish))]),
            "durations": np.full(len(vggish), np.nan, dtype=np.float32),
        }
    }
    for name in ("ast_standard", "ast_time_fine", "ast_frequency_fine"):
        loaded = np.load(ast_root / f"{name}_call_embeddings.npz")
        mask = np.isin(loaded["cat_ids"].astype(str), list(analysis_cat_ids))
        representations[name] = {
            "features": loaded["embeddings"][mask].astype(np.float32),
            "cat_ids": loaded["cat_ids"][mask].astype(str),
            "labels": loaded["labels"][mask].astype(np.int64),
            "unit_ids": loaded["call_ids"][mask].astype(str),
            "durations": loaded["durations"][mask].astype(np.float32),
        }
    return representations


def run_fold(
    representation_name: str,
    data: dict[str, np.ndarray],
    roles: pd.DataFrame,
    outer_fold: int,
    head_config: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame]:
    fold_roles = roles[roles["outer_fold"] == outer_fold]
    role_by_cat = dict(zip(fold_roles["cat_id"], fold_roles["role"], strict=True))
    unit_roles = np.asarray([role_by_cat[cat_id] for cat_id in data["cat_ids"]])
    train_mask = unit_roles == "train"
    validation_mask = unit_roles == "validation"
    test_mask = unit_roles == "test"
    if set(data["cat_ids"][train_mask]) & set(data["cat_ids"][test_mask]):
        raise RuntimeError(f"cat-ID leakage in {representation_name}, fold {outer_fold}")

    inner_scaler = StandardScaler().fit(data["features"][train_mask])
    x_train = inner_scaler.transform(data["features"][train_mask]).astype(np.float32)
    x_validation = inner_scaler.transform(data["features"][validation_mask]).astype(np.float32)
    seed = 42 + outer_fold
    model = build_model(data["features"].shape[1], head_config, seed)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=head_config["early_stopping_patience"],
        restore_best_weights=head_config["restore_best_weights"],
    )
    started = time.perf_counter()
    history = model.fit(
        x_train,
        data["labels"][train_mask],
        validation_data=(x_validation, data["labels"][validation_mask]),
        epochs=head_config["max_epochs"],
        batch_size=head_config["batch_size"],
        class_weight=balanced_class_weights(data["labels"][train_mask]),
        callbacks=[early_stopping],
        verbose=0,
        shuffle=True,
    )
    inner_seconds = time.perf_counter() - started
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    validation_probabilities = model.predict(x_validation, verbose=0)
    validation_metrics, _, _, _ = animal_level_metrics(
        validation_probabilities,
        data["cat_ids"][validation_mask],
        data["labels"][validation_mask],
    )

    outer_train_mask = train_mask | validation_mask
    outer_scaler = StandardScaler().fit(data["features"][outer_train_mask])
    x_outer_train = outer_scaler.transform(data["features"][outer_train_mask]).astype(np.float32)
    x_test = outer_scaler.transform(data["features"][test_mask]).astype(np.float32)
    model = build_model(data["features"].shape[1], head_config, seed)
    started = time.perf_counter()
    model.fit(
        x_outer_train,
        data["labels"][outer_train_mask],
        epochs=best_epoch,
        batch_size=head_config["batch_size"],
        class_weight=balanced_class_weights(data["labels"][outer_train_mask]),
        verbose=0,
        shuffle=True,
    )
    final_train_seconds = time.perf_counter() - started
    test_probabilities = model.predict(x_test, verbose=0)
    fold_animal_metrics, _, _, _ = animal_level_metrics(
        test_probabilities,
        data["cat_ids"][test_mask],
        data["labels"][test_mask],
    )
    prediction_rows = pd.DataFrame(
        {
            "representation": representation_name,
            "outer_fold": outer_fold,
            "unit_id": data["unit_ids"][test_mask],
            "cat_id": data["cat_ids"][test_mask],
            "true_label": data["labels"][test_mask],
            "duration_seconds": data["durations"][test_mask],
            "prob_kitten": test_probabilities[:, 0],
            "prob_adult": test_probabilities[:, 1],
            "prob_senior": test_probabilities[:, 2],
        }
    )
    return (
        {
            "outer_fold": outer_fold,
            "seed": seed,
            "inner_train_units": int(train_mask.sum()),
            "inner_validation_units": int(validation_mask.sum()),
            "outer_test_units": int(test_mask.sum()),
            "best_epoch": best_epoch,
            "stopped_epoch": int(len(history.history["loss"])),
            "best_val_loss": float(min(history.history["val_loss"])),
            "inner_validation_animal_metrics": validation_metrics,
            "outer_test_animal_metrics": fold_animal_metrics,
            "inner_train_seconds": inner_seconds,
            "final_train_seconds": final_train_seconds,
        },
        prediction_rows,
    )


def evaluate_predictions(frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    probability_columns = ["prob_kitten", "prob_adult", "prob_senior"]
    probabilities = frame[probability_columns].to_numpy()
    unit_metrics = categorical_metrics(frame["true_label"].to_numpy(), probabilities)
    animal_metrics, animal_ids, animal_labels, animal_probabilities = animal_level_metrics(
        probabilities,
        frame["cat_id"].to_numpy(),
        frame["true_label"].to_numpy(),
    )
    animal_frame = pd.DataFrame(
        {
            "cat_id": animal_ids,
            "true_label": animal_labels,
            "prob_kitten": animal_probabilities[:, 0],
            "prob_adult": animal_probabilities[:, 1],
            "prob_senior": animal_probabilities[:, 2],
            "predicted_label": animal_probabilities.argmax(axis=1),
        }
    )
    return {"animal_level": animal_metrics, "prediction_unit_level": unit_metrics}, animal_frame


def stratified_paired_bootstrap(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    repeats: int = 2000,
    seed: int = 20260825,
) -> dict[str, float]:
    probability_columns = ["prob_kitten", "prob_adult", "prob_senior"]
    merged = reference.merge(candidate, on=["cat_id", "true_label"], suffixes=("_ref", "_cand"))
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(merged["true_label"].to_numpy() == label) for label in range(3)]
    differences = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        labels = merged["true_label"].to_numpy()[sampled]
        ref_probabilities = merged[[f"{column}_ref" for column in probability_columns]].to_numpy()[sampled]
        candidate_probabilities = merged[
            [f"{column}_cand" for column in probability_columns]
        ].to_numpy()[sampled]
        differences[repeat] = (
            categorical_metrics(labels, candidate_probabilities)["macro_f1"]
            - categorical_metrics(labels, ref_probabilities)["macro_f1"]
        )
    return {
        "repeats": repeats,
        "seed": seed,
        "ci_lower": float(np.quantile(differences, 0.025)),
        "ci_upper": float(np.quantile(differences, 0.975)),
        "bootstrap_mean": float(differences.mean()),
    }


def main() -> None:
    args = parse_args()
    ast_root = AST_RUN_ROOT / args.ast_subdir
    run_root = RUNS_ROOT / args.output_subdir
    summary_path = run_root / "comparison_summary.json"
    tf.config.threading.set_intra_op_parallelism_threads(min(6, os.cpu_count() or 1))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    analysis_cat_ids = set(roles["cat_id"])
    if len(analysis_cat_ids) != protocol["dataset"]["expected_cats"]:
        raise ValueError("Nested role manifest has the wrong number of cats")
    representations = load_representations(analysis_cat_ids, ast_root)
    expected_units = {
        "vggish_mlp": 936,
        "ast_standard": 792,
        "ast_time_fine": 792,
        "ast_frequency_fine": 792,
    }
    for name, expected in expected_units.items():
        if len(representations[name]["features"]) != expected:
            raise ValueError(f"Expected {expected} units for {name}")

    run_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}
    animal_frames: dict[str, pd.DataFrame] = {}
    for representation_name, data in representations.items():
        print(f"Training locked head for {representation_name}", flush=True)
        fold_results = []
        fold_predictions = []
        for outer_fold in range(protocol["splits"]["outer_folds"]):
            fold_result, predictions = run_fold(
                representation_name,
                data,
                roles,
                outer_fold,
                protocol["classifier_head"],
            )
            fold_results.append(fold_result)
            fold_predictions.append(predictions)
            print(
                f"{representation_name} fold {outer_fold}: "
                f"inner F1={fold_result['inner_validation_animal_metrics']['macro_f1']:.4f}, "
                f"outer F1={fold_result['outer_test_animal_metrics']['macro_f1']:.4f}, "
                f"best epoch={fold_result['best_epoch']}",
                flush=True,
            )
        all_predictions = pd.concat(fold_predictions, ignore_index=True)
        metrics, animal_frame = evaluate_predictions(all_predictions)
        all_predictions.to_csv(run_root / f"{representation_name}_unit_predictions.csv", index=False)
        animal_frame.to_csv(run_root / f"{representation_name}_animal_predictions.csv", index=False)
        results[representation_name] = {"folds": fold_results, "overall": metrics}
        prediction_frames[representation_name] = all_predictions
        animal_frames[representation_name] = animal_frame

    ast_names = ["ast_standard", "ast_time_fine", "ast_frequency_fine"]
    selected_by_fold: dict[int, str] = {}
    selected_prediction_parts = []
    for outer_fold in range(protocol["splits"]["outer_folds"]):
        selected = max(
            ast_names,
            key=lambda name: results[name]["folds"][outer_fold]["inner_validation_animal_metrics"][
                "macro_f1"
            ],
        )
        selected_by_fold[outer_fold] = selected
        frame = prediction_frames[selected]
        selected_prediction_parts.append(frame[frame["outer_fold"] == outer_fold])
    selected_predictions = pd.concat(selected_prediction_parts, ignore_index=True)
    selected_metrics, selected_animals = evaluate_predictions(selected_predictions)
    selected_predictions.to_csv(run_root / "ast_nested_selected_unit_predictions.csv", index=False)
    selected_animals.to_csv(run_root / "ast_nested_selected_animal_predictions.csv", index=False)
    results["ast_nested_selected"] = {
        "selected_by_fold": selected_by_fold,
        "overall": selected_metrics,
    }
    animal_frames["ast_nested_selected"] = selected_animals

    contrasts = {}
    for contrast_name, candidate_name, reference_name in (
        ("time_fine_minus_standard", "ast_time_fine", "ast_standard"),
        ("nested_ast_minus_vggish", "ast_nested_selected", "vggish_mlp"),
        ("frequency_fine_minus_standard", "ast_frequency_fine", "ast_standard"),
    ):
        observed = (
            results[candidate_name]["overall"]["animal_level"]["macro_f1"]
            - results[reference_name]["overall"]["animal_level"]["macro_f1"]
        )
        contrasts[contrast_name] = {
            "candidate": candidate_name,
            "reference": reference_name,
            "observed_macro_f1_difference": float(observed),
            "practical_delta": protocol["evaluation"]["practical_delta_macro_f1"],
            "paired_stratified_animal_bootstrap": stratified_paired_bootstrap(
                animal_frames[reference_name], animal_frames[candidate_name]
            ),
        }

    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_config_sha256": sha256(CONFIG_PATH),
        "nested_roles_sha256": sha256(ROLES_PATH),
        "vggish_csv_sha256": sha256(VGGISH_PATH),
        "representations": results,
        "contrasts": contrasts,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
