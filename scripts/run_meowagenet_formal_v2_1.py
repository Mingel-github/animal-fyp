"""Run the MeowAgeNet formal-v2.1 core without crossing the protocol boundary.

``inner-only`` trains and evaluates only the nested train/validation roles.  It is
intended for runner development and smoke tests.  ``formal`` is the only scope
that can produce outer-test predictions, and it requires a completed execution
lock whose recipe and runner hashes match the files being executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for local_root in (SCRIPTS_ROOT, SRC_ROOT):
    if str(local_root) not in sys.path:
        sys.path.insert(0, str(local_root))

from animal_fyp.evaluation import (  # noqa: E402
    LABELS,
    animal_level_metrics,
    categorical_metrics,
)
import run_idea019_peft_placement as idea019  # noqa: E402


FORMAL_CONFIG_PATH = (
    REPO_ROOT / "configs" / "protocol" / "meowagenet_formal_v2_1.json"
)
PARENT_CONFIG_PATH = REPO_ROOT / "configs" / "protocol" / "meowagenet_formal_v2.json"
LOCKED_V1_CONFIG_PATH = (
    REPO_ROOT / "configs" / "protocol" / "meowagenet_locked_v1.json"
)
DEFAULT_RECIPE_PATH = (
    REPO_ROOT
    / "configs"
    / "experiment"
    / "meowagenet_formal_v2_1_probe_guided_candidate_v1.json"
)
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
VGGISH_PATH = (
    REPO_ROOT
    / "data"
    / "meowagenet"
    / "official-3d02295bef15"
    / "embeddings"
    / "vggish_looped_embeddings.csv"
)
RUNS_ROOT = REPO_ROOT / "runs"
CORE_PIPELINES = (
    "vggish_mlp",
    "ast_head_only",
    "ast_probe_guided_adapter",
)
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the frozen MeowAgeNet formal-v2.1 core."
    )
    parser.add_argument("--scope", choices=("inner-only", "formal"), required=True)
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument("--recipe-path", default=str(DEFAULT_RECIPE_PATH))
    parser.add_argument("--execution-lock")
    parser.add_argument("--pipelines")
    parser.add_argument("--repeats")
    parser.add_argument("--folds")
    parser.add_argument("--base-seeds")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--smoke-max-epochs",
        type=int,
        help="Inner-only override applied to both VGGish and AST epoch limits.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def relative_repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def parse_csv_values(value: str | None, cast: type = int) -> list[Any] | None:
    if value is None:
        return None
    parsed = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("A supplied comma-separated selection is empty")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"A supplied selection contains duplicates: {parsed}")
    return parsed


def full_model_seed(base_seed: int, repeat: int, outer_fold: int) -> int:
    return int(base_seed + 10000 * repeat + 100 * outer_fold)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def verify_entries(entries: list[dict[str, str]], label: str) -> None:
    for entry in entries:
        path = repo_path(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise RuntimeError(
                f"{label} checksum mismatch for {path}: "
                f"expected {entry['sha256']}, got {actual}"
            )


def verify_recipe(recipe_path: Path) -> dict[str, Any]:
    recipe = read_json(recipe_path)
    if recipe["protocol_id"] != "meowagenet-formal-v2.1":
        raise ValueError("Recipe protocol ID does not match formal-v2.1")
    if tuple(recipe["core_pipelines"]) != CORE_PIPELINES:
        raise ValueError("Recipe core pipelines differ from the formal-v2.1 core")
    if recipe["candidate_pipeline"] != "ast_probe_guided_adapter":
        raise ValueError("This runner revision implements the probe-guided candidate")
    verify_entries(recipe["versioned_dependencies"], "versioned dependency")
    verify_entries(recipe["local_artifacts"], "local artifact")
    return recipe


def find_null(value: Any, prefix: str = "") -> list[str]:
    if value is None:
        return [prefix or "<root>"]
    if isinstance(value, dict):
        found: list[str] = []
        for key, child in value.items():
            found.extend(find_null(child, f"{prefix}.{key}" if prefix else key))
        return found
    if isinstance(value, list):
        found = []
        for index, child in enumerate(value):
            found.extend(find_null(child, f"{prefix}[{index}]"))
        return found
    return []


def verify_execution_lock(
    lock_path: Path,
    recipe_path: Path,
    runner_path: Path,
) -> dict[str, Any]:
    lock = read_json(lock_path)
    if lock.get("protocol_id") != "meowagenet-formal-v2.1":
        raise ValueError("Execution lock protocol ID does not match formal-v2.1")
    if lock.get("status") != "locked_before_formal_outcomes":
        raise RuntimeError("Formal scope requires status=locked_before_formal_outcomes")
    if lock.get("formal_outcomes_accessed_before_lock") is not False:
        raise RuntimeError("Execution lock records prior formal outcome access")
    null_fields = find_null(lock)
    if null_fields:
        raise RuntimeError(f"Execution lock still contains null fields: {null_fields}")
    selected = lock["selected_primary_adapter"]
    if selected["pipeline_id"] != "ast_probe_guided_adapter":
        raise RuntimeError("The locked adapter is unsupported by this runner revision")
    locked_recipe_path = repo_path(selected["recipe_path"]).resolve()
    if locked_recipe_path != recipe_path.resolve():
        raise RuntimeError("Command recipe differs from the execution-lock recipe")
    if sha256(recipe_path) != selected["recipe_sha256"]:
        raise RuntimeError("Execution-lock recipe checksum does not match")
    runner = lock["runner"]
    locked_runner_path = repo_path(runner["path"]).resolve()
    if locked_runner_path != runner_path.resolve():
        raise RuntimeError("Execution lock names a different runner")
    if sha256(runner_path) != runner["sha256"]:
        raise RuntimeError("Execution-lock runner checksum does not match")
    if lock["enabled_optional_modules"]:
        raise RuntimeError(
            "This core runner revision requires enabled_optional_modules=[]; "
            "add optional-module implementations before locking them"
        )
    return lock


def choose_subset(
    requested: list[Any] | None,
    allowed: list[Any],
    default: list[Any],
    label: str,
) -> list[Any]:
    selected = default if requested is None else requested
    unexpected = set(selected) - set(allowed)
    if unexpected:
        raise ValueError(f"Unsupported {label}: {sorted(unexpected)}")
    return list(selected)


def load_vggish(analysis_cat_ids: set[str]) -> dict[str, np.ndarray]:
    frame = pd.read_csv(VGGISH_PATH, dtype={"cat_id": str})
    frame = frame[frame["cat_id"].isin(analysis_cat_ids)].reset_index(drop=True)
    feature_columns = [str(index) for index in range(128)]
    target = frame["target"].to_numpy(dtype=np.float64)
    labels = np.where(target < 0.5, 0, np.where(target < 10, 1, 2)).astype(np.int64)
    if len(frame) != 936 or frame["cat_id"].nunique() != 111:
        raise RuntimeError("Unexpected VGGish analysis scope")
    return {
        "features": frame[feature_columns].to_numpy(dtype=np.float32),
        "cat_ids": frame["cat_id"].to_numpy(dtype=str),
        "labels": labels,
        "unit_ids": np.asarray([f"vggish-row-{index:04d}" for index in range(len(frame))]),
    }


def balanced_class_weights(labels: np.ndarray) -> dict[int, float]:
    classes = np.arange(len(LABELS))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {
        int(label): float(weight)
        for label, weight in zip(classes, weights, strict=True)
    }


def build_vggish_model(recipe: dict[str, Any], seed: int) -> tf.keras.Model:
    config = recipe["vggish_mlp"]
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(int(config["features"]),)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(float(config["dropout"]), seed=seed),
            tf.keras.layers.Dense(len(LABELS), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(
            learning_rate=float(config["learning_rate"])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def animal_prediction_frame(
    unit_frame: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    probabilities = unit_frame[list(PROBABILITY_COLUMNS)].to_numpy()
    metrics, cat_ids, labels, animal_probabilities = animal_level_metrics(
        probabilities,
        unit_frame["cat_id"].to_numpy(),
        unit_frame["true_label"].to_numpy(),
    )
    animals = pd.DataFrame(
        {
            "cat_id": cat_ids,
            "true_label": labels,
            "prob_kitten": animal_probabilities[:, 0],
            "prob_adult": animal_probabilities[:, 1],
            "prob_senior": animal_probabilities[:, 2],
            "predicted_label": animal_probabilities.argmax(axis=1),
        }
    )
    return metrics, animals


def role_by_cat(roles: pd.DataFrame, repeat: int, outer_fold: int) -> dict[str, str]:
    selected = roles[
        (roles["repeat"] == repeat) & (roles["outer_fold"] == outer_fold)
    ]
    if len(selected) != 111 or selected["cat_id"].nunique() != 111:
        raise RuntimeError(f"Incomplete role set for repeat={repeat}, fold={outer_fold}")
    mapping = dict(zip(selected["cat_id"], selected["role"], strict=True))
    if set(mapping.values()) != {"train", "validation", "test"}:
        raise RuntimeError("Nested roles do not contain train, validation, and test")
    return mapping


def vggish_masks(
    store: dict[str, np.ndarray],
    roles: pd.DataFrame,
    repeat: int,
    outer_fold: int,
    include_test: bool,
) -> dict[str, np.ndarray]:
    mapping = role_by_cat(roles, repeat, outer_fold)
    unit_roles = np.asarray([mapping[cat_id] for cat_id in store["cat_ids"]])
    masks = {
        "train": unit_roles == "train",
        "validation": unit_roles == "validation",
    }
    if include_test:
        masks["test"] = unit_roles == "test"
    return masks


def vggish_unit_frame(
    store: dict[str, np.ndarray], mask: np.ndarray, probabilities: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": store["unit_ids"][mask],
            "cat_id": store["cat_ids"][mask],
            "true_label": store["labels"][mask],
            "prob_kitten": probabilities[:, 0],
            "prob_adult": probabilities[:, 1],
            "prob_senior": probabilities[:, 2],
        }
    )


def fit_vggish(
    scope: str,
    recipe: dict[str, Any],
    store: dict[str, np.ndarray],
    roles: pd.DataFrame,
    repeat: int,
    outer_fold: int,
    seed: int,
    smoke_max_epochs: int | None,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    masks = vggish_masks(
        store, roles, repeat, outer_fold, include_test=scope == "formal"
    )
    config = recipe["vggish_mlp"]
    maximum_epochs = int(config["maximum_epochs"])
    if smoke_max_epochs is not None:
        maximum_epochs = min(maximum_epochs, smoke_max_epochs)
    scaler = StandardScaler().fit(store["features"][masks["train"]])
    x_train = scaler.transform(store["features"][masks["train"]]).astype(np.float32)
    x_validation = scaler.transform(store["features"][masks["validation"]]).astype(
        np.float32
    )
    model = build_vggish_model(recipe, seed)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=int(config["early_stopping_patience"]),
        restore_best_weights=True,
    )
    started = time.perf_counter()
    history = model.fit(
        x_train,
        store["labels"][masks["train"]],
        validation_data=(x_validation, store["labels"][masks["validation"]]),
        epochs=maximum_epochs,
        batch_size=int(config["batch_size"]),
        class_weight=balanced_class_weights(store["labels"][masks["train"]]),
        callbacks=[callback],
        verbose=0,
        shuffle=True,
    )
    inner_seconds = time.perf_counter() - started
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    validation_probabilities = model.predict(x_validation, verbose=0)
    validation_frame = vggish_unit_frame(
        store, masks["validation"], validation_probabilities
    )
    validation_metrics, _ = animal_prediction_frame(validation_frame)
    audit: dict[str, Any] = {
        "best_epoch": best_epoch,
        "stopped_epoch": len(history.history["loss"]),
        "best_validation_loss": float(min(history.history["val_loss"])),
        "best_validation_animal_metrics": validation_metrics,
        "history": [
            {
                "epoch": index + 1,
                "train_loss": float(history.history["loss"][index]),
                "validation_loss": float(history.history["val_loss"][index]),
            }
            for index in range(len(history.history["loss"]))
        ],
        "train_seconds": inner_seconds,
        "inner_train_units": int(masks["train"].sum()),
        "inner_validation_units": int(masks["validation"].sum()),
    }
    if scope == "inner-only":
        return audit, None

    outer_train_mask = masks["train"] | masks["validation"]
    outer_scaler = StandardScaler().fit(store["features"][outer_train_mask])
    x_outer_train = outer_scaler.transform(store["features"][outer_train_mask]).astype(
        np.float32
    )
    x_test = outer_scaler.transform(store["features"][masks["test"]]).astype(np.float32)
    model = build_vggish_model(recipe, seed)
    started = time.perf_counter()
    model.fit(
        x_outer_train,
        store["labels"][outer_train_mask],
        epochs=best_epoch,
        batch_size=int(config["batch_size"]),
        class_weight=balanced_class_weights(store["labels"][outer_train_mask]),
        verbose=0,
        shuffle=True,
    )
    final_train_seconds = time.perf_counter() - started
    test_probabilities = model.predict(x_test, verbose=0)
    test_frame = vggish_unit_frame(store, masks["test"], test_probabilities)
    test_metrics, _ = animal_prediction_frame(test_frame)
    audit["outer"] = {
        "epochs": best_epoch,
        "outer_train_units": int(outer_train_mask.sum()),
        "outer_test_units": int(masks["test"].sum()),
        "train_and_predict_seconds": final_train_seconds,
        "test_animal_metrics": test_metrics,
    }
    return audit, test_frame


def ast_indices(
    store: Any,
    roles: pd.DataFrame,
    repeat: int,
    outer_fold: int,
    include_test: bool,
) -> dict[str, np.ndarray]:
    mapping = role_by_cat(roles, repeat, outer_fold)
    call_roles = np.asarray([mapping[cat_id] for cat_id in store.cat_ids])
    selected = {
        "train": np.flatnonzero(call_roles == "train"),
        "validation": np.flatnonzero(call_roles == "validation"),
    }
    if include_test:
        selected["test"] = np.flatnonzero(call_roles == "test")
    return selected


def ast_training_args(
    recipe: dict[str, Any], smoke_max_epochs: int | None
) -> SimpleNamespace:
    shared = recipe["ast_shared"]
    maximum_epochs = int(shared["maximum_epochs"])
    if smoke_max_epochs is not None:
        maximum_epochs = min(maximum_epochs, smoke_max_epochs)
    return SimpleNamespace(
        adapter_width=int(recipe["ast_probe_guided_adapter"]["adapter_width"]),
        adapter_learning_rate=float(
            recipe["ast_probe_guided_adapter"]["adapter_learning_rate"]
        ),
        head_learning_rate=float(shared["head_learning_rate"]),
        batch_size=int(shared["micro_batch_size"]),
        accumulation_steps=int(shared["gradient_accumulation_steps"]),
        max_epochs=maximum_epochs,
        patience=int(shared["early_stopping_patience"]),
        gradient_clip=float(shared["gradient_clip"]),
        probe_cv_folds=3,
    )


def load_or_compute_probe(
    run_root: Path,
    layer_store: Any,
    train_indices: np.ndarray,
    repeat: int,
    outer_fold: int,
    probe_seed: int,
) -> dict[str, Any]:
    path = run_root / "probes" / f"repeat_{repeat}_fold_{outer_fold}.json"
    if path.exists():
        result = read_json(path)
        if result["seed"] != probe_seed:
            raise RuntimeError(f"Existing probe seed differs: {path}")
        return result
    result = idea019.probe_layer_utilities(
        layer_store, train_indices, cv_folds=3, seed=probe_seed
    )
    write_json(path, result)
    return result


def fit_ast(
    scope: str,
    pipeline: str,
    recipe: dict[str, Any],
    locked_model_protocol: dict[str, Any],
    store: Any,
    indices: dict[str, np.ndarray],
    probe_result: dict[str, Any] | None,
    device: torch.device,
    seed: int,
    smoke_max_epochs: int | None,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    if pipeline == "ast_head_only":
        mode = "head_only"
        placement = None
    elif pipeline == "ast_probe_guided_adapter":
        mode = "adapter_probe_guided"
        if probe_result is None:
            raise RuntimeError("Probe-guided adapter requires a layer-probe result")
        placement = idea019.placement_for_mode(mode, probe_result)
    else:
        raise ValueError(f"Unsupported AST pipeline: {pipeline}")
    training_args = ast_training_args(recipe, smoke_max_epochs)
    best_epoch, inner_audit = idea019.fit_inner(
        mode,
        placement,
        locked_model_protocol,
        store,
        indices["train"],
        indices["validation"],
        training_args,
        device,
        seed,
    )
    audit: dict[str, Any] = {
        "placement_layers_zero_based": list(placement) if placement else [],
        "placement_layers_one_based": (
            [layer + 1 for layer in placement] if placement else []
        ),
        "inner_train_calls": int(len(indices["train"])),
        "inner_validation_calls": int(len(indices["validation"])),
        "inner": inner_audit,
    }
    if scope == "inner-only":
        return audit, None
    outer_train_indices = np.concatenate((indices["train"], indices["validation"]))
    test_frame, outer_audit = idea019.fit_outer_and_predict(
        mode,
        placement,
        locked_model_protocol,
        store,
        outer_train_indices,
        indices["test"],
        best_epoch,
        training_args,
        device,
        seed,
    )
    audit["outer_test_calls"] = int(len(indices["test"]))
    audit["outer"] = outer_audit
    return audit, test_frame


def fit_directory(
    run_root: Path, pipeline: str, repeat: int, outer_fold: int, base_seed: int
) -> Path:
    return (
        run_root
        / "fits"
        / pipeline
        / f"repeat_{repeat}"
        / f"fold_{outer_fold}"
        / f"base_seed_{base_seed}"
    )


def add_prediction_identity(
    frame: pd.DataFrame,
    pipeline: str,
    repeat: int,
    outer_fold: int,
    base_seed: int,
    seed: int,
) -> pd.DataFrame:
    frame = frame.copy()
    for position, (column, value) in enumerate(
        (
            ("pipeline", pipeline),
            ("repeat", repeat),
            ("outer_fold", outer_fold),
            ("base_seed", base_seed),
            ("full_seed", seed),
        )
    ):
        frame.insert(position, column, value)
    return frame


def prepare_run_root(run_root: Path, manifest: dict[str, Any], resume: bool) -> None:
    manifest_path = run_root / "run_manifest.json"
    if run_root.exists():
        if not resume:
            raise FileExistsError(
                f"Run directory already exists; pass --resume after reviewing it: {run_root}"
            )
        if not manifest_path.is_file():
            raise RuntimeError("Existing run directory has no run_manifest.json")
        existing = read_json(manifest_path)
        keys = (
            "protocol_id",
            "scope",
            "recipe_sha256",
            "runner_sha256",
            "roles_sha256",
            "selected_pipelines",
            "selected_repeats",
            "selected_folds",
            "selected_base_seeds",
            "execution_lock_sha256",
        )
        differences = [key for key in keys if existing.get(key) != manifest.get(key)]
        if differences:
            raise RuntimeError(f"Resume manifest differs in fields: {differences}")
        return
    run_root.mkdir(parents=True)
    write_json(manifest_path, manifest)


def collect_oof(
    run_root: Path,
    pipelines: list[str],
    repeats: list[int],
    base_seeds: list[int],
) -> tuple[dict[str, Any], dict[str, dict[tuple[int, int], pd.DataFrame]]]:
    oof_results: dict[str, Any] = {}
    animal_sets: dict[str, dict[tuple[int, int], pd.DataFrame]] = {
        pipeline: {} for pipeline in pipelines
    }
    for pipeline in pipelines:
        pipeline_results: dict[str, Any] = {}
        for repeat in repeats:
            for base_seed in base_seeds:
                parts = []
                for outer_fold in range(4):
                    path = (
                        fit_directory(
                            run_root, pipeline, repeat, outer_fold, base_seed
                        )
                        / "outer_test_unit_predictions.csv"
                    )
                    if not path.is_file():
                        raise RuntimeError(f"Missing formal prediction artifact: {path}")
                    parts.append(pd.read_csv(path, dtype={"cat_id": str}))
                units = pd.concat(parts, ignore_index=True)
                metrics, animals = animal_prediction_frame(units)
                if animals["cat_id"].nunique() != 111 or len(animals) != 111:
                    raise RuntimeError("A complete formal OOF set must contain 111 cats")
                animals.insert(0, "base_seed", base_seed)
                animals.insert(0, "repeat", repeat)
                animals.insert(0, "pipeline", pipeline)
                output_path = (
                    run_root
                    / "oof"
                    / pipeline
                    / f"repeat_{repeat}_base_seed_{base_seed}_animal_predictions.csv"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                animals.to_csv(output_path, index=False)
                animal_sets[pipeline][(repeat, base_seed)] = animals
                pipeline_results[f"repeat_{repeat}_base_seed_{base_seed}"] = metrics
        oof_results[pipeline] = pipeline_results
    return oof_results, animal_sets


def macro_f1(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(categorical_metrics(labels, probabilities)["macro_f1"])


def hierarchical_paired_bootstrap(
    reference_sets: dict[tuple[int, int], pd.DataFrame],
    candidate_sets: dict[tuple[int, int], pd.DataFrame],
    repeats: list[int],
    base_seeds: list[int],
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    first_key = (repeats[0], base_seeds[0])
    canonical = reference_sets[first_key].sort_values("cat_id")
    cat_ids = canonical["cat_id"].to_numpy(dtype=str)
    labels = canonical["true_label"].to_numpy(dtype=np.int64)

    def probability_cube(
        frames: dict[tuple[int, int], pd.DataFrame]
    ) -> np.ndarray:
        cube = np.empty((len(repeats), len(base_seeds), len(cat_ids), 3), np.float64)
        for repeat_index, repeat in enumerate(repeats):
            for seed_index, base_seed in enumerate(base_seeds):
                frame = frames[(repeat, base_seed)].sort_values("cat_id")
                if not np.array_equal(frame["cat_id"].to_numpy(dtype=str), cat_ids):
                    raise RuntimeError("Paired OOF cat order differs between evaluations")
                if not np.array_equal(
                    frame["true_label"].to_numpy(dtype=np.int64), labels
                ):
                    raise RuntimeError("Paired OOF labels differ between evaluations")
                cube[repeat_index, seed_index] = frame[
                    list(PROBABILITY_COLUMNS)
                ].to_numpy(dtype=np.float64)
        return cube

    reference = probability_cube(reference_sets)
    candidate = probability_cube(candidate_sets)
    cell_differences = []
    for repeat_index, repeat in enumerate(repeats):
        for seed_index, base_seed in enumerate(base_seeds):
            difference = macro_f1(labels, candidate[repeat_index, seed_index]) - macro_f1(
                labels, reference[repeat_index, seed_index]
            )
            cell_differences.append(
                {
                    "repeat": repeat,
                    "base_seed": base_seed,
                    "candidate_minus_reference_macro_f1": difference,
                }
            )
    rng = np.random.default_rng(bootstrap_seed)
    class_indices = [np.flatnonzero(labels == label) for label in range(3)]
    bootstrap_values = np.empty(bootstrap_repeats, dtype=np.float64)
    for bootstrap_index in range(bootstrap_repeats):
        sampled_cats = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        sampled_repeats = rng.integers(0, len(repeats), size=len(repeats))
        sampled_seeds = rng.integers(0, len(base_seeds), size=len(base_seeds))
        differences = []
        sampled_labels = labels[sampled_cats]
        for repeat_index in sampled_repeats:
            for seed_index in sampled_seeds:
                differences.append(
                    macro_f1(
                        sampled_labels,
                        candidate[repeat_index, seed_index, sampled_cats],
                    )
                    - macro_f1(
                        sampled_labels,
                        reference[repeat_index, seed_index, sampled_cats],
                    )
                )
        bootstrap_values[bootstrap_index] = float(np.mean(differences))
    observed_values = np.asarray(
        [row["candidate_minus_reference_macro_f1"] for row in cell_differences]
    )
    return {
        "complete_oof_differences": cell_differences,
        "mean_candidate_minus_reference_macro_f1": float(observed_values.mean()),
        "positive_complete_oof_evaluations": int((observed_values > 0).sum()),
        "total_complete_oof_evaluations": int(len(observed_values)),
        "hierarchical_paired_bootstrap": {
            "repeats": bootstrap_repeats,
            "seed": bootstrap_seed,
            "ci_lower": float(np.quantile(bootstrap_values, 0.025)),
            "ci_upper": float(np.quantile(bootstrap_values, 0.975)),
            "bootstrap_mean": float(bootstrap_values.mean()),
        },
    }


def aggregate_formal_run(
    run_root: Path,
    recipe: dict[str, Any],
    pipelines: list[str],
    repeats: list[int],
    folds: list[int],
    base_seeds: list[int],
) -> dict[str, Any] | None:
    if folds != [0, 1, 2, 3]:
        return None
    oof_results, animal_sets = collect_oof(
        run_root, pipelines, repeats, base_seeds
    )
    summary: dict[str, Any] = {
        "status": "complete_for_selected_scope",
        "oof_metrics": oof_results,
    }
    if set(CORE_PIPELINES) <= set(pipelines):
        analysis = recipe["analysis"]
        summary["contrasts"] = {
            "H048_adapter_minus_vggish": hierarchical_paired_bootstrap(
                animal_sets["vggish_mlp"],
                animal_sets["ast_probe_guided_adapter"],
                repeats,
                base_seeds,
                int(analysis["hierarchical_bootstrap_repeats"]),
                int(analysis["hierarchical_bootstrap_seed"]),
            ),
            "H019_adapter_minus_ast_head_only": hierarchical_paired_bootstrap(
                animal_sets["ast_head_only"],
                animal_sets["ast_probe_guided_adapter"],
                repeats,
                base_seeds,
                int(analysis["hierarchical_bootstrap_repeats"]),
                int(analysis["hierarchical_bootstrap_seed"]),
            ),
        }
    write_json(run_root / "formal_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    runner_path = Path(__file__).resolve()
    recipe_path = repo_path(args.recipe_path).resolve()
    formal_config = read_json(FORMAL_CONFIG_PATH)
    parent_config = read_json(PARENT_CONFIG_PATH)
    locked_v1_config = read_json(LOCKED_V1_CONFIG_PATH)
    recipe = verify_recipe(recipe_path)
    if formal_config["protocol_id"] != "meowagenet-formal-v2.1":
        raise ValueError("Unexpected formal protocol")

    execution_lock: dict[str, Any] | None = None
    lock_path: Path | None = None
    if args.scope == "formal":
        if args.smoke_max_epochs is not None:
            raise ValueError("--smoke-max-epochs is available only in inner-only scope")
        if args.execution_lock is None:
            raise RuntimeError("Formal scope requires --execution-lock")
        lock_path = repo_path(args.execution_lock).resolve()
        execution_lock = verify_execution_lock(lock_path, recipe_path, runner_path)
    elif args.execution_lock is not None:
        raise ValueError("Inner-only scope does not consume an execution lock")
    if args.smoke_max_epochs is not None and args.smoke_max_epochs < 1:
        raise ValueError("--smoke-max-epochs must be positive")

    requested_pipelines = parse_csv_values(args.pipelines, str)
    pipelines = choose_subset(
        requested_pipelines,
        list(CORE_PIPELINES),
        list(CORE_PIPELINES),
        "pipelines",
    )
    requested_repeats = parse_csv_values(args.repeats)
    requested_folds = parse_csv_values(args.folds)
    requested_base_seeds = parse_csv_values(args.base_seeds)
    if execution_lock is None:
        allowed_repeats = list(formal_config["split_bank"]["available_repeat_indices"])
        allowed_base_seeds = list(formal_config["formal_core"]["model_seeds"])
        repeats = choose_subset(requested_repeats, allowed_repeats, [0], "repeats")
        folds = choose_subset(requested_folds, [0, 1, 2, 3], [0], "folds")
        base_seeds = choose_subset(
            requested_base_seeds, allowed_base_seeds, [allowed_base_seeds[0]], "seeds"
        )
    else:
        locked_repeats = list(execution_lock["selected_repeat_indices"])
        locked_base_seeds = list(execution_lock["model_seeds"])
        repeats = choose_subset(
            requested_repeats, locked_repeats, locked_repeats, "repeats"
        )
        folds = choose_subset(requested_folds, [0, 1, 2, 3], [0, 1, 2, 3], "folds")
        base_seeds = choose_subset(
            requested_base_seeds,
            locked_base_seeds,
            locked_base_seeds,
            "seeds",
        )
    repeats = sorted(repeats)
    folds = sorted(folds)
    base_seeds = sorted(base_seeds)

    run_root = (RUNS_ROOT / args.output_subdir).resolve()
    if RUNS_ROOT.resolve() not in run_root.parents:
        raise ValueError("--output-subdir must stay below the repository runs directory")
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    analysis_cat_ids = set(roles["cat_id"])
    if len(analysis_cat_ids) != 111:
        raise RuntimeError("Formal roles must contain 111 cats")

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as error:
        raise RuntimeError("TensorFlow device state was initialized too early") from error
    tf.config.threading.set_intra_op_parallelism_threads(min(6, os.cpu_count() or 1))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    device = resolve_device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    manifest = {
        "schema_version": "1.0",
        "protocol_id": formal_config["protocol_id"],
        "scope": args.scope,
        "outer_test_accessed": args.scope == "formal",
        "recipe_path": relative_repo_path(recipe_path),
        "recipe_sha256": sha256(recipe_path),
        "runner_path": relative_repo_path(runner_path),
        "runner_sha256": sha256(runner_path),
        "git_revision_at_start": git_revision(),
        "roles_sha256": sha256(ROLES_PATH),
        "execution_lock_path": (
            relative_repo_path(lock_path) if lock_path is not None else None
        ),
        "execution_lock_sha256": sha256(lock_path) if lock_path is not None else None,
        "selected_pipelines": pipelines,
        "selected_repeats": repeats,
        "selected_folds": folds,
        "selected_base_seeds": base_seeds,
        "smoke_max_epochs": args.smoke_max_epochs,
        "seed_rule": parent_config["model_randomness"]["full_seed_rule"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tensorflow": tf.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": device_name,
        },
    }
    prepare_run_root(run_root, manifest, args.resume)
    print(
        f"formal-v2.1 scope={args.scope}; device={device} ({device_name}); "
        f"pipelines={pipelines}; repeats={repeats}; folds={folds}; seeds={base_seeds}",
        flush=True,
    )

    vggish_store = load_vggish(analysis_cat_ids) if "vggish_mlp" in pipelines else None
    ast_pipelines = [pipeline for pipeline in pipelines if pipeline.startswith("ast_")]
    ast_store = idea019.load_feature_store() if ast_pipelines else None
    layer_store = (
        idea019.load_layer_store(ast_store)
        if "ast_probe_guided_adapter" in pipelines
        else None
    )

    completed: list[dict[str, Any]] = []
    for repeat in repeats:
        for outer_fold in folds:
            fold_roles = roles[
                (roles["repeat"] == repeat) & (roles["outer_fold"] == outer_fold)
            ]
            probe_seed_values = fold_roles["inner_seed"].unique()
            if len(probe_seed_values) != 1:
                raise RuntimeError("Each repeat-fold must have one frozen inner seed")
            probe_seed = int(probe_seed_values[0])
            indices = (
                ast_indices(
                    ast_store,
                    roles,
                    repeat,
                    outer_fold,
                    include_test=args.scope == "formal",
                )
                if ast_pipelines
                else None
            )
            probe_result = None
            if layer_store is not None:
                probe_result = load_or_compute_probe(
                    run_root,
                    layer_store,
                    indices["train"],
                    repeat,
                    outer_fold,
                    probe_seed,
                )
                print(
                    f"repeat={repeat} fold={outer_fold} probe layers="
                    f"{probe_result['selected_layers_one_based']}",
                    flush=True,
                )
            for base_seed in base_seeds:
                seed = full_model_seed(base_seed, repeat, outer_fold)
                for pipeline in pipelines:
                    output_dir = fit_directory(
                        run_root, pipeline, repeat, outer_fold, base_seed
                    )
                    summary_path = output_dir / "fit_summary.json"
                    if summary_path.exists():
                        if not args.resume:
                            raise FileExistsError(f"Fit already exists: {summary_path}")
                        fit_summary = read_json(summary_path)
                        completed.append(fit_summary)
                        print(f"resume: skipping completed {summary_path}", flush=True)
                        continue
                    output_dir.mkdir(parents=True, exist_ok=True)
                    print(
                        f"=== {pipeline} repeat={repeat} fold={outer_fold} "
                        f"base_seed={base_seed} full_seed={seed} ===",
                        flush=True,
                    )
                    if pipeline == "vggish_mlp":
                        audit, test_frame = fit_vggish(
                            args.scope,
                            recipe,
                            vggish_store,
                            roles,
                            repeat,
                            outer_fold,
                            seed,
                            args.smoke_max_epochs,
                        )
                    else:
                        audit, test_frame = fit_ast(
                            args.scope,
                            pipeline,
                            recipe,
                            locked_v1_config,
                            ast_store,
                            indices,
                            probe_result,
                            device,
                            seed,
                            args.smoke_max_epochs,
                        )
                    prediction_path = None
                    if test_frame is not None:
                        test_frame = add_prediction_identity(
                            test_frame,
                            pipeline,
                            repeat,
                            outer_fold,
                            base_seed,
                            seed,
                        )
                        prediction_path = output_dir / "outer_test_unit_predictions.csv"
                        test_frame.to_csv(prediction_path, index=False)
                    fit_summary = {
                        "status": "complete",
                        "scope": args.scope,
                        "pipeline": pipeline,
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "base_seed": base_seed,
                        "full_seed": seed,
                        "probe_seed": probe_seed if pipeline == "ast_probe_guided_adapter" else None,
                        "outer_test_predictions": (
                            relative_repo_path(prediction_path)
                            if prediction_path is not None
                            else None
                        ),
                        "audit": audit,
                    }
                    write_json(summary_path, fit_summary)
                    completed.append(fit_summary)

    aggregate = None
    if args.scope == "formal":
        aggregate = aggregate_formal_run(
            run_root, recipe, pipelines, repeats, folds, base_seeds
        )
    run_summary = {
        "status": "complete",
        "scope": args.scope,
        "completed_fits": len(completed),
        "outer_test_predictions_produced": args.scope == "formal",
        "formal_aggregate_written": aggregate is not None,
        "fits": completed,
    }
    write_json(run_root / "run_summary.json", run_summary)
    print(json.dumps(run_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
