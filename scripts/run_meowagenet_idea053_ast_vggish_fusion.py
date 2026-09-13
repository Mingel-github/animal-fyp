"""Run IDEA-053 nested animal-level AST-VGGish probability fusion."""

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
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import torch
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_formal_v2_1 as formal  # noqa: E402
import run_meowagenet_idea051_cat_set as evaluation  # noqa: E402
import run_meowagenet_idea052_ast_local_residual as ast_reference  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "meowagenet_idea053_ast_vggish_probability_fusion_v1.json"
)
PLAN_PATH = REPO_ROOT / "plan" / "IDEA-053_AST_VGGish_probability_fusion.md"
DIAGNOSTIC_PATH = (
    REPO_ROOT
    / "metadata"
    / "experiments"
    / "meowagenet_idea053_ast_vggish_complementarity_v1.json"
)
DIAGNOSTIC_SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "diagnose_meowagenet_idea053_ast_vggish_complementarity.py"
)
FORMAL_RECIPE_PATH = (
    REPO_ROOT
    / "configs"
    / "experiment"
    / "meowagenet_formal_v2_1_probe_guided_candidate_v1.json"
)
ROLES_PATH = REPO_ROOT / "splits" / "meowagenet_formal_v2_nested_roles.csv"
AST_EMBEDDING_PATH = (
    REPO_ROOT
    / "runs"
    / "ast_locked_v1"
    / "gpu_rerun_2026-08-26"
    / "ast_standard_call_embeddings.npz"
)
VGGISH_EMBEDDING_PATH = formal.VGGISH_PATH
RUN_ROOT = REPO_ROOT / "runs" / "meowagenet_idea053_ast_vggish_fusion_v1"
PIPELINES = (
    "A0_ast_reference",
    "V0_vggish_reference",
    "F1_inner_ce_probability_fusion",
)
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
LABEL_NAMES = ("kitten", "adult", "senior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "evaluate"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")


def git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol["protocol_id"] != "meowagenet-idea053-ast-vggish-probability-fusion-v1":
        raise ValueError("Unexpected IDEA-053 protocol ID")
    if tuple(protocol["initial_evaluation"]["reported_pipelines"]) != PIPELINES:
        raise ValueError("IDEA-053 pipeline order differs")
    paths = {
        "idea": (PLAN_PATH, protocol["idea"]["sha256"]),
        "diagnostic": (DIAGNOSTIC_PATH, protocol["diagnostic"]["sha256"]),
        "diagnostic script": (
            DIAGNOSTIC_SCRIPT_PATH,
            protocol["diagnostic"]["script_sha256"],
        ),
        "roles": (ROLES_PATH, protocol["splits"]["roles_sha256"]),
        "AST embedding": (
            AST_EMBEDDING_PATH,
            protocol["dependencies"]["ast_embedding_sha256"],
        ),
        "VGGish embedding": (
            VGGISH_EMBEDDING_PATH,
            protocol["dependencies"]["vggish_embedding_sha256"],
        ),
        "AST reference runner": (
            REPO_ROOT / protocol["dependencies"]["ast_reference_runner"],
            protocol["dependencies"]["ast_reference_runner_sha256"],
        ),
        "VGGish reference runner": (
            REPO_ROOT / protocol["dependencies"]["vggish_reference_runner"],
            protocol["dependencies"]["vggish_reference_runner_sha256"],
        ),
        "VGGish recipe": (
            FORMAL_RECIPE_PATH,
            protocol["dependencies"]["vggish_recipe_sha256"],
        ),
    }
    for label, (path, expected) in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"IDEA-053 {label} checksum mismatch: expected {expected}, got {actual}"
            )
    if protocol["initial_evaluation"]["total_model_fits"] != 24:
        raise ValueError("IDEA-053 initial evaluation must contain 24 model fits")


def ast_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    return {"fixed_training": protocol["ast_training"]}


def load_inputs() -> tuple[
    ast_reference.ResidualStore, dict[str, np.ndarray], pd.DataFrame, dict[str, Any]
]:
    ast_store = ast_reference.load_store()
    cat_ids = set(ast_store.cat_ids.tolist())
    vggish_store = formal.load_vggish(cat_ids)
    roles = pd.read_csv(ROLES_PATH, dtype={"cat_id": str})
    recipe = read_json(FORMAL_RECIPE_PATH)
    if len(ast_store.call_ids) != 792 or len(np.unique(ast_store.cat_ids)) != 111:
        raise RuntimeError("Unexpected AST dataset scope")
    if len(vggish_store["labels"]) != 936 or len(np.unique(vggish_store["cat_ids"])) != 111:
        raise RuntimeError("Unexpected VGGish dataset scope")
    return ast_store, vggish_store, roles, recipe


def animal_cross_entropy(animals: pd.DataFrame) -> float:
    probabilities = animals[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    labels = animals["true_label"].to_numpy(dtype=np.int64)
    return float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def align_animals(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = left.sort_values("cat_id").reset_index(drop=True)
    right = right.sort_values("cat_id").reset_index(drop=True)
    if not np.array_equal(left["cat_id"].to_numpy(), right["cat_id"].to_numpy()):
        raise RuntimeError("AST and VGGish animal IDs differ")
    if not np.array_equal(
        left["true_label"].to_numpy(dtype=np.int64),
        right["true_label"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("AST and VGGish animal labels differ")
    return left, right


def fused_animals(
    ast_animals: pd.DataFrame, vggish_animals: pd.DataFrame, alpha: float
) -> pd.DataFrame:
    ast_animals, vggish_animals = align_animals(ast_animals, vggish_animals)
    ast_probabilities = ast_animals[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    vggish_probabilities = vggish_animals[list(PROBABILITY_COLUMNS)].to_numpy(
        dtype=np.float64
    )
    probabilities = alpha * ast_probabilities + (1.0 - alpha) * vggish_probabilities
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    frame = ast_animals[["cat_id", "true_label"]].copy()
    for index, column in enumerate(PROBABILITY_COLUMNS):
        frame[column] = probabilities[:, index]
    frame["predicted_label"] = probabilities.argmax(axis=1)
    return frame


def select_alpha(
    ast_animals: pd.DataFrame,
    vggish_animals: pd.DataFrame,
    alpha_grid: list[float],
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    for alpha in alpha_grid:
        animals = fused_animals(ast_animals, vggish_animals, float(alpha))
        rows.append(
            {
                "alpha": float(alpha),
                "animal_cross_entropy": animal_cross_entropy(animals),
                "animal_metrics": evaluation.animal_metrics(animals),
            }
        )
    selected = min(rows, key=lambda row: (row["animal_cross_entropy"], -row["alpha"]))
    return float(selected["alpha"]), rows


def fit_ast_path(
    protocol: dict[str, Any],
    store: ast_reference.ResidualStore,
    roles: pd.DataFrame,
    repeat: int,
    outer_fold: int,
    seed: int,
    device: torch.device,
    max_epochs_override: int | None,
    include_outer: bool,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    indices = ast_reference.reference.historical.fold_indices(
        store, roles, repeat, outer_fold, include_test=include_outer
    )
    best_epoch, inner, _, validation_calls = ast_reference.fit_inner(
        "R0_global_probability_mean",
        ast_protocol(protocol),
        store,
        indices["train"],
        indices["validation"],
        device,
        seed,
        max_epochs_override=max_epochs_override,
    )
    validation_animals = ast_reference.calls_to_animals(validation_calls)
    audit: dict[str, Any] = {
        "best_epoch": int(best_epoch),
        "inner": inner,
        "inner_validation_animal_cross_entropy": animal_cross_entropy(
            validation_animals
        ),
    }
    if not include_outer:
        return audit, validation_animals, None, None
    outer_train = np.concatenate((indices["train"], indices["validation"]))
    test_animals, test_calls, outer = ast_reference.fit_outer_and_predict(
        "R0_global_probability_mean",
        ast_protocol(protocol),
        store,
        outer_train,
        indices["test"],
        best_epoch,
        device,
        seed,
    )
    audit["outer"] = outer
    return audit, validation_animals, test_animals, test_calls


def fit_vggish_path(
    protocol: dict[str, Any],
    recipe: dict[str, Any],
    store: dict[str, np.ndarray],
    roles: pd.DataFrame,
    repeat: int,
    outer_fold: int,
    seed: int,
    max_epochs_override: int | None,
    include_outer: bool,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    masks = formal.vggish_masks(store, roles, repeat, outer_fold, include_outer)
    config = protocol["vggish_training"]
    source_config = recipe["vggish_mlp"]
    for key in ("dropout", "learning_rate", "batch_size", "maximum_epochs"):
        if float(config[key]) != float(source_config[key]):
            raise RuntimeError(f"VGGish protocol differs from source recipe for {key}")
    maximum_epochs = int(config["maximum_epochs"])
    if max_epochs_override is not None:
        maximum_epochs = min(maximum_epochs, max_epochs_override)
    scaler = StandardScaler().fit(store["features"][masks["train"]])
    x_train = scaler.transform(store["features"][masks["train"]]).astype(np.float32)
    x_validation = scaler.transform(store["features"][masks["validation"]]).astype(
        np.float32
    )
    model = formal.build_vggish_model(recipe, seed)
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
        class_weight=formal.balanced_class_weights(store["labels"][masks["train"]]),
        callbacks=[callback],
        verbose=0,
        shuffle=True,
    )
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    validation_probabilities = model.predict(x_validation, verbose=0)
    validation_units = formal.vggish_unit_frame(
        store, masks["validation"], validation_probabilities
    )
    validation_metrics, validation_animals = formal.animal_prediction_frame(
        validation_units
    )
    audit: dict[str, Any] = {
        "best_epoch": best_epoch,
        "stopped_epoch": int(len(history.history["loss"])),
        "best_validation_embedding_cross_entropy": float(min(history.history["val_loss"])),
        "inner_validation_animal_cross_entropy": animal_cross_entropy(
            validation_animals
        ),
        "inner_validation_animal_metrics": validation_metrics,
        "inner_train_rows": int(masks["train"].sum()),
        "inner_validation_rows": int(masks["validation"].sum()),
        "train_seconds": float(time.perf_counter() - started),
        "history": [
            {
                "epoch": index + 1,
                "train_loss": float(history.history["loss"][index]),
                "validation_loss": float(history.history["val_loss"][index]),
            }
            for index in range(len(history.history["loss"]))
        ],
    }
    if not include_outer:
        tf.keras.backend.clear_session()
        return audit, validation_animals, None, None
    outer_train_mask = masks["train"] | masks["validation"]
    outer_scaler = StandardScaler().fit(store["features"][outer_train_mask])
    x_outer_train = outer_scaler.transform(store["features"][outer_train_mask]).astype(
        np.float32
    )
    x_test = outer_scaler.transform(store["features"][masks["test"]]).astype(np.float32)
    model = formal.build_vggish_model(recipe, seed)
    started = time.perf_counter()
    model.fit(
        x_outer_train,
        store["labels"][outer_train_mask],
        epochs=best_epoch,
        batch_size=int(config["batch_size"]),
        class_weight=formal.balanced_class_weights(store["labels"][outer_train_mask]),
        verbose=0,
        shuffle=True,
    )
    test_probabilities = model.predict(x_test, verbose=0)
    test_units = formal.vggish_unit_frame(store, masks["test"], test_probabilities)
    test_metrics, test_animals = formal.animal_prediction_frame(test_units)
    audit["outer"] = {
        "epochs": best_epoch,
        "outer_train_rows": int(outer_train_mask.sum()),
        "outer_test_rows": int(masks["test"].sum()),
        "train_and_predict_seconds": float(time.perf_counter() - started),
        "test_animal_metrics": test_metrics,
    }
    tf.keras.backend.clear_session()
    return audit, validation_animals, test_animals, test_units


def environment_lock(protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "tensorflow": tf.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "AST_device": str(device),
        "AST_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "VGGish_device": "CPU",
        "git_revision": git_revision(),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
    }


def smoke(
    protocol: dict[str, Any],
    ast_store: ast_reference.ResidualStore,
    vggish_store: dict[str, np.ndarray],
    roles: pd.DataFrame,
    recipe: dict[str, Any],
    device: torch.device,
    resume: bool,
) -> None:
    root = RUN_ROOT / "smoke"
    summary_path = root / "summary.json"
    if summary_path.is_file() and not resume:
        raise FileExistsError(summary_path)
    settings = protocol["smoke"]
    seed = ast_reference.reference.historical.full_seed(
        int(settings["base_seed"]),
        int(settings["repeat"]),
        int(settings["outer_fold"]),
    )
    ast_audit, ast_validation, _, _ = fit_ast_path(
        protocol,
        ast_store,
        roles,
        int(settings["repeat"]),
        int(settings["outer_fold"]),
        seed,
        device,
        int(settings["epochs"]),
        include_outer=False,
    )
    vggish_audit, vggish_validation, _, _ = fit_vggish_path(
        protocol,
        recipe,
        vggish_store,
        roles,
        int(settings["repeat"]),
        int(settings["outer_fold"]),
        seed,
        int(settings["epochs"]),
        include_outer=False,
    )
    alpha, curve = select_alpha(
        ast_validation,
        vggish_validation,
        [float(value) for value in protocol["pipelines"][PIPELINES[2]]["alpha_grid"]],
    )
    fused = fused_animals(ast_validation, vggish_validation, alpha)
    probability_error = float(
        np.abs(fused[list(PROBABILITY_COLUMNS)].sum(axis=1).to_numpy() - 1.0).max()
    )
    if probability_error > 1.0e-8:
        raise RuntimeError("IDEA-053 fused probabilities are not normalized")
    write_json(root / "fit_summary.json", {
        "status": "complete",
        "stage": "inner_only_smoke",
        "outer_test_accessed": False,
        "seed": seed,
        "AST": ast_audit,
        "VGGish": vggish_audit,
        "selected_alpha": alpha,
        "inner_alpha_curve": curve,
        "fusion_validation_animal_metrics": evaluation.animal_metrics(fused),
    })
    environment_path = RUN_ROOT / "environment_lock.json"
    write_json(environment_path, environment_lock(protocol, device))
    revision = git_revision()
    if revision is None:
        raise RuntimeError("A Git commit is required before IDEA-053 execution lock")
    lock = {
        "schema_version": "1.0",
        "status": "locked_for_idea053_initial_evaluation",
        "outer_test_accessed": False,
        "protocol_id": protocol["protocol_id"],
        "code_commit": revision,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "idea_sha256": sha256(PLAN_PATH),
        "diagnostic_sha256": sha256(DIAGNOSTIC_PATH),
        "roles_sha256": sha256(ROLES_PATH),
        "AST_embedding_sha256": sha256(AST_EMBEDDING_PATH),
        "VGGish_embedding_sha256": sha256(VGGISH_EMBEDDING_PATH),
        "environment_lock_sha256": sha256(environment_path),
        "initial_evaluation": protocol["initial_evaluation"],
    }
    lock_path = RUN_ROOT / "execution_lock.json"
    write_json(lock_path, lock)
    summary = {
        "status": "complete",
        "outer_test_accessed": False,
        "excluded_from_evaluation_summary": True,
        "AST_calls": 792,
        "VGGish_rows": 936,
        "cats": 111,
        "inner_validation_cats": int(len(fused)),
        "selected_alpha": alpha,
        "probability_sum_max_error": probability_error,
        "execution_lock_sha256": sha256(lock_path),
        "environment_lock_sha256": sha256(environment_path),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


def verify_execution_lock(protocol: dict[str, Any]) -> dict[str, Any]:
    path = RUN_ROOT / "execution_lock.json"
    if not path.is_file():
        raise FileNotFoundError("IDEA-053 evaluation requires a smoke execution lock")
    lock = read_json(path)
    expected = {
        "status": "locked_for_idea053_initial_evaluation",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "idea_sha256": sha256(PLAN_PATH),
        "diagnostic_sha256": sha256(DIAGNOSTIC_PATH),
        "roles_sha256": sha256(ROLES_PATH),
        "AST_embedding_sha256": sha256(AST_EMBEDDING_PATH),
        "VGGish_embedding_sha256": sha256(VGGISH_EMBEDDING_PATH),
    }
    differences = [key for key, value in expected.items() if lock.get(key) != value]
    if differences:
        raise RuntimeError(f"IDEA-053 execution lock mismatch: {differences}")
    if lock.get("outer_test_accessed") is not False:
        raise RuntimeError("IDEA-053 smoke lock crossed the outer-test boundary")
    return lock


def write_fold_outputs(
    root: Path,
    ast_animals: pd.DataFrame,
    ast_calls: pd.DataFrame,
    vggish_animals: pd.DataFrame,
    vggish_units: pd.DataFrame,
    fused: pd.DataFrame,
) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    frames = {
        "AST_animals": (ast_animals, root / "AST_outer_test_animals.csv"),
        "AST_calls": (ast_calls, root / "AST_outer_test_calls.csv"),
        "VGGish_animals": (vggish_animals, root / "VGGish_outer_test_animals.csv"),
        "VGGish_rows": (vggish_units, root / "VGGish_outer_test_rows.csv"),
        "fusion_animals": (fused, root / "fusion_outer_test_animals.csv"),
    }
    paths = {}
    for label, (frame, path) in frames.items():
        frame.to_csv(path, index=False)
        paths[label] = repo_relative(path)
    return paths


def paired_rows(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_metrics: dict[str, Any],
    right_metrics: dict[str, Any],
    repeat: int,
) -> dict[str, Any]:
    left, right = align_animals(left, right)
    true = left["true_label"].to_numpy(dtype=np.int64)
    left_prediction = left["predicted_label"].to_numpy(dtype=np.int64)
    right_prediction = right["predicted_label"].to_numpy(dtype=np.int64)
    return {
        "repeat": repeat,
        "left_macro_f1": left_metrics["macro_f1"],
        "right_macro_f1": right_metrics["macro_f1"],
        "macro_f1_difference": right_metrics["macro_f1"] - left_metrics["macro_f1"],
        "balanced_accuracy_difference": right_metrics["balanced_accuracy"] - left_metrics["balanced_accuracy"],
        "qwk_difference": right_metrics["quadratic_weighted_kappa"] - left_metrics["quadratic_weighted_kappa"],
        "plain_accuracy_difference": right_metrics["plain_accuracy"] - left_metrics["plain_accuracy"],
        "changed_animals": int((left_prediction != right_prediction).sum()),
        "gained_correct_animals": int(((right_prediction == true) & (left_prediction != true)).sum()),
        "lost_correct_animals": int(((right_prediction != true) & (left_prediction == true)).sum()),
    }


def aggregate_results(protocol: dict[str, Any]) -> dict[str, Any]:
    settings = protocol["initial_evaluation"]
    metrics_by_pipeline = {pipeline: [] for pipeline in PIPELINES}
    animals_by_key: dict[tuple[str, int], pd.DataFrame] = {}
    alpha_rows = []
    for repeat in settings["repeats"]:
        parts = {pipeline: [] for pipeline in PIPELINES}
        for fold in settings["outer_folds"]:
            root = (
                RUN_ROOT
                / "evaluation"
                / "fits"
                / f"base_seed_{settings['base_seeds'][0]}"
                / f"repeat_{repeat}"
                / f"fold_{fold}"
            )
            paths = {
                PIPELINES[0]: root / "AST_outer_test_animals.csv",
                PIPELINES[1]: root / "VGGish_outer_test_animals.csv",
                PIPELINES[2]: root / "fusion_outer_test_animals.csv",
            }
            for pipeline, path in paths.items():
                parts[pipeline].append(pd.read_csv(path, dtype={"cat_id": str}))
            fit = read_json(root / "fit_summary.json")
            alpha_rows.append(
                {
                    "repeat": int(repeat),
                    "outer_fold": int(fold),
                    "selected_alpha": float(fit["selected_alpha"]),
                    "AST_inner_animal_cross_entropy": float(
                        fit["AST"]["inner_validation_animal_cross_entropy"]
                    ),
                    "VGGish_inner_animal_cross_entropy": float(
                        fit["VGGish"]["inner_validation_animal_cross_entropy"]
                    ),
                }
            )
        for pipeline in PIPELINES:
            animals = pd.concat(parts[pipeline], ignore_index=True).sort_values("cat_id")
            if len(animals) != 111 or animals["cat_id"].nunique() != 111:
                raise RuntimeError("IDEA-053 complete OOF must contain 111 cats")
            values = evaluation.animal_metrics(animals)
            metrics_by_pipeline[pipeline].append(
                {"base_seed": 17, "repeat": int(repeat), **values}
            )
            animals_by_key[(pipeline, int(repeat))] = animals
            output = RUN_ROOT / "evaluation" / "oof" / pipeline / f"repeat_{repeat}_animals.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            animals.to_csv(output, index=False)
    contrasts = ((PIPELINES[0], PIPELINES[1]), (PIPELINES[0], PIPELINES[2]))
    paired = {}
    paired_summary = {}
    bootstraps = {}
    changes = []
    for left_name, right_name in contrasts:
        name = f"{right_name}_minus_{left_name}"
        rows = []
        for repeat in settings["repeats"]:
            left = animals_by_key[(left_name, int(repeat))]
            right = animals_by_key[(right_name, int(repeat))]
            row = paired_rows(
                left,
                right,
                metrics_by_pipeline[left_name][int(repeat)],
                metrics_by_pipeline[right_name][int(repeat)],
                int(repeat),
            )
            rows.append(row)
            left_aligned, right_aligned = align_animals(left, right)
            changed = left_aligned["predicted_label"].to_numpy() != right_aligned["predicted_label"].to_numpy()
            frame = pd.DataFrame({
                "contrast": name,
                "repeat": int(repeat),
                "cat_id": left_aligned.loc[changed, "cat_id"],
                "true_label": left_aligned.loc[changed, "true_label"],
                "left_prediction": left_aligned.loc[changed, "predicted_label"],
                "right_prediction": right_aligned.loc[changed, "predicted_label"],
            })
            changes.append(frame)
        paired[name] = rows
        paired_summary[name] = {
            "macro_f1_differences": [row["macro_f1_difference"] for row in rows],
            "mean_macro_f1_difference": float(np.mean([row["macro_f1_difference"] for row in rows])),
            "positive_repeats": int(sum(row["macro_f1_difference"] > 0 for row in rows)),
            "mean_balanced_accuracy_difference": float(np.mean([row["balanced_accuracy_difference"] for row in rows])),
            "mean_qwk_difference": float(np.mean([row["qwk_difference"] for row in rows])),
            "mean_plain_accuracy_difference": float(np.mean([row["plain_accuracy_difference"] for row in rows])),
        }
        bootstraps[name] = evaluation.paired_cat_bootstrap(
            animals_by_key, left_name, right_name, protocol
        )
    change_path = RUN_ROOT / "evaluation" / "paired_prediction_changes.csv"
    pd.concat(changes, ignore_index=True).to_csv(change_path, index=False)
    aggregate = {
        pipeline: evaluation.aggregate_metrics(metrics_by_pipeline[pipeline])
        for pipeline in PIPELINES
    }
    fusion_name = f"{PIPELINES[2]}_minus_{PIPELINES[0]}"
    comparison = paired_summary[fusion_name]
    fusion_aggregate = aggregate[PIPELINES[2]]
    ast_aggregate = aggregate[PIPELINES[0]]
    supporting = []
    if comparison["mean_balanced_accuracy_difference"] >= 0:
        supporting.append("balanced_accuracy")
    if comparison["mean_qwk_difference"] >= 0:
        supporting.append("qwk")
    if fusion_aggregate["macro_f1_range"][0] >= ast_aggregate["macro_f1_range"][0]:
        supporting.append("worst_repeat_macro_f1")
    for label in LABEL_NAMES:
        if fusion_aggregate["mean_per_class"][label]["recall"] > ast_aggregate["mean_per_class"][label]["recall"]:
            supporting.append(f"{label}_recall")
    gate = protocol["seed_expansion_gate"]
    passed = (
        comparison["mean_macro_f1_difference"] >= float(gate["minimum_mean_macro_f1_gain"])
        and comparison["positive_repeats"] >= int(gate["minimum_positive_repeats"])
        and bool(supporting)
    )
    alpha_values = [row["selected_alpha"] for row in alpha_rows]
    return {
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "completed_model_fits": int(settings["total_model_fits"]),
        "complete_oof": metrics_by_pipeline,
        "aggregate": aggregate,
        "paired": paired,
        "paired_summary": paired_summary,
        "paired_cat_bootstrap": bootstraps,
        "alpha_selection": {
            "folds": alpha_rows,
            "values": alpha_values,
            "mean": float(np.mean(alpha_values)),
            "median": float(np.median(alpha_values)),
            "AST_only_folds": int(sum(value == 1.0 for value in alpha_values)),
            "VGGish_only_folds": int(sum(value == 0.0 for value in alpha_values)),
            "interior_fusion_folds": int(sum(0.0 < value < 1.0 for value in alpha_values)),
        },
        "seed_expansion_gate": {
            "passed": bool(passed),
            "strong_gain": bool(comparison["mean_macro_f1_difference"] >= float(gate["strong_gain"])),
            "supporting_signals": supporting,
        },
        "artifacts": {"paired_prediction_changes": repo_relative(change_path)},
    }


def evaluate(
    protocol: dict[str, Any],
    ast_store: ast_reference.ResidualStore,
    vggish_store: dict[str, np.ndarray],
    roles: pd.DataFrame,
    recipe: dict[str, Any],
    device: torch.device,
    resume: bool,
) -> None:
    lock = verify_execution_lock(protocol)
    root = RUN_ROOT / "evaluation"
    summary_path = root / "summary.json"
    if summary_path.is_file() and not resume:
        raise FileExistsError(summary_path)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "run_manifest.json", {
        "status": "running",
        "stage": "idea053_initial_evaluation",
        "outer_test_accessed": True,
        "code_commit": lock["code_commit"],
        "protocol_sha256": lock["protocol_sha256"],
        "runner_sha256": lock["runner_sha256"],
        "execution_lock_sha256": sha256(RUN_ROOT / "execution_lock.json"),
        "device": str(device),
    })
    completed = 0
    settings = protocol["initial_evaluation"]
    for base_seed in settings["base_seeds"]:
        for repeat in settings["repeats"]:
            for fold in settings["outer_folds"]:
                output = (
                    root
                    / "fits"
                    / f"base_seed_{base_seed}"
                    / f"repeat_{repeat}"
                    / f"fold_{fold}"
                )
                fit_path = output / "fit_summary.json"
                if fit_path.is_file():
                    if not resume:
                        raise FileExistsError(fit_path)
                    completed += 2
                    continue
                seed = ast_reference.reference.historical.full_seed(base_seed, repeat, fold)
                print(
                    f"IDEA053 repeat={repeat} fold={fold} seed={seed} AST",
                    flush=True,
                )
                ast_audit, ast_validation, ast_test, ast_calls = fit_ast_path(
                    protocol, ast_store, roles, repeat, fold, seed, device, None, True
                )
                print(
                    f"IDEA053 repeat={repeat} fold={fold} seed={seed} VGGish",
                    flush=True,
                )
                vggish_audit, vggish_validation, vggish_test, vggish_units = fit_vggish_path(
                    protocol, recipe, vggish_store, roles, repeat, fold, seed, None, True
                )
                alpha, curve = select_alpha(
                    ast_validation,
                    vggish_validation,
                    [float(value) for value in protocol["pipelines"][PIPELINES[2]]["alpha_grid"]],
                )
                if ast_test is None or ast_calls is None or vggish_test is None or vggish_units is None:
                    raise RuntimeError("IDEA-053 outer outputs are unexpectedly missing")
                fusion = fused_animals(ast_test, vggish_test, alpha)
                paths = write_fold_outputs(
                    output, ast_test, ast_calls, vggish_test, vggish_units, fusion
                )
                fit = {
                    "status": "complete",
                    "stage": "idea053_initial_evaluation",
                    "outer_test_accessed": True,
                    "base_seed": int(base_seed),
                    "repeat": int(repeat),
                    "outer_fold": int(fold),
                    "full_seed": int(seed),
                    "trained_models": 2,
                    "selected_alpha": alpha,
                    "inner_alpha_curve": curve,
                    "AST": ast_audit,
                    "VGGish": vggish_audit,
                    "outer_metrics": {
                        PIPELINES[0]: evaluation.animal_metrics(ast_test),
                        PIPELINES[1]: evaluation.animal_metrics(vggish_test),
                        PIPELINES[2]: evaluation.animal_metrics(fusion),
                    },
                    "prediction_paths": paths,
                }
                write_json(fit_path, fit)
                completed += 2
                print(
                    f"IDEA053 repeat={repeat} fold={fold} alpha={alpha:.1f} "
                    f"AST_F1={fit['outer_metrics'][PIPELINES[0]]['macro_f1']:.4f} "
                    f"fusion_F1={fit['outer_metrics'][PIPELINES[2]]['macro_f1']:.4f}",
                    flush=True,
                )
    summary = aggregate_results(protocol)
    inventory = evaluation.raw_prediction_inventory(root)
    inventory_path = root / "raw_prediction_inventory.json"
    write_json(inventory_path, inventory)
    summary["raw_prediction_inventory"] = {
        "path": repo_relative(inventory_path),
        "sha256": sha256(inventory_path),
        "files": inventory["files"],
        "bytes": inventory["bytes"],
        "aggregate_sha256": inventory["aggregate_sha256"],
    }
    summary["code_commit"] = lock["code_commit"]
    summary["execution_lock_sha256"] = sha256(RUN_ROOT / "execution_lock.json")
    summary["environment_lock_sha256"] = sha256(RUN_ROOT / "environment_lock.json")
    summary["runner_sha256"] = lock["runner_sha256"]
    summary["protocol_sha256"] = lock["protocol_sha256"]
    write_json(summary_path, summary)
    run_summary = {
        "status": "complete",
        "completed_model_fits": completed,
        "expected_model_fits": int(settings["total_model_fits"]),
        "summary_path": repo_relative(summary_path),
        "summary_sha256": sha256(summary_path),
        "raw_prediction_inventory_sha256": sha256(inventory_path),
        "raw_prediction_aggregate_sha256": inventory["aggregate_sha256"],
    }
    write_json(root / "run_summary.json", run_summary)
    write_json(root / "run_manifest.json", {
        "status": "complete",
        "stage": "idea053_initial_evaluation",
        "outer_test_accessed": True,
        "code_commit": lock["code_commit"],
        "protocol_sha256": lock["protocol_sha256"],
        "runner_sha256": lock["runner_sha256"],
        "execution_lock_sha256": sha256(RUN_ROOT / "execution_lock.json"),
        "device": str(device),
        "completed_model_fits": completed,
        "summary_path": repo_relative(summary_path),
    })
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as error:
        raise RuntimeError("TensorFlow device configuration occurred too late") from error
    tf.config.threading.set_intra_op_parallelism_threads(min(6, os.cpu_count() or 1))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    device = resolve_device(args.device)
    ast_store, vggish_store, roles, recipe = load_inputs()
    print(
        f"IDEA-053 stage={args.stage}; AST_device={device}; VGGish_device=CPU",
        flush=True,
    )
    if args.stage == "smoke":
        smoke(protocol, ast_store, vggish_store, roles, recipe, device, args.resume)
    else:
        evaluate(protocol, ast_store, vggish_store, roles, recipe, device, args.resume)


if __name__ == "__main__":
    main()
