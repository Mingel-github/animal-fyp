"""Run IDEA-087 fair-budget nested HPO for A0, U1, and C1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_idea068_age_sensitive_ast as idea068  # noqa: E402
import run_meowagenet_idea071_bounded_dual_path_fusion as idea071  # noqa: E402
import run_meowagenet_idea076_C1_final_seed_confirmation as idea076  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT / "configs" / "protocol" / "meowagenet_idea087_nested_hpo_v1.json"
)
PIPELINES = (
    "A0_ast_only",
    "U1_wide_unbounded_additive",
    "C1_bounded_wide_additive",
)
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
LABEL_BY_AGE = {"kitten": 0, "adult": 1, "senior": 2}
EXPECTED_PARAMETERS = {
    "A0_ast_only": 99_075,
    "U1_wide_unbounded_additive": 108_143,
    "C1_bounded_wide_additive": 108_143,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("preflight", "inner", "selection", "outer", "aggregate"),
        required=True,
    )
    parser.add_argument(
        "--output-subdir", default="meowagenet_idea087_nested_hpo_v1"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--director-authorized", action="store_true")
    parser.add_argument("--max-fits", type=int)
    parser.add_argument("--selection-lock-sha256")
    return parser.parse_args()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_run_root(output_subdir: str) -> Path:
    run_root = (idea068.idea051.RUNS_ROOT / output_subdir).resolve()
    if idea068.idea051.RUNS_ROOT.resolve() not in run_root.parents:
        raise ValueError("--output-subdir must stay below runs")
    return run_root


def configuration_map(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["config_id"]): row for row in protocol["search"]["configurations"]
    }


def derived_uint31(material: str) -> int:
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 2_147_483_647


def full_search_seed(base_seed: int, outer_fold: int, inner_fold: int) -> int:
    return int(base_seed) + 10_000 * int(outer_fold) + 100 * int(inner_fold)


def full_refit_seed(base_seed: int, outer_fold: int) -> int:
    return int(base_seed) + 10_000 * int(outer_fold)


def half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def refit_epoch(best_epochs: Iterable[int], maximum_epochs: int = 50) -> int:
    values = np.asarray(list(best_epochs), dtype=np.int64)
    if values.shape != (6,) or np.any(values < 1) or np.any(values > maximum_epochs):
        raise RuntimeError("IDEA-087 refit epoch requires six 1-based valid epochs")
    return int(np.clip(half_up(float(np.median(values))), 1, maximum_epochs))


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea087-nested-hpo-v1":
        raise RuntimeError("Unexpected IDEA-087 protocol")
    if protocol.get("status") != "locked_before_cpu_preflight":
        raise RuntimeError("IDEA-087 protocol is not locked for CPU preflight")
    if tuple(protocol["models"]["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-087 pipeline matrix changed")
    if protocol["models"]["trainable_parameters"] != EXPECTED_PARAMETERS:
        raise RuntimeError("IDEA-087 parameter counts changed")
    configurations = protocol["search"]["configurations"]
    expected = []
    index = 0
    for learning_rate in protocol["search"]["learning_rates"]:
        for dropout in protocol["search"]["dropouts"]:
            for weight_decay in protocol["search"]["weight_decays"]:
                expected.append(
                    {
                        "config_id": f"q{index:02d}",
                        "learning_rate": float(learning_rate),
                        "dropout": float(dropout),
                        "weight_decay": float(weight_decay),
                    }
                )
                index += 1
    if configurations != expected or expected[0]["config_id"] != "q00":
        raise RuntimeError("IDEA-087 configuration order changed")
    expected_inner = len(configurations) * len(PIPELINES) * 4 * 3 * 2
    if expected_inner != 576 or protocol["budget"]["inner_fits"] != expected_inner:
        raise RuntimeError("IDEA-087 inner fit budget changed")
    if protocol["budget"]["outer_refit_fits_maximum"] != 72:
        raise RuntimeError("IDEA-087 outer fit budget changed")
    if protocol["budget"]["total_fits_maximum"] != 648:
        raise RuntimeError("IDEA-087 total fit budget changed")
    split_materials = protocol["inner_roles"]["split_seed_materials"]
    if [derived_uint31(value) for value in split_materials] != protocol["inner_roles"][
        "split_seeds"
    ]:
        raise RuntimeError("IDEA-087 split seeds do not match their materials")
    for name in ("search", "refit"):
        materials = protocol[name][f"{name}_seed_materials"]
        if [derived_uint31(value) for value in materials] != protocol[name][
            f"{name}_base_seeds"
        ]:
            raise RuntimeError(f"IDEA-087 {name} seeds do not match their materials")
    dependencies = protocol["dependencies"]
    checks = {
        REPO_ROOT / dependencies["plan_path"]: dependencies["plan_sha256"],
        REPO_ROOT / dependencies["independent_design_review_path"]: dependencies[
            "independent_design_review_sha256"
        ],
        REPO_ROOT / protocol["data"]["roles_path"]: protocol["data"]["roles_sha256"],
        REPO_ROOT / protocol["data"]["dataset_manifest_path"]: protocol["data"][
            "dataset_manifest_sha256"
        ],
        REPO_ROOT / protocol["data"]["frozen_embedding_path"]: protocol["data"][
            "frozen_embedding_sha256"
        ],
        REPO_ROOT / protocol["data"]["feature_path"]: protocol["data"][
            "feature_sha256"
        ],
        REPO_ROOT / protocol["data"]["fbank_path"]: protocol["data"][
            "fbank_sha256"
        ],
        REPO_ROOT / dependencies["idea068_runner_path"]: dependencies[
            "idea068_runner_sha256"
        ],
        REPO_ROOT / dependencies["idea071_runner_path"]: dependencies[
            "idea071_runner_sha256"
        ],
        REPO_ROOT / dependencies["idea076_runner_path"]: dependencies[
            "idea076_runner_sha256"
        ],
        REPO_ROOT / dependencies["idea076_protocol_path"]: dependencies[
            "idea076_protocol_sha256"
        ],
        REPO_ROOT / dependencies["idea069_runner_path"]: dependencies[
            "idea069_runner_sha256"
        ],
        REPO_ROOT / dependencies["idea051_runner_path"]: dependencies[
            "idea051_runner_sha256"
        ],
        REPO_ROOT / dependencies["global_weighting_runner_path"]: dependencies[
            "global_weighting_runner_sha256"
        ],
        REPO_ROOT / dependencies["historical_runner_path"]: dependencies[
            "historical_runner_sha256"
        ],
        REPO_ROOT / dependencies["idea019_runner_path"]: dependencies[
            "idea019_runner_sha256"
        ],
        REPO_ROOT / dependencies["ast_finetuning_runner_path"]: dependencies[
            "ast_finetuning_runner_sha256"
        ],
        REPO_ROOT / dependencies["evaluation_module_path"]: dependencies[
            "evaluation_module_sha256"
        ],
        REPO_ROOT / dependencies["locked_base_protocol_path"]: dependencies[
            "locked_base_protocol_sha256"
        ],
        Path(__file__).resolve(): dependencies["runner_sha256"],
        REPO_ROOT / dependencies["tests_path"]: dependencies["tests_sha256"],
    }
    for path, expected_hash in checks.items():
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"IDEA-087 dependency mismatch: {path}")


def load_inputs(
    protocol: dict[str, Any],
) -> tuple[Any, np.ndarray, pd.DataFrame]:
    store = idea068.idea051.reference.historical.idea019.load_feature_store()
    if len(store.call_ids) != 792 or len(np.unique(store.cat_ids.astype(str))) != 111:
        raise RuntimeError("IDEA-087 expected 792 calls from 111 cats")
    feature_path = REPO_ROOT / protocol["data"]["feature_path"]
    loaded = np.load(feature_path)
    if tuple(loaded["feature_names"].astype(str)) != idea068.FEATURE_NAMES:
        raise RuntimeError("IDEA-087 acoustic feature definition changed")
    if not np.array_equal(loaded["call_ids"].astype(str), store.call_ids.astype(str)):
        raise RuntimeError("IDEA-087 acoustic feature call order changed")
    features = loaded["features"].astype(np.float32)
    if features.shape != (792, 20):
        raise RuntimeError("IDEA-087 acoustic feature matrix changed")
    roles = pd.read_csv(
        REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str}
    )
    roles = roles[roles["repeat"] == int(protocol["data"]["roles_repeat"])].copy()
    if len(roles) != 444:
        raise RuntimeError("IDEA-087 repeat-0 role table changed")
    return store, features, roles


def build_inner_roles(protocol: dict[str, Any], outer_roles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outer_fold, split_seed in zip(
        protocol["data"]["outer_folds"], protocol["inner_roles"]["split_seeds"]
    ):
        cell = outer_roles[outer_roles["outer_fold"] == int(outer_fold)].copy()
        if cell["cat_id"].duplicated().any() or len(cell) != 111:
            raise RuntimeError("IDEA-087 outer role cell is not one row per cat")
        outer_train = cell[cell["role"].isin(("train", "validation"))].copy()
        outer_test = cell[cell["role"] == "test"].copy()
        outer_train = outer_train.sort_values("cat_id").reset_index(drop=True)
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(split_seed))
        labels = outer_train["age_group"].astype(str).to_numpy()
        seen_validation: list[str] = []
        for inner_fold, (train_pos, validation_pos) in enumerate(
            splitter.split(np.zeros(len(outer_train)), labels)
        ):
            train_cats = set(outer_train.iloc[train_pos]["cat_id"].astype(str))
            validation_cats = set(
                outer_train.iloc[validation_pos]["cat_id"].astype(str)
            )
            if train_cats & validation_cats:
                raise RuntimeError("IDEA-087 inner cats overlap")
            if train_cats | validation_cats != set(outer_train["cat_id"].astype(str)):
                raise RuntimeError("IDEA-087 inner cats do not cover outer train")
            for role_name, cats in (("train", train_cats), ("validation", validation_cats)):
                selected = outer_train[outer_train["cat_id"].astype(str).isin(cats)]
                if set(selected["age_group"].astype(str)) != set(LABEL_BY_AGE):
                    raise RuntimeError("IDEA-087 inner role is missing a class")
                for row in selected.itertuples(index=False):
                    rows.append(
                        {
                            "outer_fold": int(outer_fold),
                            "inner_fold": int(inner_fold),
                            "split_seed": int(split_seed),
                            "cat_id": str(row.cat_id),
                            "age_group": str(row.age_group),
                            "call_count": int(row.call_count),
                            "role": role_name,
                        }
                    )
            seen_validation.extend(sorted(validation_cats))
        if len(seen_validation) != len(set(seen_validation)) or set(
            seen_validation
        ) != set(outer_train["cat_id"].astype(str)):
            raise RuntimeError("IDEA-087 inner validation folds do not form complete OOF")
        if set(outer_train["cat_id"].astype(str)) & set(
            outer_test["cat_id"].astype(str)
        ):
            raise RuntimeError("IDEA-087 outer train/test cats overlap")
    frame = pd.DataFrame(rows).sort_values(
        ["outer_fold", "inner_fold", "role", "cat_id"]
    ).reset_index(drop=True)
    if len(frame) != 3 * sum(
        len(
            outer_roles[
                (outer_roles["outer_fold"] == fold)
                & outer_roles["role"].isin(("train", "validation"))
            ]
        )
        for fold in protocol["data"]["outer_folds"]
    ):
        raise RuntimeError("IDEA-087 inner role row count changed")
    return frame


def indices_for_cats(store: Any, cats: Iterable[str]) -> np.ndarray:
    cat_set = set(str(value) for value in cats)
    indices = np.flatnonzero(np.isin(store.cat_ids.astype(str), sorted(cat_set))).astype(
        np.int64
    )
    if set(store.cat_ids[indices].astype(str)) != cat_set:
        raise RuntimeError("IDEA-087 failed to resolve every cat to calls")
    return indices


def inner_indices(
    store: Any, inner_roles: pd.DataFrame, outer_fold: int, inner_fold: int
) -> dict[str, np.ndarray]:
    cell = inner_roles[
        (inner_roles["outer_fold"] == int(outer_fold))
        & (inner_roles["inner_fold"] == int(inner_fold))
    ]
    result = {
        role: indices_for_cats(store, cell[cell["role"] == role]["cat_id"].astype(str))
        for role in ("train", "validation")
    }
    if np.intersect1d(result["train"], result["validation"]).size:
        raise RuntimeError("IDEA-087 inner call leakage")
    return result


def outer_indices(
    store: Any, outer_roles: pd.DataFrame, outer_fold: int
) -> dict[str, np.ndarray]:
    cell = outer_roles[outer_roles["outer_fold"] == int(outer_fold)]
    train_cats = cell[cell["role"].isin(("train", "validation"))]["cat_id"].astype(str)
    test_cats = cell[cell["role"] == "test"]["cat_id"].astype(str)
    result = {
        "train": indices_for_cats(store, train_cats),
        "test": indices_for_cats(store, test_cats),
    }
    if np.intersect1d(result["train"], result["test"]).size:
        raise RuntimeError("IDEA-087 outer call leakage")
    return result


def build_model(
    pipeline: str,
    config: dict[str, Any],
    store: Any,
    features: np.ndarray,
    train_indices: np.ndarray,
) -> torch.nn.Module:
    embeddings = store.frozen_embeddings[train_indices]
    common = {
        "ast_mean": embeddings.mean(axis=0),
        "ast_scale": embeddings.std(axis=0),
        "age_train": features[train_indices],
        "dropout": float(config["dropout"]),
    }
    if pipeline == PIPELINES[0]:
        return idea068.AgeResidualClassifier(
            pipeline="A0_ast_only", age_hidden_units=32, **common
        )
    if pipeline == PIPELINES[1]:
        return idea076.WideUnboundedAdditiveClassifier(**common)
    if pipeline == PIPELINES[2]:
        return idea071.BoundedWideAdditiveClassifier(**common)
    raise ValueError(pipeline)


def state_digest(model: torch.nn.Module, common_only: bool = False) -> str:
    digest = hashlib.sha256()
    common_names = ("ast_mean", "ast_scale", "ast_linear.", "batch_norm.", "output.")
    for name, tensor in sorted(model.state_dict().items()):
        if common_only and not any(name == prefix or name.startswith(prefix) for prefix in common_names):
            continue
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def training_role_identity(store: Any, indices: np.ndarray) -> dict[str, Any]:
    cats = sorted(set(store.cat_ids[indices].astype(str)))
    labels = store.labels[indices].astype(np.int64)
    weights = idea068.class_weights(labels)
    return {
        "cats": len(cats),
        "calls": int(len(indices)),
        "cat_ids_sha256": text_sha256("\n".join(cats)),
        "call_indices_sha256": hashlib.sha256(
            np.sort(indices.astype("<i8")).tobytes()
        ).hexdigest(),
        "call_class_counts": np.bincount(labels, minlength=3).astype(int).tolist(),
        "call_class_weights": weights.astype(float).tolist(),
    }


def historical_full_seeds() -> set[int]:
    """Build a conservative registry from locked IDEA-068 through IDEA-086 protocols."""

    bases: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"base_seeds", "search_base_seeds", "refit_base_seeds"} and isinstance(child, list):
                    bases.update(int(item) for item in child if isinstance(item, int))
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    protocol_root = REPO_ROOT / "configs" / "protocol"
    for idea_number in range(68, 87):
        for path in sorted(protocol_root.glob(f"meowagenet_idea{idea_number:03d}*.json")):
            visit(read_json(path))
    result = set(bases)
    for base in bases:
        for repeat in range(3):
            for fold in range(4):
                result.add(base + 10_000 * repeat + 100 * fold)
    return result


def seed_audit(protocol: dict[str, Any]) -> dict[str, Any]:
    search = {
        full_search_seed(base, outer_fold, inner_fold)
        for base in protocol["search"]["search_base_seeds"]
        for outer_fold in protocol["data"]["outer_folds"]
        for inner_fold in range(3)
    }
    refit = {
        full_refit_seed(base, outer_fold)
        for base in protocol["refit"]["refit_base_seeds"]
        for outer_fold in protocol["data"]["outer_folds"]
    }
    if len(search) != 24 or len(refit) != 12 or search & refit:
        raise RuntimeError("IDEA-087 new full seeds collide")
    historical = historical_full_seeds()
    overlap = sorted((search | refit) & historical)
    if overlap:
        raise RuntimeError(f"IDEA-087 new seeds collide with IDEA-068..086: {overlap}")
    return {
        "search_full_seeds": sorted(search),
        "search_unique": len(search),
        "refit_full_seeds": sorted(refit),
        "refit_unique": len(refit),
        "search_refit_overlap": [],
        "historical_seed_registry_size": len(historical),
        "historical_overlap": [],
    }


def predict(
    model: torch.nn.Module,
    store: Any,
    features: np.ndarray,
    indices: np.ndarray,
    cat_batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return idea076.predict_with_perturbation_reset(
        model, store, features, indices, cat_batch_size, device, seed
    )


def metric_bundle(animals: pd.DataFrame) -> dict[str, Any]:
    ordered = animals.sort_values("cat_id").reset_index(drop=True)
    labels = ordered["true_label"].to_numpy(dtype=np.int64)
    probabilities = ordered[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    recalls = recall_score(
        labels, predictions, labels=[0, 1, 2], average=None, zero_division=0
    )
    clipped = np.clip(probabilities, 1.0e-12, 1.0)
    targets = np.eye(3, dtype=np.float64)[labels]
    return {
        "n": int(len(ordered)),
        "plain_accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(np.mean(recalls)),
        "class_recall": {
            "kitten": float(recalls[0]),
            "adult": float(recalls[1]),
            "senior": float(recalls[2]),
        },
        "cross_entropy": float(-np.mean(np.log(clipped[np.arange(len(labels)), labels]))),
        "brier": float(np.mean(np.sum((probabilities - targets) ** 2, axis=1))),
    }


def validate_probabilities(calls: pd.DataFrame, animals: pd.DataFrame) -> None:
    for frame, identity in ((calls, "calls"), (animals, "animals")):
        probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"IDEA-087 nonfinite {identity} probabilities")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise RuntimeError(f"IDEA-087 out-of-range {identity} probabilities")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0):
            raise RuntimeError(f"IDEA-087 unnormalized {identity} probabilities")
    reconstructed = idea068.idea051.calls_to_animals(calls)
    left = animals.sort_values("cat_id").reset_index(drop=True)
    right = reconstructed.sort_values("cat_id").reset_index(drop=True)
    for column in ("cat_id", "true_label", "call_count", "predicted_label"):
        if not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
            raise RuntimeError(f"IDEA-087 call-to-cat mismatch in {column}")
    if not np.allclose(
        left[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64),
        right[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64),
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise RuntimeError("IDEA-087 call-to-cat probability mismatch")


def train_fit(
    pipeline: str,
    config: dict[str, Any],
    protocol: dict[str, Any],
    store: Any,
    features: np.ndarray,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    device: torch.device,
    seed: int,
    fixed_epochs: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    idea068.idea051.reference.historical.set_seed(seed)
    model = build_model(pipeline, config, store, features, train_indices).to(device)
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    if parameters != EXPECTED_PARAMETERS[pipeline]:
        raise RuntimeError("IDEA-087 model parameter count changed")
    initial_common_digest = state_digest(model, common_only=True)
    initial_full_digest = state_digest(model)
    fixed = protocol["fixed_training"]
    training_seed = seed + int(fixed["post_build_seed_offset"])
    idea068.idea051.reference.historical.set_seed(training_seed)
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=float(config["learning_rate"]),
        eps=float(fixed["optimizer_epsilon"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    dataset = idea068.idea051.CatSetDataset(store, train_indices)
    loader = idea068.idea051.build_set_loader(
        dataset, int(fixed["cat_batch_size"]), True, seed
    )
    call_weights = torch.from_numpy(
        idea068.class_weights(store.labels[train_indices])
    ).to(device)
    maximum_epochs = (
        int(fixed_epochs) if fixed_epochs is not None else int(fixed["maximum_epochs"])
    )
    if maximum_epochs < 1 or maximum_epochs > int(fixed["maximum_epochs"]):
        raise RuntimeError("IDEA-087 fixed epoch is out of range")
    best_loss = float("inf")
    best_epoch = 1
    best_state = idea068.idea051.cpu_state_dict(model)
    best_animals: pd.DataFrame | None = None
    best_calls: pd.DataFrame | None = None
    history: list[dict[str, Any]] = []
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        train_loss, train_audit = idea068.train_one_epoch(
            model,
            loader,
            optimizer,
            scaler,
            store,
            features,
            call_weights,
            device,
            float(fixed["gradient_clip"]),
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_call_loss": train_loss,
            "train_audit": train_audit,
        }
        if fixed_epochs is None:
            animals, calls = predict(
                model,
                store,
                features,
                evaluation_indices,
                int(fixed["cat_batch_size"]) * 2,
                device,
                seed,
            )
            bundle = metric_bundle(animals)
            row.update(
                {
                    "validation_animal_cross_entropy": bundle["cross_entropy"],
                    "validation_animal_brier": bundle["brier"],
                    "validation_animal_metrics": bundle,
                }
            )
            if bundle["cross_entropy"] < best_loss - 1.0e-6:
                best_loss = bundle["cross_entropy"]
                best_epoch = epoch
                best_state = idea068.idea051.cpu_state_dict(model)
                best_animals = animals.copy()
                best_calls = calls.copy()
                stale = 0
            else:
                stale += 1
        history.append(row)
        print(
            f"{pipeline} {config['config_id']} epoch={epoch} call_CE={train_loss:.4f}",
            flush=True,
        )
        if fixed_epochs is None and stale >= int(fixed["early_stopping_patience"]):
            break
    if fixed_epochs is None:
        if best_animals is None or best_calls is None:
            raise RuntimeError("IDEA-087 inner fit selected no checkpoint")
        model.load_state_dict(best_state)
        final_animals, final_calls = predict(
            model,
            store,
            features,
            evaluation_indices,
            int(fixed["cat_batch_size"]) * 2,
            device,
            seed,
        )
        reload_difference = float(
            np.abs(
                final_animals[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
                - best_animals[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
            ).max()
        )
        if reload_difference > 1.0e-7:
            raise RuntimeError("IDEA-087 best-state reload mismatch")
    else:
        best_epoch = int(fixed_epochs)
        final_animals, final_calls = predict(
            model,
            store,
            features,
            evaluation_indices,
            int(fixed["cat_batch_size"]) * 2,
            device,
            seed,
        )
        reload_difference = None
    validate_probabilities(final_calls, final_animals)
    audit = {
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(len(history)),
        "fixed_epoch_training": fixed_epochs is not None,
        "seed": int(seed),
        "training_seed": int(training_seed),
        "config": config,
        "train_role": training_role_identity(store, train_indices),
        "initial_common_state_sha256": initial_common_digest,
        "initial_full_state_sha256": initial_full_digest,
        "checkpoint_reload_applicable": fixed_epochs is None,
        "best_state_reload_max_probability_difference": reload_difference,
        "evaluation_metrics": metric_bundle(final_animals),
        "model": model.audit(),
        "train_seconds": float(time.perf_counter() - started),
        "history": history,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return audit, final_animals, final_calls


def manifest_payload(protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    dependencies = protocol["dependencies"]
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tests_sha256": dependencies["tests_sha256"],
        "roles_sha256": protocol["data"]["roles_sha256"],
        "embedding_sha256": protocol["data"]["frozen_embedding_sha256"],
        "feature_sha256": protocol["data"]["feature_sha256"],
        "pipelines": list(PIPELINES),
        "configurations": protocol["search"]["configurations"],
        "budget": protocol["budget"],
        "outer_test_accessed_at_manifest_creation": False,
        "outer_stage_requires_separate_authorization": True,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }


def require_manifest(
    run_root: Path, protocol: dict[str, Any], device: torch.device, resume: bool
) -> dict[str, Any]:
    payload = manifest_payload(protocol, device)
    path = run_root / "run_manifest.json"
    if path.exists():
        if not resume or read_json(path) != payload:
            raise RuntimeError("Existing IDEA-087 run manifest differs")
    else:
        if resume:
            raise RuntimeError("IDEA-087 resume requested without run manifest")
        write_json(path, payload)
    return payload


def require_cpu_preflight(run_root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    path = run_root / "cpu_preflight.json"
    if not path.is_file():
        raise RuntimeError("IDEA-087 CPU preflight is missing")
    preflight = read_json(path)
    expected = {
        "status": "GO",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
    }
    for key, value in expected.items():
        if preflight.get(key) != value:
            raise RuntimeError(f"IDEA-087 CPU preflight mismatch: {key}")
    return preflight


def inner_fit_path(
    run_root: Path,
    pipeline: str,
    config_id: str,
    outer_fold: int,
    inner_fold: int,
    base_seed: int,
) -> Path:
    return (
        run_root
        / "inner"
        / pipeline
        / config_id
        / f"outer_{outer_fold}"
        / f"inner_{inner_fold}"
        / f"search_seed_{base_seed}"
        / "fit_summary.json"
    )


def outer_fit_path(
    run_root: Path,
    pipeline: str,
    policy: str,
    outer_fold: int,
    base_seed: int,
) -> Path:
    return (
        run_root
        / "outer"
        / pipeline
        / policy
        / f"outer_{outer_fold}"
        / f"refit_seed_{base_seed}"
        / "fit_summary.json"
    )


def validate_saved_prediction_paths(
    fit: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for kind in ("animal", "call"):
        path = REPO_ROOT / fit[f"prediction_{kind}_path"]
        if not path.is_file() or sha256(path) != fit[f"prediction_{kind}_sha256"]:
            raise RuntimeError(f"IDEA-087 saved {kind} prediction changed")
        dtype = {"cat_id": str, "call_id": str} if kind == "call" else {"cat_id": str}
        frames[kind] = pd.read_csv(path, dtype=dtype)
    validate_probabilities(frames["call"], frames["animal"])
    return frames["animal"], frames["call"]


def validate_inner_fit(
    fit: dict[str, Any],
    protocol: dict[str, Any],
    pipeline: str,
    config: dict[str, Any],
    outer_fold: int,
    inner_fold: int,
    base_seed: int,
    train_identity: dict[str, Any],
    validation_cat_hash: str,
) -> None:
    expected = {
        "status": "complete",
        "stage": "inner",
        "pipeline": pipeline,
        "config_id": config["config_id"],
        "config": config,
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "search_base_seed": int(base_seed),
        "full_seed": full_search_seed(base_seed, outer_fold, inner_fold),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "train_role": train_identity,
        "validation_cat_ids_sha256": validation_cat_hash,
        "outer_test_accessed": False,
    }
    for key, value in expected.items():
        if fit.get(key) != value:
            raise RuntimeError(f"IDEA-087 inner resume mismatch: {key}")
    animals, _ = validate_saved_prediction_paths(fit)
    actual_hash = text_sha256("\n".join(sorted(animals["cat_id"].astype(str))))
    if actual_hash != validation_cat_hash:
        raise RuntimeError("IDEA-087 saved inner validation cats changed")


def select_configuration(
    config_scores: list[dict[str, Any]], tolerance: float
) -> tuple[dict[str, Any], list[str]]:
    if len(config_scores) != 8:
        raise RuntimeError("IDEA-087 selection requires eight configuration scores")
    best_f1 = max(float(row["mean_macro_f1"]) for row in config_scores)
    candidates = [
        row
        for row in config_scores
        if float(row["mean_macro_f1"]) >= best_f1 - float(tolerance) - 1.0e-12
    ]
    candidates.sort(
        key=lambda row: (
            float(row["mean_brier"]),
            float(row["macro_f1_sample_sd"]),
            -float(row["mean_plain_accuracy"]),
            str(row["config_id"]),
        )
    )
    return candidates[0], [str(row["config_id"]) for row in candidates]


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cpu":
        raise RuntimeError("IDEA-087 preflight must run on CPU")
    idea068.configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = resolve_run_root(args.output_subdir)
    if run_root.exists() and not args.resume:
        raise FileExistsError(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    store, features, roles = load_inputs(protocol)
    inner_roles = build_inner_roles(protocol, roles)
    inner_roles_path = run_root / protocol["outputs"]["inner_roles"]
    csv_bytes = inner_roles.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if inner_roles_path.exists():
        if not args.resume or inner_roles_path.read_bytes() != csv_bytes:
            raise RuntimeError("IDEA-087 inner roles changed")
    else:
        inner_roles_path.write_bytes(csv_bytes)
    configs = configuration_map(protocol)
    first_indices = inner_indices(store, inner_roles, 0, 0)
    probe = first_indices["validation"][: min(32, len(first_indices["validation"]))]
    initial_logits: dict[str, np.ndarray] = {}
    common_digests: dict[str, str] = {}
    full_digests: dict[str, str] = {}
    parameters: dict[str, int] = {}
    for pipeline in PIPELINES:
        idea068.idea051.reference.historical.set_seed(
            int(protocol["search"]["search_base_seeds"][0])
        )
        model = build_model(pipeline, configs["q00"], store, features, first_indices["train"])
        parameters[pipeline] = int(sum(value.numel() for value in model.parameters()))
        common_digests[pipeline] = state_digest(model, common_only=True)
        full_digests[pipeline] = state_digest(model)
        model.eval()
        with torch.no_grad():
            initial_logits[pipeline] = model(
                torch.from_numpy(store.frozen_embeddings[probe]),
                torch.from_numpy(features[probe]),
            ).numpy()
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError("IDEA-087 preflight parameter mismatch")
    if len(set(common_digests.values())) != 1:
        raise RuntimeError("IDEA-087 common model states differ")
    if full_digests[PIPELINES[1]] != full_digests[PIPELINES[2]]:
        raise RuntimeError("IDEA-087 U1/C1 initial states differ")
    logit_differences = {
        pipeline: float(
            np.max(np.abs(initial_logits[pipeline] - initial_logits[PIPELINES[0]]))
        )
        for pipeline in PIPELINES[1:]
    }
    if any(value != 0.0 for value in logit_differences.values()):
        raise RuntimeError("IDEA-087 initial logits differ")
    q00 = configs["q00"]
    source_protocol = read_json(
        REPO_ROOT / protocol["dependencies"]["idea076_protocol_path"]
    )
    q00_match = {
        "learning_rate": q00["learning_rate"]
        == source_protocol["fixed_training"]["learning_rate"],
        "dropout": q00["dropout"] == source_protocol["fixed_training"]["dropout"],
        "weight_decay": q00["weight_decay"] == 0.0,
    }
    if not all(q00_match.values()):
        raise RuntimeError("IDEA-087 q00 no longer matches the original training setup")
    cell_audits = []
    for outer_fold in protocol["data"]["outer_folds"]:
        outer = outer_indices(store, roles, outer_fold)
        validation_cats: list[str] = []
        for inner_fold in range(3):
            indices = inner_indices(store, inner_roles, outer_fold, inner_fold)
            train_identity = training_role_identity(store, indices["train"])
            validation_identity = training_role_identity(store, indices["validation"])
            validation_cats.extend(
                sorted(set(store.cat_ids[indices["validation"]].astype(str)))
            )
            cell_audits.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "train": train_identity,
                    "validation": validation_identity,
                }
            )
        outer_train_cats = set(store.cat_ids[outer["train"]].astype(str))
        if set(validation_cats) != outer_train_cats or len(validation_cats) != len(
            set(validation_cats)
        ):
            raise RuntimeError("IDEA-087 preflight OOF coverage failed")
    result = {
        "status": "GO",
        "read_only_outer_test": True,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "inner_roles_path": inner_roles_path.relative_to(REPO_ROOT).as_posix(),
        "inner_roles_sha256": sha256(inner_roles_path),
        "parameters": parameters,
        "common_initial_state_equal": True,
        "U1_C1_full_initial_state_equal": True,
        "initial_logit_differences": logit_differences,
        "q00_matches_original": q00_match,
        "seed_audit": seed_audit(protocol),
        "class_weight_scope": "current fit training calls only",
        "inner_cells": cell_audits,
        "expected_inner_fits": 576,
        "maximum_outer_refits": 72,
        "outer_test_predictions_generated": False,
        "device": "cpu",
    }
    path = run_root / protocol["outputs"]["cpu_preflight"]
    if path.exists():
        if not args.resume or read_json(path) != result:
            raise RuntimeError("IDEA-087 CPU preflight output differs")
    else:
        write_json(path, result)
    return result


def run_inner(args: argparse.Namespace) -> dict[str, Any]:
    if not args.director_authorized:
        raise RuntimeError("IDEA-087 inner GPU work requires director authorization")
    if args.max_fits is not None and args.max_fits < 1:
        raise ValueError("--max-fits must be positive")
    idea068.configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = resolve_run_root(args.output_subdir)
    require_cpu_preflight(run_root, protocol)
    device = idea068.idea051.reference.historical.idea019.resolve_device(args.device)
    require_manifest(run_root, protocol, device, args.resume)
    store, features, roles = load_inputs(protocol)
    inner_roles_path = run_root / protocol["outputs"]["inner_roles"]
    inner_roles = pd.read_csv(inner_roles_path, dtype={"cat_id": str})
    if sha256(inner_roles_path) != read_json(run_root / "cpu_preflight.json")[
        "inner_roles_sha256"
    ]:
        raise RuntimeError("IDEA-087 inner role file changed after preflight")
    configs = configuration_map(protocol)
    completed = 0
    created = 0
    for outer_fold in protocol["data"]["outer_folds"]:
        for inner_fold in range(3):
            indices = inner_indices(store, inner_roles, outer_fold, inner_fold)
            train_identity = training_role_identity(store, indices["train"])
            validation_cats = sorted(set(store.cat_ids[indices["validation"]].astype(str)))
            validation_cat_hash = text_sha256("\n".join(validation_cats))
            for base_seed in protocol["search"]["search_base_seeds"]:
                seed = full_search_seed(base_seed, outer_fold, inner_fold)
                for config in protocol["search"]["configurations"]:
                    for pipeline in PIPELINES:
                        summary_path = inner_fit_path(
                            run_root,
                            pipeline,
                            config["config_id"],
                            outer_fold,
                            inner_fold,
                            base_seed,
                        )
                        if summary_path.exists():
                            if not args.resume:
                                raise FileExistsError(summary_path)
                            fit = read_json(summary_path)
                            validate_inner_fit(
                                fit,
                                protocol,
                                pipeline,
                                config,
                                outer_fold,
                                inner_fold,
                                base_seed,
                                train_identity,
                                validation_cat_hash,
                            )
                        else:
                            if args.max_fits is not None and created >= args.max_fits:
                                return {
                                    "status": "partial",
                                    "completed_or_validated": completed,
                                    "new_fits": created,
                                    "expected_inner_fits": 576,
                                    "outer_test_accessed": False,
                                }
                            print(
                                f"=== inner {pipeline} {config['config_id']} outer={outer_fold} "
                                f"inner={inner_fold} search_seed={base_seed} full_seed={seed} ===",
                                flush=True,
                            )
                            audit, animals, calls = train_fit(
                                pipeline,
                                config,
                                protocol,
                                store,
                                features,
                                indices["train"],
                                indices["validation"],
                                device,
                                seed,
                            )
                            if sorted(animals["cat_id"].astype(str)) != validation_cats:
                                raise RuntimeError("IDEA-087 inner validation cats changed")
                            output_dir = summary_path.parent
                            output_dir.mkdir(parents=True, exist_ok=True)
                            animal_path = output_dir / "validation_animal_predictions.csv"
                            call_path = output_dir / "validation_call_predictions.csv"
                            animals.to_csv(animal_path, index=False, lineterminator="\n")
                            calls.to_csv(call_path, index=False, lineterminator="\n")
                            fit = {
                                "status": "complete",
                                "stage": "inner",
                                "pipeline": pipeline,
                                "config_id": config["config_id"],
                                "config": config,
                                "outer_fold": int(outer_fold),
                                "inner_fold": int(inner_fold),
                                "search_base_seed": int(base_seed),
                                "full_seed": int(seed),
                                "protocol_sha256": sha256(PROTOCOL_PATH),
                                "runner_sha256": sha256(Path(__file__).resolve()),
                                "tests_sha256": protocol["dependencies"]["tests_sha256"],
                                "train_role": train_identity,
                                "validation_cat_ids_sha256": validation_cat_hash,
                                "prediction_animal_path": animal_path.relative_to(
                                    REPO_ROOT
                                ).as_posix(),
                                "prediction_animal_sha256": sha256(animal_path),
                                "prediction_call_path": call_path.relative_to(
                                    REPO_ROOT
                                ).as_posix(),
                                "prediction_call_sha256": sha256(call_path),
                                "audit": audit,
                                "outer_test_accessed": False,
                            }
                            write_json(summary_path, fit)
                            created += 1
                        completed += 1
    result = {
        "status": "complete",
        "completed_or_validated": completed,
        "new_fits": created,
        "expected_inner_fits": 576,
        "outer_test_accessed": False,
    }
    write_json(run_root / "inner_run_summary.json", result)
    return result


def load_all_inner_fits(
    run_root: Path,
    protocol: dict[str, Any],
    store: Any,
    inner_roles: pd.DataFrame,
) -> list[dict[str, Any]]:
    fits = []
    for outer_fold in protocol["data"]["outer_folds"]:
        for inner_fold in range(3):
            indices = inner_indices(store, inner_roles, outer_fold, inner_fold)
            identity = training_role_identity(store, indices["train"])
            validation_cat_hash = text_sha256(
                "\n".join(
                    sorted(set(store.cat_ids[indices["validation"]].astype(str)))
                )
            )
            for base_seed in protocol["search"]["search_base_seeds"]:
                for config in protocol["search"]["configurations"]:
                    for pipeline in PIPELINES:
                        path = inner_fit_path(
                            run_root,
                            pipeline,
                            config["config_id"],
                            outer_fold,
                            inner_fold,
                            base_seed,
                        )
                        if not path.is_file():
                            raise RuntimeError(f"IDEA-087 missing inner fit: {path}")
                        fit = read_json(path)
                        validate_inner_fit(
                            fit,
                            protocol,
                            pipeline,
                            config,
                            outer_fold,
                            inner_fold,
                            base_seed,
                            identity,
                            validation_cat_hash,
                        )
                        fit["_summary_path"] = path.relative_to(REPO_ROOT).as_posix()
                        fit["_summary_sha256"] = sha256(path)
                        fits.append(fit)
    if len(fits) != 576:
        raise RuntimeError("IDEA-087 inner fit count changed")
    return fits


def run_selection(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_fits is not None:
        raise RuntimeError("--max-fits is not valid for selection")
    idea068.configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = resolve_run_root(args.output_subdir)
    require_cpu_preflight(run_root, protocol)
    store, _, roles = load_inputs(protocol)
    inner_roles_path = run_root / protocol["outputs"]["inner_roles"]
    preflight = read_json(run_root / protocol["outputs"]["cpu_preflight"])
    if sha256(inner_roles_path) != preflight["inner_roles_sha256"]:
        raise RuntimeError("IDEA-087 inner roles changed before selection")
    inner_roles = pd.read_csv(inner_roles_path, dtype={"cat_id": str})
    fits = load_all_inner_fits(run_root, protocol, store, inner_roles)
    fit_lookup = {
        (
            fit["pipeline"],
            fit["config_id"],
            fit["outer_fold"],
            fit["inner_fold"],
            fit["search_base_seed"],
        ): fit
        for fit in fits
    }
    locks = []
    tolerance = float(protocol["selection"]["macro_f1_candidate_tolerance"])
    for pipeline in PIPELINES:
        for outer_fold in protocol["data"]["outer_folds"]:
            expected_cats = set(
                roles[
                    (roles["outer_fold"] == outer_fold)
                    & roles["role"].isin(("train", "validation"))
                ]["cat_id"].astype(str)
            )
            config_scores = []
            for config in protocol["search"]["configurations"]:
                seed_bundles = []
                best_epochs = []
                seed_oof_evidence = []
                for base_seed in protocol["search"]["search_base_seeds"]:
                    frames = []
                    for inner_fold in range(3):
                        fit = fit_lookup[
                            (
                                pipeline,
                                config["config_id"],
                                outer_fold,
                                inner_fold,
                                base_seed,
                            )
                        ]
                        frame = pd.read_csv(
                            REPO_ROOT / fit["prediction_animal_path"],
                            dtype={"cat_id": str},
                        )
                        frames.append(frame)
                        best_epochs.append(int(fit["audit"]["best_epoch"]))
                    oof = pd.concat(frames, ignore_index=True)
                    if oof["cat_id"].duplicated().any() or set(
                        oof["cat_id"].astype(str)
                    ) != expected_cats:
                        raise RuntimeError("IDEA-087 incomplete or duplicated seed OOF")
                    bundle = metric_bundle(oof)
                    seed_bundles.append(bundle)
                    seed_oof_evidence.append(
                        {
                            "search_base_seed": int(base_seed),
                            "cats": int(len(oof)),
                            "cat_ids_sha256": text_sha256(
                                "\n".join(sorted(oof["cat_id"].astype(str)))
                            ),
                            "metrics": bundle,
                        }
                    )
                f1_values = np.asarray(
                    [bundle["macro_f1"] for bundle in seed_bundles], dtype=np.float64
                )
                score = {
                    "config_id": config["config_id"],
                    "config": config,
                    "mean_macro_f1": float(f1_values.mean()),
                    "macro_f1_sample_sd": float(f1_values.std(ddof=1)),
                    "mean_brier": float(
                        np.mean([bundle["brier"] for bundle in seed_bundles])
                    ),
                    "mean_plain_accuracy": float(
                        np.mean(
                            [bundle["plain_accuracy"] for bundle in seed_bundles]
                        )
                    ),
                    "seed_oof": seed_oof_evidence,
                    "best_epochs_1_based": best_epochs,
                    "refit_epoch": refit_epoch(
                        best_epochs, int(protocol["fixed_training"]["maximum_epochs"])
                    ),
                }
                config_scores.append(score)
            selected, candidate_ids = select_configuration(config_scores, tolerance)
            original = next(row for row in config_scores if row["config_id"] == "q00")
            locks.append(
                {
                    "pipeline": pipeline,
                    "outer_fold": int(outer_fold),
                    "selected_config_id": selected["config_id"],
                    "selected_refit_epoch": selected["refit_epoch"],
                    "original_config_id": "q00",
                    "original_refit_epoch": original["refit_epoch"],
                    "candidate_pool_config_ids": candidate_ids,
                    "config_scores": config_scores,
                }
            )
    evidence_lines = [f"{fit['_summary_path']}:{fit['_summary_sha256']}" for fit in fits]
    evidence_digest = text_sha256("\n".join(sorted(evidence_lines)))
    selection = {
        "schema_version": "1.0",
        "status": "complete_locked_before_outer_evaluation",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "inner_roles_sha256": sha256(inner_roles_path),
        "inner_fits": 576,
        "inner_fit_evidence_sha256": evidence_digest,
        "selection_rule": protocol["selection"],
        "selection_locks": locks,
        "complete_locks": len(locks),
        "outer_test_accessed": False,
    }
    if len(locks) != 12:
        raise RuntimeError("IDEA-087 selection lock count changed")
    path = run_root / protocol["outputs"]["selection_lock"]
    if path.exists():
        if not args.resume or path.read_bytes() != canonical_json_bytes(selection):
            raise RuntimeError("IDEA-087 existing selection lock differs")
    else:
        write_json(path, selection)
    return {**selection, "selection_lock_sha256": sha256(path)}


def selection_row(
    selection: dict[str, Any], pipeline: str, outer_fold: int
) -> dict[str, Any]:
    rows = [
        row
        for row in selection["selection_locks"]
        if row["pipeline"] == pipeline and row["outer_fold"] == int(outer_fold)
    ]
    if len(rows) != 1:
        raise RuntimeError("IDEA-087 expected one selection row")
    return rows[0]


def validate_selection_lock(
    selection: dict[str, Any],
    selection_path: Path,
    run_root: Path,
    protocol: dict[str, Any],
) -> str:
    selection_sha = sha256(selection_path)
    preflight = require_cpu_preflight(run_root, protocol)
    expected = {
        "status": "complete_locked_before_outer_evaluation",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "inner_roles_sha256": preflight["inner_roles_sha256"],
        "inner_fits": 576,
        "complete_locks": 12,
        "outer_test_accessed": False,
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise RuntimeError(f"IDEA-087 selection lock mismatch: {key}")
    identities = [
        (str(row.get("pipeline")), int(row.get("outer_fold", -1)))
        for row in selection.get("selection_locks", [])
    ]
    expected_identities = [
        (pipeline, int(outer_fold))
        for pipeline in PIPELINES
        for outer_fold in protocol["data"]["outer_folds"]
    ]
    if sorted(identities) != sorted(expected_identities) or len(set(identities)) != 12:
        raise RuntimeError("IDEA-087 selection lock identities are incomplete")
    return selection_sha


def validate_outer_fit(
    fit: dict[str, Any],
    protocol: dict[str, Any],
    pipeline: str,
    policy: str,
    config: dict[str, Any],
    epoch: int,
    outer_fold: int,
    base_seed: int,
    selection_sha: str,
    train_identity: dict[str, Any],
) -> None:
    expected = {
        "stage": "outer",
        "pipeline": pipeline,
        "policy": policy,
        "config_id": config["config_id"],
        "config": config,
        "refit_epoch": int(epoch),
        "outer_fold": int(outer_fold),
        "refit_base_seed": int(base_seed),
        "full_seed": full_refit_seed(base_seed, outer_fold),
        "selection_lock_sha256": selection_sha,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "train_role": train_identity,
        "outer_test_accessed": True,
    }
    for key, value in expected.items():
        if fit.get(key) != value:
            raise RuntimeError(f"IDEA-087 outer resume mismatch: {key}")
    if fit.get("status") == "complete":
        validate_saved_prediction_paths(fit)
    elif fit.get("status") == "alias":
        source_path = REPO_ROOT / fit["alias_source_summary_path"]
        if not source_path.is_file() or sha256(source_path) != fit["alias_source_summary_sha256"]:
            raise RuntimeError("IDEA-087 alias source changed")
        source = read_json(source_path)
        source_expected = dict(expected)
        source_expected["policy"] = "original"
        for key, value in source_expected.items():
            if source.get(key) != value:
                raise RuntimeError(f"IDEA-087 alias source identity mismatch: {key}")
        if source.get("status") != "complete" or source.get("physical_fit") is not True:
            raise RuntimeError("IDEA-087 alias source is not a physical complete fit")
        for key in (
            "prediction_animal_path",
            "prediction_animal_sha256",
            "prediction_call_path",
            "prediction_call_sha256",
            "audit",
        ):
            if fit.get(key) != source.get(key):
                raise RuntimeError(f"IDEA-087 alias differs from source: {key}")
        validate_saved_prediction_paths(fit)
    else:
        raise RuntimeError("IDEA-087 invalid outer fit status")


def run_outer(args: argparse.Namespace) -> dict[str, Any]:
    if not args.director_authorized:
        raise RuntimeError("IDEA-087 outer evaluation requires director authorization")
    if not args.selection_lock_sha256:
        raise RuntimeError("IDEA-087 outer evaluation requires selection lock SHA-256")
    if args.max_fits is not None and args.max_fits < 1:
        raise ValueError("--max-fits must be positive")
    idea068.configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = resolve_run_root(args.output_subdir)
    require_cpu_preflight(run_root, protocol)
    selection_path = run_root / protocol["outputs"]["selection_lock"]
    selection = read_json(selection_path)
    actual_selection_sha = validate_selection_lock(
        selection, selection_path, run_root, protocol
    )
    if args.selection_lock_sha256.lower() != actual_selection_sha:
        raise RuntimeError("IDEA-087 outer selection SHA-256 argument differs")
    device = idea068.idea051.reference.historical.idea019.resolve_device(args.device)
    require_manifest(run_root, protocol, device, True)
    authorization = {
        "status": "director_authorized_outer_transition",
        "director_authorized": True,
        "selection_lock_sha256": actual_selection_sha,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "device": str(device),
        "outer_test_accessed_by_this_stage": True,
    }
    authorization_path = run_root / "outer_authorization_record.json"
    if authorization_path.exists():
        if not args.resume or read_json(authorization_path) != authorization:
            raise RuntimeError("IDEA-087 outer authorization record differs")
    else:
        write_json(authorization_path, authorization)
    store, features, roles = load_inputs(protocol)
    configs = configuration_map(protocol)
    completed = 0
    created = 0
    aliases = 0
    for outer_fold in protocol["data"]["outer_folds"]:
        indices = outer_indices(store, roles, outer_fold)
        train_identity = training_role_identity(store, indices["train"])
        expected_test_cats = sorted(set(store.cat_ids[indices["test"]].astype(str)))
        for base_seed in protocol["refit"]["refit_base_seeds"]:
            seed = full_refit_seed(base_seed, outer_fold)
            for pipeline in PIPELINES:
                lock = selection_row(selection, pipeline, outer_fold)
                policy_specs = {
                    "original": (
                        lock["original_config_id"],
                        int(lock["original_refit_epoch"]),
                    ),
                    "selected": (
                        lock["selected_config_id"],
                        int(lock["selected_refit_epoch"]),
                    ),
                }
                source_by_spec: dict[tuple[str, int], Path] = {}
                for policy in ("original", "selected"):
                    config_id, epoch = policy_specs[policy]
                    config = configs[config_id]
                    summary_path = outer_fit_path(
                        run_root, pipeline, policy, outer_fold, base_seed
                    )
                    spec = (config_id, epoch)
                    if summary_path.exists():
                        if not args.resume:
                            raise FileExistsError(summary_path)
                        fit = read_json(summary_path)
                        validate_outer_fit(
                            fit,
                            protocol,
                            pipeline,
                            policy,
                            config,
                            epoch,
                            outer_fold,
                            base_seed,
                            actual_selection_sha,
                            train_identity,
                        )
                        if fit["status"] == "complete":
                            source_by_spec[spec] = summary_path
                    elif spec in source_by_spec:
                        source_path = source_by_spec[spec]
                        source = read_json(source_path)
                        fit = {
                            "status": "alias",
                            "physical_fit": False,
                            "stage": "outer",
                            "pipeline": pipeline,
                            "policy": policy,
                            "config_id": config_id,
                            "config": config,
                            "refit_epoch": epoch,
                            "outer_fold": int(outer_fold),
                            "refit_base_seed": int(base_seed),
                            "full_seed": int(seed),
                            "selection_lock_sha256": actual_selection_sha,
                            "protocol_sha256": sha256(PROTOCOL_PATH),
                            "runner_sha256": sha256(Path(__file__).resolve()),
                            "tests_sha256": protocol["dependencies"]["tests_sha256"],
                            "train_role": train_identity,
                            "alias_source_summary_path": source_path.relative_to(
                                REPO_ROOT
                            ).as_posix(),
                            "alias_source_summary_sha256": sha256(source_path),
                            "prediction_animal_path": source["prediction_animal_path"],
                            "prediction_animal_sha256": source[
                                "prediction_animal_sha256"
                            ],
                            "prediction_call_path": source["prediction_call_path"],
                            "prediction_call_sha256": source["prediction_call_sha256"],
                            "audit": source["audit"],
                            "outer_test_accessed": True,
                        }
                        write_json(summary_path, fit)
                        aliases += 1
                    else:
                        if args.max_fits is not None and created >= args.max_fits:
                            return {
                                "status": "partial",
                                "completed_or_validated": completed,
                                "new_physical_fits": created,
                                "new_aliases": aliases,
                                "maximum_outer_refits": 72,
                                "selection_lock_sha256": actual_selection_sha,
                                "director_authorized": True,
                                "protocol_sha256": sha256(PROTOCOL_PATH),
                                "runner_sha256": sha256(Path(__file__).resolve()),
                                "tests_sha256": protocol["dependencies"]["tests_sha256"],
                                "device": str(device),
                                "outer_test_accessed": True,
                            }
                        print(
                            f"=== outer {pipeline} {policy} {config_id} outer={outer_fold} "
                            f"refit_seed={base_seed} full_seed={seed} epochs={epoch} ===",
                            flush=True,
                        )
                        audit, animals, calls = train_fit(
                            pipeline,
                            config,
                            protocol,
                            store,
                            features,
                            indices["train"],
                            indices["test"],
                            device,
                            seed,
                            fixed_epochs=epoch,
                        )
                        if sorted(animals["cat_id"].astype(str)) != expected_test_cats:
                            raise RuntimeError("IDEA-087 outer test cats changed")
                        output_dir = summary_path.parent
                        output_dir.mkdir(parents=True, exist_ok=True)
                        animal_path = output_dir / "outer_test_animal_predictions.csv"
                        call_path = output_dir / "outer_test_call_predictions.csv"
                        animals.to_csv(animal_path, index=False, lineterminator="\n")
                        calls.to_csv(call_path, index=False, lineterminator="\n")
                        fit = {
                            "status": "complete",
                            "physical_fit": True,
                            "stage": "outer",
                            "pipeline": pipeline,
                            "policy": policy,
                            "config_id": config_id,
                            "config": config,
                            "refit_epoch": epoch,
                            "outer_fold": int(outer_fold),
                            "refit_base_seed": int(base_seed),
                            "full_seed": int(seed),
                            "selection_lock_sha256": actual_selection_sha,
                            "protocol_sha256": sha256(PROTOCOL_PATH),
                            "runner_sha256": sha256(Path(__file__).resolve()),
                            "tests_sha256": protocol["dependencies"]["tests_sha256"],
                            "train_role": train_identity,
                            "prediction_animal_path": animal_path.relative_to(
                                REPO_ROOT
                            ).as_posix(),
                            "prediction_animal_sha256": sha256(animal_path),
                            "prediction_call_path": call_path.relative_to(
                                REPO_ROOT
                            ).as_posix(),
                            "prediction_call_sha256": sha256(call_path),
                            "audit": audit,
                            "outer_test_accessed": True,
                        }
                        write_json(summary_path, fit)
                        source_by_spec[spec] = summary_path
                        created += 1
                    completed += 1
    result = {
        "status": "complete",
        "completed_or_validated": completed,
        "new_physical_fits": created,
        "new_aliases": aliases,
        "maximum_outer_refits": 72,
        "selection_lock_sha256": actual_selection_sha,
        "director_authorized": True,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "device": str(device),
        "outer_test_accessed": True,
    }
    write_json(run_root / "outer_run_summary.json", result)
    return result


def comparison_summary(
    rows: list[dict[str, Any]], candidate: str, control: str
) -> dict[str, Any]:
    tolerance = 1.0e-12

    def directions(values: np.ndarray) -> dict[str, int]:
        return {
            "positive": int((values > tolerance).sum()),
            "tied": int((np.abs(values) <= tolerance).sum()),
            "negative": int((values < -tolerance).sum()),
        }

    metric_names = (
        "plain_accuracy",
        "macro_f1",
        "balanced_accuracy",
        "kitten_recall",
        "adult_recall",
        "senior_recall",
    )
    result: dict[str, Any] = {"candidate": candidate, "control": control}
    for metric in metric_names:
        values = np.asarray(
            [row[f"{candidate}_{metric}"] - row[f"{control}_{metric}"] for row in rows],
            dtype=np.float64,
        )
        result[f"{metric}_delta"] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            **directions(values),
            "worst": float(values.min()),
            "best": float(values.max()),
        }
    for metric in ("cross_entropy", "brier"):
        values = np.asarray(
            [row[f"{control}_{metric}"] - row[f"{candidate}_{metric}"] for row in rows],
            dtype=np.float64,
        )
        result[f"{metric}_gain"] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            **directions(values),
        }
    result["direction_tolerance"] = tolerance
    return result


def run_aggregate(args: argparse.Namespace) -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = resolve_run_root(args.output_subdir)
    require_cpu_preflight(run_root, protocol)
    selection_path = run_root / protocol["outputs"]["selection_lock"]
    selection = read_json(selection_path)
    selection_sha = validate_selection_lock(selection, selection_path, run_root, protocol)
    store, _, roles = load_inputs(protocol)
    expected_all_cats = set(store.cat_ids.astype(str))
    configs = configuration_map(protocol)
    fit_lookup: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    physical = 0
    aliases = 0
    for pipeline in PIPELINES:
        for policy in ("original", "selected"):
            for outer_fold in protocol["data"]["outer_folds"]:
                lock = selection_row(selection, pipeline, outer_fold)
                config_id = lock[f"{policy}_config_id"]
                epoch = int(lock[f"{policy}_refit_epoch"])
                indices = outer_indices(store, roles, outer_fold)
                identity = training_role_identity(store, indices["train"])
                for base_seed in protocol["refit"]["refit_base_seeds"]:
                    path = outer_fit_path(
                        run_root, pipeline, policy, outer_fold, base_seed
                    )
                    if not path.is_file():
                        raise RuntimeError(f"IDEA-087 missing outer fit: {path}")
                    fit = read_json(path)
                    validate_outer_fit(
                        fit,
                        protocol,
                        pipeline,
                        policy,
                        configs[config_id],
                        epoch,
                        outer_fold,
                        base_seed,
                        selection_sha,
                        identity,
                    )
                    physical += int(fit["status"] == "complete")
                    aliases += int(fit["status"] == "alias")
                    fit_lookup[(pipeline, policy, outer_fold, base_seed)] = fit
    seed_rows = []
    cell_rows = []
    for base_seed in protocol["refit"]["refit_base_seeds"]:
        seed_row: dict[str, Any] = {"refit_base_seed": int(base_seed)}
        for outer_fold in protocol["data"]["outer_folds"]:
            cell_row: dict[str, Any] = {
                "refit_base_seed": int(base_seed),
                "outer_fold": int(outer_fold),
            }
            for pipeline in PIPELINES:
                for policy in ("original", "selected"):
                    name = f"{pipeline}_{policy}"
                    fit = fit_lookup[(pipeline, policy, outer_fold, base_seed)]
                    frame = pd.read_csv(
                        REPO_ROOT / fit["prediction_animal_path"], dtype={"cat_id": str}
                    )
                    bundle = metric_bundle(frame)
                    flat = {
                        "plain_accuracy": bundle["plain_accuracy"],
                        "macro_f1": bundle["macro_f1"],
                        "balanced_accuracy": bundle["balanced_accuracy"],
                        "kitten_recall": bundle["class_recall"]["kitten"],
                        "adult_recall": bundle["class_recall"]["adult"],
                        "senior_recall": bundle["class_recall"]["senior"],
                        "cross_entropy": bundle["cross_entropy"],
                        "brier": bundle["brier"],
                    }
                    for metric, value in flat.items():
                        cell_row[f"{name}_{metric}"] = value
            cell_rows.append(cell_row)
        for pipeline in PIPELINES:
            for policy in ("original", "selected"):
                name = f"{pipeline}_{policy}"
                frames = [
                    pd.read_csv(
                        REPO_ROOT
                        / fit_lookup[(pipeline, policy, fold, base_seed)][
                            "prediction_animal_path"
                        ],
                        dtype={"cat_id": str},
                    )
                    for fold in protocol["data"]["outer_folds"]
                ]
                oof = pd.concat(frames, ignore_index=True)
                if (
                    len(oof) != 111
                    or oof["cat_id"].duplicated().any()
                    or set(oof["cat_id"].astype(str)) != expected_all_cats
                ):
                    raise RuntimeError("IDEA-087 outer folds do not form 111-cat OOF")
                bundle = metric_bundle(oof)
                seed_row[f"{name}_plain_accuracy"] = bundle["plain_accuracy"]
                seed_row[f"{name}_macro_f1"] = bundle["macro_f1"]
                seed_row[f"{name}_balanced_accuracy"] = bundle["balanced_accuracy"]
                seed_row[f"{name}_kitten_recall"] = bundle["class_recall"]["kitten"]
                seed_row[f"{name}_adult_recall"] = bundle["class_recall"]["adult"]
                seed_row[f"{name}_senior_recall"] = bundle["class_recall"]["senior"]
                seed_row[f"{name}_cross_entropy"] = bundle["cross_entropy"]
                seed_row[f"{name}_brier"] = bundle["brier"]
        seed_rows.append(seed_row)
    pipeline_policy_means: dict[str, Any] = {}
    for pipeline in PIPELINES:
        for policy in ("original", "selected"):
            name = f"{pipeline}_{policy}"
            pipeline_policy_means[name] = {
                metric: float(np.mean([row[f"{name}_{metric}"] for row in seed_rows]))
                for metric in (
                    "plain_accuracy",
                    "macro_f1",
                    "balanced_accuracy",
                    "kitten_recall",
                    "adult_recall",
                    "senior_recall",
                    "cross_entropy",
                    "brier",
                )
            }
    comparisons = {}
    for pipeline in PIPELINES:
        comparisons[f"{pipeline}_tuned_minus_fixed_seed_oof"] = comparison_summary(
            seed_rows, f"{pipeline}_selected", f"{pipeline}_original"
        )
        comparisons[f"{pipeline}_tuned_minus_fixed_cells"] = comparison_summary(
            cell_rows, f"{pipeline}_selected", f"{pipeline}_original"
        )
    for policy, label in (("original", "fixed"), ("selected", "tuned")):
        for pipeline in PIPELINES[1:]:
            comparisons[f"{pipeline}_{label}_minus_A0_{label}_seed_oof"] = comparison_summary(
                seed_rows, f"{pipeline}_{policy}", f"{PIPELINES[0]}_{policy}"
            )
            comparisons[f"{pipeline}_{label}_minus_A0_{label}_cells"] = comparison_summary(
                cell_rows, f"{pipeline}_{policy}", f"{PIPELINES[0]}_{policy}"
            )
        comparisons[f"C1_minus_U1_{label}_seed_oof"] = comparison_summary(
            seed_rows, f"{PIPELINES[2]}_{policy}", f"{PIPELINES[1]}_{policy}"
        )
        comparisons[f"C1_minus_U1_{label}_cells"] = comparison_summary(
            cell_rows, f"{PIPELINES[2]}_{policy}", f"{PIPELINES[1]}_{policy}"
        )
    summary = {
        "status": "complete",
        "exploratory_not_independent_confirmation": True,
        "no_global_gate": True,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "selection_lock_sha256": selection_sha,
        "selection_locks": selection["selection_locks"],
        "physical_outer_fits": physical,
        "outer_alias_records": aliases,
        "maximum_outer_refits": 72,
        "refit_seed_oof_estimates": 3,
        "paired_refit_seed_by_outer_fold_cells": 12,
        "repeated_animal_occurrences_per_pipeline_policy": 333,
        "unique_cats": 111,
        "pipeline_policy_seed_oof_means": pipeline_policy_means,
        "comparisons": comparisons,
        "seed_oof_results": seed_rows,
        "cell_results": cell_rows,
        "outer_test_accessed": True,
    }
    summary_path = run_root / protocol["outputs"]["summary"]
    if summary_path.exists():
        if not args.resume or summary_path.read_bytes() != canonical_json_bytes(summary):
            raise RuntimeError("IDEA-087 existing aggregate differs")
    else:
        write_json(summary_path, summary)
    compact = {
        "status": "complete",
        "summary_path": summary_path.relative_to(REPO_ROOT).as_posix(),
        "summary_sha256": sha256(summary_path),
        "selection_lock_sha256": selection_sha,
        "physical_outer_fits": physical,
        "outer_alias_records": aliases,
        "outer_test_accessed": True,
    }
    write_json(run_root / "run_summary.json", compact)
    final_manifest = {
        "status": "complete",
        "initial_manifest_path": (run_root / "run_manifest.json").relative_to(
            REPO_ROOT
        ).as_posix(),
        "initial_manifest_sha256": sha256(run_root / "run_manifest.json"),
        "outer_authorization_record_path": (
            run_root / "outer_authorization_record.json"
        ).relative_to(REPO_ROOT).as_posix(),
        "outer_authorization_record_sha256": sha256(
            run_root / "outer_authorization_record.json"
        ),
        "selection_lock_sha256": selection_sha,
        "summary_sha256": sha256(summary_path),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "outer_test_accessed": True,
    }
    write_json(run_root / "final_stage_manifest.json", final_manifest)
    return summary


def main() -> None:
    args = parse_args()
    if args.stage == "preflight":
        result = run_preflight(args)
    elif args.stage == "inner":
        result = run_inner(args)
    elif args.stage == "selection":
        result = run_selection(args)
    elif args.stage == "outer":
        result = run_outer(args)
    else:
        result = run_aggregate(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
