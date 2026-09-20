"""Run IDEA-089 unified MeowAgeNet core comparisons.

The runner has four evidence-producing transitions:

1. CPU preflight (no real-data training),
2. train/validation epoch selection,
3. a complete epoch-selection lock,
4. fixed-epoch outer refits followed by one locked test inference.

Historical code and data are imported read-only.  Every output is written below the
new IDEA-089 run directory.
"""

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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_meowagenet_idea068_age_sensitive_ast as idea068  # noqa: E402
import run_meowagenet_idea071_bounded_dual_path_fusion as idea071  # noqa: E402
import run_meowagenet_idea076_C1_final_seed_confirmation as idea076  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT / "configs" / "protocol" / "meowagenet_idea089_unified_core_v1.json"
)
DEFAULT_RUN_SUBDIR = "meowagenet_idea089_unified_core_v1"
VGG_PIPELINES = ("VGG128_no_f0", "VGG129_with_f0")
AST_PIPELINES = (
    "A0_ast_only",
    "D0_direct_concat",
    "U1_wide_unbounded_additive",
    "C1_bounded_wide_additive",
)
PIPELINES = VGG_PIPELINES + AST_PIPELINES
PROBABILITY_COLUMNS = ("prob_kitten", "prob_adult", "prob_senior")
LABEL_NAMES = ("kitten", "adult", "senior")
LABEL_BY_AGE = {name: index for index, name in enumerate(LABEL_NAMES)}
EXPECTED_TRAINABLE_PARAMETERS = {
    "VGG128_no_f0": 17_155,
    "VGG129_with_f0": 17_283,
    "A0_ast_only": 99_075,
    "D0_direct_concat": 101_635,
    "U1_wide_unbounded_additive": 108_143,
    "C1_bounded_wide_additive": 108_143,
}
EXPECTED_TOTAL_STATE_PARAMETERS = {
    **EXPECTED_TRAINABLE_PARAMETERS,
    "VGG128_no_f0": 17_411,
    "VGG129_with_f0": 17_539,
}
COMPARISONS = {
    "C1_minus_A0": ("C1_bounded_wide_additive", "A0_ast_only"),
    "C1_minus_U1": (
        "C1_bounded_wide_additive",
        "U1_wide_unbounded_additive",
    ),
    "C1_minus_D0": ("C1_bounded_wide_additive", "D0_direct_concat"),
    "U1_minus_A0": ("U1_wide_unbounded_additive", "A0_ast_only"),
    "D0_minus_A0": ("D0_direct_concat", "A0_ast_only"),
    "U1_minus_D0": ("U1_wide_unbounded_additive", "D0_direct_concat"),
    "VGG129_minus_VGG128": ("VGG129_with_f0", "VGG128_no_f0"),
    "A0_minus_VGG129": ("A0_ast_only", "VGG129_with_f0"),
    "C1_minus_VGG129": ("C1_bounded_wide_additive", "VGG129_with_f0"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("preflight", "selection", "lock", "outer", "aggregate"),
        required=True,
    )
    parser.add_argument("--output-subdir", default=DEFAULT_RUN_SUBDIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--director-authorized", action="store_true")
    parser.add_argument(
        "--authorization-scope",
        choices=(
            "initial-six-selection",
            "full-selection",
            "initial-six-outer",
            "full-outer",
        ),
    )
    parser.add_argument("--preflight-sha256")
    parser.add_argument("--selection-lock-sha256")
    parser.add_argument("--pipeline", choices=PIPELINES)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--max-fits", type=int)
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


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def resolve_run_root(output_subdir: str) -> Path:
    if "idea089" not in output_subdir.lower():
        raise ValueError("--output-subdir must name a new IDEA-089 directory")
    root = (REPO_ROOT / "runs" / output_subdir).resolve()
    runs_root = (REPO_ROOT / "runs").resolve()
    if runs_root not in root.parents:
        raise ValueError("--output-subdir must remain below runs")
    return root


def full_seed(base_seed: int, repeat: int, fold: int) -> int:
    return int(base_seed) + 10_000 * int(repeat) + 100 * int(fold)


def configure_determinism() -> None:
    idea068.configure_determinism()
    tf.config.experimental.enable_op_determinism()


def _dependency_checks(protocol: dict[str, Any]) -> dict[Path, str]:
    dependencies = protocol["dependencies"]
    checks = {
        REPO_ROOT / protocol["data"]["roles_path"]: protocol["data"][
            "roles_sha256"
        ],
        REPO_ROOT / protocol["data"]["dataset_manifest_path"]: protocol["data"][
            "dataset_manifest_sha256"
        ],
        REPO_ROOT / protocol["data"]["vggish_path"]: protocol["data"][
            "vggish_sha256"
        ],
        REPO_ROOT / protocol["data"]["frozen_embedding_path"]: protocol["data"][
            "frozen_embedding_sha256"
        ],
        REPO_ROOT / protocol["data"]["feature_path"]: protocol["data"][
            "feature_sha256"
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
        REPO_ROOT / dependencies["plan_path"]: dependencies["plan_sha256"],
        REPO_ROOT / dependencies["tests_path"]: dependencies["tests_sha256"],
        Path(__file__).resolve(): dependencies["runner_sha256"],
    }
    independent_path = dependencies.get("independent_design_audit_path")
    independent_hash = dependencies.get("independent_design_audit_sha256")
    if independent_path and independent_hash:
        checks[REPO_ROOT / independent_path] = independent_hash
    return checks


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "meowagenet-idea089-unified-core-v1":
        raise RuntimeError("Unexpected IDEA-089 protocol")
    if protocol.get("status") != "locked_before_cpu_preflight":
        raise RuntimeError("IDEA-089 protocol is not locked before CPU preflight")
    if tuple(protocol["models"]["pipelines"]) != PIPELINES:
        raise RuntimeError("IDEA-089 pipeline matrix changed")
    if protocol["models"]["trainable_parameter_counts"] != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("IDEA-089 trainable parameter counts changed")
    if protocol["models"]["total_state_parameter_counts"] != EXPECTED_TOTAL_STATE_PARAMETERS:
        raise RuntimeError("IDEA-089 total/state parameter counts changed")
    if protocol["evaluation"]["comparisons"] != list(COMPARISONS):
        raise RuntimeError("IDEA-089 comparison order changed")
    if protocol["data"]["repeats"] != [0, 1, 2]:
        raise RuntimeError("IDEA-089 repeat scope changed")
    if protocol["data"]["folds"] != [0, 1, 2, 3]:
        raise RuntimeError("IDEA-089 fold scope changed")
    if protocol["training"]["model_seeds"] != [17, 43, 101]:
        raise RuntimeError("IDEA-089 seed scope changed")
    if int(protocol["budget"]["selection_fits"]) != 216:
        raise RuntimeError("IDEA-089 selection budget changed")
    if int(protocol["budget"]["outer_fits"]) != 216:
        raise RuntimeError("IDEA-089 outer budget changed")
    if int(protocol["budget"]["physical_fits"]) != 432:
        raise RuntimeError("IDEA-089 total budget changed")
    if float(protocol["training"]["minimum_delta"]) != 1.0e-6:
        raise RuntimeError("IDEA-089 minimum delta changed")
    for path, expected in _dependency_checks(protocol).items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"IDEA-089 dependency mismatch: {path}")


def set_seed(seed: int) -> None:
    idea068.idea051.reference.historical.set_seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))


def load_inputs(
    protocol: dict[str, Any],
) -> tuple[Any, np.ndarray, pd.DataFrame, dict[str, np.ndarray]]:
    store = idea068.idea051.reference.historical.idea019.load_feature_store()
    if len(store.call_ids) != 792 or len(np.unique(store.cat_ids.astype(str))) != 111:
        raise RuntimeError("IDEA-089 expected 792 calls from 111 cats")
    loaded = np.load(REPO_ROOT / protocol["data"]["feature_path"])
    if tuple(loaded["feature_names"].astype(str)) != idea068.FEATURE_NAMES:
        raise RuntimeError("IDEA-089 acoustic feature names changed")
    if not np.array_equal(loaded["call_ids"].astype(str), store.call_ids.astype(str)):
        raise RuntimeError("IDEA-089 acoustic feature order changed")
    acoustic = loaded["features"].astype(np.float32)
    if acoustic.shape != (792, 20):
        raise RuntimeError("IDEA-089 acoustic feature shape changed")

    roles = pd.read_csv(
        REPO_ROOT / protocol["data"]["roles_path"], dtype={"cat_id": str}
    )
    roles = roles[roles["repeat"].isin(protocol["data"]["repeats"])].copy()
    if len(roles) != 3 * 4 * 111:
        raise RuntimeError("IDEA-089 role table scope changed")

    frame = pd.read_csv(
        REPO_ROOT / protocol["data"]["vggish_path"], dtype={"cat_id": str}
    )
    analysis_cats = set(store.cat_ids.astype(str))
    frame = frame[frame["cat_id"].astype(str).isin(analysis_cats)].reset_index(drop=True)
    labels = np.where(
        frame["target"].to_numpy(dtype=np.float64) < 0.5,
        0,
        np.where(frame["target"].to_numpy(dtype=np.float64) < 10.0, 1, 2),
    ).astype(np.int64)
    feature128 = frame[[str(index) for index in range(128)]].to_numpy(
        dtype=np.float32
    )
    feature129 = frame[
        [str(index) for index in range(128)] + ["mean_freq"]
    ].to_numpy(dtype=np.float32)
    vggish = {
        "features128": feature128,
        "features129": feature129,
        "cat_ids": frame["cat_id"].astype(str).to_numpy(),
        "labels": labels,
        "unit_ids": np.asarray(
            [f"vggish-row-{index:04d}" for index in range(len(frame))], dtype=str
        ),
    }
    if len(frame) != 936 or len(np.unique(vggish["cat_ids"])) != 111:
        raise RuntimeError("IDEA-089 VGGish clean111 projection changed")
    return store, acoustic, roles, vggish


def role_mapping(
    roles: pd.DataFrame, repeat: int, fold: int
) -> dict[str, str]:
    selected = roles[
        (roles["repeat"] == int(repeat)) & (roles["outer_fold"] == int(fold))
    ]
    if len(selected) != 111 or selected["cat_id"].duplicated().any():
        raise RuntimeError("IDEA-089 incomplete role cell")
    mapping = dict(
        zip(selected["cat_id"].astype(str), selected["role"].astype(str), strict=True)
    )
    if set(mapping.values()) != {"train", "validation", "test"}:
        raise RuntimeError("IDEA-089 role cell lacks train/validation/test")
    return mapping


def indices_by_role(
    cat_ids: np.ndarray, roles: pd.DataFrame, repeat: int, fold: int
) -> dict[str, np.ndarray]:
    mapping = role_mapping(roles, repeat, fold)
    unit_roles = np.asarray([mapping[str(cat)] for cat in cat_ids.astype(str)])
    result = {
        role: np.flatnonzero(unit_roles == role).astype(np.int64)
        for role in ("train", "validation", "test")
    }
    if any(len(value) == 0 for value in result.values()):
        raise RuntimeError("IDEA-089 empty unit role")
    return result


def validate_role_bank(
    store: Any, roles: pd.DataFrame, vggish: dict[str, np.ndarray], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    expected_cats = set(store.cat_ids.astype(str))
    audits: list[dict[str, Any]] = []
    for repeat in protocol["data"]["repeats"]:
        test_seen: list[str] = []
        for fold in protocol["data"]["folds"]:
            mapping = role_mapping(roles, repeat, fold)
            if set(mapping) != expected_cats:
                raise RuntimeError("IDEA-089 role cats differ from analysis cats")
            role_sets = {
                name: {cat for cat, role in mapping.items() if role == name}
                for name in ("train", "validation", "test")
            }
            if any(role_sets[left] & role_sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
                raise RuntimeError("IDEA-089 cats cross roles")
            cell = roles[
                (roles["repeat"] == repeat) & (roles["outer_fold"] == fold)
            ]
            for name in ("train", "validation", "test"):
                labels = set(cell[cell["role"] == name]["age_group"].astype(str))
                if labels != set(LABEL_NAMES):
                    raise RuntimeError("IDEA-089 role is missing an age class")
            test_seen.extend(sorted(role_sets["test"]))
            audits.append(
                {
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "train_cats": len(role_sets["train"]),
                    "validation_cats": len(role_sets["validation"]),
                    "test_cats": len(role_sets["test"]),
                }
            )
        if len(test_seen) != 111 or len(set(test_seen)) != 111:
            raise RuntimeError("IDEA-089 outer folds do not form complete cat OOF")

    role_label = roles[["cat_id", "age_group"]].drop_duplicates()
    if role_label["cat_id"].duplicated().any():
        raise RuntimeError("IDEA-089 role labels disagree across cells")
    expected_labels = {
        str(row.cat_id): LABEL_BY_AGE[str(row.age_group)]
        for row in role_label.itertuples(index=False)
    }
    for source_cat_ids, source_labels in (
        (store.cat_ids.astype(str), store.labels.astype(np.int64)),
        (vggish["cat_ids"].astype(str), vggish["labels"].astype(np.int64)),
    ):
        for cat in np.unique(source_cat_ids):
            labels = np.unique(source_labels[source_cat_ids == cat])
            if labels.shape != (1,) or int(labels[0]) != expected_labels[str(cat)]:
                raise RuntimeError(f"IDEA-089 label mismatch for {cat}")
    return audits


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int64), minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("IDEA-089 training role is missing a class")
    return (len(labels) / (3.0 * counts)).astype(np.float32)


def standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale = np.where(scale > 1.0e-12, scale, 1.0).astype(np.float32)
    return mean, scale


def acoustic_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        median = np.nanmedian(values, axis=0)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    imputed = np.where(np.isfinite(values), values, median[None, :])
    mean = imputed.mean(axis=0).astype(np.float32)
    scale = imputed.std(axis=0).astype(np.float32)
    scale = np.where(scale > 1.0e-8, scale, 1.0).astype(np.float32)
    return median, mean, scale


class DirectConcatClassifier(torch.nn.Module):
    """Simple standardized [AST768, acoustic20] -> 128 -> 3 control."""

    def __init__(
        self,
        ast_mean: np.ndarray,
        ast_scale: np.ndarray,
        acoustic_train: np.ndarray,
        dropout: float,
    ) -> None:
        super().__init__()
        safe_ast_scale = np.where(ast_scale > 1.0e-12, ast_scale, 1.0).astype(
            np.float32
        )
        median, acoustic_mean, acoustic_scale = acoustic_standardizer(acoustic_train)
        self.register_buffer("ast_mean", torch.from_numpy(ast_mean.astype(np.float32)))
        self.register_buffer("ast_scale", torch.from_numpy(safe_ast_scale))
        self.register_buffer("age_median", torch.from_numpy(median))
        self.register_buffer("age_mean", torch.from_numpy(acoustic_mean))
        self.register_buffer("age_scale", torch.from_numpy(acoustic_scale))
        self.concat_linear = torch.nn.Linear(788, 128)
        self.relu = torch.nn.ReLU()
        self.batch_norm = torch.nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01)
        self.dropout = torch.nn.Dropout(float(dropout))
        self.output = torch.nn.Linear(128, 3)

    def forward(
        self, ast_embeddings: torch.Tensor, age_features: torch.Tensor
    ) -> torch.Tensor:
        ast = (ast_embeddings - self.ast_mean) / self.ast_scale
        imputed = torch.where(torch.isfinite(age_features), age_features, self.age_median)
        acoustic = (imputed - self.age_mean) / self.age_scale
        hidden = self.relu(self.concat_linear(torch.cat((ast, acoustic), dim=1)))
        hidden = self.batch_norm(hidden)
        hidden = self.dropout(hidden)
        return self.output(hidden)

    def audit(self) -> dict[str, Any]:
        extra = self.concat_linear.weight[:, 768:].detach()
        return {
            "fusion": "direct_standardized_concat",
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in self.parameters())
            ),
            "current_extra_column_max_abs": float(extra.abs().max().cpu()),
        }


def build_ast_model(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    acoustic: np.ndarray,
    train_indices: np.ndarray,
) -> torch.nn.Module:
    embeddings = store.frozen_embeddings[train_indices]
    common = {
        "ast_mean": embeddings.mean(axis=0),
        "ast_scale": embeddings.std(axis=0),
        "age_train": acoustic[train_indices],
        "dropout": float(protocol["training"]["ast"]["dropout"]),
    }
    if pipeline == "A0_ast_only":
        return idea068.AgeResidualClassifier(
            pipeline="A0_ast_only", age_hidden_units=32, **common
        )
    if pipeline == "U1_wide_unbounded_additive":
        return idea076.WideUnboundedAdditiveClassifier(**common)
    if pipeline == "C1_bounded_wide_additive":
        return idea071.BoundedWideAdditiveClassifier(**common)
    if pipeline == "D0_direct_concat":
        reference = idea068.AgeResidualClassifier(
            pipeline="A0_ast_only", age_hidden_units=32, **common
        )
        model = DirectConcatClassifier(
            ast_mean=common["ast_mean"],
            ast_scale=common["ast_scale"],
            acoustic_train=common["age_train"],
            dropout=common["dropout"],
        )
        with torch.no_grad():
            model.concat_linear.weight[:, :768].copy_(reference.ast_linear.weight)
            model.concat_linear.weight[:, 768:].zero_()
            model.concat_linear.bias.copy_(reference.ast_linear.bias)
            model.batch_norm.load_state_dict(reference.batch_norm.state_dict())
            model.output.load_state_dict(reference.output.state_dict())
        return model
    raise ValueError(pipeline)


def build_plain_vgg(input_dim: int, seed: int, protocol: dict[str, Any]) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(seed))
    config = protocol["training"]["vgg"]
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(int(input_dim),)),
            tf.keras.layers.Dense(128, activation="relu", name="hidden"),
            tf.keras.layers.BatchNormalization(name="batch_norm"),
            tf.keras.layers.Dropout(float(config["dropout"]), seed=int(seed), name="dropout"),
            tf.keras.layers.Dense(3, activation="softmax", name="output"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(
            learning_rate=float(config["learning_rate"]),
            epsilon=float(config["optimizer_epsilon"]),
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    )
    return model


def build_vgg_model(
    pipeline: str, seed: int, protocol: dict[str, Any]
) -> tf.keras.Model:
    reference = build_plain_vgg(128, seed, protocol)
    reference_weights = [np.asarray(value).copy() for value in reference.get_weights()]
    if pipeline == "VGG128_no_f0":
        return reference
    if pipeline != "VGG129_with_f0":
        raise ValueError(pipeline)
    model = build_plain_vgg(129, seed, protocol)
    kernel = np.zeros((129, 128), dtype=reference_weights[0].dtype)
    kernel[:128] = reference_weights[0]
    model.set_weights([kernel, *reference_weights[1:]])
    return model


def torch_state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def keras_state_digest(model: tf.keras.Model) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(model.get_weights()):
        array = np.ascontiguousarray(value)
        digest.update(f"weight_{index:03d}".encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_keras_weights(path: Path, model: tf.keras.Model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            f"weight_{index:03d}": value
            for index, value in enumerate(model.get_weights())
        },
    )


def load_keras_weights(path: Path, model: tf.keras.Model) -> None:
    loaded = np.load(path)
    keys = sorted(loaded.files)
    model.set_weights([loaded[key] for key in keys])


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
        "accuracy": float(accuracy_score(labels, predictions)),
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


def units_to_animals(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cat_id, group in units.groupby("cat_id", sort=True):
        labels = group["true_label"].to_numpy(dtype=np.int64)
        if len(np.unique(labels)) != 1:
            raise RuntimeError(f"IDEA-089 inconsistent unit labels for {cat_id}")
        probabilities = group[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64).mean(axis=0)
        rows.append(
            {
                "cat_id": str(cat_id),
                "true_label": int(labels[0]),
                "unit_count": int(len(group)),
                "prob_kitten": float(probabilities[0]),
                "prob_adult": float(probabilities[1]),
                "prob_senior": float(probabilities[2]),
                "predicted_label": int(probabilities.argmax()),
            }
        )
    return pd.DataFrame(rows).sort_values("cat_id").reset_index(drop=True)


def validate_probability_frames(units: pd.DataFrame, animals: pd.DataFrame) -> None:
    for frame in (units, animals):
        probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("IDEA-089 nonfinite probabilities")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise RuntimeError("IDEA-089 out-of-range probabilities")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0):
            raise RuntimeError("IDEA-089 unnormalized probabilities")
    rebuilt = units_to_animals(units)
    left = animals.sort_values("cat_id").reset_index(drop=True)
    for column in ("cat_id", "true_label", "predicted_label"):
        if not np.array_equal(left[column].to_numpy(), rebuilt[column].to_numpy()):
            raise RuntimeError(f"IDEA-089 unit-to-cat mismatch: {column}")
    if not np.allclose(
        left[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64),
        rebuilt[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64),
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise RuntimeError("IDEA-089 unit-to-cat probability mismatch")


def vgg_unit_frame(
    store: dict[str, np.ndarray], indices: np.ndarray, probabilities: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": store["unit_ids"][indices],
            "cat_id": store["cat_ids"][indices],
            "true_label": store["labels"][indices],
            "prob_kitten": probabilities[:, 0],
            "prob_adult": probabilities[:, 1],
            "prob_senior": probabilities[:, 2],
        }
    ).sort_values("unit_id").reset_index(drop=True)


def predict_vgg(
    model: tf.keras.Model,
    features: np.ndarray,
    store: dict[str, np.ndarray],
    indices: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transformed = ((features[indices] - mean) / scale).astype(np.float32)
    probabilities = model.predict(transformed, batch_size=int(batch_size), verbose=0)
    units = vgg_unit_frame(store, indices, probabilities)
    animals = units_to_animals(units)
    validate_probability_frames(units, animals)
    return animals, units


def predict_ast(
    model: torch.nn.Module,
    store: Any,
    acoustic: np.ndarray,
    indices: np.ndarray,
    cat_batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    animals, calls = idea076.predict_with_perturbation_reset(
        model,
        store,
        acoustic,
        indices,
        int(cat_batch_size),
        device,
        int(seed),
    )
    calls = calls.rename(columns={"call_id": "unit_id"})
    if "call_count" in animals.columns:
        animals = animals.rename(columns={"call_count": "unit_count"})
    validate_probability_frames(calls, animals)
    return animals, calls


def save_preprocessing(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def ast_preprocessing(model: torch.nn.Module) -> dict[str, np.ndarray]:
    values = {
        "ast_mean": model.ast_mean.detach().cpu().numpy(),
        "ast_scale": model.ast_scale.detach().cpu().numpy(),
    }
    for name in ("age_median", "age_mean", "age_scale"):
        if hasattr(model, name):
            values[name] = getattr(model, name).detach().cpu().numpy()
    return values


def training_role_identity(
    cat_ids: np.ndarray, labels: np.ndarray, indices: np.ndarray
) -> dict[str, Any]:
    selected_labels = labels[indices].astype(np.int64)
    cats = sorted(set(cat_ids[indices].astype(str)))
    return {
        "cats": len(cats),
        "units": int(len(indices)),
        "cat_ids_sha256": text_sha256("\n".join(cats)),
        "unit_indices_sha256": hashlib.sha256(
            np.sort(indices.astype("<i8")).tobytes()
        ).hexdigest(),
        "unit_class_counts": np.bincount(selected_labels, minlength=3)
        .astype(int)
        .tolist(),
        "unit_class_weights": class_weights(selected_labels).astype(float).tolist(),
    }


def train_vgg(
    pipeline: str,
    protocol: dict[str, Any],
    store: dict[str, np.ndarray],
    train_indices: np.ndarray,
    seed: int,
    validation_indices: np.ndarray | None,
    fixed_epochs: int | None,
) -> tuple[
    tf.keras.Model,
    dict[str, np.ndarray],
    dict[str, Any],
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    if (validation_indices is None) == (fixed_epochs is None):
        raise ValueError("IDEA-089 VGG fit must be selection xor fixed-epoch")
    config = protocol["training"]["vgg"]
    feature_key = "features128" if pipeline == VGG_PIPELINES[0] else "features129"
    features = store[feature_key]
    mean, scale = standardizer(features[train_indices])
    x_train = ((features[train_indices] - mean) / scale).astype(np.float32)
    y_train = store["labels"][train_indices].astype(np.int64)
    weights = class_weights(y_train)
    set_seed(seed)
    model = build_vgg_model(pipeline, seed, protocol)
    trainable_parameters = int(
        sum(int(np.prod(weight.shape)) for weight in model.trainable_weights)
    )
    total_state_parameters = int(model.count_params())
    if trainable_parameters != EXPECTED_TRAINABLE_PARAMETERS[pipeline]:
        raise RuntimeError("IDEA-089 VGG trainable parameter count changed")
    if total_state_parameters != EXPECTED_TOTAL_STATE_PARAMETERS[pipeline]:
        raise RuntimeError("IDEA-089 VGG parameter count changed")
    initial_digest = keras_state_digest(model)
    training_seed = int(seed) + int(protocol["training"]["post_build_seed_offset"])
    tf.keras.utils.set_random_seed(training_seed)
    maximum_epochs = (
        int(fixed_epochs) if fixed_epochs is not None else int(config["maximum_epochs"])
    )
    if maximum_epochs < 1 or maximum_epochs > int(config["maximum_epochs"]):
        raise RuntimeError("IDEA-089 invalid VGG epoch budget")
    best_loss = float("inf")
    best_epoch = 1
    best_weights = [np.asarray(value).copy() for value in model.get_weights()]
    best_animals: pd.DataFrame | None = None
    best_units: pd.DataFrame | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    batch_size = int(config["batch_size"])
    started = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        order = np.random.default_rng(training_seed + epoch).permutation(len(x_train))
        batch_losses: list[float] = []
        batch_counts: list[int] = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            result = model.train_on_batch(
                x_train[batch],
                y_train[batch],
                sample_weight=weights[y_train[batch]],
                return_dict=True,
            )
            loss = float(result["loss"] if isinstance(result, dict) else result)
            if not math.isfinite(loss):
                raise RuntimeError("IDEA-089 nonfinite VGG training loss")
            batch_losses.append(loss)
            batch_counts.append(len(batch))
        row: dict[str, Any] = {
            "epoch": epoch,
            "training_loss": float(np.average(batch_losses, weights=batch_counts)),
            "unit_order_sha256": hashlib.sha256(
                train_indices[order].astype("<i8").tobytes()
            ).hexdigest(),
        }
        if fixed_epochs is None:
            if validation_indices is None:
                raise RuntimeError("IDEA-089 missing VGG validation indices")
            animals, units = predict_vgg(
                model,
                features,
                store,
                validation_indices,
                mean,
                scale,
                batch_size,
            )
            metrics = metric_bundle(animals)
            row["validation_animal_metrics"] = metrics
            value = float(metrics["cross_entropy"])
            if strict_improvement(
                value, best_loss, float(protocol["training"]["minimum_delta"])
            ):
                best_loss = value
                best_epoch = epoch
                best_weights = [np.asarray(item).copy() for item in model.get_weights()]
                best_animals = animals.copy()
                best_units = units.copy()
                stale = 0
            else:
                stale += 1
        history.append(row)
        print(
            f"IDEA-089 {pipeline} epoch={epoch} train_loss={row['training_loss']:.6f}",
            flush=True,
        )
        if fixed_epochs is None and stale >= int(config["early_stopping_patience"]):
            break
    if fixed_epochs is None:
        if best_animals is None or best_units is None:
            raise RuntimeError("IDEA-089 VGG selection found no best checkpoint")
        model.set_weights(best_weights)
    else:
        best_epoch = int(fixed_epochs)
    audit = {
        "framework": f"TensorFlow {tf.__version__}",
        "best_epoch": int(best_epoch),
        "stopped_epoch": len(history),
        "fixed_epoch_training": fixed_epochs is not None,
        "seed": int(seed),
        "training_seed": int(training_seed),
        "initial_state_sha256": initial_digest,
        "train_role": training_role_identity(
            store["cat_ids"], store["labels"], train_indices
        ),
        "trainable_parameters": trainable_parameters,
        "total_state_parameters": total_state_parameters,
        "history": history,
        "train_seconds": float(time.perf_counter() - started),
    }
    if fixed_epochs is None:
        audit["best_validation_animal_metrics"] = metric_bundle(best_animals)
    return model, {"mean": mean, "scale": scale}, audit, best_animals, best_units


def train_ast(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    acoustic: np.ndarray,
    train_indices: np.ndarray,
    device: torch.device,
    seed: int,
    validation_indices: np.ndarray | None,
    fixed_epochs: int | None,
) -> tuple[
    torch.nn.Module,
    dict[str, np.ndarray],
    dict[str, Any],
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    if (validation_indices is None) == (fixed_epochs is None):
        raise ValueError("IDEA-089 AST fit must be selection xor fixed-epoch")
    config = protocol["training"]["ast"]
    set_seed(seed)
    model = build_ast_model(pipeline, protocol, store, acoustic, train_indices).to(device)
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    if parameters != EXPECTED_TRAINABLE_PARAMETERS[pipeline]:
        raise RuntimeError("IDEA-089 AST parameter count changed")
    initial_digest = torch_state_digest(model)
    training_seed = int(seed) + int(protocol["training"]["post_build_seed_offset"])
    set_seed(training_seed)
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=float(config["learning_rate"]),
        eps=float(config["optimizer_epsilon"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    dataset = idea068.idea051.CatSetDataset(store, train_indices)
    loader = idea068.idea051.build_set_loader(
        dataset, int(config["cat_batch_size"]), True, int(seed)
    )
    call_weights = torch.from_numpy(class_weights(store.labels[train_indices])).to(device)
    maximum_epochs = (
        int(fixed_epochs) if fixed_epochs is not None else int(config["maximum_epochs"])
    )
    if maximum_epochs < 1 or maximum_epochs > int(config["maximum_epochs"]):
        raise RuntimeError("IDEA-089 invalid AST epoch budget")
    best_loss = float("inf")
    best_epoch = 1
    best_state = idea068.idea051.cpu_state_dict(model)
    best_animals: pd.DataFrame | None = None
    best_units: pd.DataFrame | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        train_loss, train_audit = idea068.train_one_epoch(
            model,
            loader,
            optimizer,
            scaler,
            store,
            acoustic,
            call_weights,
            device,
            float(config["gradient_clip"]),
        )
        if not math.isfinite(train_loss):
            raise RuntimeError("IDEA-089 nonfinite AST training loss")
        row: dict[str, Any] = {
            "epoch": epoch,
            "training_call_loss": float(train_loss),
            "train_audit": train_audit,
        }
        if fixed_epochs is None:
            if validation_indices is None:
                raise RuntimeError("IDEA-089 missing AST validation indices")
            animals, units = predict_ast(
                model,
                store,
                acoustic,
                validation_indices,
                int(config["cat_batch_size"]) * 2,
                device,
                seed,
            )
            metrics = metric_bundle(animals)
            row["validation_animal_metrics"] = metrics
            value = float(metrics["cross_entropy"])
            if strict_improvement(
                value, best_loss, float(protocol["training"]["minimum_delta"])
            ):
                best_loss = value
                best_epoch = epoch
                best_state = idea068.idea051.cpu_state_dict(model)
                best_animals = animals.copy()
                best_units = units.copy()
                stale = 0
            else:
                stale += 1
        history.append(row)
        print(
            f"IDEA-089 {pipeline} epoch={epoch} train_loss={train_loss:.6f}",
            flush=True,
        )
        if fixed_epochs is None and stale >= int(config["early_stopping_patience"]):
            break
    if fixed_epochs is None:
        if best_animals is None or best_units is None:
            raise RuntimeError("IDEA-089 AST selection found no best checkpoint")
        model.load_state_dict(best_state)
    else:
        best_epoch = int(fixed_epochs)
    audit = {
        "framework": f"PyTorch {torch.__version__}",
        "best_epoch": int(best_epoch),
        "stopped_epoch": len(history),
        "fixed_epoch_training": fixed_epochs is not None,
        "seed": int(seed),
        "training_seed": int(training_seed),
        "initial_state_sha256": initial_digest,
        "train_role": training_role_identity(
            store.cat_ids, store.labels, train_indices
        ),
        "trainable_parameters": parameters,
        "total_state_parameters": parameters,
        "history": history,
        "train_seconds": float(time.perf_counter() - started),
        "model": model.audit(),
    }
    if fixed_epochs is None:
        audit["best_validation_animal_metrics"] = metric_bundle(best_animals)
    preprocessing = ast_preprocessing(model)
    return model, preprocessing, audit, best_animals, best_units


def selected_pipelines(args: argparse.Namespace) -> list[str]:
    return [args.pipeline] if args.pipeline else list(PIPELINES)


def selected_cells(
    args: argparse.Namespace, protocol: dict[str, Any]
) -> list[tuple[int, int, int]]:
    cells = [
        (int(repeat), int(fold), int(seed))
        for repeat in protocol["data"]["repeats"]
        for fold in protocol["data"]["folds"]
        for seed in protocol["training"]["model_seeds"]
        if args.repeat is None or int(repeat) == int(args.repeat)
        if args.fold is None or int(fold) == int(args.fold)
        if args.base_seed is None or int(seed) == int(args.base_seed)
    ]
    return cells


def authorized_workset(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    stage: str,
) -> tuple[list[tuple[int, int, int]], list[str], int | None]:
    expected_initial = f"initial-six-{stage}"
    expected_full = f"full-{stage}"
    if args.authorization_scope not in {expected_initial, expected_full}:
        raise RuntimeError(
            f"IDEA-089 {stage} requires --authorization-scope "
            f"{expected_initial} or {expected_full}"
        )
    if args.authorization_scope == expected_initial:
        if any(
            value is not None
            for value in (args.pipeline, args.repeat, args.fold, args.base_seed)
        ):
            raise RuntimeError("IDEA-089 initial-six scope forbids fit filters")
        if args.max_fits not in (None, 6):
            raise RuntimeError("IDEA-089 initial-six scope is exactly six fits")
        return [(0, 0, 17)], list(PIPELINES), 6
    return selected_cells(args, protocol), selected_pipelines(args), args.max_fits


def record_authorization(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    run_root: Path,
    stage: str,
    preflight_sha256: str,
    selection_lock_sha256: str | None = None,
) -> None:
    value = {
        "stage": stage,
        "authorization_scope": args.authorization_scope,
        "director_authorized": True,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "preflight_sha256": preflight_sha256,
        "selection_lock_sha256": selection_lock_sha256,
        "filters": {
            "pipeline": args.pipeline,
            "repeat": args.repeat,
            "fold": args.fold,
            "base_seed": args.base_seed,
            "max_fits": args.max_fits,
        },
    }
    path = run_root / "authorizations" / f"{args.authorization_scope}.json"
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError("IDEA-089 authorization record changed")
    else:
        write_json(path, value)


def fit_directory(
    run_root: Path,
    stage: str,
    pipeline: str,
    repeat: int,
    fold: int,
    base_seed: int,
) -> Path:
    return (
        run_root
        / stage
        / pipeline
        / f"repeat_{repeat}"
        / f"fold_{fold}"
        / f"seed_{base_seed}"
    )


def base_fit_identity(
    protocol: dict[str, Any],
    stage: str,
    pipeline: str,
    repeat: int,
    fold: int,
    base_seed: int,
) -> dict[str, Any]:
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "stage": stage,
        "pipeline": pipeline,
        "repeat": int(repeat),
        "fold": int(fold),
        "base_seed": int(base_seed),
        "full_seed": full_seed(base_seed, repeat, fold),
    }


def save_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def save_model(path: Path, pipeline: str, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pipeline in VGG_PIPELINES:
        save_keras_weights(path, model)
    else:
        torch.save(idea068.idea051.cpu_state_dict(model), path)


def reload_model(
    path: Path,
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    acoustic: np.ndarray,
    train_indices: np.ndarray,
    seed: int,
    device: torch.device,
) -> Any:
    set_seed(seed)
    if pipeline in VGG_PIPELINES:
        model = build_vgg_model(pipeline, seed, protocol)
        load_keras_weights(path, model)
        return model
    model = build_ast_model(pipeline, protocol, store, acoustic, train_indices).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    return model


def model_digest(pipeline: str, model: Any) -> str:
    return keras_state_digest(model) if pipeline in VGG_PIPELINES else torch_state_digest(model)


def train_dispatch(
    pipeline: str,
    protocol: dict[str, Any],
    store: Any,
    acoustic: np.ndarray,
    vggish: dict[str, np.ndarray],
    train_indices: np.ndarray,
    device: torch.device,
    seed: int,
    validation_indices: np.ndarray | None,
    fixed_epochs: int | None,
) -> tuple[Any, dict[str, np.ndarray], dict[str, Any], pd.DataFrame | None, pd.DataFrame | None]:
    if pipeline in VGG_PIPELINES:
        return train_vgg(
            pipeline,
            protocol,
            vggish,
            train_indices,
            seed,
            validation_indices,
            fixed_epochs,
        )
    return train_ast(
        pipeline,
        protocol,
        store,
        acoustic,
        train_indices,
        device,
        seed,
        validation_indices,
        fixed_epochs,
    )


def data_indices(
    pipeline: str,
    store: Any,
    vggish: dict[str, np.ndarray],
    roles: pd.DataFrame,
    repeat: int,
    fold: int,
) -> dict[str, np.ndarray]:
    cat_ids = vggish["cat_ids"] if pipeline in VGG_PIPELINES else store.cat_ids
    return indices_by_role(cat_ids, roles, repeat, fold)


def predict_dispatch(
    pipeline: str,
    model: Any,
    protocol: dict[str, Any],
    store: Any,
    acoustic: np.ndarray,
    vggish: dict[str, np.ndarray],
    indices: np.ndarray,
    preprocessing: dict[str, np.ndarray],
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pipeline in VGG_PIPELINES:
        feature_key = "features128" if pipeline == VGG_PIPELINES[0] else "features129"
        return predict_vgg(
            model,
            vggish[feature_key],
            vggish,
            indices,
            preprocessing["mean"],
            preprocessing["scale"],
            int(protocol["training"]["vgg"]["batch_size"]),
        )
    return predict_ast(
        model,
        store,
        acoustic,
        indices,
        int(protocol["training"]["ast"]["cat_batch_size"]) * 2,
        device,
        seed,
    )


def fit_artifact_pairs(stage: str) -> tuple[tuple[str, str], ...]:
    common = (
        ("model_weights", "model_weights_sha256"),
        ("preprocessing", "preprocessing_sha256"),
        ("training_history", "training_history_sha256"),
    )
    if stage == "selection":
        return common + (
            ("validation_unit_predictions", "validation_unit_predictions_sha256"),
            ("validation_cat_predictions", "validation_cat_predictions_sha256"),
        )
    if stage == "outer":
        return common + (
            ("checkpoint_lock", "checkpoint_lock_sha256"),
            ("test_unit_predictions", "test_unit_predictions_sha256"),
            ("test_cat_predictions", "test_cat_predictions_sha256"),
        )
    raise ValueError(stage)


def validate_completed_fit(
    summary_path: Path,
    protocol: dict[str, Any],
    stage: str,
    pipeline: str,
    repeat: int,
    fold: int,
    base_seed: int,
    selection_lock_sha256: str | None = None,
) -> dict[str, Any]:
    value = read_json(summary_path)
    if value.get("status") != "complete":
        raise RuntimeError(f"IDEA-089 incomplete fit: {summary_path}")
    expected = base_fit_identity(
        protocol, stage, pipeline, repeat, fold, base_seed
    )
    for key, item in expected.items():
        if value.get(key) != item:
            raise RuntimeError(f"IDEA-089 resume identity mismatch: {key}")
    if stage == "outer" and value.get("selection_lock_sha256") != selection_lock_sha256:
        raise RuntimeError("IDEA-089 outer selection lock changed")
    for path_key, hash_key in fit_artifact_pairs(stage):
        path = REPO_ROOT / value[path_key]
        if not path.is_file() or sha256(path) != value[hash_key]:
            raise RuntimeError(f"IDEA-089 fit artifact changed: {path_key}")
    if stage == "outer" and int(value.get("test_prediction_calls", -1)) != 1:
        raise RuntimeError("IDEA-089 outer test prediction count changed")
    return value


def require_preflight(
    args: argparse.Namespace, protocol: dict[str, Any], run_root: Path
) -> dict[str, Any]:
    path = run_root / protocol["outputs"]["cpu_preflight"]
    if not path.is_file():
        raise RuntimeError("IDEA-089 CPU preflight is missing")
    if not args.preflight_sha256 or args.preflight_sha256 != sha256(path):
        raise RuntimeError("IDEA-089 exact preflight SHA-256 is required")
    value = read_json(path)
    if value.get("status") != "GO" or value.get("training_started") is not False:
        raise RuntimeError("IDEA-089 CPU preflight is not a no-training GO")
    if value.get("protocol_sha256") != sha256(PROTOCOL_PATH):
        raise RuntimeError("IDEA-089 preflight protocol changed")
    if value.get("runner_sha256") != sha256(Path(__file__).resolve()):
        raise RuntimeError("IDEA-089 preflight runner changed")
    return value


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = resolve_run_root(args.output_subdir)
    store, acoustic, roles, vggish = load_inputs(protocol)
    role_audits = validate_role_bank(store, roles, vggish, protocol)
    repeat, fold, base_seed = 0, 0, 17
    seed = full_seed(base_seed, repeat, fold)
    ast_indices = data_indices(
        "A0_ast_only", store, vggish, roles, repeat, fold
    )["train"]
    probe = ast_indices[: min(32, len(ast_indices))]
    ast_models: dict[str, torch.nn.Module] = {}
    ast_logits: dict[str, np.ndarray] = {}
    trainable_parameter_counts: dict[str, int] = {}
    total_state_parameter_counts: dict[str, int] = {}
    for pipeline in AST_PIPELINES:
        set_seed(seed)
        model = build_ast_model(pipeline, protocol, store, acoustic, ast_indices).eval()
        ast_models[pipeline] = model
        trainable_parameter_counts[pipeline] = int(
            sum(parameter.numel() for parameter in model.parameters())
        )
        total_state_parameter_counts[pipeline] = trainable_parameter_counts[pipeline]
        with torch.no_grad():
            ast_logits[pipeline] = model(
                torch.from_numpy(store.frozen_embeddings[probe]),
                torch.from_numpy(acoustic[probe]),
            ).numpy()
    if any(
        trainable_parameter_counts[name] != EXPECTED_TRAINABLE_PARAMETERS[name]
        for name in AST_PIPELINES
    ):
        raise RuntimeError("IDEA-089 AST preflight parameter mismatch")
    a0 = ast_models["A0_ast_only"]
    d0 = ast_models["D0_direct_concat"]
    if not torch.equal(d0.concat_linear.weight[:, :768], a0.ast_linear.weight):
        raise RuntimeError("IDEA-089 D0 AST columns do not match A0")
    if not torch.equal(d0.concat_linear.bias, a0.ast_linear.bias):
        raise RuntimeError("IDEA-089 D0 bias does not match A0")
    if torch.count_nonzero(d0.concat_linear.weight[:, 768:]).item() != 0:
        raise RuntimeError("IDEA-089 D0 acoustic columns are not zero-initialized")
    for name in ("batch_norm", "output"):
        left = getattr(a0, name).state_dict()
        right = getattr(d0, name).state_dict()
        if set(left) != set(right) or any(
            not torch.equal(left[key], right[key]) for key in left
        ):
            raise RuntimeError(f"IDEA-089 D0 {name} state does not match A0")
    common_keys = set(a0.state_dict())
    for pipeline in ("U1_wide_unbounded_additive", "C1_bounded_wide_additive"):
        state = ast_models[pipeline].state_dict()
        if not common_keys.issubset(state) or any(
            not torch.equal(a0.state_dict()[key], state[key]) for key in common_keys
        ):
            raise RuntimeError(f"IDEA-089 common AST head differs for {pipeline}")
    if torch_state_digest(ast_models["U1_wide_unbounded_additive"]) != torch_state_digest(
        ast_models["C1_bounded_wide_additive"]
    ):
        raise RuntimeError("IDEA-089 U1/C1 complete initial states differ")
    ast_logit_differences = {
        name: float(np.max(np.abs(ast_logits[name] - ast_logits["A0_ast_only"])))
        for name in AST_PIPELINES[1:]
    }
    if any(value > 1.0e-6 for value in ast_logit_differences.values()):
        raise RuntimeError("IDEA-089 AST initial logits exceed paired tolerance")
    set_seed(seed)
    gradient_model = build_ast_model(
        "D0_direct_concat", protocol, store, acoustic, ast_indices
    )
    gradient_model.train()
    logits = gradient_model(
        torch.from_numpy(store.frozen_embeddings[probe]),
        torch.from_numpy(acoustic[probe]),
    )
    torch.nn.functional.cross_entropy(
        logits, torch.from_numpy(store.labels[probe].astype(np.int64))
    ).backward()
    gradient_max = float(
        gradient_model.concat_linear.weight.grad[:, 768:].abs().max().item()
    )
    if not math.isfinite(gradient_max) or gradient_max <= 0.0:
        raise RuntimeError("IDEA-089 D0 acoustic columns receive no gradient")
    residual_projection_gradients: dict[str, float] = {}
    for pipeline in ("U1_wide_unbounded_additive", "C1_bounded_wide_additive"):
        set_seed(seed)
        gradient_residual = build_ast_model(
            pipeline, protocol, store, acoustic, ast_indices
        )
        gradient_residual.train()
        residual_logits = gradient_residual(
            torch.from_numpy(store.frozen_embeddings[probe]),
            torch.from_numpy(acoustic[probe]),
        )
        torch.nn.functional.cross_entropy(
            residual_logits,
            torch.from_numpy(store.labels[probe].astype(np.int64)),
        ).backward()
        value = float(gradient_residual.age_output.weight.grad.abs().max().item())
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(
                f"IDEA-089 {pipeline} zero projection receives no gradient"
            )
        residual_projection_gradients[pipeline] = value

    vgg_indices = data_indices(
        "VGG128_no_f0", store, vggish, roles, repeat, fold
    )["train"]
    vgg_probe = vgg_indices[: min(32, len(vgg_indices))]
    vgg_models: dict[str, tf.keras.Model] = {}
    vgg_logits: dict[str, np.ndarray] = {}
    for pipeline in VGG_PIPELINES:
        set_seed(seed)
        model = build_vgg_model(pipeline, seed, protocol)
        vgg_models[pipeline] = model
        trainable_parameter_counts[pipeline] = int(
            sum(int(np.prod(weight.shape)) for weight in model.trainable_weights)
        )
        total_state_parameter_counts[pipeline] = int(model.count_params())
        key = "features128" if pipeline == VGG_PIPELINES[0] else "features129"
        mean, scale = standardizer(vggish[key][vgg_indices])
        standardized_probe = (
            (vggish[key][vgg_probe] - mean) / scale
        ).astype(np.float32)
        vgg_logits[pipeline] = model(standardized_probe, training=False).numpy()
    if any(
        trainable_parameter_counts[name] != EXPECTED_TRAINABLE_PARAMETERS[name]
        or total_state_parameter_counts[name] != EXPECTED_TOTAL_STATE_PARAMETERS[name]
        for name in VGG_PIPELINES
    ):
        raise RuntimeError("IDEA-089 VGG preflight parameter mismatch")
    vgg_difference = float(
        np.max(np.abs(vgg_logits[VGG_PIPELINES[1]] - vgg_logits[VGG_PIPELINES[0]]))
    )
    if vgg_difference > 1.0e-7:
        raise RuntimeError("IDEA-089 VGG paired initial logits differ")
    vgg129_kernel = vgg_models[VGG_PIPELINES[1]].get_layer("hidden").get_weights()[0]
    if np.count_nonzero(vgg129_kernel[128]) != 0:
        raise RuntimeError("IDEA-089 VGG F0 column is not zero-initialized")
    gradient_vgg = build_vgg_model("VGG129_with_f0", seed, protocol)
    gradient_mean, gradient_scale = standardizer(vggish["features129"][vgg_indices])
    gradient_input = (
        (vggish["features129"][vgg_probe] - gradient_mean) / gradient_scale
    ).astype(np.float32)
    with tf.GradientTape() as tape:
        probabilities = gradient_vgg(
            tf.convert_to_tensor(gradient_input), training=True
        )
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                tf.convert_to_tensor(vggish["labels"][vgg_probe]), probabilities
            )
        )
    vgg_gradients = tape.gradient(loss, gradient_vgg.trainable_weights)
    vgg_f0_gradient_max = float(
        np.max(np.abs(vgg_gradients[0].numpy()[128]))
    )
    if not math.isfinite(vgg_f0_gradient_max) or vgg_f0_gradient_max <= 0.0:
        raise RuntimeError("IDEA-089 VGG F0 column receives no gradient")

    result = {
        "status": "GO",
        "training_started": False,
        "outer_test_accessed": False,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "dependency_hashes_verified": len(_dependency_checks(protocol)),
        "roles": role_audits,
        "trainable_parameters": trainable_parameter_counts,
        "total_state_parameters": total_state_parameter_counts,
        "ast_initial_logit_max_abs_differences_vs_A0": ast_logit_differences,
        "ast_initial_logit_tolerance": 1.0e-6,
        "D0_common_weights_exact": True,
        "D0_acoustic_columns_zero": True,
        "D0_acoustic_gradient_max_abs": gradient_max,
        "A0_U1_C1_common_head_state_exact": True,
        "U1_C1_complete_initial_state_exact": True,
        "U1_C1_zero_projection_gradient_max_abs": residual_projection_gradients,
        "VGG129_minus_VGG128_initial_logit_max_abs_difference": vgg_difference,
        "VGG129_F0_column_zero": True,
        "VGG129_F0_gradient_max_abs": vgg_f0_gradient_max,
        "selection_fits": int(protocol["budget"]["selection_fits"]),
        "outer_fits": int(protocol["budget"]["outer_fits"]),
        "physical_fits": int(protocol["budget"]["physical_fits"]),
        "complete_oof_sets_per_pipeline": 9,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tensorflow": tf.__version__,
            "device": "cpu",
        },
    }
    path = run_root / protocol["outputs"]["cpu_preflight"]
    if path.exists():
        if not args.resume or read_json(path) != result:
            raise RuntimeError("IDEA-089 preflight output differs")
    else:
        write_json(path, result)
    return result


def _prediction_difference(left: pd.DataFrame, right: pd.DataFrame, key: str) -> float:
    first = left.sort_values(key).reset_index(drop=True)
    second = right.sort_values(key).reset_index(drop=True)
    if not np.array_equal(first[key].to_numpy(), second[key].to_numpy()):
        raise RuntimeError("IDEA-089 reload prediction identities changed")
    return float(
        np.max(
            np.abs(
                first[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
                - second[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
            )
        )
    )


def run_selection(args: argparse.Namespace) -> dict[str, Any]:
    if not args.director_authorized:
        raise RuntimeError("IDEA-089 selection requires director authorization")
    if args.max_fits is not None and args.max_fits < 1:
        raise ValueError("--max-fits must be positive")
    configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = resolve_run_root(args.output_subdir)
    require_preflight(args, protocol, run_root)
    cells, pipelines, fit_limit = authorized_workset(args, protocol, "selection")
    record_authorization(
        args,
        protocol,
        run_root,
        "selection",
        str(args.preflight_sha256),
    )
    device = idea068.idea051.reference.historical.idea019.resolve_device(args.device)
    store, acoustic, roles, vggish = load_inputs(protocol)
    completed = 0
    skipped = 0
    stop = False
    for repeat, fold, base_seed in cells:
        for pipeline in pipelines:
            if fit_limit is not None and completed >= fit_limit:
                stop = True
                break
            output_dir = fit_directory(
                run_root, "selection", pipeline, repeat, fold, base_seed
            )
            summary_path = output_dir / "fit_summary.json"
            if output_dir.exists() and not summary_path.exists() and any(output_dir.iterdir()):
                raise RuntimeError(
                    f"IDEA-089 partial selection directory requires audit: {output_dir}"
                )
            if summary_path.exists():
                if not args.resume:
                    raise RuntimeError(f"IDEA-089 selection fit exists: {summary_path}")
                validate_completed_fit(
                    summary_path,
                    protocol,
                    "selection",
                    pipeline,
                    repeat,
                    fold,
                    base_seed,
                )
                skipped += 1
                continue
            indices = data_indices(pipeline, store, vggish, roles, repeat, fold)
            seed = full_seed(base_seed, repeat, fold)
            output_dir.mkdir(parents=True, exist_ok=True)
            model, preprocessing, audit, best_animals, best_units = train_dispatch(
                pipeline,
                protocol,
                store,
                acoustic,
                vggish,
                indices["train"],
                device,
                seed,
                indices["validation"],
                None,
            )
            if best_animals is None or best_units is None:
                raise RuntimeError("IDEA-089 selection predictions are missing")
            weights_path = output_dir / (
                "best_model_weights.npz"
                if pipeline in VGG_PIPELINES
                else "best_model_weights.pt"
            )
            preprocessing_path = output_dir / "training_preprocessing.npz"
            history_path = output_dir / "training_history.json"
            unit_path = output_dir / "validation_unit_predictions.csv"
            cat_path = output_dir / "validation_cat_predictions.csv"
            save_model(weights_path, pipeline, model)
            save_preprocessing(preprocessing_path, preprocessing)
            write_json(history_path, {"history": audit["history"]})
            state_before = model_digest(pipeline, model)
            reloaded = reload_model(
                weights_path,
                pipeline,
                protocol,
                store,
                acoustic,
                indices["train"],
                seed,
                device,
            )
            state_after = model_digest(pipeline, reloaded)
            if state_before != state_after:
                raise RuntimeError("IDEA-089 selection checkpoint state mismatch")
            reload_animals, reload_units = predict_dispatch(
                pipeline,
                reloaded,
                protocol,
                store,
                acoustic,
                vggish,
                indices["validation"],
                preprocessing,
                device,
                seed,
            )
            unit_key = "unit_id"
            max_difference = max(
                _prediction_difference(best_units, reload_units, unit_key),
                _prediction_difference(best_animals, reload_animals, "cat_id"),
            )
            if max_difference > 1.0e-6:
                raise RuntimeError("IDEA-089 selection reload predictions changed")
            save_frame(unit_path, reload_units)
            save_frame(cat_path, reload_animals)
            del audit["history"]
            summary = {
                "status": "complete",
                **base_fit_identity(
                    protocol, "selection", pipeline, repeat, fold, base_seed
                ),
                "outer_test_accessed": False,
                "best_epoch": int(audit["best_epoch"]),
                "stopped_epoch": int(audit["stopped_epoch"]),
                "selection_metric": "unweighted_validation_cat_cross_entropy",
                "minimum_delta": float(protocol["training"]["minimum_delta"]),
                "validation_metrics": metric_bundle(reload_animals),
                "audit": audit,
                "checkpoint_reload_state_match": True,
                "checkpoint_reload_max_probability_difference": max_difference,
                "model_weights": weights_path.relative_to(REPO_ROOT).as_posix(),
                "model_weights_sha256": sha256(weights_path),
                "preprocessing": preprocessing_path.relative_to(REPO_ROOT).as_posix(),
                "preprocessing_sha256": sha256(preprocessing_path),
                "training_history": history_path.relative_to(REPO_ROOT).as_posix(),
                "training_history_sha256": sha256(history_path),
                "validation_unit_predictions": unit_path.relative_to(REPO_ROOT).as_posix(),
                "validation_unit_predictions_sha256": sha256(unit_path),
                "validation_cat_predictions": cat_path.relative_to(REPO_ROOT).as_posix(),
                "validation_cat_predictions_sha256": sha256(cat_path),
                "test_prediction_calls": 0,
            }
            write_json(summary_path, summary)
            completed += 1
            del model, reloaded
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if stop:
            break
    result = {
        "status": "selection_stage_returned",
        "completed_new_fits": completed,
        "validated_resume_fits": skipped,
        "outer_test_accessed": False,
        "filters": {
            "pipeline": args.pipeline,
            "repeat": args.repeat,
            "fold": args.fold,
            "base_seed": args.base_seed,
            "max_fits": args.max_fits,
            "authorization_scope": args.authorization_scope,
        },
    }
    write_json(run_root / "selection_stage_status.json", result)
    return result


def all_fit_coordinates(
    protocol: dict[str, Any]
) -> Iterable[tuple[str, int, int, int]]:
    for repeat in protocol["data"]["repeats"]:
        for fold in protocol["data"]["folds"]:
            for base_seed in protocol["training"]["model_seeds"]:
                for pipeline in PIPELINES:
                    yield pipeline, int(repeat), int(fold), int(base_seed)


def selection_lock_path(run_root: Path, protocol: dict[str, Any]) -> Path:
    return run_root / protocol["outputs"]["selection_lock"]


def run_lock(args: argparse.Namespace) -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = resolve_run_root(args.output_subdir)
    require_preflight(args, protocol, run_root)
    entries = []
    for pipeline, repeat, fold, base_seed in all_fit_coordinates(protocol):
        summary_path = (
            fit_directory(
                run_root, "selection", pipeline, repeat, fold, base_seed
            )
            / "fit_summary.json"
        )
        if not summary_path.is_file():
            raise RuntimeError(f"IDEA-089 missing selection fit: {summary_path}")
        summary = validate_completed_fit(
            summary_path,
            protocol,
            "selection",
            pipeline,
            repeat,
            fold,
            base_seed,
        )
        if summary.get("outer_test_accessed") is not False:
            raise RuntimeError("IDEA-089 selection accessed outer test")
        entries.append(
            {
                "pipeline": pipeline,
                "repeat": repeat,
                "fold": fold,
                "base_seed": base_seed,
                "full_seed": full_seed(base_seed, repeat, fold),
                "best_epoch": int(summary["best_epoch"]),
                "selection_summary": summary_path.relative_to(REPO_ROOT).as_posix(),
                "selection_summary_sha256": sha256(summary_path),
                "selection_model_weights_sha256": summary["model_weights_sha256"],
                "selection_preprocessing_sha256": summary["preprocessing_sha256"],
                "selection_unit_predictions_sha256": summary[
                    "validation_unit_predictions_sha256"
                ],
                "selection_cat_predictions_sha256": summary[
                    "validation_cat_predictions_sha256"
                ],
            }
        )
    value = {
        "status": "complete_locked_before_outer_evaluation",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "selection_fits": len(entries),
        "outer_test_accessed": False,
        "entries": entries,
    }
    path = selection_lock_path(run_root, protocol)
    if path.exists():
        if not args.resume or read_json(path) != value:
            raise RuntimeError("IDEA-089 epoch-selection lock differs")
    else:
        write_json(path, value)
    return {**value, "selection_lock_sha256": sha256(path)}


def require_selection_lock(
    args: argparse.Namespace, protocol: dict[str, Any], run_root: Path
) -> tuple[dict[str, Any], str]:
    path = selection_lock_path(run_root, protocol)
    if not path.is_file():
        raise RuntimeError("IDEA-089 selection lock is missing")
    actual = sha256(path)
    if not args.selection_lock_sha256 or args.selection_lock_sha256 != actual:
        raise RuntimeError("IDEA-089 exact selection-lock SHA-256 is required")
    value = read_json(path)
    if value.get("status") != "complete_locked_before_outer_evaluation":
        raise RuntimeError("IDEA-089 selection lock is incomplete")
    if value.get("selection_fits") != 216 or len(value.get("entries", [])) != 216:
        raise RuntimeError("IDEA-089 selection lock budget changed")
    if value.get("protocol_sha256") != sha256(PROTOCOL_PATH):
        raise RuntimeError("IDEA-089 selection lock protocol changed")
    if value.get("runner_sha256") != sha256(Path(__file__).resolve()):
        raise RuntimeError("IDEA-089 selection lock runner changed")
    return value, actual


def selection_entry_map(lock: dict[str, Any]) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    result = {
        (
            str(row["pipeline"]),
            int(row["repeat"]),
            int(row["fold"]),
            int(row["base_seed"]),
        ): row
        for row in lock["entries"]
    }
    if len(result) != 216:
        raise RuntimeError("IDEA-089 selection lock identities are not unique")
    return result


def run_outer(args: argparse.Namespace) -> dict[str, Any]:
    if not args.director_authorized:
        raise RuntimeError("IDEA-089 outer evaluation requires director authorization")
    if args.max_fits is not None and args.max_fits < 1:
        raise ValueError("--max-fits must be positive")
    configure_determinism()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = resolve_run_root(args.output_subdir)
    require_preflight(args, protocol, run_root)
    selection_lock, lock_sha = require_selection_lock(args, protocol, run_root)
    lock_map = selection_entry_map(selection_lock)
    cells, pipelines, fit_limit = authorized_workset(args, protocol, "outer")
    record_authorization(
        args,
        protocol,
        run_root,
        "outer",
        str(args.preflight_sha256),
        lock_sha,
    )
    device = idea068.idea051.reference.historical.idea019.resolve_device(args.device)
    store, acoustic, roles, vggish = load_inputs(protocol)
    completed = 0
    skipped = 0
    stop = False
    for repeat, fold, base_seed in cells:
        for pipeline in pipelines:
            if fit_limit is not None and completed >= fit_limit:
                stop = True
                break
            output_dir = fit_directory(
                run_root, "outer", pipeline, repeat, fold, base_seed
            )
            summary_path = output_dir / "fit_summary.json"
            if output_dir.exists() and not summary_path.exists() and any(output_dir.iterdir()):
                raise RuntimeError(
                    f"IDEA-089 partial outer directory requires audit and may not rerun: {output_dir}"
                )
            if summary_path.exists():
                if not args.resume:
                    raise RuntimeError(f"IDEA-089 outer fit exists: {summary_path}")
                validate_completed_fit(
                    summary_path,
                    protocol,
                    "outer",
                    pipeline,
                    repeat,
                    fold,
                    base_seed,
                    lock_sha,
                )
                skipped += 1
                continue
            entry = lock_map[(pipeline, repeat, fold, base_seed)]
            refit_epoch = int(entry["best_epoch"])
            indices = data_indices(pipeline, store, vggish, roles, repeat, fold)
            outer_train = np.sort(
                np.concatenate((indices["train"], indices["validation"]))
            ).astype(np.int64)
            seed = full_seed(base_seed, repeat, fold)
            output_dir.mkdir(parents=True, exist_ok=True)
            model, preprocessing, audit, no_animals, no_units = train_dispatch(
                pipeline,
                protocol,
                store,
                acoustic,
                vggish,
                outer_train,
                device,
                seed,
                None,
                refit_epoch,
            )
            if no_animals is not None or no_units is not None:
                raise RuntimeError("IDEA-089 outer training predicted before checkpoint lock")
            weights_path = output_dir / (
                "model_weights.npz" if pipeline in VGG_PIPELINES else "model_weights.pt"
            )
            preprocessing_path = output_dir / "training_preprocessing.npz"
            history_path = output_dir / "training_history.json"
            checkpoint_path = output_dir / "checkpoint_lock.json"
            unit_path = output_dir / "test_unit_predictions.csv"
            cat_path = output_dir / "test_cat_predictions.csv"
            save_model(weights_path, pipeline, model)
            save_preprocessing(preprocessing_path, preprocessing)
            write_json(history_path, {"history": audit["history"]})
            state_before = model_digest(pipeline, model)
            reloaded = reload_model(
                weights_path,
                pipeline,
                protocol,
                store,
                acoustic,
                outer_train,
                seed,
                device,
            )
            state_after = model_digest(pipeline, reloaded)
            if state_before != state_after:
                raise RuntimeError("IDEA-089 outer checkpoint state mismatch")
            checkpoint = {
                **base_fit_identity(
                    protocol, "outer", pipeline, repeat, fold, base_seed
                ),
                "selection_lock_sha256": lock_sha,
                "selection_summary_sha256": entry["selection_summary_sha256"],
                "refit_epoch": refit_epoch,
                "model_state_sha256": state_after,
                "model_weights_sha256": sha256(weights_path),
                "preprocessing_sha256": sha256(preprocessing_path),
                "training_history_sha256": sha256(history_path),
                "locked_before_test_prediction": True,
                "test_prediction_calls_before_lock": 0,
            }
            write_json(checkpoint_path, checkpoint)
            prediction_started = time.perf_counter()
            test_animals, test_units = predict_dispatch(
                pipeline,
                reloaded,
                protocol,
                store,
                acoustic,
                vggish,
                indices["test"],
                preprocessing,
                device,
                seed,
            )
            prediction_seconds = float(time.perf_counter() - prediction_started)
            save_frame(unit_path, test_units)
            save_frame(cat_path, test_animals)
            del audit["history"]
            summary = {
                "status": "complete",
                **base_fit_identity(
                    protocol, "outer", pipeline, repeat, fold, base_seed
                ),
                "selection_lock_sha256": lock_sha,
                "selection_summary": entry["selection_summary"],
                "selection_summary_sha256": entry["selection_summary_sha256"],
                "refit_epoch": refit_epoch,
                "outer_train_role": (
                    training_role_identity(
                        vggish["cat_ids"], vggish["labels"], outer_train
                    )
                    if pipeline in VGG_PIPELINES
                    else training_role_identity(store.cat_ids, store.labels, outer_train)
                ),
                "test_cats": int(test_animals["cat_id"].nunique()),
                "test_units": int(len(test_units)),
                "metrics": metric_bundle(test_animals),
                "audit": audit,
                "prediction_seconds": prediction_seconds,
                "checkpoint_reload_state_match": True,
                "checkpoint_lock": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
                "checkpoint_lock_sha256": sha256(checkpoint_path),
                "model_weights": weights_path.relative_to(REPO_ROOT).as_posix(),
                "model_weights_sha256": sha256(weights_path),
                "preprocessing": preprocessing_path.relative_to(REPO_ROOT).as_posix(),
                "preprocessing_sha256": sha256(preprocessing_path),
                "training_history": history_path.relative_to(REPO_ROOT).as_posix(),
                "training_history_sha256": sha256(history_path),
                "test_unit_predictions": unit_path.relative_to(REPO_ROOT).as_posix(),
                "test_unit_predictions_sha256": sha256(unit_path),
                "test_cat_predictions": cat_path.relative_to(REPO_ROOT).as_posix(),
                "test_cat_predictions_sha256": sha256(cat_path),
                "test_prediction_calls": 1,
                "outer_test_accessed": True,
            }
            write_json(summary_path, summary)
            completed += 1
            del model, reloaded
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if stop:
            break
    result = {
        "status": "outer_stage_returned",
        "completed_new_fits": completed,
        "validated_resume_fits": skipped,
        "selection_lock_sha256": lock_sha,
        "filters": {
            "pipeline": args.pipeline,
            "repeat": args.repeat,
            "fold": args.fold,
            "base_seed": args.base_seed,
            "max_fits": args.max_fits,
            "authorization_scope": args.authorization_scope,
        },
    }
    write_json(run_root / "outer_stage_status.json", result)
    return result


def flatten_metrics(bundle: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy": float(bundle["accuracy"]),
        "macro_f1": float(bundle["macro_f1"]),
        "balanced_accuracy": float(bundle["balanced_accuracy"]),
        "recall_kitten": float(bundle["class_recall"]["kitten"]),
        "recall_adult": float(bundle["class_recall"]["adult"]),
        "recall_senior": float(bundle["class_recall"]["senior"]),
        "cross_entropy": float(bundle["cross_entropy"]),
        "brier": float(bundle["brier"]),
    }


def mean_and_sample_sd(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) < 2:
        raise RuntimeError("IDEA-089 summary needs at least two values")
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
    }


def strict_improvement(value: float, best: float, minimum_delta: float) -> bool:
    return float(value) < float(best) - float(minimum_delta)


def stratified_bootstrap_indices(
    labels: np.ndarray, replicates: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    pieces = []
    for label in range(3):
        positions = np.flatnonzero(labels == label)
        if len(positions) == 0:
            raise RuntimeError("IDEA-089 bootstrap labels are incomplete")
        pieces.append(
            rng.choice(positions, size=(int(replicates), len(positions)), replace=True)
        )
    return np.concatenate(pieces, axis=1).astype(np.int64)


def bootstrap_model_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_indices: np.ndarray,
    chunk_size: int = 250,
) -> dict[str, np.ndarray]:
    if probabilities.shape[1:] != (len(labels), 3):
        raise RuntimeError("IDEA-089 bootstrap probability shape changed")
    replicates = len(sample_indices)
    names = (
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "recall_kitten",
        "recall_adult",
        "recall_senior",
        "cross_entropy",
        "brier",
    )
    result = {name: np.empty(replicates, dtype=np.float64) for name in names}
    targets = np.eye(3, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = sample_indices[start:stop]
        sampled_labels = labels[indices]
        accumulated = {
            name: np.zeros(stop - start, dtype=np.float64) for name in names
        }
        for oof_probabilities in probabilities:
            sampled_probabilities = oof_probabilities[indices]
            predictions = sampled_probabilities.argmax(axis=2)
            recalls = []
            f1_values = []
            for label in range(3):
                true = sampled_labels == label
                predicted = predictions == label
                tp = np.sum(true & predicted, axis=1).astype(np.float64)
                fn = np.sum(true & ~predicted, axis=1).astype(np.float64)
                fp = np.sum(~true & predicted, axis=1).astype(np.float64)
                recalls.append(tp / np.maximum(tp + fn, 1.0))
                denominator = 2.0 * tp + fp + fn
                f1_values.append(
                    np.divide(
                        2.0 * tp,
                        denominator,
                        out=np.zeros_like(tp),
                        where=denominator > 0.0,
                    )
                )
            true_probabilities = np.take_along_axis(
                sampled_probabilities, sampled_labels[..., None], axis=2
            )[..., 0]
            sampled_targets = targets[sampled_labels]
            accumulated["accuracy"] += np.mean(
                predictions == sampled_labels, axis=1
            )
            accumulated["macro_f1"] += np.mean(np.stack(f1_values, axis=1), axis=1)
            accumulated["balanced_accuracy"] += np.mean(
                np.stack(recalls, axis=1), axis=1
            )
            accumulated["recall_kitten"] += recalls[0]
            accumulated["recall_adult"] += recalls[1]
            accumulated["recall_senior"] += recalls[2]
            accumulated["cross_entropy"] += -np.mean(
                np.log(np.clip(true_probabilities, 1.0e-12, 1.0)), axis=1
            )
            accumulated["brier"] += np.mean(
                np.sum((sampled_probabilities - sampled_targets) ** 2, axis=2),
                axis=1,
            )
        divisor = float(len(probabilities))
        for name in names:
            result[name][start:stop] = accumulated[name] / divisor
    return result


def run_aggregate(args: argparse.Namespace) -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = resolve_run_root(args.output_subdir)
    require_preflight(args, protocol, run_root)
    selection_lock, lock_sha = require_selection_lock(args, protocol, run_root)
    del selection_lock
    fit_lookup: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for pipeline, repeat, fold, base_seed in all_fit_coordinates(protocol):
        summary_path = (
            fit_directory(run_root, "outer", pipeline, repeat, fold, base_seed)
            / "fit_summary.json"
        )
        if not summary_path.is_file():
            raise RuntimeError(f"IDEA-089 missing outer fit: {summary_path}")
        fit_lookup[(pipeline, repeat, fold, base_seed)] = validate_completed_fit(
            summary_path,
            protocol,
            "outer",
            pipeline,
            repeat,
            fold,
            base_seed,
            lock_sha,
        )

    oof_frames: dict[str, list[pd.DataFrame]] = {name: [] for name in PIPELINES}
    oof_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in PIPELINES}
    timing: dict[str, dict[str, Any]] = {}
    for pipeline in PIPELINES:
        selection_seconds = []
        outer_seconds = []
        prediction_seconds = []
        refit_epochs = []
        for repeat in protocol["data"]["repeats"]:
            for base_seed in protocol["training"]["model_seeds"]:
                pieces = []
                for fold in protocol["data"]["folds"]:
                    fit = fit_lookup[(pipeline, repeat, fold, base_seed)]
                    unit_frame = pd.read_csv(
                        REPO_ROOT / fit["test_unit_predictions"],
                        dtype={"cat_id": str, "unit_id": str},
                    )
                    cat_frame = pd.read_csv(
                        REPO_ROOT / fit["test_cat_predictions"],
                        dtype={"cat_id": str},
                    )
                    validate_probability_frames(unit_frame, cat_frame)
                    recomputed = flatten_metrics(metric_bundle(cat_frame))
                    stored = flatten_metrics(fit["metrics"])
                    if any(
                        abs(recomputed[name] - stored[name]) > 1.0e-12
                        for name in recomputed
                    ):
                        raise RuntimeError("IDEA-089 stored outer metrics changed")
                    pieces.append(cat_frame)
                    outer_seconds.append(float(fit["audit"]["train_seconds"]))
                    prediction_seconds.append(float(fit["prediction_seconds"]))
                    refit_epochs.append(int(fit["refit_epoch"]))
                    selection = read_json(REPO_ROOT / fit["selection_summary"])
                    selection_seconds.append(float(selection["audit"]["train_seconds"]))
                frame = pd.concat(pieces, ignore_index=True).sort_values("cat_id").reset_index(drop=True)
                if len(frame) != 111 or frame["cat_id"].nunique() != 111:
                    raise RuntimeError("IDEA-089 outer folds do not form 111-cat OOF")
                frame.insert(0, "repeat", int(repeat))
                frame.insert(1, "base_seed", int(base_seed))
                oof_path = (
                    run_root
                    / "oof_predictions"
                    / pipeline
                    / f"repeat_{repeat}_seed_{base_seed}.csv"
                )
                save_frame(oof_path, frame)
                metrics = metric_bundle(frame)
                oof_frames[pipeline].append(frame)
                oof_rows[pipeline].append(
                    {
                        "repeat": int(repeat),
                        "base_seed": int(base_seed),
                        "oof_predictions": oof_path.relative_to(REPO_ROOT).as_posix(),
                        "oof_predictions_sha256": sha256(oof_path),
                        "metrics": metrics,
                    }
                )
        timing[pipeline] = {
            "selection_train_seconds": mean_and_sample_sd(selection_seconds),
            "outer_refit_seconds": mean_and_sample_sd(outer_seconds),
            "test_prediction_seconds": mean_and_sample_sd(prediction_seconds),
            "refit_epoch": mean_and_sample_sd(refit_epochs),
            "scope": "head/fusion training and cached-feature inference only; excludes audio feature extraction and AST precomputation",
        }

    pipeline_results: dict[str, Any] = {}
    for pipeline in PIPELINES:
        flattened = [flatten_metrics(row["metrics"]) for row in oof_rows[pipeline]]
        names = tuple(flattened[0])
        pipeline_results[pipeline] = {
            "complete_oof_sets": 9,
            "oof_metrics": oof_rows[pipeline],
            "summary_over_nine_complete_oof_sets": {
                name: mean_and_sample_sd(row[name] for row in flattened) for name in names
            },
            "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS[pipeline],
            "total_state_parameters": EXPECTED_TOTAL_STATE_PARAMETERS[pipeline],
            "timing": timing[pipeline],
        }

    paired: dict[str, Any] = {}
    for name, (left, right) in COMPARISONS.items():
        deltas = []
        for index, (left_row, right_row) in enumerate(
            zip(oof_rows[left], oof_rows[right], strict=True)
        ):
            if (
                left_row["repeat"] != right_row["repeat"]
                or left_row["base_seed"] != right_row["base_seed"]
            ):
                raise RuntimeError("IDEA-089 paired OOF identities changed")
            left_metrics = flatten_metrics(left_row["metrics"])
            right_metrics = flatten_metrics(right_row["metrics"])
            deltas.append(
                {
                    "repeat": left_row["repeat"],
                    "base_seed": left_row["base_seed"],
                    **{
                        metric: float(left_metrics[metric] - right_metrics[metric])
                        for metric in left_metrics
                    },
                }
            )
        metric_names = [key for key in deltas[0] if key not in {"repeat", "base_seed"}]
        paired[name] = {
            "left": left,
            "right": right,
            "nine_complete_oof_deltas": deltas,
            "summary": {
                metric: mean_and_sample_sd(row[metric] for row in deltas)
                for metric in metric_names
            },
            "macro_f1_signs": {
                "positive": sum(row["macro_f1"] > 0.0 for row in deltas),
                "tied": sum(row["macro_f1"] == 0.0 for row in deltas),
                "negative": sum(row["macro_f1"] < 0.0 for row in deltas),
            },
        }

    reference = oof_frames[PIPELINES[0]][0].sort_values("cat_id").reset_index(drop=True)
    cat_ids = reference["cat_id"].astype(str).to_numpy()
    labels = reference["true_label"].to_numpy(dtype=np.int64)
    probability_arrays: dict[str, np.ndarray] = {}
    for pipeline in PIPELINES:
        arrays = []
        for frame in oof_frames[pipeline]:
            ordered = frame.sort_values("cat_id").reset_index(drop=True)
            if not np.array_equal(ordered["cat_id"].astype(str).to_numpy(), cat_ids):
                raise RuntimeError("IDEA-089 OOF cat order differs across models")
            if not np.array_equal(ordered["true_label"].to_numpy(dtype=np.int64), labels):
                raise RuntimeError("IDEA-089 OOF labels differ across models")
            arrays.append(ordered[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64))
        probability_arrays[pipeline] = np.stack(arrays)

    bootstrap_config = protocol["evaluation"]["bootstrap"]
    sample_indices = stratified_bootstrap_indices(
        labels,
        int(bootstrap_config["replicates"]),
        int(bootstrap_config["seed"]),
    )
    model_bootstrap = {
        pipeline: bootstrap_model_metrics(labels, probability_arrays[pipeline], sample_indices)
        for pipeline in PIPELINES
    }
    for name, (left, right) in COMPARISONS.items():
        intervals = {}
        for metric in model_bootstrap[left]:
            difference = model_bootstrap[left][metric] - model_bootstrap[right][metric]
            intervals[metric] = {
                "lower_2_5_percentile": float(np.quantile(difference, 0.025)),
                "upper_97_5_percentile": float(np.quantile(difference, 0.975)),
            }
        paired[name]["stratified_cat_bootstrap_interval"] = intervals

    result = {
        "schema_version": "1.0",
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "tests_sha256": protocol["dependencies"]["tests_sha256"],
        "selection_lock_sha256": lock_sha,
        "physical_fits": 432,
        "selection_fits": 216,
        "outer_fits": 216,
        "unique_cats": 111,
        "prediction_occurrences_per_pipeline": 999,
        "independent_animal_count": 111,
        "claims_external_confirmation": False,
        "pipelines": pipeline_results,
        "paired_comparisons": paired,
        "bootstrap": {
            "replicates": int(bootstrap_config["replicates"]),
            "seed": int(bootstrap_config["seed"]),
            "stratification": "three age groups",
            "shared_cat_draws_across_all_models_and_nine_oof_sets": True,
            "interpretation": "descriptive interval conditional on these animals, predictions, and prior development history",
        },
        "claim_boundary": "Unified internal validation on the historically reused clean111 cats; not independent animals or external confirmation.",
    }
    aggregate_path = run_root / protocol["outputs"]["aggregate_results"]
    write_json(aggregate_path, result)
    metadata = {
        **result,
        "run_summary": aggregate_path.relative_to(REPO_ROOT).as_posix(),
        "run_summary_sha256": sha256(aggregate_path),
    }
    metadata_path = REPO_ROOT / protocol["outputs"]["metadata"]
    write_json(metadata_path, metadata)
    return metadata


def main() -> None:
    args = parse_args()
    if args.stage == "preflight":
        result = run_preflight(args)
    elif args.stage == "selection":
        result = run_selection(args)
    elif args.stage == "lock":
        result = run_lock(args)
    elif args.stage == "outer":
        result = run_outer(args)
    else:
        result = run_aggregate(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
