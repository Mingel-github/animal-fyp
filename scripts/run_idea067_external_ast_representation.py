"""Prepare and evaluate IDEA-067 external frozen-AST representation probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import ASTFeatureExtractor, ASTModel


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import extract_ast_embeddings as ast_extract  # noqa: E402


PROTOCOL_PATH = (
    REPO_ROOT
    / "configs"
    / "protocol"
    / "idea067_external_ast_representation_v1.json"
)
RUNS_ROOT = REPO_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare-dog", "extract", "evaluate"), required=True)
    parser.add_argument("--dataset", choices=("catmeows", "canine"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output-subdir", default="idea067_external_ast_representation_v1"
    )
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


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "idea067-external-ast-representation-v1":
        raise RuntimeError("Unexpected IDEA-067 protocol")
    if protocol.get("status") != "locked_before_embedding_and_evaluation":
        raise RuntimeError("IDEA-067 protocol is not locked")
    dependencies = protocol["dependencies"]
    checks = {
        REPO_ROOT / dependencies["idea_path"]: dependencies["idea_sha256"],
        REPO_ROOT / dependencies["locked_ast_protocol_path"]: dependencies[
            "locked_ast_protocol_sha256"
        ],
        REPO_ROOT / dependencies["extraction_helper_path"]: dependencies[
            "extraction_helper_sha256"
        ],
        Path(__file__).resolve(): dependencies["runner_sha256"],
    }
    for path, expected in checks.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"IDEA-067 dependency checksum mismatch: {path}")
    for dataset in protocol["datasets"].values():
        source_path = dataset.get("archive_path") or dataset.get("metadata_path")
        source_hash = dataset.get("archive_sha256") or dataset.get("metadata_sha256")
        path = REPO_ROOT / source_path
        if not path.is_file() or sha256(path) != source_hash:
            raise RuntimeError(f"IDEA-067 dataset source checksum mismatch: {path}")


def catmeows_manifest(spec: dict[str, Any]) -> pd.DataFrame:
    root = REPO_ROOT / spec["audio_root"]
    context = {"B": "brushing", "F": "food_waiting", "I": "isolation"}
    rows = []
    for path in sorted(root.rglob("*.wav")):
        parts = path.stem.split("_")
        if len(parts) != 6 or parts[0] not in context:
            raise RuntimeError(f"Unexpected CatMeows filename: {path.name}")
        rows.append(
            {
                "recording_id": path.stem,
                "animal_id": parts[1],
                "label_name": context[parts[0]],
                "audio_path": path.relative_to(REPO_ROOT).as_posix(),
                "breed": parts[2],
                "sex": parts[3],
                "owner_id": parts[4],
                "session": parts[5][0],
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != int(spec["expected_recordings"]):
        raise RuntimeError("CatMeows recording count changed")
    if frame["animal_id"].nunique() != int(spec["expected_animals"]):
        raise RuntimeError("CatMeows animal count changed")
    return frame


def evenly_spaced_sample(group: pd.DataFrame, cap: int) -> pd.DataFrame:
    ordered = group.sort_values("barkunit_audio").reset_index(drop=True)
    count = min(cap, len(ordered))
    positions = np.unique(
        np.rint(np.linspace(0, len(ordered) - 1, count)).astype(np.int64)
    )
    if len(positions) != count:
        raise RuntimeError("Evenly spaced sampling produced duplicate positions")
    return ordered.iloc[positions].copy()


def canine_sample_manifest(spec: dict[str, Any]) -> pd.DataFrame:
    metadata = pd.read_csv(REPO_ROOT / spec["metadata_path"], sep="\t")
    parts = [
        evenly_spaced_sample(group, int(spec["cap_per_dog_age_group"]))
        for _, group in metadata.groupby(["dog_id", "age_group"], sort=True)
    ]
    sampled = pd.concat(parts, ignore_index=True).sort_values(
        ["dog_id", "age_group", "barkunit_audio"]
    )
    root = REPO_ROOT / spec["sample_audio_root"]
    frame = pd.DataFrame(
        {
            "recording_id": sampled["bark_unit_id"].astype(str),
            "animal_id": sampled["dog_id"].astype(str),
            "label_name": sampled["age_group"].astype(str),
            "audio_path": sampled["barkunit_audio"].map(
                lambda value: (root / str(value)).relative_to(REPO_ROOT).as_posix()
            ),
            "breed": sampled["breed"].astype(str),
            "age_months": sampled["age_months"].astype(float),
            "source_member": sampled["barkunit_audio"].astype(str),
        }
    )
    if len(frame) != int(spec["expected_sample_recordings"]):
        raise RuntimeError("Canine sample count changed")
    if frame["animal_id"].nunique() != int(spec["expected_animals"]):
        raise RuntimeError("Canine sample animal count changed")
    return frame.reset_index(drop=True)


def dog_shard_name(animal_id: str) -> str:
    number = int(animal_id.split("_")[-1])
    starts = ((1, 25), (26, 50), (51, 75), (76, 100), (101, 125))
    for start, stop in starts:
        if start <= number <= stop:
            return f"audio_barkunits_dogs_{start:06d}_{stop:06d}.tar"
    raise ValueError(animal_id)


def prepare_dog_audio(protocol: dict[str, Any], run_root: Path) -> dict[str, Any]:
    spec = protocol["datasets"]["canine"]
    manifest = canine_sample_manifest(spec)
    sample_root = REPO_ROOT / spec["sample_audio_root"]
    shard_root = REPO_ROOT / spec["shard_root"]
    extracted = 0
    missing = 0
    for shard_name, group in manifest.groupby(
        manifest["animal_id"].map(dog_shard_name), sort=True
    ):
        shard_path = shard_root / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        with tarfile.open(shard_path, "r") as archive:
            for row in group.itertuples(index=False):
                destination = REPO_ROOT / row.audio_path
                if destination.is_file():
                    continue
                try:
                    member = archive.getmember(row.source_member)
                except KeyError:
                    missing += 1
                    continue
                source = archive.extractfile(member)
                if source is None:
                    missing += 1
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                extracted += 1
    if missing:
        raise RuntimeError(f"Canine sample is missing {missing} tar members")
    manifest_path = run_root / "canine" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    present = sum((REPO_ROOT / value).is_file() for value in manifest["audio_path"])
    if present != len(manifest):
        raise RuntimeError("Canine sample extraction is incomplete")
    summary = {
        "status": "complete",
        "sample_recordings": len(manifest),
        "animals": int(manifest["animal_id"].nunique()),
        "newly_extracted": extracted,
        "manifest_path": manifest_path.relative_to(REPO_ROOT).as_posix(),
        "manifest_sha256": sha256(manifest_path),
    }
    write_json(run_root / "canine" / "prepare_summary.json", summary)
    return summary


def dataset_manifest(
    dataset: str, protocol: dict[str, Any], run_root: Path
) -> pd.DataFrame:
    spec = protocol["datasets"][dataset]
    if dataset == "catmeows":
        frame = catmeows_manifest(spec)
        path = run_root / dataset / "manifest.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return frame
    path = run_root / dataset / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError("Run --stage prepare-dog before canine extraction")
    return pd.read_csv(path, dtype={"animal_id": str})


def extract_embeddings(
    dataset: str,
    protocol: dict[str, Any],
    run_root: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    manifest = dataset_manifest(dataset, protocol, run_root)
    classes = list(protocol["datasets"][dataset]["classes"])
    label_to_id = {label: index for index, label in enumerate(classes)}
    if set(manifest["label_name"]) != set(classes):
        raise RuntimeError(f"{dataset} labels differ from the locked classes")
    locked = read_json(REPO_ROOT / protocol["dependencies"]["locked_ast_protocol_path"])
    ast_config = locked["ast"]
    representation = protocol["representation"]
    extractor = ASTFeatureExtractor.from_pretrained(
        representation["checkpoint"],
        revision=representation["revision"],
        cache_dir=ast_extract.HF_CACHE,
    )
    extractor.max_length = int(ast_config["max_length_frames"])
    extractor.num_mel_bins = int(ast_config["num_mel_bins"])
    model = ASTModel.from_pretrained(
        representation["checkpoint"],
        revision=representation["revision"],
        cache_dir=ast_extract.HF_CACHE,
        use_safetensors=True,
    )
    geometry = ast_extract.adapt_geometry(
        model,
        int(ast_config["max_length_frames"]),
        int(ast_config["variants"]["ast_standard"]["frequency_stride"]),
        int(ast_config["variants"]["ast_standard"]["time_stride"]),
    )
    model.to(device).eval()
    recording_sums = np.zeros((len(manifest), 768), dtype=np.float64)
    recording_counts = np.zeros(len(manifest), dtype=np.int64)
    buffer_features: list[np.ndarray] = []
    buffer_recordings: list[int] = []

    def flush() -> None:
        if not buffer_features:
            return
        values = np.stack(buffer_features).astype(np.float32)
        with torch.inference_mode():
            output = model(input_values=torch.from_numpy(values).to(device)).pooler_output
        vectors = output.float().cpu().numpy()
        np.add.at(recording_sums, np.asarray(buffer_recordings), vectors)
        np.add.at(recording_counts, np.asarray(buffer_recordings), 1)
        buffer_features.clear()
        buffer_recordings.clear()

    started = time.perf_counter()
    for index, row in enumerate(manifest.itertuples(index=False)):
        waveform = ast_extract.load_audio(
            REPO_ROOT / row.audio_path, int(ast_config["sample_rate_hz"])
        )
        segments = ast_extract.segment_waveform(
            waveform,
            round(int(ast_config["sample_rate_hz"]) * float(representation["segment_seconds"])),
            round(
                int(ast_config["sample_rate_hz"])
                * float(representation["segment_hop_seconds"])
            ),
        )
        features = extractor(
            segments,
            sampling_rate=int(ast_config["sample_rate_hz"]),
            return_tensors="np",
        )["input_values"].astype(np.float32)
        for feature in features:
            buffer_features.append(feature)
            buffer_recordings.append(index)
            if len(buffer_features) >= batch_size:
                flush()
        if (index + 1) % 100 == 0 or index + 1 == len(manifest):
            print(f"{dataset}: prepared {index + 1}/{len(manifest)} recordings", flush=True)
    flush()
    if np.any(recording_counts == 0):
        raise RuntimeError("At least one external recording has no AST segment")
    embeddings = (recording_sums / recording_counts[:, None]).astype(np.float32)
    output_path = run_root / dataset / "ast_standard_recording_embeddings.npz"
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        recording_ids=np.asarray(manifest["recording_id"].astype(str).tolist(), dtype="U"),
        animal_ids=np.asarray(manifest["animal_id"].astype(str).tolist(), dtype="U"),
        labels=manifest["label_name"].map(label_to_id).to_numpy(dtype=np.int8),
        label_names=np.asarray(classes),
        source_paths=np.asarray(manifest["audio_path"].astype(str).tolist(), dtype="U"),
        segment_counts=recording_counts.astype(np.int16),
    )
    summary = {
        "status": "complete",
        "dataset": dataset,
        "recordings": len(manifest),
        "animals": int(manifest["animal_id"].nunique()),
        "segments": int(recording_counts.sum()),
        "segment_count_range": [int(recording_counts.min()), int(recording_counts.max())],
        "embedding_seconds": float(time.perf_counter() - started),
        "device": str(device),
        "geometry": geometry,
        "output_path": output_path.relative_to(REPO_ROOT).as_posix(),
        "output_sha256": sha256(output_path),
    }
    write_json(run_root / dataset / "embedding_summary.json", summary)
    return summary


def metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    animal_ids: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    per_class = recall_score(
        labels, predictions, labels=np.arange(len(class_names)), average=None, zero_division=0
    )
    animal_accuracy = pd.DataFrame(
        {"animal_id": animal_ids, "correct": predictions == labels}
    ).groupby("animal_id")["correct"].mean()
    return {
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=np.arange(len(class_names)),
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "plain_accuracy": float(accuracy_score(labels, predictions)),
        "mean_per_animal_call_accuracy": float(animal_accuracy.mean()),
        "per_class_recall": {
            name: float(per_class[index]) for index, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=np.arange(len(class_names))
        ).tolist(),
        "recordings": int(len(labels)),
        "animals": int(len(np.unique(animal_ids))),
    }


def evaluate(dataset: str, protocol: dict[str, Any], run_root: Path) -> dict[str, Any]:
    embedding_path = run_root / dataset / "ast_standard_recording_embeddings.npz"
    loaded = np.load(embedding_path)
    x = loaded["embeddings"].astype(np.float32)
    labels = loaded["labels"].astype(np.int64)
    animal_ids = loaded["animal_ids"].astype(str)
    recording_ids = loaded["recording_ids"].astype(str)
    class_names = loaded["label_names"].astype(str).tolist()
    settings = protocol["evaluation"]
    repeat_results = []
    prediction_rows = []
    for repeat, seed in enumerate(settings["repeat_seeds"]):
        splitter = StratifiedGroupKFold(
            n_splits=int(settings["folds"]), shuffle=True, random_state=int(seed)
        )
        ast_probabilities = np.zeros((len(labels), len(class_names)), dtype=np.float64)
        dummy_probabilities = np.zeros_like(ast_probabilities)
        fold_assignments = np.full(len(labels), -1, dtype=np.int64)
        for fold, (train, test) in enumerate(
            splitter.split(x, labels, groups=animal_ids)
        ):
            if set(animal_ids[train]).intersection(animal_ids[test]):
                raise RuntimeError("Animal leakage in IDEA-067 split")
            if len(np.unique(labels[train])) != len(class_names):
                raise RuntimeError("A training fold is missing a class")
            scaler = StandardScaler().fit(x[train])
            classifier = LogisticRegression(
                C=float(settings["C"]),
                class_weight="balanced",
                solver=str(settings["solver"]),
                max_iter=int(settings["maximum_iterations"]),
                multi_class="auto",
                random_state=int(seed),
            )
            classifier.fit(scaler.transform(x[train]), labels[train])
            predicted = classifier.predict_proba(scaler.transform(x[test]))
            ast_probabilities[test[:, None], classifier.classes_[None, :]] = predicted
            priors = np.bincount(labels[train], minlength=len(class_names)).astype(float)
            priors /= priors.sum()
            dummy_probabilities[test] = priors
            fold_assignments[test] = fold
        if np.any(fold_assignments < 0):
            raise RuntimeError("Incomplete IDEA-067 OOF assignment")
        ast_metrics = metrics(labels, ast_probabilities, animal_ids, class_names)
        dummy_metrics = metrics(labels, dummy_probabilities, animal_ids, class_names)
        repeat_results.append(
            {
                "repeat": repeat,
                "seed": int(seed),
                "ast": ast_metrics,
                "dummy": dummy_metrics,
                "delta_macro_f1": ast_metrics["macro_f1"]
                - dummy_metrics["macro_f1"],
            }
        )
        for index in range(len(labels)):
            prediction_rows.append(
                {
                    "repeat": repeat,
                    "seed": int(seed),
                    "fold": int(fold_assignments[index]),
                    "recording_id": recording_ids[index],
                    "animal_id": animal_ids[index],
                    "true_label": int(labels[index]),
                    **{
                        f"ast_prob_{name}": float(ast_probabilities[index, class_index])
                        for class_index, name in enumerate(class_names)
                    },
                    **{
                        f"dummy_prob_{name}": float(
                            dummy_probabilities[index, class_index]
                        )
                        for class_index, name in enumerate(class_names)
                    },
                }
            )
    predictions_path = run_root / dataset / "grouped_oof_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    ast_values = [row["ast"]["macro_f1"] for row in repeat_results]
    dummy_values = [row["dummy"]["macro_f1"] for row in repeat_results]
    summary = {
        "status": "complete",
        "dataset": dataset,
        "independence_unit": protocol["datasets"][dataset]["group"],
        "repeat_results": repeat_results,
        "macro_f1": {
            "ast_mean": float(np.mean(ast_values)),
            "ast_std": float(np.std(ast_values, ddof=1)),
            "dummy_mean": float(np.mean(dummy_values)),
            "dummy_std": float(np.std(dummy_values, ddof=1)),
            "mean_delta": float(np.mean(np.asarray(ast_values) - np.asarray(dummy_values))),
            "ast_above_dummy_all_repeats": bool(
                all(ast > dummy for ast, dummy in zip(ast_values, dummy_values))
            ),
        },
        "predictions_path": predictions_path.relative_to(REPO_ROOT).as_posix(),
        "predictions_sha256": sha256(predictions_path),
    }
    write_json(run_root / dataset / "evaluation_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_PATH)
    verify_protocol(protocol)
    run_root = (RUNS_ROOT / args.output_subdir).resolve()
    if RUNS_ROOT.resolve() not in run_root.parents:
        raise ValueError("--output-subdir must stay below runs")
    run_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare-dog":
        if args.dataset != "canine":
            raise ValueError("prepare-dog requires --dataset canine")
        result = prepare_dog_audio(protocol, run_root)
    elif args.stage == "extract":
        device = ast_extract.resolve_device(args.device)
        result = extract_embeddings(
            args.dataset, protocol, run_root, device, args.batch_size
        )
    else:
        result = evaluate(args.dataset, protocol, run_root)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
