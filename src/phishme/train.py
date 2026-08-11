from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

from .features import FEATURE_VERSION, vectorize


@dataclass(frozen=True)
class TrainConfig:
    alpha: float = 1e-4
    epochs: int = 3
    batch_size: int = 2048
    seed: int = 42

    def __post_init__(self) -> None:
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real) or self.alpha <= 0:
            raise ValueError("alpha must be positive")
        _require_positive_integer("epochs", self.epochs)
        _require_positive_integer("batch_size", self.batch_size)
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")


def fit_incremental(frame, config: TrainConfig, include_dom) -> SGDClassifier:
    _validate_training_frame(frame)
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(config.alpha),
        random_state=int(config.seed),
    )
    rng = np.random.default_rng(int(config.seed))
    first = True
    classes = np.array([0, 1], dtype=np.int8)
    for _ in range(int(config.epochs)):
        order = rng.permutation(len(frame))
        shuffled = frame.iloc[order]
        for batch in _batches(shuffled, int(config.batch_size)):
            labels = batch["label"].to_numpy(dtype=np.int8)
            kwargs = {"classes": classes} if first else {}
            model.partial_fit(vectorize(batch, include_dom=include_dom), labels, **kwargs)
            first = False
    return model


def predict_scores(model, frame, include_dom, batch_size=4096) -> np.ndarray:
    _require_positive_integer("batch_size", batch_size)
    chunks = [
        model.predict_proba(vectorize(batch, include_dom=include_dom))[:, 1]
        for batch in _batches(frame, int(batch_size))
    ]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=float)


def save_checkpoint(model, metadata, path) -> Path:
    checkpoint_path = Path(path)
    checkpoint_metadata = _checkpoint_metadata(metadata)
    payload = {"model": model, "metadata": checkpoint_metadata}
    temporary_path = _temporary_checkpoint_path(checkpoint_path)

    try:
        joblib.dump(payload, temporary_path)
        _fsync_file(temporary_path)
        os.replace(temporary_path, checkpoint_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return checkpoint_path


def load_checkpoint(path, expected_dataset_revision):
    if expected_dataset_revision is None:
        raise ValueError("expected dataset revision is required")

    payload = joblib.load(Path(path))
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    if "model" not in payload or "metadata" not in payload:
        raise ValueError("checkpoint must contain model and metadata")

    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    metadata = dict(metadata)

    feature_version = metadata.get("feature_version")
    if feature_version != FEATURE_VERSION:
        raise ValueError(
            f"checkpoint feature version mismatch: expected {FEATURE_VERSION!r}, "
            f"found {feature_version!r}"
        )

    dataset_revision = metadata.get("dataset_revision")
    if dataset_revision != expected_dataset_revision:
        raise ValueError(
            "checkpoint dataset revision mismatch: "
            f"expected {expected_dataset_revision!r}, found {dataset_revision!r}"
        )
    return payload["model"], metadata


def atomic_checkpoint(model, metadata, path) -> Path:
    return save_checkpoint(model, metadata, path)


def _batches(frame, size):
    for start in range(0, len(frame), int(size)):
        yield frame.iloc[start : start + int(size)]


def _validate_training_frame(frame) -> None:
    if len(frame) == 0:
        raise ValueError("training frame must not be empty")
    if "label" not in frame:
        raise ValueError("training frame must include a label column")
    labels = frame["label"].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("training labels must contain only 0 and 1")


def _checkpoint_metadata(metadata) -> dict:
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    checkpoint_metadata = dict(metadata)
    dataset_revision = checkpoint_metadata.get("dataset_revision")
    if dataset_revision is None:
        raise ValueError("checkpoint metadata must include dataset_revision")
    feature_version = checkpoint_metadata.get("feature_version", FEATURE_VERSION)
    if feature_version != FEATURE_VERSION:
        raise ValueError(
            f"checkpoint feature version mismatch: expected {FEATURE_VERSION!r}, "
            f"found {feature_version!r}"
        )
    checkpoint_metadata["feature_version"] = FEATURE_VERSION
    return checkpoint_metadata


def _temporary_checkpoint_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDWR)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_positive_integer(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
