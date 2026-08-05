from dataclasses import FrozenInstanceError
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from phishme.features import FEATURE_VERSION
from phishme.train import (
    TrainConfig,
    fit_incremental,
    load_checkpoint,
    predict_scores,
    save_checkpoint,
)


def _tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"url": "https://safe.example/home", "title": "welcome", "label": 0},
            {"url": "https://safe.example/about", "title": "about", "label": 0},
            {"url": "http://steal.example/login", "title": "verify password", "label": 1},
            {"url": "http://steal.example/account", "title": "urgent login", "label": 1},
        ]
        * 8
    )


def test_train_config_is_frozen_and_validated():
    config = TrainConfig()

    with pytest.raises(FrozenInstanceError):
        config.alpha = 0.5

    with pytest.raises(ValueError, match="alpha"):
        TrainConfig(alpha=0.0)
    with pytest.raises(ValueError, match="epochs"):
        TrainConfig(epochs=0)
    with pytest.raises(ValueError, match="batch_size"):
        TrainConfig(batch_size=0)


def test_incremental_fit_rejects_empty_input():
    empty = pd.DataFrame(columns=["url", "title", "label"])

    with pytest.raises(ValueError, match="must not be empty"):
        fit_incremental(empty, TrainConfig(), include_dom=False)


def test_incremental_model_learns_tiny_signal():
    frame = _tiny_frame()

    model = fit_incremental(
        frame, TrainConfig(alpha=1e-4, epochs=5, batch_size=8, seed=42), include_dom=False
    )
    scores = predict_scores(model, frame, include_dom=False, batch_size=8)

    assert scores[frame.label.to_numpy() == 1].mean() > scores[frame.label.to_numpy() == 0].mean()


def test_incremental_training_is_deterministic():
    frame = _tiny_frame()
    config = TrainConfig(alpha=1e-4, epochs=4, batch_size=5, seed=7)

    left = fit_incremental(frame, config, include_dom=False)
    right = fit_incremental(frame, config, include_dom=False)

    np.testing.assert_allclose(
        predict_scores(left, frame, include_dom=False, batch_size=3),
        predict_scores(right, frame, include_dom=False, batch_size=3),
    )


def test_predict_scores_handles_empty_input_after_fit():
    frame = _tiny_frame()
    model = fit_incremental(frame, TrainConfig(epochs=1, batch_size=4), include_dom=False)
    empty = pd.DataFrame(columns=["url", "title", "label"])

    scores = predict_scores(model, empty, include_dom=False, batch_size=2)

    assert scores.shape == (0,)


def test_checkpoint_round_trip_and_rejects_mismatches(tmp_path: Path):
    frame = _tiny_frame()
    model = fit_incremental(frame, TrainConfig(epochs=1, batch_size=4), include_dom=False)
    checkpoint = tmp_path / "model.joblib"
    metadata = {"dataset_revision": "rev-1", "run": "tiny"}

    saved_path = save_checkpoint(model, metadata, checkpoint)
    loaded_model, loaded_metadata = load_checkpoint(checkpoint, expected_dataset_revision="rev-1")

    assert saved_path == checkpoint
    assert loaded_metadata == {**metadata, "feature_version": FEATURE_VERSION}
    np.testing.assert_allclose(
        predict_scores(model, frame, include_dom=False, batch_size=3),
        predict_scores(loaded_model, frame, include_dom=False, batch_size=3),
    )
    assert not checkpoint.with_suffix(checkpoint.suffix + ".tmp").exists()

    with pytest.raises(ValueError, match="dataset revision"):
        load_checkpoint(checkpoint, expected_dataset_revision="rev-2")


def test_checkpoint_rejects_missing_and_mismatched_feature_versions(tmp_path: Path):
    frame = _tiny_frame()
    model = fit_incremental(frame, TrainConfig(epochs=1, batch_size=4), include_dom=False)
    missing = tmp_path / "missing-feature.joblib"
    mismatch = tmp_path / "mismatched-feature.joblib"

    save_checkpoint(model, {"dataset_revision": "rev-1"}, missing)
    payload = joblib.load(missing)
    payload["metadata"].pop("feature_version")
    joblib.dump(payload, missing)

    save_checkpoint(model, {"dataset_revision": "rev-1"}, mismatch)
    payload = joblib.load(mismatch)
    payload["metadata"]["feature_version"] = "wrong"
    joblib.dump(payload, mismatch)

    with pytest.raises(ValueError, match="feature version"):
        load_checkpoint(missing, expected_dataset_revision="rev-1")
    with pytest.raises(ValueError, match="feature version"):
        load_checkpoint(mismatch, expected_dataset_revision="rev-1")
