import numpy as np
import pandas as pd
import pytest
from phishme.models import TreeConfig, fit_tree, predict_tree_scores


def _synthetic_frame(n=120):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        phish = i % 2 == 1
        rows.append({
            "url": f"http{'s' if not phish else ''}://d{i % 20}.example/p{i}",
            "title": "verify password" if phish else "welcome",
            "label": 1 if phish else 0,
        })
    return pd.DataFrame(rows)


def test_tree_requires_lightgbm_optional():
    lightgbm = pytest.importorskip("lightgbm")  # skip when phishme[tree] not installed
    assert lightgbm is not None


def test_fit_tree_predicts_probabilities():
    pytest.importorskip("lightgbm")
    frame = _synthetic_frame()
    config = TreeConfig(n_estimators=10, seed=0)  # small for tests
    model = fit_tree(frame, config, include_dom=False)
    scores = predict_tree_scores(model, frame, include_dom=False)
    assert scores.shape == (len(frame),)
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    assert scores[frame["label"].to_numpy() == 1].mean() > scores[frame["label"].to_numpy() == 0].mean()


def test_tree_config_validates():
    with pytest.raises(ValueError):
        TreeConfig(n_estimators=0)
    with pytest.raises(ValueError):
        TreeConfig(learning_rate=0.0)
