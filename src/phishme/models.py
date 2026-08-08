from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from phishme.features import vectorize
from phishme.train import _batches, _require_positive_integer


@dataclass(frozen=True)
class TreeConfig:
    n_estimators: int = 300
    num_leaves: int = 31
    learning_rate: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        _require_positive_integer("n_estimators", self.n_estimators)
        _require_positive_integer("num_leaves", self.num_leaves)
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, Real) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")


def _require_lightgbm():
    try:
        import lightgbm
    except ImportError as exc:  # pragma: no cover - depends on optional tree extra
        raise RuntimeError("optional dependency 'lightgbm' is required; install phishme[tree]") from exc
    return lightgbm


def fit_tree(frame, config: TreeConfig, include_dom: bool):
    lightgbm = _require_lightgbm()
    if len(frame) == 0:
        raise ValueError("training frame must not be empty")
    if not np.isin(frame["label"].to_numpy(), [0, 1]).all():
        raise ValueError("training labels must contain only 0 and 1")
    model = lightgbm.LGBMClassifier(
        n_estimators=int(config.n_estimators),
        num_leaves=int(config.num_leaves),
        learning_rate=float(config.learning_rate),
        random_state=int(config.seed),
        verbose=-1,
    )
    model.fit(vectorize(frame, include_dom=include_dom), frame["label"].to_numpy(dtype=np.int8))
    return model


def predict_tree_scores(model, frame, include_dom, batch_size=4096) -> np.ndarray:
    _require_positive_integer("batch_size", batch_size)
    chunks = [
        model.predict_proba(vectorize(batch, include_dom=include_dom))[:, 1]
        for batch in _batches(frame, int(batch_size))
    ]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=float)


@dataclass(frozen=True)
class HybridConfig:
    meta_c: float = 1.0
    oof_folds: int = 3
    seed: int = 42

    def __post_init__(self) -> None:
        if isinstance(self.meta_c, bool) or not isinstance(self.meta_c, Real) or self.meta_c <= 0:
            raise ValueError("meta_c must be positive")
        _require_positive_integer("oof_folds", self.oof_folds)
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")


def _fit_linear(frame, config, include_dom):
    """Thin wrapper around fit_incremental for hybrid stacking."""
    from phishme.train import fit_incremental

    return fit_incremental(frame, config, include_dom)


def _linear_scores(model, frame, include_dom, batch_size=4096):
    """Thin wrapper around predict_scores for hybrid stacking."""
    from phishme.train import predict_scores

    return predict_scores(model, frame, include_dom, batch_size)


def fit_hybrid(frame, config, include_dom, *, linear_config, tree_config) -> dict:
    """Out-of-fold stacking: linear + tree scores -> logistic meta-classifier."""
    if len(frame) == 0:
        raise ValueError("training frame must not be empty")
    labels = frame["label"].to_numpy(dtype=np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("training labels must contain only 0 and 1")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(
        n_splits=int(config.oof_folds), shuffle=True, random_state=int(config.seed)
    )
    oof_linear = np.empty(len(frame), dtype=float)
    oof_tree = np.empty(len(frame), dtype=float)

    for train_idx, holdout_idx in splitter.split(frame, labels):
        train_part = frame.iloc[train_idx]
        holdout_part = frame.iloc[holdout_idx]
        linear_model = _fit_linear(train_part, linear_config, include_dom)
        tree_model = fit_tree(train_part, tree_config, include_dom)
        oof_linear[holdout_idx] = _linear_scores(linear_model, holdout_part, include_dom)
        oof_tree[holdout_idx] = predict_tree_scores(tree_model, holdout_part, include_dom)

    meta = LogisticRegression(
        C=float(config.meta_c), random_state=int(config.seed), max_iter=1000
    )
    meta.fit(np.column_stack((oof_linear, oof_tree)), labels)

    linear_full = _fit_linear(frame, linear_config, include_dom)
    tree_full = fit_tree(frame, tree_config, include_dom)
    return {"linear": linear_full, "tree": tree_full, "meta": meta}


def predict_hybrid_scores(model, frame, include_dom, batch_size=4096) -> np.ndarray:
    """Predict scores via hybrid stacking: linear + tree -> meta."""
    _require_positive_integer("batch_size", batch_size)
    linear_scores = _linear_scores(model["linear"], frame, include_dom, batch_size)
    tree_scores = predict_tree_scores(model["tree"], frame, include_dom, batch_size)

    n = len(frame)
    batch = int(batch_size)
    chunks = [
        model["meta"].predict_proba(
            np.column_stack((linear_scores[i : i + batch], tree_scores[i : i + batch]))
        )[:, 1]
        for i in range(0, n, batch)
    ]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=float)
