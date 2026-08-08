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
