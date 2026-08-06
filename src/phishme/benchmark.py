from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from numbers import Integral, Real

import numpy as np

from phishme.evaluate import metrics_report

BASE_RATE_PREVALENCES = (0.0005, 0.001, 0.005, 0.01, 0.05)


def base_rate_manifest(labels, prevalence, seed) -> np.ndarray:
    """Return deterministic shuffled original row indices for a requested base rate."""

    y = _validate_binary_integer_labels(labels)
    requested_prevalence = _validate_prevalence(prevalence)
    seed_value = _validate_seed(seed)

    negative_indices = np.flatnonzero(y == 0)
    positive_indices = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed_value)

    positive_count = round(requested_prevalence * len(negative_indices) / (1 - requested_prevalence))
    if positive_count <= len(positive_indices):
        selected_negatives = negative_indices
        selected_positives = _sample_without_replacement(rng, positive_indices, positive_count)
    else:
        negative_count = round(len(positive_indices) * (1 - requested_prevalence) / requested_prevalence)
        selected_negatives = _sample_without_replacement(
            rng,
            negative_indices,
            min(len(negative_indices), negative_count),
        )
        selected_positives = positive_indices

    selected = np.concatenate((selected_negatives, selected_positives)).astype(np.int64, copy=False)
    rng.shuffle(selected)
    return selected


def evaluate_base_rates(labels, scores, threshold, rates, seed, *, sample_ids=None) -> dict:
    y = _validate_binary_integer_labels(labels)
    score_values = _validate_scores(scores, len(y))
    threshold_value = _validate_threshold(threshold)
    rate_values = _validate_rates(rates)
    sample_id_values = _validate_sample_ids(sample_ids, len(y))
    seed_value = _validate_seed(seed)

    reports = []
    for rate in rate_values:
        selected = base_rate_manifest(y, rate, seed_value)
        selected_labels = y[selected]
        selected_sample_ids = [sample_id_values[index] for index in selected.tolist()]
        phishing_count = int(np.count_nonzero(selected_labels == 1))
        benign_count = int(np.count_nonzero(selected_labels == 0))
        total = len(selected)
        reports.append(
            {
                "requested_prevalence": float(rate),
                "actual_prevalence": float(phishing_count / total) if total else 0.0,
                "counts": {
                    "total": total,
                    "benign": benign_count,
                    "phishing": phishing_count,
                },
                "selected_indices": selected.astype(int).tolist(),
                "selected_sample_ids": selected_sample_ids,
                "sample_id_manifest_sha256": sample_id_list_sha256(selected_sample_ids),
                "metrics": metrics_report(
                    selected_labels,
                    score_values[selected],
                    threshold_value,
                ),
            }
        )

    return {
        "schema": "phishme-base-rate-evaluation-v1",
        "seed": seed_value,
        "threshold": threshold_value,
        "rates": reports,
    }


def sample_id_list_sha256(sample_ids: Sequence[str]) -> str:
    payload = json.dumps(
        list(sample_ids),
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_binary_integer_labels(labels) -> np.ndarray:
    y = np.asarray(labels)
    if y.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if len(y) == 0:
        raise ValueError("labels must not be empty")
    if y.dtype == np.bool_ or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("labels must be integer labels containing only 0 and 1")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1")
    if len(np.unique(y)) != 2:
        raise ValueError("labels must contain both classes")
    return y.astype(np.int8, copy=False)


def _validate_prevalence(prevalence) -> float:
    if isinstance(prevalence, bool) or not isinstance(prevalence, Real):
        raise TypeError("prevalence must be a finite float between 0 and 1")
    value = float(prevalence)
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError("prevalence must be finite with 0 < prevalence < 1")
    return value


def _validate_seed(seed) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be a non-boolean integer")
    value = int(seed)
    if value < 0:
        raise ValueError("seed must be a non-negative integer")
    return value


def _validate_scores(scores, expected_length: int) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if len(values) != expected_length:
        raise ValueError("scores must have the same length as labels")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return values


def _validate_threshold(threshold) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError("threshold must be finite")
    value = float(threshold)
    if not math.isfinite(value):
        raise ValueError("threshold must be finite")
    return value


def _validate_rates(rates) -> list[float]:
    if isinstance(rates, (str, bytes)) or not isinstance(rates, Iterable):
        raise TypeError("rates must be an iterable of unique prevalences")
    values = [_validate_prevalence(rate) for rate in rates]
    if not values:
        raise ValueError("rates must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("rates must be unique")
    return values


def _validate_sample_ids(sample_ids, expected_length: int) -> list[str]:
    if sample_ids is None:
        return [str(index) for index in range(expected_length)]
    sample_id_array = np.asarray(sample_ids, dtype=object)
    if sample_id_array.ndim != 1:
        raise ValueError("sample_ids must be one-dimensional")
    values = sample_id_array.tolist()
    if len(values) != expected_length:
        raise ValueError("sample_ids must have the same length as labels")
    output = []
    for value in values:
        if not isinstance(value, str) or value == "" or "\x00" in value:
            raise ValueError("sample_ids must be unique non-empty strings")
        output.append(value)
    if len(set(output)) != len(output):
        raise ValueError("sample_ids must be unique non-empty strings")
    return output


def _sample_without_replacement(
    rng: np.random.Generator,
    values: np.ndarray,
    count: int,
) -> np.ndarray:
    if count == len(values):
        return values.copy()
    if count == 0:
        return np.array([], dtype=np.int64)
    return rng.choice(values, size=count, replace=False)
