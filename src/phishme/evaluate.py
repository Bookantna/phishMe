from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_threshold(y_true, scores) -> float:
    y, score_values = _validate_labels_and_scores(y_true, scores)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        warnings.simplefilter("ignore", UserWarning)
        precision, recall, thresholds = precision_recall_curve(y, score_values)
    if len(thresholds) == 0:
        raise ValueError("scores must not be empty")

    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
    return float(thresholds[int(np.nanargmax(f1_values))])


def metrics_report(y_true, scores, threshold) -> dict:
    y, score_values = _validate_labels_and_scores(y_true, scores)
    predictions = (score_values >= float(threshold)).astype(np.int8)
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    true_negatives, false_positives = matrix[0]
    false_positive_denominator = true_negatives + false_positives
    false_positive_rate = (
        false_positives / false_positive_denominator if false_positive_denominator else 0.0
    )

    return {
        "average_precision": _average_precision(y, score_values),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "roc_auc": _roc_auc(y, score_values),
        "false_positive_rate": float(false_positive_rate),
        "brier": float(brier_score_loss(y, score_values)),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def paired_bootstrap(
    y,
    left,
    right,
    left_threshold,
    right_threshold,
    seed,
    resamples,
) -> dict:
    y_values, left_scores, right_scores = _validate_labels_and_scores(y, left, right)
    if isinstance(resamples, bool) or int(resamples) != resamples or resamples <= 0:
        raise ValueError("resamples must be a positive integer")

    rng = np.random.default_rng(seed)
    ap_differences = np.empty(int(resamples), dtype=float)
    f1_differences = np.empty(int(resamples), dtype=float)
    for index in range(int(resamples)):
        sample_indices = rng.integers(0, len(y_values), size=len(y_values))
        sample_y = y_values[sample_indices]
        sample_left = left_scores[sample_indices]
        sample_right = right_scores[sample_indices]
        ap_differences[index] = _average_precision(sample_y, sample_left) - _average_precision(
            sample_y, sample_right
        )
        f1_differences[index] = _f1(sample_y, sample_left, left_threshold) - _f1(
            sample_y, sample_right, right_threshold
        )

    return {
        "average_precision_diff": _percentile_interval(ap_differences),
        "f1_diff": _percentile_interval(f1_differences),
    }


def _validate_labels_and_scores(y_true, *score_arrays):
    y = np.asarray(y_true)
    if y.ndim != 1:
        raise ValueError("y_true must be one-dimensional")
    if len(y) == 0:
        raise ValueError("inputs must not be empty")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1 labels")
    y = y.astype(np.int8)

    validated_scores = []
    for scores in score_arrays:
        score_values = np.asarray(scores, dtype=float)
        if score_values.ndim != 1:
            raise ValueError("scores must be one-dimensional")
        if len(score_values) != len(y):
            raise ValueError("scores must have the same length as y_true")
        if not np.isfinite(score_values).all():
            raise ValueError("scores must be finite")
        validated_scores.append(score_values)
    return (y, *validated_scores)


def _average_precision(y, scores) -> float:
    if not np.any(y == 1):
        return 0.0
    if not np.any(y == 0):
        return 1.0
    return float(average_precision_score(y, scores))


def _roc_auc(y, scores) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


def _f1(y, scores, threshold) -> float:
    predictions = (scores >= float(threshold)).astype(np.int8)
    return float(f1_score(y, predictions, zero_division=0))


def _percentile_interval(values: np.ndarray) -> dict[str, float]:
    low, median, high = np.percentile(values, [2.5, 50.0, 97.5])
    return {"p2.5": float(low), "p50": float(median), "p97.5": float(high)}
