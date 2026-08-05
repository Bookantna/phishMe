import json

import numpy as np
import pytest

from phishme.evaluate import metrics_report, paired_bootstrap, select_threshold


def _assert_strict_json_serializable(report: dict) -> None:
    assert json.dumps(report, allow_nan=False)


def test_threshold_uses_validation_scores():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.05, 0.2, 0.7, 0.9])

    threshold = select_threshold(y, scores)

    assert 0.2 < threshold <= 0.7


def test_threshold_returns_first_f1_tie():
    y = np.array([1, 0, 0, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4])

    assert select_threshold(y, scores) == 0.1


def test_report_contains_required_metrics():
    report = metrics_report(np.array([0, 1]), np.array([0.1, 0.9]), 0.5)

    assert {
        "average_precision",
        "f1",
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
        "false_positive_rate",
        "brier",
        "confusion_matrix",
    } <= set(report)
    assert report["roc_auc"] == 1.0
    assert isinstance(report["roc_auc"], float)
    _assert_strict_json_serializable(report)


def test_report_has_stable_zero_division_and_2x2_confusion_matrix():
    report = metrics_report(np.array([0, 0]), np.array([0.1, 0.2]), 0.9)

    assert report["precision"] == 0.0
    assert report["recall"] == 0.0
    assert report["f1"] == 0.0
    assert report["false_positive_rate"] == 0.0
    assert report["roc_auc"] is None
    assert report["confusion_matrix"] == [[2, 0], [0, 0]]
    _assert_strict_json_serializable(report)


def test_metrics_reject_empty_input():
    with pytest.raises(ValueError, match="must not be empty"):
        select_threshold(np.array([]), np.array([]))

    with pytest.raises(ValueError, match="must not be empty"):
        metrics_report(np.array([]), np.array([]), 0.5)


def test_paired_bootstrap_is_deterministic():
    y = np.array([0, 0, 1, 1])
    left = np.array([0.1, 0.2, 0.8, 0.9])
    right = np.array([0.2, 0.3, 0.7, 0.8])

    assert paired_bootstrap(y, left, right, 0.5, 0.5, 7, 20) == paired_bootstrap(
        y, left, right, 0.5, 0.5, 7, 20
    )


def test_paired_bootstrap_reports_percentile_differences():
    y = np.array([0, 0, 1, 1, 1])
    left = np.array([0.05, 0.35, 0.6, 0.8, 0.95])
    right = np.array([0.2, 0.25, 0.55, 0.7, 0.9])

    report = paired_bootstrap(y, left, right, 0.5, 0.5, seed=123, resamples=30)

    assert set(report) == {"average_precision_diff", "f1_diff"}
    assert set(report["average_precision_diff"]) == {"p2.5", "p50", "p97.5"}
    assert set(report["f1_diff"]) == {"p2.5", "p50", "p97.5"}


def test_paired_bootstrap_uses_same_rows_for_both_models():
    y = np.array([0, 1, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.7, 0.8, 0.9])

    report = paired_bootstrap(y, scores, scores, 0.5, 0.5, seed=11, resamples=40)

    assert all(value == 0.0 for value in report["average_precision_diff"].values())
    assert all(value == 0.0 for value in report["f1_diff"].values())
