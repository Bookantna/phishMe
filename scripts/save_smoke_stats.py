"""Write phresh-smoke-stats.json: full V3 model stats on validation + PhreshPhish smoke slice.

Reads a v3-train output dir (validation-report.json, models/*.joblib) and a frozen
PhreshPhish records JSONL, scores all variants with the project's own evaluation
code, and writes a machine-readable stats file next to the artifacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from phishme import evaluate, models as M, train
from phishme.cross_dataset import VARIANT_KINDS, compute_delta

OUTPUT_DIR = Path(r"D:\phishme-dataset\artifacts\phishpedia-full-v3")
RECORDS = OUTPUT_DIR / "phresh-test-records.jsonl"
STATS_PATH = OUTPUT_DIR / "phresh-smoke-stats.json"

METRIC_KEYS = [
    "average_precision", "f1", "accuracy", "precision", "recall",
    "roc_auc", "false_positive_rate", "brier", "confusion_matrix",
]


def score_variant(kind, model, frame):
    if kind == "linear":
        return train.predict_scores(model, frame, include_dom=True)
    if kind == "tree":
        return M.predict_tree_scores(model, frame, include_dom=True)
    return M.predict_hybrid_scores(model, frame, include_dom=True)


def operating_points(labels, scores):
    """TPR at fixed FPR thresholds (rank-based, mirror of benchmark.evaluate_base_rates)."""
    order = np.argsort(-scores)
    fpr = np.cumsum(labels[order] == 0) / max(1, int((labels == 0).sum()))
    tpr = np.cumsum(labels[order] == 1) / max(1, int((labels == 1).sum()))
    return {
        f"tpr_at_{int(fpr_target * 1000):d}permille_fpr": float(tpr[min(np.searchsorted(fpr, fpr_target), len(tpr) - 1)])
        for fpr_target in (0.001, 0.01, 0.05, 0.10, 0.20)
    }


def main() -> int:
    report = json.loads((OUTPUT_DIR / "validation-report.json").read_text(encoding="utf-8"))
    frame = pd.read_json(RECORDS, lines=True)
    labels = frame["label"].to_numpy(dtype=np.int8)

    stats = {
        "schema": "phishme-v3-smoke-stats-v1",
        "records": {
            "path": str(RECORDS),
            "count": int(len(frame)),
            "benign": int((labels == 0).sum()),
            "phish": int((labels == 1).sum()),
            "phish_rate": round(float(labels.mean()), 6),
            "note": "first 500 rows of the pinned PhreshPhish test split in stream order",
        },
        "seed": report.get("seed"),
        "variants": {},
    }

    for kind in VARIANT_KINDS:
        variant = report["variants"][kind]
        model = joblib.load(OUTPUT_DIR / "models" / f"{kind}.joblib")
        threshold = variant["threshold"]
        scores = score_variant(kind, model, frame)

        validation_metrics = {k: variant["validation_metrics"][k] for k in METRIC_KEYS}
        phresh_metrics = evaluate.metrics_report(labels, scores, threshold)
        phresh_metrics = {k: phresh_metrics[k] for k in METRIC_KEYS}
        delta = compute_delta(variant["validation_metrics"], phresh_metrics)

        stats["variants"][kind] = {
            "threshold": threshold,
            "threshold_source": "validation",
            "validation_metrics": validation_metrics,
            "phreshphish_metrics": phresh_metrics,
            "delta_vs_validation": delta,
            "operating_points": operating_points(labels, scores),
        }

    STATS_PATH.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {STATS_PATH}")
    print(json.dumps({k: v for k, v in stats.items() if k != "variants"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
