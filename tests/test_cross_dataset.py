import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from phishme.cross_dataset import (
    CROSS_SCHEMA,
    VALIDATION_SCHEMA,
    VARIANT_KINDS,
    align_phishlang_scores,
    compute_delta,
    materialize_phresh_test,
    run_v3_eval,
    run_v3_train,
)
from phishme.data import _sample_id, registrable_domain
from phishme.features import NUMERIC_FEATURES


def _synthetic_phishpedia_frame(n=80):
    rows = []
    for i in range(n):
        phish = i % 2 == 1
        rows.append({
            "url": f"http{'s' if not phish else ''}://d{i % 10}.example/p{i}",
            "title": "verify password" if phish else "welcome",
            "label": 1 if phish else 0,
        })
    frame = pd.DataFrame(rows)
    frame["sample_id"] = frame["url"].map(_sample_id)
    frame["group"] = frame["url"].map(registrable_domain)
    for name in NUMERIC_FEATURES:
        frame[f"dom_{name}"] = 0.0
    return frame


def test_compute_delta_signs():
    validation = {"average_precision": 0.95, "f1": 0.90}
    phresh = {"average_precision": 0.80, "f1": 0.75}
    delta = compute_delta(validation, phresh)
    assert delta["delta_ap"] == pytest.approx(0.15)
    assert delta["delta_f1"] == pytest.approx(0.15)


def test_align_phishlang_scores_matches_by_sample_id():
    sample_ids = [f"s{i}" for i in range(4)]
    phresh_scores = np.array([0.9, 0.8, 0.2, 0.1])
    phresh_labels = np.array([1, 1, 0, 0], dtype=np.int8)
    phishlang_rows = pd.DataFrame({
        "sample_id": [sample_ids[1], sample_ids[0], sample_ids[3], sample_ids[2]],
        "label": [1, 1, 0, 0],
        "score": [0.7, 0.6, 0.1, 0.2],
        "model": ["phishlang"] * 4,
        "source_commit": ["abc"] * 4,
    })
    aligned = align_phishlang_scores(phishlang_rows, sample_ids, phresh_scores, phresh_labels)
    # phishMe scores reordered to PhishLang CSV row order: s1, s0, s3, s2
    assert aligned["left"].tolist() == [0.8, 0.9, 0.1, 0.2]
    assert aligned["right"].tolist() == [0.7, 0.6, 0.1, 0.2]
    assert aligned["labels"].tolist() == [1, 1, 0, 0]


def test_run_v3_train_writes_artifacts(tmp_path: Path):
    frame = _synthetic_phishpedia_frame()
    train_frame = frame.iloc[:50].reset_index(drop=True)
    val_frame = frame.iloc[50:].reset_index(drop=True)
    empty_test_frame = frame.iloc[:0].reset_index(drop=True)
    summary = run_v3_train(
        train_frame,
        val_frame,
        test_frame=empty_test_frame,
        output_dir=tmp_path,
        seed=42,
    )
    assert set(summary["variants"]) == {"linear", "tree", "hybrid"}
    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "validation-report.json").exists()
    report = json.loads((tmp_path / "validation-report.json").read_text(encoding="utf-8"))
    assert all("holdout_metrics" not in variant for variant in report["variants"].values())


def test_run_v3_train_validation_report_schema(tmp_path: Path):
    frame = _synthetic_phishpedia_frame()
    train_frame = frame.iloc[:50].reset_index(drop=True)
    val_frame = frame.iloc[50:].reset_index(drop=True)
    run_v3_train(train_frame, val_frame, output_dir=tmp_path, seed=42)
    report = json.loads((tmp_path / "validation-report.json").read_text(encoding="utf-8"))
    assert report["schema"] == VALIDATION_SCHEMA
    for kind in VARIANT_KINDS:
        v = report["variants"][kind]
        assert "selected" in v
        assert "threshold" in v
        assert v["threshold_source"] == "validation"
        assert "validation_metrics" in v
        assert "validation_base_rates" in v


def test_materialize_phresh_test_writes_jsonl(tmp_path: Path):
    raw_rows = [
        {"url": "https://a.example/p1", "html": "<p>phish</p>", "date": "2023-01-01", "label": "phish"},
        {"url": "https://b.example/p2", "html": "<p>ok</p>", "date": "2023-01-02", "label": "benign"},
        {"url": "https://c.example/p3", "html": "<p>bad</p>", "date": "2023-01-03", "label": "phish"},
    ]
    output_path = tmp_path / "frozen.jsonl"
    results = materialize_phresh_test(iter(raw_rows), output_path)
    assert results == {"processed": 3, "accepted": 3, "rejected": 0}
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        rec = json.loads(line)
        assert "url" in rec
        assert "html" in rec
        assert "sample_id" in rec
        assert rec["label"] in (0, 1)
    # canonical labels: phish -> 1, benign -> 0
    labels = [json.loads(line)["label"] for line in lines]
    assert labels == [1, 0, 1]


def test_materialize_phresh_test_limit(tmp_path: Path):
    raw_rows = [
        {"url": f"https://d{i}.example/p{i}", "html": f"<p>x{i}</p>", "date": "2023-01-01",
         "label": "benign" if i % 2 == 0 else "phish"}
        for i in range(10)
    ]
    output_path = tmp_path / "frozen.jsonl"
    results = materialize_phresh_test(iter(raw_rows), output_path, limit=3)
    assert results["processed"] == 3
    assert results["accepted"] == 3
    lines = output_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3


def test_materialize_phresh_test_rejects_existing_path(tmp_path: Path):
    raw_rows = [{"url": "https://a.example/p1", "html": "<p>x</p>", "date": "2023-01-01", "label": "benign"}]
    output_path = tmp_path / "frozen.jsonl"
    output_path.write_text("existing")
    with pytest.raises(ValueError, match="exists"):
        materialize_phresh_test(iter(raw_rows), output_path)


def test_run_v3_eval_per_variant_paired_bootstrap(tmp_path: Path):
    """Regression test: per-variant paired bootstrap must use each variant's own scores.

    Before the C1 fix, all three variants reused the linear variant's scores for
    the paired bootstrap — so the paired dicts were identical.  After the fix,
    each variant aligns its own scores against PhishLang.  We use XOR-pattern
    features so that linear and tree models produce different predictions,
    guaranteeing at least one paired-bootstrap dict differs.
    """
    pytest.importorskip("lightgbm")

    rng = np.random.RandomState(42)
    n = 60
    rows = []
    # Pre-generate XOR feature values so we can reuse them consistently
    f0_vals = rng.uniform(0, 1, n)
    f1_vals = rng.uniform(0, 1, n)
    for i in range(n):
        # XOR: label = 1 if (f0 > 0.5) XOR (f1 > 0.5)
        label = 1 if (f0_vals[i] > 0.5) != (f1_vals[i] > 0.5) else 0
        rows.append({
            "url": f"http{'s' if label else ''}://d{i % 10}.example/p{i}",
            "title": "verify password" if label else "welcome",
            "label": label,
            "_f0": f0_vals[i],
            "_f1": f1_vals[i],
        })
    frame = pd.DataFrame(rows)
    frame["sample_id"] = frame["url"].map(_sample_id)
    frame["group"] = frame["url"].map(registrable_domain)
    # Put XOR signal into the first two dom_* columns; zero out the rest
    feature_names = list(NUMERIC_FEATURES)
    for j, name in enumerate(feature_names):
        if j == 0:
            frame[f"dom_{name}"] = frame["_f0"]
        elif j == 1:
            frame[f"dom_{name}"] = frame["_f1"]
        else:
            frame[f"dom_{name}"] = 0.0
    frame.drop(columns=["_f0", "_f1"], inplace=True)

    train_frame = frame.iloc[:40].reset_index(drop=True)
    val_frame = frame.iloc[40:50].reset_index(drop=True)
    test_frame = frame.iloc[50:].reset_index(drop=True)

    # -- train -----------------------------------------------------------
    train_dir = tmp_path / "train"
    train_summary = run_v3_train(
        train_frame, val_frame,
        output_dir=train_dir, seed=42, epochs=1,
    )

    # -- frozen JSONL ----------------------------------------------------
    frozen_path = tmp_path / "frozen.jsonl"
    records = []
    for _, row in test_frame.iterrows():
        rec = {
            "url": row["url"],
            "title": row["title"],
            "date": "2023-01-01",
            "label": int(row["label"]),
            "sample_id": row["sample_id"],
            "html": "<p>x</p>",
            "_parse_status": "ok",
        }
        for name in NUMERIC_FEATURES:
            rec[f"dom_{name}"] = float(row[f"dom_{name}"])
        records.append(rec)
    frozen_path.write_text(
        "\n".join(json.dumps(r, separators=(",", ":"), sort_keys=True) for r in records) + "\n",
        encoding="utf-8",
    )

    # -- fake PhishLang CSV ----------------------------------------------
    phishlang_csv = tmp_path / "phishlang.csv"
    pl_rows = []
    for _, row in test_frame.iterrows():
        pl_rows.append({
            "sample_id": row["sample_id"],
            "label": int(row["label"]),
            "score": round(rng.uniform(0.3, 0.7), 6),
            "model": "phishlang",
            "source_commit": "abc123",
        })
    pd.DataFrame(pl_rows).to_csv(phishlang_csv, index=False)

    # -- build variant_models + validation_metrics -----------------------
    variant_models: dict = {"variants": {}}
    for kind in VARIANT_KINDS:
        model = joblib.load(train_dir / "models" / f"{kind}.joblib")
        variant_models["variants"][kind] = {
            "model": model,
            "threshold": train_summary["variants"][kind]["threshold"],
        }
    validation_metrics = {
        kind: train_summary["variants"][kind]["validation_metrics"]
        for kind in VARIANT_KINDS
    }

    # -- eval ------------------------------------------------------------
    eval_dir = tmp_path / "eval"
    report = run_v3_eval(
        variant_models, validation_metrics,
        frozen_path, phishlang_csv,
        output_dir=eval_dir, seed=42, resamples=50,
    )

    # -- assertions ------------------------------------------------------
    report_path = eval_dir / "cross-dataset-report.json"
    assert report_path.exists()
    on_disk = json.loads(report_path.read_text(encoding="utf-8"))
    assert on_disk["schema"] == CROSS_SCHEMA

    for kind in VARIANT_KINDS:
        v = report["variants"][kind]
        assert "phresh_metrics" in v
        assert "delta" in v
        assert "base_rates" in v
        pb = report["phishlang"]["paired_bootstrap"].get(kind)
        assert pb is not None, f"missing paired_bootstrap for {kind}"

    # Regression: the three paired-bootstrap dicts are NOT all identical
    pbs = [report["phishlang"]["paired_bootstrap"][k] for k in VARIANT_KINDS]
    assert not (pbs[0] == pbs[1] == pbs[2]), (
        "paired bootstrap dicts are all identical — "
        "per-variant alignment is broken"
    )
