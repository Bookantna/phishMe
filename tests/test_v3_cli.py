# tests/test_v3_cli.py
"""CLI tests for v3-train, v3-eval, and v3-browser subcommands."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from phishme import __main__
from phishme.cross_dataset import run_v3_train
from phishme.features import NUMERIC_FEATURES
from phishme.phishpedia import SPLIT_CSV_NAME

SPLIT_CSV = SPLIT_CSV_NAME
LIGHTGBM = pytest.importorskip("lightgbm")


def _load_measure_browser():
    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_browser.py"
    spec = importlib.util.spec_from_file_location("measure_browser", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _make_html(is_phish: bool, idx: int) -> str:
    if is_phish:
        links = "<a href='/link'>link</a>" * (idx % 5)
        return (
            f"<html><head><title>alert {idx}</title></head>"
            f"<body>{links}"
            f"<form action='https://evil{idx}.example/login'>"
            f"<input type='password'></form></body></html>"
        )
    else:
        links = "<a href='/link'>link</a>" * (idx % 5)
        return (
            f"<html><head><title>page {idx}</title></head>"
            f"<body><p>Welcome to page {idx}</p>"
            f"{links}</body></html>"
        )


@pytest.fixture
def phishpedia_v3_site(tmp_path: Path):
    """Build a synthetic PhishPedia site with a split column and enough rows for training."""
    phish_root = tmp_path / "phish_html"
    phish_root.mkdir()
    benign_root = tmp_path / "benign_html"
    benign_root.mkdir()

    rows = []
    # 0-15: phishing, 16-31: benign  (32 total)
    for idx in range(32):
        is_phish = idx < 16
        label = 1 if is_phish else 0
        fname = f"{'p' if is_phish else 'b'}{idx}"
        url = (
            f"https://evil{idx}.example/login"
            if is_phish
            else f"https://good{idx}.example/page"
        )
        html = _make_html(is_phish, idx)

        if is_phish:
            (phish_root / f"{fname}.html").write_text(html)
        else:
            (benign_root / f"{fname}.html").write_text(html)

        if idx < 10 or (16 <= idx < 26):
            split = "train"
        elif idx < 13 or (26 <= idx < 29):
            split = "validation"
        else:
            split = "test"

        rows.append(
            {
                "file_name": fname,
                "label": label,
                "url": url,
                "type": "phishing" if is_phish else "benign",
                "split": split,
            }
        )

    csv_path = tmp_path / SPLIT_CSV
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, phish_root, benign_root


# ---------------------------------------------------------------------------
# v3-train CLI tests
# ---------------------------------------------------------------------------


def test_v3_train_writes_report(tmp_path: Path, phishpedia_v3_site):
    """v3-train exits 0 and writes validation-report.json with three variants."""
    csv_path, phish_root, benign_root = phishpedia_v3_site
    out_dir = tmp_path / "out"
    exit_code = __main__.main(
        [
            "v3-train",
            "--csv", str(csv_path),
            "--phish-html", str(phish_root),
            "--benign-html", str(benign_root),
            "--output", str(out_dir),
            "--limit", "50",
            "--seed", "42",
        ]
    )
    assert exit_code == 0

    report_path = out_dir / "validation-report.json"
    assert report_path.exists(), f"missing {report_path}"

    report = json.loads(report_path.read_text())
    assert set(report["variants"]) == {"linear", "tree", "hybrid"}

    # Verify the split_source is recorded
    assert "split_source" in report
    assert report["split_source"] == "official_csv_split"


# ---------------------------------------------------------------------------
# v3-eval CLI tests
# ---------------------------------------------------------------------------


def test_v3_eval_with_records_path(tmp_path: Path, phishpedia_v3_site):
    """v3-eval --records exits 0 and writes cross-dataset-report.json."""
    csv_path, phish_root, benign_root = phishpedia_v3_site

    from phishme.phishpedia import load_phishpedia

    frame = load_phishpedia(csv_path, phish_root, benign_root)
    train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
    val_frame = frame[frame["split"] == "validation"].reset_index(drop=True)

    # Build a run dir with run_v3_train
    run_dir = tmp_path / "run"
    run_v3_train(train_frame, val_frame, output_dir=run_dir, seed=42)

    # Build frozen records from validation frame (pretend they're PhreshPhish)
    # Use the same DOM features the model was trained on
    records_path = tmp_path / "frozen-records.jsonl"
    records = []
    for _, row in val_frame.iterrows():
        rec = {
            "url": row["url"].replace("example", "phresh.example"),
            "title": row["title"],
            "date": "2024-01-15T00:00:00Z",
            "label": int(row["label"]),
            "sample_id": row["sample_id"] + "_phresh",
            "html": f"<html><title>{row['title']}</title></html>",
            "_parse_status": "ok",
        }
        for name in NUMERIC_FEATURES:
            dom_key = f"dom_{name}"
            if dom_key in row:
                rec[dom_key] = row[dom_key]
        records.append(rec)

    with records_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True, allow_nan=False) + "\n")

    # Build a fake PhishLang CSV
    phishlang_path = tmp_path / "phishlang.csv"
    pl_rows = []
    for _, row in val_frame.iterrows():
        pl_rows.append(
            {
                "sample_id": row["sample_id"] + "_phresh",
                "label": int(row["label"]),
                "score": 0.5,
            }
        )
    pd.DataFrame(pl_rows).to_csv(phishlang_path, index=False)

    exit_code = __main__.main(
        [
            "v3-eval",
            "--output", str(run_dir),
            "--phishlang-csv", str(phishlang_path),
            "--records", str(records_path),
            "--resamples", "50",
            "--seed", "42",
        ]
    )
    assert exit_code == 0

    x_report_path = run_dir / "cross-dataset-report.json"
    assert x_report_path.exists(), f"missing {x_report_path}"

    x_report = json.loads(x_report_path.read_text())
    assert x_report["schema"] == "phishme-v3-cross-dataset-v1"
    assert set(x_report["variants"]) == {"linear", "tree", "hybrid"}
    assert "phishlang" in x_report


# ---------------------------------------------------------------------------
# v3-browser CLI tests
# ---------------------------------------------------------------------------


def test_parse_benchmark_json():
    """parse_benchmark_json extracts strict JSON from the benchmark-results pre element."""
    module = _load_measure_browser()

    html = (
        '<html><body><pre id="benchmark-results">'
        '{"avg_latency_ms":12.5,"p95_latency_ms":20.1,'
        '"memory_bytes":1048576,"pages":50,"model_bytes":5913068,'
        '"latencies":[1.2,3.4],"scorer_version":"phishme-features-v1"}'
        "</pre></body></html>"
    )
    result = module.parse_benchmark_json(html)
    assert result["avg_latency_ms"] == 12.5
    assert result["p95_latency_ms"] == 20.1
    assert result["memory_bytes"] == 1048576
    assert result["pages"] == 50
    assert result["model_bytes"] == 5913068
    assert result["scorer_version"] == "phishme-features-v1"
    assert result["latencies"] == [1.2, 3.4]
