# phishMe

phishMe is a leakage-aware lightweight phishing detection research pipeline for a
sparse, browser-computable classifier.

## References

- [V3 Cross-Dataset Design](docs/superpowers/specs/2026-08-08-phishme-v3-cross-dataset-design.md)
- [Original Approved Design](docs/superpowers/specs/2026-08-05-phishme-research-pipeline-design.md)
- [Scientific method](METHOD.md)
- [Implementation log](Implement.md)

## V3: Cross-Dataset Generalization Study

### Research Question

Can phishMe variants (Linear SGD, Tree LightGBM, Hybrid OOF stacking) trained on
PhishPedia generalize to the frozen PhreshPhish test split, compared to the
official pretrained PhishLang MobileBERT reference?

### V3 Commands

Test-verified V3 CLI; real dataset runs require PhishPedia download (Google
Drive) and a networked runtime for PhreshPhish streaming:

- `python -m phishme v3-train --csv PATH --phish-html PATH --benign-html PATH --output DIR`

  Trains all three phishMe variants on the PhishPedia benchmark, selects
  thresholds on PhishPedia validation, and exports model artifacts plus a
  `validation-report.json`. Accepts optional `--limit`, `--epochs`,
  `--batch-size`, and `--seed`.

  **Flags:**
  - `--csv PATH` (required) — PhishPedia split CSV (`train_test_val_split_30.csv`)
  - `--phish-html PATH` (required) — phishing HTML root directory
  - `--benign-html PATH` (required) — benign HTML root directory
  - `--output DIR` (required) — directory for V3 artifacts (models, reports)
  - `--limit N` — optional row cap for smoke testing
  - `--epochs N` — epochs for linear model (default: 3)
  - `--batch-size N` — SGD mini-batch size (default: 2048)
  - `--seed N` — training and split seed (default: 42)

- `python -m phishme v3-eval --output DIR --phishlang-csv PATH --records PATH`

  Scores all trained variants on the frozen PhreshPhish test set, computes ΔAP/
  ΔF1 vs validation, runs low-base-rate evaluation, aligns PhishLang predictions
  by sample_id, and computes paired bootstrap (10,000 resamples, 95% CI). Writes
  `cross-dataset-report.json` to the output directory.

  **Flags:**
  - `--output DIR` (required) — V3 run directory (from `v3-train`)
  - `--phishlang-csv PATH` (required) — frozen PhishLang predictions CSV
  - `--records PATH` (optional) — frozen PhreshPhish test records JSONL; when omitted, streams and materializes the pinned PhreshPhish test split first (requires network)
  - `--resamples N` — paired bootstrap resamples (default: 10000)
  - `--seed N` — bootstrap seed (default: 42)

- `python -m phishme v3-browser --model PATH --output DIR`

  Serves `web/` and the exported linear model artifact, launches headless Chrome,
  collects per-page scoring latencies and memory usage, and writes
  `browser-metrics.json` with schema `phishme-browser-metrics-v1`. Reports gate
  booleans: payload ≤ 25 MB, Chrome p95 ≤ 250 ms. Requires Chrome/Chromium on
  the host (checked via `CHROME_BIN` env or common macOS/Linux paths).

  **Flags:**
  - `--model PATH` (required) — exported `model.json` artifact
  - `--output DIR` (required) — directory for `browser-metrics.json`

### V3 Data Policy

- **PhishPedia** (CC0-1.0): downloaded from the official Google Drive link
  `https://drive.google.com/file/d/12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g/view`
  (referenced from `github.com/lindsey98/Phishpedia`). The archive contains
  `train_test_val_split_30.csv`, a phishing HTML directory, and a benign HTML
  directory. Kept under `dataset/` (git-ignored). No rows, HTML, or derivatives
  are committed.

- **PhreshPhish** (CC BY 4.0): streamed from Hugging Face datasets at pinned
  revision `eabec4b7a66324b79cc8a0ad856d1731dc26fe1a`. Test split is frozen
  once into a local JSONL; the JSONL is git-ignored. Raw HTML is parsed and
  discarded.

- **PhishLang** (official pretrained): predictions CSV produced by
  `scripts/run_phishlang.py` from a clean checkout of
  `github.com/UTA-SPRLab/phishlang.git` with local model files only. Neither
  the model (98 MB) nor predictions are committed.

### What is Verified Locally vs Pending Cloud

| Item | Status |
|------|--------|
| All 135 pytest tests + 1 skip | ✅ Passed locally (`env -u PHISHLANG_DIR`) |
| Ruff lint | ✅ Clean (`ruff check .` passes) |
| Node parity (7 tests) | ✅ Passed |
| CLI smoke with synthetic data | ✅ `v3-train`, `v3-eval`, `v3-browser` exercised |
| Headless Chrome `v3-browser` smoke (tiny model) | ✅ avg 0.13 ms, p95 0.20 ms, ~19.3 MB, both gates passing |
| PhishPedia archive download + layout verification | 🔲 Manual (Google Drive) |
| Full PhreshPhish test streaming | 🔲 Needs networked runtime (HF access) |
| Official PhishLang predictions run | 🔲 Needs `PHISHLANG_DIR` (clean checkout + 98 MB model) |

### V3 Browser Smoke Results

Headless Chrome `v3-browser` smoke on a tiny exported linear model:

| Metric | Value |
|--------|-------|
| Average latency | 0.13 ms |
| p95 latency | 0.20 ms |
| Memory | ~19.3 MB |
| Payload ≤ 25 MB gate | ✅ passing |
| p95 ≤ 250 ms gate | ✅ passing |

**Note:** These values were measured on a tiny smoke model, not a fully-trained
PhishPedia model. Real trained-model latency is pending PhishPedia download and
full training.

## Data Policy

Raw datasets, HTML corpora, model checkpoints, generated caches, reports, and
large artifacts are excluded from Git. Local users must obtain datasets from
authorized sources and keep them outside committed source files.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,cloud]'
python -m pytest tests/test_package.py -q
```

For V3 with LightGBM:
```bash
python -m pip install -e '.[tree,dev,cloud]'
```

## Commands (V2 Baseline)

Verified by synthetic integration tests and by the full local PhiUSIIL baseline:

- `python -m phishme audit --csv PATH`
- `python -m phishme local --csv PATH --output DIR`

`local` writes `DIR/run.json`, `DIR/splits.json`, `DIR/model.json`, and
`DIR/test-metrics.json`. It accepts optional `--epochs`, `--batch-size`, and
`--seed` arguments.

The verified local baseline used 233,971 canonical rows with domain-grouped,
disjoint train/validation/test partitions. The selected DOM-enabled model
achieved test average precision `0.9999888939542867`, F1
`0.9998234240597331`, and accuracy `0.999850411368736`. Full provenance,
split hashes, and artifact hashes are recorded in `Implement.md`; generated
artifacts remain excluded from Git.

Test-verified cloud interface; the local network attempt failed and Colab execution is pending:

- `python -m phishme phresh-smoke --limit 1000 --output DIR`

`phresh-smoke` streams the pinned PhreshPhish Hugging Face train split, derives
the pinned temporal cutoff from projected `date,label` columns, checkpoints a
bounded smoke model under `DIR/checkpoints/`, and writes `DIR/model.json` plus
`DIR/run.json`. Training records are always filtered to `date` values strictly
before the cutoff; rows at or after the cutoff are never used for training in
smoke or full PhreshPhish modes. The `datasets` dependency is optional and lazy;
install `.[cloud]` before running this command. Smoke artifacts use threshold
`0.5` only as a plumbing check and do not claim benchmark metrics.

Planned or separately verified commands:

- `node --test web/parity.test.mjs` (verified: seven tests passed)
- Open `web/benchmark.html` in Chrome

The exact network smoke command must be rerun in Colab/a networked host:
`python -m phishme phresh-smoke --limit 1000 --output artifacts/phresh-smoke`.
