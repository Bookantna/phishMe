# phishMe

phishMe is a leakage-aware lightweight phishing detection research pipeline for a
sparse, browser-computable classifier.

## References

- [V3 Cross-Dataset Design](docs/superpowers/specs/2026-08-08-phishme-v3-cross-dataset-design.md)
- [Original Approved Design](docs/superpowers/specs/2026-08-05-phishme-research-pipeline-design.md)
- [Scientific method](METHOD.md)
- [Implementation log](Implement.md)

## Achievements

### V3 PhishPedia training run (complete)

Full V3 training executed on the official PhishPedia paired archives
(46,786 canonical rows; 28,068 train / 9,356 validation / 9,362 test,
group-disjoint by registrable domain and phishing `family_id`, seed 42).
All three variants were fitted, thresholds were selected on validation only,
and the holdout was scored exactly once. Artifacts (models, model-linear.json,
validation-report.json, run.json) live in
`D:\phishme-dataset\artifacts\phishpedia-full-v3` and are audited by manifest
sha256 hashes.

### Cross-dataset evaluation on the frozen PhreshPhish test split (smoke slice)

The pinned PhreshPhish test split (Hugging Face revision
`eabec4b7a66324b79cc8a0ad856d1731dc26fe1a`, verified reachable via the
Hugging Face MCP) was streamed and frozen into a local JSONL. A 500-record
smoke slice (275 benign / 225 phish, first rows in stream order) was scored
with the trained variants using the project's own evaluation code. Full
per-variant metrics are saved in
`D:\phishme-dataset\artifacts\phishpedia-full-v3\phresh-smoke-stats.json`
(schema `phishme-v3-smoke-stats-v1`).

| Variant | Val AP | Val F1 | Phresh AP | Phresh F1 | ΔAP | ΔF1 | Phresh prec | Phresh rec | Phresh FPR | AUC |
|---------|--------|--------|-----------|-----------|-----|-----|-------------|------------|------------|-----|
| linear  | 0.9999 | 0.9979 | 0.4398    | 0.5365    | 0.5601 | 0.4614 | 0.4341 | 0.7022 | 0.7491 | 0.4623 |
| tree    | 1.0000 | 0.9994 | 0.3970    | 0.5373    | 0.6030 | 0.4621 | 0.4045 | 0.8000 | 0.9636 | 0.3933 |
| hybrid  | 1.0000 | 0.9995 | 0.4024    | 0.5236    | 0.5976 | 0.4759 | 0.3981 | 0.7644 | 0.9455 | 0.4169 |

**Key finding (negative result, honestly reported):** variants that are
near-perfect on PhishPedia validation (AP ≥ 0.9999, F1 ≥ 0.9979) do **not**
generalize to the PhreshPhish slice. Precision collapses to 0.40–0.43, FPR
explodes to 75–96%, accuracy (0.37–0.45) falls below the always-benign
baseline, and AUC 0.39–0.46 indicates the score ranking is mildly
anti-correlated with labels. This is directional evidence — the slice is the
first 500 test rows in stream order, so error bars are wide; the full 168,060
record test set is required for the definitive generalization read.

### PhishLang official baseline pipeline

- Official PhishLang repository (UTA-SPRLab/phishlang, commit `6b72838`) cloned
  with the 98 MB MobileBERT checkpoint; repo cleanliness and remote origin are
  validated before every scoring run.
- GPU scoring pipeline built on the project's `run_phishlang` entry point
  (CUDA model placement + tokenizer wrapper for transformers 5.x
  `BatchEncoding`), exercised end-to-end on records and verified for
  correctness on sample rows.
- **Latency finding:** the project's scorer runs one 128-token window per
  forward pass (~100–120 ms/window on an RTX 4060 Ti). PhreshPhish pages are
  large (measured window distribution over the 500-record slice: median 154,
  mean 706, p90 1,439, max 21,651 windows per page; 352,921 windows total), so
  batch-1 scoring costs hours. The PhishLang paper's 0.39 s median is not
  comparable: it is a median over their smaller PhishPedia-derived pages on a
  Xeon W + 4×A5000 rig. A batched-window rewrite of the scorer
  (semantics-preserving max-over-windows aggregation, ~10–30× speedup) is the
  next step before the full paired-bootstrap comparison can complete in
  reasonable time.

### Test suite

All 146 pytest tests pass locally (1 skip, PHISHLANG_DIR-gated). Ruff lint is
clean.

## V3: Cross-Dataset Generalization Study

### Research Question

Can phishMe variants (Linear SGD, Tree LightGBM, Hybrid OOF stacking) trained on
PhishPedia generalize to the frozen PhreshPhish test split, compared to the
official pretrained PhishLang MobileBERT reference?

### V3 Commands

Test-verified V3 CLI. The official PhishPedia phishing and benign archives are
read directly as ZIP files; extraction is neither required nor recommended:

- `python -m phishme v3-train --phish-zip PATH --benign-zip PATH --output DIR`

  Loads URL metadata and `html.txt` directly from both archives, rejects rows
  missing either field, removes canonical-URL duplicates, and constructs
  connected split groups over registrable domains and phishing `family_id`
  values. It trains all three variants, selects thresholds on validation data,
  declares one primary variant using validation metrics only, scores the
  untouched local holdout once, and exports model artifacts plus a
  `validation-report.json`. `run.json` includes accepted, skipped, malformed,
  missing-field, and duplicate counts for both archives.

  **Flags:**
  - `--phish-zip PATH` (required in archive mode) — official phishing archive
  - `--benign-zip PATH` (required in archive mode) — official benign archive
  - `--output DIR` (required) — directory for V3 artifacts (models, reports)
  - `--limit N` — optional approximately class-balanced row cap for smoke testing; minimum 30
  - `--epochs N` — epochs for linear model (default: 3)
  - `--batch-size N` — SGD mini-batch size (default: 2048)
  - `--seed N` — training and split seed (default: 42)

  The former `--csv`, `--phish-html`, and `--benign-html` flags remain only for
  synthetic fixtures and compatibility with earlier experiments; they do not
  describe the official archive layout.

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

- **PhishPedia**: the
  [official project site](https://sites.google.com/view/phishpedia-site/)
  publishes separate phishing and benign archives. Each site is a top-level
  directory containing some combination of
  `info.txt`, `html.txt`, screenshots, and annotations. The phishing archive's
  `info.txt` is a metadata dictionary; the benign archive's `info.txt` is the
  URL. There is no official train/validation/test CSV in these archives. Raw
  archives and generated artifacts remain outside Git. Because the two labels
  come from separate archives, local metrics are within-corpus diagnostics;
  frozen PhreshPhish evaluation is required for cross-dataset claims.

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
| All 146 pytest tests + 1 skip | ✅ Passed locally (`env -u PHISHLANG_DIR`) |
| Ruff lint | ✅ Clean (`ruff check .` passes) |
| Node parity (7 tests) | ✅ Passed |
| CLI smoke with synthetic data | ✅ `v3-train`, `v3-eval`, `v3-browser` exercised |
| Headless Chrome `v3-browser` smoke (tiny model) | ✅ avg 0.13 ms, p95 0.20 ms, ~19.3 MB, both gates passing |
| Paired PhishPedia archive download + layout verification | ✅ Verified locally; ZIP-native smoke passed |
| Full reviewed PhishPedia V3 training | ✅ 46,786 rows; models and audited manifests written |
| PhreshPhish test streaming + materialization | ✅ Verified on a 500-record smoke slice (pinned revision, via HF) |
| PhreshPhish full test materialization (168,060 records) | 🔲 Pending (≈25–35 GB JSONL; scheduled as needed) |
| Official PhishLang predictions (500-record slice) | 🔲 Batch-1 loop proven correct but ~100–120 ms/window; batched-window rewrite pending |
| Full cross-dataset report (`cross-dataset-report.json`) | 🔲 Blocked on PhishLang predictions CSV |

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

For PhishLang GPU scoring (torch + transformers, CUDA build):
```bash
uv pip install --python .venv/Scripts/python.exe torch transformers bs4 \
  --index-url https://download.pytorch.org/whl/cu128
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
