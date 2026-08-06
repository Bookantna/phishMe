# phishMe

phishMe is a leakage-aware lightweight phishing detection research pipeline for a
sparse, browser-computable classifier.

## References

- [Approved design](docs/superpowers/specs/2026-08-05-phishme-research-pipeline-design.md)
- [Scientific method](METHOD.md)
- [Implementation log](Implement.md)

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

## Commands

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
