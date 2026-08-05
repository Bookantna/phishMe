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

Verified by synthetic integration tests:

- `python -m phishme audit --csv PATH`
- `python -m phishme local --csv PATH --output DIR`

`local` writes `DIR/run.json`, `DIR/splits.json`, `DIR/model.json`, and
`DIR/test-metrics.json`. It accepts optional `--epochs`, `--batch-size`, and
`--seed` arguments.

The real PhiUSIIL local baseline is pending host execution; no real baseline
metrics are verified in this repository state.

Planned or separately verified commands:

- `python -m phishme phresh-smoke`
- `node --test web/parity.test.mjs`
- Open `web/benchmark.html` in Chrome
