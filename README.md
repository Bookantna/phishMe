# phishMe

phishMe is a leakage-aware lightweight phishing detection research pipeline for a
sparse, browser-computable classifier. The current repository state is the Task 1
package shell and research protocol.

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

The following commands are planned and are not verified until Task 9:

- `python -m phishme audit`
- `python -m phishme local`
- `python -m phishme phresh-smoke`
- `node --test web/parity.test.mjs`
- Open `web/benchmark.html` in Chrome
