# phishMe Implementation Log

## Status
- In progress: real local baseline execution on the full PhiUSIIL CSV.
- Complete: Task 6 synthetic end-to-end local CLI implementation.
- Complete: Task 2 canonical PhiUSIIL adapter and leakage-safe local split.
- Blocked: full PhreshPhish training requires a cloud runtime.
- Not run: controlled PhishLang comparison.

## Commands and observed results
- `.venv/bin/python -m pytest tests/test_data.py tests/test_splits.py -q` before implementation:
  failed during collection with `ModuleNotFoundError: No module named 'phishme.data'`
  for both focused test modules.
- `.venv/bin/python -m pytest tests/test_data.py tests/test_splits.py -q` after implementation:
  `4 passed in 20.34s`.
- `.venv/bin/python - <<'PY' ... PY` real CSV audit on
  `dataset/PhiUSIIL_Phishing_URL_Dataset.csv`:
  loaded 233,971 canonical rows and removed 1,824 canonical duplicates;
  `{'train': {'rows': 140382, 'phishing': 59473, 'benign': 80909, 'domains': 105309}, 'validation': {'rows': 46794, 'phishing': 19824, 'benign': 26970, 'domains': 35101}, 'test': {'rows': 46795, 'phishing': 19825, 'benign': 26970, 'domains': 35099}}`.
  tldextract emitted a cache warning inside the Codex sandbox, so the final extractor disables
  its cache and uses only the bundled public-suffix snapshot; the host audit then ran cleanly.
- `.venv/bin/python -m pytest tests/test_pipeline.py -q` before Task 6 implementation:
  failed with `No module named phishme.__main__; 'phishme' is a package and cannot be
  directly executed`.
- `.venv/bin/python -m pytest tests/test_pipeline.py -q` after Task 6 implementation:
  `2 passed in 5.57s`.
- `.venv/bin/python -m pytest -q` after Task 6 implementation:
  `51 passed in 6.55s`.
- `.venv/bin/ruff check .` after Task 6 implementation:
  `All checks passed!`.
- `node --test web/parity.test.mjs` after Task 6 implementation:
  `7` tests passed.
- `.venv/bin/python -m phishme audit --csv /tmp/.../synthetic.csv` smoke:
  exited 0 with strict JSON, `50` raw rows, `50` canonical rows, `0` duplicate rows,
  canonical label counts `{"0": 26, "1": 24}`, and disjoint split manifests.
- `.venv/bin/python -m phishme local --csv /tmp/.../synthetic.csv --output /tmp/.../run --epochs 1`
  smoke: exited 0 and wrote `run.json`, `splits.json`, `model.json`, and
  `test-metrics.json`; synthetic `model.json` was `1,051,759` bytes. The selected
  synthetic config was `include_dom=false`, `alpha=1e-05`, `threshold=1.0`.

## Decisions and deviations
- Used `.venv/bin/python` because the shell `python` resolves to a different virtualenv outside
  this worktree.
- Task 2 copies only `url`, `title`, inverted `label`, `sample_id`, and `group`; no dataset
  feature columns are exposed.
- Task 6 `audit` and `local` use `argparse`. `local` uses the existing
  `load_phiusill`, `split_local`, `fit_incremental`, `predict_scores`,
  `select_threshold`, `metrics_report`, and `export_model` interfaces.
- Task 6 trains exactly six local candidates from DOM off/on crossed with alpha
  `1e-5`, `1e-4`, and `1e-3`; ranking is `(validation_ap, validation_f1, -alpha,
  -int(include_dom))`.
- `run.json` records byte sizes and SHA-256 hashes for `splits.json`, `model.json`,
  and `test-metrics.json`. The command stdout summary includes the final `run.json`
  byte size and SHA-256 because embedding an exact self-hash inside `run.json` would
  be self-referential.
- The real `dataset/PhiUSIIL_Phishing_URL_Dataset.csv` baseline command was not run
  in this worker. It remains pending host execution; no real AP/F1/accuracy values
  are claimed.

## Failed or negative results
- Expected red test run confirmed missing `phishme.data` before implementation.
- Expected Task 6 red test run confirmed missing `phishme.__main__` before CLI implementation.
- The shell `python -m pytest tests/test_pipeline.py -q` command resolved to a different
  virtualenv without pytest; verification used `.venv/bin/python`.

## Artifacts
- `src/phishme/data.py`
- `tests/test_data.py`
- `tests/test_splits.py`
- `src/phishme/__main__.py`
- `tests/conftest.py`
- `tests/test_pipeline.py`

## Next actions
- Host should run the full real local baseline:
  `python -m phishme audit --csv dataset/PhiUSIIL_Phishing_URL_Dataset.csv` and
  `python -m phishme local --csv dataset/PhiUSIIL_Phishing_URL_Dataset.csv --output artifacts/local-v1`.
