# phishMe Implementation Log

## Status
- Complete: Task 7 local PHRESH streaming/checkpoint interfaces and Colab notebook.
- Complete: Task 6 end-to-end local CLI and real PhiUSIIL baseline.
- Complete: Task 2 canonical PhiUSIIL adapter and leakage-safe local split.
- Pending: Colab PhreshPhish smoke after the host network failure; full PhreshPhish training
  remains a cloud/runtime task.
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
- `.venv/bin/python -m phishme audit --csv dataset/PhiUSIIL_Phishing_URL_Dataset.csv`:
  exited 0; the dataset SHA-256 was
  `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`;
  it found 235,795 raw rows, 233,971 canonical rows, and 1,824 duplicates.
  The domain-disjoint splits were train 140,382 rows (80,909 benign, 59,473 phishing,
  105,309 groups), validation 46,794 rows (26,970 benign, 19,824 phishing, 35,101
  groups), and test 46,795 rows (26,970 benign, 19,825 phishing, 35,099 groups).
  Their sample-manifest SHA-256 values were respectively
  `15523e49c9768a7bd8945a8be21582219e800481fc9fb4a8f0f00653e7364a7f`,
  `d3c92344e3d6fc5cc6854b8fb51a3a4e508f298fda6cc0bc91913877c08e2b23`, and
  `bc7e5bfafabe59c966b14f4a0705c5e96eac68ea14b4b1aae3b5826661bfc2f9`.
- `.venv/bin/python -m phishme local --csv dataset/PhiUSIIL_Phishing_URL_Dataset.csv
  --output artifacts/local-v1`: exited 0 using source commit
  `6c939803267f67fe94e19585fc389551c4959475`; all six candidates succeeded and
  the test split was scored once. The selected candidate was DOM-enabled with alpha
  `1e-05`, three epochs, batch size 2,048, threshold `0.9717452287519257`, validation
  AP `0.9999989381709102`, and validation F1 `0.99979820401574`.
- The untouched test result was average precision `0.9999888939542867`, F1
  `0.9998234240597331`, accuracy `0.999850411368736`, ROC AUC
  `0.9999906598382865`, precision `1.0`, recall `0.9996469104665826`, false-positive
  rate `0.0`, Brier score `0.00010941090093052358`, and confusion matrix
  `[[26970, 0], [7, 19818]]`.
- Real baseline artifact outputs were: `model.json` 5,913,068 bytes, SHA-256
  `3304bd9327506e0b3f63b6d0e5a7a6c044c740cefe30ebbfd22ee0db774fc00d`;
  `run.json` 6,449 bytes, SHA-256
  `5028da5a4bfc0a6babedd02b1e89ce5353068db53c6bd779f78d4f1810ba9cb2`;
  `splits.json` 1,163 bytes, SHA-256
  `691ce334d1f18be2c925616eeb8899c55dacc739e60f25c3944eeb32484ce8e8`;
  and `test-metrics.json` 1,004 bytes, SHA-256
  `fadf8a42991590c79e5e339c0939e8dc4cd8b072586f9f278baf31873a82aeee`.
- Final Task 6 verification: `.venv/bin/python -m pytest -q` passed 51 tests,
  `node --test web/parity.test.mjs` passed seven tests, and Ruff passed.
- `.venv/bin/python -m pytest tests/test_phresh.py -q` before Task 7 implementation:
  failed during collection with `ModuleNotFoundError: No module named 'phishme.phresh'`.
- `.venv/bin/python -m pytest tests/test_phresh.py -q` after Task 7 implementation:
  `11 passed in 0.12s`.
- `.venv/bin/python - <<'PY' ... PY` notebook validation for
  `notebooks/phishme_colab.ipynb`: nbformat validation succeeded and printed `8`.
- `.venv/bin/python -m phishme phresh-smoke --help`: exited 0 and showed
  `--limit`, required `--output`, `--alpha`, `--batch-size`, `--seed`, and
  `--no-resume`.
- Final Task 7 local verification: `.venv/bin/python -m pytest -q` passed
  `62` tests in `6.04s`; `.venv/bin/ruff check .` reported `All checks passed!`;
  `node --test web/parity.test.mjs` passed seven tests; and
  `node --check web/scorer.js && node --check web/parity.test.mjs && git diff --check`
  exited 0.
- Task 7 network-failure hardening red run:
  `.venv/bin/python -m pytest tests/test_phresh.py -q` failed with `11 failed, 17 passed`
  before the classifier/client-reset/cutoff-reconstruction implementation.
- Task 7 network-failure hardening focused green run:
  `.venv/bin/python -m pytest tests/test_phresh.py -q` passed `28 passed in 2.31s`.
- Task 7 network-failure hardening final verification:
  `.venv/bin/python -m pytest -q` passed `79 passed in 8.42s`;
  `.venv/bin/ruff check .` reported `All checks passed!`;
  `node --test web/parity.test.mjs` passed seven tests;
  `node --check web/scorer.js` exited 0;
  `node --check web/parity.test.mjs` exited 0;
  notebook validation printed `nbformat valid: 8 cells`; and
  `git diff --check` exited 0.

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
- The executable pipeline was committed before the real baseline so `run.json` records
  the exact source commit used. The observed baseline was then recorded separately;
  generated `artifacts/` remain ignored.
- Task 7 keeps `datasets` as a lazy optional import inside `phishme.phresh`; base
  imports and tests do not require the cloud extra.
- Task 7 `iter_phresh` yields parsed canonical records without an `html` field.
  Missing URL/label/date rows become reject records for stream counters; non-empty
  unknown labels, unsupported revisions, and schema drift are hard failures.
- Task 7 checkpoints use `checkpoints/checkpoint.json` metadata plus versioned
  `model-<position>-<hash>.joblib` files. A new joblib is fsynced and atomically
  moved before metadata is atomically replaced, so the previous valid metadata keeps
  pointing at a valid old model if metadata replacement fails.
- Task 7 retry classification now covers built-in stream/network exceptions, lazy optional
  `httpx.TransportError`, and only the exact closed Hugging Face client
  `RuntimeError("Cannot send a request, as the client has been closed.")`. Retries close
  the Hugging Face global client before iterator reconstruction; schema, label, generic
  runtime, type, and programming failures remain non-retryable.

## Failed or negative results
- Expected red test run confirmed missing `phishme.data` before implementation.
- Expected Task 6 red test run confirmed missing `phishme.__main__` before CLI implementation.
- Expected Task 7 red test run confirmed missing `phishme.phresh` before implementation.
- The shell `python -m pytest tests/test_pipeline.py -q` command resolved to a different
  virtualenv without pytest; verification used `.venv/bin/python`.
- The host PhreshPhish smoke attempt produced zero artifacts. The pinned cutoff probe failed
  while reading `train-001.parquet` after a read timeout/DNS failure, and Hugging Face
  ultimately raised `RuntimeError: Cannot send a request, as the client has been closed.`
  No PhiUSIIL or other substitute data was used. After the failure, PyArrow shutdown
  deadlocked and the process required termination. This is a host network failure, not an
  interface block; Colab smoke is still required.

## Artifacts
- `src/phishme/data.py`
- `tests/test_data.py`
- `tests/test_splits.py`
- `src/phishme/__main__.py`
- `tests/conftest.py`
- `tests/test_pipeline.py`
- `src/phishme/phresh.py`
- `tests/test_phresh.py`
- `notebooks/phishme_colab.ipynb`

## Next actions
- Colab/networked runtime: run
  `python -m phishme phresh-smoke --limit 1000 --output artifacts/phresh-smoke`
  and record the exact success or failure without substituting PhiUSIIL data.

## Host hardening
- Red: `.venv/bin/python -m pytest tests/test_phresh.py -q` failed with
  `3 failed, 14 passed` before the hardening implementation. The failing tests
  covered checkpoint metadata count consistency, strict-before-cutoff smoke CLI
  helper behavior, and notebook smoke/full train filtering.
- Green: `.venv/bin/python -m pytest tests/test_phresh.py -q` passed
  `17 passed in 0.16s` after implementation.
- Green: `.venv/bin/python -m pytest -q` passed `68 passed in 6.10s`.
- Green: `.venv/bin/ruff check .` reported `All checks passed!`.
- Green: `node --test web/parity.test.mjs` passed seven tests.
- Green: notebook validation for `notebooks/phishme_colab.ipynb` succeeded with
  `nbformat valid: 8 cells`.
- Green: `node --check web/scorer.js`, `node --check web/parity.test.mjs`, and
  `git diff --check` exited 0.
- Host PhreshPhish smoke was attempted but failed during pinned cutoff streaming with the
  `train-001.parquet` timeout/DNS/client-closed sequence described above; Colab smoke and
  full-data training remain pending for a networked/cloud runtime.
- Final Task 7 verification after independent review resolution:
  `.venv/bin/python -m pytest tests/test_phresh.py -q` passed `29` tests,
  `.venv/bin/python -m pytest -q` passed `80` tests, Node passed seven tests,
  Ruff passed, the eight-cell notebook validated with nbformat, JavaScript syntax
  checks passed, and `git diff --check` passed.
- Review finding resolved: production `PhreshTrainConfig` now defaults to a positive
  one-second initial retry delay with a 2x multiplier and at most three attempts;
  retry-focused unit tests explicitly set zero delay to remain fast.
