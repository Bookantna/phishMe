# phishMe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a leakage-aware sparse phishing-classification pipeline, browser scorer, and resumable Colab workflow using PhiUSIIL locally and streamed PhreshPhish in the cloud.

**Architecture:** Source adapters normalize both datasets into a shared frame, deterministic domain/temporal controls produce frozen manifests, and shared FNV-1a hashed character n-grams plus browser-computable numeric features feed incremental logistic regression. Evaluation, JSON export, JavaScript scoring, and cloud streaming reuse the same package contracts so research and browser inference cannot silently drift.

**Tech Stack:** Python 3.9+, NumPy, pandas, SciPy, scikit-learn, tldextract, lxml, Hugging Face `datasets`, pytest, Node.js 20+, vanilla browser JavaScript.

## Global Constraints

- Canonical labels are exactly `1 = phishing`, `0 = benign`.
- Hash space is exactly `2^18`; character n-grams are exactly lengths 3–5.
- Model is `SGDClassifier(loss="log_loss", penalty="l2")` and supports `partial_fit`.
- PhiUSIIL is split by registrable domain; no random row split is permitted.
- PhreshPhish revision is pinned to `eabec4b7a66324b79cc8a0ad856d1731dc26fe1a` and its official test split remains untouched.
- Primary metrics are average precision and validation-threshold F1; accuracy is secondary.
- The threshold is selected on validation data only and frozen before test scoring.
- The browser payload must be no larger than 25 MB and Chrome p95 inference no slower than 250 ms per page.
- No raw datasets, HTML, model checkpoints, generated caches, or large artifacts are committed.
- No external reputation, DNS, WHOIS, certificate, or blocklist feature is allowed.
- No claim of superiority over PhishLang without identical frozen examples and paired uncertainty estimates.
- Keep `METHOD.md` stable and `Implement.md` updated with actual commands, results, failures, and negative findings.

---

## File Map

- `pyproject.toml`: package metadata, runtime dependencies, pytest/ruff configuration.
- `README.md`: installation and verified local/cloud commands.
- `METHOD.md`: frozen scientific protocol and threats to validity.
- `Implement.md`: execution log containing only observed results.
- `src/phishme/data.py`: URL canonicalization, labels, PhiUSIIL loading, domain splits.
- `src/phishme/features.py`: feature names, HTML/URL extraction, FNV hashing, sparse matrices.
- `src/phishme/train.py`: incremental fitting, validation search, checkpointing.
- `src/phishme/evaluate.py`: threshold selection, metrics, base-rate and paired-bootstrap reports.
- `src/phishme/export.py`: compact artifact export/load and Python reference scorer.
- `src/phishme/phresh.py`: pinned Hugging Face streaming, temporal partitioning, resumable batches.
- `src/phishme/__main__.py`: `audit`, `local`, and `phresh-smoke` commands.
- `tests/`: synthetic tests only; no unlicensed dataset rows.
- `web/scorer.js`: JavaScript hash, transform, and logistic scorer.
- `web/parity.test.mjs`: Python/JavaScript golden-vector check.
- `web/benchmark.html`: browser payload and latency benchmark.
- `notebooks/phishme_colab.ipynb`: thin Colab UI around package APIs.
- `scripts/run_phishlang.py`: pinned official-model comparator producing canonical predictions.

---

### Task 1: Package shell and research protocol

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `METHOD.md`
- Create: `Implement.md`
- Create: `src/phishme/__init__.py`
- Create: `tests/test_package.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: importable package version `phishme.__version__ == "0.1.0"`.
- Produces: editable development install via `python -m pip install -e '.[dev,cloud]'`.

- [ ] **Step 1: Write the package smoke test**

```python
# tests/test_package.py
import phishme


def test_version_is_public():
    assert phishme.__version__ == "0.1.0"
```

- [ ] **Step 2: Run the test and confirm the package does not exist yet**

Run: `python -m pytest tests/test_package.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'phishme'`.

- [ ] **Step 3: Add minimal package metadata**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "phishme"
version = "0.1.0"
description = "Leakage-aware lightweight phishing detection research pipeline"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
  "joblib>=1.3,<2",
  "lxml>=5,<7",
  "numpy>=1.26,<3",
  "pandas>=2.1,<3",
  "scikit-learn>=1.4,<2",
  "scipy>=1.11,<2",
  "tldextract>=5,<6",
]

[project.optional-dependencies]
cloud = ["datasets>=3,<5", "nbformat>=5,<6"]
dev = ["pytest>=8,<9", "ruff>=0.12,<1"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py39"
```

```python
# src/phishme/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 4: Write the protocol documents**

`METHOD.md` must freeze these exact items before test evaluation: research question; canonical labels; data provenance and licenses; domain and temporal split rules; duplicate definition; feature version; FNV-1a 32-bit hash; `2^18` buckets; character 3–5 grams; numeric transforms; excluded leakage features; SGD log-loss model; validation-only hyperparameters and threshold; AP/F1 primary metrics; low-base-rate evaluation at 0.05%, 0.1%, 0.5%, 1%, and 5%; paired bootstrap; 25 MB/250 ms gates; limitations.

`Implement.md` starts with these sections and no invented numbers:

```markdown
# phishMe Implementation Log

## Status
- In progress: package and local baseline implementation.
- Blocked: full PhreshPhish training requires a cloud runtime.
- Not run: controlled PhishLang comparison.

## Commands and observed results

## Decisions and deviations

## Failed or negative results

## Artifacts

## Next actions
```

`README.md` links the approved design and `METHOD.md`, states that raw datasets are excluded, and documents `python -m phishme audit`, `python -m phishme local`, `python -m phishme phresh-smoke`, `node --test web/parity.test.mjs`, and opening `web/benchmark.html` in Chrome. Commands that are not yet verified are labeled “planned” until Task 9.

- [ ] **Step 5: Extend ignore rules**

Append exactly:

```gitignore
*.joblib
*.pkl
artifacts/
reports/
runs/
```

- [ ] **Step 6: Install and verify**

Run: `python -m pip install -e '.[dev,cloud]' && python -m pytest tests/test_package.py -q`

Expected: `1 passed`.

- [ ] **Step 7: Commit the independently testable shell**

```bash
git add pyproject.toml README.md METHOD.md Implement.md .gitignore src/phishme/__init__.py tests/test_package.py
git commit -m "chore: initialize phishMe research package"
```

---

### Task 2: Canonical PhiUSIIL adapter and leakage-safe split

**Files:**
- Create: `src/phishme/data.py`
- Create: `tests/test_data.py`
- Create: `tests/test_splits.py`
- Modify: `Implement.md`

**Interfaces:**
- Produces: `canonicalize_url(url: str) -> str`.
- Produces: `registrable_domain(url: str) -> str`.
- Produces: `load_phiusill(path: Path) -> pandas.DataFrame` with `url`, `title`, `label`, `group`, and `sample_id`; Task 3 adds only the explicit shared `dom_*` columns.
- Produces: `split_local(frame: DataFrame, seed: int = 42) -> dict[str, DataFrame]` with keys `train`, `validation`, `test`.

- [ ] **Step 1: Write failing label and duplicate tests**

```python
# tests/test_data.py
from pathlib import Path

import pandas as pd
import pytest

from phishme.data import canonicalize_url, load_phiusill


def test_canonical_url_removes_fragment_and_normalizes_host():
    assert canonicalize_url("HTTPS://Example.COM/#login") == "https://example.com"


def test_phiusill_inverts_labels_and_deduplicates(tmp_path: Path):
    frame = pd.DataFrame({
        "URL": ["https://safe.example/", "https://evil.example/#x", "https://evil.example/"],
        "Title": ["Safe", "Login", "Login"],
        "label": [1, 0, 0],
    })
    path = tmp_path / "tiny.csv"
    frame.to_csv(path, index=False)
    loaded = load_phiusill(path)
    assert loaded["label"].tolist() == [0, 1]
    assert loaded["sample_id"].is_unique


def test_phiusill_rejects_unknown_label(tmp_path: Path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"URL": ["https://x.test"], "Title": ["x"], "label": [7]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown PhiUSIIL labels"):
        load_phiusill(path)
```

- [ ] **Step 2: Write the failing group split test**

```python
# tests/test_splits.py
import pandas as pd

from phishme.data import split_local


def test_split_has_no_sample_or_domain_overlap():
    rows = []
    for domain_id in range(25):
        for page_id in range(2):
            rows.append({
                "url": f"https://d{domain_id}.example/p{page_id}",
                "title": "login",
                "label": domain_id % 2,
                "group": f"d{domain_id}.example",
                "sample_id": f"{domain_id}-{page_id}",
            })
    parts = split_local(pd.DataFrame(rows))
    assert set(parts) == {"train", "validation", "test"}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert set(parts[left].group).isdisjoint(parts[right].group)
        assert set(parts[left].sample_id).isdisjoint(parts[right].sample_id)
    assert all(set(part.label) == {0, 1} for part in parts.values())
```

- [ ] **Step 3: Run tests and confirm missing implementation**

Run: `python -m pytest tests/test_data.py tests/test_splits.py -q`

Expected: import errors for `phishme.data`.

- [ ] **Step 4: Implement canonicalization, labels, domains, and split**

```python
# src/phishme/data.py
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
import tldextract
from sklearn.model_selection import StratifiedGroupKFold

_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())


def canonicalize_url(url: str) -> str:
    raw = str(url).strip()
    if not raw:
        raise ValueError("empty URL")
    parts = urlsplit(raw if "://" in raw else "//" + raw)
    host = (parts.hostname or "").lower()
    if not host:
        raise ValueError(f"URL has no host: {raw!r}")
    port = f":{parts.port}" if parts.port else ""
    path = "" if parts.path in ("", "/") else parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), host + port, path, parts.query, ""))


def registrable_domain(url: str) -> str:
    host = urlsplit(url if "://" in url else "//" + url).hostname or ""
    result = _EXTRACT(host)
    return result.top_domain_under_public_suffix or host.lower()


def _sample_id(url: str) -> str:
    return sha256(canonicalize_url(url).encode("utf-8")).hexdigest()


def load_phiusill(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"URL", "Title", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing PhiUSIIL columns: {missing}")
    labels = set(frame["label"].dropna().astype(int).unique())
    if not labels <= {0, 1}:
        raise ValueError(f"unknown PhiUSIIL labels: {sorted(labels)}")
    out = pd.DataFrame({
        "url": frame["URL"].astype(str),
        "title": frame["Title"].fillna("").astype(str),
        "label": 1 - frame["label"].astype(int),
    })
    out["sample_id"] = out["url"].map(_sample_id)
    out = out.drop_duplicates("sample_id", keep="first").reset_index(drop=True)
    out["group"] = out["url"].map(registrable_domain)
    return out


def split_local(frame: pd.DataFrame, seed: int = 42) -> dict:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold = np.full(len(frame), -1, dtype=np.int8)
    for number, (_, held_out) in enumerate(splitter.split(frame, frame["label"], frame["group"])):
        fold[held_out] = number
    parts = {
        "train": frame.loc[fold >= 2].reset_index(drop=True),
        "validation": frame.loc[fold == 1].reset_index(drop=True),
        "test": frame.loc[fold == 0].reset_index(drop=True),
    }
    for name, part in parts.items():
        if set(part["label"]) != {0, 1}:
            raise ValueError(f"{name} split does not contain both classes")
    names = tuple(parts)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if set(parts[left]["group"]) & set(parts[right]["group"]):
                raise AssertionError(f"domain overlap: {left}/{right}")
            if set(parts[left]["sample_id"]) & set(parts[right]["sample_id"]):
                raise AssertionError(f"sample overlap: {left}/{right}")
    return parts
```

Task 2 deliberately copies no dataset feature columns. Task 3 adds an allowlist; excluded fields therefore cannot enter the matrix by default.

- [ ] **Step 5: Run tests and audit the real CSV**

Run: `python -m pytest tests/test_data.py tests/test_splits.py -q`

Expected: all tests pass.

Run:

```bash
python - <<'PY'
from pathlib import Path
from phishme.data import load_phiusill, split_local
f = load_phiusill(Path('dataset/PhiUSIIL_Phishing_URL_Dataset.csv'))
p = split_local(f)
print({k: {"rows": len(v), "phishing": int(v.label.sum()), "domains": int(v.group.nunique())} for k, v in p.items()})
PY
```

Expected: three non-empty parts, both classes in each, no exception.

- [ ] **Step 6: Record observed counts and commit**

Update `Implement.md` with the actual printed counts and test result, then:

```bash
git add src/phishme/data.py tests/test_data.py tests/test_splits.py Implement.md
git commit -m "feat: add leakage-safe PhiUSIIL adapter"
```

---

### Task 3: Shared feature extraction and deterministic hashing

**Files:**
- Create: `src/phishme/features.py`
- Create: `tests/test_features.py`
- Modify: `src/phishme/data.py`
- Modify: `METHOD.md`

**Interfaces:**
- Produces: `fnv1a_32(text: str) -> int`.
- Produces: `url_numeric_features(url: str) -> dict[str, float]`.
- Produces: `html_features(url: str, html: str) -> tuple[str, dict[str, float]]`.
- Produces: `vectorize(frame: DataFrame, include_dom: bool = True, hash_dim: int = 1 << 18) -> scipy.sparse.csr_matrix`.
- Produces: immutable `FEATURE_VERSION`, `HASH_DIM`, `NGRAM_RANGE`, and `NUMERIC_FEATURES` used by export and JavaScript.

- [ ] **Step 1: Write deterministic hash and transform tests**

```python
# tests/test_features.py
import math

import pandas as pd

from phishme.features import HASH_DIM, fnv1a_32, iter_ngrams, vectorize


def test_fnv1a_matches_published_vector():
    assert fnv1a_32("hello") == 0x4F9F2CAB


def test_ngrams_are_unicode_normalized_and_namespaced():
    grams = set(iter_ngrams("URL", "AbC"))
    assert "url:abc" in grams


def test_vectorization_is_binary_for_duplicate_ngrams():
    frame = pd.DataFrame([{"url": "https://aaaa.example", "title": "", "label": 1}])
    matrix = vectorize(frame, include_dom=False)
    assert matrix.shape == (1, HASH_DIM)
    assert matrix.data.max() == 1.0


def test_numeric_counts_use_log1p():
    frame = pd.DataFrame([{"url": "https://x.example", "title": "", "label": 0, "dom_image_count": 9}])
    matrix = vectorize(frame, include_dom=True)
    assert any(math.isclose(float(value), math.log1p(9), rel_tol=1e-6) for value in matrix.data)
```

- [ ] **Step 2: Run tests and observe missing feature module**

Run: `python -m pytest tests/test_features.py -q`

Expected: import failure for `phishme.features`.

- [ ] **Step 3: Implement the shared feature contract**

```python
# src/phishme/features.py
import math
import unicodedata

import numpy as np
from scipy.sparse import csr_matrix

FEATURE_VERSION = "phishme-features-v1"
HASH_DIM = 1 << 18
NGRAM_RANGE = (3, 5)
COUNT_FEATURES = (
    "url_length", "domain_length", "tld_length", "subdomain_count",
    "letter_count", "digit_count", "obfuscated_count", "query_count",
    "equals_count", "ampersand_count", "special_count", "html_line_count",
    "largest_line_length", "iframe_count", "image_count", "css_count",
    "javascript_count", "self_ref_count", "empty_ref_count", "external_ref_count",
)
RATIO_FEATURES = ("letter_ratio", "digit_ratio", "obfuscated_ratio", "special_ratio")
BOOLEAN_FEATURES = (
    "is_domain_ip", "is_https", "has_title", "has_favicon", "is_responsive",
    "has_description", "external_form_submit", "has_social", "has_submit_button",
    "has_hidden_fields", "has_password_field", "mentions_bank", "mentions_pay",
    "mentions_crypto", "has_copyright",
)
NUMERIC_FEATURES = COUNT_FEATURES + RATIO_FEATURES + BOOLEAN_FEATURES


def fnv1a_32(text: str) -> int:
    value = 0x811C9DC5
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def iter_ngrams(namespace: str, text: str):
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    for size in range(NGRAM_RANGE[0], NGRAM_RANGE[1] + 1):
        for start in range(max(0, len(normalized) - size + 1)):
            yield f"{namespace.lower()}:{normalized[start:start + size]}"


def _transform(name: str, value: float) -> float:
    value = float(value or 0.0)
    if name in COUNT_FEATURES:
        return math.log1p(max(0.0, value))
    if name in RATIO_FEATURES:
        return min(1.0, max(0.0, value))
    return 1.0 if value else 0.0


def vectorize(frame, include_dom=True, hash_dim=HASH_DIM):
    rows, columns, values = [], [], []
    for row_number, row in enumerate(frame.to_dict("records")):
        hashed = {fnv1a_32(token) % hash_dim for token in iter_ngrams("url", row["url"])}
        hashed.update(fnv1a_32(token) % hash_dim for token in iter_ngrams("title", row.get("title", "")))
        for column in sorted(hashed):
            rows.append(row_number); columns.append(column); values.append(1.0)
        if include_dom:
            for offset, name in enumerate(NUMERIC_FEATURES):
                value = _transform(name, row.get("dom_" + name, 0.0))
                if value:
                    rows.append(row_number); columns.append(hash_dim + offset); values.append(value)
    width = hash_dim + (len(NUMERIC_FEATURES) if include_dom else 0)
    return csr_matrix((np.asarray(values, dtype=np.float32), (rows, columns)), shape=(len(frame), width))
```

`url_numeric_features` uses `urlsplit` and `ipaddress.ip_address`, counts `%HH` sequences as obfuscation, and derives every URL count/ratio from the raw URL. `html_features` uses tolerant `lxml.html.fromstring`, resolves links with `urllib.parse.urljoin`, and classifies references by registrable domain. It counts only the exact `NUMERIC_FEATURES` above. Empty or malformed HTML returns an empty title and zero DOM values plus status `empty` or `parse_error`; callers aggregate those statuses.

- [ ] **Step 4: Add the allowlisted local source map**

In `data.py`, define exactly this allowlist and copy these values before duplicate removal so row alignment remains intact:

```python
PHIUSIIL_DOM_MAP = {
    "LineOfCode": "html_line_count",
    "LargestLineLength": "largest_line_length",
    "HasTitle": "has_title",
    "HasFavicon": "has_favicon",
    "IsResponsive": "is_responsive",
    "HasDescription": "has_description",
    "NoOfiFrame": "iframe_count",
    "HasExternalFormSubmit": "external_form_submit",
    "HasSocialNet": "has_social",
    "HasSubmitButton": "has_submit_button",
    "HasHiddenFields": "has_hidden_fields",
    "HasPasswordField": "has_password_field",
    "Bank": "mentions_bank",
    "Pay": "mentions_pay",
    "Crypto": "mentions_crypto",
    "HasCopyrightInfo": "has_copyright",
    "NoOfImage": "image_count",
    "NoOfCSS": "css_count",
    "NoOfJS": "javascript_count",
    "NoOfSelfRef": "self_ref_count",
    "NoOfEmptyRef": "empty_ref_count",
    "NoOfExternalRef": "external_ref_count",
}
```

Recompute all URL-derived features with `url_numeric_features`. Add a test asserting none of `URLSimilarityIndex`, `TLDLegitimateProb`, `CharContinuationRate`, `URLCharProb`, `DomainTitleMatchScore`, `URLTitleMatchScore`, `Robots`, redirect counts, popup counts, `FILENAME`, or the source label appears in loaded feature columns.

- [ ] **Step 5: Freeze feature definitions and run tests**

Update `METHOD.md` with the exact tuples and FNV byte algorithm above.

Run: `python -m pytest tests/test_features.py tests/test_data.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/phishme/features.py src/phishme/data.py tests/test_features.py tests/test_data.py METHOD.md
git commit -m "feat: add browser-parity feature hashing"
```

---

### Task 4: Incremental training, thresholding, and scientific metrics

**Files:**
- Create: `src/phishme/train.py`
- Create: `src/phishme/evaluate.py`
- Create: `tests/test_train.py`
- Create: `tests/test_evaluate.py`

**Interfaces:**
- Produces: `TrainConfig(alpha: float, epochs: int, batch_size: int, seed: int)`.
- Produces: `fit_incremental(frame, config, include_dom) -> SGDClassifier`.
- Produces: `predict_scores(model, frame, include_dom, batch_size) -> ndarray`.
- Produces: `select_threshold(y_true, scores) -> float`.
- Produces: `metrics_report(y_true, scores, threshold) -> dict`.
- Produces: `paired_bootstrap(y, left, right, left_threshold, right_threshold, seed, resamples) -> dict`.

- [ ] **Step 1: Write failing evaluation tests**

```python
# tests/test_evaluate.py
import numpy as np

from phishme.evaluate import metrics_report, paired_bootstrap, select_threshold


def test_threshold_uses_validation_scores():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.05, 0.2, 0.7, 0.9])
    threshold = select_threshold(y, scores)
    assert 0.2 < threshold <= 0.7


def test_report_contains_required_metrics():
    report = metrics_report(np.array([0, 1]), np.array([0.1, 0.9]), 0.5)
    assert {"average_precision", "f1", "accuracy", "precision", "recall", "roc_auc", "false_positive_rate", "brier", "confusion_matrix"} <= set(report)


def test_paired_bootstrap_is_deterministic():
    y = np.array([0, 0, 1, 1])
    left = np.array([0.1, 0.2, 0.8, 0.9])
    right = np.array([0.2, 0.3, 0.7, 0.8])
    assert paired_bootstrap(y, left, right, 0.5, 0.5, 7, 20) == paired_bootstrap(y, left, right, 0.5, 0.5, 7, 20)
```

- [ ] **Step 2: Write the incremental fit test**

```python
# tests/test_train.py
import pandas as pd

from phishme.train import TrainConfig, fit_incremental, predict_scores


def test_incremental_model_learns_tiny_signal():
    frame = pd.DataFrame([
        {"url": "https://safe.example/home", "title": "welcome", "label": 0},
        {"url": "https://safe.example/about", "title": "about", "label": 0},
        {"url": "http://steal.example/login", "title": "verify password", "label": 1},
        {"url": "http://steal.example/account", "title": "urgent login", "label": 1},
    ] * 8)
    model = fit_incremental(frame, TrainConfig(alpha=1e-4, epochs=5, batch_size=8, seed=42), include_dom=False)
    scores = predict_scores(model, frame, include_dom=False, batch_size=8)
    assert scores[frame.label.to_numpy() == 1].mean() > scores[frame.label.to_numpy() == 0].mean()
```

- [ ] **Step 3: Run and observe missing modules**

Run: `python -m pytest tests/test_train.py tests/test_evaluate.py -q`

Expected: import errors.

- [ ] **Step 4: Implement incremental fitting and prediction**

```python
# src/phishme/train.py
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import SGDClassifier

from .features import vectorize


@dataclass(frozen=True)
class TrainConfig:
    alpha: float = 1e-4
    epochs: int = 3
    batch_size: int = 2048
    seed: int = 42


def _batches(frame, size):
    for start in range(0, len(frame), size):
        yield frame.iloc[start:start + size]


def fit_incremental(frame, config, include_dom):
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=config.alpha, random_state=config.seed)
    rng = np.random.default_rng(config.seed)
    first = True
    for _ in range(config.epochs):
        order = rng.permutation(len(frame))
        shuffled = frame.iloc[order]
        for batch in _batches(shuffled, config.batch_size):
            kwargs = {"classes": np.array([0, 1])} if first else {}
            model.partial_fit(vectorize(batch, include_dom=include_dom), batch["label"].to_numpy(), **kwargs)
            first = False
    return model


def predict_scores(model, frame, include_dom, batch_size=4096):
    chunks = [model.predict_proba(vectorize(batch, include_dom=include_dom))[:, 1] for batch in _batches(frame, batch_size)]
    return np.concatenate(chunks) if chunks else np.empty(0)
```

Add `atomic_checkpoint(model, metadata, path)` using `joblib.dump` to `path.with_suffix(path.suffix + ".tmp")`, `os.replace`, and strict `FEATURE_VERSION`/dataset revision validation on load.

- [ ] **Step 5: Implement metrics and bootstrap**

Use scikit-learn’s `precision_recall_curve`, `average_precision_score`, `f1_score`, `accuracy_score`, `precision_score`, `recall_score`, `roc_auc_score`, `brier_score_loss`, and `confusion_matrix`. `select_threshold` computes F1 for every returned threshold, uses `np.nanargmax`, and returns the first maximum. `paired_bootstrap` samples the same row indices for both models and returns 2.5/50/97.5 percentiles for AP and F1 differences.

- [ ] **Step 6: Verify and commit**

Run: `python -m pytest tests/test_train.py tests/test_evaluate.py -q`

Expected: all tests pass.

```bash
git add src/phishme/train.py src/phishme/evaluate.py tests/test_train.py tests/test_evaluate.py
git commit -m "feat: add incremental training and evaluation"
```

---

### Task 5: Compact export and JavaScript parity

**Files:**
- Create: `src/phishme/export.py`
- Create: `tests/test_export.py`
- Create: `web/scorer.js`
- Create: `web/parity.test.mjs`
- Create: `web/benchmark.html`

**Interfaces:**
- Produces: `export_model(model, threshold, include_dom, metadata, path) -> Path`.
- Produces: `load_model(path) -> dict`.
- Produces: `score_record(artifact, record) -> float` reference probability.
- Produces JS: `fnv1a32`, `vectorizeRecord`, `scoreRecord`, `extractDomFeatures`.

- [ ] **Step 1: Write the Python round-trip test**

```python
# tests/test_export.py
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier

from phishme.export import export_model, load_model


def test_export_is_compact_and_float32(tmp_path: Path):
    model = SGDClassifier(loss="log_loss")
    model.classes_ = np.array([0, 1])
    model.coef_ = np.array([[0.25, -0.5]], dtype=np.float64)
    model.intercept_ = np.array([0.1])
    path = export_model(model, 0.6, False, {"dataset_revision": "test"}, tmp_path / "model.json", hash_dim=2)
    payload = load_model(path)
    assert payload["threshold"] == 0.6
    assert payload["weights"] == [0.25, -0.5]
    assert json.loads(path.read_text())["feature_version"]
```

- [ ] **Step 2: Implement compact JSON export**

The artifact schema is exactly:

```json
{
  "schema": "phishme-model-v1",
  "feature_version": "phishme-features-v1",
  "hash": {"name": "fnv1a-32", "dimension": 262144, "ngram_min": 3, "ngram_max": 5},
  "include_dom": true,
  "numeric_features": [],
  "weights": [],
  "intercept": 0.0,
  "threshold": 0.5,
  "metadata": {}
}
```

Convert coefficients and intercept through `np.float32`, reject dimension mismatches, serialize with `json.dumps(payload, separators=(",", ":"), sort_keys=True)`, write through a temporary file, then `os.replace`.

- [ ] **Step 3: Implement the JavaScript scorer with the identical hash**

```javascript
// web/scorer.js
export function fnv1a32(text) {
  let value = 0x811c9dc5;
  for (const byte of new TextEncoder().encode(text)) {
    value ^= byte;
    value = Math.imul(value, 0x01000193) >>> 0;
  }
  return value >>> 0;
}

export function sigmoid(value) {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exp = Math.exp(value);
  return exp / (1 + exp);
}

const COUNT_FEATURES = new Set([
  "url_length", "domain_length", "tld_length", "subdomain_count", "letter_count",
  "digit_count", "obfuscated_count", "query_count", "equals_count", "ampersand_count",
  "special_count", "html_line_count", "largest_line_length", "iframe_count", "image_count",
  "css_count", "javascript_count", "self_ref_count", "empty_ref_count", "external_ref_count",
]);
const RATIO_FEATURES = new Set(["letter_ratio", "digit_ratio", "obfuscated_ratio", "special_ratio"]);

function transformNumeric(name, raw) {
  const value = Number(raw || 0);
  if (COUNT_FEATURES.has(name)) return Math.log1p(Math.max(0, value));
  if (RATIO_FEATURES.has(name)) return Math.min(1, Math.max(0, value));
  return value ? 1 : 0;
}

export function scoreRecord(model, record) {
  if (model.schema !== "phishme-model-v1") throw new Error("unsupported model schema");
  const seen = new Set();
  for (const [namespace, raw] of [["url", record.url], ["title", record.title || ""]]) {
    const characters = Array.from(String(raw).normalize("NFKC").toLowerCase());
    for (let size = model.hash.ngram_min; size <= model.hash.ngram_max; size += 1) {
      for (let start = 0; start + size <= characters.length; start += 1) {
        seen.add(fnv1a32(`${namespace}:${characters.slice(start, start + size).join("")}`) % model.hash.dimension);
      }
    }
  }
  let logit = model.intercept;
  for (const index of seen) logit += model.weights[index];
  if (model.include_dom) {
    model.numeric_features.forEach((name, offset) => {
      logit += model.weights[model.hash.dimension + offset] * transformNumeric(name, record.dom?.[name]);
    });
  }
  const probability = sigmoid(logit);
  return {probability, label: Number(probability >= model.threshold), featureVersion: model.feature_version};
}
```

`extractDomFeatures(document, location.href)` derives the frozen DOM counts from `querySelectorAll`, document text, and resolved element URLs; it emits no key outside `model.numeric_features` and performs no network requests. URL lexical values are computed by the same named helper used by `scoreRecord`, not accepted from arbitrary page data.

- [ ] **Step 4: Add cross-language golden-vector test**

`tests/test_export.py` writes a deterministic tiny model and record to a temporary JSON fixture and invokes:

```bash
node web/parity.test.mjs /absolute/model.json /absolute/record.json /absolute/python-score.txt
```

`web/parity.test.mjs` imports `scoreRecord`, loads all three files, and asserts `Math.abs(jsProbability - pythonProbability) <= 1e-6` plus `fnv1a32("hello") === 0x4f9f2cab`.

- [ ] **Step 5: Add the browser benchmark**

`web/benchmark.html` loads a selected artifact via file input, runs 20 warmups and 200 measured scores over fixed synthetic records, sorts durations, and displays JSON with browser user agent, payload bytes, sample count, p50, p95, maximum, and booleans for `bytes <= 25 * 1024 * 1024` and `p95 <= 250`.

- [ ] **Step 6: Verify and commit**

Run: `python -m pytest tests/test_export.py -q && node --test web/parity.test.mjs`

Expected: Python export tests and Node static hash tests pass.

```bash
git add src/phishme/export.py tests/test_export.py web/scorer.js web/parity.test.mjs web/benchmark.html
git commit -m "feat: export browser-compatible phishing scorer"
```

---

### Task 6: End-to-end local experiment command

**Files:**
- Create: `src/phishme/__main__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_pipeline.py`
- Modify: `README.md`
- Modify: `Implement.md`

**Interfaces:**
- Produces CLI: `python -m phishme audit --csv PATH`.
- Produces CLI: `python -m phishme local --csv PATH --output DIR`.
- Produces: `DIR/run.json`, `DIR/splits.json`, `DIR/model.json`, and `DIR/test-metrics.json`.

- [ ] **Step 1: Write a failing CLI integration test**

Create the synthetic fixture, then the integration test:

```python
# tests/conftest.py
import pandas as pd
import pytest


@pytest.fixture
def synthetic_phiusill_csv(tmp_path):
    rows = []
    for domain_id in range(25):
        for page_id in range(2):
            phishing = domain_id % 2 == 1
            rows.append({
                "URL": f"http{'s' if not phishing else ''}://d{domain_id}.example/p{page_id}",
                "Title": "verify password" if phishing else "welcome",
                "label": 0 if phishing else 1,
            })
    path = tmp_path / "synthetic.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
```

```python
# tests/test_pipeline.py
import json
import subprocess
import sys


def test_local_cli_writes_reproducible_artifacts(synthetic_phiusill_csv, tmp_path):
    output = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, "-m", "phishme", "local", "--csv", str(synthetic_phiusill_csv), "--output", str(output), "--epochs", "1"],
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "model.json").exists()
    metrics = json.loads((output / "test-metrics.json").read_text())
    assert "average_precision" in metrics
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["test_scored_once"] is True
```

- [ ] **Step 2: Implement the command flow**

The `local` command must:

1. load and split once;
2. write split counts and SHA-256 sample-manifest hashes;
3. train six validation candidates: DOM off/on crossed with alpha `1e-5`, `1e-4`, `1e-3`;
4. select each candidate’s threshold on validation only;
5. rank candidates with `max(key=(validation_ap, validation_f1, -alpha, -int(include_dom)))`, so ties prefer smaller alpha and then URL-only;
6. score the untouched test partition exactly once with the selected candidate;
7. export model and compact JSON reports atomically;
8. record Python/dependency versions, Git commit, seed, feature version, dataset file SHA-256, arguments, counts, failures, and artifact hashes.

Use `argparse`; do not add a CLI framework.

- [ ] **Step 3: Verify the synthetic integration**

Run: `python -m pytest tests/test_pipeline.py -q`

Expected: pass and all four expected artifacts exist.

- [ ] **Step 4: Execute the real local baseline**

Run:

```bash
python -m phishme audit --csv dataset/PhiUSIIL_Phishing_URL_Dataset.csv
python -m phishme local --csv dataset/PhiUSIIL_Phishing_URL_Dataset.csv --output artifacts/local-v1
python -m pytest -q
node --test web/parity.test.mjs
```

Expected: commands exit 0; `artifacts/local-v1/test-metrics.json` contains all required metrics; `model.json` is at most 25 MB. Do not invent expected metric values.

- [ ] **Step 5: Record actual baseline and update docs**

Copy only actual command outputs, selected hyperparameters, split counts, AP/F1/accuracy, artifact bytes, failures, and hashes into `Implement.md`. Change README local commands from “planned” to “verified” only if all commands pass.

- [ ] **Step 6: Commit source and compact summary only**

Do not add `artifacts/`. Commit code/docs:

```bash
git add src/phishme/__main__.py tests/conftest.py tests/test_pipeline.py README.md Implement.md
git commit -m "feat: run reproducible local phishing baseline"
```

---

### Task 7: Pinned PhreshPhish streaming and resumable Colab workflow

**Files:**
- Create: `src/phishme/phresh.py`
- Create: `tests/test_phresh.py`
- Create: `notebooks/phishme_colab.ipynb`
- Modify: `src/phishme/__main__.py`
- Modify: `README.md`
- Modify: `Implement.md`

**Interfaces:**
- Produces: `iter_phresh(split, revision, limit=None) -> Iterator[dict]`.
- Produces: `derive_temporal_cutoff(revision) -> str` using projected date/label columns.
- Produces: `train_stream(records, checkpoint_dir, config, resume) -> SGDClassifier`.
- Produces CLI: `python -m phishme phresh-smoke --limit 1000 --output artifacts/phresh-smoke`.

- [ ] **Step 1: Write a failing stream/checkpoint test**

```python
# tests/test_phresh.py
from pathlib import Path

import pytest

from phishme.phresh import PHRESH_REVISION, validate_checkpoint


def test_revision_is_pinned():
    assert PHRESH_REVISION == "eabec4b7a66324b79cc8a0ad856d1731dc26fe1a"


def test_checkpoint_rejects_feature_mismatch(tmp_path: Path):
    path = tmp_path / "checkpoint.json"
    path.write_text('{"feature_version":"wrong","dataset_revision":"eabec4b7a66324b79cc8a0ad856d1731dc26fe1a"}')
    with pytest.raises(ValueError, match="feature version"):
        validate_checkpoint(path)
```

- [ ] **Step 2: Implement pinned streaming**

Use:

```python
from datasets import load_dataset

PHRESH_DATASET = "phreshphish/phreshphish"
PHRESH_REVISION = "eabec4b7a66324b79cc8a0ad856d1731dc26fe1a"


def iter_phresh(split, revision=PHRESH_REVISION, limit=None):
    if revision != PHRESH_REVISION:
        raise ValueError("unsupported PhreshPhish revision")
    stream = load_dataset(PHRESH_DATASET, split=split, revision=revision, streaming=True)
    for index, row in enumerate(stream):
        if limit is not None and index >= limit:
            break
        label = {"benign": 0, "phish": 1}.get(str(row["label"]).lower())
        if label is None:
            raise ValueError(f"unknown PhreshPhish label: {row['label']!r}")
        yield {"url": row["url"], "html": row.get("html", ""), "date": row.get("date"), "label": label}
```

Validate required columns `url`, `html`, `date`, and `label` on the first row and fail with the available-column list if the published schema changed. Derive the chronological 90th-percentile validation cutoff from `stream.select_columns(["date", "label"])`, parse dates as UTC, and store the resulting ISO-8601 cutoff in the run manifest so raw HTML is not transferred for the cutoff pass.

- [ ] **Step 3: Implement resumable streaming batches**

Every checkpoint contains processed position, model joblib path and SHA-256, feature version, dataset revision, split, cutoff, alpha, batch size, class counts, parse/reject counts, seed, and write time. Write model and metadata to temporary files, fsync, and atomically replace. Resume only if every invariant matches command arguments. Use a bounded three-attempt exponential retry around iterator reconstruction; after the third failure stop and preserve the last valid checkpoint.

- [ ] **Step 4: Create the thin notebook**

The notebook contains cells that:

1. select `MODE = "smoke"` or `"full"` and mount Drive;
2. clone `https://github.com/Bookantna/phishMe`, check out a displayed commit, and install `.[cloud]`;
3. print dataset revision and cutoff;
4. call package streaming/training APIs with a Drive checkpoint directory;
5. evaluate the untouched official test stream;
6. export model, reports, manifests, and hashes to Drive;
7. print exact next commands and limitations.

No feature, model, or metric implementation is duplicated in notebook cells.

- [ ] **Step 5: Run tests and smoke mode**

Run: `python -m pytest tests/test_phresh.py -q`.

Then run `python -m phishme phresh-smoke --limit 1000 --output artifacts/phresh-smoke` on a networked runtime. Expected: 1,000 or fewer accepted examples, at least one checkpoint, and an exported smoke artifact. If local bandwidth blocks it, record the failure exactly and run the same command in Colab; do not substitute local PhiUSIIL data.

- [ ] **Step 6: Validate notebook syntax and commit**

Run:

```bash
python - <<'PY'
import nbformat
nb = nbformat.read('notebooks/phishme_colab.ipynb', as_version=4)
nbformat.validate(nb)
print(len(nb.cells))
PY
```

Expected: validation succeeds and cell count is non-zero.

```bash
git add src/phishme/phresh.py src/phishme/__main__.py tests/test_phresh.py notebooks/phishme_colab.ipynb README.md Implement.md
git commit -m "feat: add resumable PhreshPhish Colab pipeline"
```

---

### Task 8: Low-base-rate benchmarks and controlled PhishLang adapter

**Files:**
- Create: `src/phishme/benchmark.py`
- Create: `tests/test_benchmark.py`
- Create: `scripts/run_phishlang.py`
- Modify: `METHOD.md`
- Modify: `Implement.md`

**Interfaces:**
- Produces: `base_rate_manifest(labels, prevalence, seed) -> ndarray`.
- Produces: `evaluate_base_rates(labels, scores, threshold, rates, seed) -> dict`.
- Produces script output CSV columns exactly `sample_id,label,score,model,source_commit`.

- [ ] **Step 1: Write deterministic prevalence tests**

```python
# tests/test_benchmark.py
import numpy as np

from phishme.benchmark import base_rate_manifest


def test_base_rate_manifest_is_deterministic_and_close():
    labels = np.array([0] * 10000 + [1] * 1000)
    left = base_rate_manifest(labels, prevalence=0.01, seed=42)
    right = base_rate_manifest(labels, prevalence=0.01, seed=42)
    assert np.array_equal(left, right)
    assert abs(labels[left].mean() - 0.01) < 0.001
```

- [ ] **Step 2: Implement base-rate manifests**

Use all available negatives and sample `round(prevalence * negatives / (1 - prevalence))` positives without replacement. If positives are insufficient, reduce negatives deterministically rather than duplicate samples. Store sample IDs and SHA-256 of each manifest for 0.05%, 0.1%, 0.5%, 1%, and 5%.

- [ ] **Step 3: Implement the official PhishLang runner**

Pin source using:

```bash
git clone https://github.com/UTA-SPRLab/phishlang.git external/phishlang
git -C external/phishlang rev-parse HEAD
```

`external/` remains ignored. `scripts/run_phishlang.py` imports the official repository’s `generate_text_representation`, loads `external/phishlang/src/model` with `MobileBertTokenizer` and `MobileBertForSequenceClassification`, scores exactly the frozen manifest’s HTML in batches, and emits canonical phishing probabilities. It records the exact Git commit and model SHA-256. It must not install or execute the `.deb` package.

Because the official published prediction helper can assign zero probability to token sequences shorter than its 128-token window, the adapter must preserve official behavior for the main comparison and may report a separately labeled corrected short-input sensitivity analysis. Never merge those two result sets.

- [ ] **Step 4: Test adapter contracts without downloading the 98 MB model**

Use a fake tokenizer/model to assert output columns, sample order, and probability mapping. Mark a live official-model smoke test with `pytest.mark.integration` and skip unless `PHISHLANG_DIR` exists.

- [ ] **Step 5: Run the controlled comparison in cloud**

On the frozen PhreshPhish manifest, run phishMe and official PhishLang, then call `paired_bootstrap` with the same sample order. Save compact prediction/report files outside Git and add only hashes, metrics, confidence intervals, source commit, model hash, and limitations to `Implement.md`.

No superiority sentence is allowed unless both AP and F1 are higher, paired intervals support the claim, and phishMe passes payload/latency gates.

- [ ] **Step 6: Verify and commit**

Run: `python -m pytest tests/test_benchmark.py -q`.

```bash
git add src/phishme/benchmark.py tests/test_benchmark.py scripts/run_phishlang.py METHOD.md Implement.md .gitignore
git commit -m "feat: add fair PhishLang comparison protocol"
```

---

### Task 9: Full verification, documentation truth pass, and publication

**Files:**
- Modify: `README.md`
- Modify: `METHOD.md`
- Modify: `Implement.md`

**Interfaces:**
- Consumes all prior tasks.
- Produces a clean, reproducible repository with verified local evidence and explicit cloud-only remaining work.

- [ ] **Step 1: Run all local quality gates**

```bash
python -m ruff check src tests scripts
python -m pytest -q
node --test web/parity.test.mjs
python -m phishme audit --csv dataset/PhiUSIIL_Phishing_URL_Dataset.csv
python -m phishme local --csv dataset/PhiUSIIL_Phishing_URL_Dataset.csv --output artifacts/local-v1
```

Expected: lint and tests pass; audit/local exit 0; generated artifact and report hashes are stable on a repeated run with the same environment and seed.

- [ ] **Step 2: Measure artifact size and browser parity**

Run `wc -c artifacts/local-v1/model.json` and record the observed byte count. Open `web/benchmark.html` in Chrome, load the artifact, run the benchmark, and save the JSON result outside Git. Record browser version, platform, payload bytes, p50, p95, max, and pass/fail in `Implement.md`.

- [ ] **Step 3: Truth-check documentation**

Replace every “planned” marker in README only when the corresponding command actually passed. Keep full PhreshPhish and PhishLang work explicitly “not run” until cloud outputs exist. Ensure `METHOD.md` describes the protocol, while `Implement.md` contains all deviations and observed values. Search for unsupported claims:

```bash
python - <<'PY'
from pathlib import Path
terms = ("better than", "outperform", "state of the art", "production ready", "planned")
for path in map(Path, ("README.md", "METHOD.md", "Implement.md")):
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if any(term in line.lower() for term in terms):
            print(f"{path}:{number}:{line}")
PY
```

Expected: no unsupported superiority or readiness claim; remaining planned/cloud work is clearly labeled.

- [ ] **Step 4: Review Git hygiene**

```bash
git status --short
git check-ignore dataset/PhiUSIIL_Phishing_URL_Dataset.csv paper/*.pdf artifacts/local-v1/model.json
python - <<'PY'
import re
import subprocess
tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
for path in tracked:
    if re.search(r"^(dataset|paper|artifacts)/|\.(joblib|pkl)$", path):
        print(path)
PY
```

Expected: local data/artifacts are ignored and the final command prints nothing.

- [ ] **Step 5: Commit the verified documentation**

```bash
git add README.md METHOD.md Implement.md
git commit -m "docs: record verified phishMe baseline"
```

- [ ] **Step 6: Push only after reviewing the diff**

```bash
git diff origin/main...HEAD --stat
git status --short --branch
git push origin main
```

Expected: push succeeds; local `main` equals `origin/main`; GitHub contains code/docs but no raw dataset or generated model.
