# phishMe Method

## Research Question

Can phishMe variants — a regularized sparse linear classifier (SGD log-loss),
a LightGBM gradient-boosted tree, and an out-of-fold stacking hybrid — trained
on the PhishPedia 30k benchmark generalize to the frozen PhreshPhish test
split, and how do they compare against the official pretrained PhishLang
MobileBERT reference on identical frozen sample IDs under paired bootstrap
uncertainty and browser deployment constraints?

Any claim of improvement over PhishLang requires identical frozen PhreshPhish
test samples, paired bootstrap 95% confidence intervals, and satisfaction of the
browser size (≤ 25 MB) and latency (p95 ≤ 250 ms) gates.

## Canonical Labels

Canonical labels are fixed as `1 = phishing` and `0 = benign`. Source adapters
must normalize source-specific conventions into this representation and reject
unknown labels.

## Data Provenance and Licenses

PhishPedia is the V3 training source, licensed CC0-1.0. The published benchmark
is obtained from the official Google Drive link
`https://drive.google.com/file/d/12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g/view`
(referenced from `github.com/lindsey98/Phishpedia`). It contains a split CSV
(`train_test_val_split_30.csv`), a phishing HTML directory, and a benign HTML
directory. The CSV columns include `file_name`, `url`, `label`, and optionally
`type` and `split`. Labels are normalized from either the numeric `label` column
or the `type` column. Unknown labels and missing HTML files are rejected.

PhreshPhish is the V3 frozen cross-dataset test set, used via its Hugging Face
dataset source, pinned to revision
`eabec4b7a66324b79cc8a0ad856d1731dc26fe1a`. Its dataset card lists CC BY 4.0
and restricts use to anti-phishing research. Raw HTML is parsed inside the
runtime and discarded rather than committed. PhreshPhish test examples are
never used for training, threshold selection, or hyperparameter tuning.

Dataset source, collection date, split membership, and labels disguised as
metadata are never model features.

## Split Rules

PhishPedia uses its published official train/validation/test split from
`train_test_val_split_30.csv` as-is. The CSV is never re-split by the project.
Registrable-domain grouping is used only for duplicate and overlap accounting,
not for re-partitioning.

PhreshPhish preserves the official test split as untouched test data. The
official training split is ordered by collection date; its final temporal portion
is reserved for validation, and earlier examples are used for training. Test
examples are never moved into training or validation.

In V3, the PhreshPhish test split is the sole cross-dataset test set. It is
materialized once into a frozen JSONL of canonical records. Threshold selection
for all phishMe variants occurs exclusively on PhishPedia validation data.
PhishLang uses the official fixed threshold `0.5` without tuning. PhreshPhish
scores and metrics are never used for threshold or hyperparameter selection.

PhreshPhish train-split rows are never used in V3; the training source is
PhishPedia only.

## Duplicate Definition

Duplicate detection uses a canonicalized URL: scheme and host casing are
normalized, fragments are removed, and a trailing root slash is normalized. The
original URL remains the feature input. Exact duplicate samples are removed
before splitting, and removal counts are recorded.

## Feature Contract

The initial feature version is `phishme-features-v1`.

Text features are URL and title character grams of lengths 3, 4, and 5. Text is
Unicode-normalized and lowercased; URL punctuation is preserved. Tokens are
namespaced by source, such as `url:` and `title:`, and binary presence is used
instead of raw frequency.

Feature hashing uses FNV-1a 32-bit over UTF-8 bytes with offset basis
`0x811C9DC5`, prime `0x01000193`, and arithmetic modulo `2^32`. Text features
map into exactly `2^18` buckets with non-negative values.

URL numeric features are computed over the raw URL string as Unicode code
points, without NFKC normalization. `url_length`, `domain_length`, and
`tld_length` are code-point lengths. `letter_count` counts code points whose
Unicode General_Category starts with `L`; `digit_count` counts code points whose
General_Category starts with `N`; `special_count` counts code points whose
General_Category starts with neither `L` nor `N`. Browser scoring uses
ECMAScript Unicode property escapes `\p{L}` and `\p{N}` with the `u` flag;
Python mirrors this with `unicodedata.category`.

Feature-side URL authority parsing is frozen, no-network, and deliberately not
the Python `urllib.parse` or browser WHATWG parser. The scheme is recognized
only from the first literal `://`. If there is no literal scheme and the input
starts with `//`, those two slashes are skipped for protocol-relative input;
otherwise authority parsing starts at the first code point. The authority ends
at the first literal `/`, `?`, or `#`. The last `@` removes userinfo. A
bracketed host starts with `[` and uses the content through the first `]`, or
the remaining content when `]` is absent. An unbracketed host always strips the
final ASCII colon component, regardless of whether the text after the colon is
numeric. No Unicode normalization, IDNA conversion, percent decoding, network
lookup, or validity rejection is applied.

Feature-side IP literal classification uses the host text emitted by that
authority parser. IPv4 is true only for exactly four ASCII decimal components
separated by `.`, each `0` or a non-zero digit followed by up to two digits,
with integer value no greater than 255. IPv6 is true only for ASCII hex, `.`,
and `:` host text containing `:`; it allows at most one `::`, requires every
explicit hex group to be 1-4 hex digits, requires exactly eight groups without
`::`, and requires fewer than eight explicit groups with `::`. A final dotted
IPv4 tail is allowed only after the final `:` and counts as two IPv6 groups
after satisfying the same IPv4 rule. Zone IDs, IPvFuture, non-ASCII digits, and
other address forms are not IP literals.

Feature-side suffix handling is frozen and deliberately browser-computable. It
does not call `tldextract`, fetch the Public Suffix List, or use reputation or
network data. Hosts are lowercased, split on `.`, and empty labels are dropped.
Empty hosts and IP literals have suffix `""` and subdomain count `0`. Otherwise,
if at least one label precedes the longest exact trailing suffix in
`FROZEN_MULTI_LABEL_SUFFIXES`, that suffix is used and
`subdomain_count = max(0, label_count - suffix_label_count - 1)`. If no frozen
multi-label suffix matches, the final host label is the suffix; known single
label suffixes use `max(0, label_count - 2)`, and unknown single-label suffixes
use `max(0, label_count - 1)`.

The frozen suffix catalogs for `phishme-features-v1` are:

```python
FROZEN_SINGLE_LABEL_SUFFIXES = (
    "ai", "app", "au", "biz", "br", "cn", "co", "com", "dev",
    "edu", "gov", "info", "int", "io", "jp", "kr", "mil", "mx",
    "net", "nz", "org", "pl", "sa", "sg", "th", "tr", "uk", "us",
    "za",
)
FROZEN_MULTI_LABEL_SUFFIXES = (
    "ac.th", "ac.uk", "co.jp", "co.kr", "co.nz", "co.th", "co.uk",
    "co.za", "com.au", "com.br", "com.cn", "com.mx", "com.pl",
    "com.sa", "com.sg", "com.tr", "edu.au", "go.th", "gov.au",
    "gov.uk", "ltd.uk", "me.uk", "ne.jp", "net.au", "net.nz",
    "or.jp", "or.th", "org.nz", "org.uk",
)
```

DOM reference classification uses the same feature-side suffix algorithm.
Empty strings, fragments, and references beginning with `javascript:`,
`mailto:`, `tel:`, `data:`, or `about:` are empty references. Relative
references resolve to the base host. Only references containing literal `://`
or beginning with `//` are absolute for feature-side domain classification.
Absolute and protocol-relative references are self references when their
feature-side registrable domain equals the base feature-side registrable
domain, and external otherwise.

The frozen feature constants are:

```python
FEATURE_VERSION = "phishme-features-v1"
HASH_DIM = 1 << 18
NGRAM_RANGE = (3, 5)
COUNT_FEATURES = (
    "url_length", "domain_length", "tld_length", "subdomain_count",
    "letter_count", "digit_count", "obfuscated_count", "query_count",
    "equals_count", "ampersand_count", "special_count", "html_line_count",
    "largest_line_length", "iframe_count", "image_count", "css_count",
    "javascript_count", "self_ref_count", "empty_ref_count",
    "external_ref_count",
)
RATIO_FEATURES = (
    "letter_ratio", "digit_ratio", "obfuscated_ratio", "special_ratio",
)
BOOLEAN_FEATURES = (
    "is_domain_ip", "is_https", "has_title", "has_favicon",
    "is_responsive", "has_description", "external_form_submit",
    "has_social", "has_submit_button", "has_hidden_fields",
    "has_password_field", "mentions_bank", "mentions_pay",
    "mentions_crypto", "has_copyright",
)
NUMERIC_FEATURES = COUNT_FEATURES + RATIO_FEATURES + BOOLEAN_FEATURES
```

The FNV-1a byte algorithm is:

```python
value = 0x811C9DC5
for byte in text.encode("utf-8"):
    value ^= byte
    value = (value * 0x01000193) & 0xFFFFFFFF
```

Numeric count features use `log1p` after clipping to non-negative values.
Ratios are clipped to `[0, 1]`. Boolean values are encoded as `0` or `1`. No
fitted scaler is used.

Excluded leakage features include filename, dataset source, source labels, URL
similarity indices, TLD legitimacy probabilities, target brand, language score,
collection date, redirect counters that cannot be reproduced in-browser, popup
counters, robots metadata, and any feature requiring reputation services, DNS,
WHOIS, certificate checks, blocklists, or other external network lookup.
Features whose semantics cannot be matched across PhiUSIIL columns,
PhreshPhish parsing, and the Chrome extractor are removed rather than
approximated silently.

## Model

V3 evaluates four systems on the same frozen PhreshPhish test samples:

### PhishLang (reference)

PhishLang is the official pretrained MobileBERT model (`github.com/UTA-SPRLab/phishlang`),
used as a fixed reference. It is never retrained on PhishPedia or any V3 data.
Predictions use `generate_text_representation` from
`src/patched_parser_prediction.py`, 128-token windows with 64-token stride,
and `MobileBertForSequenceClassification` with `local_files_only=True`. The
threshold is fixed at `0.5` (official untuned). PhishLang scores are aligned to
phishMe scores by frozen `sample_id` for paired comparisons. Short inputs that
produce no full 128-token window yield phishing probability `0.0` (official
behavior).

### Linear SGD

The classifier is an SGD log-loss model equivalent to regularized logistic
regression, implemented as `SGDClassifier(loss="log_loss", penalty="l2")` with
incremental training support. Hyperparameters are selected by grid search over
α ∈ {1e-5, 1e-4, 1e-3} on PhishPedia validation data (ranking by AP, F1,
−α). Three training epochs, batch size 2048.

### Tree (LightGBM)

A gradient-boosted decision tree implemented as `LGBMClassifier` with grid
search over n_estimators ∈ {100, 500}, num_leaves=31, learning_rate=0.1.
Ranking by AP, F1, −n_estimators on PhishPedia validation. LightGBM is an
optional dependency (`phishme[tree]`); tests use `pytest.importorskip`.

### Hybrid OOF Stacking

An out-of-fold stacking ensemble with two base models (Linear SGD at α=1e-4
and LightGBM at n_estimators=300) and a `LogisticRegression` meta-classifier
(C=1.0). The meta-classifier is trained on 3-fold stratified out-of-fold
scores from the PhishPedia training split. After meta-training, both base
models are refit on the full training frame. No hyperparameter grid is used
for the hybrid; it is a single configuration. OOF score targets are never
derived from PhishPedia validation or PhreshPhish data.

All phishMe variants share the same `phishme-features-v1` vectorizer. The
operating threshold for each variant is selected once on PhishPedia validation
data before any PhreshPhish scoring.

## Evaluation

Primary metrics are average precision and F1 at the validation-selected
threshold. Accuracy is secondary and cannot alone justify model selection.

### Cross-Dataset Metrics

For each variant, after scoring the frozen PhreshPhish test:

- **ΔAP** = validation_AP − PhreshPhish_AP
- **ΔF1** = validation_F1 − PhreshPhish_F1

Positive values represent generalization drop from the in-distribution
validation set to the unseen test distribution. Secondary metrics (precision,
recall, ROC-AUC, Brier, confusion matrix) are also reported.

### Threshold Protocol

For phishMe variants: thresholds are selected exclusively on PhishPedia
validation data via `evaluate.select_threshold`, maximizing F1. These
thresholds are then applied to PhreshPhish test scores without adjustment.

For PhishLang: the threshold is fixed at `0.5` (official pretrained default)
and recorded as `threshold_source: "official_fixed_0.5"`. No tuning is
performed on any dataset.

### Low-Base-Rate Evaluation

Low-base-rate evaluation is reported at exact requested phishing prevalences
`0.0005`, `0.001`, `0.005`, `0.01`, and `0.05`. For a requested prevalence
`p`, a base-rate manifest is a deterministic shuffled array of original row
indices. With `n_neg` available benign examples, the normal case keeps every
benign example and samples `round(p * n_neg / (1 - p))` phishing examples
without replacement. If there are not enough phishing examples, the manifest
keeps every phishing example and samples
`round(n_pos * (1 - p) / p)` benign examples without replacement, capped to the
available benign count. Manifests never duplicate rows.

Base-rate reports store requested prevalence, actual selected prevalence, class
counts, selected original indices, selected frozen sample IDs, metrics from
`metrics_report`, and a SHA-256 over the canonical UTF-8 strict JSON sample-ID
list. The canonical list hash is computed from
`json.dumps(selected_sample_ids, separators=(",", ":"), ensure_ascii=False,
allow_nan=False).encode("utf-8")`.

### Paired Bootstrap

phishMe versus PhishLang comparisons use paired bootstrap estimates for
average-precision and F1 differences on the same frozen sample IDs. Each
variant's scores are aligned with PhishLang scores via `align_phishlang_scores`,
which reorders both score vectors to identical row ordering by `sample_id` and
validates label agreement. Paired bootstrap uses `evaluate.paired_bootstrap`
with 10,000 resamples and seed 42. No superiority claim is made without a 95%
confidence interval excluding zero.

### Frozen Test Protocol

The PhreshPhish test set is materialized once via `materialize_phresh_test` into
a frozen JSONL file with canonical records carrying `url`, `title`, `date`,
`label`, `sample_id`, `html`, and `dom_*` features. This file is the single
source of truth for all cross-dataset scoring. Re-materialization on an existing
frozen file fails. Scores, thresholds, and metrics are computed deterministically
from this file.

### Success Criteria

| Tier | Requirement |
|------|-------------|
| **Bronze** | PhishPedia training completes; all three variants produce validation metrics; PhreshPhish frozen test scored; all Δ metrics and paired bootstrap intervals computed |
| **Silver** | At least one phishMe variant achieves ΔAP 95% CI wholly above zero against PhishLang on identical frozen sample IDs |
| **Gold** | Silver holds AND both browser gates pass (≤ 25 MB payload, ≤ 250 ms Chrome p95) |

### Browser Deployment Gates

Runtime acceptance gates are an exported browser payload no larger than 25 MB
and Chrome p95 inference no slower than 250 ms per page. Browser metrics are
collected via headless Chrome (`--headless=new --dump-dom
--virtual-time-budget=30000`) running `web/benchmark.html` against the exported
linear model artifact. Results are recorded in `phishme-browser-metrics-v1`
schema with model bytes, extension bytes (web/ static files + model artifact),
average latency, p95 latency, memory usage, and gate booleans.

## Limitations

### V2 Limitations (still applicable)

PhiUSIIL lacks a trustworthy temporal field, so its domain-grouped test is less
realistic than the PhreshPhish temporal test. Offline HTML may differ from the
live DOM available to a Chrome extension. Linear models can miss semantic and
nonlinear deception. Feature hashing introduces collisions. Base rates and
attacker behavior drift over time. Dataset labels and collection methods may
encode source artifacts. Offline success does not prove production readiness or
long-term evasion resistance.

### V3-Specific Limitations

**PhishLang is not retrained on identical data.** PhishLang is the official
pretrained MobileBERT reference. Any advantage PhishLang shows on PhreshPhish
may stem from its own training distribution rather than superior generalization.
Conversely, any phishMe advantage carries stronger evidence.

**OOF stacking optimism.** The hybrid meta-classifier uses 3-fold out-of-fold
scores computed on the PhishPedia training split. While these OOF scores are
holdout estimates, the meta-classifier itself is trained on the same split.
Residual optimism is bounded by validation-only threshold selection, but the
hybrid may exhibit slightly inflated cross-dataset estimates relative to a
strictly held-out stacking pipeline.

**HTML-only proxy for PhishPedia.** PhishPedia pages are parsed from offline
HTML files. The PhishPedia benchmark was originally designed for screenshot-based
phishing detection; the V3 protocol uses HTML DOM features only, which may
differ from the live-DOM behavior of the same pages.

**Single frozen test set.** PhreshPhish test is one temporal distribution (phishing
samples collected up to early 2023). Conclusions generalize only as far as that
distribution and timeframe allow. No claim of universal cross-dataset
generalization is made.

**Browser gate measurement caveat.** The browser latency gates (p95 ≤ 250 ms,
≤ 25 MB payload) have been verified with the infrastructure via headless Chrome
smoke on a tiny exported model. Real trained-model latency measurements are
pending full training on the downloaded PhishPedia corpus.

**Pending external-runtime items.** Full PhreshPhish test streaming requires a
networked runtime (Hugging Face access). The official PhishLang predictions CSV
requires a machine with `PHISHLANG_DIR` (clean checkout + 98 MB model). Both are
pending cloud execution; CLI interfaces are verified with synthetic data locally.
