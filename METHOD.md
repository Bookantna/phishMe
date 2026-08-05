# phishMe Method

## Research Question

Can a regularized sparse linear classifier using URL character n-grams and
browser-computable DOM features outperform PhishLang on the same PhreshPhish
evaluation examples while remaining no larger than 25 MB and no slower than
250 ms at p95 in Chrome?

Any claim of improvement over PhishLang requires identical frozen evaluation
examples, paired uncertainty estimates, and satisfaction of the browser size and
latency gates.

## Canonical Labels

Canonical labels are fixed as `1 = phishing` and `0 = benign`. Source adapters
must normalize source-specific conventions into this representation and reject
unknown labels.

## Data Provenance and Licenses

PhiUSIIL is expected locally at `dataset/PhiUSIIL_Phishing_URL_Dataset.csv`.
The current project directory does not contain license metadata for that file,
so raw rows and derivative redistributed data are excluded unless the license is
independently verified.

PhreshPhish is used through its Hugging Face dataset source, pinned to revision
`eabec4b7a66324b79cc8a0ad856d1731dc26fe1a` for experiments. Its dataset card
lists CC BY 4.0 and restricts use to anti-phishing research. Raw HTML is parsed
inside the runtime and discarded rather than committed.

Dataset source, collection date, split membership, and labels disguised as
metadata are never model features.

## Split Rules

PhiUSIIL has no trustworthy temporal field, so it uses deterministic
registrable-domain grouping. No registrable domain or sample ID may cross the
training, validation, and test partitions. Each partition must contain both
classes.

PhreshPhish preserves the official test split as untouched test data. The
official training split is ordered by collection date; its final temporal portion
is reserved for validation, and earlier examples are used for training. Test
examples are never moved into training or validation.

Combined-data experiments may add PhiUSIIL training data to PhreshPhish training
data, but threshold selection remains validation-only and the untouched
PhreshPhish test set remains frozen.

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

The classifier is an SGD log-loss model equivalent to regularized logistic
regression, implemented as `SGDClassifier(loss="log_loss", penalty="l2")` with
incremental training support. Hyperparameters are selected only from validation
data, and the operating threshold is selected once on validation data before any
test scoring.

## Evaluation

Primary metrics are average precision and F1 at the validation-selected
threshold. Accuracy is secondary and cannot alone justify model selection.

Low-base-rate evaluation is reported at 0.05%, 0.1%, 0.5%, 1%, and 5% phishing
base rates. phishMe versus PhishLang comparisons use paired bootstrap estimates
for average-precision and F1 differences on the same frozen sample IDs.

Runtime acceptance gates are an exported browser payload no larger than 25 MB
and Chrome p95 inference no slower than 250 ms per page.

## Limitations

PhiUSIIL lacks a trustworthy temporal field, so its domain-grouped test is less
realistic than the PhreshPhish temporal test. Offline HTML may differ from the
live DOM available to a Chrome extension. Linear models can miss semantic and
nonlinear deception. Feature hashing introduces collisions. Base rates and
attacker behavior drift over time. Dataset labels and collection methods may
encode source artifacts. Offline success does not prove production readiness or
long-term evasion resistance. A controlled PhishLang comparison depends on a
runnable official artifact.
