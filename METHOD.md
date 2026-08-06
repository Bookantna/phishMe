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

The classifier is an SGD log-loss model equivalent to regularized logistic
regression, implemented as `SGDClassifier(loss="log_loss", penalty="l2")` with
incremental training support. Hyperparameters are selected only from validation
data, and the operating threshold is selected once on validation data before any
test scoring.

## Evaluation

Primary metrics are average precision and F1 at the validation-selected
threshold. Accuracy is secondary and cannot alone justify model selection.

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

phishMe versus PhishLang comparisons use paired bootstrap estimates for
average-precision and F1 differences on the same frozen sample IDs. The
controlled PhishLang adapter reads a frozen CSV with at least
`sample_id,label,html` and writes prediction CSV columns exactly
`sample_id,label,score,model,source_commit` in the same order. The official
mode uses a clean checkout of `https://github.com/UTA-SPRLab/phishlang.git`,
imports `generate_text_representation` from
`src/patched_parser_prediction.py`, loads the local `src/model` with
`MobileBertTokenizer` and `MobileBertForSequenceClassification` using local
files only, records the clean Git commit, records a deterministic SHA-256 over
the sorted model-tree file hash manifest, and preserves the official
short-input behavior where no full 128-token window yields phishing probability
`0.0`. The adapter's `batch_size` only bounds CSV input chunks; it is not
vectorized MobileBERT batching. Each sample keeps the official per-sample
128-token window and 64-token stride semantics. Corrected short-input
sensitivity results, if ever produced, must use a distinct model label and
separate output.

Runtime acceptance gates are an exported browser payload no larger than 25 MB
and Chrome p95 inference no slower than 250 ms per page.

## Limitations

PhiUSIIL lacks a trustworthy temporal field, so its domain-grouped test is less
realistic than the PhreshPhish temporal test. Offline HTML may differ from the
live DOM available to a Chrome extension. Linear models can miss semantic and
nonlinear deception. Feature hashing introduces collisions. Base rates and
attacker behavior drift over time. Dataset labels and collection methods may
encode source artifacts. Offline success does not prove production readiness or
long-term evasion resistance. The controlled PhishLang comparison has not yet
been run in cloud with the official model, paired intervals are not yet
available, and the Chrome p95 latency gate is still missing.
