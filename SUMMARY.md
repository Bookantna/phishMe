# PhishPedia Local Training Summary

## Overview

The original repository contained a Colab-oriented PhishPedia workflow that did
not match the actual official dataset layout. The workflow was replaced with a
local, ZIP-native pipeline, the official phishing and benign archives were
validated, and three phishing-detection model variants were trained and
independently reviewed.

The final reviewed artifacts are located at:

```text
D:\phishme-dataset\artifacts\phishpedia-full-v3-reviewed
```

This directory supersedes the earlier run at:

```text
D:\phishme-dataset\artifacts\phishpedia-full-v3
```

## Dataset Inspection

The phishing archive is located at:

```text
C:\Users\User\codespace\phishMe\dataset\phish_sample_30k.zip
```

It contains:

- 29,496 phishing site directories
- Phishing metadata, HTML, and screenshots
- No benign examples
- No `train_test_val_split_30.csv`

The benign archive is located at:

```text
D:\phishme-dataset\benign_sample_30k.zip
```

Its verified SHA-256 is:

```text
bc7f6c950829ac34dfddd7bcf320bea9fe9a4222c9c063e49aef379c08179752
```

The phishing archive's verified SHA-256 is:

```text
3ea1865f31934aa667cb7940f37fea73b32c8fe79da556c879ee5cae08d8468b
```

## Local Workflow Changes

The old workflow incorrectly assumed that the dataset provided:

- One combined phishing/benign archive
- Extracted phishing and benign HTML directories
- An official train/validation/test CSV
- A Colab runtime

The replacement workflow:

- Reads the phishing and benign ZIP archives directly
- Avoids extracting more than 40 GiB of data
- Parses the real metadata format used by each archive
- Runs locally on Windows
- Records archive hashes and detailed audit statistics
- Uses a deterministic local notebook rather than requiring Colab

## Final Dataset Audit

After rejecting unusable records and removing canonical-URL duplicates, the
final dataset contains:

| Class | Accepted records |
|---|---:|
| Phishing | 24,535 |
| Benign | 22,251 |
| **Total** | **46,786** |

Detailed audit results:

### Phishing archive

- Site directories: 29,496
- URL and HTML candidate pairs: 29,048
- Accepted unique records: 24,535
- Duplicate canonical URLs removed: 4,513
- Missing HTML: 448
- Missing metadata: 0
- Invalid metadata: 0
- Invalid URLs: 0

### Benign archive

- Site directories: 30,649
- URL and HTML candidate pairs: 22,252
- Accepted unique records: 22,251
- Duplicate canonical URLs: 0
- Missing HTML: 8,397
- Missing metadata: 4,248
- Invalid metadata: 0
- Invalid URLs: 1

No canonical URL appeared with conflicting phishing and benign labels.

## Leakage-Controlled Splits

Records are grouped using connected components. Two records remain in the same
component whenever they share either:

- A registrable domain, or
- A valid phishing `family_id`

Null and common sentinel family values are treated as missing. This prevents
unrelated records with values such as `None`, `N/A`, `0`, or `-1` from being
merged into a false family.

The corrected split is:

| Split | Rows |
|---|---:|
| Train | 28,073 |
| Validation | 9,357 |
| Holdout | 9,356 |

Verification established:

- Zero sample-ID overlap between splits
- Zero connected domain/family-group overlap between splits
- Hyperparameters and thresholds selected only on validation data
- Holdout data scored only after selection

## Models Trained

Three model variants were trained:

1. Linear SGD classifier
2. LightGBM tree classifier
3. Hybrid linear/tree classifier

One primary variant is declared using validation results only. Variants are
ranked by:

1. Validation average precision, descending
2. Validation F1, descending
3. Validation Brier score, ascending
4. Fixed variant order as the final tie-breaker

The validation-selected primary model is the **hybrid** variant.

## Corrected Holdout Results

### Hybrid primary model

| Metric | Result |
|---|---:|
| Accuracy | 0.9988243 |
| Average precision | 0.9999978 |
| F1 | 0.9988795 |
| Precision | 0.9983710 |
| Recall | 0.9993885 |
| False-positive rate | 0.0017978 |
| ROC AUC | 0.9999976 |
| Brier score | 0.0006311 |

Confusion matrix:

```text
[[4442, 8],
 [   3, 4903]]
```

This corresponds to:

- 4,442 true negatives
- 8 false positives
- 3 false negatives
- 4,903 true positives

## Independent Review and Corrections

The first complete run revealed a subtle grouping problem during independent
review:

- 783 phishing records had `family_id=None`
- The initial loader converted `None` to the string `"None"`
- Unrelated records were consequently placed into one large family group

This did not introduce train/test leakage, but it was overly conservative and
incorrect. The loader was fixed, and the full dataset was retrained into the
reviewed artifact directory.

The review also led to the following hardening:

- Null and common sentinel family IDs are treated as missing
- Malformed metadata and URLs are counted and skipped per sample
- Duplicate ZIP members are rejected
- Invalid ZIP files produce a clear error
- Metadata reads are limited to 1 MiB per member
- HTML reads are limited to 16 MiB per member
- Cross-label URL conflicts fail closed
- Empty legacy holdouts no longer crash after training
- Archive smoke limits below 30 produce a clear error
- Detailed accepted, rejected, missing, malformed, and duplicate counts are
  included in `run.json`
- A primary model is declared from validation results before holdout inspection
- Notebook paths are configurable rather than tied to one computer

## Final Artifacts

The reviewed output directory contains:

```text
D:\phishme-dataset\artifacts\phishpedia-full-v3-reviewed\
├── model-linear.json
├── models\
│   ├── hybrid.joblib
│   ├── linear.joblib
│   └── tree.joblib
├── run.json
└── validation-report.json
```

Artifact purposes:

- `models\hybrid.joblib`: validation-selected primary model
- `models\linear.joblib`: trained linear variant
- `models\tree.joblib`: trained tree variant
- `model-linear.json`: browser-compatible linear artifact
- `validation-report.json`: selected configurations, thresholds, validation
  metrics, holdout metrics, and primary-variant declaration
- `run.json`: input hashes, dataset audit, split manifests, runtime versions,
  and artifact hashes

All three reviewed `.joblib` artifacts were loaded back from disk and scored 30
real archive records. Every returned score was finite and within `[0, 1]`.

## Verification

Final verification results:

- Python test suite: 146 passed
- Expected external PhishLang skip: 1
- Directly affected test modules: 26 passed
- Browser parity tests: 7 passed
- Ruff linting: passed
- Notebook validation: passed
- `git diff --check`: passed
- Independent re-review: passed
- Remaining security or logic findings: none

## Important Interpretation Limitation

The very high local validation and holdout scores are within-corpus diagnostics,
not proof of real-world deployment performance.

The phishing and benign classes came from separately published archives and may
retain differences related to:

- Crawler behavior
- Collection time
- HTML completeness
- Missing assets
- Source-specific formatting or storage artifacts

Domain/family-disjoint splitting prevents direct duplicate leakage, but it
cannot remove every possible corpus shortcut. Comparative or deployment claims
require evaluation on the frozen PhreshPhish cross-dataset benchmark.

## Hardware Note

The RTX 4060 Ti was not used by this workflow. The current project trains
scikit-learn and LightGBM models over URL, title, and DOM-derived features, so
the completed training run was CPU-based. The GPU would be relevant for a
separate screenshot, logo-detection, or neural-network model.

## Repository State

The implementation, tests, notebook, and documentation changes remain
uncommitted. No commit or push was performed because it was not requested.
