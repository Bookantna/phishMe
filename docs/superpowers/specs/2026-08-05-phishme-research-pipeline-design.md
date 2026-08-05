# phishMe Research Pipeline Design

Date: 2026-08-05
Status: Approved design
Repository: `Bookantna/phishMe` (public)

## 1. Objective

phishMe will be a reproducible research and deployment pipeline for a lightweight, fully client-side phishing classifier. The project will train a sparse linear model from URL text and lightweight DOM/HTML signals, evaluate it under realistic dataset shift and low phishing base rates, and export it for use inside a Chrome extension.

The research question is:

> Can a regularized sparse linear classifier using URL character n-grams and browser-computable DOM features outperform PhishLang on the same PhreshPhish evaluation examples while remaining no larger than 25 MB and no slower than 250 ms at p95 in Chrome?

A result will be described as better than PhishLang only if the paired evaluation supports that statement. The project will preserve and report negative results.

## 2. Goals

1. Build and execute a verified local baseline using `dataset/PhiUSIIL_Phishing_URL_Dataset.csv`.
2. Provide a Google Colab notebook that streams PhreshPhish without transferring the 36.6 GB corpus through the user's home connection.
3. Prevent duplicate and domain leakage between training, validation, and test data.
4. Evaluate cross-dataset and temporal generalization rather than relying on a random row split.
5. Compare URL-only and URL-plus-DOM models through a controlled ablation.
6. Compare phishMe and PhishLang on an identical frozen evaluation manifest.
7. Export a deterministic browser scorer whose output matches Python.
8. Keep the model at or below 25 MB and Chrome p95 inference latency at or below 250 ms per page.
9. Document the scientific method in `METHOD.md` and implementation progress in `Implement.md`.

## 3. Non-goals

The first phase will not:

- build a complete Chrome extension UI;
- use remote reputation APIs, DNS/WHOIS lookups, certificate checks, or blocklists at inference time;
- train a transformer, large language model, vision model, or teacher-student distillation system;
- introduce MLflow, DVC, Hydra, a database, or a hosted inference service;
- redistribute either raw dataset;
- claim production readiness from offline evaluation alone.

The output will be integration-ready model artifacts and a JavaScript scorer, not a finished browser product.

## 4. Observed Project Context

The current directory contains:

- `dataset/PhiUSIIL_Phishing_URL_Dataset.csv`: 235,795 rows, 55 input columns plus the label, and no missing values observed during the initial audit;
- `paper/Paper 1 PhishLang A Real-Time, Fully Client-Side Phishing Detection Framework Using MobileBERT.pdf`;
- no existing nested Git repository or project source files.

The local dataset contains 425 duplicate URL rows beyond the first occurrence, 220,086 unique domains, and 15,709 domain repetitions beyond the first occurrence. Fifty-four domains contain both labels. These properties make random row splitting inappropriate.

PhreshPhish currently contains 666,315 URL/HTML examples and occupies approximately 36.6 GB. Its published train and test sets are temporally separated and filtered to reduce similarity leakage. The project will pin the exact Hugging Face dataset revision used by each experiment.

PhishLang uses a 25-million-parameter MobileBERT model. Its paper reports approximately 74 MB memory and 0.39 seconds median inference in its evaluated environment, exceeding this project's model-size and latency targets. Its original controlled evaluation used a random 70/30 split of PhishPedia. phishMe will focus on cross-dataset and temporal evaluation instead.

## 5. Scientific Hypothesis

The proposed model may generalize better than PhishLang because:

- character n-grams preserve suspicious URL substrings and obfuscation patterns without a fixed tokenizer vocabulary;
- the model examines the entire bounded URL and title representation instead of truncating a page representation to 512 tokens;
- structural DOM signals provide evidence complementary to URL text;
- linear L2 regularization limits reliance on individual templates and high-dimensional coincidences;
- every feature is available at browser inference time, preventing training-serving skew from unavailable reputation features;
- feature hashing provides a bounded model size independent of vocabulary growth.

These arguments support an experiment; they do not guarantee superiority. Dataset shift, collection artifacts, label noise, and adversarial adaptation may invalidate the hypothesis.

## 6. Architecture

### 6.1 End-to-end flow

```text
PhiUSIIL CSV or PhreshPhish stream
  -> source adapter and schema validation
  -> label normalization and URL canonicalization
  -> deduplication and split controls
  -> shared URL/title/DOM feature extraction
  -> sparse feature hashing and numeric transforms
  -> incremental logistic regression
  -> validation threshold selection
  -> frozen test evaluation and reports
  -> versioned model artifact
  -> JavaScript scorer and browser benchmark
```

### 6.2 Minimal repository structure

```text
phishMe/
├── README.md
├── METHOD.md
├── Implement.md
├── pyproject.toml
├── notebooks/
│   └── phishme_colab.ipynb
├── src/phishme/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── export.py
├── tests/
│   ├── test_features.py
│   ├── test_splits.py
│   └── test_pipeline.py
└── web/
    ├── scorer.js
    ├── benchmark.html
    └── parity.test.mjs
```

Generated datasets, caches, checkpoints, models, and reports will be ignored by Git. Only compact fixtures and explicitly selected summary reports may be committed.

## 7. Data Contracts

### 7.1 Canonical sample

Every source adapter will produce this logical record:

- `url`: original URL used for feature extraction;
- `title`: page title when available, otherwise an empty string;
- `dom`: normalized browser-computable structural fields;
- `label`: integer, where `1` means phishing and `0` means benign;
- `source`: dataset identifier used for analysis but never as a model feature;
- `date`: collection date when provided, used for temporal splitting but never as a model feature;
- `group`: registrable domain used to prevent group leakage;
- `sample_id`: deterministic hash used in manifests and duplicate detection.

### 7.2 Label normalization

The PhiUSIIL adapter will convert its observed convention (`0` phishing, `1` legitimate) into the canonical convention (`1` phishing, `0` benign). The adapter will include sanity checks against known benign and phishing examples and will fail on any value other than `0` or `1`.

The PhreshPhish adapter will map `phish` to `1` and `benign` to `0` and fail on every other value.

### 7.3 Dataset licensing and publication

PhreshPhish is licensed under CC BY 4.0 and restricted by its dataset card to anti-phishing research. The repository will cite the dataset and paper.

The local PhiUSIIL file has no license metadata in the current directory. The project will not commit or redistribute it. `METHOD.md` will record its local path and require users to obtain it from an authorized source. Publication of any derivative data from PhiUSIIL is outside this phase unless its license is independently verified.

## 8. Feature Specification

A single versioned feature specification will be implemented in Python and JavaScript.

### 8.1 Text features

- URL character n-grams of lengths 3, 4, and 5, prefixed with the `url` namespace.
- Page-title character n-grams of lengths 3, 4, and 5, prefixed with the `title` namespace.
- Text is Unicode-normalized and lowercased, while URL punctuation is preserved.
- N-grams use binary presence rather than raw frequency so repeated filler cannot increase a feature without bound.
- Features are mapped into `2^18` buckets with a documented 32-bit hash implementation shared by Python and JavaScript.
- The hash uses non-negative feature values to simplify browser parity.

### 8.2 Numeric and Boolean features

Only static URL and live-DOM feature names available from both sources and from a Chrome content script are eligible. URL features will be recomputed from the raw URL for both datasets. PhiUSIIL has no raw HTML, so its DOM values must be mapped from the provided columns; PhreshPhish and Chrome will derive those values from HTML or the live DOM. This unavoidable extractor difference is treated as source shift, documented in run manifests, and tested by the combined-data ablation rather than assumed harmless. The initial feature set includes:

- URL length, domain length, IP-domain flag, TLD length, and subdomain count;
- counts and ratios for letters, digits, obfuscated characters, query markers, equals signs, ampersands, and other special characters;
- HTTPS flag;
- HTML line count and largest line length;
- title, favicon, responsive metadata, and description flags;
- iframe count;
- external form-submit flag;
- social-link, submit-button, hidden-field, and password-field flags;
- bank, payment, and cryptocurrency term flags;
- copyright flag;
- image, CSS, JavaScript, self-reference, empty-reference, and external-reference counts.

Count features use `log1p` transformation. Ratios are clipped to `[0, 1]`. Boolean values are encoded as `0` or `1`. No fitted scaler is required, reducing training-serving parity risk.

The model will exclude filename, dataset source, labels disguised as metadata, URL similarity indices, TLD legitimacy probabilities, target brand, language score, collection date, and any feature requiring an external network lookup.

Features whose semantics cannot be matched sufficiently across the PhiUSIIL columns, the PhreshPhish parser, and the browser extractor will be removed from the shared model rather than approximated silently. The final ordered feature list and definitions will be frozen in `METHOD.md` before the first untouched test evaluation.

### 8.3 Model

The classifier is regularized logistic regression trained incrementally with stochastic gradient descent:

```text
p(y=1 | x) = sigmoid(w*x + b)
```

Training uses log loss and L2 regularization. The fixed hash space makes the model size bounded. Hyperparameter selection is restricted to a small validation-only comparison of regularization strength and the URL-only versus URL-plus-DOM ablation. Test results will not influence hyperparameters or threshold selection.

## 9. Data Preparation and Splits

### 9.1 Canonicalization and duplicates

URLs will be canonicalized only for duplicate detection: scheme and host casing are normalized, fragments are removed, and a trailing root slash is normalized. The original URL remains the feature input. Exact duplicate samples are removed before splitting, with removal counts recorded.

### 9.2 Local dataset

The local dataset has no trustworthy collection date. It will use deterministic registrable-domain grouping. A five-fold stratified group split will allocate three folds to training, one to validation, and one to test, producing approximately 60/20/20 partitions while ensuring no registrable domain crosses partitions.

The split process will assert:

- no sample ID overlap;
- no registrable-domain overlap;
- both classes in each partition;
- recorded class counts and rates.

### 9.3 PhreshPhish

The official test split remains untouched. The official train split will be ordered by collection date, and its final temporal portion will be reserved as validation. Earlier examples will be used for training. The exact cutoff and counts will be recorded in a manifest generated by the pipeline.

Similarity and temporal controls already applied by PhreshPhish will be preserved. phishMe may remove exact duplicates but will not move examples from the official test split into training or validation.

### 9.4 Combined-data experiment

The main PhreshPhish model will be compared with a model trained on PhiUSIIL training data plus PhreshPhish training data. Both will use the same PhreshPhish validation threshold procedure and untouched test set. Combined training will be retained as the recommended model only if it improves the primary metrics without violating the runtime budget.

## 10. Training Workflows

### 10.1 Local baseline

The local command-line workflow will:

1. validate and audit the CSV;
2. write split manifests;
3. train URL-only and URL-plus-DOM models;
4. choose regularization and threshold using validation data;
5. evaluate once on the local test split;
6. export the selected model and report;
7. run JavaScript parity and size checks.

The implementation must execute this workflow successfully before the project is described as locally verified.

### 10.2 Colab workflow

The notebook will run in two modes:

- `smoke`: a small deterministic subset that validates installation, streaming, checkpoints, training, evaluation, and export;
- `full`: the pinned PhreshPhish train and test splits.

The full workflow streams records directly between Hugging Face infrastructure and the Colab runtime. Raw HTML is parsed, converted to bounded features, and discarded. It is not downloaded to the user's machine.

Training processes bounded batches and saves atomic checkpoints to Google Drive. A checkpoint includes model weights, intercept, feature version, processed position, dataset revision, run arguments, and random seed. Restarting the notebook resumes only from a compatible checkpoint. A mismatched dataset revision or feature version causes a hard failure rather than an unsafe resume.

Network failures use bounded retries provided by the dataset client. Persistent failures stop the run with the last valid checkpoint preserved.

## 11. Experiments

### 11.1 Local validation

Compare URL-only and URL-plus-DOM models on the domain-separated PhiUSIIL test partition.

Purpose: verify the end-to-end pipeline and quantify the contribution of DOM features without random domain leakage.

### 11.2 Cross-dataset generalization

Train on PhiUSIIL training data only and evaluate on a frozen PhreshPhish evaluation manifest without fine-tuning.

Purpose: expose source shift and determine whether high local performance transfers.

### 11.3 Temporal PhreshPhish evaluation

Train on the official historical train split and evaluate on the official future test split.

Purpose: measure performance under temporal drift and the dataset's published leakage controls.

### 11.4 Combined-data ablation

Train with and without PhiUSIIL added to PhreshPhish training data.

Purpose: determine whether the older feature dataset adds diversity or introduces harmful source artifacts.

### 11.5 PhishLang comparison

The preferred comparator is the official PhishLang parser and model artifact pinned to a repository commit. PhishLang and phishMe will score the same frozen sample IDs, and evaluation will use the same canonical labels and metrics.

If a runnable official artifact cannot be obtained, the project will explicitly report that the controlled comparison was not completed. The user's prior 50–60% accuracy observation may be recorded as context but cannot support a superiority claim. No reconstructed substitute will be labeled as the official PhishLang model.

## 12. Evaluation

### 12.1 Metrics

Primary metrics:

- average precision;
- F1 at the validation-selected threshold.

Secondary metrics:

- accuracy;
- precision;
- recall;
- ROC-AUC;
- false-positive rate;
- confusion matrix;
- Brier score and a calibration summary.

Accuracy will be reported but not used alone to choose a model.

### 12.2 Threshold policy

The operating threshold is selected once on validation data by maximum F1. The selected threshold is frozen before test predictions are summarized. No test-specific threshold will be reported as the main result.

### 12.3 Base-rate evaluation

Results will be measured on the PhreshPhish benchmark base rates of 0.05%, 0.1%, 0.5%, 1%, and 5%. Official benchmark manifests will be used when available. If they are not distributed in directly consumable form, deterministic resampling will follow the published PhreshPhish benchmark procedure, and generated manifest hashes will be recorded.

### 12.4 Statistical comparison

For phishMe versus PhishLang, paired bootstrap resampling will estimate 95% confidence intervals for differences in average precision and F1. The random seed, number of resamples, sample IDs, and interval method will be recorded in the run manifest.

The project may claim better performance only when:

1. both models scored the same frozen examples;
2. phishMe has higher average precision and F1;
3. the paired intervals support the reported difference rather than an unqualified tie;
4. phishMe satisfies the model-size and browser-latency limits.

## 13. Browser Artifact and Runtime

The export step will create a versioned model artifact containing:

- float32 coefficients and intercept;
- hash-space size, n-gram range, and hash specification;
- ordered numeric feature specification and transforms;
- canonical label mapping;
- validation-selected threshold;
- feature version;
- source commit and dataset revisions;
- validation and test summary references.

`web/scorer.js` will:

1. collect the URL, title, and approved DOM features;
2. reproduce feature normalization and hashing;
3. compute the sparse dot product and sigmoid probability;
4. return probability, threshold, predicted label, feature version, and timing.

A complete browser extension is out of scope. The scorer is the stable integration boundary for a later extension.

## 14. Runtime Acceptance Tests

The selected artifact must satisfy:

- exported model payload at or below 25 MB;
- p95 scorer latency at or below 250 ms per page in Chrome;
- exact feature version match between model and scorer;
- Python/JavaScript probabilities matching within a documented floating-point tolerance on golden examples.

`web/benchmark.html` will measure warm and cold runs over representative benign and phishing fixtures and export a JSON summary containing browser version, platform, model size, sample count, p50, p95, and maximum latency.

## 15. Error Handling and Data Quality

The pipeline will fail immediately on:

- unknown labels;
- missing required columns;
- an unsupported dataset revision;
- incompatible checkpoints;
- feature-version mismatches;
- domain overlap between partitions;
- sample ID overlap between partitions;
- a model artifact that cannot be parsed or whose dimensions are inconsistent.

Malformed HTML will be parsed tolerantly. Each fallback, empty title, empty document, and skipped record will be counted by reason. Records missing a valid URL or label will be rejected rather than assigned defaults. Reports will include processed, rejected, duplicate, and retained counts.

Checkpoint writes and final artifacts will use temporary files followed by atomic replacement. Interrupted writes must not appear as valid checkpoints.

## 16. Verification Strategy

The smallest checks that protect the research claims are:

1. feature tests for URL normalization, title extraction, numeric transforms, and hash determinism;
2. split tests proving no sample or registrable-domain overlap;
3. label-mapping tests for both datasets;
4. a tiny end-to-end training test that creates a loadable artifact and metrics report;
5. golden Python/JavaScript parity vectors;
6. a notebook smoke-mode run;
7. actual local baseline execution on PhiUSIIL;
8. browser artifact size and latency measurement.

Tests will use small synthetic or explicitly permitted fixtures, never committed raw dataset samples without confirmed redistribution rights.

## 17. Research and Implementation Documentation

### 17.1 `METHOD.md`

`METHOD.md` is the authoritative scientific protocol. It will contain:

- research question and hypotheses;
- theoretical motivation;
- dataset provenance, versions, labels, and licensing;
- feature and model definitions;
- split, deduplication, and leakage controls;
- training and checkpoint procedure;
- metrics, threshold selection, base-rate analysis, and statistics;
- runtime evaluation;
- reproducibility requirements;
- assumptions, limitations, and threats to validity.

Method changes made after seeing test results must be identified as post-hoc and must not overwrite the original result silently.

### 17.2 `Implement.md`

`Implement.md` is the operational record. It will contain:

- pending, in-progress, completed, blocked, and rejected work;
- commands actually executed;
- decisions and deviations from `METHOD.md`;
- test, training, evaluation, size, and latency outputs;
- run IDs, artifact locations, and hashes;
- known failures and next actions.

It will be updated as work occurs. It will not invent outputs for Colab runs that have not been executed.

## 18. Reproducibility

Every run manifest will record:

- code commit;
- Python and dependency versions;
- dataset name and pinned revision;
- split manifest hashes;
- random seed;
- feature version and hash parameters;
- model hyperparameters;
- threshold and how it was selected;
- processing counts and failures;
- model and report hashes;
- hardware/runtime details when measuring performance.

One documented command will reproduce the local baseline. The Colab notebook will expose the same core package functions rather than maintain a second implementation.

## 19. Risks and Limitations

- PhiUSIIL lacks a trustworthy temporal field, so its domain-grouped test remains less realistic than the PhreshPhish temporal test.
- HTML collected offline may differ from the live DOM visible to a Chrome extension.
- Linear models can miss nonlinear relationships and semantic deception.
- Feature hashing introduces collisions; the fixed dimension trades a small amount of representational fidelity for bounded size.
- Phishing base rates and attacker behavior change over time.
- Dataset labels and collection methods may encode source-specific artifacts.
- Offline benchmark success does not measure user experience, evasion resistance, or long-term concept drift.
- A controlled PhishLang comparison depends on obtaining a runnable official artifact.

These limitations will be reported with results rather than hidden by aggregate accuracy.

## 20. Delivery Sequence

1. Create the public `Bookantna/phishMe` repository without committing raw datasets.
2. Add `METHOD.md`, `Implement.md`, project metadata, and tests.
3. Implement and verify shared features and leakage-safe splits.
4. Train and evaluate the local URL-only and URL-plus-DOM baselines.
5. Export and verify the browser scorer.
6. Build and smoke-test the Colab notebook.
7. Run the full PhreshPhish training and evaluation when Colab compute is available.
8. Run the controlled PhishLang comparison if its official artifact is executable.
9. Publish only claims supported by frozen-test and runtime evidence.

The implementation phase begins only after this written specification is reviewed and an implementation plan is approved.
