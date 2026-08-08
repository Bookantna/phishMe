# phishMe V3 Cross-Dataset Generalization — Design Record

**Spec:** 2026-08-08-phishme-v3-cross-dataset-design  
**Status:** implemented  
**Plan:** `.hermes/plans/2026-08-08_213237-phishme-v3-cross-dataset.md` (PLANV3)

---

## Research Question

Can phishMe variants — a regularized sparse linear classifier (SGD log-loss), a
LightGBM gradient-boosted tree, and an out-of-fold stacking hybrid — trained on
the PhishPedia 30k benchmark generalize to the frozen PhreshPhish test split,
and how do they compare against the official pretrained PhishLang MobileBERT
reference on identical frozen sample IDs under paired bootstrap uncertainty and
browser deployment constraints?

---

## Motivation

The original phishMe pipeline trained and evaluated on PhiUSIIL, a single
dataset without a trustworthy temporal field, using only a linear model. V3
repositions phishMe as a cross-dataset generalization study:

1. **Training source**: PhishPedia (CC0-1.0), a published 30k phishing/benign
   benchmark with an official train/validation/test split and offline HTML
   corpus — resolving prior PhiUSIIL licensing uncertainty.
2. **Test-only destination**: PhreshPhish (CC BY 4.0), the temporally-ordered
   phishing detection benchmark, used exclusively as frozen test data.
3. **Baseline**: PhishLang, the official pretrained MobileBERT model, never
   retrained on PhishPedia — kept as the fixed reference for cross-dataset
   comparison.
4. **Model zoo**: Three phishMe variants exercising different inductive biases
   over the same frozen `phishme-features-v1` contract.

---

## Four Systems

| # | System | Architecture | Training |
|---|--------|-------------|----------|
| 1 | **PhishLang** | Official pretrained MobileBERT (98 MB), 128-token windows, 64-token stride | Frozen pretrained reference; threshold fixed at 0.5 |
| 2 | **Linear SGD** | `SGDClassifier(loss="log_loss", penalty="l2")` over FNV-1a hashed URL/title 3-5-grams + browser-computable DOM features | PhishPedia train split; grid α ∈ {1e-5, 1e-4, 1e-3} × 3 epochs |
| 3 | **Tree** | `LGBMClassifier` over the same feature matrix | PhishPedia train split; grid n_estimators ∈ {100, 500}, num_leaves=31 |
| 4 | **Hybrid OOF** | 3-fold stratified out-of-fold stacking: Linear + Tree base → LogisticRegression meta-classifier, refit on full train | PhishPedia train split; single config (meta_c=1.0, α=1e-4, n_estimators=300) |

All phishMe variants use the identical frozen `phishme-features-v1` contract
(FNV-1a 32-bit hashing into 2^18 buckets, URL/title character 3-5-grams,
browser-computable numeric + DOM features). No feature is derived from
PhishPedia-specific columns or metadata.

---

## Split Strategy

### PhishPedia (training source)

- Published split CSV: `train_test_val_split_30.csv` with columns `file_name,
  url, label` (and optional `type`, `split`).
- Official split partitions used as-is: train → training, validation →
  threshold selection, test → in-distribution reporting.
- Registrable-domain grouping is used for duplicate/overlap accounting only;
  the CSV split is never re-partitioned.
- HTML corpus: phishing pages under a `phish_html/` directory, benign pages
  under `benign_html/`. File names map verbatim (`<file_name>.html`).
- Label normalization: `label ∈ {0, 1}` or `type ∈ {phishing, benign}`.
  Mismatched or ambiguous labels are rejected.

### PhreshPhish (frozen test)

- PhreshPhish preserves the official test split as untouched test data. The
  official training split is ordered by collection date; its final temporal
  portion is reserved for validation, and earlier examples are used for
  training. Test examples are never moved into training or validation.
- In V3, the PhreshPhish official test split is the sole cross-dataset test
  set. It is never used for training, threshold selection, or hyperparameter
  tuning. PhreshPhish train-split rows are never used in V3.
- The test set is materialized once (via `materialize_phresh_test`) into a
  frozen JSONL file with canonical records carrying sample_id, label, parsed
  HTML, and DOM features.
- Dataset source pinned to Hugging Face revision
  `eabec4b7a66324b79cc8a0ad856d1731dc26fe1a`.

---

## Metrics

### Primary

- **Average precision (AP)** and **F1 at validation-selected threshold**.
  Accuracy is secondary and cannot alone justify model selection.

### Cross-dataset

- **ΔAP** = validation_AP − PhreshPhish_AP (positive = generalization drop).
- **ΔF1** = validation_F1 − PhreshPhish_F1.
- Reported per variant.

### Secondary

- Precision, recall, ROC-AUC, Brier score, confusion matrix.
- All metrics from `evaluate.metrics_report` at selected thresholds.

---

## Statistical Analysis

- **Paired bootstrap**: 10,000 resamples (seed 42) on identical frozen sample
  IDs, via the existing `evaluate.paired_bootstrap`. Each variant's scores are
  aligned with PhishLang scores by sample_id (reordered to identical row
  ordering via `align_phishlang_scores`).
- **95% confidence intervals**: reported for ΔAP and ΔF1. No superiority claim
  is made without a 95% CI excluding zero.
- **Threshold selection**: occurs on PhishPedia validation only, before any
  PhreshPhish scoring. PhishLang threshold is fixed at 0.5 (untuned).
- **Frozen sample IDs**: every canonical record carries a deterministic
  `sample_id = SHA-256(canonicalized_url)`. This is the pairing key for all
  cross-system comparisons.

---

## Low-Base-Rate Evaluation

Reported at phishing prevalences `0.0005, 0.001, 0.005, 0.01, 0.05` on the
PhreshPhish test split. Manifests use deterministic shuffled row indices at a
fixed seed. Canonical sample-ID list SHA-256 recorded for reproducibility.

---

## Browser Deployment Metrics

Measured by headless Chrome via `v3-browser`:

| Gate | Threshold | Measurement |
|------|-----------|-------------|
| Payload size | ≤ 25 MB | Exported `model.json` (linear only, browser-deployable) + all `web/` static files |
| p95 latency | ≤ 250 ms | `web/benchmark.html` → per-page scoring → JSON emission → headless Chrome `--dump-dom` collection |

Reported via `phishme-browser-metrics-v1` schema: model_bytes, extension_bytes,
avg_latency_ms, p95_latency_ms, memory_bytes, pages, chrome binary, gate booleans.

---

## Success Criteria (Bronze / Silver / Gold)

| Tier | Requirement |
|------|-------------|
| **Bronze** | PhishPedia training completes; all three variants produce valid validation metrics; PhreshPhish frozen test scored; all Δ metrics and paired bootstrap intervals computed |
| **Silver** | At least one phishMe variant achieves ΔAP 95% CI wholly above zero relative to PhishLang on identical frozen sample IDs |
| **Gold** | Silver holds AND both browser gates pass (≤ 25 MB payload, ≤ 250 ms p95) |

Bronze is the minimum deliverable; Silver and Gold are aspirational outcomes
requiring statistical evidence.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| **PhishLang training distribution advantage**: PhishLang may have been trained on data overlapping with PhreshPhish. This is a stated limitation — PhishLang is the official pretrained reference and is not retrained on identical PhishPedia data. | Document as limitation; any phishMe advantage is the stronger claim. |
| **OOF stacking optimism**: Hybrid uses 3-fold OOF scores from PhishPedia train split. Residual optimism is bounded by validation-only threshold selection. | Document; PhreshPhish test is never used for any selection. |
| **HTML-only proxy**: PhishPedia pages are parsed from offline HTML. Live-DOM behavior may differ from the training distribution. | PhishPedia is an offline benchmark by design; this is an inherent limitation of the dataset. |
| **Single frozen test set**: PhreshPhish test is one temporal distribution; conclusions generalize only as far as that distribution. | Document; no claim of universal generalization. |
| **Base-rate drift**: PhreshPhish phishing prevalence differs from deployment environments. | Low-base-rate manifest evaluation covers a range of simulated prevalences. |
| **Browser gate measurement**: Current smoke was performed on a tiny exported model (not the fully-trained model). Real trained-model latency is pending. | Document as limitation; gates validated with the smoke infrastructure. |
| **PhreshPhish streaming + PhishLang run pending**: Both require a networked runtime with Hugging Face access and PHISHLANG_DIR. | Pending items recorded in Implement.md; CLI interfaces are verified with synthetic data. |

---

## Expected Contributions

- **First cross-dataset comparison**: phishMe variants (text-gram + DOM
  features) vs. PhishLang MobileBERT on identical frozen PhreshPhish test
  samples, providing evidence about whether lightweight browser-computable
  features can match a 98 MB pretrained transformer for phishing detection.
- **Reproducible protocol**: Frozen feature contract, pinned dataset revisions,
  deterministic sample IDs, and paired bootstrap CI on identical row ordering.
- **Browser viability evidence**: Whether the model-payload and latency gates
  can be met by a JavaScript-deployable linear classifier.

---

## Implemented Modules

| Module | Purpose |
|--------|---------|
| `src/phishme/phishpedia.py` | PhishPedia split CSV + HTML → canonical frame adapter |
| `src/phishme/models.py` | TreeConfig / fit_tree / predict_tree_scores; HybridConfig / fit_hybrid / predict_hybrid_scores (3-fold OOF stacking) |
| `src/phishme/cross_dataset.py` | run_v3_train (grid + threshold on PhishPedia validation), materialize_phresh_test, run_v3_eval (frozen PhreshPhish scoring + PhishLang alignment + paired bootstrap), compute_delta, align_phishlang_scores |
| `src/phishme/__main__.py` | CLI: `v3-train`, `v3-eval`, `v3-browser` |
| `scripts/measure_browser.py` | Headless Chrome → `browser-metrics.json` collector |
| `web/benchmark.html` | Browser scoring harness with machine-readable JSON emission |
| `src/phishme/phresh.py` | sample_id in canonical records (pairing key for PhishLang alignment) |
