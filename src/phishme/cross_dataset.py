from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from phishme import benchmark, evaluate, export, features, models, phresh, train

VARIANT_KINDS = ("linear", "tree", "hybrid")
LINEAR_ALPHAS = (1e-5, 1e-4, 1e-3)
TREE_ESTIMATOR_GRID = (100, 500)
DEFAULT_RESAMPLES = 10000
CROSS_SCHEMA = "phishme-v3-cross-dataset-v1"
VALIDATION_SCHEMA = "phishme-v3-validation-v1"


def run_v3_train(
    train_frame,
    validation_frame,
    *,
    test_frame=None,
    output_dir,
    seed,
    epochs=3,
    batch_size=2048,
) -> dict:
    """Fit every variant on PhishPedia train, select thresholds on validation, export artifacts.

    Writes under output_dir: validation-report.json, models/<variant>.joblib,
    model-linear.json (browser artifact), artifacts manifest. Returns a summary dict
    with per-variant validation metrics and selected thresholds.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    models_dir = output_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    val_labels = validation_frame["label"].to_numpy(dtype=np.int8)
    val_sample_ids = validation_frame["sample_id"].tolist()
    include_dom = True

    variants = {}

    # -- linear: grid over LINEAR_ALPHAS --------------------------------
    linear_best = None
    linear_best_key = None
    for alpha in LINEAR_ALPHAS:
        config = train.TrainConfig(alpha=alpha, epochs=epochs, batch_size=batch_size, seed=seed)
        model = train.fit_incremental(train_frame, config, include_dom=include_dom)
        scores = train.predict_scores(model, validation_frame, include_dom=include_dom)
        ap = evaluate._average_precision(val_labels, scores)
        f1 = evaluate._f1(val_labels, scores, 0.5)
        key = (ap, f1, -alpha)
        if linear_best is None or key > linear_best_key:
            linear_best = (model, config, scores)
            linear_best_key = key
    linear_model, linear_config, linear_scores = linear_best

    # -- tree: grid over TREE_ESTIMATOR_GRID ----------------------------
    tree_best = None
    tree_best_key = None
    for n_est in TREE_ESTIMATOR_GRID:
        config = models.TreeConfig(n_estimators=n_est, seed=seed)
        model = models.fit_tree(train_frame, config, include_dom=include_dom)
        scores = models.predict_tree_scores(model, validation_frame, include_dom=include_dom)
        ap = evaluate._average_precision(val_labels, scores)
        f1 = evaluate._f1(val_labels, scores, 0.5)
        key = (ap, f1, -n_est)
        if tree_best is None or key > tree_best_key:
            tree_best = (model, config, scores)
            tree_best_key = key
    tree_model, tree_config, tree_scores = tree_best

    # -- hybrid: single config ------------------------------------------
    hybrid_cfg = models.HybridConfig(seed=seed)
    hybrid_linear_cfg = train.TrainConfig(alpha=1e-4, epochs=epochs, batch_size=batch_size, seed=seed)
    hybrid_tree_cfg = models.TreeConfig(n_estimators=300, seed=seed)
    hybrid_model = models.fit_hybrid(
        train_frame,
        hybrid_cfg,
        include_dom=include_dom,
        linear_config=hybrid_linear_cfg,
        tree_config=hybrid_tree_cfg,
    )
    hybrid_scores = models.predict_hybrid_scores(hybrid_model, validation_frame, include_dom=include_dom)

    # -- thresholds and metrics -----------------------------------------
    variant_data = [
        ("linear", linear_model, linear_scores, linear_config, linear_best_key),
        ("tree", tree_model, tree_scores, tree_config, tree_best_key),
        ("hybrid", hybrid_model, hybrid_scores, hybrid_cfg, None),
    ]

    for kind, model, scores, config, _best_key in variant_data:
        threshold = evaluate.select_threshold(val_labels, scores)
        metrics = evaluate.metrics_report(val_labels, scores, threshold)
        base_rates = benchmark.evaluate_base_rates(
            val_labels, scores, threshold,
            benchmark.BASE_RATE_PREVALENCES, seed,
            sample_ids=val_sample_ids,
        )
        variants[kind] = {
            "selected": _config_summary(kind, config),
            "threshold": threshold,
            "threshold_source": "validation",
            "validation_metrics": metrics,
            "validation_base_rates": base_rates,
        }
        if test_frame is not None and len(test_frame) > 0:
            test_labels = test_frame["label"].to_numpy(dtype=np.int8)
            if kind == "linear":
                test_scores = train.predict_scores(model, test_frame, include_dom=include_dom)
            elif kind == "tree":
                test_scores = models.predict_tree_scores(model, test_frame, include_dom=include_dom)
            else:
                test_scores = models.predict_hybrid_scores(model, test_frame, include_dom=include_dom)
            variants[kind]["holdout_metrics"] = evaluate.metrics_report(
                test_labels, test_scores, threshold,
            )

    primary_kind = max(
        VARIANT_KINDS,
        key=lambda kind: (
            variants[kind]["validation_metrics"]["average_precision"],
            variants[kind]["validation_metrics"]["f1"],
            -variants[kind]["validation_metrics"]["brier"],
            -VARIANT_KINDS.index(kind),
        ),
    )
    primary_variant = {
        "kind": primary_kind,
        "selection_source": "validation",
        "ranking": ["average_precision_desc", "f1_desc", "brier_asc", "variant_order"],
    }

    # -- persist models -------------------------------------------------
    linear_model_path = models_dir / "linear.joblib"
    tree_model_path = models_dir / "tree.joblib"
    hybrid_model_path = models_dir / "hybrid.joblib"
    _dump_joblib_atomically(linear_model, linear_model_path)
    _dump_joblib_atomically(tree_model, tree_model_path)
    _dump_joblib_atomically(hybrid_model, hybrid_model_path)

    # -- browser artifact for linear ------------------------------------
    metadata = {"seed": seed, "variant": "linear", "dataset": "phishpedia"}
    linear_json_path = output_path / "model-linear.json"
    export.export_model(linear_model, variants["linear"]["threshold"], True, metadata, linear_json_path)

    # -- validation report + manifest -----------------------------------
    artifact_paths = {
        "linear": linear_model_path,
        "tree": tree_model_path,
        "hybrid": hybrid_model_path,
        "linear_browser": linear_json_path,
    }
    manifest = phresh.artifact_manifest(artifact_paths)

    report = {
        "schema": VALIDATION_SCHEMA,
        "seed": int(seed),
        "primary_variant": primary_variant,
        "variants": variants,
        "artifacts": manifest,
    }
    report_path = output_path / "validation-report.json"
    phresh.write_json_atomically(report, report_path)

    return {
        "primary_variant": primary_variant,
        "variants": variants,
        "output_dir": str(output_path),
    }


def materialize_phresh_test(
    raw_rows: Iterator[dict],
    output_path,
    *,
    limit: int | None = None,
) -> dict:
    """Stream raw PhreshPhish rows and write a frozen JSONL.

    Accepts an ITERATOR OF RAW ROWS (dicts with keys url, html, date, label).
    Each row is canonicalized via phresh._canonicalize_raw_phresh_row.
    Rejected records are counted and skipped. Accepted records include the
    'html' field and dom_* feature columns.

    Returns {"processed": int, "accepted": int, "rejected": int}.
    """
    destination = Path(output_path)
    if destination.exists():
        raise ValueError(f"output path already exists: {destination}")

    iterator = iter(raw_rows)
    try:
        first = next(iterator)
    except StopIteration:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("", encoding="utf-8")
        return {"processed": 0, "accepted": 0, "rejected": 0}

    phresh._validate_raw_schema(first)

    processed = 0
    accepted = 0
    rejected = 0

    def _rows():
        nonlocal processed
        yield first
        for row in iterator:
            if limit is not None and processed >= limit:
                return
            yield row

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")

    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for row in _rows():
                if limit is not None and processed >= limit:
                    break
                processed += 1
                canonical = phresh._canonicalize_raw_phresh_row(row)
                if "_reject_reason" in canonical:
                    rejected += 1
                    continue
                canonical["html"] = str(row.get("html", ""))
                accepted += 1
                # Reorder keys for stable output
                ordered = _order_frozen_record(canonical)
                handle.write(json.dumps(ordered, separators=(",", ":"), sort_keys=True, allow_nan=False))
                handle.write("\n")
                handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, destination)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return {"processed": processed, "accepted": accepted, "rejected": rejected}


def score_frozen_records(variant_models, records_path, *, batch_size=4096) -> dict:
    """Score every variant on the frozen records file.

    Returns {variant: {"labels": [...], "scores": [...], "sample_ids": [...]}}.
    """
    frame = _read_jsonl_frame(Path(records_path))
    labels = frame["label"].to_numpy(dtype=np.int8)
    sample_ids = frame["sample_id"].tolist()
    include_dom = True

    result = {}
    for kind in VARIANT_KINDS:
        model = variant_models["variants"][kind]["model"]
        if kind == "linear":
            scores = train.predict_scores(model, frame, include_dom=include_dom, batch_size=batch_size)
        elif kind == "tree":
            scores = models.predict_tree_scores(model, frame, include_dom=include_dom, batch_size=batch_size)
        else:
            scores = models.predict_hybrid_scores(model, frame, include_dom=include_dom, batch_size=batch_size)
        result[kind] = {
            "labels": labels.tolist(),
            "scores": scores.tolist(),
            "sample_ids": sample_ids,
        }
    return result


def run_v3_eval(
    variant_models,
    validation_metrics,
    records_path,
    phishlang_csv,
    *,
    output_dir,
    seed,
    resamples=DEFAULT_RESAMPLES,
) -> dict:
    """Compute PhreshPhish metrics per variant, ΔAP/ΔF1 vs validation, low-base-rate reports,
    align PhishLang predictions CSV by sample_id, compute PhishLang metrics + paired bootstrap
    for each variant vs PhishLang. Writes cross-dataset-report.json and returns the dict.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frame = _read_jsonl_frame(Path(records_path))
    labels = frame["label"].to_numpy(dtype=np.int8)
    sample_ids = frame["sample_id"].tolist()
    include_dom = True

    variants_report = {}
    for kind in VARIANT_KINDS:
        model = variant_models["variants"][kind]["model"]
        threshold = variant_models["variants"][kind]["threshold"]

        if kind == "linear":
            scores = train.predict_scores(model, frame, include_dom=include_dom)
        elif kind == "tree":
            scores = models.predict_tree_scores(model, frame, include_dom=include_dom)
        else:
            scores = models.predict_hybrid_scores(model, frame, include_dom=include_dom)

        phresh_metrics = evaluate.metrics_report(labels, scores, threshold)
        delta = compute_delta(validation_metrics[kind], phresh_metrics)
        base_rates = benchmark.evaluate_base_rates(
            labels, scores, threshold,
            benchmark.BASE_RATE_PREVALENCES, seed,
            sample_ids=sample_ids,
        )

        variants_report[kind] = {
            "validation_metrics": validation_metrics[kind],
            "phresh_metrics": phresh_metrics,
            "delta": delta,
            "base_rates": base_rates,
        }

    # PhishLang alignment — per-variant scores reordered to CSV order
    phishlang_frame = pd.read_csv(Path(phishlang_csv))
    phresh_labels_arr = np.array(labels, dtype=np.int8)

    paired = {}
    phishlang_metrics = None
    for kind in VARIANT_KINDS:
        model = variant_models["variants"][kind]["model"]
        threshold = variant_models["variants"][kind]["threshold"]

        if kind == "linear":
            variant_scores = train.predict_scores(model, frame, include_dom=include_dom)
        elif kind == "tree":
            variant_scores = models.predict_tree_scores(model, frame, include_dom=include_dom)
        else:
            variant_scores = models.predict_hybrid_scores(model, frame, include_dom=include_dom)

        aligned_for_variant = align_phishlang_scores(
            phishlang_frame, sample_ids, variant_scores, phresh_labels_arr,
        )
        paired[kind] = evaluate.paired_bootstrap(
            aligned_for_variant["labels"], aligned_for_variant["left"],
            aligned_for_variant["right"],
            threshold, aligned_for_variant["phishlang_threshold"],
            seed, resamples,
        )
        if phishlang_metrics is None:
            phishlang_metrics = evaluate.metrics_report(
                aligned_for_variant["labels"], aligned_for_variant["right"],
                aligned_for_variant["phishlang_threshold"],
            )

    report = {
        "schema": CROSS_SCHEMA,
        "seed": int(seed),
        "variants": variants_report,
        "phishlang": {
            "threshold": 0.5,
            "threshold_source": "official_fixed_0.5",
            "metrics": phishlang_metrics,
            "paired_bootstrap": paired,
        },
    }
    report_path = output_path / "cross-dataset-report.json"
    phresh.write_json_atomically(report, report_path)

    return report


def align_phishlang_scores(
    phishlang_frame,
    sample_ids,
    phresh_scores,
    phresh_labels,
) -> dict:
    """Align phishMe scores to the PhishLang CSV row order by sample_id.

    Returns {"labels", "left" (phishMe scores in CSV order), "right" (PhishLang scores),
    "phishlang_threshold": 0.5}. Raises ValueError on sample_id set mismatch or label conflict.
    """
    csv_ids = list(phishlang_frame["sample_id"])
    csv_label_map = {}
    csv_score_map = {}
    for _, row in phishlang_frame.iterrows():
        sid = str(row["sample_id"])
        csv_label_map[sid] = int(row["label"])
        csv_score_map[sid] = float(row["score"])

    csv_id_set = set(csv_ids)
    phresh_id_set = set(sample_ids)
    if csv_id_set != phresh_id_set:
        missing = sorted(phresh_id_set - csv_id_set)
        extra = sorted(csv_id_set - phresh_id_set)
        details = []
        if missing:
            details.append(f"missing from CSV: {missing}")
        if extra:
            details.append(f"extra in CSV: {extra}")
        raise ValueError("sample_id set mismatch: " + "; ".join(details))

    phresh_id_to_label = dict(zip(sample_ids, phresh_labels))
    phresh_id_to_score = dict(zip(sample_ids, phresh_scores))

    for sid in csv_ids:
        if csv_label_map[sid] != phresh_id_to_label[sid]:
            raise ValueError(
                f"label conflict for sample_id {sid!r}: "
                f"CSV label={csv_label_map[sid]}, phresh label={phresh_id_to_label[sid]}"
            )

    left = np.array([phresh_id_to_score[sid] for sid in csv_ids], dtype=np.float64)
    right = np.array([csv_score_map[sid] for sid in csv_ids], dtype=np.float64)
    labels = np.array([csv_label_map[sid] for sid in csv_ids], dtype=np.int8)

    return {
        "labels": labels,
        "left": left,
        "right": right,
        "phishlang_threshold": 0.5,
    }


def compute_delta(validation_metrics, phresh_metrics) -> dict:
    """Return {"delta_ap": val_ap - phresh_ap, "delta_f1": val_f1 - phresh_f1}."""
    return {
        "delta_ap": validation_metrics["average_precision"] - phresh_metrics["average_precision"],
        "delta_f1": validation_metrics["f1"] - phresh_metrics["f1"],
    }


# ---- internal helpers --------------------------------------------------------


def _config_summary(kind: str, config) -> dict:
    if kind == "linear":
        return {"kind": kind, "alpha": config.alpha}
    elif kind == "tree":
        return {"kind": kind, "n_estimators": config.n_estimators, "num_leaves": config.num_leaves,
                "learning_rate": config.learning_rate}
    else:
        return {"kind": kind, "meta_c": config.meta_c, "oof_folds": config.oof_folds}


def _dump_joblib_atomically(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        joblib.dump(model, tmp)
        fd = os.open(tmp, os.O_RDWR)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _order_frozen_record(record: Mapping) -> dict:
    """Return a dict with keys in a stable order for JSONL output."""
    ordered_keys = [
        "url", "title", "date", "label", "sample_id", "html", "_parse_status",
    ] + [f"dom_{name}" for name in features.NUMERIC_FEATURES]
    ordered = {}
    for key in ordered_keys:
        if key in record:
            ordered[key] = record[key]
    for key in sorted(record):
        if key not in ordered:
            ordered[key] = record[key]
    return ordered


def _read_jsonl_frame(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"frozen JSONL is empty: {path}")
    return pd.DataFrame(records)
