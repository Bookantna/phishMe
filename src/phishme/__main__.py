from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from importlib import metadata
from numbers import Integral, Real
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from phishme import __version__
from phishme.data import load_phiusill, split_local
from phishme.evaluate import metrics_report, select_threshold
from phishme.export import export_model
from phishme.features import FEATURE_VERSION
from phishme.train import TrainConfig, fit_incremental, predict_scores

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None


ALPHAS = (1e-5, 1e-4, 1e-3)
INCLUDE_DOM_OPTIONS = (False, True)
SPLIT_NAMES = ("train", "validation", "test")
REPORT_ARTIFACTS = ("splits.json", "model.json", "test-metrics.json")
MAX_JSON_DEPTH = 12
UNSAFE_JSON_KEYS = {"__proto__", "constructor", "prototype"}
PIPELINE_ERRORS = (
    ArithmeticError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    pd.errors.ParserError,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except PIPELINE_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m phishme")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="audit a PhiUSIIL CSV and planned local split")
    audit.add_argument("--csv", required=True, type=Path, help="path to PhiUSIIL CSV")
    audit.add_argument("--seed", default=42, type=int, help="split seed")
    audit.set_defaults(func=_cmd_audit)

    local = subparsers.add_parser("local", help="run the reproducible local baseline")
    local.add_argument("--csv", required=True, type=Path, help="path to PhiUSIIL CSV")
    local.add_argument("--output", required=True, type=Path, help="directory for JSON artifacts")
    local.add_argument("--epochs", default=3, type=int, help="epochs for each candidate")
    local.add_argument("--batch-size", default=2048, type=int, help="SGD mini-batch size")
    local.add_argument("--seed", default=42, type=int, help="split and training seed")
    local.set_defaults(func=_cmd_local)

    phresh_smoke = subparsers.add_parser(
        "phresh-smoke",
        help="run a bounded pinned PhreshPhish streaming smoke train",
    )
    phresh_smoke.add_argument("--limit", default=1000, type=int, help="maximum train rows to stream")
    phresh_smoke.add_argument("--output", required=True, type=Path, help="directory for smoke artifacts")
    phresh_smoke.add_argument("--alpha", default=1e-4, type=float, help="SGD L2 regularization")
    phresh_smoke.add_argument("--batch-size", default=256, type=int, help="streaming SGD batch size")
    phresh_smoke.add_argument("--seed", default=42, type=int, help="training seed")
    phresh_smoke.add_argument(
        "--no-resume",
        action="store_true",
        help="fail if a compatible smoke checkpoint already exists",
    )
    phresh_smoke.set_defaults(func=_cmd_phresh_smoke)

    v3_train = subparsers.add_parser("v3-train", help="train PhishPedia V3 variants and validate")
    v3_train.add_argument("--csv", required=True, type=Path, help="PhishPedia split CSV")
    v3_train.add_argument("--phish-html", required=True, type=Path, help="phishing HTML root")
    v3_train.add_argument("--benign-html", required=True, type=Path, help="benign HTML root")
    v3_train.add_argument("--output", required=True, type=Path, help="directory for V3 artifacts")
    v3_train.add_argument("--limit", default=None, type=int, help="optional row cap (smoke)")
    v3_train.add_argument("--epochs", default=3, type=int, help="epochs for linear model")
    v3_train.add_argument("--batch-size", default=2048, type=int, help="SGD mini-batch size")
    v3_train.add_argument("--seed", default=42, type=int, help="training and split seed")
    v3_train.set_defaults(func=_cmd_v3_train)

    v3_eval = subparsers.add_parser("v3-eval", help="evaluate V3 variants on frozen PhreshPhish test")
    v3_eval.add_argument("--output", required=True, type=Path, help="V3 run directory (from v3-train)")
    v3_eval.add_argument("--phishlang-csv", required=True, type=Path, help="frozen PhishLang predictions CSV")
    v3_eval.add_argument("--records", default=None, type=Path, help="frozen PhreshPhish test records JSONL (optional; streams if omitted)")
    v3_eval.add_argument("--resamples", default=10000, type=int, help="paired bootstrap resamples")
    v3_eval.add_argument("--limit", default=None, type=int, help="optional row cap for streaming (only when --records omitted)")
    v3_eval.add_argument("--seed", default=42, type=int, help="bootstrap seed")
    v3_eval.set_defaults(func=_cmd_v3_eval)

    return parser


def _cmd_audit(args: argparse.Namespace) -> None:
    csv_path = _validate_csv_path(args.csv)
    seed = _validate_integer("seed", args.seed, minimum=0)
    raw = _read_raw_csv(csv_path)
    dataset_sha256 = _sha256_file(csv_path)

    canonical = load_phiusill(csv_path)
    splits = split_local(canonical, seed=seed)
    split_report = _split_report(splits, seed)

    report = {
        "schema": "phishme-audit-v1",
        "command": "audit",
        "normalized_arguments": {
            "command": "audit",
            "csv": str(csv_path),
            "seed": seed,
        },
        "dataset": {
            "path": str(csv_path),
            "sha256": dataset_sha256,
        },
        "counts": {
            "raw_rows": len(raw),
            "canonical_rows": len(canonical),
            "duplicate_rows": int(len(raw) - len(canonical)),
        },
        "label_counts": {
            "raw": _raw_label_counts(raw),
            "canonical": _label_counts(canonical),
            "classes": _class_counts(canonical),
        },
        "splits": split_report["splits"],
        "disjointness": split_report["disjointness"],
    }
    _print_json(report)


def _cmd_local(args: argparse.Namespace) -> None:
    csv_path = _validate_csv_path(args.csv)
    epochs = _validate_integer("epochs", args.epochs, minimum=1)
    batch_size = _validate_integer("batch_size", args.batch_size, minimum=1)
    seed = _validate_integer("seed", args.seed, minimum=0)
    output_dir = _validate_output_dir(args.output)
    paths = _artifact_paths(output_dir)
    dataset_sha256 = _sha256_file(csv_path)

    frame = load_phiusill(csv_path)
    splits = split_local(frame, seed=seed)
    operation_counts = {
        "load_phiusill": 1,
        "split_local": 1,
        "test_predict_scores": 0,
    }
    split_report = _split_report(splits, seed)

    successful, failures, models = _train_validation_candidates(
        splits["train"],
        splits["validation"],
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    if not successful:
        raise RuntimeError(f"all validation candidates failed: {failures}")

    selected = max(
        successful,
        key=lambda candidate: (
            candidate["validation_ap"],
            candidate["validation_f1"],
            -candidate["alpha"],
            -int(candidate["include_dom"]),
        ),
    )
    selected_model = models[selected["candidate_id"]]

    test_scores = predict_scores(
        selected_model,
        splits["test"],
        include_dom=selected["include_dom"],
        batch_size=batch_size,
    )
    operation_counts["test_predict_scores"] += 1
    test_metrics = metrics_report(splits["test"]["label"].to_numpy(), test_scores, selected["threshold"])
    test_scored_once = operation_counts["test_predict_scores"] == 1

    selected_config = _selected_config(selected, epochs=epochs, batch_size=batch_size, seed=seed)
    common_run_fields = {
        "schema": "phishme-local-run-v1",
        "command": "local",
        "normalized_arguments": {
            "command": "local",
            "csv": str(csv_path),
            "output": str(output_dir),
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
        },
        "dataset": {
            "path": str(csv_path),
            "sha256": dataset_sha256,
            "canonical_rows": len(frame),
        },
        "feature_version": FEATURE_VERSION,
        "seed": seed,
        "git_commit": _git_commit(),
        "versions": _runtime_versions(),
        "operation_counts": operation_counts,
        "test_scored_once": test_scored_once,
        "candidate_grid": {
            "include_dom": list(INCLUDE_DOM_OPTIONS),
            "alpha": list(ALPHAS),
        },
        "candidate_attempt_count": len(INCLUDE_DOM_OPTIONS) * len(ALPHAS),
        "candidate_failure_count": len(failures),
        "candidate_failures": failures,
        "validation_candidates": successful,
        "selected_config": selected_config,
    }

    _write_json_atomically(
        {
            "schema": "phishme-splits-v1",
            "dataset": common_run_fields["dataset"],
            "seed": seed,
            **split_report,
        },
        paths["splits.json"],
    )
    _write_json_atomically(
        {
            "schema": "phishme-test-metrics-v1",
            "dataset": common_run_fields["dataset"],
            "feature_version": FEATURE_VERSION,
            "seed": seed,
            "threshold_source": "validation",
            "selected_config": selected_config,
            **test_metrics,
        },
        paths["test-metrics.json"],
    )
    export_model(
        selected_model,
        selected["threshold"],
        selected["include_dom"],
        {
            "dataset_revision": dataset_sha256,
            "dataset_sha256": dataset_sha256,
            "feature_version": FEATURE_VERSION,
            "seed": seed,
            "selected_config": selected_config,
            "validation_metrics": selected["validation_metrics"],
        },
        paths["model.json"],
    )

    artifacts = _artifact_manifest(paths, REPORT_ARTIFACTS)
    _write_json_atomically({**common_run_fields, "artifacts": artifacts}, paths["run.json"])

    run_artifacts = {
        **artifacts,
        "run.json": _file_manifest(paths["run.json"]),
    }
    _print_json(
        {
            "schema": "phishme-local-summary-v1",
            "output": str(output_dir),
            "selected_config": selected_config,
            "artifacts": run_artifacts,
        }
    )


def _cmd_phresh_smoke(args: argparse.Namespace) -> None:
    from phishme import phresh

    limit = _validate_integer("limit", args.limit, minimum=1)
    batch_size = _validate_integer("batch_size", args.batch_size, minimum=1)
    seed = _validate_integer("seed", args.seed, minimum=0)
    output_dir = _validate_phresh_output_dir(args.output)
    paths = _phresh_smoke_paths(output_dir)
    limitations = [
        "bounded smoke sample only; not a benchmark",
        "threshold 0.5 is smoke-only and not validation-selected",
        "no official test metrics are claimed by phresh-smoke",
    ]

    cutoff = phresh.derive_temporal_cutoff(phresh.PHRESH_REVISION)
    config = phresh.PhreshTrainConfig(
        alpha=float(args.alpha),
        batch_size=batch_size,
        seed=seed,
        include_dom=True,
        split="train",
        cutoff=cutoff,
    )
    model = phresh.train_stream(
        lambda: _phresh_smoke_train_records(phresh, limit=limit, cutoff=cutoff),
        paths["checkpoints"],
        config,
        resume=not bool(args.no_resume),
    )
    _, checkpoint = phresh.validate_checkpoint(paths["checkpoint.json"], config)

    export_model(
        model,
        0.5,
        True,
        {
            "dataset": phresh.PHRESH_DATASET,
            "dataset_revision": phresh.PHRESH_REVISION,
            "feature_version": FEATURE_VERSION,
            "seed": seed,
            "split": "train",
            "cutoff": cutoff,
            "checkpoint": {
                "processed_position": checkpoint["processed_position"],
                "model_joblib": checkpoint["model_joblib"],
                "model_sha256": checkpoint["model_sha256"],
            },
            "threshold_source": "smoke_only_not_validation_selected",
            "limitations": limitations,
        },
        paths["model.json"],
    )

    artifacts = {
        "model.json": _file_manifest(paths["model.json"]),
        "checkpoints/checkpoint.json": _file_manifest(paths["checkpoint.json"]),
        f"checkpoints/{checkpoint['model_joblib']}": _file_manifest(
            paths["checkpoints"] / checkpoint["model_joblib"]
        ),
    }
    manifest = {
        "schema": "phishme-phresh-smoke-run-v1",
        "command": "phresh-smoke",
        "normalized_arguments": {
            "command": "phresh-smoke",
            "limit": limit,
            "output": str(output_dir),
            "alpha": float(config.alpha),
            "batch_size": batch_size,
            "seed": seed,
            "resume": not bool(args.no_resume),
        },
        "dataset": {
            "name": phresh.PHRESH_DATASET,
            "revision": phresh.PHRESH_REVISION,
            "split": "train",
        },
        "feature_version": FEATURE_VERSION,
        "cutoff": {
            "value": cutoff,
            "percentile": phresh.CUTOFF_PERCENTILE,
            "index_rule": phresh.CUTOFF_INDEX_RULE,
        },
        "training": {
            "processed_position": checkpoint["processed_position"],
            "accepted_examples": checkpoint["class_counts"]["0"] + checkpoint["class_counts"]["1"],
            "class_counts": checkpoint["class_counts"],
            "parse_counts": checkpoint["parse_counts"],
            "reject_counts": checkpoint["reject_counts"],
            "checkpoint": checkpoint,
        },
        "model": {
            "threshold": 0.5,
            "threshold_source": "smoke_only_not_validation_selected",
        },
        "limitations": limitations,
        "artifacts": artifacts,
    }
    _write_json_atomically(manifest, paths["run.json"])
    _print_json(
        {
            "schema": "phishme-phresh-smoke-summary-v1",
            "output": str(output_dir),
            "processed_position": checkpoint["processed_position"],
            "accepted_examples": manifest["training"]["accepted_examples"],
            "artifacts": {
                **artifacts,
                "run.json": _file_manifest(paths["run.json"]),
            },
            "limitations": limitations,
        }
    )


def _phresh_smoke_train_records(phresh_module, *, limit: int, cutoff: str):
    base = phresh_module.iter_phresh(
        "train",
        revision=phresh_module.PHRESH_REVISION,
        limit=limit,
    )
    return phresh_module.filter_by_cutoff(base, cutoff, keep="before")


def _cmd_v3_train(args: argparse.Namespace) -> None:
    from phishme import cross_dataset, phishpedia

    csv_path = _validate_csv_path(args.csv)
    phish_html = _validate_dir_path("phish-html", args.phish_html)
    benign_html = _validate_dir_path("benign-html", args.benign_html)
    output_dir = _validate_phresh_output_dir(args.output)
    epochs = _validate_integer("epochs", args.epochs, minimum=1)
    batch_size = _validate_integer("batch-size", args.batch_size, minimum=1)
    seed = _validate_integer("seed", args.seed, minimum=0)
    limit = args.limit

    frame = phishpedia.load_phishpedia(
        csv_path, phish_html, benign_html, limit=limit,
    )

    # Split the frame
    if "split" in frame.columns:
        train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
        validation_frame = frame[frame["split"] == "validation"].reset_index(drop=True)
        extra = set(frame["split"]) - {"train", "validation", "test"}
        if extra:
            raise ValueError(f"unknown split values: {sorted(extra)}")
        split_source = "official_csv_split"
    else:
        splits = split_local(frame, seed=seed)
        train_frame = splits["train"]
        validation_frame = splits["validation"]
        split_source = "split_local_domain_grouped"

    result = cross_dataset.run_v3_train(
        train_frame, validation_frame,
        output_dir=output_dir, seed=seed,
        epochs=epochs, batch_size=batch_size,
    )

    # Read back the validation report to include split_source
    report_path = output_dir / "validation-report.json"
    report = json.loads(report_path.read_text())
    report["split_source"] = split_source
    _write_json_atomically(report, report_path)

    # Write run.json manifest
    run_path = output_dir / "run.json"
    artifacts = {
        "validation-report.json": _file_manifest(report_path),
        "models/linear.joblib": _file_manifest(output_dir / "models" / "linear.joblib"),
        "models/tree.joblib": _file_manifest(output_dir / "models" / "tree.joblib"),
        "models/hybrid.joblib": _file_manifest(output_dir / "models" / "hybrid.joblib"),
        "model-linear.json": _file_manifest(output_dir / "model-linear.json"),
    }
    manifest = {
        "schema": "phishme-v3-train-run-v1",
        "command": "v3-train",
        "normalized_arguments": {
            "command": "v3-train",
            "csv": str(csv_path),
            "phish_html": str(phish_html),
            "benign_html": str(benign_html),
            "output": str(output_dir),
            "limit": limit,
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
        },
        "split_source": split_source,
        "feature_version": FEATURE_VERSION,
        "seed": seed,
        "git_commit": _git_commit(),
        "versions": _runtime_versions(),
        "artifacts": artifacts,
    }
    _write_json_atomically(manifest, run_path)

    _print_json({
        "schema": "phishme-v3-train-summary-v1",
        "output": str(output_dir),
        "split_source": split_source,
        "variants": list(result["variants"]),
        "artifacts": {
            **artifacts,
            "run.json": _file_manifest(run_path),
        },
    })


def _cmd_v3_eval(args: argparse.Namespace) -> None:
    from phishme import cross_dataset, phresh

    output_dir = _validate_phresh_output_dir(args.output)
    phishlang_csv = _validate_csv_path(args.phishlang_csv)
    resamples = _validate_integer("resamples", args.resamples, minimum=1)
    seed = _validate_integer("seed", args.seed, minimum=0)
    records_path = args.records
    limit = args.limit

    # Read validation-report.json
    val_report_path = output_dir / "validation-report.json"
    if not val_report_path.exists():
        raise ValueError(f"validation-report.json not found in {output_dir}")
    val_report = json.loads(val_report_path.read_text())

    # Reconstruct variant_models from disk
    variant_models = {"variants": {}}
    validation_metrics = {}
    for kind in cross_dataset.VARIANT_KINDS:
        variant = val_report["variants"][kind]
        model_path = output_dir / "models" / f"{kind}.joblib"
        if not model_path.exists():
            raise ValueError(f"model not found: {model_path}")
        model = joblib.load(model_path)
        variant_models["variants"][kind] = {
            "model": model,
            "threshold": variant["threshold"],
        }
        validation_metrics[kind] = variant["validation_metrics"]

    # Materialize records if --records not given
    if records_path is None:
        raw_stream = phresh._load_dataset(
            phresh.PHRESH_DATASET, split="test",
            revision=phresh.PHRESH_REVISION, streaming=True,
        )
        materialized_path = output_dir / "phresh-test-records.jsonl"
        materialize_result = cross_dataset.materialize_phresh_test(
            iter(raw_stream), materialized_path, limit=limit,
        )
        records_path = materialized_path
        if materialize_result["accepted"] == 0:
            raise ValueError("no accepted PhreshPhish test records")
    else:
        records_path = _validate_records_path(args.records)

    result = cross_dataset.run_v3_eval(
        variant_models, validation_metrics,
        records_path, phishlang_csv,
        output_dir=output_dir, seed=seed, resamples=resamples,
    )

    _print_json({
        "schema": "phishme-v3-eval-summary-v1",
        "output": str(output_dir),
        "variants": list(result["variants"]),
    })


def _validate_dir_path(name: str, path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.is_dir():
        raise ValueError(f"--{name} must be an existing directory: {resolved}")
    return resolved


def _train_validation_candidates(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[list[dict], list[dict], dict[str, object]]:
    successful = []
    failures = []
    models: dict[str, object] = {}

    for include_dom in INCLUDE_DOM_OPTIONS:
        for alpha in ALPHAS:
            candidate_id = _candidate_id(include_dom, alpha)
            try:
                config = TrainConfig(alpha=alpha, epochs=epochs, batch_size=batch_size, seed=seed)
                model = fit_incremental(train_frame, config, include_dom=include_dom)
                validation_scores = predict_scores(
                    model,
                    validation_frame,
                    include_dom=include_dom,
                    batch_size=batch_size,
                )
                threshold = select_threshold(
                    validation_frame["label"].to_numpy(),
                    validation_scores,
                )
                validation_metrics = metrics_report(
                    validation_frame["label"].to_numpy(),
                    validation_scores,
                    threshold,
                )
                validation_ap = float(validation_metrics["average_precision"])
                validation_f1 = float(validation_metrics["f1"])
                candidate = {
                    "candidate_id": candidate_id,
                    "include_dom": include_dom,
                    "alpha": float(alpha),
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "seed": seed,
                    "threshold": threshold,
                    "threshold_source": "validation",
                    "validation_ap": validation_ap,
                    "validation_f1": validation_f1,
                    "validation_metrics": validation_metrics,
                    "rank_key_fields": [
                        "validation_ap",
                        "validation_f1",
                        "-alpha",
                        "-include_dom",
                    ],
                    "rank_key": [
                        validation_ap,
                        validation_f1,
                        -float(alpha),
                        -int(include_dom),
                    ],
                }
                successful.append(candidate)
                models[candidate_id] = model
            except PIPELINE_ERRORS as exc:
                failures.append(
                    {
                        "candidate_id": candidate_id,
                        "include_dom": include_dom,
                        "alpha": float(alpha),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
    return successful, failures, models


def _selected_config(candidate: Mapping, *, epochs: int, batch_size: int, seed: int) -> dict:
    return {
        "candidate_id": candidate["candidate_id"],
        "include_dom": candidate["include_dom"],
        "alpha": candidate["alpha"],
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "threshold": candidate["threshold"],
        "threshold_source": "validation",
        "validation_ap": candidate["validation_ap"],
        "validation_f1": candidate["validation_f1"],
        "rank_key_fields": list(candidate["rank_key_fields"]),
        "rank_key": list(candidate["rank_key"]),
    }


def _split_report(parts: Mapping[str, pd.DataFrame], seed: int) -> dict:
    return {
        "seed": seed,
        "splits": {name: _split_summary(parts[name]) for name in SPLIT_NAMES},
        "disjointness": _split_disjointness(parts),
    }


def _split_summary(frame: pd.DataFrame) -> dict:
    label_counts = _label_counts(frame)
    return {
        "rows": len(frame),
        "label_counts": label_counts,
        "class_counts": _class_counts(frame),
        "groups": int(frame["group"].nunique()),
        "sample_manifest_sha256": _sample_manifest_sha256(frame["sample_id"]),
    }


def _split_disjointness(parts: Mapping[str, pd.DataFrame]) -> dict:
    sample_intersections = {}
    group_intersections = {}
    for left_index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[left_index + 1 :]:
            pair = f"{left_name}_{right_name}"
            sample_intersections[pair] = len(
                set(parts[left_name]["sample_id"]) & set(parts[right_name]["sample_id"])
            )
            group_intersections[pair] = len(
                set(parts[left_name]["group"]) & set(parts[right_name]["group"])
            )
    return {
        "sample_id_disjoint": all(value == 0 for value in sample_intersections.values()),
        "group_disjoint": all(value == 0 for value in group_intersections.values()),
        "sample_id_intersections": sample_intersections,
        "group_intersections": group_intersections,
    }


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["label"].value_counts().to_dict()
    return {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}


def _raw_label_counts(frame: pd.DataFrame) -> dict[str, int]:
    if "label" not in frame:
        return {}
    counts = frame["label"].fillna("<NA>").astype(str).value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
    labels = _label_counts(frame)
    return {"benign": labels["0"], "phishing": labels["1"]}


def _sample_manifest_sha256(sample_ids: pd.Series) -> str:
    manifest = "\n".join(sorted(str(value) for value in sample_ids))
    if manifest:
        manifest += "\n"
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def _artifact_manifest(paths: Mapping[str, Path], names: Sequence[str]) -> dict[str, dict[str, object]]:
    return {name: _file_manifest(paths[name]) for name in names}


def _file_manifest(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_raw_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"label": "string"})


def _validate_csv_path(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"--csv does not exist: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"--csv must be a file: {resolved}")
    return resolved


def _validate_records_path(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"--records does not exist: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"--records must be a file: {resolved}")
    return resolved


def _validate_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"--output must be a directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    for artifact_path in _artifact_paths(resolved).values():
        _ensure_child(resolved, artifact_path)
    return resolved


def _validate_phresh_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"--output must be a directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "run.json": output_dir / "run.json",
        "splits.json": output_dir / "splits.json",
        "model.json": output_dir / "model.json",
        "test-metrics.json": output_dir / "test-metrics.json",
    }


def _phresh_smoke_paths(output_dir: Path) -> dict[str, Path]:
    checkpoints = output_dir / "checkpoints"
    return {
        "run.json": output_dir / "run.json",
        "model.json": output_dir / "model.json",
        "checkpoints": checkpoints,
        "checkpoint.json": checkpoints / "checkpoint.json",
    }


def _ensure_child(parent: Path, child: Path) -> None:
    parent_resolved = parent.resolve(strict=False)
    child_resolved = child.resolve(strict=False)
    if child_resolved.parent != parent_resolved:
        raise ValueError(f"artifact path escapes output directory: {child}")


def _validate_integer(name: str, value: int, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if minimum is not None and integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return integer


def _write_json_atomically(payload: Mapping, path: Path) -> None:
    safe_payload = _json_safe(payload)
    text = json.dumps(safe_payload, separators=(",", ":"), sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def _print_json(payload: Mapping) -> None:
    safe_payload = _json_safe(payload)
    print(json.dumps(safe_payload, separators=(",", ":"), sort_keys=True, allow_nan=False))


def _json_safe(value, *, depth: int = 0):
    if depth > MAX_JSON_DEPTH:
        raise ValueError("JSON payload is too deeply nested")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON numbers must be finite")
        return number
    if isinstance(value, np.floating):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON numbers must be finite")
        return number
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            if key in UNSAFE_JSON_KEYS or "\x00" in key:
                raise ValueError("JSON object contains unsafe key")
            output[key] = _json_safe(item, depth=depth + 1)
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item, depth=depth + 1) for item in value]
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _runtime_versions() -> dict:
    dependencies = {}
    for requirement in _declared_dependencies():
        name = _requirement_name(requirement)
        if not name:
            continue
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        dependencies[name] = {"declared": requirement, "installed": installed}

    return {
        "phishme": __version__,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "dependencies": dependencies,
    }


def _declared_dependencies() -> list[str]:
    root = _repo_root()
    pyproject = root / "pyproject.toml"
    if tomllib is not None and pyproject.exists():
        with pyproject.open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
        return [str(requirement) for requirement in project.get("dependencies", [])]
    return [str(requirement) for requirement in metadata.requires("phishme") or []]


def _requirement_name(requirement: str) -> str:
    return re.split(r"\s*(?:[<>=!~;\[])", requirement, maxsplit=1)[0].strip()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_id(include_dom: bool, alpha: float) -> str:
    return f"dom-{'on' if include_dom else 'off'}-alpha-{alpha:.0e}"


if __name__ == "__main__":
    raise SystemExit(main())
