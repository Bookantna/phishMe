from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from numbers import Integral, Real
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from .features import (
    FEATURE_VERSION,
    HASH_DIM,
    NUMERIC_FEATURES,
    _html_features_with_status,
    vectorize,
)
from .train import predict_scores

UTC = timezone.utc
PHRESH_DATASET = "phreshphish/phreshphish"
PHRESH_REVISION = "eabec4b7a66324b79cc8a0ad856d1731dc26fe1a"
PHRESH_SPLITS = frozenset(("train", "test"))

CHECKPOINT_SCHEMA = "phishme-phresh-checkpoint-v1"
CHECKPOINT_FILENAME = "checkpoint.json"
CUTOFF_COLUMNS = ["date", "label"]
CUTOFF_PERCENTILE = 0.90
CUTOFF_INDEX_RULE = "ceil(0.90 * n) - 1 over UTC-sorted timestamps"
CUTOFF_RETRY_MAX_ATTEMPTS = 3
CUTOFF_RETRY_INITIAL_DELAY_SECONDS = 1.0

_REQUIRED_RAW_COLUMNS = frozenset(("url", "html", "date", "label"))
_PROJECTED_COLUMNS = frozenset(CUTOFF_COLUMNS)
_LABELS = {"benign": 0, "phish": 1}
_CLASSES = np.array([0, 1], dtype=np.int8)
_RETRYABLE_BUILTIN_STREAM_ERRORS = (ConnectionError, TimeoutError, OSError)
_CLOSED_HF_CLIENT_RUNTIME_ERROR = "Cannot send a request, as the client has been closed."
_CHECKPOINT_KEYS = frozenset(
    (
        "schema",
        "processed_position",
        "model_joblib",
        "model_sha256",
        "feature_version",
        "dataset_revision",
        "split",
        "cutoff",
        "alpha",
        "batch_size",
        "class_counts",
        "parse_counts",
        "reject_counts",
        "seed",
        "include_dom",
        "retry_policy",
        "write_time_utc",
    )
)
_RETRY_POLICY_KEYS = frozenset(
    ("max_attempts", "initial_delay_seconds", "backoff_multiplier")
)
_UNSAFE_JSON_KEYS = frozenset(("__proto__", "constructor", "prototype"))


class _RetryableStreamError(RuntimeError):
    pass


@dataclass(frozen=True)
class PhreshTrainConfig:
    alpha: float = 1e-4
    batch_size: int = 2048
    seed: int = 42
    include_dom: bool = True
    split: str = "train"
    cutoff: str | None = None
    dataset_revision: str = PHRESH_REVISION
    retry_max_attempts: int = 3
    retry_initial_delay_seconds: float = 1.0
    retry_backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real) or self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not math.isfinite(float(self.alpha)):
            raise ValueError("alpha must be finite")
        _require_positive_integer("batch_size", self.batch_size)
        _require_non_negative_integer("seed", self.seed)
        if not isinstance(self.include_dom, bool):
            raise TypeError("include_dom must be a boolean")
        _validate_split(self.split)
        if self.dataset_revision != PHRESH_REVISION:
            raise ValueError("unsupported PhreshPhish revision")
        if self.cutoff is not None:
            object.__setattr__(
                self,
                "cutoff",
                _format_utc(_parse_utc_datetime(self.cutoff, "cutoff")),
            )
        attempts = _require_positive_integer("retry_max_attempts", self.retry_max_attempts)
        if attempts > 3:
            raise ValueError("retry_max_attempts must be at most 3")
        _require_non_negative_float(
            "retry_initial_delay_seconds",
            self.retry_initial_delay_seconds,
        )
        multiplier = _require_positive_float(
            "retry_backoff_multiplier",
            self.retry_backoff_multiplier,
        )
        if multiplier < 1.0:
            raise ValueError("retry_backoff_multiplier must be at least 1")

    def retry_policy(self) -> dict[str, int | float]:
        return {
            "max_attempts": int(self.retry_max_attempts),
            "initial_delay_seconds": float(self.retry_initial_delay_seconds),
            "backoff_multiplier": float(self.retry_backoff_multiplier),
        }


@dataclass
class _TrainingState:
    model: SGDClassifier
    committed_position: int
    class_counts: Counter[str]
    parse_counts: Counter[str]
    reject_counts: Counter[str]
    first_fit: bool


def iter_phresh(
    split: str,
    revision: str = PHRESH_REVISION,
    limit: int | None = None,
    *,
    load_dataset_func: Callable | None = None,
) -> Iterator[dict]:
    """Stream pinned PhreshPhish rows as canonical feature records.

    The raw `html` column is parsed into title and bounded DOM feature columns,
    then discarded before yielding each record.
    """

    _validate_split(split)
    _validate_revision(revision)
    limit_value = _validate_limit(limit)
    load_dataset = load_dataset_func or _load_dataset
    stream = load_dataset(PHRESH_DATASET, split=split, revision=revision, streaming=True)

    return _iter_canonical_phresh(stream, limit_value)


def derive_temporal_cutoff(
    revision: str = PHRESH_REVISION,
    *,
    load_dataset_func: Callable | None = None,
    retry_initial_delay_seconds: float = CUTOFF_RETRY_INITIAL_DELAY_SECONDS,
    retry_backoff_multiplier: float = 2.0,
) -> str:
    """Return the UTC chronological 90th-percentile cutoff for the train split.

    Index rule: sort all projected train timestamps in ascending UTC order and
    choose zero-based index `ceil(0.90 * n) - 1`, clamped into `[0, n - 1]`.
    """

    _validate_revision(revision)
    retry_delay = _require_non_negative_float(
        "retry_initial_delay_seconds",
        retry_initial_delay_seconds,
    )
    retry_multiplier = _require_positive_float(
        "retry_backoff_multiplier",
        retry_backoff_multiplier,
    )
    if retry_multiplier < 1.0:
        raise ValueError("retry_backoff_multiplier must be at least 1")
    load_dataset = load_dataset_func or _load_dataset
    failures = 0

    while True:
        try:
            return _derive_temporal_cutoff_once(load_dataset, revision)
        except Exception as exc:
            if not _is_retryable_stream_exception(exc):
                raise
            failures += 1
            if failures >= CUTOFF_RETRY_MAX_ATTEMPTS:
                raise RuntimeError(
                    "PhreshPhish cutoff stream failed after "
                    f"{CUTOFF_RETRY_MAX_ATTEMPTS} attempts"
                ) from exc
            _reset_huggingface_client()
            _sleep_before_retry_values(retry_delay, retry_multiplier, failures)


def _derive_temporal_cutoff_once(load_dataset: Callable, revision: str) -> str:
    stream = load_dataset(PHRESH_DATASET, split="train", revision=revision, streaming=True)
    projected = stream.select_columns(CUTOFF_COLUMNS)

    iterator = iter(projected)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("PhreshPhish train split is empty") from exc
    _validate_projected_schema(first)

    dates = [_projected_date(first)]
    for row in iterator:
        _canonical_label(row.get("label"))
        dates.append(_parse_utc_datetime(row.get("date"), "date"))

    dates.sort()
    index = max(0, min(len(dates) - 1, math.ceil(CUTOFF_PERCENTILE * len(dates)) - 1))
    return _format_utc(dates[index])


def filter_by_cutoff(records: Iterable[Mapping], cutoff: str, *, keep: str) -> Iterator[Mapping]:
    cutoff_dt = _parse_utc_datetime(cutoff, "cutoff")
    if keep not in {"before", "at_or_after"}:
        raise ValueError("keep must be 'before' or 'at_or_after'")

    for record in records:
        if isinstance(record, Mapping) and "_reject_reason" in record:
            yield record
            continue
        row_dt = _parse_utc_datetime(record.get("date"), "date")
        if (keep == "before" and row_dt < cutoff_dt) or (
            keep == "at_or_after" and row_dt >= cutoff_dt
        ):
            yield record


def train_stream(records, checkpoint_dir, config: PhreshTrainConfig, resume: bool) -> SGDClassifier:
    if not isinstance(config, PhreshTrainConfig):
        raise TypeError("config must be a PhreshTrainConfig")
    if not isinstance(resume, bool):
        raise TypeError("resume must be a boolean")
    if resume and not callable(records):
        raise TypeError("records must be callable when resume=True")

    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / CHECKPOINT_FILENAME
    _ensure_checkpoint_child(directory, checkpoint_path)
    factory = _records_factory(records)

    if checkpoint_path.exists():
        if not resume:
            raise ValueError("checkpoint exists; pass resume=True or use a new checkpoint directory")
        model, metadata = validate_checkpoint(checkpoint_path, config)
        state = _state_from_checkpoint(model, metadata)
    else:
        state = _new_state(config)

    failures_since_progress = 0
    while True:
        start_committed = state.committed_position
        try:
            _run_training_attempt(factory, directory, config, state)
            if state.first_fit:
                raise ValueError("no accepted PhreshPhish training records")
            return state.model
        except _RetryableStreamError as exc:
            if state.committed_position > start_committed:
                failures_since_progress = 0
            failures_since_progress += 1
            if failures_since_progress >= int(config.retry_max_attempts):
                raise RuntimeError(
                    "PhreshPhish stream failed after "
                    f"{config.retry_max_attempts} attempts; last checkpoint is preserved"
                ) from (exc.__cause__ or exc)
            _reset_huggingface_client()
            _sleep_before_retry(config, failures_since_progress)


def validate_checkpoint(
    path,
    config: PhreshTrainConfig | None = None,
) -> tuple[SGDClassifier, dict]:
    checkpoint_path = Path(path)
    payload = _read_strict_json(checkpoint_path)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")

    feature_version = payload.get("feature_version")
    if feature_version is not None and feature_version != FEATURE_VERSION:
        raise ValueError(
            f"checkpoint feature version mismatch: expected {FEATURE_VERSION!r}, "
            f"found {feature_version!r}"
        )
    dataset_revision = payload.get("dataset_revision")
    if dataset_revision is not None and dataset_revision != PHRESH_REVISION:
        raise ValueError(
            "checkpoint dataset revision mismatch: "
            f"expected {PHRESH_REVISION!r}, found {dataset_revision!r}"
        )

    keys = set(payload)
    if keys != _CHECKPOINT_KEYS:
        missing = sorted(_CHECKPOINT_KEYS - keys)
        extra = sorted(keys - _CHECKPOINT_KEYS)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected keys {extra}")
        raise ValueError("checkpoint schema mismatch: " + ", ".join(details))

    metadata = _validate_checkpoint_metadata(payload, checkpoint_path.parent)
    if config is not None:
        _validate_checkpoint_matches_config(metadata, config)
    model_path = checkpoint_path.parent / metadata["model_joblib"]
    model_sha256 = sha256_file(model_path)
    if model_sha256 != metadata["model_sha256"]:
        raise ValueError(
            "checkpoint model SHA-256 mismatch: "
            f"expected {metadata['model_sha256']!r}, found {model_sha256!r}"
        )
    model = joblib.load(model_path)
    _validate_model(model, include_dom=metadata["include_dom"])
    return model, metadata


def predict_stream_scores(
    model,
    records,
    *,
    include_dom: bool,
    batch_size: int = 4096,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not isinstance(include_dom, bool):
        raise TypeError("include_dom must be a boolean")
    batch_size_value = _require_positive_integer("batch_size", batch_size)
    limit_value = _validate_limit(limit)
    factory = _records_factory(records)
    y_true: list[int] = []
    scores: list[np.ndarray] = []
    batch: list[dict] = []
    counts = _empty_counts()
    processed = 0

    for record in factory():
        if limit_value is not None and processed >= limit_value:
            break
        processed += 1
        accepted = _accepted_training_record(record)
        if accepted is None:
            counts["reject_counts"][_reject_reason(record)] += 1
            continue
        counts["parse_counts"][_parse_status(accepted)] += 1
        counts["class_counts"][str(accepted["label"])] += 1
        batch.append(accepted)
        y_true.append(int(accepted["label"]))
        if len(batch) == batch_size_value:
            scores.append(predict_scores(model, pd.DataFrame(batch), include_dom, batch_size_value))
            batch = []

    if batch:
        scores.append(predict_scores(model, pd.DataFrame(batch), include_dom, batch_size_value))

    counts["processed_position"] = processed
    return (
        np.asarray(y_true, dtype=np.int8),
        np.concatenate(scores) if scores else np.empty(0, dtype=float),
        _counts_to_json(counts),
    )


def write_json_atomically(payload: Mapping, path) -> Path:
    destination = Path(path)
    text = json.dumps(_json_safe(payload), separators=(",", ":"), sort_keys=True, allow_nan=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return destination


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest(paths: Mapping[str, Path]) -> dict[str, dict[str, object]]:
    return {
        name: {"bytes": Path(path).stat().st_size, "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional cloud extra
        raise RuntimeError("optional dependency 'datasets' is required; install phishme[cloud]") from exc
    return load_dataset(*args, **kwargs)


def _iter_canonical_phresh(stream, limit: int | None) -> Iterator[dict]:
    iterator = iter(stream)
    if limit == 0:
        return
    try:
        first = next(iterator)
    except StopIteration:
        return
    _validate_raw_schema(first)

    for yielded, row in enumerate(chain((first,), iterator)):
        if limit is not None and yielded >= limit:
            break
        canonical = _canonicalize_raw_phresh_row(row)
        yield canonical


def _canonicalize_raw_phresh_row(row: Mapping) -> dict:
    label = _maybe_canonical_label(row.get("label"))
    if label is None:
        return _rejected_raw_record("missing_label")
    if _is_missing_text(row.get("url")):
        return _rejected_raw_record("missing_url")
    if _is_missing_text(row.get("date")):
        return _rejected_raw_record("missing_date")
    url = str(row.get("url")).strip()
    date = _format_utc(_parse_utc_datetime(row.get("date"), "date"))
    title, dom_values, status = _html_features_with_status(url, row.get("html", ""))
    if status == "empty":
        status = "empty_document"
    if status == "ok" and not title:
        status = "empty_title"
    canonical = {
        "url": url,
        "title": title,
        "date": date,
        "label": label,
        "_parse_status": status,
    }
    canonical.update({f"dom_{name}": float(dom_values[name]) for name in NUMERIC_FEATURES})
    return canonical


def _validate_raw_schema(row: Mapping) -> None:
    if not isinstance(row, Mapping):
        raise TypeError("PhreshPhish rows must be mappings")
    columns = set(row)
    missing = sorted(_REQUIRED_RAW_COLUMNS - columns)
    if missing:
        raise ValueError(
            "PhreshPhish schema changed: "
            f"missing required columns {missing}; available columns: {sorted(columns)}"
        )


def _validate_projected_schema(row: Mapping) -> None:
    if not isinstance(row, Mapping):
        raise TypeError("projected PhreshPhish rows must be mappings")
    columns = set(row)
    if columns != _PROJECTED_COLUMNS:
        raise ValueError(
            "projected PhreshPhish schema changed: "
            f"expected columns {CUTOFF_COLUMNS}; available columns: {sorted(columns)}"
        )


def _projected_date(row: Mapping) -> datetime:
    _canonical_label(row.get("label"))
    return _parse_utc_datetime(row.get("date"), "date")


def _canonical_label(value) -> int:
    text = "" if value is None else str(value).strip().lower()
    if text not in _LABELS:
        raise ValueError(f"unknown PhreshPhish label: {value!r}")
    return _LABELS[text]


def _maybe_canonical_label(value) -> int | None:
    if _is_missing_text(value):
        return None
    return _canonical_label(value)


def _rejected_raw_record(reason: str) -> dict:
    return {"_reject_reason": reason, "_parse_status": "skipped"}


def _is_missing_text(value) -> bool:
    return value is None or not str(value).strip()


def _parse_utc_datetime(value, name: str) -> datetime:
    if value is None:
        raise ValueError(f"{name} is required")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} is required")
    try:
        if len(text) == 10 and text[4] == "-" and text[7] == "-":
            parsed = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC)
        else:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be a strict ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a UTC offset")
    return parsed.astimezone(UTC)


def _format_utc(value: datetime) -> str:
    utc = value.astimezone(UTC)
    if utc.microsecond:
        return utc.isoformat().replace("+00:00", "Z")
    return utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_split(split: str) -> str:
    if not isinstance(split, str):
        raise TypeError("split must be a string")
    if split not in PHRESH_SPLITS:
        raise ValueError("split must be one of ['test', 'train']")
    return split


def _validate_revision(revision: str) -> str:
    if revision != PHRESH_REVISION:
        raise ValueError("unsupported PhreshPhish revision")
    return revision


def _validate_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, Integral) or int(limit) < 0:
        raise ValueError("limit must be a non-negative integer")
    return int(limit)


def _records_factory(records) -> Callable[[], Iterator]:
    if callable(records):
        return lambda: iter(records())
    used = False

    def factory() -> Iterator:
        nonlocal used
        if used:
            raise RuntimeError("records must be callable for retry or resume")
        used = True
        return iter(records)

    return factory


def _new_state(config: PhreshTrainConfig) -> _TrainingState:
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(config.alpha),
        random_state=int(config.seed),
    )
    return _TrainingState(
        model=model,
        committed_position=0,
        class_counts=Counter({"0": 0, "1": 0}),
        parse_counts=Counter(),
        reject_counts=Counter(),
        first_fit=True,
    )


def _state_from_checkpoint(model: SGDClassifier, metadata: Mapping) -> _TrainingState:
    return _TrainingState(
        model=model,
        committed_position=int(metadata["processed_position"]),
        class_counts=Counter(metadata["class_counts"]),
        parse_counts=Counter(metadata["parse_counts"]),
        reject_counts=Counter(metadata["reject_counts"]),
        first_fit=False,
    )


def _run_training_attempt(
    factory: Callable[[], Iterator],
    checkpoint_dir: Path,
    config: PhreshTrainConfig,
    state: _TrainingState,
) -> None:
    iterator = _open_stream(factory)
    _skip_committed(iterator, state.committed_position)
    processed_position = state.committed_position
    pending_records: list[dict] = []
    pending_counts = _empty_counts()

    while True:
        try:
            record = _next_stream_record(iterator)
        except StopIteration:
            break

        processed_position += 1
        accepted = _accepted_training_record(record)
        if accepted is None:
            pending_counts["reject_counts"][_reject_reason(record)] += 1
        else:
            pending_records.append(accepted)
            pending_counts["parse_counts"][_parse_status(accepted)] += 1
            pending_counts["class_counts"][str(accepted["label"])] += 1

        if len(pending_records) >= int(config.batch_size):
            _commit_pending_batch(
                pending_records,
                pending_counts,
                processed_position,
                checkpoint_dir,
                config,
                state,
            )
            pending_records = []
            pending_counts = _empty_counts()

    if pending_records or _has_pending_counts(pending_counts):
        if state.first_fit and not pending_records:
            raise ValueError("no accepted PhreshPhish training records")
        _commit_pending_batch(
            pending_records,
            pending_counts,
            processed_position,
            checkpoint_dir,
            config,
            state,
        )


def _skip_committed(iterator: Iterator, position: int) -> None:
    for _ in range(position):
        try:
            _next_stream_record(iterator)
        except StopIteration as exc:
            raise RuntimeError("stream ended before checkpoint position") from exc


def _open_stream(factory: Callable[[], Iterator]) -> Iterator:
    try:
        return factory()
    except Exception as exc:
        if _is_retryable_stream_exception(exc):
            raise _RetryableStreamError(str(exc)) from exc
        raise


def _next_stream_record(iterator: Iterator):
    try:
        return next(iterator)
    except StopIteration:
        raise
    except Exception as exc:
        if _is_retryable_stream_exception(exc):
            raise _RetryableStreamError(str(exc)) from exc
        raise


def _is_retryable_stream_exception(exc: BaseException) -> bool:
    if isinstance(exc, _RETRYABLE_BUILTIN_STREAM_ERRORS):
        return True
    if _is_httpx_transport_error(exc):
        return True
    return type(exc) is RuntimeError and str(exc) == _CLOSED_HF_CLIENT_RUNTIME_ERROR


def _is_httpx_transport_error(exc: BaseException) -> bool:
    try:
        import httpx
    except ImportError:
        return False
    return isinstance(exc, httpx.TransportError)


def _reset_huggingface_client() -> None:
    try:
        from huggingface_hub.utils import close_session
    except ModuleNotFoundError as exc:
        if exc.name == "huggingface_hub":
            return
        raise
    close_session()


def _commit_pending_batch(
    records: list[dict],
    counts: dict[str, Counter[str]],
    processed_position: int,
    checkpoint_dir: Path,
    config: PhreshTrainConfig,
    state: _TrainingState,
) -> None:
    if records:
        frame = pd.DataFrame(records)
        labels = frame["label"].to_numpy(dtype=np.int8)
        kwargs = {"classes": _CLASSES} if state.first_fit else {}
        state.model.partial_fit(vectorize(frame, include_dom=config.include_dom), labels, **kwargs)
        state.first_fit = False

    state.class_counts.update(counts["class_counts"])
    state.parse_counts.update(counts["parse_counts"])
    state.reject_counts.update(counts["reject_counts"])
    state.committed_position = processed_position
    _write_checkpoint(checkpoint_dir, config, state)


def _accepted_training_record(record) -> dict | None:
    if not isinstance(record, Mapping):
        raise TypeError("stream records must be mappings")
    if "_reject_reason" in record:
        return None
    if "label" not in record:
        return None
    if "url" not in record:
        return None
    label = record["label"]
    if isinstance(label, bool) or label not in (0, 1):
        raise ValueError(f"training label must be 0 or 1, found {label!r}")
    url = str(record["url"]).strip()
    if not url:
        return None
    accepted = dict(record)
    accepted["url"] = url
    accepted["label"] = int(label)
    accepted.setdefault("title", "")
    accepted.setdefault("_parse_status", "ok")
    return accepted


def _reject_reason(record) -> str:
    if isinstance(record, Mapping):
        reason = record.get("_reject_reason")
        if reason is not None:
            return _safe_reason(reason)
        if "label" not in record:
            return "missing_label"
        if "url" not in record or not str(record.get("url", "")).strip():
            return "missing_url"
    return "invalid_record"


def _parse_status(record: Mapping) -> str:
    return _safe_reason(record.get("_parse_status", "ok"))


def _safe_reason(value) -> str:
    text = str(value).strip()
    if not text:
        return "unknown"
    if text in _UNSAFE_JSON_KEYS or "\x00" in text:
        raise ValueError("reason contains an unsafe key")
    return text


def _empty_counts() -> dict[str, Counter[str]]:
    return {
        "class_counts": Counter({"0": 0, "1": 0}),
        "parse_counts": Counter(),
        "reject_counts": Counter(),
    }


def _has_pending_counts(counts: Mapping[str, Counter[str]]) -> bool:
    return any(sum(counter.values()) for counter in counts.values())


def _write_checkpoint(
    checkpoint_dir: Path,
    config: PhreshTrainConfig,
    state: _TrainingState,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / CHECKPOINT_FILENAME
    model_path, model_sha256 = _write_model_joblib_atomically(
        state.model,
        checkpoint_dir,
        state.committed_position,
    )
    payload = _checkpoint_payload(
        state,
        config,
        model_path.name,
        model_sha256,
    )
    _write_checkpoint_metadata_atomically(payload, checkpoint_path)


def _write_model_joblib_atomically(
    model: SGDClassifier,
    checkpoint_dir: Path,
    processed_position: int,
) -> tuple[Path, str]:
    temporary_path = checkpoint_dir / f".model-{processed_position:012d}.joblib.tmp"
    try:
        joblib.dump(model, temporary_path)
        _fsync_file(temporary_path)
        digest = sha256_file(temporary_path)
        model_path = checkpoint_dir / f"model-{processed_position:012d}-{digest[:16]}.joblib"
        os.replace(temporary_path, model_path)
        _fsync_directory(checkpoint_dir)
        return model_path, digest
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def _write_checkpoint_metadata_atomically(payload: Mapping, path: Path) -> None:
    write_json_atomically(payload, path)


def _checkpoint_payload(
    state: _TrainingState,
    config: PhreshTrainConfig,
    model_joblib: str,
    model_sha256: str,
) -> dict:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "processed_position": int(state.committed_position),
        "model_joblib": model_joblib,
        "model_sha256": model_sha256,
        "feature_version": FEATURE_VERSION,
        "dataset_revision": PHRESH_REVISION,
        "split": config.split,
        "cutoff": config.cutoff,
        "alpha": float(config.alpha),
        "batch_size": int(config.batch_size),
        "class_counts": _counter_to_json(state.class_counts, required=("0", "1")),
        "parse_counts": _counter_to_json(state.parse_counts),
        "reject_counts": _counter_to_json(state.reject_counts),
        "seed": int(config.seed),
        "include_dom": bool(config.include_dom),
        "retry_policy": config.retry_policy(),
        "write_time_utc": _format_utc(datetime.now(UTC)),
    }


def _read_strict_json(path: Path):
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"strict checkpoint JSON is invalid: {exc}") from exc
    except ValueError as exc:
        if "duplicate" in str(exc) or "JSON constant" in str(exc):
            raise
        raise ValueError(f"strict checkpoint JSON is invalid: {exc}") from exc


def _object_without_duplicate_keys(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate key {key!r}")
        output[key] = value
    return output


def _reject_json_constant(value: str):
    raise ValueError(f"invalid JSON constant {value}")


def _validate_checkpoint_metadata(payload: Mapping, directory: Path) -> dict:
    if payload["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema mismatch")
    processed_position = _require_non_negative_integer(
        "processed_position",
        payload["processed_position"],
    )
    model_joblib = _validate_model_filename(payload["model_joblib"])
    model_path = directory / model_joblib
    _ensure_checkpoint_child(directory, model_path)
    if not model_path.is_file():
        raise ValueError(f"checkpoint model is missing: {model_joblib}")
    model_sha256 = _validate_sha256(payload["model_sha256"], "model_sha256")
    if payload["feature_version"] != FEATURE_VERSION:
        raise ValueError("checkpoint feature version mismatch")
    if payload["dataset_revision"] != PHRESH_REVISION:
        raise ValueError("checkpoint dataset revision mismatch")
    split = _validate_split(payload["split"])
    cutoff = payload["cutoff"]
    if cutoff is not None:
        cutoff = _format_utc(_parse_utc_datetime(cutoff, "cutoff"))
    alpha = _require_positive_float("alpha", payload["alpha"])
    batch_size = _require_positive_integer("batch_size", payload["batch_size"])
    class_counts = _validate_counts(
        payload["class_counts"],
        "class_counts",
        required_keys=frozenset(("0", "1")),
    )
    parse_counts = _validate_counts(payload["parse_counts"], "parse_counts")
    reject_counts = _validate_counts(payload["reject_counts"], "reject_counts")
    _validate_checkpoint_count_consistency(
        processed_position,
        class_counts,
        parse_counts,
        reject_counts,
    )
    seed = _require_non_negative_integer("seed", payload["seed"])
    include_dom = payload["include_dom"]
    if not isinstance(include_dom, bool):
        raise TypeError("checkpoint include_dom must be a boolean")
    retry_policy = _validate_retry_policy(payload["retry_policy"])
    write_time_utc = _format_utc(_parse_utc_datetime(payload["write_time_utc"], "write_time_utc"))

    return {
        "schema": CHECKPOINT_SCHEMA,
        "processed_position": processed_position,
        "model_joblib": model_joblib,
        "model_sha256": model_sha256,
        "feature_version": FEATURE_VERSION,
        "dataset_revision": PHRESH_REVISION,
        "split": split,
        "cutoff": cutoff,
        "alpha": alpha,
        "batch_size": batch_size,
        "class_counts": class_counts,
        "parse_counts": parse_counts,
        "reject_counts": reject_counts,
        "seed": seed,
        "include_dom": include_dom,
        "retry_policy": retry_policy,
        "write_time_utc": write_time_utc,
    }


def _validate_checkpoint_matches_config(metadata: Mapping, config: PhreshTrainConfig) -> None:
    expected = {
        "dataset_revision": PHRESH_REVISION,
        "split": config.split,
        "cutoff": config.cutoff,
        "alpha": float(config.alpha),
        "batch_size": int(config.batch_size),
        "seed": int(config.seed),
        "include_dom": bool(config.include_dom),
        "retry_policy": config.retry_policy(),
    }
    for key, expected_value in expected.items():
        if metadata[key] != expected_value:
            raise ValueError(
                f"checkpoint {key.replace('_', ' ')} mismatch: "
                f"expected {expected_value!r}, found {metadata[key]!r}"
            )


def _validate_model(model, *, include_dom: bool) -> None:
    if not isinstance(model, SGDClassifier):
        raise TypeError("checkpoint model must be an SGDClassifier")
    if not hasattr(model, "classes_") or not np.array_equal(model.classes_, _CLASSES):
        raise ValueError("checkpoint model class order must be exactly [0, 1]")
    expected_width = HASH_DIM + (len(NUMERIC_FEATURES) if include_dom else 0)
    coefficients = np.asarray(getattr(model, "coef_", None), dtype=np.float64)
    if coefficients.shape != (1, expected_width):
        raise ValueError(
            "checkpoint model dimension mismatch: "
            f"expected coefficient shape (1, {expected_width}), found {coefficients.shape}"
        )
    intercept = np.asarray(getattr(model, "intercept_", None), dtype=np.float64)
    if intercept.shape != (1,):
        raise ValueError(f"checkpoint model intercept dimension mismatch: found {intercept.shape}")
    if not np.isfinite(coefficients).all() or not np.isfinite(intercept).all():
        raise ValueError("checkpoint model parameters must be finite")


def _validate_model_filename(value) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("checkpoint model_joblib must be a filename")
    path = Path(value)
    if path.name != value or path.is_absolute() or ".." in path.parts:
        raise ValueError("checkpoint model_joblib path escapes checkpoint directory")
    if not value.endswith(".joblib"):
        raise ValueError("checkpoint model_joblib must reference a .joblib file")
    return value


def _ensure_checkpoint_child(directory: Path, child: Path) -> None:
    parent = directory.resolve(strict=False)
    resolved = child.resolve(strict=False)
    if resolved.parent != parent:
        raise ValueError(f"checkpoint path escapes directory: {child}")


def _validate_sha256(value, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from exc
    return value.lower()


def _validate_counts(value, name: str, *, required_keys: frozenset[str] | None = None) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"checkpoint {name} must be a mapping")
    keys = set(value)
    if required_keys is not None and keys != required_keys:
        raise ValueError(f"checkpoint {name} keys mismatch")
    output = {}
    for key, item in value.items():
        if not isinstance(key, str) or key in _UNSAFE_JSON_KEYS or "\x00" in key:
            raise ValueError(f"checkpoint {name} contains unsafe key")
        output[key] = _require_non_negative_integer(f"{name}[{key!r}]", item)
    return dict(sorted(output.items()))


def _validate_checkpoint_count_consistency(
    processed_position: int,
    class_counts: Mapping[str, int],
    parse_counts: Mapping[str, int],
    reject_counts: Mapping[str, int],
) -> None:
    accepted = sum(class_counts.values())
    parsed = sum(parse_counts.values())
    rejected = sum(reject_counts.values())
    if accepted != parsed:
        raise ValueError(
            "checkpoint metadata inconsistent: "
            "sum(class_counts) must equal sum(parse_counts); "
            f"found {accepted} and {parsed}"
        )
    if accepted + rejected != processed_position:
        raise ValueError(
            "checkpoint metadata inconsistent: "
            "accepted examples plus reject_counts must equal processed_position; "
            f"found {accepted} + {rejected} != {processed_position}"
        )


def _validate_retry_policy(value) -> dict[str, int | float]:
    if not isinstance(value, Mapping) or set(value) != _RETRY_POLICY_KEYS:
        raise ValueError("checkpoint retry policy mismatch")
    attempts = _require_positive_integer("retry_policy.max_attempts", value["max_attempts"])
    if attempts > 3:
        raise ValueError("checkpoint retry policy max_attempts must be at most 3")
    delay = _require_non_negative_float(
        "retry_policy.initial_delay_seconds",
        value["initial_delay_seconds"],
    )
    multiplier = _require_positive_float(
        "retry_policy.backoff_multiplier",
        value["backoff_multiplier"],
    )
    if multiplier < 1.0:
        raise ValueError("checkpoint retry policy backoff_multiplier must be at least 1")
    return {
        "max_attempts": attempts,
        "initial_delay_seconds": delay,
        "backoff_multiplier": multiplier,
    }


def _counter_to_json(counter: Counter[str], *, required: Sequence[str] = ()) -> dict[str, int]:
    output = {str(key): int(value) for key, value in counter.items() if int(value) != 0}
    for key in required:
        output.setdefault(str(key), 0)
    return dict(sorted(output.items()))


def _counts_to_json(counts: Mapping[str, object]) -> dict:
    return {
        "processed_position": int(counts["processed_position"]),
        "class_counts": _counter_to_json(counts["class_counts"], required=("0", "1")),
        "parse_counts": _counter_to_json(counts["parse_counts"]),
        "reject_counts": _counter_to_json(counts["reject_counts"]),
    }


def _json_safe(value, *, depth: int = 0):
    if depth > 12:
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
            if key in _UNSAFE_JSON_KEYS or "\x00" in key:
                raise ValueError("JSON object contains unsafe key")
            output[key] = _json_safe(item, depth=depth + 1)
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item, depth=depth + 1) for item in value]
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sleep_before_retry(config: PhreshTrainConfig, failures_since_progress: int) -> None:
    _sleep_before_retry_values(
        float(config.retry_initial_delay_seconds),
        float(config.retry_backoff_multiplier),
        failures_since_progress,
    )


def _sleep_before_retry_values(
    initial_delay_seconds: float,
    backoff_multiplier: float,
    failures_since_progress: int,
) -> None:
    delay = float(initial_delay_seconds)
    if delay <= 0:
        return
    multiplier = float(backoff_multiplier)
    time.sleep(delay * (multiplier ** max(0, failures_since_progress - 1)))


def _require_positive_integer(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _require_non_negative_integer(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return integer


def _require_positive_float(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _require_non_negative_float(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be non-negative")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number
