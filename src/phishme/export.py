from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .features import (
    DOM_NUMERIC_FEATURES,
    FEATURE_VERSION,
    HASH_DIM,
    NGRAM_RANGE,
    NUMERIC_FEATURES,
    URL_NUMERIC_FEATURES,
    _as_text,
    _transform,
    fnv1a_32,
    iter_ngrams,
    url_numeric_features,
)

SCHEMA = "phishme-model-v1"
HASH_NAME = "fnv1a-32"
_TOP_LEVEL_KEYS = {
    "schema",
    "feature_version",
    "hash",
    "include_dom",
    "numeric_features",
    "weights",
    "intercept",
    "threshold",
    "metadata",
}
_HASH_KEYS = {"name", "dimension", "ngram_min", "ngram_max"}
_UNSAFE_METADATA_KEYS = {"__proto__", "constructor", "prototype"}
_MAX_METADATA_DEPTH = 12


def export_model(
    model,
    threshold,
    include_dom,
    metadata,
    path,
    *,
    hash_dim: int = HASH_DIM,
) -> Path:
    """Export a fitted binary SGD model as a compact browser-compatible artifact."""

    if not isinstance(include_dom, bool):
        raise TypeError("include_dom must be a boolean")
    dimension = _require_positive_integer("hash dimension", hash_dim)
    _validate_class_order(model)

    numeric_features = list(NUMERIC_FEATURES) if include_dom else []
    expected_width = dimension + len(numeric_features)
    coefficients = _model_coefficients(model, expected_width)
    intercept = _model_intercept(model)
    threshold_value = _finite_float("threshold", threshold)
    if not 0.0 <= threshold_value <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    payload = {
        "schema": SCHEMA,
        "feature_version": FEATURE_VERSION,
        "hash": {
            "name": HASH_NAME,
            "dimension": dimension,
            "ngram_min": NGRAM_RANGE[0],
            "ngram_max": NGRAM_RANGE[1],
        },
        "include_dom": include_dom,
        "numeric_features": numeric_features,
        "weights": [float(value) for value in coefficients],
        "intercept": float(intercept),
        "threshold": threshold_value,
        "metadata": _safe_metadata(metadata),
    }
    validated = _validate_payload(payload)
    destination = Path(path)
    _write_json_atomically(validated, destination)
    return destination


def load_model(path) -> dict:
    """Load and validate a strict JSON phishMe artifact."""

    artifact_path = Path(path)
    try:
        payload = json.loads(
            artifact_path.read_text(),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"strict JSON artifact is invalid: {exc}") from exc
    except ValueError as exc:
        if "duplicate" in str(exc) or "strict JSON" in str(exc):
            raise
        raise ValueError(f"strict JSON artifact is invalid: {exc}") from exc
    return _validate_payload(payload)


def score_record(artifact: Mapping, record: Mapping) -> float:
    """Return the Python reference phishing probability for one record."""

    model = _validate_payload(artifact)
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")

    dimension = model["hash"]["dimension"]
    weights = model["weights"]
    seen = _hashed_text_indices(model, record)

    logit = model["intercept"]
    for index in seen:
        logit += weights[index]

    if model["include_dom"]:
        url_values = url_numeric_features(_as_text(record.get("url", "")))
        dom = record.get("dom", {})
        if not isinstance(dom, Mapping):
            dom = {}
        for offset, name in enumerate(model["numeric_features"]):
            if name in URL_NUMERIC_FEATURES:
                raw_value = url_values[name]
            elif name in DOM_NUMERIC_FEATURES:
                raw_value = dom.get(name, 0.0)
            else:
                raw_value = 0.0
            logit += weights[dimension + offset] * _transform(name, raw_value)

    return _sigmoid(logit)


def _validate_payload(payload) -> dict:
    if not isinstance(payload, Mapping):
        raise TypeError("artifact must be a mapping")
    keys = set(payload)
    if keys != _TOP_LEVEL_KEYS:
        missing = sorted(_TOP_LEVEL_KEYS - keys)
        extra = sorted(keys - _TOP_LEVEL_KEYS)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected top-level keys {extra}")
        raise ValueError("artifact top-level schema mismatch: " + ", ".join(details))
    if payload["schema"] != SCHEMA:
        raise ValueError("artifact schema mismatch")
    if payload["feature_version"] != FEATURE_VERSION:
        raise ValueError("artifact feature version mismatch")
    if not isinstance(payload["include_dom"], bool):
        raise TypeError("artifact include_dom must be a boolean")

    hash_config = _validate_hash(payload["hash"])
    numeric_features = _validate_numeric_features(payload["include_dom"], payload["numeric_features"])
    weights = _finite_float_list("weights", payload["weights"])
    expected_width = hash_config["dimension"] + len(numeric_features)
    if len(weights) != expected_width:
        raise ValueError(
            f"artifact dimension mismatch: expected {expected_width} weights, found {len(weights)}"
        )

    return {
        "schema": SCHEMA,
        "feature_version": FEATURE_VERSION,
        "hash": hash_config,
        "include_dom": payload["include_dom"],
        "numeric_features": numeric_features,
        "weights": weights,
        "intercept": _finite_float("intercept", payload["intercept"]),
        "threshold": _validate_threshold(payload["threshold"]),
        "metadata": _safe_metadata(payload["metadata"]),
    }


def _validate_hash(value) -> dict:
    if not isinstance(value, Mapping):
        raise TypeError("artifact hash must be a mapping")
    if set(value) != _HASH_KEYS:
        raise ValueError("artifact hash configuration mismatch")
    if value["name"] != HASH_NAME:
        raise ValueError("artifact hash name mismatch")
    dimension = _require_positive_integer("hash dimension", value["dimension"])
    if value["ngram_min"] != NGRAM_RANGE[0] or value["ngram_max"] != NGRAM_RANGE[1]:
        raise ValueError("artifact hash ngram range mismatch")
    return {
        "name": HASH_NAME,
        "dimension": dimension,
        "ngram_min": NGRAM_RANGE[0],
        "ngram_max": NGRAM_RANGE[1],
    }


def _validate_numeric_features(include_dom: bool, value) -> list[str]:
    if not isinstance(value, list):
        raise TypeError("artifact numeric_features must be a list")
    expected = list(NUMERIC_FEATURES) if include_dom else []
    if value != expected:
        raise ValueError("artifact numeric_features order mismatch")
    return list(value)


def _model_coefficients(model, expected_width: int) -> np.ndarray:
    if not hasattr(model, "coef_"):
        raise ValueError("model must have coefficients")
    coefficients = np.asarray(model.coef_, dtype=np.float64)
    if coefficients.shape != (1, expected_width):
        raise ValueError(
            f"model dimension mismatch: expected coefficient shape (1, {expected_width}), "
            f"found {coefficients.shape}"
        )
    if not np.isfinite(coefficients).all():
        raise ValueError("model coefficients must be finite")
    coefficients32 = coefficients.astype(np.float32, copy=False).reshape(-1)
    if not np.isfinite(coefficients32).all():
        raise ValueError("model coefficients must be finite after float32 conversion")
    return coefficients32


def _model_intercept(model) -> np.float32:
    if not hasattr(model, "intercept_"):
        raise ValueError("model must have an intercept")
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    if intercept.shape != (1,):
        raise ValueError(f"model intercept dimension mismatch: found {intercept.shape}")
    if not np.isfinite(intercept).all():
        raise ValueError("model intercept must be finite")
    intercept32 = intercept.astype(np.float32, copy=False)[0]
    if not np.isfinite(intercept32):
        raise ValueError("model intercept must be finite after float32 conversion")
    return intercept32


def _validate_class_order(model) -> None:
    if not hasattr(model, "classes_"):
        raise ValueError("model must have class order")
    classes = np.asarray(model.classes_)
    if classes.shape != (2,) or not np.array_equal(classes, np.array([0, 1])):
        raise ValueError("model class order must be exactly [0, 1]")


def _finite_float_list(name: str, value) -> list[float]:
    if not isinstance(value, list):
        raise TypeError(f"artifact {name} must be a list")
    return [_finite_float(f"{name}[{index}]", item) for index, item in enumerate(value)]


def _finite_float(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_threshold(value) -> float:
    threshold = _finite_float("threshold", value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return threshold


def _require_positive_integer(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _safe_metadata(value, *, depth: int = 0):
    if depth > _MAX_METADATA_DEPTH:
        raise ValueError("metadata is too deeply nested")
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")

    safe = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        if key in _UNSAFE_METADATA_KEYS:
            raise ValueError("metadata contains unsafe key")
        if "\x00" in key:
            raise ValueError("metadata contains unsafe key")
        safe[key] = _safe_metadata_value(item, depth=depth + 1)
    return safe


def _safe_metadata_value(value, *, depth: int):
    if depth > _MAX_METADATA_DEPTH:
        raise ValueError("metadata is too deeply nested")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("metadata numbers must be finite")
        return number
    if isinstance(value, Mapping):
        return _safe_metadata(value, depth=depth)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe_metadata_value(item, depth=depth + 1) for item in value]
    raise TypeError("metadata values must be JSON-compatible")


def _hashed_text_indices(model: Mapping, record: Mapping) -> set[int]:
    dimension = model["hash"]["dimension"]
    seen: set[int] = set()
    for namespace, raw in (("url", record.get("url", "")), ("title", record.get("title", ""))):
        for token in iter_ngrams(namespace, _as_text(raw)):
            seen.add(fnv1a_32(token) % dimension)
    return seen


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1 + exp)


def _write_json_atomically(payload: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False)
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


def _object_without_duplicate_keys(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_json_constant(value: str):
    raise ValueError(f"strict JSON artifact rejects {value}")
