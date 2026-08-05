import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import SGDClassifier

from phishme.export import export_model, load_model, score_record
from phishme.features import (
    NUMERIC_FEATURES,
    _transform,
    fnv1a_32,
    iter_ngrams,
    url_numeric_features,
)


def _model(weights, intercept=0.0, classes=(0, 1)) -> SGDClassifier:
    model = SGDClassifier(loss="log_loss")
    model.classes_ = np.array(classes)
    model.coef_ = np.array([weights], dtype=np.float64)
    model.intercept_ = np.array([intercept], dtype=np.float64)
    return model


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1 + exp)


def _valid_payload(hash_dim=2, include_dom=False):
    numeric_features = list(NUMERIC_FEATURES) if include_dom else []
    return {
        "schema": "phishme-model-v1",
        "feature_version": "phishme-features-v1",
        "hash": {"name": "fnv1a-32", "dimension": hash_dim, "ngram_min": 3, "ngram_max": 5},
        "include_dom": include_dom,
        "numeric_features": numeric_features,
        "weights": [0.25] * (hash_dim + len(numeric_features)),
        "intercept": 0.125,
        "threshold": 0.6,
        "metadata": {"dataset_revision": "test"},
    }


def _write_payload(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return path


def _assert_js_probability_parity(tmp_path: Path, artifact: dict, record: dict) -> None:
    python_probability = score_record(artifact, record)
    record_path = _write_payload(tmp_path / "record.json", record)
    score_path = tmp_path / "python-score.txt"
    score_path.write_text(f"{python_probability:.17g}")

    result = subprocess.run(
        ["node", "web/parity.test.mjs", str(tmp_path / "model.json"), str(record_path), str(score_path)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_export_is_compact_text_only_and_float32(tmp_path: Path):
    model = _model([0.1, -0.2])

    path = export_model(
        model,
        0.6,
        False,
        {"dataset_revision": "test"},
        tmp_path / "model.json",
        hash_dim=2,
    )
    text = path.read_text()
    payload = load_model(path)

    assert path == tmp_path / "model.json"
    assert ": " not in text
    assert ", " not in text
    assert payload["schema"] == "phishme-model-v1"
    assert payload["feature_version"] == "phishme-features-v1"
    assert payload["threshold"] == 0.6
    assert payload["hash"] == {"name": "fnv1a-32", "dimension": 2, "ngram_min": 3, "ngram_max": 5}
    assert payload["include_dom"] is False
    assert payload["numeric_features"] == []
    assert payload["weights"] == [float(np.float32(0.1)), float(np.float32(-0.2))]
    assert json.loads(text)["feature_version"]
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_include_dom_export_preserves_numeric_order_and_validates_model_shape(tmp_path: Path):
    weights = np.arange(4 + len(NUMERIC_FEATURES), dtype=np.float64) / 100
    path = export_model(
        _model(weights, intercept=0.25),
        0.5,
        True,
        {"dataset_revision": "dom"},
        tmp_path / "dom.json",
        hash_dim=4,
    )

    payload = load_model(path)

    assert payload["include_dom"] is True
    assert payload["numeric_features"] == list(NUMERIC_FEATURES)
    assert len(payload["weights"]) == 4 + len(NUMERIC_FEATURES)
    assert payload["intercept"] == 0.25

    with pytest.raises(ValueError, match="class order"):
        export_model(_model([0.0, 0.0], classes=(1, 0)), 0.5, False, {}, tmp_path / "bad.json", hash_dim=2)
    with pytest.raises(ValueError, match="dimension"):
        export_model(_model([0.0]), 0.5, False, {}, tmp_path / "bad.json", hash_dim=2)
    with pytest.raises(ValueError, match="finite"):
        export_model(_model([0.0, math.inf]), 0.5, False, {}, tmp_path / "bad.json", hash_dim=2)
    with pytest.raises(ValueError, match="finite"):
        export_model(_model([0.0, 0.0], intercept=math.nan), 0.5, False, {}, tmp_path / "bad.json", hash_dim=2)
    with pytest.raises(ValueError, match="threshold"):
        export_model(_model([0.0, 0.0]), math.nan, False, {}, tmp_path / "bad.json", hash_dim=2)


def test_load_model_rejects_malformed_strict_json_and_artifact_mismatches(tmp_path: Path):
    cases = []

    extra = _valid_payload()
    extra["unexpected"] = True
    cases.append(("top-level", extra))

    wrong_schema = _valid_payload()
    wrong_schema["schema"] = "wrong"
    cases.append(("schema", wrong_schema))

    wrong_feature = _valid_payload()
    wrong_feature["feature_version"] = "wrong"
    cases.append(("feature version", wrong_feature))

    wrong_hash = _valid_payload()
    wrong_hash["hash"]["name"] = "murmur"
    cases.append(("hash", wrong_hash))

    wrong_dimension = _valid_payload()
    wrong_dimension["weights"].append(0.0)
    cases.append(("dimension", wrong_dimension))

    wrong_order = _valid_payload(include_dom=True)
    wrong_order["numeric_features"] = list(reversed(wrong_order["numeric_features"]))
    cases.append(("numeric_features", wrong_order))

    unsafe_metadata = _valid_payload()
    unsafe_metadata["metadata"] = {"__proto__": {"polluted": True}}
    cases.append(("metadata", unsafe_metadata))

    for name, payload in cases:
        with pytest.raises((TypeError, ValueError), match=name):
            load_model(_write_payload(tmp_path / f"{name}.json", payload))

    nan_path = tmp_path / "nan.json"
    nan_path.write_text('{"schema":"phishme-model-v1","weights":[NaN]}')
    with pytest.raises(ValueError, match="strict JSON"):
        load_model(nan_path)

    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text('{"schema":"phishme-model-v1","schema":"phishme-model-v1"}')
    with pytest.raises(ValueError, match="duplicate"):
        load_model(duplicate_path)


def test_score_record_recomputes_url_features_and_uses_only_allowlisted_dom(tmp_path: Path):
    hash_dim = 8
    weights = np.zeros(hash_dim + len(NUMERIC_FEATURES), dtype=np.float64)
    weights[hash_dim + NUMERIC_FEATURES.index("is_https")] = 0.75
    weights[hash_dim + NUMERIC_FEATURES.index("url_length")] = 0.125
    weights[hash_dim + NUMERIC_FEATURES.index("image_count")] = -0.5
    artifact = load_model(
        export_model(
            _model(weights, intercept=-0.2),
            0.5,
            True,
            {"dataset_revision": "dom"},
            tmp_path / "dom.json",
            hash_dim=hash_dim,
        )
    )
    record = {
        "url": "https://example.com/login",
        "title": "",
        "dom": {
            "url_length": 1_000_000,
            "is_https": 0,
            "image_count": 3,
            "not_allowlisted": 1_000_000,
        },
    }

    probability = score_record(artifact, record)

    url_values = url_numeric_features(record["url"])
    expected_logit = (
        artifact["intercept"]
        + artifact["weights"][hash_dim + NUMERIC_FEATURES.index("is_https")]
        * _transform("is_https", url_values["is_https"])
        + artifact["weights"][hash_dim + NUMERIC_FEATURES.index("url_length")]
        * _transform("url_length", url_values["url_length"])
        + artifact["weights"][hash_dim + NUMERIC_FEATURES.index("image_count")]
        * _transform("image_count", 3)
    )
    assert probability == pytest.approx(_sigmoid(expected_logit), abs=1e-12)


def test_score_record_uses_binary_hash_buckets_for_collisions(tmp_path: Path):
    artifact = load_model(
        export_model(
            _model([0.75]),
            0.5,
            False,
            {"dataset_revision": "collision"},
            tmp_path / "collision.json",
            hash_dim=1,
        )
    )

    probability = score_record(artifact, {"url": "https://aaaa.example/aaaa", "title": "aaaa"})

    assert probability == pytest.approx(_sigmoid(0.75), abs=1e-12)


def test_score_record_matches_python_vector_definition_for_astral_text(tmp_path: Path):
    hash_dim = 64
    weights = np.zeros(hash_dim, dtype=np.float64)
    record = {"url": "https://example.com/𠮷abc", "title": "Login 𠮷abc"}
    seen = {fnv1a_32(token) % hash_dim for token in iter_ngrams("url", record["url"])}
    seen.update(fnv1a_32(token) % hash_dim for token in iter_ngrams("title", record["title"]))
    for index in seen:
        weights[index] = (index + 1) / 1000
    artifact = load_model(
        export_model(
            _model(weights, intercept=-0.125),
            0.5,
            False,
            {"dataset_revision": "astral"},
            tmp_path / "astral.json",
            hash_dim=hash_dim,
        )
    )

    expected = _sigmoid(artifact["intercept"] + sum(artifact["weights"][index] for index in seen))

    assert score_record(artifact, record) == pytest.approx(expected, abs=1e-12)


def test_js_parity_matches_python_score_with_astral_text_and_dom(tmp_path: Path):
    hash_dim = 128
    weights = np.zeros(hash_dim + len(NUMERIC_FEATURES), dtype=np.float64)
    record = {
        "url": "https://example.com/𠮷abc?x=1&y=%2F",
        "title": "Secure 𠮷abc portal",
        "dom": {
            "image_count": 4,
            "has_password_field": 1,
            "external_ref_count": 2,
            "url_length": 999999,
        },
    }
    seen = {fnv1a_32(token) % hash_dim for token in iter_ngrams("url", record["url"])}
    seen.update(fnv1a_32(token) % hash_dim for token in iter_ngrams("title", record["title"]))
    for index in seen:
        weights[index] = ((index % 7) - 3) / 100
    for name, value in {
        "is_https": 0.4,
        "url_length": 0.025,
        "image_count": -0.1,
        "has_password_field": 0.35,
        "external_ref_count": -0.05,
    }.items():
        weights[hash_dim + NUMERIC_FEATURES.index(name)] = value
    artifact = load_model(
        export_model(
            _model(weights, intercept=0.03125),
            0.42,
            True,
            {"dataset_revision": "parity"},
            tmp_path / "model.json",
            hash_dim=hash_dim,
        )
    )
    _assert_js_probability_parity(tmp_path, artifact, record)


def test_js_parity_matches_python_score_with_frozen_suffix_and_unicode_numbers(tmp_path: Path):
    hash_dim = 8
    weights = np.zeros(hash_dim + len(NUMERIC_FEATURES), dtype=np.float64)
    record = {
        "url": "https://login.shop.example.co.nz/pay/𠮷é?code=①²&step=௧",
        "title": "",
        "dom": {
            "image_count": 2,
            "url_length": 999999,
            "digit_count": 999999,
        },
    }
    for name, value in {
        "tld_length": 0.071,
        "subdomain_count": -0.113,
        "letter_count": 0.017,
        "digit_count": -0.029,
        "special_count": 0.043,
        "image_count": 0.059,
    }.items():
        weights[hash_dim + NUMERIC_FEATURES.index(name)] = value
    artifact = load_model(
        export_model(
            _model(weights, intercept=-0.1875),
            0.5,
            True,
            {"dataset_revision": "unicode-suffix-parity"},
            tmp_path / "model.json",
            hash_dim=hash_dim,
        )
    )
    _assert_js_probability_parity(tmp_path, artifact, record)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:abc/login",
        "http://exa\uff0fmple.com/login",
    ],
)
def test_js_parity_matches_python_score_with_frozen_authority_parser(
    tmp_path: Path, url: str
):
    hash_dim = 8
    weights = np.zeros(hash_dim + len(NUMERIC_FEATURES), dtype=np.float64)
    for name, value in {
        "url_length": 0.013,
        "domain_length": -0.071,
        "tld_length": 0.109,
        "subdomain_count": -0.127,
        "is_domain_ip": 0.173,
        "is_https": -0.191,
    }.items():
        weights[hash_dim + NUMERIC_FEATURES.index(name)] = value
    artifact = load_model(
        export_model(
            _model(weights, intercept=0.0625),
            0.5,
            True,
            {"dataset_revision": "authority-parser-parity"},
            tmp_path / "model.json",
            hash_dim=hash_dim,
        )
    )
    record = {
        "url": url,
        "title": "",
        "dom": {
            "url_length": 999999,
            "domain_length": 999999,
            "tld_length": 999999,
            "subdomain_count": 999999,
            "is_domain_ip": 1,
            "is_https": 1,
        },
    }

    _assert_js_probability_parity(tmp_path, artifact, record)


@pytest.mark.parametrize(
    ("url", "expected_is_ip"),
    [
        ("http://192.0.2.1/login", 1.0),
        ("http://256.0.2.1/login", 0.0),
        ("http://192.0.02.1/login", 0.0),
        ("http://[2001:db8::1]/login", 1.0),
        ("http://[::ffff:192.0.2.1]/login", 1.0),
        ("http://[2001:db8:::1]/login", 0.0),
    ],
)
def test_js_parity_matches_python_score_with_frozen_ip_classifier(
    tmp_path: Path, url: str, expected_is_ip: float
):
    hash_dim = 8
    weights = np.zeros(hash_dim + len(NUMERIC_FEATURES), dtype=np.float64)
    weights[hash_dim + NUMERIC_FEATURES.index("is_domain_ip")] = 0.625
    artifact = load_model(
        export_model(
            _model(weights, intercept=-0.25),
            0.5,
            True,
            {"dataset_revision": "ip-parser-parity"},
            tmp_path / "model.json",
            hash_dim=hash_dim,
        )
    )
    record = {"url": url, "title": "", "dom": {"is_domain_ip": 1 - expected_is_ip}}
    expected_logit = artifact["intercept"] + 0.625 * expected_is_ip

    assert score_record(artifact, record) == pytest.approx(_sigmoid(expected_logit), abs=1e-12)
    _assert_js_probability_parity(tmp_path, artifact, record)
