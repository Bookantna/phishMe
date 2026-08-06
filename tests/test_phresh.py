from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import pytest

from phishme.phresh import (
    PHRESH_REVISION,
    PhreshTrainConfig,
    derive_temporal_cutoff,
    filter_by_cutoff,
    iter_phresh,
    train_stream,
    validate_checkpoint,
)
from phishme.train import predict_scores


class FakeStream:
    def __init__(self, rows):
        self.rows = list(rows)
        self.selected_columns = None

    def select_columns(self, columns):
        self.selected_columns = list(columns)
        return FakeStream([{column: row[column] for column in columns if column in row} for row in self.rows])

    def __iter__(self):
        return iter(self.rows)


class _ScriptedProjectedStream:
    def __init__(self, rows, *, exception=None, fail_after: int | None = None, scan_log=None):
        self.rows = list(rows)
        self.exception = exception
        self.fail_after = fail_after
        self.scan_log = scan_log

    def __iter__(self):
        for index, row in enumerate(self.rows):
            if self.scan_log is not None:
                self.scan_log.append((index, tuple(row)))
            yield row
            if self.exception is not None and index == self.fail_after:
                raise self.exception
        if self.exception is not None and self.fail_after is None:
            raise self.exception


class _ScriptedCutoffStream:
    def __init__(self, rows, *, exception=None, fail_after: int | None = None, scan_log=None):
        self.rows = list(rows)
        self.exception = exception
        self.fail_after = fail_after
        self.scan_log = scan_log
        self.selected_columns = None

    def select_columns(self, columns):
        self.selected_columns = list(columns)
        projected_rows = [{column: row[column] for column in columns if column in row} for row in self.rows]
        return _ScriptedProjectedStream(
            projected_rows,
            exception=self.exception,
            fail_after=self.fail_after,
            scan_log=self.scan_log,
        )

    def __iter__(self):
        raise AssertionError("cutoff derivation must iterate only projected date/label columns")


def _raw_row(index: int, label: str = "benign", html: str | None = None) -> dict:
    phish = str(label).lower() == "phish"
    return {
        "url": f"http{'s' if not phish else ''}://site{index}.example/login",
        "html": html if html is not None else f"<html><title>row {index}</title></html>",
        "date": f"2024-01-{index + 1:02d}T00:00:00Z",
        "label": label,
    }


def _canonical_record(index: int, label: int, *, status: str = "ok") -> dict:
    title = "verify password" if label else "welcome"
    return {
        "url": f"http{'s' if not label else ''}://site{index}.example/login",
        "title": title,
        "date": f"2024-01-{index + 1:02d}T00:00:00Z",
        "label": label,
        "_parse_status": status,
    }


def _checkpoint_path(directory: Path) -> Path:
    return directory / "checkpoint.json"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_checkpoint_metadata(path: Path, metadata: dict) -> None:
    path.write_text(
        json.dumps(metadata, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


def _install_fake_httpx(monkeypatch):
    fake_httpx = types.ModuleType("httpx")

    class FakeTransportError(Exception):
        pass

    fake_httpx.TransportError = FakeTransportError
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    return FakeTransportError


def _closed_client_error() -> RuntimeError:
    return RuntimeError("Cannot send a request, as the client has been closed.")


def test_revision_is_pinned():
    assert PHRESH_REVISION == "eabec4b7a66324b79cc8a0ad856d1731dc26fe1a"


def test_training_retry_policy_defaults_to_exponential_backoff():
    config = PhreshTrainConfig()

    assert config.retry_policy() == {
        "max_attempts": 3,
        "initial_delay_seconds": 1.0,
        "backoff_multiplier": 2.0,
    }


def test_checkpoint_rejects_feature_mismatch(tmp_path: Path):
    path = tmp_path / "checkpoint.json"
    path.write_text(
        '{"feature_version":"wrong",'
        '"dataset_revision":"eabec4b7a66324b79cc8a0ad856d1731dc26fe1a"}'
    )
    with pytest.raises(ValueError, match="feature version"):
        validate_checkpoint(path)


def test_iter_phresh_streams_with_pinned_revision_and_discards_html():
    calls = []

    def load_dataset(name, *, split, revision, streaming):
        calls.append((name, split, revision, streaming))
        return FakeStream(
            [
                _raw_row(
                    0,
                    "Phish",
                    "<html><title>Sign in</title><input type='password'></html>",
                )
            ]
        )

    rows = list(iter_phresh("train", limit=1, load_dataset_func=load_dataset))

    assert calls == [("phreshphish/phreshphish", "train", PHRESH_REVISION, True)]
    assert len(rows) == 1
    assert rows[0]["label"] == 1
    assert rows[0]["title"] == "Sign in"
    assert rows[0]["dom_has_password_field"] == 1.0
    assert "html" not in rows[0]


def test_iter_phresh_validates_inputs_and_first_row_schema():
    def load_missing_schema(name, *, split, revision, streaming):
        return FakeStream([{"url": "https://example.test", "date": "2024-01-01", "label": "benign"}])

    with pytest.raises(ValueError, match="split must be one of"):
        list(iter_phresh("validation", load_dataset_func=load_missing_schema))
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        list(iter_phresh("train", limit=-1, load_dataset_func=load_missing_schema))
    with pytest.raises(ValueError, match="unsupported PhreshPhish revision"):
        list(iter_phresh("train", revision="other", load_dataset_func=load_missing_schema))
    with pytest.raises(ValueError, match=r"available columns: \['date', 'label', 'url'\]"):
        list(iter_phresh("train", load_dataset_func=load_missing_schema))


def test_iter_phresh_rejects_unknown_labels():
    def load_dataset(name, *, split, revision, streaming):
        return FakeStream([_raw_row(0, "mystery")])

    with pytest.raises(ValueError, match="unknown PhreshPhish label"):
        list(iter_phresh("train", load_dataset_func=load_dataset))


def test_derive_temporal_cutoff_projects_columns_and_uses_chronological_index_rule():
    source = FakeStream([_raw_row(index, "phish" if index % 2 else "benign") for index in range(10)])
    calls = []

    def load_dataset(name, *, split, revision, streaming):
        calls.append((name, split, revision, streaming))
        return source

    cutoff = derive_temporal_cutoff(
        load_dataset_func=load_dataset,
        retry_initial_delay_seconds=0.0,
    )

    assert calls == [("phreshphish/phreshphish", "train", PHRESH_REVISION, True)]
    assert source.selected_columns == ["date", "label"]
    assert cutoff == "2024-01-09T00:00:00Z"


def test_retryable_stream_classifier_is_narrow_and_optional(monkeypatch):
    from phishme import phresh

    FakeTransportError = _install_fake_httpx(monkeypatch)

    assert phresh._is_retryable_stream_exception(ConnectionError("network"))
    assert phresh._is_retryable_stream_exception(TimeoutError("timeout"))
    assert phresh._is_retryable_stream_exception(OSError("socket reset"))
    assert phresh._is_retryable_stream_exception(FakeTransportError("dns"))
    assert phresh._is_retryable_stream_exception(_closed_client_error())
    assert not phresh._is_retryable_stream_exception(RuntimeError("Cannot send a request"))
    assert not phresh._is_retryable_stream_exception(RuntimeError("programming bug"))
    assert not phresh._is_retryable_stream_exception(ValueError("schema changed"))
    assert not phresh._is_retryable_stream_exception(TypeError("bad record"))


@pytest.mark.parametrize("error_factory", ["httpx", "closed_client"])
def test_derive_temporal_cutoff_retries_transient_stream_errors_and_resets_client(
    error_factory,
    monkeypatch,
):
    from phishme import phresh

    if error_factory == "httpx":
        FakeTransportError = _install_fake_httpx(monkeypatch)
        error = FakeTransportError("temporary DNS failure")
    else:
        error = _closed_client_error()
    rows = [_raw_row(index, "phish" if index % 2 else "benign") for index in range(5)]
    calls = []
    scans: list[list[tuple[int, tuple[str, ...]]]] = []
    reset_calls = []

    def load_dataset(name, *, split, revision, streaming):
        calls.append((name, split, revision, streaming))
        scan_log: list[tuple[int, tuple[str, ...]]] = []
        scans.append(scan_log)
        if len(calls) == 1:
            return _ScriptedCutoffStream(rows, exception=error, fail_after=1, scan_log=scan_log)
        return _ScriptedCutoffStream(rows, scan_log=scan_log)

    monkeypatch.setattr(phresh, "_reset_huggingface_client", lambda: reset_calls.append("reset"))

    cutoff = derive_temporal_cutoff(load_dataset_func=load_dataset)

    assert cutoff == "2024-01-05T00:00:00Z"
    assert len(calls) == 2
    assert reset_calls == ["reset"]
    assert scans[0] == [(0, ("date", "label")), (1, ("date", "label"))]
    assert scans[1] == [(index, ("date", "label")) for index in range(5)]


def test_derive_temporal_cutoff_stops_after_three_retryable_attempts_and_resets_between_retries(
    monkeypatch,
):
    from phishme import phresh

    rows = [_raw_row(index, "benign") for index in range(3)]
    streams = []
    reset_calls = []

    def load_dataset(name, *, split, revision, streaming):
        stream = _ScriptedCutoffStream(rows, exception=TimeoutError("train-001 timeout"), fail_after=0)
        streams.append(stream)
        return stream

    monkeypatch.setattr(phresh, "_reset_huggingface_client", lambda: reset_calls.append("reset"))

    with pytest.raises(RuntimeError, match="cutoff stream failed after 3 attempts"):
        derive_temporal_cutoff(
            load_dataset_func=load_dataset,
            retry_initial_delay_seconds=0.0,
        )

    assert len(streams) == 3
    assert [stream.selected_columns for stream in streams] == [["date", "label"]] * 3
    assert reset_calls == ["reset", "reset"]


@pytest.mark.parametrize(
    ("stream_factory", "match"),
    [
        (
            lambda rows: _ScriptedCutoffStream(
                rows,
                exception=RuntimeError("generic programming failure"),
                fail_after=0,
            ),
            "generic programming failure",
        ),
        (lambda rows: _ScriptedCutoffStream([{"date": rows[0]["date"]}]), "projected PhreshPhish schema"),
        (lambda rows: _ScriptedCutoffStream([{**rows[0], "label": "unknown"}]), "unknown PhreshPhish label"),
    ],
)
def test_derive_temporal_cutoff_does_not_retry_schema_label_or_generic_runtime_errors(
    stream_factory,
    match,
    monkeypatch,
):
    from phishme import phresh

    rows = [_raw_row(index, "benign") for index in range(2)]
    calls = []
    reset_calls = []

    def load_dataset(name, *, split, revision, streaming):
        calls.append((name, split, revision, streaming))
        return stream_factory(rows)

    monkeypatch.setattr(phresh, "_reset_huggingface_client", lambda: reset_calls.append("reset"))

    with pytest.raises((RuntimeError, ValueError), match=match):
        derive_temporal_cutoff(load_dataset_func=load_dataset)

    assert len(calls) == 1
    assert reset_calls == []


def test_train_stream_retries_then_matches_clean_training(tmp_path: Path):
    records = [_canonical_record(index, index % 2) for index in range(5)]
    yielded_by_call: list[list[int]] = []

    def flaky_factory():
        call_index = len(yielded_by_call)
        yielded_by_call.append([])
        for index, record in enumerate(records):
            yielded_by_call[call_index].append(index)
            yield record
            if call_index == 0 and index == 2:
                raise TimeoutError("temporary network break")

    config = PhreshTrainConfig(
        batch_size=2,
        include_dom=False,
        cutoff="2024-01-09T00:00:00Z",
        retry_initial_delay_seconds=0.0,
    )
    model = train_stream(flaky_factory, tmp_path / "flaky", config, resume=False)
    clean_model = train_stream(lambda: iter(records), tmp_path / "clean", config, resume=False)

    frame = __import__("pandas").DataFrame(records)
    np.testing.assert_allclose(
        predict_scores(model, frame, include_dom=False, batch_size=2),
        predict_scores(clean_model, frame, include_dom=False, batch_size=2),
    )
    assert yielded_by_call[0] == [0, 1, 2]
    assert yielded_by_call[1][:2] == [0, 1]


def test_train_stream_retries_httpx_transport_error_during_iterator_construction(
    tmp_path: Path,
    monkeypatch,
):
    from phishme import phresh

    FakeTransportError = _install_fake_httpx(monkeypatch)
    records = [_canonical_record(index, index % 2) for index in range(4)]
    calls = []
    reset_calls = []

    def flaky_factory():
        calls.append("factory")
        if len(calls) == 1:
            raise FakeTransportError("temporary DNS failure")
        return iter(records)

    monkeypatch.setattr(phresh, "_reset_huggingface_client", lambda: reset_calls.append("reset"))

    config = PhreshTrainConfig(
        batch_size=2,
        include_dom=False,
        cutoff="2024-01-09T00:00:00Z",
        retry_initial_delay_seconds=0.0,
    )
    train_stream(flaky_factory, tmp_path, config, resume=False)

    assert calls == ["factory", "factory"]
    assert reset_calls == ["reset"]


def test_train_stream_retries_exact_closed_client_from_iterator_next(tmp_path: Path, monkeypatch):
    from phishme import phresh

    records = [_canonical_record(index, index % 2) for index in range(4)]
    calls = []
    reset_calls = []

    def flaky_factory():
        call_index = len(calls)
        calls.append("factory")
        for index, record in enumerate(records):
            yield record
            if call_index == 0 and index == 0:
                raise _closed_client_error()

    monkeypatch.setattr(phresh, "_reset_huggingface_client", lambda: reset_calls.append("reset"))

    config = PhreshTrainConfig(
        batch_size=2,
        include_dom=False,
        cutoff="2024-01-09T00:00:00Z",
        retry_initial_delay_seconds=0.0,
    )
    train_stream(flaky_factory, tmp_path, config, resume=False)

    assert calls == ["factory", "factory"]
    assert reset_calls == ["reset"]


@pytest.mark.parametrize(
    "records_factory",
    [
        lambda: (_ for _ in ()).throw(RuntimeError("programming bug")),
        lambda: iter([{"url": "https://example.test", "label": 2}]),
    ],
)
def test_train_stream_does_not_retry_generic_runtime_or_label_errors(
    records_factory,
    tmp_path: Path,
    monkeypatch,
):
    from phishme import phresh

    calls = []
    reset_calls = []

    def factory():
        calls.append("factory")
        return records_factory()

    monkeypatch.setattr(phresh, "_reset_huggingface_client", lambda: reset_calls.append("reset"))

    config = PhreshTrainConfig(batch_size=2, include_dom=False, cutoff="2024-01-09T00:00:00Z")
    with pytest.raises((RuntimeError, ValueError)):
        train_stream(factory, tmp_path, config, resume=False)

    assert calls == ["factory"]
    assert reset_calls == []


def test_train_stream_resumes_exact_position_and_rejects_incompatible_config(tmp_path: Path):
    first_records = [_canonical_record(index, index % 2) for index in range(4)]
    more_records = [_canonical_record(index, index % 2) for index in range(6)]
    config = PhreshTrainConfig(batch_size=2, include_dom=False, cutoff="2024-01-09T00:00:00Z")

    train_stream(lambda: iter(first_records), tmp_path, config, resume=False)
    seen = []

    def resumed_factory():
        for index, record in enumerate(more_records):
            seen.append(index)
            yield record

    resumed_model = train_stream(resumed_factory, tmp_path, config, resume=True)
    clean_model = train_stream(lambda: iter(more_records), tmp_path / "clean", config, resume=False)
    metadata = validate_checkpoint(_checkpoint_path(tmp_path))[1]
    frame = __import__("pandas").DataFrame(more_records)

    assert seen == list(range(6))
    assert metadata["processed_position"] == 6
    assert metadata["class_counts"] == {"0": 3, "1": 3}
    np.testing.assert_allclose(
        predict_scores(resumed_model, frame, include_dom=False, batch_size=2),
        predict_scores(clean_model, frame, include_dom=False, batch_size=2),
    )

    incompatible = PhreshTrainConfig(
        alpha=1e-3,
        batch_size=2,
        include_dom=False,
        cutoff="2024-01-09T00:00:00Z",
    )
    with pytest.raises(ValueError, match="checkpoint alpha mismatch"):
        train_stream(lambda: iter(more_records), tmp_path, incompatible, resume=True)


def test_validate_checkpoint_rejects_model_hash_and_type_mismatches(tmp_path: Path):
    config = PhreshTrainConfig(batch_size=2, include_dom=False, cutoff="2024-01-09T00:00:00Z")
    train_stream(
        lambda: iter([_canonical_record(0, 0), _canonical_record(1, 1)]),
        tmp_path / "hash",
        config,
        resume=False,
    )
    checkpoint = _checkpoint_path(tmp_path / "hash")
    metadata = validate_checkpoint(checkpoint)[1]
    model_path = checkpoint.parent / metadata["model_joblib"]

    model_path.write_bytes(model_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="model SHA-256"):
        validate_checkpoint(checkpoint)

    train_stream(
        lambda: iter([_canonical_record(0, 0), _canonical_record(1, 1)]),
        tmp_path / "type",
        config,
        resume=False,
    )
    checkpoint = _checkpoint_path(tmp_path / "type")
    metadata = validate_checkpoint(checkpoint)[1]
    model_path = checkpoint.parent / metadata["model_joblib"]
    joblib.dump({"not": "a classifier"}, model_path)
    metadata_text = checkpoint.read_text(encoding="utf-8").replace(
        metadata["model_sha256"],
        __import__("hashlib").sha256(model_path.read_bytes()).hexdigest(),
    )
    checkpoint.write_text(metadata_text, encoding="utf-8")
    with pytest.raises(TypeError, match="SGDClassifier"):
        validate_checkpoint(checkpoint)


def test_checkpoint_preserves_last_valid_metadata_when_metadata_replace_fails(tmp_path: Path, monkeypatch):
    from phishme import phresh

    config = PhreshTrainConfig(batch_size=2, include_dom=False, cutoff="2024-01-09T00:00:00Z")
    train_stream(
        lambda: iter([_canonical_record(0, 0), _canonical_record(1, 1)]),
        tmp_path,
        config,
        resume=False,
    )
    old_model, old_metadata = validate_checkpoint(_checkpoint_path(tmp_path))

    def fail_metadata_replace(payload, path):
        raise RuntimeError("metadata write failed")

    monkeypatch.setattr(phresh, "_write_checkpoint_metadata_atomically", fail_metadata_replace)
    with pytest.raises(RuntimeError, match="metadata write failed"):
        train_stream(
            lambda: iter([_canonical_record(index, index % 2) for index in range(4)]),
            tmp_path,
            config,
            resume=True,
        )

    model, metadata = validate_checkpoint(_checkpoint_path(tmp_path))
    assert metadata == old_metadata
    np.testing.assert_allclose(model.coef_, old_model.coef_)
    assert (tmp_path / old_metadata["model_joblib"]).exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_train_stream_final_partial_batch_counts_and_no_temp_leftovers(tmp_path: Path):
    records = [
        _canonical_record(0, 0, status="ok"),
        {"_reject_reason": "missing_url"},
        _canonical_record(1, 1, status="empty_document"),
        _canonical_record(2, 1, status="empty_title"),
    ]
    config = PhreshTrainConfig(batch_size=2, include_dom=False, cutoff="2024-01-09T00:00:00Z")

    train_stream(lambda: iter(records), tmp_path, config, resume=False)
    metadata = validate_checkpoint(_checkpoint_path(tmp_path))[1]

    assert metadata["processed_position"] == 4
    assert metadata["class_counts"] == {"0": 1, "1": 2}
    assert metadata["parse_counts"] == {"empty_document": 1, "empty_title": 1, "ok": 1}
    assert metadata["reject_counts"] == {"missing_url": 1}
    assert not list(tmp_path.rglob("*.tmp"))


def test_validate_checkpoint_rejects_inconsistent_metadata_counts(tmp_path: Path):
    records = [
        _canonical_record(0, 0),
        {"_reject_reason": "missing_url"},
        _canonical_record(1, 1),
    ]
    config = PhreshTrainConfig(batch_size=2, include_dom=False, cutoff="2024-01-09T00:00:00Z")
    train_stream(lambda: iter(records), tmp_path, config, resume=False)
    checkpoint = _checkpoint_path(tmp_path)
    metadata = _checkpoint_metadata(checkpoint)

    parse_tampered = json.loads(json.dumps(metadata))
    parse_tampered["parse_counts"]["ok"] += 1
    _write_checkpoint_metadata(checkpoint, parse_tampered)
    with pytest.raises(ValueError, match=r"sum\(class_counts\).*sum\(parse_counts\)"):
        validate_checkpoint(checkpoint)

    processed_tampered = json.loads(json.dumps(metadata))
    processed_tampered["reject_counts"]["missing_url"] += 1
    _write_checkpoint_metadata(checkpoint, processed_tampered)
    with pytest.raises(ValueError, match="accepted examples plus reject_counts.*processed_position"):
        validate_checkpoint(checkpoint)


def test_train_stream_resume_failures_preserve_valid_checkpoint(tmp_path: Path):
    config = PhreshTrainConfig(
        batch_size=2,
        include_dom=False,
        cutoff="2024-01-09T00:00:00Z",
        retry_initial_delay_seconds=0.0,
    )
    train_stream(
        lambda: iter([_canonical_record(0, 0), _canonical_record(1, 1)]),
        tmp_path,
        config,
        resume=False,
    )
    checkpoint = _checkpoint_path(tmp_path)
    old_model, old_metadata = validate_checkpoint(checkpoint, config)
    old_model_path = tmp_path / old_metadata["model_joblib"]
    old_model_hash = _file_sha256(old_model_path)
    attempts = 0

    def failing_factory():
        nonlocal attempts
        attempts += 1
        raise TimeoutError("stream unavailable")

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        train_stream(failing_factory, tmp_path, config, resume=True)

    model, metadata = validate_checkpoint(checkpoint, config)
    assert attempts == 3
    assert metadata == old_metadata
    assert _file_sha256(old_model_path) == old_model_hash == old_metadata["model_sha256"]
    np.testing.assert_allclose(model.coef_, old_model.coef_)
    assert not list(tmp_path.rglob("*.tmp"))


def test_validate_checkpoint_rejects_strict_json_adversarial_inputs(tmp_path: Path):
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        '{"schema":"phishme-phresh-checkpoint-v1","schema":"duplicate"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key 'schema'"):
        validate_checkpoint(duplicate_path)

    nan_path = tmp_path / "nan.json"
    nan_path.write_text('{"schema":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON constant NaN"):
        validate_checkpoint(nan_path)


def test_validate_checkpoint_rejects_missing_unknown_keys_and_model_path_escape(
    tmp_path: Path,
):
    config = PhreshTrainConfig(batch_size=2, include_dom=False, cutoff="2024-01-09T00:00:00Z")
    train_stream(
        lambda: iter([_canonical_record(0, 0), _canonical_record(1, 1)]),
        tmp_path,
        config,
        resume=False,
    )
    checkpoint = _checkpoint_path(tmp_path)
    metadata = _checkpoint_metadata(checkpoint)

    missing_key = dict(metadata)
    missing_key.pop("seed")
    _write_checkpoint_metadata(checkpoint, missing_key)
    with pytest.raises(ValueError, match=r"missing \['seed'\]"):
        validate_checkpoint(checkpoint)

    unknown_key = {**metadata, "unexpected": True}
    _write_checkpoint_metadata(checkpoint, unknown_key)
    with pytest.raises(ValueError, match=r"unexpected keys \['unexpected'\]"):
        validate_checkpoint(checkpoint)

    escaped_path = {**metadata, "model_joblib": "../escape.joblib"}
    _write_checkpoint_metadata(checkpoint, escaped_path)
    with pytest.raises(ValueError, match="model_joblib path escapes"):
        validate_checkpoint(checkpoint)


def test_phresh_smoke_train_records_filters_strictly_before_cutoff():
    from phishme import __main__ as cli

    calls = []
    rows = [
        _canonical_record(0, 0),
        {**_canonical_record(1, 1), "date": "2024-01-09T00:00:00Z"},
        {**_canonical_record(2, 1), "date": "2024-01-10T00:00:00Z"},
        {**_canonical_record(3, 0), "date": "2024-01-08T23:59:59Z"},
    ]

    class FakePhreshModule:
        PHRESH_REVISION = PHRESH_REVISION

        @staticmethod
        def iter_phresh(split, *, revision, limit):
            calls.append(("iter", split, revision, limit))
            return iter(rows)

        @staticmethod
        def filter_by_cutoff(records, cutoff, *, keep):
            calls.append(("filter", cutoff, keep))
            return filter_by_cutoff(records, cutoff, keep=keep)

    filtered = list(
        cli._phresh_smoke_train_records(
            FakePhreshModule,
            limit=4,
            cutoff="2024-01-09T00:00:00Z",
        )
    )

    assert calls == [
        ("iter", "train", PHRESH_REVISION, 4),
        ("filter", "2024-01-09T00:00:00Z", "before"),
    ]
    assert [record["date"] for record in filtered] == [
        "2024-01-01T00:00:00Z",
        "2024-01-08T23:59:59Z",
    ]


def test_notebook_training_factory_filters_smoke_and_full_before_cutoff():
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "phishme_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    train_cell = next(cell for cell in notebook["cells"] if cell["id"] == "stream-train")
    source = "".join(train_cell["source"])

    assert 'return phresh.filter_by_cutoff(base, cutoff, keep="before")' in source
    assert "return base" not in source
