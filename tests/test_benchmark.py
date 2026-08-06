import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from phishme.benchmark import base_rate_manifest, evaluate_base_rates


def _strict_json_hash(values: list[str]) -> str:
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_run_phishlang():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_phishlang.py"
    spec = importlib.util.spec_from_file_location("run_phishlang", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _fake_phishlang_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "phishlang"
    repo.mkdir()
    _git("init", cwd=repo)
    _git("config", "user.email", "test@example.test", cwd=repo)
    _git("config", "user.name", "Test User", cwd=repo)
    _git("remote", "add", "origin", "https://github.com/UTA-SPRLab/phishlang.git", cwd=repo)

    source = repo / "src"
    model = source / "model"
    model.mkdir(parents=True)
    (source / "patched_parser_prediction.py").write_text(
        "def generate_text_representation(html_content):\n"
        "    return 'official:' + html_content\n",
        encoding="utf-8",
    )
    (model / "config.json").write_text('{"model_type":"mobilebert"}', encoding="utf-8")
    (model / "vocab.txt").write_text("[PAD]\n[UNK]\n", encoding="utf-8")
    _git("add", ".", cwd=repo)
    _git("commit", "-m", "fake official fixture", cwd=repo)
    commit = _git("rev-parse", "HEAD", cwd=repo)

    manifest = []
    for path in sorted(model.rglob("*")):
        if path.is_file():
            manifest.append(
                [
                    path.relative_to(model).as_posix(),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                ]
            )
    canonical = json.dumps(manifest, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return repo, commit, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_base_rate_manifest_is_deterministic_and_close():
    labels = np.array([0] * 10000 + [1] * 1000)
    left = base_rate_manifest(labels, prevalence=0.01, seed=42)
    right = base_rate_manifest(labels, prevalence=0.01, seed=42)

    assert np.array_equal(left, right)
    assert abs(labels[left].mean() - 0.01) < 0.001
    assert set(np.flatnonzero(labels == 0)) <= set(left.tolist())
    assert len(left) == len(set(left.tolist()))


def test_base_rate_manifest_samples_negatives_when_positives_are_insufficient():
    labels = np.array([0] * 100 + [1] * 2)

    selected = base_rate_manifest(labels, prevalence=0.05, seed=9)

    assert np.array_equal(selected, base_rate_manifest(labels, prevalence=0.05, seed=9))
    assert len(selected) == 40
    assert np.count_nonzero(labels[selected] == 1) == 2
    assert np.count_nonzero(labels[selected] == 0) == 38
    assert len(selected) == len(set(selected.tolist()))


@pytest.mark.parametrize(
    ("labels", "prevalence", "seed", "match"),
    [
        (np.array([[0, 1]]), 0.1, 1, "one-dimensional"),
        (np.array([0, 1, 2]), 0.1, 1, "0 and 1"),
        (np.array([0, 0, 0]), 0.1, 1, "both classes"),
        (np.array([False, True]), 0.1, 1, "integer labels"),
        (np.array([0, 1]), 0.0, 1, "prevalence"),
        (np.array([0, 1]), 1.0, 1, "prevalence"),
        (np.array([0, 1]), math.nan, 1, "prevalence"),
        (np.array([0, 1]), 0.1, True, "seed"),
        (np.array([0, 1]), 0.1, 1.5, "seed"),
    ],
)
def test_base_rate_manifest_rejects_invalid_inputs(labels, prevalence, seed, match):
    with pytest.raises((TypeError, ValueError), match=match):
        base_rate_manifest(labels, prevalence=prevalence, seed=seed)


def test_evaluate_base_rates_records_metrics_ids_and_canonical_hashes():
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.05, 0.1, 0.6, 0.2, 0.8, 0.9])
    sample_ids = np.array([f"id-{index}" for index in range(len(labels))])

    report = evaluate_base_rates(labels, scores, 0.5, [0.5, 0.25], 123, sample_ids=sample_ids)

    assert report["schema"] == "phishme-base-rate-evaluation-v1"
    assert report["threshold"] == 0.5
    assert [entry["requested_prevalence"] for entry in report["rates"]] == [0.5, 0.25]
    first = report["rates"][0]
    selected_ids = [sample_ids[index] for index in first["selected_indices"]]
    assert first["selected_sample_ids"] == selected_ids
    assert first["sample_id_manifest_sha256"] == _strict_json_hash(selected_ids)
    assert first["counts"] == {"total": 4, "benign": 2, "phishing": 2}
    assert first["actual_prevalence"] == 0.5
    assert first["metrics"]["confusion_matrix"] == [[1, 1], [0, 2]]
    json.dumps(report, separators=(",", ":"), sort_keys=True, allow_nan=False)


@pytest.mark.parametrize(
    ("labels", "scores", "threshold", "rates", "sample_ids", "match"),
    [
        (np.array([0, 1]), np.array([0.1]), 0.5, [0.5], None, "same length"),
        (np.array([0, 1]), np.array([0.1, math.inf]), 0.5, [0.5], None, "finite"),
        (np.array([0, 1]), np.array([0.1, 0.9]), math.nan, [0.5], None, "threshold"),
        (np.array([0, 1]), np.array([0.1, 0.9]), 0.5, [0.5, 0.5], None, "unique"),
        (np.array([0, 1]), np.array([0.1, 0.9]), 0.5, [1.0], None, "prevalence"),
        (np.array([0, 1]), np.array([0.1, 0.9]), 0.5, [0.5], ["a", "a"], "sample_ids"),
        (np.array([0, 1]), np.array([0.1, 0.9]), 0.5, [0.5], ["a"], "sample_ids"),
    ],
)
def test_evaluate_base_rates_rejects_invalid_inputs(
    labels,
    scores,
    threshold,
    rates,
    sample_ids,
    match,
):
    with pytest.raises((TypeError, ValueError), match=match):
        evaluate_base_rates(labels, scores, threshold, rates, 7, sample_ids=sample_ids)


class _FakeTokenIds:
    def __init__(self, values):
        self._values = list(values)

    def squeeze(self):
        return self

    def size(self, dimension):
        assert dimension == 0
        return len(self._values)

    def __getitem__(self, key):
        return _FakeTokenIds(self._values[key])


class _FakeBatch(dict):
    def __init__(self, token_count: int):
        super().__init__(input_ids=_FakeTokenIds(range(token_count)))
        self.input_ids = self["input_ids"]


class _FakeTokenizer:
    def __call__(self, text, **_kwargs):
        return _FakeBatch(130 if "long" in text else 12)

    def decode(self, _window, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "decoded-long-window"


class _FakeScalar:
    def __init__(self, value: float):
        self._value = value

    def item(self):
        return self._value


class _FakeProbabilities:
    def __init__(self, phishing_probability: float):
        self._phishing_probability = phishing_probability

    def __getitem__(self, key):
        assert key == (0, 1)
        return _FakeScalar(self._phishing_probability)


class _FakeTorch:
    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    @staticmethod
    def no_grad():
        return _FakeTorch._NoGrad()

    @staticmethod
    def softmax(logits, dim):
        assert dim == -1
        benign, phishing = logits[0]
        denominator = math.exp(benign) + math.exp(phishing)
        return _FakeProbabilities(math.exp(phishing) / denominator)


class _FakeOutputs:
    def __init__(self):
        self.logits = [[0.0, math.log(3.0)]]


class _MissingLogitsOutputs:
    pass


class _FakeModel:
    outputs_class = _FakeOutputs

    def eval(self):
        self.evaluated = True
        return self

    def __call__(self, **inputs):
        assert "input_ids" in inputs
        return self.outputs_class()


class _MissingLogitsModel(_FakeModel):
    outputs_class = _MissingLogitsOutputs


class _ProbabilityTorch:
    def __init__(self, phishing_probability: float):
        self._phishing_probability = phishing_probability

    @staticmethod
    def no_grad():
        return _FakeTorch._NoGrad()

    def softmax(self, logits, dim):
        assert logits == [[0.0, math.log(3.0)]]
        assert dim == -1
        return _FakeProbabilities(self._phishing_probability)


def _write_input_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "html", "extra"])
        writer.writeheader()
        writer.writerow({"sample_id": "s-1", "label": "0", "html": "<p>short</p>", "extra": "kept out"})
        writer.writerow({"sample_id": "s-2", "label": "1", "html": "<p>long</p>", "extra": "kept out"})


def _write_long_input_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "html"])
        writer.writeheader()
        writer.writerow({"sample_id": "s-1", "label": "1", "html": "<p>long</p>"})


def _assert_no_temporary_artifacts(module, output_csv: Path) -> None:
    sidecar_path = module.manifest_path_for(output_csv)
    assert not output_csv.with_suffix(output_csv.suffix + ".tmp").exists()
    assert not sidecar_path.with_suffix(sidecar_path.suffix + ".tmp").exists()


def _assert_no_prediction_artifacts(module, output_csv: Path) -> None:
    assert not output_csv.exists()
    assert not module.manifest_path_for(output_csv).exists()
    _assert_no_temporary_artifacts(module, output_csv)


def test_run_phishlang_writes_exact_columns_order_scores_and_sidecar(tmp_path: Path):
    module = _load_run_phishlang()
    repo, commit, model_hash = _fake_phishlang_repo(tmp_path)
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "predictions.csv"
    _write_input_csv(input_csv)
    expected_long_score = format(math.exp(math.log(3.0)) / (1.0 + math.exp(math.log(3.0))), ".17g")

    manifest = module.run_phishlang(
        input_csv,
        output_csv,
        repo,
        tokenizer_factory=lambda _path: _FakeTokenizer(),
        model_factory=lambda _path: _FakeModel(),
        torch_module=_FakeTorch,
    )

    with output_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "sample_id": "s-1",
            "label": "0",
            "score": "0",
            "model": "phishlang-official-mobilebert",
            "source_commit": commit,
        },
        {
            "sample_id": "s-2",
            "label": "1",
            "score": expected_long_score,
            "model": "phishlang-official-mobilebert",
            "source_commit": commit,
        },
    ]
    assert list(rows[0]) == ["sample_id", "label", "score", "model", "source_commit"]
    sidecar = json.loads((tmp_path / "predictions.csv.manifest.json").read_text(encoding="utf-8"))
    assert sidecar == manifest
    assert sidecar["source_commit"] == commit
    assert sidecar["model_tree_sha256"] == model_hash
    assert sidecar["mode"] == "official"
    assert sidecar["input"]["rows"] == 2
    assert sidecar["output"]["columns"] == ["sample_id", "label", "score", "model", "source_commit"]
    assert sidecar["output"]["sha256"] == hashlib.sha256(output_csv.read_bytes()).hexdigest()


@pytest.mark.parametrize("phishing_probability", [math.nan, math.inf, -0.01, 1.01])
def test_run_phishlang_rejects_non_finite_or_out_of_range_phishing_probability(
    tmp_path: Path,
    phishing_probability: float,
):
    module = _load_run_phishlang()
    repo, _commit, _model_hash = _fake_phishlang_repo(tmp_path)
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "predictions.csv"
    _write_long_input_csv(input_csv)

    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        module.run_phishlang(
            input_csv,
            output_csv,
            repo,
            tokenizer_factory=lambda _path: _FakeTokenizer(),
            model_factory=lambda _path: _FakeModel(),
            torch_module=_ProbabilityTorch(phishing_probability),
        )

    _assert_no_prediction_artifacts(module, output_csv)


def test_run_phishlang_rejects_non_string_text_representation_without_outputs(tmp_path: Path):
    module = _load_run_phishlang()
    repo, _commit, _model_hash = _fake_phishlang_repo(tmp_path)
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "predictions.csv"
    _write_long_input_csv(input_csv)

    with pytest.raises(TypeError, match="return a string"):
        module.run_phishlang(
            input_csv,
            output_csv,
            repo,
            tokenizer_factory=lambda _path: _FakeTokenizer(),
            model_factory=lambda _path: _FakeModel(),
            torch_module=_FakeTorch,
            generate_text_representation=lambda _html: None,
        )

    _assert_no_prediction_artifacts(module, output_csv)


def test_run_phishlang_rejects_model_output_missing_logits_without_outputs(tmp_path: Path):
    module = _load_run_phishlang()
    repo, _commit, _model_hash = _fake_phishlang_repo(tmp_path)
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "predictions.csv"
    _write_long_input_csv(input_csv)

    with pytest.raises(TypeError, match="missing logits"):
        module.run_phishlang(
            input_csv,
            output_csv,
            repo,
            tokenizer_factory=lambda _path: _FakeTokenizer(),
            model_factory=lambda _path: _MissingLogitsModel(),
            torch_module=_FakeTorch,
        )

    _assert_no_prediction_artifacts(module, output_csv)


def test_run_phishlang_rejects_overwriting_input_without_temporary_files(tmp_path: Path):
    module = _load_run_phishlang()
    repo, _commit, _model_hash = _fake_phishlang_repo(tmp_path)
    input_csv = tmp_path / "input.csv"
    _write_input_csv(input_csv)
    original_input = input_csv.read_bytes()

    with pytest.raises(ValueError, match="overwrite --input"):
        module.run_phishlang(
            input_csv,
            input_csv,
            repo,
            tokenizer_factory=lambda _path: _FakeTokenizer(),
            model_factory=lambda _path: _FakeModel(),
            torch_module=_FakeTorch,
        )

    assert input_csv.read_bytes() == original_input
    assert not module.manifest_path_for(input_csv).exists()
    _assert_no_temporary_artifacts(module, input_csv)


def test_run_phishlang_rejects_output_inside_phishlang_repo_without_outputs(tmp_path: Path):
    module = _load_run_phishlang()
    repo, _commit, _model_hash = _fake_phishlang_repo(tmp_path)
    input_csv = tmp_path / "input.csv"
    output_csv = repo / "predictions.csv"
    _write_input_csv(input_csv)

    with pytest.raises(ValueError, match="outside the PhishLang repository"):
        module.run_phishlang(
            input_csv,
            output_csv,
            repo,
            tokenizer_factory=lambda _path: _FakeTokenizer(),
            model_factory=lambda _path: _FakeModel(),
            torch_module=_FakeTorch,
        )

    _assert_no_prediction_artifacts(module, output_csv)


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([{"sample_id": "", "label": "0", "html": "<p>x</p>"}], "sample_id"),
        ([{"sample_id": "a", "label": "2", "html": "<p>x</p>"}], "label"),
        ([{"sample_id": "a", "label": "0", "html": ""}], "html"),
        (
            [
                {"sample_id": "a", "label": "0", "html": "<p>x</p>"},
                {"sample_id": "a", "label": "1", "html": "<p>y</p>"},
            ],
            "sample_id",
        ),
    ],
)
def test_run_phishlang_rejects_malformed_input_rows(tmp_path: Path, rows, match):
    module = _load_run_phishlang()
    repo, _commit, _model_hash = _fake_phishlang_repo(tmp_path)
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "predictions.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "html"])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=match):
        module.run_phishlang(
            input_csv,
            output_csv,
            repo,
            tokenizer_factory=lambda _path: _FakeTokenizer(),
            model_factory=lambda _path: _FakeModel(),
            torch_module=_FakeTorch,
        )


def test_run_phishlang_rejects_dirty_or_wrong_source_repositories(tmp_path: Path):
    module = _load_run_phishlang()
    repo, _commit, _model_hash = _fake_phishlang_repo(tmp_path)
    input_csv = tmp_path / "input.csv"
    _write_input_csv(input_csv)

    _git("remote", "set-url", "origin", "https://github.com/example/not-phishlang.git", cwd=repo)
    with pytest.raises(ValueError, match="official PhishLang"):
        module.run_phishlang(
            input_csv,
            tmp_path / "bad-source.csv",
            repo,
            tokenizer_factory=lambda _path: _FakeTokenizer(),
            model_factory=lambda _path: _FakeModel(),
            torch_module=_FakeTorch,
        )

    _git("remote", "set-url", "origin", "https://github.com/UTA-SPRLab/phishlang.git", cwd=repo)
    (repo / "src" / "model" / "dirty.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        module.run_phishlang(
            input_csv,
            tmp_path / "dirty.csv",
            repo,
            tokenizer_factory=lambda _path: _FakeTokenizer(),
            model_factory=lambda _path: _FakeModel(),
            torch_module=_FakeTorch,
        )


@pytest.mark.integration
def test_live_phishlang_smoke_scores_one_frozen_row_with_local_official_model(
    tmp_path: Path,
    monkeypatch,
):
    phishlang_dir = os.environ.get("PHISHLANG_DIR")
    if not phishlang_dir:
        pytest.skip("PHISHLANG_DIR is not set")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    module = _load_run_phishlang()
    repository = Path(phishlang_dir)
    source = module.inspect_phishlang_source(repository)
    input_csv = tmp_path / "frozen.csv"
    output_csv = tmp_path / "predictions.csv"
    visible_text = " ".join(["secure account login verification invoice payment portal"] * 220)
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "html"])
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "live-1",
                "label": "1",
                "html": f"<html><body><main><p>{visible_text}</p></main></body></html>",
            }
        )

    manifest = module.run_phishlang(input_csv, output_csv, repository, batch_size=1)

    with output_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(module.OUTPUT_COLUMNS)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "live-1"
    assert rows[0]["label"] == "1"
    assert rows[0]["model"] == module.MODEL_LABELS["official"]
    assert rows[0]["source_commit"] == source["source_commit"]
    score = float(rows[0]["score"])
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0

    sidecar = json.loads(module.manifest_path_for(output_csv).read_text(encoding="utf-8"))
    assert sidecar == manifest
    assert sidecar["output"]["rows"] == 1
    assert sidecar["output"]["columns"] == list(module.OUTPUT_COLUMNS)
    assert sidecar["source_commit"] == source["source_commit"]
    assert source["generate_text_representation_path"] == "src/patched_parser_prediction.py"
    assert Path(sidecar["model_path"]) == repository.resolve() / module.MODEL_RELATIVE_PATH
    assert sidecar["model_tree_sha256"] == module.model_tree_sha256(
        repository.resolve() / module.MODEL_RELATIVE_PATH
    )
