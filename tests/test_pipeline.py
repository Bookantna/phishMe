from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from phishme.features import FEATURE_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_NAMES = ("run.json", "splits.json", "model.json", "test-metrics.json")
MAX_ARTIFACT_BYTES = 25 * 1024 * 1024


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else src_path + os.pathsep + env["PYTHONPATH"]
    )
    return subprocess.run(
        [sys.executable, "-m", "phishme", *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _strict_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_json_constant,
    )


def _strict_json_text(text: str):
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_json_constant,
    )


def _reject_duplicate_keys(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate key {key!r}")
        output[key] = value
    return output


def _reject_json_constant(value: str):
    raise ValueError(f"invalid JSON constant {value}")


def _artifact_hashes(output: Path) -> dict[str, str]:
    return {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in ARTIFACT_NAMES}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_local_cli_writes_reproducible_artifacts(synthetic_phiusill_csv: Path, tmp_path: Path):
    output = tmp_path / "run"

    first = _run_cli(
        "local",
        "--csv",
        str(synthetic_phiusill_csv),
        "--output",
        str(output),
        "--epochs",
        "1",
    )
    assert first.returncode == 0, first.stderr
    summary = _strict_json_text(first.stdout)

    for name in ARTIFACT_NAMES:
        assert (output / name).exists()
        assert (output / name).stat().st_size <= MAX_ARTIFACT_BYTES
        _strict_json(output / name)
        assert summary["artifacts"][name]["bytes"] == (output / name).stat().st_size
        assert summary["artifacts"][name]["sha256"] == _file_sha256(output / name)
    assert not list(output.rglob("*.tmp"))

    first_hashes = _artifact_hashes(output)
    second = _run_cli(
        "local",
        "--csv",
        str(synthetic_phiusill_csv),
        "--output",
        str(output),
        "--epochs",
        "1",
    )
    assert second.returncode == 0, second.stderr
    assert _artifact_hashes(output) == first_hashes
    assert not list(output.rglob("*.tmp"))

    run = _strict_json(output / "run.json")
    splits = _strict_json(output / "splits.json")
    metrics = _strict_json(output / "test-metrics.json")
    model = _strict_json(output / "model.json")
    dataset_sha256 = _file_sha256(synthetic_phiusill_csv)

    assert metrics["average_precision"] >= 0.0
    assert run["feature_version"] == FEATURE_VERSION
    assert run["seed"] == 42
    assert run["dataset"]["sha256"] == dataset_sha256
    assert model["metadata"]["dataset_revision"] == dataset_sha256
    assert model["feature_version"] == FEATURE_VERSION
    assert run["git_commit"] and len(run["git_commit"]) == 40
    assert run["normalized_arguments"]["command"] == "local"
    assert run["normalized_arguments"]["csv"] == str(synthetic_phiusill_csv.resolve())
    assert run["normalized_arguments"]["output"] == str(output.resolve())
    assert run["normalized_arguments"]["epochs"] == 1
    assert run["normalized_arguments"]["batch_size"] == 2048
    assert run["normalized_arguments"]["seed"] == 42
    assert run["versions"]["python"]["version"]
    for dependency in ("joblib", "lxml", "numpy", "pandas", "scikit-learn", "scipy", "tldextract"):
        assert run["versions"]["dependencies"][dependency]["declared"]
        assert run["versions"]["dependencies"][dependency]["installed"]
    assert run["test_scored_once"] is True
    assert run["operation_counts"] == {
        "load_phiusill": 1,
        "split_local": 1,
        "test_predict_scores": 1,
    }

    assert run["candidate_attempt_count"] == 6
    assert run["candidate_failure_count"] == 0
    assert run["candidate_failures"] == []
    candidates = run["validation_candidates"]
    assert len(candidates) == 6
    assert {(item["include_dom"], item["alpha"]) for item in candidates} == {
        (False, 1e-5),
        (False, 1e-4),
        (False, 1e-3),
        (True, 1e-5),
        (True, 1e-4),
        (True, 1e-3),
    }
    for candidate in candidates:
        assert candidate["threshold_source"] == "validation"
        assert candidate["rank_key"] == [
            candidate["validation_ap"],
            candidate["validation_f1"],
            -candidate["alpha"],
            -int(candidate["include_dom"]),
        ]
        assert set(candidate["rank_key_fields"]) == {
            "validation_ap",
            "validation_f1",
            "-alpha",
            "-include_dom",
        }

    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["validation_ap"],
            candidate["validation_f1"],
            -candidate["alpha"],
            -int(candidate["include_dom"]),
        ),
    )
    assert run["selected_config"]["candidate_id"] == selected["candidate_id"]
    assert run["selected_config"]["include_dom"] == model["include_dom"]
    assert run["selected_config"]["threshold"] == model["threshold"]

    assert set(splits["splits"]) == {"train", "validation", "test"}
    for summary in splits["splits"].values():
        assert summary["rows"] == summary["class_counts"]["benign"] + summary["class_counts"]["phishing"]
        assert summary["rows"] == summary["label_counts"]["0"] + summary["label_counts"]["1"]
        assert len(summary["sample_manifest_sha256"]) == 64
    assert splits["disjointness"]["sample_id_disjoint"] is True
    assert splits["disjointness"]["group_disjoint"] is True

    for name in ("splits.json", "model.json", "test-metrics.json"):
        artifact = run["artifacts"][name]
        assert artifact["bytes"] == (output / name).stat().st_size
        assert artifact["sha256"] == hashlib.sha256((output / name).read_bytes()).hexdigest()


def test_audit_cli_prints_split_manifest_without_training(
    synthetic_phiusill_csv: Path, tmp_path: Path
):
    output = tmp_path / "run"
    local = _run_cli(
        "local",
        "--csv",
        str(synthetic_phiusill_csv),
        "--output",
        str(output),
        "--epochs",
        "1",
    )
    assert local.returncode == 0, local.stderr

    audit = _run_cli("audit", "--csv", str(synthetic_phiusill_csv))
    assert audit.returncode == 0, audit.stderr
    assert audit.stderr == ""
    report = _strict_json_text(audit.stdout)

    assert report["command"] == "audit"
    assert report["counts"] == {
        "raw_rows": 50,
        "canonical_rows": 50,
        "duplicate_rows": 0,
    }
    assert report["label_counts"]["canonical"] == {"0": 26, "1": 24}
    assert report["splits"] == _strict_json(output / "splits.json")["splits"]
    assert report["disjointness"]["sample_id_disjoint"] is True
    assert report["disjointness"]["group_disjoint"] is True
    assert "validation_candidates" not in report
    assert "selected_config" not in report
