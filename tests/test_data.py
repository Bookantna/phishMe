from pathlib import Path

import pandas as pd
import pytest

from phishme.data import canonicalize_url, load_phiusill


def test_canonical_url_removes_fragment_and_normalizes_host():
    assert canonicalize_url("HTTPS://Example.COM/#login") == "https://example.com"


def test_phiusill_inverts_labels_and_deduplicates(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "URL": ["https://safe.example/", "https://evil.example/#x", "https://evil.example/"],
            "Title": ["Safe", "Login", "Login"],
            "label": [1, 0, 0],
        }
    )
    path = tmp_path / "tiny.csv"
    frame.to_csv(path, index=False)
    loaded = load_phiusill(path)
    assert loaded["label"].tolist() == [0, 1]
    assert loaded["sample_id"].is_unique


def test_phiusill_rejects_unknown_label(tmp_path: Path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"URL": ["https://x.test"], "Title": ["x"], "label": [7]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="unknown PhiUSIIL labels"):
        load_phiusill(path)


@pytest.mark.parametrize(
    "bad_label",
    [
        pytest.param(0.5, id="fractional"),
        pytest.param(None, id="null"),
        pytest.param("not-a-number", id="non-numeric"),
    ],
)
def test_phiusill_rejects_malformed_raw_labels(tmp_path: Path, bad_label):
    path = tmp_path / "bad-raw-label.csv"
    pd.DataFrame({"URL": ["https://x.test"], "Title": ["x"], "label": [bad_label]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="unknown PhiUSIIL labels"):
        load_phiusill(path)
