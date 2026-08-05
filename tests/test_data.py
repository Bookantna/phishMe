from pathlib import Path

import pandas as pd
import pytest

from phishme.data import PHIUSIIL_DOM_MAP, canonicalize_url, load_phiusill
from phishme.features import NUMERIC_FEATURES, url_numeric_features


def _phiusill_frame(rows: list[dict]) -> pd.DataFrame:
    defaults = {source: 0 for source in PHIUSIIL_DOM_MAP}
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_canonical_url_removes_fragment_and_normalizes_host():
    assert canonicalize_url("HTTPS://Example.COM/#login") == "https://example.com"


def test_phiusill_inverts_labels_and_deduplicates(tmp_path: Path):
    frame = _phiusill_frame(
        [
            {"URL": "https://safe.example/", "Title": "Safe", "label": 1},
            {"URL": "https://evil.example/#x", "Title": "Login", "label": 0},
            {"URL": "https://evil.example/", "Title": "Login", "label": 0},
        ]
    )
    path = tmp_path / "tiny.csv"
    frame.to_csv(path, index=False)
    loaded = load_phiusill(path)
    assert loaded["label"].tolist() == [0, 1]
    assert loaded["sample_id"].is_unique


def test_phiusill_rejects_unknown_label(tmp_path: Path):
    path = tmp_path / "bad.csv"
    _phiusill_frame([{"URL": "https://x.test", "Title": "x", "label": 7}]).to_csv(path, index=False)
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
    _phiusill_frame([{"URL": "https://x.test", "Title": "x", "label": bad_label}]).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="unknown PhiUSIIL labels"):
        load_phiusill(path)


def test_phiusill_uses_explicit_dom_allowlist_and_recomputed_url_values(tmp_path: Path):
    url = "https://safe.example/login?token=%2Fabc"
    source_row = {
        "URL": url,
        "Title": "Safe",
        "label": 1,
        "LineOfCode": 12,
        "NoOfImage": 7,
        "URLLength": 999,
        "NoOfLettersInURL": 999,
        "URLSimilarityIndex": 0.99,
        "TLDLegitimateProb": 0.98,
        "CharContinuationRate": 0.97,
        "URLCharProb": 0.96,
        "DomainTitleMatchScore": 0.95,
        "URLTitleMatchScore": 0.94,
        "Robots": 1,
        "NoOfURLRedirect": 8,
        "NoOfSelfRedirect": 9,
        "NoOfPopup": 10,
        "NoOfPopUp": 11,
        "FILENAME": "source.csv",
    }
    path = tmp_path / "features.csv"
    _phiusill_frame([source_row]).to_csv(path, index=False)

    loaded = load_phiusill(path)
    feature_columns = [column for column in loaded.columns if column.startswith("dom_")]

    assert feature_columns == [f"dom_{name}" for name in NUMERIC_FEATURES]
    assert loaded.loc[0, "dom_html_line_count"] == 12
    assert loaded.loc[0, "dom_image_count"] == 7
    for name, value in url_numeric_features(url).items():
        assert loaded.loc[0, f"dom_{name}"] == value

    leakage_fields = {
        "URLSimilarityIndex",
        "TLDLegitimateProb",
        "CharContinuationRate",
        "URLCharProb",
        "DomainTitleMatchScore",
        "URLTitleMatchScore",
        "Robots",
        "NoOfURLRedirect",
        "NoOfSelfRedirect",
        "NoOfPopup",
        "NoOfPopUp",
        "FILENAME",
        "label",
    }
    normalized_feature_columns = {column.removeprefix("dom_") for column in feature_columns}
    assert leakage_fields.isdisjoint(normalized_feature_columns)
    assert all(field not in loaded.columns for field in leakage_fields - {"label"})
    assert "dom_label" not in loaded.columns
