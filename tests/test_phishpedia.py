# tests/test_phishpedia.py
from pathlib import Path

import pandas as pd
import pytest

from phishme.phishpedia import load_phishpedia

SPLIT_CSV = "train_test_val_split_30.csv"


@pytest.fixture
def phishpedia_tree(tmp_path: Path):
    phish_root = tmp_path / "phish_html"
    phish_root.mkdir()
    benign_root = tmp_path / "benign_html"
    benign_root.mkdir()
    (phish_root / "p1.html").write_text(
        "<html><title>verify password</title>"
        "<form action='https://evil.example/x'></form></html>"
    )
    (phish_root / "p2.html").write_text("<html><title>account alert</title></html>")
    (benign_root / "b1.html").write_text(
        "<html><title>welcome</title><a href='/home'>home</a></html>"
    )
    rows = [
        {"file_name": "p1", "label": 1, "url": "https://evil.example/login", "type": "phishing"},
        {"file_name": "p2", "label": 1, "url": "https://scam.example/alert", "type": "phishing"},
        {"file_name": "b1", "label": 0, "url": "https://good.example/", "type": "benign"},
    ]
    csv_path = tmp_path / SPLIT_CSV
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, phish_root, benign_root


def test_load_phishpedia_canonical_shape(phishpedia_tree):
    csv_path, phish_root, benign_root = phishpedia_tree
    frame = load_phishpedia(csv_path, phish_root, benign_root)

    # Required base columns present.
    assert {"url", "title", "label", "sample_id", "group"} <= set(frame.columns)

    # At least the four named DOM feature columns exist.
    required_dom = {"dom_html_line_count", "dom_external_ref_count",
                    "dom_has_title", "dom_has_password_field"}
    assert required_dom <= set(frame.columns)

    # Labels are only 0 and 1, sample IDs are unique, order matches input.
    assert set(frame["label"]) == {0, 1}
    assert frame["sample_id"].is_unique
    assert (frame["label"] == [1, 1, 0]).all()


def test_load_phishpedia_rejects_unknown_label(phishpedia_tree):
    csv_path, phish_root, benign_root = phishpedia_tree
    original = pd.read_csv(csv_path)
    extra = pd.DataFrame([{"file_name": "p9", "label": 2, "url": "https://x.example/", "type": "weird"}])
    pd.concat([original, extra], ignore_index=True).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="unknown"):
        load_phishpedia(csv_path, phish_root, benign_root)


def test_load_phishpedia_rejects_missing_html_file(phishpedia_tree):
    csv_path, phish_root, benign_root = phishpedia_tree
    original = pd.read_csv(csv_path)
    extra = pd.DataFrame([{"file_name": "missing", "label": 1, "url": "https://m.example/", "type": "phishing"}])
    pd.concat([original, extra], ignore_index=True).to_csv(csv_path, index=False)
    frame = load_phishpedia(csv_path, phish_root, benign_root)
    assert len(frame) == 3  # missing file row rejected, not crashed
