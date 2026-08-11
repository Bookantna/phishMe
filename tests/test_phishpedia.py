# tests/test_phishpedia.py
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from phishme import phishpedia
from phishme.phishpedia import load_phishpedia, load_phishpedia_archives

SPLIT_CSV = "train_test_val_split_30.csv"


def _write_zip(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_load_phishpedia_archives_reads_official_paired_layouts(tmp_path: Path):
    phish_zip = tmp_path / "phish.zip"
    benign_zip = tmp_path / "benign.zip"
    _write_zip(
        phish_zip,
        {
            "PayPal+2020-01-01/info.txt": repr(
                {
                    "url": "https://evil.example/login",
                    "brand": "PayPal Inc.",
                    "family_id": "kit-1",
                    "isotime": "2020-01-01T00:00:00Z",
                }
            ),
            "PayPal+2020-01-01/html.txt": "<html><title>verify account</title></html>",
        },
    )
    _write_zip(
        benign_zip,
        {
            "good.example/info.txt": "https://good.example/",
            "good.example/html.txt": "<html><title>welcome</title></html>",
        },
    )

    frame = load_phishpedia_archives(phish_zip, benign_zip)

    assert frame["label"].tolist() == [1, 0]
    assert frame["title"].tolist() == ["verify account", "welcome"]
    assert frame["sample_id"].is_unique
    assert set(frame["source_class"]) == {"phishing", "benign"}
    assert frame.loc[frame["label"] == 1, "source_family_id"].item() == "kit-1"
    assert frame.loc[frame["label"] == 0, "source_family_id"].item() == ""


def test_load_phishpedia_archives_limit_is_class_balanced(tmp_path: Path):
    phish_zip = tmp_path / "phish.zip"
    benign_zip = tmp_path / "benign.zip"
    _write_zip(
        phish_zip,
        {
            **{
                f"p{index}/info.txt": repr(
                    {"url": f"https://evil{0 if index == 1 else index}.example/"}
                )
                for index in range(4)
            },
            **{f"p{index}/html.txt": "<html></html>" for index in range(4)},
        },
    )
    _write_zip(
        benign_zip,
        {
            **{f"b{index}/info.txt": f"https://good{index}.example/" for index in range(3)},
            **{f"b{index}/html.txt": "<html></html>" for index in range(3)},
        },
    )

    frame = load_phishpedia_archives(phish_zip, benign_zip, limit=4)

    assert frame["label"].value_counts().to_dict() == {1: 2, 0: 2}


def test_load_phishpedia_archives_groups_connected_domains_and_families(tmp_path: Path):
    phish_zip = tmp_path / "phish.zip"
    benign_zip = tmp_path / "benign.zip"
    _write_zip(
        phish_zip,
        {
            "p1/info.txt": repr({"url": "https://one.example/a", "family_id": "kit-a"}),
            "p1/html.txt": "<html></html>",
            "p2/info.txt": repr({"url": "https://two.example/b", "family_id": "kit-a"}),
            "p2/html.txt": "<html></html>",
            "p3/info.txt": repr({"url": "https://one.example/c", "family_id": "kit-b"}),
            "p3/html.txt": "<html></html>",
        },
    )
    _write_zip(
        benign_zip,
        {
            "good.example/info.txt": "https://good.example/",
            "good.example/html.txt": "<html></html>",
        },
    )

    frame = load_phishpedia_archives(phish_zip, benign_zip)
    phishing = frame[frame["label"] == 1]

    assert phishing["group"].nunique() == 1
    assert set(phishing["registrable_domain"]) == {"one.example", "two.example"}


def test_load_phishpedia_archives_treats_none_family_as_missing(tmp_path: Path):
    phish_zip = tmp_path / "phish.zip"
    benign_zip = tmp_path / "benign.zip"
    _write_zip(
        phish_zip,
        {
            "p1/info.txt": repr({"url": "https://one.example/a", "family_id": None}),
            "p1/html.txt": "<html></html>",
            "p2/info.txt": repr({"url": "https://two.example/b", "family_id": None}),
            "p2/html.txt": "<html></html>",
        },
    )
    _write_zip(
        benign_zip,
        {
            "b1/info.txt": "https://good.example/",
            "b1/html.txt": "<html></html>",
        },
    )

    frame = load_phishpedia_archives(phish_zip, benign_zip)
    phishing = frame[frame["label"] == 1]

    assert phishing["source_family_id"].tolist() == ["", ""]
    assert phishing["group"].nunique() == 2


def test_load_phishpedia_archives_skips_malformed_samples(tmp_path: Path):
    phish_zip = tmp_path / "phish.zip"
    benign_zip = tmp_path / "benign.zip"
    _write_zip(
        phish_zip,
        {
            "bad-metadata/info.txt": "{'url': }",
            "bad-metadata/html.txt": "<html></html>",
            "bad-url/info.txt": repr({"url": "http:///missing-host"}),
            "bad-url/html.txt": "<html></html>",
            "shot-only/shot.png": "not used",
            "valid/info.txt": repr({"url": "https://evil.example/"}),
            "valid/html.txt": "<html></html>",
        },
    )
    _write_zip(
        benign_zip,
        {
            "valid/info.txt": "https://good.example/",
            "valid/html.txt": "<html></html>",
        },
    )

    frame = load_phishpedia_archives(phish_zip, benign_zip)

    assert frame["label"].value_counts().to_dict() == {1: 1, 0: 1}
    assert frame.attrs["archive_audit"]["phishing"] == {
        "accepted": 1,
        "candidate_pairs": 3,
        "duplicate_url": 0,
        "invalid_metadata": 1,
        "invalid_url": 1,
        "missing_html": 1,
        "missing_info": 1,
        "sample_directories": 4,
        "scan_capped": False,
        "scanned_directories": 4,
    }


def test_load_phishpedia_archives_rejects_oversized_members(tmp_path: Path, monkeypatch):
    phish_zip = tmp_path / "phish.zip"
    benign_zip = tmp_path / "benign.zip"
    _write_zip(
        phish_zip,
        {
            "p/info.txt": repr({"url": "https://evil.example/"}),
            "p/html.txt": "123456",
        },
    )
    _write_zip(
        benign_zip,
        {
            "b/info.txt": "https://good.example/",
            "b/html.txt": "<html></html>",
        },
    )
    monkeypatch.setattr(phishpedia, "MAX_HTML_BYTES", 5)

    with pytest.raises(ValueError, match="html.txt is too large"):
        load_phishpedia_archives(phish_zip, benign_zip)


def test_load_phishpedia_archives_rejects_duplicate_members(tmp_path: Path):
    phish_zip = tmp_path / "phish.zip"
    benign_zip = tmp_path / "benign.zip"
    with zipfile.ZipFile(phish_zip, "w") as archive:
        archive.writestr("p/info.txt", repr({"url": "https://evil.example/"}))
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("p/info.txt", repr({"url": "https://shadow.example/"}))
        archive.writestr("p/html.txt", "<html></html>")
    _write_zip(
        benign_zip,
        {
            "b/info.txt": "https://good.example/",
            "b/html.txt": "<html></html>",
        },
    )

    with pytest.raises(ValueError, match="duplicate info.txt member"):
        load_phishpedia_archives(phish_zip, benign_zip)


def test_load_phishpedia_archives_rejects_cross_label_url_conflicts(tmp_path: Path):
    phish_zip = tmp_path / "phish.zip"
    benign_zip = tmp_path / "benign.zip"
    _write_zip(
        phish_zip,
        {
            "p/info.txt": repr({"url": "https://shared.example/"}),
            "p/html.txt": "<html></html>",
        },
    )
    _write_zip(
        benign_zip,
        {
            "b/info.txt": "https://shared.example/",
            "b/html.txt": "<html></html>",
        },
    )

    with pytest.raises(ValueError, match="conflicting labels"):
        load_phishpedia_archives(phish_zip, benign_zip)


def test_load_phishpedia_archives_reports_bad_zip(tmp_path: Path):
    phish_zip = tmp_path / "phish.zip"
    benign_zip = tmp_path / "benign.zip"
    phish_zip.write_bytes(b"not a zip")
    _write_zip(
        benign_zip,
        {
            "b/info.txt": "https://good.example/",
            "b/html.txt": "<html></html>",
        },
    )

    with pytest.raises(ValueError, match="invalid PhishPedia ZIP archive"):
        load_phishpedia_archives(phish_zip, benign_zip)


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


def test_load_phishpedia_carries_split_column(tmp_path: Path):
    """When the CSV has a split column, it is carried through to the frame."""
    phish_root = tmp_path / "phish_html"
    phish_root.mkdir()
    benign_root = tmp_path / "benign_html"
    benign_root.mkdir()
    (phish_root / "p1.html").write_text(
        "<html><title>verify password</title>"
        "<form action='https://evil.example/x'></form></html>"
    )
    (benign_root / "b1.html").write_text(
        "<html><title>welcome</title><a href='/home'>home</a></html>"
    )
    rows = [
        {"file_name": "p1", "label": 1, "url": "https://evil.example/login", "type": "phishing", "split": "train"},
        {"file_name": "b1", "label": 0, "url": "https://good.example/", "type": "benign", "split": "validation"},
    ]
    csv_path = tmp_path / SPLIT_CSV
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    frame = load_phishpedia(csv_path, phish_root, benign_root)
    assert "split" in frame.columns
    assert list(frame["split"]) == ["train", "validation"]


def test_load_phishpedia_rejects_missing_html_file(phishpedia_tree):
    csv_path, phish_root, benign_root = phishpedia_tree
    original = pd.read_csv(csv_path)
    extra = pd.DataFrame([{"file_name": "missing", "label": 1, "url": "https://m.example/", "type": "phishing"}])
    pd.concat([original, extra], ignore_index=True).to_csv(csv_path, index=False)
    frame = load_phishpedia(csv_path, phish_root, benign_root)
    assert len(frame) == 3  # missing file row rejected, not crashed
