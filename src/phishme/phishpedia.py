from __future__ import annotations

from pathlib import Path

import pandas as pd

from phishme.data import _sample_id, canonicalize_url, registrable_domain
from phishme.features import NUMERIC_FEATURES, _html_features_with_status

SPLIT_CSV_NAME = "train_test_val_split_30.csv"
PHISHPEDIA_URL = "https://drive.google.com/file/d/12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g/view"
PHISHPEDIA_LICENSE = "CC0-1.0"
SPLITS = ("train", "validation", "test")
_REQUIRED_COLUMNS = frozenset({"file_name", "url", "label"})
_LABEL_MAP = {"phishing": 1, "benign": 0}


def load_phishpedia(
    csv_path,
    phish_html_root,
    benign_html_root,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load the PhishPedia split CSV and HTML corpus into the canonical frame."""
    frame = pd.read_csv(Path(csv_path), dtype=str)
    missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"missing PhishPedia columns: {missing}")

    # Normalize labels from either the numeric `label` or the `type` column.
    labels = []
    for _, row in frame.iterrows():
        labels.append(_canonical_label(row))
    frame = frame.assign(label=labels, url=frame["url"].str.strip())

    rows = []
    for _, row in frame.iterrows():
        if limit is not None and len(rows) >= limit:
            break
        if not row["url"]:
            continue
        html_path = _html_path(row, phish_html_root, benign_html_root)
        if html_path is None or not html_path.exists():
            continue
        title, dom_values, status = _html_features_with_status(
            row["url"], html_path.read_text(errors="replace")
        )
        canonical = {
            "url": row["url"],
            "title": title,
            "label": int(row["label"]),
            "sample_id": _sample_id(row["url"]),
            "group": registrable_domain(row["url"]),
        }
        canonical.update({f"dom_{name}": float(dom_values[name]) for name in NUMERIC_FEATURES})
        rows.append(canonical)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("no usable PhishPedia rows")
    out = out.drop_duplicates("sample_id", keep="first").reset_index(drop=True)
    return out


def _canonical_label(row: pd.Series) -> int:
    raw_label = str(row.get("label", "")).strip().lower()
    raw_type = str(row.get("type", "")).strip().lower()
    values = set()
    if raw_label in {"1", "0"}:
        values.add(int(raw_label))
    if raw_type in _LABEL_MAP:
        values.add(_LABEL_MAP[raw_type])
    if len(values) != 1:
        raise ValueError(
            f"unknown PhishPedia label: label={row.get('label')!r} type={row.get('type')!r}"
        )
    return values.pop()


def _html_path(row: pd.Series, phish_html_root, benign_html_root) -> Path | None:
    name = str(row["file_name"])
    if not name.endswith(".html"):
        name += ".html"
    label = int(row["label"])
    root = Path(phish_html_root) if label == 1 else Path(benign_html_root)
    return root / name
