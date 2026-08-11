from __future__ import annotations

import ast
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

from phishme.data import _sample_id, registrable_domain
from phishme.features import NUMERIC_FEATURES, _html_features_with_status

SPLIT_CSV_NAME = "train_test_val_split_30.csv"
PHISHPEDIA_URL = "https://drive.google.com/file/d/12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g/view"
PHISHPEDIA_LICENSE = "CC0-1.0"
SPLITS = ("train", "validation", "test")
_REQUIRED_COLUMNS = frozenset({"file_name", "url", "label"})
_LABEL_MAP = {"phishing": 1, "benign": 0}
_MISSING_FAMILY_IDS = frozenset({"", "-1", "0", "n/a", "na", "nan", "none", "null", "unknown"})
MAX_METADATA_BYTES = 1 * 1024 * 1024
MAX_HTML_BYTES = 16 * 1024 * 1024


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
        title, dom_values, _ = _html_features_with_status(
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
        if "split" in frame.columns:
            canonical["split"] = row["split"]
        rows.append(canonical)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("no usable PhishPedia rows")
    out = out.drop_duplicates("sample_id", keep="first").reset_index(drop=True)
    return out


def load_phishpedia_archives(phish_zip, benign_zip, *, limit: int | None = None) -> pd.DataFrame:
    """Load the official paired PhishPedia archives without extracting them."""
    class_limit = None if limit is None else max(1, int(limit) // 2)
    try:
        phish_rows, phish_audit = _load_archive_rows(phish_zip, label=1, limit=class_limit)
        benign_rows, benign_audit = _load_archive_rows(benign_zip, label=0, limit=class_limit)
        rows = [*phish_rows, *benign_rows]
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid PhishPedia ZIP archive: {exc}") from exc
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("no usable PhishPedia archive rows")
    conflicting_ids = out.groupby("sample_id")["label"].nunique()
    conflict_count = int((conflicting_ids > 1).sum())
    if conflict_count:
        raise ValueError(f"PhishPedia archives contain {conflict_count} URL(s) with conflicting labels")
    out = out.drop_duplicates("sample_id", keep="first").reset_index(drop=True)
    out["registrable_domain"] = out["group"]
    out["group"] = _connected_split_groups(out)
    out.attrs["archive_audit"] = {
        "phishing": phish_audit,
        "benign": benign_audit,
        "cross_label_conflicts": conflict_count,
    }
    return out


def _connected_split_groups(frame: pd.DataFrame) -> list[str]:
    """Connect rows sharing either a registrable domain or phishing-family ID."""
    parents: dict[str, str] = {}

    def find(token: str) -> str:
        parents.setdefault(token, token)
        while parents[token] != token:
            parents[token] = parents[parents[token]]
            token = parents[token]
        return token

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parents[second] = first

    row_tokens = []
    for _, row in frame.iterrows():
        tokens = [f"domain:{row['registrable_domain']}"]
        family_id = str(row["source_family_id"]).strip()
        if family_id:
            tokens.append(f"family:{family_id}")
            union(tokens[0], tokens[1])
        else:
            find(tokens[0])
        row_tokens.append(tokens)
    return [find(tokens[0]) for tokens in row_tokens]


def _load_archive_rows(archive_path, *, label: int, limit: int | None) -> tuple[list[dict], dict]:
    rows = []
    seen_sample_ids = set()
    audit = {
        "accepted": 0,
        "candidate_pairs": 0,
        "duplicate_url": 0,
        "invalid_metadata": 0,
        "invalid_url": 0,
        "missing_html": 0,
        "missing_info": 0,
        "sample_directories": 0,
        "scan_capped": False,
        "scanned_directories": 0,
    }
    with zipfile.ZipFile(Path(archive_path)) as archive:
        members: dict[str, dict[str, zipfile.ZipInfo]] = {}
        for member in archive.infolist():
            parts = PurePosixPath(member.filename.replace("\\", "/")).parts
            if member.is_dir() or len(parts) != 2:
                continue
            basename = parts[1].lower()
            sample_members = members.setdefault(parts[0], {})
            if basename in {"info.txt", "html.txt"}:
                if basename in sample_members:
                    raise ValueError(
                        f"duplicate {basename} member for PhishPedia sample {parts[0]!r}"
                    )
                sample_members[basename] = member

        audit["sample_directories"] = len(members)

        for source_member in sorted(members):
            audit["scanned_directories"] += 1
            sample = members[source_member]
            if "info.txt" not in sample:
                audit["missing_info"] += 1
            if "html.txt" not in sample:
                audit["missing_html"] += 1
            if not {"info.txt", "html.txt"} <= set(sample):
                continue
            audit["candidate_pairs"] += 1
            metadata = _read_archive_member(
                archive, sample["info.txt"], MAX_METADATA_BYTES, "info.txt"
            ).decode("utf-8", errors="replace").strip()
            if label == 1:
                try:
                    parsed = ast.literal_eval(metadata)
                except (SyntaxError, TypeError, ValueError):
                    audit["invalid_metadata"] += 1
                    continue
                if not isinstance(parsed, dict):
                    audit["invalid_metadata"] += 1
                    continue
                url = _metadata_text(parsed.get("url"))
                family_id = _metadata_text(parsed.get("family_id"))
                if family_id.lower() in _MISSING_FAMILY_IDS:
                    family_id = ""
                brand = _metadata_text(parsed.get("brand"))
                collected_at = _metadata_text(parsed.get("isotime"))
            else:
                url = metadata
                family_id = ""
                brand = ""
                collected_at = ""
            if not url:
                audit["invalid_url"] += 1
                continue
            try:
                sample_id = _sample_id(url)
            except (TypeError, ValueError):
                audit["invalid_url"] += 1
                continue
            if sample_id in seen_sample_ids:
                audit["duplicate_url"] += 1
                continue

            html = _read_archive_member(
                archive, sample["html.txt"], MAX_HTML_BYTES, "html.txt"
            ).decode("utf-8", errors="replace")
            title, dom_values, _ = _html_features_with_status(url, html)
            canonical = {
                "url": url,
                "title": title,
                "label": label,
                "sample_id": sample_id,
                "group": registrable_domain(url),
                "source_class": "phishing" if label == 1 else "benign",
                "source_member": source_member,
                "source_family_id": family_id,
                "source_brand": brand,
                "source_collected_at": collected_at,
            }
            canonical.update({f"dom_{name}": float(dom_values[name]) for name in NUMERIC_FEATURES})
            rows.append(canonical)
            seen_sample_ids.add(sample_id)
            audit["accepted"] += 1
            if limit is not None and len(rows) >= limit:
                audit["scan_capped"] = audit["scanned_directories"] < audit["sample_directories"]
                break
    return rows, audit


def _metadata_text(value) -> str:
    return "" if value is None else str(value).strip()


def _read_archive_member(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    max_bytes: int,
    label: str,
) -> bytes:
    if member.file_size > max_bytes:
        raise ValueError(f"{label} is too large: {member.filename!r}")
    with archive.open(member) as handle:
        payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"{label} is too large: {member.filename!r}")
    return payload


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
