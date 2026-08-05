from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
import tldextract
from sklearn.model_selection import StratifiedGroupKFold

_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


def canonicalize_url(url: str) -> str:
    raw = str(url).strip()
    if not raw:
        raise ValueError("empty URL")
    parts = urlsplit(raw if "://" in raw else "//" + raw)
    host = (parts.hostname or "").lower()
    if not host:
        raise ValueError(f"URL has no host: {raw!r}")
    port = f":{parts.port}" if parts.port else ""
    path = "" if parts.path in ("", "/") else parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), host + port, path, parts.query, ""))


def registrable_domain(url: str) -> str:
    host = urlsplit(url if "://" in url else "//" + url).hostname or ""
    result = _EXTRACT(host)
    return result.top_domain_under_public_suffix or host.lower()


def _sample_id(url: str) -> str:
    return sha256(canonicalize_url(url).encode("utf-8")).hexdigest()


def load_phiusill(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"label": "string"})
    required = {"URL", "Title", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing PhiUSIIL columns: {missing}")
    raw_labels = frame["label"]
    unknown = raw_labels[raw_labels.isna() | ~raw_labels.isin(["0", "1"])]
    if not unknown.empty:
        labels = sorted(unknown.fillna("<NA>").unique())
        raise ValueError(f"unknown PhiUSIIL labels: {labels}")
    labels = raw_labels.astype(int)
    out = pd.DataFrame(
        {
            "url": frame["URL"].astype(str),
            "title": frame["Title"].fillna("").astype(str),
            "label": 1 - labels,
        }
    )
    out["sample_id"] = out["url"].map(_sample_id)
    out = out.drop_duplicates("sample_id", keep="first").reset_index(drop=True)
    out["group"] = out["url"].map(registrable_domain)
    return out


def split_local(frame: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold = np.full(len(frame), -1, dtype=np.int8)
    for number, (_, held_out) in enumerate(splitter.split(frame, frame["label"], frame["group"])):
        fold[held_out] = number
    parts = {
        "train": frame.loc[fold >= 2].reset_index(drop=True),
        "validation": frame.loc[fold == 1].reset_index(drop=True),
        "test": frame.loc[fold == 0].reset_index(drop=True),
    }
    for name, part in parts.items():
        if set(part["label"]) != {0, 1}:
            raise ValueError(f"{name} split does not contain both classes")
    names = tuple(parts)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if set(parts[left]["group"]) & set(parts[right]["group"]):
                raise AssertionError(f"domain overlap: {left}/{right}")
            if set(parts[left]["sample_id"]) & set(parts[right]["sample_id"]):
                raise AssertionError(f"sample overlap: {left}/{right}")
    return parts
