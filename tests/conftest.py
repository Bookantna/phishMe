from pathlib import Path

import pandas as pd
import pytest

from phishme.data import PHIUSIIL_DOM_MAP


@pytest.fixture
def synthetic_phiusill_csv(tmp_path: Path) -> Path:
    rows = []
    dom_defaults = {source: 0 for source in PHIUSIIL_DOM_MAP}
    for domain_id in range(25):
        for page_id in range(2):
            phishing = domain_id % 2 == 1
            rows.append(
                {
                    **dom_defaults,
                    "URL": f"http{'s' if not phishing else ''}://d{domain_id}.example/p{page_id}",
                    "Title": "verify password" if phishing else "welcome",
                    "label": 0 if phishing else 1,
                }
            )
    path = tmp_path / "synthetic.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
