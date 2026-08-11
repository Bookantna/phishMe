"""Materialize a limited slice of the pinned PhreshPhish test split to JSONL."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from phishme import phresh
from phishme.cross_dataset import materialize_phresh_test

OUTPUT = Path(r"D:\phishme-dataset\artifacts\phishpedia-full-v3\phresh-test-records.jsonl")
LIMIT = 500


def main() -> int:
    if OUTPUT.exists():
        print(f"already exists: {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
        return 0
    print(f"streaming {phresh.PHRESH_DATASET}@{phresh.PHRESH_REVISION[:12]} split=test, limit={LIMIT}")
    raw_stream = phresh._load_dataset(
        phresh.PHRESH_DATASET, split="test",
        revision=phresh.PHRESH_REVISION, streaming=True,
    )
    result = materialize_phresh_test(iter(raw_stream), OUTPUT, limit=LIMIT)
    print(json.dumps(result, sort_keys=True))
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
