"""Build the local paired-archive PhishPedia notebook."""

import json
from pathlib import Path

CELLS = [
    {
        "id": "local-overview",
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Local PhishPedia paired-archive workflow\n",
            "\n",
            "This notebook replaces the obsolete Colab download/extract workflow. The official ",
            "PhishPedia site publishes separate phishing and benign ZIP archives; neither archive ",
            "contains the previously assumed split CSV. `phishme` reads both ZIPs directly.\n",
        ],
    },
    {
        "id": "local-paths",
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "from pathlib import Path\n",
            "\n",
            "PROJECT_ROOT = Path.cwd().resolve()\n",
            "PHISH_ZIP = Path(os.environ.get(\n",
            "    'PHISHPEDIA_PHISH_ZIP', PROJECT_ROOT / 'dataset' / 'phish_sample_30k.zip'\n",
            "))\n",
            "BENIGN_ZIP = Path(os.environ.get(\n",
            "    'PHISHPEDIA_BENIGN_ZIP', 'D:/phishme-dataset/benign_sample_30k.zip'\n",
            "))\n",
            "OUTPUT_DIR = Path(os.environ.get(\n",
            "    'PHISHPEDIA_OUTPUT_DIR', 'D:/phishme-dataset/artifacts/phishpedia-full-v3'\n",
            "))\n",
            "\n",
            "for archive in (PHISH_ZIP, BENIGN_ZIP):\n",
            "    assert archive.is_file(), f\"missing archive: {archive}\"\n",
            "    print(f\"{archive}: {archive.stat().st_size:,} bytes\")\n",
        ],
    },
    {
        "id": "zip-layout",
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import zipfile\n",
            "from collections import Counter\n",
            "from pathlib import PurePosixPath\n",
            "\n",
            "for archive_path in (PHISH_ZIP, BENIGN_ZIP):\n",
            "    with zipfile.ZipFile(archive_path) as archive:\n",
            "        files = [item for item in archive.infolist() if not item.is_dir()]\n",
            "        basenames = Counter(\n",
            "            PurePosixPath(item.filename.replace('\\\\', '/')).name.lower()\n",
            "            for item in files\n",
            "        )\n",
            "        roots = {item.filename.replace('\\\\', '/').split('/')[0] for item in files}\n",
            "    print(archive_path.name, {'files': len(files), 'sites': len(roots)})\n",
            "    print(basenames.most_common(10))\n",
        ],
    },
    {
        "id": "loader-smoke",
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from phishme.phishpedia import load_phishpedia_archives\n",
            "\n",
            "frame = load_phishpedia_archives(PHISH_ZIP, BENIGN_ZIP, limit=200)\n",
            "print('rows:', len(frame))\n",
            "print('labels:', frame['label'].value_counts().sort_index().to_dict())\n",
            "print('unique sample IDs:', frame['sample_id'].nunique())\n",
            "print('leakage-control groups:', frame['group'].nunique())\n",
            "print('archive audit:', frame.attrs['archive_audit'])\n",
        ],
    },
    {
        "id": "training-command",
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Full local training\n",
            "\n",
            "Run from the repository's Git Bash shell:\n",
            "\n",
            "```bash\n",
            ".venv/Scripts/python.exe -m phishme v3-train \\\n",
            "  --phish-zip '/path/to/phish_sample_30k.zip' \\\n",
            "  --benign-zip '/path/to/benign_sample_30k.zip' \\\n",
            "  --output '/path/to/artifacts/phishpedia-full-v3' \\\n",
            "  --epochs 3 --batch-size 2048 --seed 42\n",
            "```\n",
            "\n",
            "The command records archive SHA-256 values, constructs family/domain-disjoint ",
            "train/validation/holdout splits, trains linear, LightGBM, and hybrid variants, ",
            "selects thresholds only on validation, and scores the holdout once.\n",
        ],
    },
]

NOTEBOOK = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "phishMe (.venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "cells": CELLS,
}

path = Path(__file__).resolve().parent.parent / "notebooks" / "phishpedia_colab.ipynb"
path.write_text(json.dumps(NOTEBOOK, indent=1) + "\n", encoding="utf-8", newline="\n")
print(f"wrote {path} ({len(CELLS)} cells)")
