"""Run the official PhishLang MobileBERT over frozen PhreshPhish records on GPU.

Pipeline: records JSONL -> phishlang input CSV -> run_phishlang (CUDA) ->
predictions CSV (sample_id,label,score,model,source_commit) + manifest.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

from run_phishlang import run_phishlang

PHISHME_ROOT = Path(r"C:\Users\User\codespace\phishMe")
RECORDS = Path(r"D:\phishme-dataset\artifacts\phishpedia-full-v3\phresh-test-records.jsonl")
INPUT_CSV = Path(r"D:\phishme-dataset\artifacts\phishpedia-full-v3\phishlang-input.csv")
OUTPUT_CSV = Path(r"D:\phishme-dataset\artifacts\phishpedia-full-v3\phishlang-predictions.csv")
PHISHLANG_DIR = PHISHME_ROOT / "external" / "phishlang"


def records_to_input_csv(records_path: Path, output_path: Path) -> int:
    if output_path.exists():
        print(f"input csv already exists: {output_path}")
        return sum(1 for _ in open(output_path, encoding="utf-8")) - 1
    n = 0
    with records_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.writer(dst)
        writer.writerow(["sample_id", "label", "html"])
        for line in src:
            row = json.loads(line)
            writer.writerow([row["sample_id"], row["label"], row["html"]])
            n += 1
    print(f"wrote {n} rows to {output_path}")
    return n


def model_factory(model_dir):
    from transformers import MobileBertForSequenceClassification

    model = MobileBertForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    assert torch.cuda.is_available(), "CUDA not available"
    return model.to("cuda")


class _CudaTokenizer:
    """Tokenizer wrapper: encode on GPU, decode on CPU (the script's model lives on CUDA)."""

    def __init__(self, tokenizer):
        self._tok = tokenizer

    def __call__(self, *args, **kwargs):
        out = self._tok(*args, **kwargs)
        if hasattr(out, "to"):  # transformers 5.x BatchEncoding has .to() but is not a dict
            return out.to("cuda")
        if isinstance(out, dict):
            return {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in out.items()}
        return out

    def decode(self, *args, **kwargs):
        args = [a.cpu() if hasattr(a, "cpu") else a for a in args]
        return self._tok.decode(*args, **kwargs)


def tokenizer_factory(model_dir):
    from transformers import MobileBertTokenizer

    return _CudaTokenizer(MobileBertTokenizer.from_pretrained(str(model_dir), local_files_only=True))


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE", file=sys.stderr)
        return 2
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    records_to_input_csv(RECORDS, INPUT_CSV)

    # HTML fields can exceed Python's default csv field limit (131072)
    csv.field_size_limit(2**31 - 1)

    manifest = run_phishlang(
        INPUT_CSV,
        OUTPUT_CSV,
        PHISHLANG_DIR,
        batch_size=16,
        mode="official",
        tokenizer_factory=tokenizer_factory,
        model_factory=model_factory,
        torch_module=torch,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
