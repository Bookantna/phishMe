from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from pathlib import Path

OFFICIAL_REMOTE = "https://github.com/UTA-SPRLab/phishlang.git"
OFFICIAL_REMOTE_ALIASES = frozenset(
    (
        OFFICIAL_REMOTE,
        "https://github.com/UTA-SPRLab/phishlang",
        "git@github.com:UTA-SPRLab/phishlang.git",
    )
)
GENERATOR_RELATIVE_PATH = Path("src") / "patched_parser_prediction.py"
MODEL_RELATIVE_PATH = Path("src") / "model"
MODEL_LABELS = {"official": "phishlang-official-mobilebert"}
OUTPUT_COLUMNS = ("sample_id", "label", "score", "model", "source_commit")
REQUIRED_INPUT_COLUMNS = frozenset(("sample_id", "label", "html"))
WINDOW_SIZE = 128
STRIDE = 64
MANIFEST_SCHEMA = "phishme-phishlang-predictions-v1"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = run_phishlang(
            args.input,
            args.output,
            args.phishlang_dir,
            batch_size=args.batch_size,
            mode=args.mode,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, separators=(",", ":"), sort_keys=True, allow_nan=False))
    return 0


def run_phishlang(
    input_csv: Path | str,
    output_csv: Path | str,
    phishlang_dir: Path | str,
    *,
    batch_size: int = 16,
    mode: str = "official",
    tokenizer_factory: Callable[[Path], object] | None = None,
    model_factory: Callable[[Path], object] | None = None,
    torch_module: object | None = None,
    generate_text_representation: Callable[[str], str] | None = None,
) -> dict:
    if mode != "official":
        raise ValueError("mode must be 'official'")
    batch_size_value = _validate_positive_integer("batch_size", batch_size)
    input_path = _validate_input_path(Path(input_csv))
    repository = _validate_repository_path(Path(phishlang_dir))
    output_path = _validate_output_path(Path(output_csv), input_path, repository)

    source = inspect_phishlang_source(repository)
    model_dir = repository / MODEL_RELATIVE_PATH
    model_hash = model_tree_sha256(model_dir)
    rows = _read_input_csv(input_path)
    generator = generate_text_representation or import_generate_text_representation(repository)
    tokenizer = _load_tokenizer(model_dir, tokenizer_factory)
    model = _load_model(model_dir, model_factory)
    if hasattr(model, "eval"):
        model.eval()
    torch_api = torch_module or _import_torch()

    predictions = []
    for batch in _chunks(rows, batch_size_value):
        for row in batch:
            text_representation = generator(row["html"])
            if not isinstance(text_representation, str):
                raise TypeError("generate_text_representation must return a string")
            probability = official_phishing_probability(
                model,
                tokenizer,
                torch_api,
                text_representation,
            )
            predictions.append(
                {
                    "sample_id": row["sample_id"],
                    "label": row["label"],
                    "score": format_probability(probability),
                    "model": MODEL_LABELS[mode],
                    "source_commit": source["source_commit"],
                }
            )

    _write_predictions_csv_atomically(predictions, output_path)
    sidecar_path = manifest_path_for(output_path)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "mode": mode,
        "model": MODEL_LABELS[mode],
        "source_repository": OFFICIAL_REMOTE,
        "observed_remote": source["observed_remote"],
        "source_commit": source["source_commit"],
        "source_status": "clean",
        "generate_text_representation_path": source["generate_text_representation_path"],
        "model_path": source["model_path"],
        "model_tree_sha256": model_hash,
        "input": {
            "path": str(input_path),
            "sha256": _file_sha256(input_path),
            "rows": len(rows),
            "columns_required": sorted(REQUIRED_INPUT_COLUMNS),
        },
        "output": {
            "path": str(output_path),
            "sha256": _file_sha256(output_path),
            "rows": len(predictions),
            "columns": list(OUTPUT_COLUMNS),
        },
    }
    _write_json_atomically(manifest, sidecar_path)
    return manifest


def inspect_phishlang_source(phishlang_dir: Path | str) -> dict[str, str]:
    repository = _validate_repository_path(Path(phishlang_dir))
    top_level = Path(_git(repository, "rev-parse", "--show-toplevel")).resolve()
    if top_level != repository:
        raise ValueError("--phishlang-dir must point to the PhishLang repository root")
    remote = _git(repository, "config", "--get", "remote.origin.url")
    if remote not in OFFICIAL_REMOTE_ALIASES:
        raise ValueError("--phishlang-dir must be the official PhishLang repository")
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError("PhishLang repository must be clean before scoring")
    commit = _git(repository, "rev-parse", "HEAD")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("PhishLang source commit is malformed")

    generator_path = repository / GENERATOR_RELATIVE_PATH
    model_path = repository / MODEL_RELATIVE_PATH
    if not generator_path.is_file():
        raise ValueError(f"missing official generator: {GENERATOR_RELATIVE_PATH.as_posix()}")
    if not model_path.is_dir():
        raise ValueError(f"missing official model directory: {MODEL_RELATIVE_PATH.as_posix()}")

    return {
        "source_commit": commit,
        "observed_remote": remote,
        "generate_text_representation_path": GENERATOR_RELATIVE_PATH.as_posix(),
        "model_path": str(model_path),
    }


def import_generate_text_representation(phishlang_dir: Path | str) -> Callable[[str], str]:
    repository = _validate_repository_path(Path(phishlang_dir))
    module_path = repository / GENERATOR_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location("_phishlang_official_prediction", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {GENERATOR_RELATIVE_PATH.as_posix()}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    generator = getattr(module, "generate_text_representation", None)
    if not callable(generator):
        raise TypeError("official generate_text_representation is missing")
    return generator


def official_phishing_probability(
    model,
    tokenizer,
    torch_module,
    text: str,
    *,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> float:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    token_ids = _input_ids(encoded).squeeze()
    total_tokens = _tensor_size(token_ids)
    max_phishing_probability = 0.0

    for start in range(0, total_tokens, stride):
        window = token_ids[start : start + window_size]
        if _tensor_size(window) < window_size:
            break
        inputs = tokenizer.decode(window, skip_special_tokens=True)
        encoded_window = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        if not isinstance(encoded_window, Mapping):
            raise TypeError("tokenizer window output must be a mapping")
        with torch_module.no_grad():
            outputs = model(**encoded_window)
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise TypeError("model output is missing logits")
        probabilities = torch_module.softmax(logits, dim=-1)
        phishing_probability = _probability_item(probabilities)
        _validate_probability(phishing_probability)
        max_phishing_probability = max(max_phishing_probability, phishing_probability)

    return max_phishing_probability


def model_tree_sha256(model_dir: Path | str) -> str:
    directory = Path(model_dir)
    if not directory.is_dir():
        raise ValueError("model directory is missing")
    entries = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            entries.append(
                [
                    path.relative_to(directory).as_posix(),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                ]
            )
    if not entries:
        raise ValueError("model directory contains no files")
    canonical = json.dumps(entries, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def manifest_path_for(output_path: Path) -> Path:
    return Path(f"{output_path}.manifest.json")


def format_probability(value: float) -> str:
    probability = _validate_probability(value)
    if probability == 0.0:
        return "0"
    if probability == 1.0:
        return "1"
    return format(probability, ".17g")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a frozen CSV with the official PhishLang MobileBERT adapter.",
    )
    parser.add_argument("--input", required=True, type=Path, help="frozen CSV with sample_id,label,html")
    parser.add_argument("--output", required=True, type=Path, help="destination prediction CSV")
    parser.add_argument(
        "--phishlang-dir",
        default=Path(os.environ.get("PHISHLANG_DIR", "external/phishlang")),
        type=Path,
        help="clean checkout of https://github.com/UTA-SPRLab/phishlang.git",
    )
    parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
        help="number of CSV rows per input chunk; MobileBERT scoring remains per sample",
    )
    parser.add_argument("--mode", default="official", choices=sorted(MODEL_LABELS), help="scoring mode")
    return parser


def _load_tokenizer(model_dir: Path, tokenizer_factory):
    if tokenizer_factory is not None:
        return tokenizer_factory(model_dir)
    try:
        from transformers import MobileBertTokenizer
    except ImportError as exc:
        raise ImportError("transformers with MobileBertTokenizer is required") from exc
    return MobileBertTokenizer.from_pretrained(str(model_dir), local_files_only=True)


def _load_model(model_dir: Path, model_factory):
    if model_factory is not None:
        return model_factory(model_dir)
    try:
        from transformers import MobileBertForSequenceClassification
    except ImportError as exc:
        raise ImportError("transformers with MobileBertForSequenceClassification is required") from exc
    return MobileBertForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required") from exc
    return torch


def _read_input_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("input CSV must have a header")
        missing = REQUIRED_INPUT_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"input CSV missing required columns: {sorted(missing)}")
        rows = []
        seen_sample_ids = set()
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"input CSV row {row_number} has too many fields")
            sample_id = _validate_sample_id(row.get("sample_id"), row_number)
            if sample_id in seen_sample_ids:
                raise ValueError(f"input CSV row {row_number} has duplicate sample_id")
            seen_sample_ids.add(sample_id)
            label = _validate_label(row.get("label"), row_number)
            html = _validate_html(row.get("html"), row_number)
            rows.append({"sample_id": sample_id, "label": label, "html": html})
    if not rows:
        raise ValueError("input CSV must contain at least one row")
    return rows


def _validate_sample_id(value, row_number: int) -> str:
    if not isinstance(value, str) or value == "" or any(character in value for character in "\x00\r\n"):
        raise ValueError(f"input CSV row {row_number} has malformed sample_id")
    return value


def _validate_label(value, row_number: int) -> str:
    if value not in {"0", "1"}:
        raise ValueError(f"input CSV row {row_number} has malformed label")
    return value


def _validate_html(value, row_number: int) -> str:
    if not isinstance(value, str) or value == "" or "\x00" in value:
        raise ValueError(f"input CSV row {row_number} has malformed html")
    return value


def _write_predictions_csv_atomically(rows: Sequence[Mapping[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(OUTPUT_COLUMNS),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def _write_json_atomically(payload: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False)
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def _validate_input_path(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"--input does not exist: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"--input must be a file: {resolved}")
    return resolved


def _validate_repository_path(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"--phishlang-dir does not exist: {path}") from exc
    if not resolved.is_dir():
        raise ValueError(f"--phishlang-dir must be a directory: {resolved}")
    return resolved


def _validate_output_path(path: Path, input_path: Path, repository: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if resolved.suffix.lower() != ".csv":
        raise ValueError("--output must end with .csv")
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"--output must be a file path: {resolved}")
    if resolved == input_path:
        raise ValueError("--output must not overwrite --input")
    if _is_relative_to(resolved, repository):
        raise ValueError("--output must be outside the PhishLang repository")
    if resolved.parent.exists() and not resolved.parent.is_dir():
        raise ValueError("--output parent must be a directory")
    sidecar_path = manifest_path_for(resolved)
    if sidecar_path.exists() and not sidecar_path.is_file():
        raise ValueError("--output sidecar path must be a file")
    return resolved


def _validate_positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _validate_probability(value) -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("phishing probability must be finite and in [0, 1]")
    return probability


def _input_ids(encoded):
    if hasattr(encoded, "input_ids"):
        return encoded.input_ids
    if isinstance(encoded, Mapping) and "input_ids" in encoded:
        return encoded["input_ids"]
    raise TypeError("tokenizer output is missing input_ids")


def _tensor_size(values) -> int:
    if not hasattr(values, "size"):
        raise TypeError("tokenizer input_ids must expose size(0)")
    return int(values.size(0))


def _probability_item(probabilities) -> float:
    value = probabilities[0, 1]
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _chunks(rows: Sequence[dict[str, str]], size: int):
    for index in range(0, len(rows), size):
        yield rows[index : index + size]


def _git(repository: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or "git command failed"
        raise ValueError(message) from exc
    return result.stdout.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
