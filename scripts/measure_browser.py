from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
    "chromium-browser",
]


def parse_benchmark_json(html: str) -> dict:
    """Extract the strict JSON inside <pre id="benchmark-results"> from dumped DOM."""
    marker = 'id="benchmark-results"'
    index = html.find(marker)
    if index < 0:
        raise ValueError("benchmark-results marker not found in dumped DOM")
    start = html.index(">", index) + 1
    end = html.index("</pre>", start)
    raw = html[start:end].strip()
    if not raw:
        raise ValueError("benchmark-results element is empty")
    return json.loads(raw)


def find_chrome() -> str:
    """Return the path to a usable Chrome/Chromium binary.

    Checks CHROME_BIN env var first, then known paths, then PATH.
    Raises RuntimeError if no binary is found.
    """
    env_bin = os.environ.get("CHROME_BIN")
    if env_bin and Path(env_bin).is_file():
        return env_bin

    for candidate in _CHROME_CANDIDATES[:1]:
        if Path(candidate).is_file():
            return candidate

    for candidate in _CHROME_CANDIDATES[1:]:
        found = shutil.which(candidate)
        if found:
            return found

    raise RuntimeError(
        f"Chrome binary not found — checked CHROME_BIN, {', '.join(_CHROME_CANDIDATES)}"
    )


def run(chrome_bin: str, model_path: str | Path, output_dir: str | Path, *, port: int = 0) -> dict:
    """Serve web/ and the model artifact, launch headless Chrome, collect benchmark JSON.

    Returns the browser-metrics dict and writes ``browser-metrics.json`` to *output_dir*.
    """
    model_path = Path(model_path).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute model file size
    model_bytes = model_path.stat().st_size

    # Build a temporary directory containing web/ files + model.json
    tmp_dir = Path(tempfile.mkdtemp(prefix="phishme-browser-"))
    try:
        # Copy web/ files
        for entry in _WEB_DIR.iterdir():
            dest = tmp_dir / entry.name
            if entry.is_file():
                shutil.copy2(entry, dest)
            elif entry.is_dir():
                shutil.copytree(entry, dest, dirs_exist_ok=True)

        # Copy model artifact as model.json
        dest_model = tmp_dir / "model.json"
        shutil.copy2(model_path, dest_model)

        # Compute extension bytes (all files in web/ + model file)
        web_bytes = _dir_bytes(_WEB_DIR)
        extension_bytes = web_bytes + model_bytes

        # Find free port
        actual_port = port if port != 0 else _find_free_port()

        # Start HTTP server in a background thread
        server = HTTPServer(
            ("127.0.0.1", actual_port),
            _handler_for(tmp_dir),
        )
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{actual_port}/benchmark.html?model=model.json"
            result = subprocess.run(
                [
                    chrome_bin,
                    "--headless=new",
                    "--dump-dom",
                    "--virtual-time-budget=30000",
                    "--disable-gpu",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=90,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()[:500]
                raise RuntimeError(
                    f"Chrome exited with code {result.returncode}: {stderr}"
                )

            html = result.stdout
            metrics = parse_benchmark_json(html)

            avg_latency_ms = float(metrics["avg_latency_ms"])
            p95_latency_ms = float(metrics["p95_latency_ms"])
            memory_bytes = metrics.get("memory_bytes")
            pages = int(metrics["pages"])

            payload = {
                "schema": "phishme-browser-metrics-v1",
                "model_bytes": int(metrics["model_bytes"]),
                "extension_bytes": extension_bytes,
                "avg_latency_ms": avg_latency_ms,
                "p95_latency_ms": p95_latency_ms,
                "memory_bytes": memory_bytes,
                "pages": pages,
                "chrome": chrome_bin,
                "gates": {
                    "payload_le_25mb": int(metrics["model_bytes"]) <= 25 * 1024 * 1024,
                    "p95_le_250ms": p95_latency_ms <= 250,
                },
            }

            metrics_path = output_dir / "browser-metrics.json"
            _write_json_atomically(payload, metrics_path)

            return payload

        finally:
            server.shutdown()
            server_thread.join(timeout=5)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _handler_for(directory: Path):
    """Return a partial SimpleHTTPRequestHandler configured for *directory*."""
    from functools import partial

    return partial(SimpleHTTPRequestHandler, directory=str(directory))


def _dir_bytes(directory: Path) -> int:
    """Sum file sizes recursively under *directory*."""
    total = 0
    for entry in directory.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def _write_json_atomically(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False)
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
