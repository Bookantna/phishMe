from __future__ import annotations

import argparse
import binascii
import struct
import zlib
from pathlib import Path

SIZES = (16, 32, 48, 128)
NAVY = (25, 55, 109, 255)
WHITE = (255, 255, 255, 255)
TRANSPARENT = (0, 0, 0, 0)


def png_chunk(kind: bytes, data: bytes) -> bytes:
    payload = kind + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", binascii.crc32(payload) & 0xFFFFFFFF)


def make_icon(size: int) -> bytes:
    pixels = []
    center = (size - 1) / 2
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            nx = abs(x - center) / max(center, 1)
            ny = y / max(size - 1, 1)
            shield = ny < 0.08 or (ny <= 0.70 and nx <= 0.78) or (ny > 0.70 and nx <= 0.78 * (1 - (ny - 0.70) / 0.30))
            color = NAVY if shield else TRANSPARENT

            # A compact white P: vertical stem, bowl cap, and bowl right edge.
            stem = 0.31 <= x / size <= 0.43 and 0.23 <= y / size <= 0.72
            top = 0.31 <= x / size <= 0.65 and 0.23 <= y / size <= 0.34
            middle = 0.31 <= x / size <= 0.62 and 0.45 <= y / size <= 0.55
            right = 0.55 <= x / size <= 0.66 and 0.28 <= y / size <= 0.50
            if shield and (stem or top or middle or right):
                color = WHITE
            row.extend(color)
        pixels.append(bytes(row))

    raw = b"".join(pixels)
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return signature + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", zlib.compress(raw, 9)) + png_chunk(b"IEND", b"")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic phishMe PNG icons")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for size in SIZES:
        (args.output / f"icon{size}.png").write_bytes(make_icon(size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
