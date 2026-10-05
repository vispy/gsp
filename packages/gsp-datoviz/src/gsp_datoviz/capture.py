"""PNG encoding for caller-owned native RGBA capture."""

from __future__ import annotations

from struct import pack
from zlib import compress, crc32


def _encode_rgba8_png(width: int, height: int, rgba: bytes) -> bytes:
    """Encode tightly packed RGBA8 screenshot pixels without filesystem round-tripping."""
    if width <= 0 or height <= 0:
        raise ValueError("PNG dimensions must be positive")
    expected_size = width * height * 4
    if len(rgba) != expected_size:
        raise ValueError(f"RGBA8 payload must contain exactly {expected_size} bytes")

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = crc32(kind)
        checksum = crc32(payload, checksum) & 0xFFFFFFFF
        return pack(">I", len(payload)) + kind + payload + pack(">I", checksum)

    stride = width * 4
    scanlines = b"".join(b"\x00" + rgba[row * stride : (row + 1) * stride] for row in range(height))
    header = pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", compress(scanlines))
        + chunk(b"IEND", b"")
    )
