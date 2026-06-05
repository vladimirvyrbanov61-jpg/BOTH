"""Block representation: words, bytes, and bit vectors (SPECK 32/64)."""

from __future__ import annotations

import struct
from typing import Literal

import numpy as np

WORD_MASK = np.uint16(0xFFFF)
BLOCK_BYTES = 4
BLOCK_BITS = 32
WORD_BITS = 16

Endian = Literal["little", "big"]


def validate_block(words: np.ndarray) -> bool:
    """Return True if *words* is a valid 32-bit block (2 x uint16)."""
    w = np.asarray(words)
    if w.shape != (2,):
        return False
    for x in w.flat:
        xi = int(x)
        if xi < 0 or xi > int(WORD_MASK):
            return False
    return True


def validate_blocks(blocks: np.ndarray) -> np.ndarray:
    """Per-row validation for shape (N, 2). Returns boolean mask of length N."""
    b = np.asarray(blocks)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"blocks must have shape (N, 2), got {b.shape}")
    b = b.astype(np.uint16)
    ok = (b >= 0) & (b <= WORD_MASK)
    return ok.all(axis=1)


def block_to_bytes(
    words: np.ndarray,
    endian: Endian = "little",
) -> bytes:
    w = np.asarray(words, dtype=np.uint16).reshape(2)
    fmt = "<HH" if endian == "little" else ">HH"
    return struct.pack(fmt, int(w[0]), int(w[1]))


def bytes_to_block(
    data: bytes,
    endian: Endian = "little",
) -> np.ndarray:
    if len(data) != BLOCK_BYTES:
        raise ValueError(f"expected {BLOCK_BYTES} bytes, got {len(data)}")
    fmt = "<HH" if endian == "little" else ">HH"
    x, y = struct.unpack(fmt, data)
    return np.array([x, y], dtype=np.uint16)


def bytes_to_blocks(
    data: bytes,
    endian: Endian = "little",
) -> np.ndarray:
    if len(data) % BLOCK_BYTES != 0:
        raise ValueError(f"byte length must be multiple of {BLOCK_BYTES}")
    n = len(data) // BLOCK_BYTES
    out = np.empty((n, 2), dtype=np.uint16)
    for i in range(n):
        out[i] = bytes_to_block(data[i * BLOCK_BYTES : (i + 1) * BLOCK_BYTES], endian)
    return out


def block_to_bits(
    words: np.ndarray,
    n: int = WORD_BITS,
) -> np.ndarray:
    w = np.asarray(words, dtype=np.uint16).reshape(2)
    bits = np.empty(BLOCK_BITS, dtype=np.uint8)
    for i, word in enumerate(w):
        for b in range(n):
            bits[i * n + (n - 1 - b)] = (int(word) >> b) & 1
    return bits


def blocks_to_bits(blocks: np.ndarray, n: int = WORD_BITS) -> np.ndarray:
    b = np.asarray(blocks, dtype=np.uint16)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"blocks must be (N, 2), got {b.shape}")
    out = np.empty((b.shape[0], 2 * n), dtype=np.uint8)
    for i in range(b.shape[0]):
        out[i] = block_to_bits(b[i], n)
    return out
