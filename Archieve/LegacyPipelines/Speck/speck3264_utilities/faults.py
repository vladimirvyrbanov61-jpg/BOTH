"""Synthetic fault injection for SPECK 32/64 anomaly labels."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from speck3264.cipher import Speck3264
from speck3264.encoding import WORD_MASK

FaultMode = Literal["random", "flip", "swap", "xor", "identity", "wrong_rounds"]


def random_blocks(
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform random 32-bit blocks, shape (n, 2) uint16."""
    return rng.integers(0, int(WORD_MASK) + 1, size=(n, 2), dtype=np.uint16)


def corrupt_block(
    blocks: np.ndarray,
    *,
    flip_bits: Optional[int] = None,
    swap_halves: bool = False,
    xor_mask: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    out = np.asarray(blocks, dtype=np.uint16).copy()
    n_rows = out.shape[0]

    if swap_halves:
        out[:, 0], out[:, 1] = out[:, 1].copy(), out[:, 0].copy()

    if xor_mask is not None:
        m = np.asarray(xor_mask, dtype=np.uint16)
        if m.shape == (2,):
            m = m[np.newaxis, :]
        out ^= m

    if flip_bits is not None and flip_bits > 0:
        gen = rng if rng is not None else np.random.default_rng()
        for _ in range(flip_bits):
            row = gen.integers(0, n_rows)
            word = gen.integers(0, 2)
            bit = gen.integers(0, 16)
            out[row, word] ^= np.uint16(1 << bit)

    return out


def identity_or_swap(
    blocks: np.ndarray,
    mode: Literal["identity", "swap"] = "identity",
) -> np.ndarray:
    b = np.asarray(blocks, dtype=np.uint16).copy()
    if mode == "swap":
        b[:, 0], b[:, 1] = b[:, 1].copy(), b[:, 0].copy()
    return b


def wrong_rounds_encrypt(
    cipher: Speck3264,
    plaintext: np.ndarray,
    key: np.ndarray,
    num_rounds: int,
) -> np.ndarray:
    return cipher.encrypt_rounds(plaintext, key, num_rounds)
