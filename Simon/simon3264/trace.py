"""Round traces and subkey feature export."""

from __future__ import annotations

from typing import Optional

import numpy as np

from simon import encrypt_blocks_trace
from simon3264.cipher import N_BITS, ROUNDS, Simon3264
from simon3264.encoding import blocks_to_bits


def encrypt_trace(
    cipher: Simon3264,
    plaintext: np.ndarray,
    key: np.ndarray,
    *,
    rounds: Optional[int] = None,
) -> np.ndarray:
    """State after each round including plaintext.

    Returns shape (rounds + 1, N, 2).
    """
    r = ROUNDS if rounds is None else rounds
    pt = cipher._coerce_blocks(plaintext)
    sk = cipher.get_subkeys(key, rounds=r)
    return encrypt_blocks_trace(pt, sk, N_BITS, r)


def encrypt_stop_at_round(
    cipher: Simon3264,
    plaintext: np.ndarray,
    key: np.ndarray,
    round_index: int,
) -> np.ndarray:
    """Ciphertext after exactly *round_index* encryption rounds."""
    trace = encrypt_trace(cipher, plaintext, key, rounds=round_index)
    return trace[round_index].copy()


def subkey_bits(
    subkeys: np.ndarray,
    n: int = N_BITS,
) -> np.ndarray:
    """Flatten round keys to a bit vector (MSB-first per word)."""
    sk = np.asarray(subkeys)
    if sk.ndim == 2:
        sk = sk.reshape(-1)
    words = sk.astype(np.uint16).ravel()
    bits_list = []
    for w in words:
        for b in range(n - 1, -1, -1):
            bits_list.append((int(w) >> b) & 1)
    return np.array(bits_list, dtype=np.uint8)


def subkey_summary_stats(subkeys: np.ndarray) -> dict[str, float]:
    """Scalar features from a subkey schedule (tabular anomaly cues)."""
    sk = np.asarray(subkeys, dtype=np.uint16).ravel()
    hw = np.array([bin(int(w)).count("1") for w in sk], dtype=np.float64)
    return {
        "subkey_hw_mean": float(hw.mean()),
        "subkey_hw_std": float(hw.std()) if len(hw) > 1 else 0.0,
        "subkey_hw_min": float(hw.min()),
        "subkey_hw_max": float(hw.max()),
    }
