"""Statistical and sequence features for anomaly detection."""

from __future__ import annotations

import numpy as np

from simon3264.encoding import WORD_MASK, blocks_to_bits


def _hamming_weight16(words: np.ndarray) -> np.ndarray:
    """Per-word HW for uint16 array of arbitrary shape ending in word axis."""
    flat = np.asarray(words, dtype=np.uint16).ravel()
    return np.array([bin(int(w)).count("1") for w in flat], dtype=np.float64)


def block_stats(blocks: np.ndarray) -> np.ndarray:
    """Per-block tabular features, shape (N, F).

    Columns: hw_left, hw_right, hw_total, xor_halves_hw, left, right (normalized).
    """
    b = np.asarray(blocks, dtype=np.uint16)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"blocks must be (N, 2), got {b.shape}")
    n = b.shape[0]
    hw_l = np.array([bin(int(x)).count("1") for x in b[:, 0]], dtype=np.float64)
    hw_r = np.array([bin(int(x)).count("1") for x in b[:, 1]], dtype=np.float64)
    xor_hw = np.array(
        [bin(int(b[i, 0] ^ b[i, 1])).count("1") for i in range(n)],
        dtype=np.float64,
    )
    norm = float(WORD_MASK)
    return np.column_stack(
        [
            hw_l,
            hw_r,
            hw_l + hw_r,
            xor_hw,
            b[:, 0].astype(np.float64) / norm,
            b[:, 1].astype(np.float64) / norm,
        ]
    )


def _entropy_bits(bits: np.ndarray) -> float:
    p1 = bits.mean()
    if p1 <= 0.0 or p1 >= 1.0:
        return 0.0
    p0 = 1.0 - p1
    return float(-(p0 * np.log2(p0) + p1 * np.log2(p1)))


def blocks_to_feature_matrix(
    blocks: np.ndarray,
    *,
    include_bits: bool = False,
) -> np.ndarray:
    """Combine tabular stats and optional 32-bit expansion."""
    stats = block_stats(blocks)
    if not include_bits:
        return stats
    bits = blocks_to_bits(blocks).astype(np.float64)
    return np.hstack([stats, bits])


def batch_hw_stats(blocks: np.ndarray) -> dict[str, float]:
    """Session-level HW mean/variance over a batch of blocks."""
    b = np.asarray(blocks, dtype=np.uint16)
    hw = _hamming_weight16(b)
    per_block = hw.reshape(-1, 2).sum(axis=1)
    return {
        "batch_hw_mean": float(per_block.mean()),
        "batch_hw_var": float(per_block.var()) if len(per_block) > 1 else 0.0,
    }


def chi_square_hw_vs_reference(
    blocks: np.ndarray,
    reference_hw_hist: np.ndarray,
) -> float:
    """Chi-square distance of HW histogram (0..32) vs reference counts."""
    b = np.asarray(blocks, dtype=np.uint16)
    total_hw = np.array(
        [
            bin(int(b[i, 0])).count("1") + bin(int(b[i, 1])).count("1")
            for i in range(b.shape[0])
        ],
        dtype=np.int64,
    )
    obs, _ = np.histogram(total_hw, bins=33, range=(0, 33))
    ref = np.asarray(reference_hw_hist, dtype=np.float64).ravel()
    if ref.shape != (33,):
        raise ValueError("reference_hw_hist must have length 33")
    ref = ref + 1e-6
    obs = obs.astype(np.float64) + 1e-6
    obs /= obs.sum()
    ref /= ref.sum()
    return float(np.sum((obs - ref) ** 2 / ref))


def sliding_window_xor_features(
    blocks: np.ndarray,
) -> np.ndarray:
    """Consecutive-block XOR features, shape (N-1, 32) as bits of C_i xor C_{i+1}."""
    b = np.asarray(blocks, dtype=np.uint16)
    if b.shape[0] < 2:
        return np.empty((0, 32), dtype=np.uint8)
    xored = b[:-1].copy()
    xored ^= b[1:]
    return blocks_to_bits(xored.reshape(-1, 2)).reshape(len(xored), -1)
