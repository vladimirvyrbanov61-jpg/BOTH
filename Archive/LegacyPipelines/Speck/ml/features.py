"""ml/features.py — Extended feature engineering for anomaly detection."""

from __future__ import annotations

import numpy as np

from speck3264.encoding import BLOCK_BITS, WORD_BITS, blocks_to_bits
from speck3264.features import block_stats


def bit_transitions(blocks: np.ndarray) -> np.ndarray:
    bits = blocks_to_bits(np.asarray(blocks, dtype=np.uint16))
    diffs = np.diff(bits.astype(np.int8), axis=1)
    rise = (diffs == 1).sum(axis=1).astype(np.float64)
    fall = (diffs == -1).sum(axis=1).astype(np.float64)
    return np.column_stack([rise, fall])


def word_entropy(blocks: np.ndarray) -> np.ndarray:
    b = np.asarray(blocks, dtype=np.uint16)
    hw_l = np.array([bin(int(x)).count("1") for x in b[:, 0]], dtype=np.float64)
    hw_r = np.array([bin(int(x)).count("1") for x in b[:, 1]], dtype=np.float64)
    n_bits = float(WORD_BITS)

    def _ent(hw: np.ndarray) -> np.ndarray:
        p = hw / n_bits
        eps = 1e-10
        p0 = np.clip(1.0 - p, eps, 1.0 - eps)
        p1 = np.clip(p, eps, 1.0 - eps)
        return -(p0 * np.log2(p0) + p1 * np.log2(p1))

    return np.column_stack([_ent(hw_l), _ent(hw_r)])


def build_feature_matrix(
    blocks: np.ndarray,
    *,
    include_bits: bool = True,
    include_stats: bool = True,
    include_transitions: bool = True,
    include_entropy: bool = True,
) -> np.ndarray:
    parts: list[np.ndarray] = []

    if include_stats:
        parts.append(block_stats(np.asarray(blocks, dtype=np.uint16)))
    if include_transitions:
        parts.append(bit_transitions(blocks))
    if include_entropy:
        parts.append(word_entropy(blocks))
    if include_bits:
        parts.append(blocks_to_bits(np.asarray(blocks, dtype=np.uint16)).astype(np.float64))

    if not parts:
        raise ValueError("At least one feature group must be enabled.")
    return np.hstack(parts)


def feature_names(
    *,
    include_bits: bool = True,
    include_stats: bool = True,
    include_transitions: bool = True,
    include_entropy: bool = True,
) -> list[str]:
    names: list[str] = []
    if include_stats:
        names += ["hw_left", "hw_right", "hw_total", "xor_hw", "norm_left", "norm_right"]
    if include_transitions:
        names += ["transitions_01", "transitions_10"]
    if include_entropy:
        names += ["entropy_left", "entropy_right"]
    if include_bits:
        names += [f"bit_{i}" for i in range(BLOCK_BITS)]
    return names


def build_hw_reference(n_samples: int = 100_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 0x10000, size=(n_samples, 2), dtype=np.uint16)
    total_hw = np.array(
        [bin(int(b[0])).count("1") + bin(int(b[1])).count("1") for b in blocks],
        dtype=np.int64,
    )
    hist, _ = np.histogram(total_hw, bins=33, range=(0, 33))
    return hist.astype(np.float64) / hist.sum()


def build_hw_reference_from_blocks(blocks: np.ndarray) -> np.ndarray:
    b = np.asarray(blocks, dtype=np.uint16)
    hw = np.array(
        [bin(int(b[i, 0])).count("1") + bin(int(b[i, 1])).count("1") for i in range(len(b))],
        dtype=np.int64,
    )
    hist, _ = np.histogram(hw, bins=33, range=(0, 33))
    total = hist.sum()
    return hist.astype(np.float64) / max(total, 1)
