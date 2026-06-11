"""Synthetic labeled datasets for SPECK 32/64 (same layout as simon3264.dataset)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

from speck3264.cipher import Speck3264
from speck3264.encoding import blocks_to_bits
from speck3264.faults import (
    corrupt_block,
    identity_or_swap,
    random_blocks,
    wrong_rounds_encrypt,
)

FaultType = Literal["random", "flip", "swap", "xor", "identity", "wrong_rounds"]


@dataclass
class DatasetConfig:
    seed: int = 0
    n_samples: int = 1024
    anomaly_fraction: float = 0.2
    fault_types: list[FaultType] = field(
        default_factory=lambda: ["random", "flip", "wrong_rounds"]
    )
    wrong_rounds_values: list[int] = field(default_factory=lambda: [6, 11])
    flip_bits: int = 3
    feature_bits: bool = True


def sample_keys(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random 64-bit keys, shape (n, 4) uint16, big-endian word order."""
    return rng.integers(0, 0x10000, size=(n, 4), dtype=np.uint16)


def sample_plaintexts(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random plaintext blocks, shape (n, 2) uint16."""
    return rng.integers(0, 0x10000, size=(n, 2), dtype=np.uint16)


def _apply_fault(
    fault: FaultType,
    cipher: Speck3264,
    pt: np.ndarray,
    key: np.ndarray,
    rng: np.random.Generator,
    config: DatasetConfig,
) -> tuple[np.ndarray, str]:
    if fault == "random":
        return random_blocks(pt.shape[0], rng), fault
    if fault == "flip":
        ct = cipher.encrypt(pt, key)
        return corrupt_block(ct, flip_bits=config.flip_bits, rng=rng), fault
    if fault == "swap":
        ct = cipher.encrypt(pt, key)
        return corrupt_block(ct, swap_halves=True), fault
    if fault == "xor":
        ct = cipher.encrypt(pt, key)
        mask = rng.integers(1, 0x10000, size=(1, 2), dtype=np.uint16)
        return corrupt_block(ct, xor_mask=mask), fault
    if fault == "identity":
        return identity_or_swap(pt, mode="identity"), fault
    if fault == "wrong_rounds":
        r = int(rng.choice(config.wrong_rounds_values))
        return wrong_rounds_encrypt(cipher, pt, key, r), f"wrong_rounds_{r}"
    raise ValueError(f"unknown fault type: {fault}")


def labeled_batch(
    n: int,
    cipher: Speck3264,
    rng: np.random.Generator,
    config: DatasetConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    n_anom = int(round(n * config.anomaly_fraction))
    n_norm = n - n_anom
    blocks_list: list[np.ndarray] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []

    if n_norm > 0:
        keys = sample_keys(n_norm, rng)
        pts = sample_plaintexts(n_norm, rng)
        for i in range(n_norm):
            ct = cipher.encrypt(pts[i : i + 1], keys[i])
            blocks_list.append(ct[0])
            labels.append(0)
            meta.append({"fault": "normal", "key_index": i})

    if n_anom > 0:
        keys = sample_keys(n_anom, rng)
        pts = sample_plaintexts(n_anom, rng)
        faults = config.fault_types
        for i in range(n_anom):
            fault = faults[i % len(faults)]
            blk, fault_name = _apply_fault(fault, cipher, pts[i : i + 1], keys[i], rng, config)
            blocks_list.append(blk[0])
            labels.append(1)
            meta.append({"fault": fault_name, "key_index": i})

    blocks = np.stack(blocks_list, axis=0)
    y = np.array(labels, dtype=np.int8)
    rng.shuffle(indices := np.arange(n))
    return blocks[indices], y[indices], [meta[i] for i in indices]


def generate_labeled_dataset(
    config: Optional[DatasetConfig] = None,
) -> dict[str, Any]:
    cfg = config or DatasetConfig()
    rng = np.random.default_rng(cfg.seed)
    cipher = Speck3264()
    blocks, y, meta = labeled_batch(cfg.n_samples, cipher, rng, cfg)

    out: dict[str, Any] = {
        "blocks": blocks,
        "labels": y,
        "meta": meta,
        "config": cfg,
    }
    if cfg.feature_bits:
        out["bits"] = blocks_to_bits(blocks)
    return out


def stratified_split(
    labels: np.ndarray,
    meta: list[dict[str, Any]],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.asarray(labels)
    faults = np.array([m.get("fault", "unknown") for m in meta])
    strata = np.char.add(y.astype(str), np.array(["_"] * len(y), dtype="U1"))
    strata = np.char.add(strata, faults.astype(str))

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val :].tolist())

    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx, dtype=np.int64),
        np.array(test_idx, dtype=np.int64),
    )
