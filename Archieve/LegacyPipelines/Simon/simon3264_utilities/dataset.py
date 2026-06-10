"""Synthetic labeled datasets and splits for anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

from simon3264.cipher import Simon3264
from simon3264.encoding import blocks_to_bits
from simon3264.faults import (
    corrupt_block,
    identity_or_swap,
    random_blocks,
    wrong_rounds_encrypt,
    wrong_z_encrypt,
)

FaultType = Literal[
    "random",
    "flip",
    "swap",
    "xor",
    "identity",
    "wrong_rounds",
    "wrong_z",
]


@dataclass
class DatasetConfig:
    seed: int = 0
    n_samples: int = 1024
    """Process this many samples per chunk to limit peak RAM and Python object churn."""
    chunk_size: int = 25_000
    anomaly_fraction: float = 0.2
    fault_types: list[FaultType] = field(
        default_factory=lambda: ["random", "flip", "wrong_rounds"]
    )
    wrong_rounds_values: list[int] = field(default_factory=lambda: [8, 16])
    wrong_z_index: int = 1
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
    cipher: Simon3264,
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
    if fault == "wrong_z":
        return wrong_z_encrypt(cipher, pt, key, config.wrong_z_index), fault
    raise ValueError(f"unknown fault type: {fault}")


def labeled_batch(
    n: int,
    cipher: Simon3264,
    rng: np.random.Generator,
    config: DatasetConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Single batch: blocks, labels, meta, plaintexts, keys (for ML oracle features)."""
    n_anom = int(round(n * config.anomaly_fraction))
    n_norm = n - n_anom
    blocks_list: list[np.ndarray] = []
    pts_list: list[np.ndarray] = []
    keys_list: list[np.ndarray] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []

    if n_norm > 0:
        keys = sample_keys(n_norm, rng)
        pts = sample_plaintexts(n_norm, rng)
        # Vectorised full-round encrypt (simon.encrypt_blocks supports batched keys).
        cts = cipher.encrypt(pts, keys)
        for i in range(n_norm):
            blocks_list.append(cts[i])
            pts_list.append(pts[i])
            keys_list.append(keys[i])
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
            pts_list.append(pts[i])
            keys_list.append(keys[i])
            labels.append(1)
            meta.append({"fault": fault_name, "key_index": i})

    blocks = np.stack(blocks_list, axis=0)
    plain = np.stack(pts_list, axis=0)
    keyw = np.stack(keys_list, axis=0)
    y = np.array(labels, dtype=np.int8)
    rng.shuffle(indices := np.arange(n))
    return (
        blocks[indices],
        y[indices],
        [meta[i] for i in indices],
        plain[indices],
        keyw[indices],
    )


def generate_labeled_dataset(
    config: Optional[DatasetConfig] = None,
) -> dict[str, Any]:
    """Build a reproducible labeled dataset dict with features and metadata."""
    cfg = config or DatasetConfig()
    rng = np.random.default_rng(cfg.seed)
    cipher = Simon3264()
    n = cfg.n_samples
    chunk = max(1, min(cfg.chunk_size, n))

    blocks = np.empty((n, 2), dtype=np.uint16)
    labels = np.empty(n, dtype=np.int8)
    fault_names = np.empty(n, dtype=object)
    plaintexts = np.empty((n, 2), dtype=np.uint16)
    keys = np.empty((n, 4), dtype=np.uint16)

    offset = 0
    while offset < n:
        size = min(chunk, n - offset)
        b, y, meta, pt, kw = labeled_batch(size, cipher, rng, cfg)
        blocks[offset : offset + size] = b
        labels[offset : offset + size] = y
        plaintexts[offset : offset + size] = pt
        keys[offset : offset + size] = kw
        fault_names[offset : offset + size] = [m.get("fault", "unknown") for m in meta]
        offset += size

    perm = rng.permutation(n)
    blocks = blocks[perm]
    labels = labels[perm]
    plaintexts = plaintexts[perm]
    keys = keys[perm]
    fault_names = fault_names[perm]
    meta = [{"fault": str(fault_names[i])} for i in range(n)]

    out: dict[str, Any] = {
        "blocks": blocks,
        "labels": labels,
        "meta": meta,
        "plaintexts": plaintexts,
        "keys": keys,
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
    """Indices for train / val / test, stratified by label and fault string."""
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
