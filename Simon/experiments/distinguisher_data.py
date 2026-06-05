"""Generate neural-distinguisher datasets: r-round Simon pairs vs random pairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from experiments.config import DistinguisherConfig
from ml.features import build_feature_matrix
from simon3264.cipher import Simon3264
from simon3264.dataset import sample_keys, sample_plaintexts
from simon3264.encoding import blocks_to_bits
from simon3264.faults import random_blocks


def _config_hash(cfg: DistinguisherConfig, rounds: int) -> str:
    spec = {
        "rounds": rounds,
        "n": cfg.n_samples_per_round,
        "delta": cfg.input_delta,
        "mode": cfg.feature_mode,
        "seed": cfg.seed,
    }
    return hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def apply_delta(
    plaintexts: np.ndarray,
    delta: tuple[int, int],
) -> np.ndarray:
    """Word-wise XOR of plaintext with delta (16-bit words)."""
    d = np.array(delta, dtype=np.uint16)
    out = plaintexts.copy()
    out ^= d
    return out


def pair_to_features(
    c0: np.ndarray,
    c1: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Build feature row from ciphertext pair (N,2) each -> (N, F)."""
    if mode == "xor_bits":
        xored = c0 ^ c1
        return blocks_to_bits(xored).astype(np.float64)
    if mode == "concat_bits":
        return np.hstack(
            [
                blocks_to_bits(c0).astype(np.float64),
                blocks_to_bits(c1).astype(np.float64),
            ]
        )
    if mode == "ml_features":
        xored = (c0 ^ c1).astype(np.uint16)
        return build_feature_matrix(xored)
    raise ValueError(f"unknown feature_mode: {mode!r}")


def generate_distinguisher_dataset(
    rounds: int,
    cfg: DistinguisherConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return X (N, F) float64 and y (N,) int8 — half real Simon pairs, half random."""
    n = cfg.n_samples_per_round
    n_each = n // 2
    cipher = Simon3264()
    delta = (int(cfg.input_delta[0]), int(cfg.input_delta[1]))

    keys = sample_keys(n_each, rng)
    pts = sample_plaintexts(n_each, rng)
    pts2 = apply_delta(pts, delta)

    real0 = np.empty((n_each, 2), dtype=np.uint16)
    real1 = np.empty((n_each, 2), dtype=np.uint16)
    for i in range(n_each):
        ct = cipher.encrypt_rounds(pts[i : i + 1], keys[i], rounds)
        ct2 = cipher.encrypt_rounds(pts2[i : i + 1], keys[i], rounds)
        real0[i] = ct[0]
        real1[i] = ct2[0]

    rand0 = random_blocks(n_each, rng)
    rand1 = random_blocks(n_each, rng)

    X_real = pair_to_features(real0, real1, cfg.feature_mode)
    X_rand = pair_to_features(rand0, rand1, cfg.feature_mode)
    X = np.vstack([X_real, X_rand])
    y = np.array([1] * n_each + [0] * n_each, dtype=np.int8)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def stratified_split_indices(
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for label in (0, 1):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(n * train_ratio)
        n_va = int(n * val_ratio)
        train_idx.extend(idx[:n_tr].tolist())
        val_idx.extend(idx[n_tr : n_tr + n_va].tolist())
        test_idx.extend(idx[n_tr + n_va :].tolist())
    return {
        "train": np.array(train_idx, dtype=np.int64),
        "val": np.array(val_idx, dtype=np.int64),
        "test": np.array(test_idx, dtype=np.int64),
    }


def generate_or_load_distinguisher(
    rounds: int,
    cfg: DistinguisherConfig,
    *,
    cache_dir: Optional[Path] = None,
    force_regen: bool = False,
) -> dict[str, Any]:
    if cache_dir is None:
        cache_dir = Path(cfg.data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = _config_hash(cfg, rounds)
    cache_path = cache_dir / f"distinguisher_r{rounds}_{tag}.npz"

    if not force_regen and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            out = {k: data[k] for k in data.files}
        out["rounds"] = rounds
        out["splits"] = stratified_split_indices(
            out["y"], seed=cfg.seed + rounds
        )
        return out

    rng = np.random.default_rng(cfg.seed + rounds)
    X, y = generate_distinguisher_dataset(rounds, cfg, rng)
    splits = stratified_split_indices(y, seed=cfg.seed + rounds)
    np.savez_compressed(cache_path, X=X, y=y, rounds=np.array([rounds]))
    return {"X": X, "y": y, "rounds": rounds, "splits": splits, "cache_path": cache_path}
