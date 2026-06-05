"""NPZ cache for blind distinguisher tensors (X, y only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ciphers.registry import CipherName


def config_fingerprint(
    *,
    cipher: CipherName,
    rounds: int,
    n_samples: int,
    delta: tuple[int, int],
    seed: int,
) -> str:
    spec = {
        "cipher": cipher,
        "rounds": rounds,
        "n": n_samples,
        "delta": list(delta),
        "seed": seed,
        "blind": True,
        "feature": "concat_bits_64",
    }
    return hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def cache_path(
    data_dir: Path,
    cipher: CipherName,
    rounds: int,
    tag: str,
) -> Path:
    return data_dir / f"{cipher}_r{rounds}_{tag}.npz"


def save_blind_npz(path: Path, X: np.ndarray, y: np.ndarray, rounds: int) -> None:
    """Persist only features and labels — no plaintexts or keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X.astype(np.float32), y=y.astype(np.int8), rounds=np.array([rounds]))


def load_blind_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        forbidden = {"plaintext", "key", "keys", "P", "K", "pt", "recovery"}
        for name in data.files:
            if name.lower() in forbidden or name.startswith("P_") or name.startswith("K_"):
                raise ValueError(f"cache leak: unexpected array {name!r} in {path}")
        return {k: data[k] for k in data.files}
