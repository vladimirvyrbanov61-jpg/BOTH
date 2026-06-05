"""Load/save blocks and real-data feature pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np

from speck3264.cipher import Speck3264
from speck3264.encoding import (
    BLOCK_BYTES,
    bytes_to_block,
    bytes_to_blocks,
    validate_blocks,
    blocks_to_bits,
)
from speck3264.features import blocks_to_feature_matrix

FileFormat = Literal["npz", "bin", "hex"]


def save_blocks_npz(
    path: str | Path,
    blocks: np.ndarray,
    *,
    labels: Optional[np.ndarray] = None,
    keys: Optional[np.ndarray] = None,
) -> None:
    payload: dict[str, np.ndarray] = {"blocks": np.asarray(blocks, dtype=np.uint16)}
    if labels is not None:
        payload["labels"] = np.asarray(labels)
    if keys is not None:
        payload["keys"] = np.asarray(keys, dtype=np.uint16)
    np.savez_compressed(path, **payload)


def load_blocks_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def blocks_from_file(
    path: str | Path,
    format: FileFormat = "npz",
    *,
    endian: Literal["little", "big"] = "little",
) -> np.ndarray:
    path = Path(path)
    if format == "npz":
        data = load_blocks_npz(path)
        if "blocks" not in data:
            raise ValueError("npz must contain 'blocks'")
        return np.asarray(data["blocks"], dtype=np.uint16)

    if format == "bin":
        raw = path.read_bytes()
        return bytes_to_blocks(raw, endian=endian)

    if format == "hex":
        rows = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) == 2 and all(len(p) <= 4 for p in parts):
                rows.append(
                    np.array([int(parts[0], 16), int(parts[1], 16)], dtype=np.uint16)
                )
                continue
            compact = "".join(parts)
            if len(compact) == 8:
                raw = bytes.fromhex(compact)
                rows.append(bytes_to_block(raw, endian=endian))
                continue
            raw = bytes(int(p, 16) for p in parts)
            if len(raw) != BLOCK_BYTES:
                raise ValueError(
                    f"hex line must be two words, 8 hex chars, or {BLOCK_BYTES} byte tokens"
                )
            rows.append(bytes_to_block(raw, endian=endian))
        if not rows:
            return np.empty((0, 2), dtype=np.uint16)
        return np.stack(rows, axis=0)

    raise ValueError(f"unknown format: {format}")


def decrypt_check(
    cipher: Speck3264,
    ciphertext: np.ndarray,
    key: np.ndarray,
    plaintext_expected: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray | bool | float]:
    ct = cipher._coerce_blocks(ciphertext)
    pt_dec = cipher.decrypt(ct, key)
    result: dict[str, np.ndarray | bool | float] = {"plaintext": pt_dec}
    if plaintext_expected is not None:
        exp = np.asarray(plaintext_expected, dtype=np.uint16)
        if exp.ndim == 1:
            exp = exp[np.newaxis, :]
        match = np.all(pt_dec == exp, axis=1)
        result["match"] = bool(np.all(match))
        diff = pt_dec.astype(np.int32) - exp.astype(np.int32)
        result["residual_l2"] = float(np.sqrt(np.sum(diff**2)))
    return result


def blocks_to_feature_file(
    path: str | Path,
    *,
    format: FileFormat = "npz",
    output_npz: Optional[str | Path] = None,
    include_bits: bool = True,
) -> np.ndarray:
    blocks = blocks_from_file(path, format=format)
    valid = validate_blocks(blocks)
    if not np.all(valid):
        raise ValueError(f"{np.sum(~valid)} invalid blocks in file")
    feats = blocks_to_feature_matrix(blocks, include_bits=include_bits)
    if output_npz is not None:
        np.savez_compressed(
            output_npz,
            blocks=blocks,
            features=feats,
            bits=blocks_to_bits(blocks) if include_bits else np.array([]),
        )
    return feats
