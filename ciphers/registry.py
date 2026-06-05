"""Resolve Simon32/64 and Speck32/64 profile ciphers by name."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Protocol, Union

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SIMON_ROOT = _REPO_ROOT / "Simon"
_SPECK_ROOT = _REPO_ROOT / "Speck"

CipherName = Literal["simon", "speck"]


class BlockCipherProfile(Protocol):
    """Minimal interface for thesis data generation."""

    rounds: int
    dtype: np.dtype

    def encrypt(
        self,
        plaintext: np.ndarray,
        key: np.ndarray,
        *,
        rounds: int | None = None,
    ) -> np.ndarray: ...


def _ensure_path(root: Path) -> None:
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def get_cipher(name: CipherName) -> Union["Simon3264", "Speck3264"]:
    if name == "simon":
        _ensure_path(_SIMON_ROOT)
        from simon3264.cipher import Simon3264

        return Simon3264()
    if name == "speck":
        _ensure_path(_SPECK_ROOT)
        from speck3264.cipher import Speck3264

        return Speck3264()
    raise ValueError(f"unknown cipher {name!r}; use 'simon' or 'speck'")


def max_rounds(name: CipherName) -> int:
    return int(get_cipher(name).rounds)
