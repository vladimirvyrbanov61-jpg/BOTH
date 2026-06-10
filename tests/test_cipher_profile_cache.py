"""Regression tests for batched master-key cache identity."""

from __future__ import annotations

import numpy as np
import pytest

from ciphers.registry import get_cipher


@pytest.mark.parametrize("cipher_name", ["simon", "speck"])
def test_batched_key_cache_uses_every_key(cipher_name: str) -> None:
    plaintext = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    first = np.array([1, 2, 3, 4], dtype=np.uint16)
    second_a = np.array([5, 6, 7, 8], dtype=np.uint16)
    second_b = np.array([9, 10, 11, 12], dtype=np.uint16)

    cached_cipher = get_cipher(cipher_name)
    cached_cipher.encrypt(plaintext, np.stack([first, second_a]))
    cached = cached_cipher.encrypt(plaintext, np.stack([first, second_b]))
    fresh = get_cipher(cipher_name).encrypt(
        plaintext,
        np.stack([first, second_b]),
    )

    np.testing.assert_array_equal(cached, fresh)


@pytest.mark.parametrize("cipher_name", ["simon", "speck"])
def test_batched_encryption_creates_one_cache_entry(cipher_name: str) -> None:
    cipher = get_cipher(cipher_name)
    plaintext = np.arange(20, dtype=np.uint16).reshape(10, 2)
    keys = np.arange(40, dtype=np.uint16).reshape(10, 4)

    cipher.encrypt(plaintext, keys, rounds=3)

    assert len(cipher._subkey_cache) == 1
