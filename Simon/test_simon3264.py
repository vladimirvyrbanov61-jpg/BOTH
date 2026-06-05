"""
Tests for simon3264 anomaly-detection utilities.

Run: pytest test_simon3264.py -v
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from simon import Simon
from simon3264 import (
    DatasetConfig,
    Simon3264,
    block_to_bits,
    block_to_bytes,
    blocks_from_file,
    blocks_to_bits,
    bytes_to_block,
    bytes_to_blocks,
    corrupt_block,
    encrypt_trace,
    generate_labeled_dataset,
    labeled_batch,
    load_blocks_npz,
    random_blocks,
    save_blocks_npz,
    stratified_split,
    subkey_summary_stats,
    validate_block,
    validate_blocks,
)
from simon3264.io import blocks_to_feature_file
from simon3264.features import (
    block_stats,
    chi_square_hw_vs_reference,
    sliding_window_xor_features,
)
from simon3264.io import decrypt_check
from simon3264.trace import encrypt_stop_at_round, subkey_bits


OFFICIAL_KEY = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)
OFFICIAL_PT = np.array([0x6565, 0x6877], dtype=np.uint16)
OFFICIAL_CT = np.array([0xC69B, 0xE9BB], dtype=np.uint16)


class TestEncoding:
    def test_little_endian_roundtrip(self):
        b = OFFICIAL_PT
        raw = block_to_bytes(b, "little")
        assert raw == bytes([0x65, 0x65, 0x77, 0x68])
        np.testing.assert_array_equal(bytes_to_block(raw, "little"), b)

    def test_big_endian_roundtrip(self):
        raw = block_to_bytes(OFFICIAL_PT, "big")
        np.testing.assert_array_equal(bytes_to_block(raw, "big"), OFFICIAL_PT)

    def test_bytes_to_blocks(self):
        raw = block_to_bytes(OFFICIAL_PT) + block_to_bytes(OFFICIAL_CT)
        blk = bytes_to_blocks(raw)
        assert blk.shape == (2, 2)

    def test_block_to_bits_length(self):
        bits = block_to_bits(OFFICIAL_PT)
        assert bits.shape == (32,)
        assert bits.dtype == np.uint8

    def test_blocks_to_bits_batch(self):
        batch = np.stack([OFFICIAL_PT, OFFICIAL_CT])
        bits = blocks_to_bits(batch)
        assert bits.shape == (2, 32)

    def test_validate_block(self):
        assert validate_block(OFFICIAL_PT)
        assert not validate_block(np.array([0, 0x10000], dtype=np.uint32))

    def test_validate_blocks_mask(self):
        good = np.array([[0, 1], [0xFFFF, 0]], dtype=np.uint16)
        assert validate_blocks(good).all()


class TestSimon3264Cipher:
    @pytest.fixture
    def cipher(self) -> Simon3264:
        return Simon3264()

    def test_official_vector(self, cipher: Simon3264):
        ct = cipher.encrypt(OFFICIAL_PT, OFFICIAL_KEY)
        np.testing.assert_array_equal(ct[0], OFFICIAL_CT)

    def test_decrypt_roundtrip(self, cipher: Simon3264):
        ct = cipher.encrypt(OFFICIAL_PT, OFFICIAL_KEY)
        pt = cipher.decrypt(ct, OFFICIAL_KEY)
        np.testing.assert_array_equal(pt[0], OFFICIAL_PT)

    def test_subkey_cache(self, cipher: Simon3264):
        sk1 = cipher.get_subkeys(OFFICIAL_KEY)
        sk2 = cipher.get_subkeys(OFFICIAL_KEY)
        assert sk1 is sk2
        assert sk1.shape[-1] == 32

    def test_encrypt_with_subkeys(self, cipher: Simon3264):
        sk = cipher.get_subkeys(OFFICIAL_KEY)
        ct = cipher.encrypt_with_subkeys(OFFICIAL_PT[np.newaxis, :], sk)
        np.testing.assert_array_equal(ct[0], OFFICIAL_CT)

    def test_encrypt_rounds_differs_from_full(self, cipher: Simon3264):
        ct16 = cipher.encrypt_rounds(OFFICIAL_PT, OFFICIAL_KEY, 16)
        ct32 = cipher.encrypt(OFFICIAL_PT, OFFICIAL_KEY)
        assert not np.array_equal(ct16, ct32)

    def test_encrypt_variant_wrong_z(self, cipher: Simon3264):
        ct = cipher.encrypt_variant(OFFICIAL_PT, OFFICIAL_KEY, z_index=1)
        assert ct.shape == (1, 2)

    def test_matches_core_simon(self, cipher: Simon3264):
        core = Simon(n=16, m=4)
        pt = np.random.default_rng(0).integers(0, 0x10000, size=(5, 2), dtype=np.uint16)
        key = np.random.default_rng(1).integers(0, 0x10000, size=4, dtype=np.uint16)
        np.testing.assert_array_equal(cipher.encrypt(pt, key), core.encrypt(pt, key))


class TestFaults:
    def test_random_blocks(self):
        rng = np.random.default_rng(0)
        b = random_blocks(10, rng)
        assert b.shape == (10, 2)

    def test_corrupt_swap(self):
        b = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        c = corrupt_block(b, swap_halves=True)
        assert c[0, 0] == 2 and c[0, 1] == 1

    def test_corrupt_flip(self):
        b = np.array([[0, 0]], dtype=np.uint16)
        c = corrupt_block(b, flip_bits=1, rng=np.random.default_rng(0))
        assert int(c[0, 0]) != 0 or int(c[0, 1]) != 0


class TestDataset:
    def test_labeled_batch_balance(self):
        cfg = DatasetConfig(seed=42, n_samples=100, anomaly_fraction=0.3)
        cipher = Simon3264()
        rng = np.random.default_rng(cfg.seed)
        blocks, y, meta, _pt, _kw = labeled_batch(100, cipher, rng, cfg)
        assert blocks.shape == (100, 2)
        assert y.shape == (100,)
        assert int(y.sum()) == pytest.approx(30, abs=2)
        assert len(meta) == 100

    def test_generate_labeled_dataset_bits(self):
        cfg = DatasetConfig(seed=0, n_samples=50, anomaly_fraction=0.2)
        ds = generate_labeled_dataset(cfg)
        assert ds["bits"].shape == (50, 32)

    def test_stratified_split_covers_all(self):
        cfg = DatasetConfig(seed=1, n_samples=200)
        ds = generate_labeled_dataset(cfg)
        tr, va, te = stratified_split(ds["labels"], ds["meta"], seed=0)
        union = np.unique(np.concatenate([tr, va, te]))
        assert len(union) == 200

    def test_golden_batch_reproducible(self):
        cfg = DatasetConfig(seed=12345, n_samples=64, anomaly_fraction=0.25)
        ds1 = generate_labeled_dataset(cfg)
        ds2 = generate_labeled_dataset(cfg)
        np.testing.assert_array_equal(ds1["blocks"], ds2["blocks"])
        np.testing.assert_array_equal(ds1["labels"], ds2["labels"])
        h = hashlib.sha256(ds1["blocks"].tobytes() + ds1["labels"].tobytes()).hexdigest()
        assert len(h) == 64


class TestIO:
    def test_npz_roundtrip(self):
        blocks = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        labels = np.array([0, 1], dtype=np.int8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.npz"
            save_blocks_npz(path, blocks, labels=labels)
            loaded = load_blocks_npz(path)
            np.testing.assert_array_equal(loaded["blocks"], blocks)
            np.testing.assert_array_equal(loaded["labels"], labels)

    def test_hex_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blocks.hex"
            path.write_text("6565 6877\n# comment\nc69b e9bb\n")
            blk = blocks_from_file(path, format="hex")
            assert blk.shape == (2, 2)
            np.testing.assert_array_equal(blk[0], OFFICIAL_PT)

    def test_bin_file(self):
        raw = block_to_bytes(OFFICIAL_PT) + block_to_bytes(OFFICIAL_CT)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blocks.bin"
            path.write_bytes(raw)
            blk = blocks_from_file(path, format="bin")
            assert blk.shape == (2, 2)

    def test_decrypt_check(self):
        cipher = Simon3264()
        ct = cipher.encrypt(OFFICIAL_PT, OFFICIAL_KEY)
        out = decrypt_check(cipher, ct, OFFICIAL_KEY, OFFICIAL_PT)
        assert out["match"] is True

    def test_blocks_to_feature_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blocks.hex"
            path.write_text("6565 6877\n")
            feats = blocks_to_feature_file(path, format="hex")
            assert feats.ndim == 2 and feats.shape[0] == 1


class TestFeatures:
    def test_block_stats_shape(self):
        b = np.random.default_rng(0).integers(0, 0x10000, size=(8, 2), dtype=np.uint16)
        s = block_stats(b)
        assert s.shape == (8, 6)

    def test_sliding_window(self):
        b = np.random.default_rng(0).integers(0, 0x10000, size=(5, 2), dtype=np.uint16)
        w = sliding_window_xor_features(b)
        assert w.shape == (4, 32)

    def test_chi_square(self):
        ref = np.zeros(33)
        ref[16] = 100
        b = np.array([[0xFFFF, 0xFFFF]], dtype=np.uint16)
        score = chi_square_hw_vs_reference(b, ref)
        assert score >= 0


class TestTrace:
    def test_encrypt_trace_length(self):
        cipher = Simon3264()
        tr = encrypt_trace(cipher, OFFICIAL_PT, OFFICIAL_KEY, rounds=4)
        assert tr.shape == (5, 1, 2)

    def test_stop_at_round_matches_trace(self):
        cipher = Simon3264()
        stop = encrypt_stop_at_round(cipher, OFFICIAL_PT, OFFICIAL_KEY, 8)
        tr = encrypt_trace(cipher, OFFICIAL_PT, OFFICIAL_KEY, rounds=8)
        np.testing.assert_array_equal(stop[0], tr[8, 0])

    def test_subkey_bits_length(self):
        cipher = Simon3264()
        sk = cipher.get_subkeys(OFFICIAL_KEY)
        bits = subkey_bits(sk)
        assert bits.shape == (32 * 16,)

    def test_subkey_summary(self):
        cipher = Simon3264()
        sk = cipher.get_subkeys(OFFICIAL_KEY)
        stats = subkey_summary_stats(sk)
        assert "subkey_hw_mean" in stats
