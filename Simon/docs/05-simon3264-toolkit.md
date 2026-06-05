# 05 — `simon3264/` Toolkit

## 5.1 Purpose

`simon3264` is the **SIMON 32/64 profile layer**: encoding, synthetic faults, labelled datasets, file I/O, statistical features, and encryption traces. It is the **bridge** between `simon.py` and all ML code.

**Why freeze parameters?** Anomaly models assume a fixed cipher geometry (32-bit blocks, 64-bit keys, 32 rounds). Allowing arbitrary `(n, m)` in the dataset layer would silently break feature dimensions and test vectors.

---

## 5.2 Package exports (`__init__.py`)

Typical imports:

```python
from simon3264 import (
    Simon3264,
    generate_labeled_dataset,
    DatasetConfig,
    blocks_from_file,
    blocks_to_bits,
)
```

See `simon3264/__init__.py` for the full export list.

---

## 5.3 `cipher.py` — `Simon3264`

| Constant | Value |
|----------|-------|
| `N_BITS` | 16 (word width) |
| `M_WORDS` | 4 |
| `ROUNDS` | 32 |
| `Z_INDEX` | 0 |

| Method | Description |
|--------|-------------|
| `encrypt(pt, key)` | Full 32-round encryption |
| `decrypt(ct, key)` | Inverse |
| `get_subkeys(key, use_cache=True, rounds=, z_index=)` | Expanded keys; **cached** by key bytes |
| `encrypt_with_subkeys(pt, sk)` | Encrypt without re-expanding |
| `encrypt_rounds(pt, key, num_rounds)` | Partial encryption — **Track A/B critical** |
| `encrypt_variant(pt, key, rounds=, z_index=)` | Wrong schedule parameters |
| `encrypt_ecb(pts, key)` | Alias for independent-block encryption |

**Why subkey caching?** Dataset generation performs many encryptions under the same key; caching avoids repeated `expand_key` work.

---

## 5.4 `encoding.py`

| Function | Role |
|----------|------|
| `block_to_bytes` / `bytes_to_block` | 4-byte block ↔ two words |
| `bytes_to_blocks` | Concatenated raw ciphertext file |
| `block_to_bits` / `blocks_to_bits` | MSB-first bit vectors, shape `(32,)` / `(N, 32)` |
| `validate_block` / `validate_blocks` | Reject values > `0xFFFF` before cast |

Default file endianness: **little-endian per 16-bit word**.

---

## 5.5 `faults.py` — synthetic anomalies

| Fault | Effect |
|-------|--------|
| `random` | Uniform random block (non-cipher) |
| `flip` | Encrypt then flip random bits |
| `swap` | Encrypt then swap halves |
| `xor` | Encrypt then XOR random mask |
| `identity` | Output = plaintext (no cipher) |
| `wrong_rounds` | `encrypt_rounds(..., r)` for `r ∈ wrong_rounds_values` |
| `wrong_z` | Encrypt with alternate `z` sequence |

Used by `dataset._apply_fault` when building label `1` rows.

---

## 5.6 `dataset.py`

### `DatasetConfig`

| Field | Default | Meaning |
|-------|---------|---------|
| `seed` | 0 | RNG seed |
| `n_samples` | 1024 | Total rows |
| `anomaly_fraction` | 0.2 | Fraction label `1` |
| `fault_types` | random, flip, wrong_rounds | Cycle for anomalies |
| `wrong_rounds_values` | [8, 16] | Partial round counts |
| `wrong_z_index` | 1 | Alternate z |
| `flip_bits` | 3 | Bit flips per flip fault |
| `feature_bits` | True | Include `bits` in dataset dict |

### `generate_labeled_dataset`

Returns dict:

| Key | Shape / type |
|-----|----------------|
| `blocks` | `(N, 2)` uint16 |
| `labels` | `(N,)` int8 — 0 normal, 1 anomaly |
| `meta` | `[{"fault": "normal"}, …]` |
| `bits` | `(N, 32)` optional |
| `config` | `DatasetConfig` copy |

### `stratified_split`

Splits train/val/test indices stratified by **label + fault string** (e.g. `0_normal`, `1_wrong_rounds_8`). Used by `ml/data.py`.

**Contrast:** `experiments/distinguisher_data.stratified_split_indices` stratifies by **label only** — do not assume identical split behaviour.

---

## 5.7 `io.py`

| Function | Formats |
|----------|---------|
| `save_blocks_npz` / `load_blocks_npz` | Compressed NPZ with `blocks`, optional `labels`, `keys` |
| `blocks_from_file` | `npz`, `bin` (4-byte blocks), `hex` (word lines or 8 hex chars) |
| `decrypt_check` | Optional roundtrip if key/plaintext known |
| `blocks_to_feature_file` | File → `simon3264.features` matrix → optional NPZ |

See [appendices/B-file-formats.md](appendices/B-file-formats.md).

---

## 5.8 `features.py` (simon3264-level)

Lower-level helpers: `block_stats`, `batch_hw_stats`, `chi_square_hw_vs_reference`, `sliding_window_xor_features`, `blocks_to_feature_matrix`.

**Canonical ML path** uses `ml/features.py` `build_feature_matrix` (42 dims), which **imports** `block_stats` from here.

---

## 5.9 `trace.py`

| Function | Use |
|----------|-----|
| `encrypt_trace` | Per-round state array |
| `encrypt_stop_at_round` | Ciphertext at round `r` |
| `subkey_bits`, `subkey_summary_stats` | Key schedule analysis |

Supports research extensions (e.g. visualizing diffusion) outside the default ML pipeline.

---

[← Cryptography](04-cryptography-simon.md) · [Next: ML pipeline →](06-ml-anomaly-pipeline.md)
