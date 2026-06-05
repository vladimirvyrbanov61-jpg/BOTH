# 04 — Cryptography: `simon.py`

## 4.1 Purpose

`simon.py` implements the **SIMON** family of lightweight block ciphers (Beaulieu et al., 2013) as a **standalone research primitive**. It is intentionally **decoupled** from machine learning and anomaly detection.

**Why a separate module?** Cipher correctness can be validated with unit tests and known-answer tests (KAT) without importing PyTorch or scikit-learn. Other Simon parameter sizes remain available for benchmarking even though the ML stack fixes **SIMON 32/64**.

---

## 4.2 Security disclaimer

From the module docstring:

- **Research / benchmarking / education only**
- **Not** for production cryptographic use
- **No** constant-time guarantees
- **No** side-channel mitigations

Operators must not deploy this code to protect real data.

---

## 4.3 Public API

```python
from simon import Simon, expand_key, encrypt_blocks, decrypt_blocks, encrypt_blocks_trace

cipher = Simon(n=16, m=4)   # SIMON 32/64
ct = cipher.encrypt(plaintext, key)   # (N, 2) uint16 or (2,)
pt = cipher.decrypt(ciphertext, key)
```

| Symbol | Role |
|--------|------|
| `Simon(n, m, z_index=, rounds=)` | Parameterized cipher instance |
| `expand_key(key, n, m, rounds, z_index)` | Round subkeys |
| `encrypt_blocks(pt, subkeys, n, rounds)` | Batch encryption with pre-expanded keys |
| `decrypt_blocks(ct, subkeys, n, rounds)` | Inverse |
| `encrypt_blocks_trace(...)` | States after each round, shape `(rounds+1, N, 2)` |
| `rol`, `ror`, `f_round` | Low-level primitives |

---

## 4.4 Data layout conventions

### Plaintext / ciphertext

- Shape `(N, 2)` or `(2,)` — `dtype` `uint16` for `n=16`
- `[..., 0]` = left word **x** (Feistel half where `f` applies)
- `[..., 1]` = right word **y**

### Key (API input)

- Shape `(m,)`, `(1, m)`, or `(N, m)`
- **Big-endian word order:** `[k_{m-1}, …, k_0]` — most significant key word first
- Example SIMON 32/64 test key: `[0x1918, 0x1110, 0x0908, 0x0100]`

Internally, `expand_key` reverses to `[k_0, …, k_{m-1}]` for the recurrence.

### Batch behaviour

- `(N, 2)` plaintext with broadcast key `(m,)` → same key for all blocks
- `(N, 2)` with `(N, m)` keys → per-block keys

---

## 4.5 Round function (n = 16)

For word `x`:

```
f(x) = (S¹x ∧ S⁸x) ⊕ S²x
```

One round with round key `k`:

```
x' = y ⊕ f(x) ⊕ k
y' = x
```

---

## 4.6 Supported parameter pairs

The table `_SIMON_PARAMS` maps `(n, m)` → `(z_index, default_rounds)`. Includes SIMON 32/64 and other published sizes.

**ML and `simon3264` freeze** `n=16`, `m=4`, `ROUNDS=32`, `Z_INDEX=0` in `simon3264/cipher.py`.

---

## 4.7 Official test vector (Appendix B)

| Field | Hex words |
|-------|-----------|
| Key | `1918 1110 0908 0100` |
| Plaintext | `6565 6877` |
| Ciphertext | `c69b e9bb` |

Verified in `test_simon.py` and `test_simon3264.py`.

**Endian note:** Little-endian **bytes** for words `6565 6877` are `65 65 77 68` (see [05 — simon3264 encoding](05-simon3264-toolkit.md)).

---

## 4.8 `encrypt_blocks_trace`

Returns intermediate states for cryptanalysis and visualization. Consumed by `simon3264/trace.py` (`encrypt_trace`, `encrypt_stop_at_round`).

---

## 4.9 Relationship to `Simon3264`

`Simon3264` wraps `simon.py` with:

- Frozen 32/64 parameters
- Subkey caching keyed by raw key bytes
- `encrypt_rounds` for partial encryption (fault injection / round sweeps)
- `encrypt_variant` for wrong `z` or round count

Always prefer `Simon3264` for the ML and experiments layers unless you need another Simon size from the full table.

---

[← Layout](03-repository-layout.md) · [Next: simon3264 →](05-simon3264-toolkit.md)
