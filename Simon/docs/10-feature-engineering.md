# 10 — Feature Engineering

## 10.1 Two feature systems (do not confuse)

| System | Module | Used by |
|--------|--------|---------|
| **Canonical ML (42-dim)** | `ml/features.py` → `build_feature_matrix` | `ml/data.py`, `ml/train.py`, `ml/score.py`, Track A round sweep (optional in Track B) |
| **simon3264 stats** | `simon3264/features.py` → `blocks_to_feature_matrix` | Legacy/helpers, `io.blocks_to_feature_file` |

**Rule:** Any anomaly model training or Track A scoring must use **`ml.features.build_feature_matrix`** with the same boolean flags as training.

---

## 10.2 `build_feature_matrix` composition

Default flags (all `True`): **42 columns**

| Group | Columns | Names (via `feature_names()`) |
|-------|---------|-------------------------------|
| Stats | 6 | `hw_left`, `hw_right`, `hw_total`, `xor_hw`, `norm_left`, `norm_right` |
| Transitions | 2 | `transitions_01`, `transitions_10` |
| Entropy | 2 | `entropy_left`, `entropy_right` |
| Bits | 32 | `bit_0` … `bit_31` |

### Stats (`block_stats` from simon3264)

- Hamming weights of left/right words, total, XOR-half HW
- Normalized raw word values in `[0, 1]`

### Transitions (`bit_transitions`)

On MSB-first 32-bit expansion: count 0→1 and 1→0 adjacent bit changes.

### Entropy (`word_entropy`)

Per-word entropy proxy from Hamming weight (not full byte entropy).

### Bits

`blocks_to_bits` → float64 0/1 values.

---

## 10.3 Disabling feature groups

```python
from ml.features import build_feature_matrix

X = build_feature_matrix(
    blocks,
    include_bits=False,
    include_stats=True,
    include_transitions=True,
    include_entropy=True,
)
```

Must match `configs/default.yaml` `features:` section and `ml/score.py` `_feature_kwargs(cfg)`.

---

## 10.4 Distinguisher features (Track B)

Implemented in `experiments/distinguisher_data.pair_to_features`:

| `feature_mode` | Dimension | Formula |
|----------------|-----------|---------|
| `xor_bits` | 32 | `blocks_to_bits(C ⊕ C')` |
| `concat_bits` | 64 | `[bits(C); bits(C')]` |
| `ml_features` | 42 | `build_feature_matrix(C ⊕ C')` |

**Why `xor_bits` default?** Standard neural distinguisher literature uses XOR of pair difference pattern.

**Input difference Δ:** `input_delta: [1, 0]` → XOR `0x0001` into left word of plaintext pair.

---

## 10.5 Feature scaling (models)

| Model | Scaling |
|-------|---------|
| Isolation Forest | None (sklearn on raw features) |
| NumPy AE | Min-max fit on **training normals** |
| Torch AE | Per-column min-max fit on **training normals** (`_xmin`, `_xmax` in checkpoint) |

Scoring applies the **saved** Torch normalization from the checkpoint.

---

## 10.6 Design rationale

| Choice | Why |
|--------|-----|
| Hand-crafted stats + bits | Interpretable, fixed size, no raw ciphertext bytes in NN without structure |
| Include raw bits | Lets AE learn non-linear bit patterns beyond HW |
| 42-dim fixed size | Batch training on Colab without variable-length input |
| Separate distinguisher features | Pair-based cryptanalysis tradition; XOR pattern is the signal |

---

[← Configuration](09-configuration-reference.md) · [Next: Models & APIs →](11-models-and-apis.md)
