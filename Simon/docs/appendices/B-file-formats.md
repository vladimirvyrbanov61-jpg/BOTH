# Appendix B — File Formats

## B.1 Hex text format (`--format hex`)

**One block per line.** Supported line styles:

| Style | Example |
|-------|---------|
| Two hex words | `6565 6877` |
| Eight hex chars | `65656877` |
| Four byte tokens | `65 65 77 68` |

- Lines starting with `#` are ignored.
- Words are parsed as **16-bit hex** values, not byte-swapped pairs unless using byte token form.

**Endianness:** `bytes_to_block` defaults to **little-endian** per 16-bit word.

Official vector plaintext `6565 6877` → LE bytes `65 65 77 68`.

---

## B.2 Binary format (`--format bin`)

- Concatenated **4-byte blocks** (2 words × 2 bytes).
- Length must be multiple of 4.

---

## B.3 NPZ format (`--format npz`)

### Dataset cache (`ml/data.py`)

| Array key | dtype | Shape |
|-----------|-------|-------|
| `blocks` | uint16 | (N, 2) |
| `labels` | int8 | (N,) |
| `features` | float64 | (N, F) |

Meta: companion `.meta.json` list of dicts.

### Distinguisher cache

| Key | dtype |
|-----|-------|
| `X` | float64 (N, F) |
| `y` | int8 (N,) |
| `rounds` | int (metadata) |

---

## B.4 Model files

### `torch_autoencoder.pt` (PyTorch)

| Key | Content |
|-----|---------|
| `cfg` | `TorchAutoencoderConfig` |
| `state_dict` | Weights |
| `xmin`, `xmax` | Normalization vectors |
| `input_dim` | Feature dimension |
| `train_history` | Loss list |

### `*.pkl` (Isolation Forest / NumPy AE)

Pickle dict with `cfg` and internal model state.

### `thresholds.json`

```json
{ "torch_autoencoder": 0.01, "autoencoder": 0.02, "iso_forest": 0.5 }
```

---

## B.5 Results CSV

### `predictions_<model>.csv`

```
index,score,prediction,true_label,fault
```

### `round_sweep.csv`

```
rounds,mean_score,median_score,detection_rate,false_positive_rate,auc_vs_random,n_samples,...
```

### `distinguisher/summary.csv`

```
rounds,accuracy,auc,advantage,tpr,tnr,n_test
```

---

[← Glossary](A-glossary.md) · [Appendix C →](C-references.md)
