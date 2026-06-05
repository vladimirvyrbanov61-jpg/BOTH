# 12 — CLI Reference

All commands assume **current working directory = repository root**.

---

## 12.1 `ml/train.py`

**Purpose:** Train anomaly models; write models and thresholds.

```bash
python ml/train.py [options]
```

| Flag | Description |
|------|-------------|
| `--config PATH` | YAML config (default: `configs/default.yaml`) |
| `--n-samples N` | Override dataset size |
| `--anomaly-fraction F` | Override anomaly rate |
| `--epochs N` | NumPy + Torch AE epochs |
| `--model-dir DIR` | Output models |
| `--results-dir DIR` | Output metrics |
| `--seed N` | RNG seeds |
| `--force-regen` | Rebuild dataset cache |
| `--torch-only` | Only PyTorch AE (skip IF + NumPy AE) |
| `--no-torch` | Skip PyTorch AE |
| `--no-autoencoder` | Skip NumPy AE |

**Examples:**

```bash
python ml/train.py --torch-only --n-samples 20000 --epochs 40
python ml/train.py --config configs/default.yaml --force-regen
```

**Exit:** `0` on success; `SystemExit` if PyTorch required but missing.

---

## 12.2 `ml/score.py`

**Purpose:** Evaluate on test split or score external files.

```bash
python ml/score.py evaluate [options]
python ml/score.py score-file <path> [options]
```

### `evaluate`

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | default.yaml | Experiment config |
| `--model` | torch_autoencoder | iso_forest \| autoencoder \| torch_autoencoder |
| `--threshold` | from thresholds.json | Override cutoff |

### `score-file`

| Flag | Default | Description |
|------|---------|-------------|
| `input_file` | required | Path to ciphertext file |
| `--format` | hex | hex \| bin \| npz |
| `--model` | torch_autoencoder | Model to load |
| `--threshold` | from JSON or 0.5 | Cutoff |

**Examples:**

```bash
python ml/score.py evaluate --model torch_autoencoder
python ml/score.py score-file captures.hex --format hex --model torch_autoencoder
```

---

## 12.3 `experiments/round_sweep.py`

```bash
python experiments/round_sweep.py [--config PATH] [--rounds R ...] [--model-dir DIR]
```

**Requires:** `models/torch_autoencoder.pt`, `models/thresholds.json`.

---

## 12.4 `experiments/run_distinguisher.py`

```bash
python experiments/run_distinguisher.py [--config PATH] [--force-regen] [--rounds R ...]
```

---

## 12.5 `experiments/run_all.py`

```bash
python experiments/run_all.py [options]
```

| Flag | Description |
|------|-------------|
| `--config` | Cryptanalysis YAML |
| `--quick` | Use `cryptanalysis_quick.yaml`; faster AE if training |
| `--skip-train` | Do not auto-run `ml/train.py` |
| `--skip-round-sweep` | Track A only skipped |
| `--skip-distinguisher` | Track B only skipped |
| `--force-regen` | Rebuild distinguisher NPZ caches |

---

## 12.6 `scripts/verify_assignment_setup.py`

```bash
python scripts/verify_assignment_setup.py
```

Exit `0` if structure OK; prints PyTorch and base model status.

---

## 12.7 `pytest`

```bash
pytest                          # all tests
pytest test_simon.py -v         # cipher only
pytest test_experiments.py -v   # experiments smoke
```

---

[← Models](11-models-and-apis.md) · [Next: Testing →](13-testing-and-quality.md)
