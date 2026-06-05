# 09 — Configuration Reference

## 9.1 Overview

| Config file | Loader | Used by |
|-------------|--------|---------|
| `configs/default.yaml` | `ml.config.load_config` | `ml/train.py`, `ml/score.py` |
| `configs/cryptanalysis.yaml` | `experiments.config.load_cryptanalysis_config` | `experiments/*` |
| `configs/cryptanalysis_quick.yaml` | same | `experiments/run_all.py --quick` |

**No environment variables** are read. Override via CLI flags where supported.

**Silent defaults:** YAML may omit keys; dataclass defaults in Python apply. Tables below show **dataclass defaults** where YAML is incomplete.

---

## 9.2 `configs/default.yaml` — ML experiment

### `data`

| Key | YAML value | Default if omitted | Description |
|-----|------------|-------------------|-------------|
| `seed` | 42 | 42 | Master RNG seed |
| `n_samples` | 10000 | 10000 | Total dataset rows |
| `anomaly_fraction` | 0.20 | 0.20 | Fraction label 1 |
| `fault_types` | random, flip, wrong_rounds | see dataclass | Anomaly generators |
| `wrong_rounds_values` | [8, 16] | [8, 16] | Partial rounds |
| `wrong_z_index` | 1 | 1 | Alternate z for wrong_z fault |
| `flip_bits` | 3 | 3 | Bits flipped per flip fault |
| `feature_bits` | true | true | Include bits in dataset dict (simon3264) |

### `split`

| Key | Default | Description |
|-----|---------|-------------|
| `train_ratio` | 0.70 | Train fraction per stratum |
| `val_ratio` | 0.15 | Validation fraction |
| `split_seed` | 0 | Split RNG (independent of data seed) |

### `features`

| Key | YAML | Code default | Description |
|-----|------|--------------|-------------|
| `include_bits` | true | true | 32 raw bits |
| `include_stats` | true | true | 6 HW/norm columns |
| `include_transitions` | *omitted* | **true** | 2 transition columns |
| `include_entropy` | *omitted* | **true** | 2 entropy columns |

**Effective dimension:** 6+2+2+32 = **42** with shipped YAML.

### `isolation_forest`

| Key | Default |
|-----|---------|
| `n_estimators` | 200 |
| `max_samples` | auto |
| `contamination` | auto |
| `random_state` | 42 |
| `n_jobs` | -1 |

### `autoencoder` (NumPy)

| Key | Default |
|-----|---------|
| `hidden_dims` | [64, 32, 16] |
| `latent_dim` | 8 |
| `epochs` | 60 |
| `batch_size` | 256 |
| `lr` | 0.001 |
| `patience` | 10 |

### `torch_autoencoder`

| Key | Default |
|-----|---------|
| `hidden_dims` | [128, 64, 32] |
| `latent_dim` | 16 |
| `dropout` | 0.1 |
| `epochs` | 40 |
| `batch_size` | 512 |
| `device` | auto |

### `scoring`

| Key | Default | Notes |
|-----|---------|-------|
| `target_fpr` | 0.01 | Validation threshold tuning |
| `score_agg` | mean | **Currently unused in code** — see [18](18-technical-debt-and-roadmap.md) |

### `paths`

| Key | Default |
|-----|---------|
| `data_dir` | data/ |
| `model_dir` | models/ |
| `results_dir` | results/ |

### `ml/train.py` CLI overrides

| Flag | Overrides |
|------|-----------|
| `--config` | Config file path |
| `--n-samples` | `data.n_samples` |
| `--anomaly-fraction` | `data.anomaly_fraction` |
| `--epochs` | Both AE epoch counts |
| `--model-dir` | `paths.model_dir` |
| `--results-dir` | `paths.results_dir` |
| `--seed` | data + model seeds |
| `--force-regen` | Ignore dataset cache |
| `--torch-only` | Skip IF and NumPy AE |
| `--no-torch` | Skip PyTorch AE |
| `--no-autoencoder` | Skip NumPy AE |

---

## 9.3 `configs/cryptanalysis.yaml`

### `round_sweep`

| Key | Value (shipped) |
|-----|-----------------|
| `round_values` | [8, 10, 12, 14, 16, 18, 20, 24, 28, 32] |
| `n_samples_per_round` | 2000 |
| `seed` | 42 |
| `reference_model` | torch_autoencoder |
| `model_dir` | models/ |
| `results_dir` | results/round_sweep/ |

### `distinguisher`

| Key | Value (shipped) |
|-----|-----------------|
| `round_values` | [8, 10, 12, 14, 16, 18, 20] |
| `n_samples_per_round` | 10000 |
| `input_delta` | [1, 0] |
| `feature_mode` | xor_bits |
| `hidden_dims` | [128, 64] |
| `epochs` | 30 |
| `batch_size` | 512 |
| `lr` | 0.001 |
| `patience` | 6 |
| `device` | auto |

### `paths`

Shared `data_dir`, `model_dir`, `results_dir` for reporting.

### `cryptanalysis_quick.yaml`

Reduced rounds, `n_samples_per_round` 200/400, distinguisher `epochs: 5`, smaller MLP.

---

## 9.4 Configuration decision matrix

| I want to… | Edit |
|------------|------|
| More training data | `default.yaml` → `data.n_samples` |
| Fewer false alarms | Lower `scoring.target_fpr` |
| Faster Colab train | `--torch-only --n-samples 5000 --epochs 20` |
| Faster assignment smoke | `experiments/run_all.py --quick` |
| Distinguisher bit features | `distinguisher.feature_mode: concat_bits` |
| Match ML features in distinguisher | `feature_mode: ml_features` |

---

[← Persistence](08-data-persistence-and-artifacts.md) · [Next: Features →](10-feature-engineering.md)
