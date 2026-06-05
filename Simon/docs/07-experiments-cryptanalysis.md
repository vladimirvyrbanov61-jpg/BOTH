# 07 — `experiments/` Cryptanalysis Layer

## 7.1 Purpose

The `experiments/` package is an **additive assignment layer** on top of the existing ML stack. It addresses coursework requirements that the base anomaly pipeline alone does not cover:

| Requirement | Track | Module |
|-------------|-------|--------|
| Effectiveness vs **varying round counts** (anomaly framing) | **A** | `round_sweep.py` |
| **Automated cryptanalysis** (distinguisher) | **B** | `distinguisher_*`, `run_distinguisher.py` |
| Combined report + plots | Both | `reporting.py`, `run_all.py` |

**Why separate from `ml/`?** Preserves stable `ml/train.py` / `ml/score.py` behaviour; cryptanalysis configs live in `configs/cryptanalysis*.yaml` and `experiments/config.py`.

---

## 7.2 Configuration

`experiments/config.py` defines:

- `RoundSweepConfig` — round list, samples, model paths
- `DistinguisherConfig` — rounds, samples, Δ, feature mode, MLP hyperparams
- `CryptanalysisConfig` — container + `paths`

Loader: `load_cryptanalysis_config(path)` — **independent** of `ml.config.load_config`.

| File | Use |
|------|-----|
| `configs/cryptanalysis.yaml` | Full sweeps (Colab / report) |
| `configs/cryptanalysis_quick.yaml` | Fast smoke (~minutes) |

---

## 7.3 Track A — Round sweep (`round_sweep.py`)

### Question answered

> Does the **same** Torch autoencoder (trained on 32-round normal features) flag **truncated-round** ciphertext as anomalous?

### Algorithm (per round `r`)

1. Sample keys/plaintexts; `encrypt_rounds(pt, key, r)`
2. `build_feature_matrix` → score with loaded `TorchAutoencoder`
3. Compare to `random_blocks` baseline (AUC vs random)
4. Apply fixed threshold from `models/thresholds.json`

### Metrics (per row in `round_sweep.csv`)

| Column | Meaning |
|--------|---------|
| `mean_score` | Mean AE reconstruction error at round `r` |
| `median_score` | Median error |
| `detection_rate` | If `r < 32`: fraction flagged anomalous (high good). If `r = 32`: **specificity** — fraction correctly not flagged |
| `false_positive_rate` | At `r = 32` only: fraction wrongly flagged |
| `auc_vs_random` | AUC separating Simon `r`-round vs random blocks |
| `n_samples` | Sample count |

### Outputs

- `results/round_sweep/round_sweep.csv`
- `results/round_sweep/summary.json`
- `results/round_sweep/score_vs_rounds.png`

```bash
python experiments/round_sweep.py --config configs/cryptanalysis.yaml
python experiments/round_sweep.py --rounds 8 16 32   # override list
```

---

## 7.4 Track B — Neural distinguisher

### Question answered

> Can a neural network **automate** classification of reduced-round Simon ciphertext **pairs** (fixed input difference Δ) vs **random** pairs?

This is standard **distinguisher** language — **not** key recovery.

### Data (`distinguisher_data.py`)

For each sample:

1. Random key `k`, plaintext `P`
2. `P' = P ⊕ Δ` (word-wise; default Δ = `[1, 0]` on left word)
3. **Class 1:** `(Enc_r(P,k), Enc_r(P',k))`
4. **Class 0:** independent random blocks `(R0, R1)`

**Feature modes** (`feature_mode` in config):

| Mode | Dim | Definition |
|------|-----|------------|
| `xor_bits` | 32 | `blocks_to_bits(C ⊕ C')` |
| `concat_bits` | 64 | `bits(C) ‖ bits(C')` |
| `ml_features` | 42 | `build_feature_matrix(C ⊕ C')` |

Cache: `data/distinguisher_r{r}_{hash}.npz`

Split: 70/15/15 stratified by label (`stratified_split_indices`).

### Model (`distinguisher_model.py`)

`NeuralDistinguisher` — PyTorch MLP, `BCEWithLogitsLoss`, outputs `predict_proba` = P(real).

Saved: `models/distinguisher_r{r}.pt`

### Evaluation (`run_distinguisher.py`)

Per round: accuracy, AUC, advantage `|acc − 0.5|`, **TPR**, **TNR**.

Outputs:

- `results/distinguisher/metrics_r{r}.json`
- `results/distinguisher/summary.csv`
- `results/distinguisher/summary.json`
- `results/distinguisher/advantage_vs_rounds.png`

---

## 7.5 Orchestration (`run_all.py`)

```bash
python experiments/run_all.py --config configs/cryptanalysis.yaml
python experiments/run_all.py --quick          # quick yaml + faster AE train if needed
python experiments/run_all.py --skip-train
python experiments/run_all.py --force-regen    # rebuild distinguisher caches
```

Steps:

1. If missing: subprocess `python ml/train.py --torch-only` (adds `--n-samples 2000 --epochs 10` when `--quick`)
2. `run_round_sweep(cfg)`
3. `run_distinguisher_experiment(cfg)`
4. `generate_assignment_report()` → `results/ASSIGNMENT_SUMMARY.md`

---

## 7.6 Reporting (`reporting.py`)

Merges Track A/B CSVs and embeds plot paths into Markdown with methodology boilerplate for student reports.

---

## 7.7 Verification script

```bash
python scripts/verify_assignment_setup.py
```

Checks required files exist and configs load; reports PyTorch / base model status.

---

[← ML pipeline](06-ml-anomaly-pipeline.md) · [Next: Persistence →](08-data-persistence-and-artifacts.md)
