# Thesis pivot — file migration map (incremental)

## Phase 1 (this commit) — shared crypto + blind data

| Status | Path | Role |
|--------|------|------|
| **NEW** | `ciphers/common/encoding.py` | MSB-first bit layout; `concat_pair_bits` → float32 (64,) |
| **NEW** | `ciphers/common/sampling.py` | Uniform keys (64b), plaintexts (32b), random pairs |
| **NEW** | `ciphers/registry.py` | `get_cipher("simon"\|"speck")` → profile instance |
| **NEW** | `thesis/data/generator.py` | Blind 50/50 differential dataset; X (N,64), y |
| **NEW** | `thesis/data/cache.py` | NPZ cache (X, y only — no P/K) |
| **NEW** | `thesis/config/thesis.yaml` | delta, rounds list, sample counts |
| **MOD** | `Simon/simon3264/cipher.py` | `encrypt(..., rounds=R)` optional API |
| **MOD** | `Speck/speck3264/cipher.py` | `encrypt(..., rounds=R)` optional API |
| **NEW** | `tests/test_thesis_encoding.py` | Encoding + generator smoke tests |

## Phase 2 — neural + sweep (done)

| Status | Path | Role |
|--------|------|------|
| **NEW** | `thesis/models/cnn_distinguisher.py` | 1D CNN; (B,64)→(B,4,16); sigmoid proba |
| **NEW** | `thesis/models/train.py` | Adam + BCE; early stop on val AUC |
| **NEW** | `thesis/eval/metrics.py` | Acc, AUC, TPR, TNR, advantage |
| **NEW** | `thesis/eval/round_sweep.py` | Simon + Speck sweep → CSV |
| **NEW** | `tests/test_cnn.py` | Shape + loss smoke tests |

## Phase 3 — classical DDT + comparison (done)

| Status | Path | Role |
|--------|------|------|
| **NEW** | `thesis/classical/ddt_core.py` | DDT normalization, max-trail DP |
| **NEW** | `thesis/classical/ddt_simon.py` | Exact f-DDT + analytical/empirical round |
| **NEW** | `thesis/classical/ddt_speck.py` | Monte Carlo ARX round DDT (1M samples) |
| **NEW** | `thesis/classical/characteristic.py` | R-round p_max tracking + CSV |
| **NEW** | `thesis/eval/compare.py` | AI advantage vs log₂(p_max) plots |
| **NEW** | `tests/test_classical.py` | Probability + shape validation |

**Architectural pivot: complete.** Thesis path is self-contained under `thesis/` + `ciphers/common/`.

## Legacy (unchanged, not imported by thesis/)

| Path | Disposition |
|------|-------------|
| `Simon/ml/**`, `Speck/ml/**` | Fault / AE pipeline — archive later |
| `Simon/simon3264/faults.py`, `dataset.py` | Fault injection |
| `Simon/experiments/distinguisher_data.py` | Superseded by `thesis/data/generator.py` |

## Import graph (thesis)

```
thesis/data/generator.py
  → ciphers/common/{encoding,sampling}
  → ciphers/registry.py
       → Simon/simon3264/cipher.py  OR  Speck/speck3264/cipher.py
```

No imports from `ml/`, `faults/`, or `recovery_error`.

## Phase 4 operational — sweep artifacts + orchestration (done)

| Status | Path | Role |
|--------|------|------|
| **NEW** | `thesis/config/thesis_quick.yaml` | Smoke profile (5k samples, R=3–5, short training) |
| **MOD** | `thesis/config/loader.py` | `load_profile("full"\|"quick")`, `config_path_for_profile()` |
| **MOD** | `thesis/eval/round_sweep.py` | `--profile`, `--n-samples`, `--fresh-csv`, path overrides |
| **MOD** | `thesis/eval/compare.py` | Clear error if neural CSV missing; `--profile` |
| **NEW** | `thesis/run_thesis.py` | Optional pytest → sweep → compare |
| **NEW** | `tests/test_round_sweep_smoke.py` | Tiny sweep integration (tmp dirs) |
| **NEW** | `README.md` (repo root) | Thesis-first commands |

### Artifact checklist (after a successful run)

- [ ] `results/thesis/simon_round_sweep.csv` — test rows, columns: cipher, rounds, split, n_samples, accuracy, auc_roc, tpr, tnr, advantage
- [ ] `results/thesis/speck_round_sweep.csv` (if speck in config)
- [ ] `models/thesis/{cipher}_R{R}.pt` per swept round
- [ ] `results/thesis/{cipher}_classical_bounds.csv`
- [ ] `results/thesis/{cipher}_vs_classical.png`

### Runtime order

1. `py -3 -m pytest tests/... -v` (optional; ~18 min includes exact Simon f-DDT test)
2. `py -3 -m thesis.run_thesis --profile quick --skip-tests --fresh-csv`
3. For publication curves: `py -3 -m thesis.run_thesis --profile full --fresh-csv` (GPU, long)

First `compare` on a cipher requires matching `round_sweep.csv` from step 2.
