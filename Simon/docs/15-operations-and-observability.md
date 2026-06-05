# 15 — Operations Runbook

## 15.1 Scope

This project has **no production service** (no HTTP API, no Kubernetes, no database). “Operations” means **running batch research jobs** reliably on a laptop, lab machine, or Colab VM.

---

## 15.2 Deployment model

| Environment | Role | Lifecycle |
|-------------|------|-----------|
| Local Python venv | Development | Manual |
| Google Colab | GPU training + assignment | Ephemeral VM |
| CI (future) | pytest on push | Not yet implemented |

---

## 15.3 Standard operating procedures

### SOP-1: Full ML retrain

1. Backup `models/` if needed
2. `python ml/train.py --torch-only --force-regen --n-samples 20000 --epochs 40`
3. `python ml/score.py evaluate --model torch_autoencoder`
4. Archive `results/val_metrics.json`

### SOP-2: Full assignment reproduction

1. Complete SOP-1 (or `--skip-train` if models exist)
2. `python experiments/run_all.py --config configs/cryptanalysis.yaml`
3. Verify artifacts:
   - `results/round_sweep/score_vs_rounds.png`
   - `results/distinguisher/advantage_vs_rounds.png`
   - `results/ASSIGNMENT_SUMMARY.md`
4. Copy figures into report

### SOP-3: Score external capture

1. Confirm file format (hex/bin/npz)
2. `python ml/score.py score-file <file> --format hex --model torch_autoencoder`
3. Inspect `results/scores_*.csv`

---

## 15.4 Scaling guidance

| Knob | Effect | Tradeoff |
|------|--------|----------|
| `n_samples` | More stable metrics | Time, disk |
| `epochs` | Better AE fit | Overfit risk; time |
| `batch_size` | GPU utilization | Memory |
| `round_values` length | Finer sweep plots | Distinguisher trains N models |
| `n_samples_per_round` | Smoother curves | Generation time |

**Horizontal scaling:** Not built-in. Shard by seed and merge CSVs manually if needed.

---

## 15.5 Observability (what to watch)

There is no structured logging framework. Monitor **stdout**:

| Stage | Expected output |
|-------|-----------------|
| `ml/train.py` | Split summary, per-model metric tables |
| `round_sweep.py` | Per-round mean_score, detection_rate, auc |
| `run_distinguisher.py` | Per-round acc, auc, advantage, tpr, tnr |
| `run_all.py` | `[run_all] Found ...` or training subprocess |

**Artifacts to inspect on failure:**

- `results/val_metrics.json` — training sanity
- `models/thresholds.json` — threshold presence
- Last lines of pytest output

---

## 15.6 Common failure modes

| Failure | Severity | Response |
|---------|----------|----------|
| OOM on GPU | High | Reduce `batch_size` or `n_samples` |
| Disk full in `data/` | High | Delete old `dataset_*` / `distinguisher_*` NPZ |
| NaN AUC | Medium | Single-class slice in eval — increase samples |
| Flat distinguisher advantage | Low at high rounds | Expected — document in report |
| High FPR on real data | High | Domain shift — retrain on representative normals |

---

## 15.7 Incident-style FAQ

**Q: Training hung on Colab**  
A: Check GPU runtime; reduce epochs; restart runtime and rerun with cached `data/`.

**Q: Round sweep all detection_rate ≈ 1**  
A: Threshold may be too low; inspect score distributions; retrain AE.

**Q: Cannot reproduce paper numbers**  
A: Exact values depend on seed, sample size, and hardware — compare **trends**, not absolute accuracy.

---

## 15.8 Backup and recovery

| Asset | Backup? |
|-------|---------|
| Source code | Git |
| `configs/*.yaml` | Git |
| `models/*.pt` | Optional copy (large) |
| `data/*.npz` | Regenerable — backup only if generation is costly |
| `results/` | Regenerable from models + config |

---

[← Onboarding](14-onboarding-and-troubleshooting.md) · [Next: Assignment →](16-assignment-and-colab.md)
