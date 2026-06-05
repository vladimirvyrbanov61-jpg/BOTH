# 14 — Onboarding & Troubleshooting

## 14.1 Prerequisites

| Requirement | Version / notes |
|-------------|-----------------|
| Python | 3.9+ tested (3.9.2 on Windows) |
| pip | Current |
| RAM | ≥ 4 GB for default `n_samples=10000`; more for full distinguisher sweep |
| GPU | Optional; PyTorch uses CUDA when `device: auto` |

---

## 14.2 Installation paths

### Local development (full)

```bash
git clone <repo-url> Simon
cd Simon
pip install -r requirements.txt
python scripts/verify_assignment_setup.py
pytest
```

### Google Colab

```bash
%cd Simon
!pip install -q -r requirements-colab.txt
!python scripts/verify_assignment_setup.py
```

See [16 — Assignment & Colab](16-assignment-and-colab.md).

---

## 14.3 Environment variables

**None required.** The project does not read `.env` files or OS environment variables for configuration.

| If you need… | Use instead |
|--------------|-------------|
| Different data path | `paths.data_dir` in YAML or `--model-dir` |
| GPU selection | `torch_autoencoder.device` / `distinguisher.device` (`auto`, `cpu`, `cuda`) |
| Reproducibility | `seed` fields in YAML + `lock_seeds()` in train |

---

## 14.4 First successful run (recommended order)

```bash
# 1. Verify structure
python scripts/verify_assignment_setup.py

# 2. Train PyTorch AE (~minutes on GPU, longer on CPU)
python ml/train.py --torch-only --n-samples 5000 --epochs 20

# 3. Evaluate
python ml/score.py evaluate --model torch_autoencoder

# 4. Quick assignment smoke
python experiments/run_all.py --quick
```

---

## 14.5 Troubleshooting matrix

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: torch` | PyTorch not installed | `pip install torch` or use Colab |
| `Missing models/torch_autoencoder.pt` | Track A / run_all before train | `python ml/train.py --torch-only` |
| `threshold key 'torch_autoencoder' not in thresholds.json` | Partial train or wrong model dir | Retrain or pass `--model-dir` |
| `FileNotFoundError` on config | Wrong cwd | `cd` to repo root |
| `ImportError` for simon3264 | cwd not repo root | Run CLIs not bare modules |
| Stale metrics after config change | NPZ cache | `--force-regen` |
| pytest 2 skipped | No torch | Expected on minimal install; install torch for full coverage |
| pip SSL error on torch | Corporate proxy / certs | Use Colab or fix pip certs |
| All blocks flagged anomalous | Threshold too low or feature mismatch | Retrain; align feature flags |
| Distinguisher acc ≈ 0.5 at high rounds | Expected — cipher stronger | Report as limitation |
| Very slow distinguisher gen | Python loop per sample | Reduce `n_samples_per_round` or use `--quick` |
| Hex load fails | Wrong format | See [appendix B](appendices/B-file-formats.md) |
| KAT failure after edit | Cipher regression | Revert `simon.py` changes |

---

## 14.6 Working directory and imports

Every CLI does:

```python
sys.path.insert(0, str(REPO_ROOT))
```

**Do not** run `python -m ml.train` unless the package is installed. Prefer:

```bash
python ml/train.py
```

---

## 14.7 Disk space planning

| Artifact | Rough size driver |
|----------|-------------------|
| `dataset_*.npz` | O(n_samples × 42 × 8 bytes) features |
| `distinguisher_r*.npz` | O(n_samples_per_round × 32 × 8) per round |
| `torch_autoencoder.pt` | Small (MB) |

Full `cryptanalysis.yaml` with 7 distinguisher rounds × 10000 samples: allow **hundreds of MB** in `data/`.

---

## 14.8 Python version edge cases

- Use `from __future__ import annotations` throughout (3.9 compatible).
- Type syntax avoids `X | Y` union in older-tested files.

---

[← Testing](13-testing-and-quality.md) · [Next: Operations →](15-operations-and-observability.md)
