# 03 — Repository Layout

Authoritative directory map as of documentation v2.0. Paths are relative to the repository root.

---

## 3.1 Top-level tree

```
Simon/
├── simon.py                      # SIMON cipher family (all table parameter pairs)
├── test_simon.py                 # Cipher unit tests + official KAT
├── test_simon3264.py             # simon3264 toolkit tests
├── test_ml_smoke.py              # Fast ML pipeline smoke tests
├── test_experiments.py           # experiments/ smoke tests
│
├── simon3264/                    # SIMON 32/64 anomaly-oriented API
│   ├── __init__.py               # Public re-exports
│   ├── cipher.py                 # Simon3264, encrypt_rounds, subkey cache
│   ├── encoding.py               # words ↔ bytes ↔ bits
│   ├── faults.py                 # Synthetic anomaly generators
│   ├── dataset.py                # Labeled dataset + stratified_split
│   ├── io.py                     # NPZ / hex / bin I/O
│   ├── features.py               # Lower-level statistical features
│   └── trace.py                  # Round traces, subkey bit export
│
├── ml/                           # Machine learning pipeline
│   ├── config.py                 # ExperimentConfig + YAML loader
│   ├── data.py                   # NPZ cache, DataSplit
│   ├── features.py               # 42-dim build_feature_matrix (canonical for ML)
│   ├── models.py                 # IF, NumPy AE, Torch AE
│   ├── metrics.py                # Threshold, AUC, fault breakdown
│   ├── train.py                  # Training CLI
│   └── score.py                  # evaluate / score-file CLI
│
├── experiments/                  # Assignment cryptanalysis layer
│   ├── config.py                 # CryptanalysisConfig
│   ├── round_sweep.py            # Track A
│   ├── distinguisher_data.py     # Pair dataset generation + cache
│   ├── distinguisher_model.py    # NeuralDistinguisher (PyTorch MLP)
│   ├── run_distinguisher.py      # Track B CLI
│   ├── metrics.py                # TPR/TNR for distinguisher
│   ├── reporting.py              # ASSIGNMENT_SUMMARY.md generator
│   └── run_all.py                # Full pipeline orchestrator
│
├── configs/
│   ├── default.yaml              # ML experiment defaults
│   ├── cryptanalysis.yaml        # Full assignment sweeps
│   └── cryptanalysis_quick.yaml  # Fast smoke config
│
├── data/                         # Generated caches (gitignored)
├── models/                       # Trained models (gitignored)
├── results/                      # Metrics, plots, reports (gitignored)
│
├── docs/                         # This documentation set
│   ├── PROJECT_DOCUMENTATION.md  # Master index
│   ├── 01-introduction-and-scope.md … 18-technical-debt-and-roadmap.md
│   ├── ASSIGNMENT_ALIGNMENT.md   # Rubric / report template
│   └── appendices/
│
├── notebooks/
│   └── Assignment_Colab.ipynb
├── scripts/
│   └── verify_assignment_setup.py
│
├── README.md
├── DOCUMENTATION.md              # Legacy pointer → docs/
├── COLAB.md
├── requirements.txt
└── requirements-colab.txt
```

---

## 3.2 Ownership boundaries

| Path | Owns | Must not |
|------|------|----------|
| `simon.py` | Cipher math, parameter table | Import ML packages |
| `simon3264/` | 32/64 profile, faults, dataset, I/O | Train neural nets |
| `ml/` | Features, models, train/score CLIs | Modify cipher rounds |
| `experiments/` | Round sweep, distinguisher, reports | Fork `ml/train.py` |
| `configs/` | YAML parameters | Executable logic |

---

## 3.3 Import conventions

**Recommended public imports:**

```python
from simon import Simon, encrypt_blocks
from simon3264 import Simon3264, generate_labeled_dataset, DatasetConfig
from ml.config import load_config
from ml.data import generate_or_load_dataset, DataSplit
from ml.features import build_feature_matrix
from ml.models import TorchAutoencoder
from experiments.config import load_cryptanalysis_config
```

**CLI pattern:** Each script under `ml/` and `experiments/` executes:

```python
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
```

**Why no `pip install -e .`?** The project ships without `pyproject.toml`. Consumers run from a clone with repo root on the path (implicit via CLIs).

---

## 3.4 Generated vs source artifacts

| Location | Committed? | Produced by |
|----------|------------|-------------|
| `*.py`, `configs/*.yaml`, `docs/` | Yes | Developers |
| `data/dataset_*.npz` | No | `ml/data.py` |
| `data/distinguisher_r*.npz` | No | `experiments/distinguisher_data.py` |
| `models/*.pt`, `*.pkl` | No | `ml/train.py`, `run_distinguisher.py` |
| `models/thresholds.json` | No | `ml/train.py` |
| `results/**` | No | train, score, experiments |

See [08 — Persistence & artifacts](08-data-persistence-and-artifacts.md).

---

## 3.5 Test file map

| File | Scope |
|------|--------|
| `test_simon.py` | `simon.py` primitives, KAT, batch keys, variants |
| `test_simon3264.py` | Encoding, cipher, faults, dataset, I/O, features, trace |
| `test_ml_smoke.py` | Feature dim, dataset, IF, NumPy AE, YAML load |
| `test_experiments.py` | Cryptanalysis config, distinguisher data, metrics, round sweep CSV |

---

[← Architecture](02-system-architecture.md) · [Next: Cryptography →](04-cryptography-simon.md)
