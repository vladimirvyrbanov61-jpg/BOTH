# SIMON 32/64 anomaly detection on Google Colab

**Full step-by-step guide (Simon + Speck):** [`../COLAB_GUIDE.md`](../COLAB_GUIDE.md)

## 1. Get the code on Colab

```python
# Option A: upload a zip of d:\Simon and unzip
# Option B: clone from your git remote
# !git clone https://YOUR_REPO.git Simon && %cd Simon
```

## 2. Install dependencies

```python
!pip install -q -r requirements-colab.txt
```

## 3. Quick train (PyTorch neural network)

```python
import sys
from pathlib import Path
ROOT = Path(".").resolve()  # repo root
sys.path.insert(0, str(ROOT))

!python ml/train.py --torch-only --n-samples 20000 --epochs 30 --force-regen
```

## 4. Evaluate on held-out test split

```python
!python ml/score.py evaluate --model torch_autoencoder
```

## 5. Score a hex capture file

```python
# One block per line: "6565 6877" or 8 hex chars
!python ml/score.py score-file captures.hex --format hex --model torch_autoencoder
```

## Models

| File | Description |
|------|-------------|
| `models/torch_autoencoder.pt` | **PyTorch MLP** (recommended on Colab / GPU) |
| `models/autoencoder.pkl` | NumPy MLP (CPU, no torch) |
| `models/iso_forest.pkl` | sklearn baseline |
| `models/thresholds.json` | Validation-tuned thresholds |

## Config

Edit `configs/default.yaml` or pass `--config path/to.yaml`.

Normal class = valid SIMON 32/64 ciphertext; anomalies = injected faults (random blocks, bit flips, wrong rounds, etc.) from `simon3264.dataset`.

## Assignment experiments (round sweep + neural distinguisher)

For the full assignment pipeline (varying rounds + automated distinguisher), use `notebooks/Assignment_Colab.ipynb` or run `python experiments/run_all.py --config configs/cryptanalysis.yaml`. See `docs/ASSIGNMENT_ALIGNMENT.md`.
