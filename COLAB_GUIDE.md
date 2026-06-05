# Google Colab Guide — Thesis Pipeline (Simon32/64 & Speck32/64)

This guide covers the **unified thesis path** at the repository root (`thesis/`, `ciphers/`). Use a **GPU runtime** for the full profile.

| Component | Role |
|-----------|------|
| `thesis/` | Blind data, 1D CNN, round sweep, DDT comparison |
| `Simon/`, `Speck/` | KAT-verified cipher cores (`encrypt(..., rounds=R)`) |
| `Simon/ml/`, `Speck/ml/` | **Legacy** fault/autoencoder — not used by thesis |

**Research / education only** — not for production cryptography.

---

## Thesis pipeline (primary)

### What you train

- **Task:** Binary distinguisher — real differential ciphertext pairs vs random pairs (Gohr-style).
- **Input:** 64-bit blind features `bits(C0) || bits(C1)`.
- **Model:** `thesis/models/cnn_distinguisher.py` — 1D CNN, BCE, early stop on val AUC.
- **Evaluation:** Advantage `|accuracy − 0.5|` per round R; compared to classical `log₂(p_max)` from DDT.

### Colab setup

```python
# Upload BOTH.zip or clone the repo, then:
%cd /content/BOTH   # adjust path

!pip install -q -r requirements-thesis.txt
```

### Smoke run (verify GPU + paths)

```python
!python -m thesis.run_thesis --profile quick --skip-tests --fresh-csv
```

Uses `thesis/config/thesis_quick.yaml` (5k samples/round, R = 3–5).

### Full thesis run

```python
!python -m thesis.run_thesis --profile full --skip-tests --fresh-csv
```

Uses `thesis/config/thesis.yaml` (100k samples/round, R = 3–10, both ciphers). Expect long runtime; download artifacts when done:

```python
from google.colab import files
for name in ["simon_round_sweep.csv", "speck_round_sweep.csv",
             "simon_vs_classical.png", "speck_vs_classical.png"]:
    path = f"results/thesis/{name}"
    if os.path.exists(path):
        files.download(path)
```

### Artifact paths

| Output | Path |
|--------|------|
| Neural CSV | `results/thesis/{cipher}_round_sweep.csv` |
| Checkpoints | `models/thesis/{cipher}_R{R}.pt` |
| Comparison plot | `results/thesis/{cipher}_vs_classical.png` |

`compare` requires `round_sweep.csv` first — run sweep before compare, or use `run_thesis` for both steps.

---

## Legacy per-folder tracks (appendix only)

The older **Simon/** and **Speck/** trees still contain fault-injection autoencoders and `experiments/run_distinguisher.py`. Those are **superseded** by `thesis/` for the diploma assignment. See [`README.md`](README.md) and [`thesis/MIGRATION.md`](thesis/MIGRATION.md).

<details>
<summary>Legacy Track A/B (autoencoder + experiments)</summary>

### Track A — One-class anomaly detection (PyTorch autoencoder)

- Script: `ml/train.py --torch-only` inside `Simon/` or `Speck/`.

### Track B — Binary neural distinguisher (per-folder experiments)

- Script: `experiments/run_distinguisher.py`.

### Track A extension — AE round sweep

- Produces `results/round_sweep/score_vs_rounds.png` under each cipher folder.

</details>

---

## Before you start (legacy sections below)

1. **Runtime:** Colab menu → **Runtime → Change runtime type → GPU** (T4 is enough).
2. **Upload the repo:** Zip `D:\BOTH` (or only `Simon` / `Speck`) and upload, or clone from Git if you host it.
3. **Disk:** Full training uses ~20k–50k samples; allow a few hundred MB for `data/` caches and `models/`.

---

## Part 1 — SIMON on Colab

### Step 1: Upload or clone

**Option A — Upload zip (simplest)**

1. On your PC, zip the `Simon` folder → `Simon.zip`.
2. In Colab:

```python
from google.colab import files
uploaded = files.upload()  # pick Simon.zip
!unzip -q Simon.zip
%cd Simon
```

**Option B — Google Drive**

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/path/to/Simon
```

**Option C — Git**

```python
!git clone https://YOUR_REMOTE/Simon.git
%cd Simon
```

Confirm you are in the repo root (you should see `simon.py`, `ml/`, `experiments/`):

```python
!ls
```

### Step 2: Install dependencies

```python
!pip install -q -r requirements-colab.txt
```

Packages: `numpy`, `pyyaml`, `scikit-learn`, `torch`, `matplotlib`.

### Step 3: Optional — verify the cipher

Quick sanity check (CPU is fine):

```python
!python -m pytest test_simon.py -q --tb=no
```

You should see official KAT tests and `test_algebraic_round_trip` pass.

### Step 4: Train the PyTorch autoencoder (Track A)

```python
import sys
from pathlib import Path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))

!python ml/train.py --torch-only --n-samples 20000 --epochs 40
```

| Flag | Meaning |
|------|---------|
| `--torch-only` | Train only the PyTorch AE (skip sklearn models; faster on Colab). |
| `--n-samples 20000` | Total labeled blocks to generate (70% train / 15% val / 15% test). |
| `--epochs 40` | Training epochs for the autoencoder. |
| `--force-regen` | Rebuild dataset cache even if `data/*.npz` exists. |

**Outputs:**

| Path | Description |
|------|-------------|
| `models/torch_autoencoder.pt` | Trained neural network |
| `models/thresholds.json` | Validation FPR-tuned threshold |
| `results/val_metrics.json` | Validation accuracy / FPR / TPR |
| `data/dataset_*.npz` | Cached ciphertext + labels (reused on reruns) |

**Faster smoke test** (few minutes):

```python
!python ml/train.py --torch-only --n-samples 2000 --epochs 10
```

### Step 5: Evaluate on the held-out test split

```python
!python ml/score.py evaluate --model torch_autoencoder
```

Writes `results/test_metrics_torch_autoencoder.json` and `results/predictions_torch_autoencoder.csv`.

### Step 6: Run full cryptanalysis experiments (Tracks A + B)

**Full assignment run** (~tens of minutes with GPU):

```python
!python experiments/run_all.py --config configs/cryptanalysis.yaml
```

**Quick run** (CI-style, a few minutes):

```python
!python experiments/run_all.py --quick
```

This uses `configs/cryptanalysis_quick.yaml` (fewer rounds, smaller samples).

**Skip autoencoder retrain** if Step 4 already finished:

```python
!python experiments/run_all.py --config configs/cryptanalysis.yaml --skip-train
```

### Step 7: View figures in the notebook

```python
from IPython.display import Image, display
display(Image("results/round_sweep/score_vs_rounds.png"))
display(Image("results/distinguisher/advantage_vs_rounds.png"))
```

Optional: open the generated report:

```python
from IPython.display import Markdown
display(Markdown(Path("results/ASSIGNMENT_SUMMARY.md").read_text()))
```

### Step 8: Download artifacts

```python
from google.colab import files
files.download("models/torch_autoencoder.pt")
files.download("results/round_sweep/score_vs_rounds.png")
```

Or zip everything:

```python
!zip -r simon_colab_results.zip models results data
files.download("simon_colab_results.zip")
```

### Step 9 (optional) — Score your own hex captures

One block per line (`6565 6877` or `65656877`):

```python
!python ml/score.py score-file my_capture.hex --format hex --model torch_autoencoder
```

---

## Part 2 — SPECK on Colab

The Speck tree is **standalone** (no imports from `Simon/`). Repeat the same workflow in a **new Colab session** or after switching directories.

### Step 1: Get the Speck folder

```python
# After uploading Speck.zip:
!unzip -q Speck.zip
%cd Speck
!ls   # expect speck.py, ml/, experiments/
```

### Step 2: Install dependencies

```python
!pip install -q -r requirements-colab.txt
```

(This installs `requirements.txt`: numpy, pyyaml, sklearn, torch, matplotlib, pytest.)

### Step 3: Optional — verify the cipher

```python
!python -m pytest test_speck.py -q --tb=no
```

### Step 4: Train the PyTorch autoencoder

```python
import sys
from pathlib import Path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))

!python ml/train.py --torch-only --n-samples 20000 --epochs 40
```

Speck defaults differ slightly (22 full rounds, wrong-round faults at 6 and 11) — see `configs/default.yaml`.

### Step 5: Evaluate

```python
!python ml/score.py evaluate --model torch_autoencoder
```

### Step 6: Cryptanalysis pipeline

```python
!python experiments/run_all.py --config configs/cryptanalysis.yaml
# or: !python experiments/run_all.py --quick
```

Speck round lists in `configs/cryptanalysis.yaml` go up to **22** (not 32).

### Step 7: Display results

```python
from IPython.display import Image, display
display(Image("results/round_sweep/score_vs_rounds.png"))
display(Image("results/distinguisher/advantage_vs_rounds.png"))
```

---

## Part 3 — One Colab notebook for both ciphers

If you uploaded the full `BOTH` workspace:

```python
# === SIMON ===
%cd /content/BOTH/Simon
!pip install -q -r requirements-colab.txt
!python ml/train.py --torch-only --n-samples 20000 --epochs 40
!python experiments/run_all.py --config configs/cryptanalysis.yaml --skip-train

# === SPECK ===
%cd /content/BOTH/Speck
!pip install -q -r requirements-colab.txt
!python ml/train.py --torch-only --n-samples 20000 --epochs 40
!python experiments/run_all.py --config configs/cryptanalysis.yaml --skip-train
```

Use separate `%cd` blocks so each project keeps its own `models/`, `data/`, and `results/`.

---

## Using the bundled assignment notebook (Simon only)

`Simon/notebooks/Assignment_Colab.ipynb` mirrors the steps above:

1. `%cd Simon`
2. `pip install -r requirements-colab.txt`
3. `ml/train.py --torch-only`
4. `experiments/run_all.py`
5. Display PNGs

In Colab: **File → Upload notebook** or open from Drive.

---

## Tuning training (YAML)

Edit `configs/default.yaml` (or pass `--config my.yaml`):

```yaml
data:
  n_samples: 20000        # dataset size
  anomaly_fraction: 0.20  # fraction of fault injections
  fault_types: [random, flip, wrong_rounds]

torch_autoencoder:
  hidden_dims: [128, 64, 32]
  latent_dim: 16
  epochs: 40
  batch_size: 512
  device: auto              # uses CUDA when available
```

Cryptanalysis-specific settings: `configs/cryptanalysis.yaml` (`round_values`, `n_samples_per_round`, distinguisher `epochs`).

---

## How data is generated (no external dataset)

You do **not** need external ciphertext files for training:

1. Random keys and plaintexts are drawn.
2. `simon3264` / `speck3264` encrypt at full rounds → **normal** samples.
3. Fault injectors build **anomalies** (see `simon3264/faults.py`, `speck3264/faults.py`).
4. `ml/features.py` builds the fixed-length feature matrix.
5. Splits are stratified by label and fault type.

The distinguisher track builds **positive** = reduced-round Speck/Simon pairs, **negative** = random blocks.

---

## GPU and runtime tips

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Lower `--n-samples` or `torch_autoencoder.batch_size` in YAML |
| Training slow on CPU | Enable GPU runtime; confirm `device=cuda` in train log |
| Stale dataset after config change | Add `--force-regen` to `ml/train.py` |
| `ModuleNotFoundError: simon3264` | Wrong working directory — `%cd` into `Simon` or `Speck` root |
| Empty `models/` after run_all | Run `ml/train.py` first or omit `--skip-train` |

Check PyTorch device:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

---

## What to report in coursework

| Deliverable | Simon path | Speck path |
|-------------|------------|------------|
| Trained AE | `models/torch_autoencoder.pt` | same |
| Round sweep figure | `results/round_sweep/score_vs_rounds.png` | same |
| Distinguisher figure | `results/distinguisher/advantage_vs_rounds.png` | same |
| Metrics tables | `results/round_sweep/round_sweep.csv` | same |
| | `results/distinguisher/summary.csv` | same |
| Config used | `configs/cryptanalysis.yaml` | same |

Simon rubric details: `Simon/docs/ASSIGNMENT_ALIGNMENT.md`.

---

## Limitations (state these in reports)

- Fixed profiles only: **32-bit blocks, 64-bit keys** (not 128/256 variants in the ML layer).
- No key recovery; synthetic faults and random baselines only.
- Results depend on seed and sample size.
- Implementations are not constant-time.

---

## Quick reference — copy-paste blocks

### Simon (minimal)

```python
%cd Simon
!pip install -q -r requirements-colab.txt
!python ml/train.py --torch-only --n-samples 20000 --epochs 40
!python experiments/run_all.py --config configs/cryptanalysis.yaml --skip-train
```

### Speck (minimal)

```python
%cd Speck
!pip install -q -r requirements-colab.txt
!python ml/train.py --torch-only --n-samples 20000 --epochs 40
!python experiments/run_all.py --config configs/cryptanalysis.yaml --skip-train
```

### Verify both ciphers (optional)

```python
%cd Simon && !python -m pytest test_simon.py -q --tb=no
%cd ../Speck && !python -m pytest test_speck.py -q --tb=no
```

---

## Further reading

| Document | Content |
|----------|---------|
| `Simon/COLAB.md` | Short Simon-only cheat sheet |
| `Speck/COLAB.md` | Short Speck-only cheat sheet |
| `Simon/docs/06-ml-anomaly-pipeline.md` | Feature design, thresholds, scoring |
| `Simon/docs/16-assignment-and-colab.md` | Assignment tracks and figure checklist |
| `Simon/docs/07-experiments-cryptanalysis.md` | Round sweep & distinguisher internals |

---

*Last updated for workspace layout: `BOTH/Simon` + `BOTH/Speck`.*
