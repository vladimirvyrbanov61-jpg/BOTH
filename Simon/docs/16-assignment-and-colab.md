# 16 — Assignment & Colab

## 16.1 Assignment tracks summary

| Track | Script | ML paradigm | Delivers |
|-------|--------|-------------|----------|
| **A** | `experiments/round_sweep.py` | One-class AE anomaly scores vs rounds | `score_vs_rounds.png`, `round_sweep.csv` |
| **B** | `experiments/run_distinguisher.py` | Binary distinguisher | `advantage_vs_rounds.png`, `summary.csv` |

Full rubric mapping: [`ASSIGNMENT_ALIGNMENT.md`](ASSIGNMENT_ALIGNMENT.md).

---

## 16.2 Examiner one-paragraph summary

> We implement SIMON 32/64 and apply **deep learning** in two complementary ways: (1) a **PyTorch autoencoder** trained on statistical features of genuine 32-round ciphertext to detect anomalies, evaluated over **multiple round counts**; (2) a **neural distinguisher** that automates classification of **reduced-round** Simon ciphertext pairs versus random pairs, with accuracy/advantage measured **across rounds**. Both use the same cipher oracle without altering the core library.

---

## 16.3 Colab workflow

### Notebook: `notebooks/Assignment_Colab.ipynb`

| Cell | Action |
|------|--------|
| 1 | Intro / clone instructions |
| 2 | `%cd Simon` |
| 3 | `pip install -r requirements-colab.txt` |
| 4 | `sys.path` setup |
| 5 | `python ml/train.py --torch-only --n-samples 20000 --epochs 40` |
| 6 | `python experiments/run_all.py --config configs/cryptanalysis.yaml` |
| 7 | Display PNG figures |
| 8 | Render `ASSIGNMENT_SUMMARY.md` |

### Short checklist: `COLAB.md`

Minimal copy-paste commands for train + evaluate + assignment pointer.

---

## 16.4 Report figure checklist

| Figure | Path | Caption hint |
|--------|------|--------------|
| Fig. 1 | `results/round_sweep/score_vs_rounds.png` | AE error vs round count |
| Fig. 2 | `results/distinguisher/advantage_vs_rounds.png` | Distinguisher advantage vs rounds |
| Table 1 | `results/round_sweep/round_sweep.csv` | Track A metrics |
| Table 2 | `results/distinguisher/summary.csv` | Track B acc/AUC/TPR/TNR |

---

## 16.5 Interpreting results for written reports

### Track A

- **Rising mean_score** as rounds decrease → truncated outputs less like 32-round training distribution.
- **High detection_rate** for `r < 32` → threshold flags truncated ciphertext.
- **High detection_rate at r = 32** → specificity (correctly accept normals).

### Track B

- **High advantage** at low `r` → network automates Simon-vs-random separation.
- **Advantage → 0** as `r` increases → expected; not a failure.

### Limitations to state explicitly

- No key recovery
- Synthetic / random baselines only
- Results are seed and sample-size dependent

---

## 16.6 Commands for submission package

```bash
python ml/train.py --torch-only --n-samples 20000 --epochs 40
python experiments/run_all.py --config configs/cryptanalysis.yaml
pytest test_simon.py test_experiments.py -q
```

Include in report appendix: `configs/cryptanalysis.yaml` (or diff), commit hash, Python version.

---

[← Operations](15-operations-and-observability.md) · [Next: Security →](17-security-limitations.md)
