# Assignment alignment — SIMON 32/64 cryptanalysis experiments

This document maps the project to typical automated-cryptanalysis assignment criteria and provides a report outline.

---

## 1. Assignment criteria mapping

| Criterion | Code location | Output artifact |
|-----------|---------------|-----------------|
| SIMON 32/64 implementation | `simon.py`, `simon3264/cipher.py` | `test_simon.py`, `test_simon3264.py` |
| Deep learning on ciphertext | `ml/models.py` (`TorchAutoencoder`) | `models/torch_autoencoder.pt` |
| Statistical anomaly detection | `ml/train.py`, `ml/features.py` (42 features) | `models/thresholds.json`, Track A |
| Automate cryptanalysis | `experiments/distinguisher_*.py` | `results/distinguisher/summary.csv` |
| Varying number of rounds | `experiments/round_sweep.py`, `run_distinguisher.py` | PNG + CSV under `results/` |

---

## 2. SIMON 32/64 implementation

- **Spec reference:** Simon 32/64 with 32 rounds, 64-bit block, 64-bit key (`simon3264/cipher.py`).
- **Core API:** `Simon3264.encrypt`, `encrypt_rounds(plaintext, key, r)` for reduced-round study.
- **Validation:** Known-answer tests in `test_simon.py` and round-truncation checks in `test_simon3264.py`.

---

## 3. Statistical anomaly detection (deep learning)

**Training (unchanged primary pipeline):**

1. Sample random keys/plaintexts; encrypt with full 32 rounds.
2. Extract 42-dimensional statistical features (`ml/features.py`).
3. Train `TorchAutoencoder` on **normal-only** validation split (`python ml/train.py --torch-only`).
4. Threshold from validation normal scores at target FPR (`models/thresholds.json`).

**Anomalies tested in default pipeline:** synthetic faults (wrong rounds, wrong Z, etc.) via `simon3264/faults.py`.

---

## 4. Automated cryptanalysis — neural distinguisher (Track B)

At fixed reduced round count `r`:

1. Choose input difference Δ (default word XOR: left word `0x0001`, right `0x0000`).
2. For each sample: random key `k`, plaintext `P`, `P' = P ⊕ Δ`.
3. **Class 1 (real):** `(Enc_r(P,k), Enc_r(P',k))`.
4. **Class 0 (random):** independent random 32-bit block pairs.
5. Features: default `xor_bits` = 32 bits of `C ⊕ C'`.
6. Train small MLP; report accuracy, AUC, advantage `|acc − 0.5|`.

**Interpretation:** High advantage at low `r` means the network automates distinguishing reduced-round Simon from random structure. This is a **distinguisher** study, not full key recovery.

---

## 5. Varying rounds

### Track A — Autoencoder round sweep

Same AE trained on 32-round normal features; for each `r` in `configs/cryptanalysis.yaml`:

- Score `r`-round ciphertext (label: truncated vs full).
- Compare scores to random-block baseline; record detection rate and AUC.

**Expected narrative:** Mean reconstruction error increases as `r` decreases; truncated-round outputs appear anomalous.

### Track B — Distinguisher round sweep

Train/evaluate a separate classifier per `r`; plot `advantage_vs_rounds.png`.

**Expected narrative:** Advantage decreases as `r` approaches 32.

---

## 6. Limitations

- No 32-round key recovery or slide attacks.
- Random baselines are structural null models, not other ciphers.
- Feature/threshold choices affect absolute numbers; trends matter for the report.

---

## 7. Reproduction commands

```bash
# From repository root
pip install -r requirements-colab.txt   # or requirements.txt (includes torch)

python ml/train.py --torch-only --n-samples 20000 --epochs 40

python experiments/run_all.py --config configs/cryptanalysis.yaml

# Fast smoke run (smaller samples, fewer rounds):
python experiments/run_all.py --quick

# Or tracks individually:
python experiments/round_sweep.py --config configs/cryptanalysis.yaml
python experiments/run_distinguisher.py --config configs/cryptanalysis.yaml
```

**Track A `detection_rate`:** for `r < 32`, fraction of truncated-round ciphertext flagged as anomalous (high is good); at `r = 32`, fraction correctly left unflagged (specificity, high is good).

**Track B metrics:** accuracy, AUC, advantage, plus **TPR** (real Simon pairs classified as real) and **TNR** (random pairs classified as random).

Outputs:

- `results/round_sweep/round_sweep.csv`, `score_vs_rounds.png`
- `results/distinguisher/summary.csv`, `advantage_vs_rounds.png`
- `results/ASSIGNMENT_SUMMARY.md`

---

## 8. Suggested report figures

| Figure | File | Caption suggestion |
|--------|------|-------------------|
| Fig. 1 | `score_vs_rounds.png` | PyTorch autoencoder reconstruction error vs encryption round count; trained on 32-round normal ciphertext. |
| Fig. 2 | `advantage_vs_rounds.png` | Neural distinguisher accuracy/advantage vs round count for Simon pair vs random pair classification. |
| Table 1 | `round_sweep.csv` | Detection metrics per round count (Track A). |
| Table 2 | `distinguisher/summary.csv` | Distinguisher test accuracy/AUC per round (Track B). |

---

## 9. Suggested report structure (template)

1. **Abstract** — Two DL tracks on SIMON 32/64: anomaly AE + neural distinguisher; round sweeps.
2. **Introduction** — Lightweight block ciphers; ML in cryptanalysis.
3. **Background** — SIMON structure; reduced-round cryptanalysis; distinguishers vs key recovery.
4. **Implementation** — Pointer to `simon.py` / `simon3264`; feature design.
5. **Method** — Track A one-class AE; Track B pair-based classifier; Δ choice.
6. **Experiments** — Config (`cryptanalysis.yaml`), sample sizes, hardware (Colab GPU optional).
7. **Results** — Embed figures/tables; discuss trends.
8. **Conclusion** — Criteria met; limitations; future work (more rounds, other Δ, joint training).
9. **References** — SIMON spec, Gohr-style neural cryptanalysis papers.

---

## One-paragraph examiner summary

> We implement SIMON 32/64 and apply **deep learning** in two complementary ways: (1) a **PyTorch autoencoder** trained on statistical features of genuine 32-round ciphertext to detect anomalies, evaluated over **multiple round counts**; (2) a **neural distinguisher** that automates classification of **reduced-round** Simon ciphertext pairs versus random pairs, with accuracy/advantage measured **across rounds**. Both use the same cipher oracle without altering the core library.
