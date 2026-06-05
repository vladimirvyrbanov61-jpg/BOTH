# 17 — Security & Limitations

## 17.1 Cryptographic security

| Topic | Status |
|-------|--------|
| Production use | **Prohibited** — research code only |
| Constant-time implementation | **No** |
| Side-channel resistance | **No** |
| Key management | **None** — keys are random synthetic |
| Modes of operation | **ECB-style independent blocks** only |

The SIMON implementation is intended for **correctness verification** and **ML experiments**, not for protecting data in transit or at rest.

---

## 17.2 Machine learning security

| Risk | Mitigation in project |
|------|----------------------|
| Adversarial examples | Not studied; no robust training |
| Domain shift (real vs synthetic) | Documented; user must validate on real traffic |
| Model inversion | Features are statistical summaries, not secret keys |
| Overfitting to faults | Hold-out test + multiple fault types |

---

## 17.3 Cryptanalysis scope

| Claim | Valid? |
|-------|--------|
| “Neural network detects anomalies in ciphertext” | Yes (Track A, under stated assumptions) |
| “Neural network automates distinguisher” | Yes (Track B, reduced rounds) |
| “System breaks SIMON 32/64” | **No** |
| “System recovers secret keys” | **No** |

The distinguisher distinguishes **cipher pairs from random pairs** at reduced rounds — standard academic “automated cryptanalysis” framing, not a break.

---

## 17.4 Data and privacy

- All training data is **synthetically generated**.
- No PII; no network capture built-in.
- Scoring external files is **local** — no telemetry.

---

## 17.5 Dependency supply chain

Pin versions in `requirements.txt` for reproducibility. PyTorch and sklearn should be installed from trusted sources (PyPI, official Colab image).

---

## 17.6 Known functional limitations

| Limitation | Detail |
|------------|--------|
| No CBC/GCM | Cannot model chained IV scenarios |
| No padding | Raw 32-bit blocks only |
| No CSV ingest in trainer | Pre-convert to hex/NPZ |
| Single block profile | ML fixed to SIMON 32/64 |
| Per-sample encryption loops | Slow large-N generation |

---

## 17.7 Assumptions for valid results

1. Blocks are independent SIMON 32/64 outputs (or consistent synthetic faults).
2. Training “normal” class matches deployment “normal” distribution.
3. Feature extraction flags match between train and score.
4. Threshold tuned on validation synthetic normals transfers acceptably to target data.

Violating these assumptions can produce misleading detection rates without any code bug.

---

[← Assignment](16-assignment-and-colab.md) · [Next: Technical debt →](18-technical-debt-and-roadmap.md)
