# Appendix A — Glossary

| Term | Definition |
|------|------------|
| **SIMON 32/64** | SIMON with 32-bit blocks and 64-bit keys; `n=16`, `m=4`, 32 rounds |
| **Block** | Two `uint16` words `(left, right)` representing one ciphertext or plaintext |
| **Word** | 16-bit half of a block |
| **Normal (label 0)** | Valid full-round SIMON ciphertext in the ML pipeline |
| **Anomaly (label 1)** | Synthetic fault or non-cipher sample |
| **Fault** | Named injection type in `meta["fault"]` (e.g. `wrong_rounds_16`) |
| **Feature vector** | Fixed-length float row fed to ML models (default 42-D) |
| **One-class training** | Fitting only on normal rows; anomalies at test time |
| **Reconstruction error** | MSE between AE input and output — used as anomaly score |
| **Threshold** | Score cutoff; tuned for target FPR on validation normals |
| **FPR** | False positive rate — flagging a normal as anomaly |
| **TPR** | True positive rate — detecting real Simon pairs (distinguisher) |
| **TNR** | True negative rate — detecting random pairs (distinguisher) |
| **Advantage** | `|accuracy − 0.5|` for distinguisher |
| **Δ (delta)** | Input word-wise XOR difference for distinguisher pairs |
| **Distinguisher** | Classifier separating cipher-structured pairs from random |
| **Track A** | Round sweep using TorchAutoencoder |
| **Track B** | Neural distinguisher per round count |
| **KAT** | Known-answer test vector from SIMON spec |
| **NPZ** | NumPy compressed archive used for caches |
| **ECB-style** | Independent block encryption without chaining |

[← Technical debt](../18-technical-debt-and-roadmap.md) · [Appendix B →](B-file-formats.md)
