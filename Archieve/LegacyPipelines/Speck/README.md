# SPECK 32/64 — ML cryptanalysis (standalone)

Self-contained project for **SPECK 32/64** (22 rounds, α=7, β=2): cipher implementation, synthetic datasets, anomaly-detection training, and neural distinguisher experiments.

This tree is independent of the sibling [`../Simon`](../Simon) project. Copy this folder to a second machine for Speck-only training without Simon code or shared imports.

## Layout

| Path | Purpose |
|------|---------|
| `speck.py` | SPECK family primitive (encrypt/decrypt, key schedule) |
| `speck3264/` | Frozen 32/64 profile, encoding, dataset, faults |
| `ml/` | Autoencoder / isolation forest training on ciphertext features |
| `experiments/` | Round sweep + neural distinguisher pipelines |
| `configs/` | `default.yaml`, `cryptanalysis.yaml` |

## Setup

```powershell
cd Speck
py -m pip install -r requirements.txt
py -m pytest test_speck.py test_speck3264.py test_experiments.py -q
```

## Train anomaly models (Track A base)

```powershell
py ml/train.py --torch-only
```

## Neural distinguisher (Track B)

```powershell
py experiments/run_distinguisher.py
```

## Full pipeline

```powershell
py experiments/run_all.py
```

## Parameters

- Block: 32 bits (two 16-bit words)
- Key: 64 bits (four words)
- Full rounds: **22**

Research / education only — not for production cryptography.
