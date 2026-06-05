# Simon — SIMON 32/64 cipher & anomaly detection

Research-oriented **SIMON 32/64** implementation plus a pipeline to train **anomaly detectors** (PyTorch autoencoder, Isolation Forest) on synthetic ciphertext data and score real captures.

**Full documentation:** [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md) — production-grade multi-chapter docs (architecture, crypto, ML, experiments, CLI, ops). Legacy pointer: [DOCUMENTATION.md](DOCUMENTATION.md).

**Google Colab quick start:** [COLAB.md](COLAB.md)

```bash
pip install -r requirements-colab.txt   # or requirements.txt locally
python ml/train.py --torch-only --n-samples 20000 --epochs 40
python ml/score.py evaluate --model torch_autoencoder
```

## Assignment / cryptanalysis experiments

Full rubric mapping: [docs/ASSIGNMENT_ALIGNMENT.md](docs/ASSIGNMENT_ALIGNMENT.md)

```bash
# Full pipeline (Track A round sweep + Track B neural distinguisher + report)
python experiments/run_all.py --config configs/cryptanalysis.yaml

# Fast local verification (~few minutes)
python experiments/run_all.py --quick
```

Colab: [notebooks/Assignment_Colab.ipynb](notebooks/Assignment_Colab.ipynb)
