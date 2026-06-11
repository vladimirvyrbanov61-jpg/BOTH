#!/usr/bin/env python3
"""Verify assignment layer files and imports (no training required)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

REQUIRED = [
    "simon.py",
    "configs/cryptanalysis.yaml",
    "configs/cryptanalysis_quick.yaml",
    "experiments/config.py",
    "experiments/round_sweep.py",
    "experiments/distinguisher_data.py",
    "experiments/distinguisher_model.py",
    "experiments/run_distinguisher.py",
    "experiments/reporting.py",
    "experiments/run_all.py",
    "docs/ASSIGNMENT_ALIGNMENT.md",
    "notebooks/Assignment_Colab.ipynb",
    "test_experiments.py",
]


def main() -> int:
    missing = [p for p in REQUIRED if not (_REPO / p).exists()]
    if missing:
        print("Missing files:")
        for p in missing:
            print(f"  - {p}")
        return 1

    from experiments.config import load_cryptanalysis_config

    load_cryptanalysis_config(_REPO / "configs" / "cryptanalysis.yaml")
    load_cryptanalysis_config(_REPO / "configs" / "cryptanalysis_quick.yaml")

    try:
        import torch  # noqa: F401

        print("PyTorch: installed (full pipeline can run)")
    except ImportError:
        print("PyTorch: not installed — use pip install -r requirements.txt or Colab")

    model = _REPO / "models" / "torch_autoencoder.pt"
    if model.exists():
        print(f"Base model: {model}")
    else:
        print("Base model: not trained yet — run: python ml/train.py --torch-only")

    print("Assignment layer structure: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
