#!/usr/bin/env python3
"""Verify project files and imports (no training required)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

REQUIRED = [
    "speck.py",
    "speck3264/cipher.py",
    "configs/cryptanalysis.yaml",
    "configs/cryptanalysis_quick.yaml",
    "experiments/config.py",
    "experiments/round_sweep.py",
    "experiments/distinguisher_data.py",
    "experiments/distinguisher_model.py",
    "experiments/run_distinguisher.py",
    "experiments/run_all.py",
    "ml/train.py",
    "test_speck.py",
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
    from speck3264 import Speck3264

    load_cryptanalysis_config(_REPO / "configs" / "cryptanalysis.yaml")
    _ = Speck3264()

    try:
        import torch  # noqa: F401

        print("PyTorch: installed")
    except ImportError:
        print("PyTorch: not installed — pip install -r requirements.txt")

    model = _REPO / "models" / "torch_autoencoder.pt"
    if model.exists():
        print(f"Base model: {model}")
    else:
        print("Base model: not trained — run: py ml/train.py --torch-only")

    print("Speck project structure: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
