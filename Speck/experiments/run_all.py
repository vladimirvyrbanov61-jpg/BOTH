#!/usr/bin/env python3
"""Run full SPECK 32/64 cryptanalysis pipeline: train AE + both tracks + report."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.config import load_cryptanalysis_config
from experiments.reporting import generate_assignment_report
from experiments.round_sweep import run_round_sweep
from experiments.run_distinguisher import run_distinguisher_experiment


def ensure_torch_model(model_dir: Path, *, quick: bool = False) -> None:
    model_path = model_dir / "torch_autoencoder.pt"
    if model_path.exists():
        print(f"[run_all] Found {model_path}")
        return
    print("[run_all] Training TorchAutoencoder via ml/train.py --torch-only ...")
    cmd = [sys.executable, str(_REPO / "ml" / "train.py"), "--torch-only"]
    if quick:
        cmd.extend(["--n-samples", "2000", "--epochs", "10"])
    subprocess.run(cmd, cwd=str(_REPO), check=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run Speck cryptanalysis experiments")
    parser.add_argument("--config", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-round-sweep", action="store_true")
    parser.add_argument("--skip-distinguisher", action="store_true")
    parser.add_argument("--force-regen", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config
    if args.quick and config_path is None:
        config_path = _REPO / "configs" / "cryptanalysis_quick.yaml"
    cfg = load_cryptanalysis_config(config_path)
    model_dir = Path(cfg.paths.model_dir)

    if not args.skip_train:
        ensure_torch_model(model_dir, quick=args.quick)

    if not args.skip_round_sweep:
        run_round_sweep(cfg)

    if not args.skip_distinguisher:
        run_distinguisher_experiment(cfg, force_regen=args.force_regen)

    report_path = Path(cfg.paths.results_dir) / "ASSIGNMENT_SUMMARY.md"
    text = generate_assignment_report(results_root=Path(cfg.paths.results_dir))
    print(f"\n[run_all] Report written to {report_path}")
    print(text[:800], "..." if len(text) > 800 else "")


if __name__ == "__main__":
    main()
