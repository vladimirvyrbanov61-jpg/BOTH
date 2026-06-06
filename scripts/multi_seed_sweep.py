"""Orchestrator: run round-sweep experiments across multiple random seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]


def run_seed_sweep(
    seeds: list[int],
    profile: str = "full",
    config: Optional[Path] = None,
    ciphers: Optional[list[str]] = None,
    model_base_dir: Optional[Path] = None,
    log_base_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    skip_tests: bool = False,
    force_regen: bool = False,
    fresh_csv_first_seed: bool = True,
) -> None:
    """Run round-sweep for each seed sequentially.
    
    Parameters
    ----------
    seeds : list[int]
        List of random seeds to sweep over.
    profile : str
        Config profile: 'full' or 'quick'.
    config : Path, optional
        Explicit config path (overrides profile).
    ciphers : list[str], optional
        Restrict to cipher(s), e.g., ['simon', 'speck'].
    model_base_dir : Path, optional
        Base directory for checkpoints (per-seed subdirs created).
    log_base_dir : Path, optional
        Base directory for TensorBoard logs.
    results_dir : Path, optional
        Override results directory for CSVs.
    skip_tests : bool
        Skip pytest gate.
    force_regen : bool
        Regenerate cached datasets.
    fresh_csv_first_seed : bool
        Delete multi_seed_raw.csv before first seed run (append for subsequent seeds).
    """
    if results_dir is None:
        results_dir = _REPO_ROOT / "results" / "thesis" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_type": "multi_seed_sweep",
        "profile": profile,
        "config": str(config) if config is not None else None,
        "ciphers": ciphers or ["simon", "speck"],
        "seeds": seeds,
        "results_dir": str(results_dir),
        "created_at": datetime.now().isoformat(),
    }
    with open(results_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    for i, seed in enumerate(seeds):
        print(f"\n{'='*70}")
        print(f"[SEED {i+1}/{len(seeds)}] Running seed={seed}")
        print(f"{'='*70}\n")

        # Build command
        cmd = [sys.executable, "-m", "thesis.eval.round_sweep", "--profile", profile]

        if config is not None:
            cmd.extend(["--config", str(config)])

        if ciphers:
            for c in ciphers:
                cmd.extend(["--cipher", c])

        if model_base_dir is not None:
            cmd.extend(["--model-dir", str(model_base_dir)])

        if log_base_dir is not None:
            cmd.extend(["--log-dir", str(log_base_dir)])

        if results_dir is not None:
            cmd.extend(["--results-dir", str(results_dir), "--no-timestamped-dir"])

        # Add seed
        cmd.extend(["--seed", str(seed)])

        # Fresh CSV only on first seed
        if i == 0 and fresh_csv_first_seed:
            cmd.append("--fresh-csv")

        if force_regen:
            cmd.append("--force-regen")

        print(f"[SEED {seed}] Command: {' '.join(cmd)}\n")

        # Run
        ret = subprocess.call(cmd, cwd=str(_REPO_ROOT))
        if ret != 0:
            print(f"\n[ERROR] Seed {seed} failed with code {ret}. Stopping sweep.")
            sys.exit(ret)

        print(f"\n[SEED {seed}] Completed successfully.\n")

    print(f"\n{'='*70}")
    print(f"[SUCCESS] All {len(seeds)} seeds completed.")
    print(f"Results CSV: {results_dir}/{{cipher}}_multi_seed_raw.csv")
    print(f"TensorBoard: tensorboard --logdir {log_base_dir or 'runs/thesis'}")
    print(f"Manifest: {results_dir / 'manifest.json'}")
    print(f"{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed sweep orchestrator for thesis neural distinguisher"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="List of seeds to sweep (default: 1-10)",
    )
    parser.add_argument(
        "--profile",
        choices=["full", "quick"],
        default="full",
        help="Config profile (default: full)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Explicit config path (overrides --profile)",
    )
    parser.add_argument(
        "--ciphers",
        nargs="+",
        choices=["simon", "speck"],
        default=None,
        help="Restrict to cipher(s) (default: all)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Base directory for model checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Base directory for TensorBoard logs (default: runs/thesis)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for result CSVs",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest gate",
    )
    parser.add_argument(
        "--force-regen",
        action="store_true",
        help="Regenerate cached datasets",
    )
    parser.add_argument(
        "--fresh-csv",
        action="store_true",
        help="Delete multi_seed_raw.csv before first seed run",
    )
    args = parser.parse_args()

    log_dir = args.log_dir or (_REPO_ROOT / "runs" / "thesis")

    run_seed_sweep(
        seeds=args.seeds,
        profile=args.profile,
        config=args.config,
        ciphers=args.ciphers,
        model_base_dir=args.model_dir,
        log_base_dir=log_dir,
        results_dir=args.results_dir,
        skip_tests=args.skip_tests,
        force_regen=args.force_regen,
        fresh_csv_first_seed=args.fresh_csv,
    )


if __name__ == "__main__":
    main()
