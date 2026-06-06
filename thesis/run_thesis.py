"""Unified thesis pipeline: optional tests → round sweep → AI vs classical comparison."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from thesis.config.loader import config_path_for_profile
from thesis.eval.compare import run_compare
from thesis.eval.round_sweep import run_round_sweep

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_pytest(test_paths: list[str], *, verbose: bool) -> int:
    cmd = [sys.executable, "-m", "pytest", *test_paths]
    if verbose:
        cmd.append("-v")
    print("[run_thesis] pytest:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run thesis pipeline (round sweep + classical comparison)"
    )
    parser.add_argument(
        "--profile",
        choices=["full", "quick"],
        default="quick",
        help="Config profile: thesis_quick.yaml (quick) or thesis.yaml (full)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Explicit config path (overrides --profile)",
    )
    parser.add_argument(
        "--cipher",
        action="append",
        choices=["simon", "speck"],
        help="Restrict to cipher(s)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        nargs="+",
        help="Override round list",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override samples per round",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest gate",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Run round sweep only (no DDT plots)",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Run compare only (requires existing neural CSV)",
    )
    parser.add_argument(
        "--force-regen",
        action="store_true",
        help="Regenerate cached datasets",
    )
    parser.add_argument(
        "--fresh-csv",
        action="store_true",
        help="Replace round_sweep.csv files before sweep",
    )
    parser.add_argument(
        "--force-classical",
        action="store_true",
        help="Recompute classical bounds",
    )
    parser.add_argument(
        "--pytest-verbose",
        action="store_true",
        help="Verbose pytest output",
    )
    args = parser.parse_args()

    cfg_path = args.config or config_path_for_profile(args.profile)
    print(f"[run_thesis] config: {cfg_path}")

    if not args.skip_tests:
        test_files = [
            "tests/test_thesis_encoding.py",
            "tests/test_cnn.py",
            "tests/test_classical.py",
        ]
        rc = _run_pytest(test_files, verbose=args.pytest_verbose)
        if rc != 0:
            print("[run_thesis] pytest failed; aborting.", file=sys.stderr)
            return rc

    results_path = None
    if not args.skip_sweep:
        print("[run_thesis] starting round sweep …")
        _, results_path = run_round_sweep(
            cfg_path,
            ciphers=args.cipher,
            rounds_list=args.rounds,
            n_samples=args.n_samples,
            force_regen=args.force_regen,
            fresh_csv=args.fresh_csv,
        )

    if not args.skip_compare:
        print("[run_thesis] starting AI vs classical comparison …")
        run_compare(
            cfg_path,
            ciphers=args.cipher,
            force_classical=args.force_classical,
            results_dir=results_path,
        )

    print("[run_thesis] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
