"""Orchestrator: run round-sweep experiments across multiple random seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from thesis.config.loader import config_path_for_profile, load_config
from thesis.eval.compare import run_compare
from thesis.eval.manifest import artifact_inventory, build_manifest, utc_now, write_manifest
from thesis.eval.plot_results import plot_all

_TEST_FILES = [
    "tests/test_cipher_kats.py",
    "tests/test_thesis_encoding.py",
    "tests/test_cnn.py",
    "tests/test_classical.py",
    "tests/test_round_sweep_smoke.py",
    "tests/test_aggregate.py",
    "tests/test_compare_experiments.py",
    "tests/test_config_validation.py",
]


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
    config_path = config or config_path_for_profile(profile)
    resolved_config = load_config(config_path)
    if results_dir is None:
        configured_results = Path(resolved_config.get("results_dir", "results/thesis"))
        if not configured_results.is_absolute():
            configured_results = _REPO_ROOT / configured_results
        results_dir = configured_results / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}"
    results_dir.mkdir(parents=True, exist_ok=True)
    cipher_names = ciphers or resolved_config.get("ciphers") or ["simon", "speck"]
    manifest_path = results_dir / "manifest.json"
    manifest = build_manifest(
        repo_root=_REPO_ROOT,
        run_type="multi_seed_sweep",
        config_path=config_path,
        config=resolved_config,
        parameters={
            "profile": profile,
            "seeds": seeds,
            "ciphers": cipher_names,
            "rounds": resolved_config.get("rounds"),
            "n_samples_per_round": resolved_config.get("n_samples_per_round"),
            "input_delta": resolved_config.get("input_delta"),
            "training": resolved_config.get("training", {}),
            "force_regen": force_regen,
            "fresh_csv_first_seed": fresh_csv_first_seed,
            "test_gate": not skip_tests,
        },
        paths={
            "results_dir": results_dir,
            "model_dir": model_base_dir or resolved_config.get("model_dir"),
            "log_dir": log_base_dir,
            "data_dir": resolved_config.get("data_dir"),
        },
    )
    write_manifest(manifest_path, manifest)

    if not skip_tests:
        test_cmd = [sys.executable, "-m", "pytest", *_TEST_FILES]
        print(f"[TEST GATE] {' '.join(test_cmd)}")
        ret = subprocess.call(test_cmd, cwd=str(_REPO_ROOT))
        if ret != 0:
            manifest["status"] = "failed"
            manifest["failure"] = {"stage": "test_gate", "return_code": ret}
            manifest["completed_at"] = utc_now()
            write_manifest(manifest_path, manifest)
            raise SystemExit(ret)

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
            manifest["status"] = "failed"
            manifest["progress"]["failed_seeds"].append(seed)
            manifest["failure"] = {"stage": "round_sweep", "seed": seed, "return_code": ret}
            manifest["completed_at"] = utc_now()
            manifest["artifacts"] = artifact_inventory(results_dir)
            write_manifest(manifest_path, manifest)
            print(f"\n[ERROR] Seed {seed} failed with code {ret}. Stopping sweep.")
            raise SystemExit(ret)

        manifest["progress"]["completed_seeds"].append(seed)
        manifest["artifacts"] = artifact_inventory(results_dir)
        write_manifest(manifest_path, manifest)
        print(f"\n[SEED {seed}] Completed successfully.\n")

    try:
        plot_all(results_dir, list(cipher_names))
        run_compare(config_path, ciphers=list(cipher_names), results_dir=results_dir)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["failure"] = {"stage": "classical_comparison", "reason": "keyboard_interrupt"}
        manifest["completed_at"] = utc_now()
        manifest["artifacts"] = artifact_inventory(results_dir)
        write_manifest(manifest_path, manifest)
        raise
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {
            "stage": "plot_or_classical_comparison",
            "error_type": type(exc).__name__,
            "reason": str(exc),
        }
        manifest["completed_at"] = utc_now()
        manifest["artifacts"] = artifact_inventory(results_dir)
        write_manifest(manifest_path, manifest)
        raise
    manifest["status"] = "completed"
    manifest["completed_at"] = utc_now()
    manifest["artifacts"] = artifact_inventory(results_dir)
    write_manifest(manifest_path, manifest)

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
