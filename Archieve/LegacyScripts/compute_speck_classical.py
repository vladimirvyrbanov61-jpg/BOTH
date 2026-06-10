#!/usr/bin/env python3
"""Compute Speck classical bounds for configured rounds with a custom sample count.

Usage: run from repo root in the project's venv.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis.classical.characteristic import (
    track_characteristic_over_rounds,
    load_classical_bounds_csv,
    save_classical_bounds_csv,
)
from thesis.config.loader import load_config, config_path_for_profile


def main(n_samples: int = 200_000):
    cfg = load_config(config_path_for_profile("full"))
    results_dir = ROOT / cfg.get("results_dir", "results/thesis")
    speck_csv = results_dir / "speck_classical_bounds.csv"

    rounds = cfg.get("rounds", [3,4,5,6,7,8,9,10])
    delta = tuple(cfg.get("input_delta", [0x0001, 0x0000]))
    seed = int(cfg.get("seed", 1))
    top_k = int(cfg.get("classical", {}).get("top_k_dp", 32))

    existing = {}
    if speck_csv.exists():
        existing = load_classical_bounds_csv(speck_csv)

    missing = [r for r in rounds if r not in existing]
    if not missing:
        print("No missing Speck rounds; CSV already complete.")
        return

    print(f"Existing rounds in CSV: {sorted(existing.keys())}")
    print(f"Will compute missing Speck rounds: {missing} with n_samples={n_samples}")

    rows = track_characteristic_over_rounds(
        "speck",
        missing,
        delta,
        n_samples_row=int(n_samples),
        top_k=top_k,
        seed=seed,
    )

    print(f"Computed {len(rows)} rows; appending to {speck_csv}")
    save_classical_bounds_csv(speck_csv, rows)
    all_rows = load_classical_bounds_csv(speck_csv)
    print(f"After append, rounds present: {sorted(all_rows.keys())}")

    # regenerate the speck plot
    from thesis.eval.compare import run_compare, resolve_config_path

    cfg_path = resolve_config_path(profile="full")
    print("Regenerating Speck comparison plot (will use existing CSV)...")
    run_compare(cfg_path, force_classical=False, ciphers=["speck"])
    print("Done.")


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=200000)
    args = p.parse_args()
    main(args.n_samples)
