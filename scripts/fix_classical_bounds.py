#!/usr/bin/env python3
"""Compute and append missing classical bounds (Simon) and regenerate plots.
Run from repo root using the project's venv Python.
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


def main():
    cfg = load_config(config_path_for_profile("full"))
    results_dir = ROOT / cfg.get("results_dir", "results/thesis")
    simon_csv = results_dir / "simon_classical_bounds.csv"

    desired_rounds = cfg.get("rounds", [3,4,5,6,7,8,9,10])
    existing = {}
    if simon_csv.exists():
        existing = load_classical_bounds_csv(simon_csv)
    missing = [r for r in desired_rounds if r not in existing]
    if not missing:
        print("No missing rounds. CSV already complete.")
        return

    print(f"Existing rounds in CSV: {sorted(existing.keys())}")
    print(f"Will compute missing rounds: {missing}")

    rows = track_characteristic_over_rounds(
        "simon",
        missing,
        tuple(cfg.get("input_delta", [0x0001, 0x0000])),
        n_samples_row=int(cfg.get("classical", {}).get("n_samples_simon", 250000)),
        top_k=int(cfg.get("classical", {}).get("top_k_dp", 32)),
        seed=int(cfg.get("seed", 1)),
    )

    print(f"Computed {len(rows)} rows; appending to {simon_csv}")
    save_classical_bounds_csv(simon_csv, rows)
    all_rows = load_classical_bounds_csv(simon_csv)
    print(f"After append, rounds present: {sorted(all_rows.keys())}")

    # Regenerate plots using compare module
    from thesis.eval.compare import run_compare, resolve_config_path

    cfg_path = resolve_config_path(profile="full")
    print("Regenerating comparison plots...")
    run_compare(cfg_path, force_classical=False)
    print("Done.")


if __name__ == '__main__':
    main()
