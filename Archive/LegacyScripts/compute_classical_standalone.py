#!/usr/bin/env python3
"""Compute classical bounds for Simon/Speck and generate complete comparison."""

from pathlib import Path
from thesis.classical.characteristic import (
    track_characteristic_over_rounds,
    save_classical_bounds_csv,
    load_classical_bounds_csv,
)
from thesis.data.generator import DEFAULT_INPUT_DELTA
from thesis.config.loader import load_config

def main():
    # Load config
    config = load_config(Path("thesis/config/thesis_quick.yaml"))
    rounds = config.get("rounds", [3, 4, 5, 6, 7, 8, 9, 10])
    delta = tuple(config.get("input_delta", list(DEFAULT_INPUT_DELTA)))
    seed = int(config.get("seed", 1))
    top_k = int(config.get("classical", {}).get("top_k_dp", 32))
    
    results_dir = Path("results/thesis")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute for both ciphers
    for cipher in ["simon", "speck"]:
        print(f"\n[classical] Computing {cipher.upper()}...", flush=True)
        bounds_path = results_dir / f"{cipher}_classical_bounds.csv"
        
        # Delete old file
        if bounds_path.exists():
            bounds_path.unlink()
            print(f"[classical] Deleted old {cipher} bounds", flush=True)
        
        # Get sampling parameters
        n_samples = config.get("classical", {}).get("n_samples_simon", 250000) if cipher == "simon" else config.get("classical", {}).get("n_samples_speck", 1000000)
        print(f"[classical] Using n_samples={n_samples} for {cipher}", flush=True)
        
        # Compute
        print(f"[classical] Computing {len(rounds)} rounds for {cipher}...", flush=True)
        rows = track_characteristic_over_rounds(
            cipher,
            rounds,
            delta,
            n_samples_row=int(n_samples),
            top_k=top_k,
            seed=seed,
        )
        
        # Save
        print(f"[classical] Saving {len(rows)} rows to {bounds_path}", flush=True)
        save_classical_bounds_csv(bounds_path, rows)
        
        # Verify
        loaded = load_classical_bounds_csv(bounds_path)
        print(f"[classical] Verified {len(loaded)} rounds in output: {sorted(loaded.keys())}", flush=True)
    
    print("\n[classical] COMPLETE", flush=True)

if __name__ == "__main__":
    main()
