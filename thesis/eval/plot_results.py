"""Plot aggregated multi-seed results and classical DDT bounds.

Creates one PNG per metric (accuracy, auc_roc, tpr, tnr, advantage) for each cipher
from the `{cipher}_multi_seed_raw.csv` files. Also creates DDT plots from
`{cipher}_classical_bounds.csv`.

Outputs are written to `thesis/eval/results/`.
"""
from __future__ import annotations

from pathlib import Path
import math
import csv
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = _REPO_ROOT / "thesis" / "eval" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = _REPO_ROOT / "results" / "thesis"

METRICS = ["accuracy", "auc_roc", "tpr", "tnr", "advantage"]


def load_multi_seed_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def aggregate_by_round(df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    out = {}
    if df.empty:
        return out
    # Group by rounds and compute mean and std for metrics
    g = df.groupby("rounds")
    for r, grp in g:
        out[r] = {}
        for m in METRICS:
            if m in grp.columns:
                vals = grp[m].dropna().astype(float)
                out[r][f"{m}_mean"] = float(vals.mean()) if len(vals) else float('nan')
                out[r][f"{m}_std"] = float(vals.std(ddof=0)) if len(vals) else float('nan')
            else:
                out[r][f"{m}_mean"] = float('nan')
                out[r][f"{m}_std"] = float('nan')
    return out


def plot_metric_rounds(rounds: List[int], means: List[float], stds: List[float], title: str, out_path: Path, ylabel: str):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    rounds = list(rounds)
    ax.errorbar(rounds, means, yerr=stds, fmt='o-', capsize=4)
    ax.set_xticks(rounds)
    ax.set_xlabel("Round count R")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_multi_seed_for_cipher(cipher: str):
    csv_path = DATA_DIR / f"{cipher}_multi_seed_raw.csv"
    df = load_multi_seed_csv(csv_path)
    if df.empty:
        print(f"[plot] No multi-seed CSV for {cipher} at {csv_path}")
        return

    agg = aggregate_by_round(df)
    rounds = sorted(agg.keys())
    if not rounds:
        print(f"[plot] No rounds in data for {cipher}")
        return

    for m in METRICS:
        means = [agg[r].get(f"{m}_mean", math.nan) for r in rounds]
        stds = [agg[r].get(f"{m}_std", math.nan) for r in rounds]
        title = f"{cipher.upper()}32/64 — {m} (mean ± std across seeds)"
        out_path = RESULTS_DIR / f"{cipher}_{m}.png"
        # For advantage, also plot as absolute advantage if needed (already advantage is |acc-0.5|)
        plot_metric_rounds(rounds, means, stds, title, out_path, ylabel=m)
        print(f"[plot] saved {out_path}")


def plot_classical_ddt(cipher: str):
    csv_path = DATA_DIR / f"{cipher}_classical_bounds.csv"
    if not csv_path.exists():
        print(f"[plot] No classical DDT CSV for {cipher} at {csv_path}")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[plot] classical CSV empty for {cipher}")
        return
    df = df.sort_values("rounds")
    rounds = df["rounds"].astype(int).tolist()
    probs = df["max_characteristic_prob"].astype(float).tolist()

    # Plot probability
    out_prob = RESULTS_DIR / f"{cipher}_classical_prob.png"
    title = f"{cipher.upper()}32/64 — Classical max characteristic probability"
    plot_metric_rounds(rounds, probs, [0]*len(probs), title, out_prob, ylabel="max_characteristic_prob")

    # Plot log2(p)
    log2p = [math.log2(max(p, 1e-300)) for p in probs]
    out_log2 = RESULTS_DIR / f"{cipher}_classical_log2p.png"
    title2 = f"{cipher.upper()}32/64 — Classical log2(max characteristic prob)"
    plot_metric_rounds(rounds, log2p, [0]*len(log2p), title2, out_log2, ylabel="log2(p)")
    print(f"[plot] saved {out_prob} and {out_log2}")


def main():
    ciphers = [p.stem.replace("_multi_seed_raw", "") for p in DATA_DIR.glob("*_multi_seed_raw.csv")]
    # Deduplicate
    ciphers = sorted(set(ciphers))
    if not ciphers:
        print("[plot] No multi-seed CSVs found in results/thesis")

    for cipher in ciphers:
        plot_multi_seed_for_cipher(cipher)
        plot_classical_ddt(cipher)


if __name__ == "__main__":
    main()
