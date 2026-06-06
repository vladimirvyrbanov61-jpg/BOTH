"""Publication plots: neural distinguisher advantage vs classical DDT bounds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from thesis.classical.characteristic import (
    load_classical_bounds_csv,
    save_classical_bounds_csv,
    track_characteristic_over_rounds,
)
from thesis.config.loader import config_path_for_profile, load_config
from thesis.data.generator import DEFAULT_INPUT_DELTA

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = config_path_for_profile("full")


def _resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def load_neural_sweep(csv_path: Path) -> dict[int, float]:
    """Map rounds → mean test advantage from round_sweep.csv."""
    out: dict[int, list[float]] = {}
    if not csv_path.exists():
        return {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split", "test") != "test":
                continue
            r = int(row["rounds"])
            if "advantage_abs" in row and row["advantage_abs"] != "":
                value = float(row["advantage_abs"])
            elif "advantage" in row and row["advantage"] != "":
                value = float(row["advantage"])
            else:
                raise KeyError(
                    f"CSV {csv_path} missing advantage column for round {r}. "
                    "Expected advantage_abs or advantage."
                )
            out.setdefault(r, []).append(value)
    return {r: sum(vals) / len(vals) for r, vals in out.items()}


def require_neural_overlap(
    cipher: str,
    rounds: list[int],
    neural: dict[int, float],
    neural_path: Path,
) -> None:
    """Raise a clear error if round_sweep.csv is missing or has no test rows."""
    if not neural_path.exists():
        raise FileNotFoundError(
            f"Neural sweep results missing for {cipher}: {neural_path}\n"
            "Run the round sweep first, e.g.:\n"
            "  py -3 -m thesis.eval.round_sweep --profile quick --fresh-csv\n"
            "  py -3 -m thesis.run_thesis --profile quick"
        )
    overlap = sorted(set(rounds) & set(neural.keys()))
    if not overlap:
        raise ValueError(
            f"No test-split neural metrics for {cipher} at rounds {rounds} in {neural_path}.\n"
            "The CSV exists but has no matching 'test' rows. Re-run:\n"
            "  py -3 -m thesis.eval.round_sweep --profile quick --cipher "
            f"{cipher} --fresh-csv"
        )


def ensure_classical_bounds(
    cipher: str,
    rounds: list[int],
    delta: tuple[int, int],
    results_dir: Path,
    *,
    n_samples_row: int | None,
    top_k: int,
    seed: int,
    force: bool,
) -> dict[int, float]:
    bounds_path = results_dir / f"{cipher}_classical_bounds.csv"
    if not force and bounds_path.exists():
        loaded = load_classical_bounds_csv(bounds_path)
        if all(r in loaded for r in rounds):
            return loaded

    rows = track_characteristic_over_rounds(
        cipher,
        rounds,
        delta,
        n_samples_row=n_samples_row,
        top_k=top_k,
        seed=seed,
    )
    if bounds_path.exists() and not force:
        bounds_path.unlink()
    save_classical_bounds_csv(bounds_path, rows)
    return load_classical_bounds_csv(bounds_path)


def plot_cipher_comparison(
    cipher: str,
    rounds: list[int],
    neural: dict[int, float],
    classical: dict[int, float],
    out_path: Path,
) -> None:
    xs = sorted(set(rounds) & set(neural.keys()) & set(classical.keys()))
    if not xs:
        raise ValueError(
            f"No overlapping rounds for {cipher} (config rounds={rounds}, "
            f"neural={sorted(neural.keys())}, classical={sorted(classical.keys())})."
        )

    adv = [neural[r] for r in xs]
    probs = [max(classical[r], 1e-300) for r in xs]
    log2p = [np.log2(p) for p in probs]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_ai = "#1f77b4"
    color_cl = "#d62728"

    ax1.plot(xs, adv, "o-", color=color_ai, linewidth=2, markersize=7, label="AI advantage |acc − 0.5|")
    ax1.set_xlabel("Round count R")
    ax1.set_ylabel("Neural advantage", color=color_ai)
    ax1.tick_params(axis="y", labelcolor=color_ai)
    ax1.set_xticks(xs)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, log2p, "s--", color=color_cl, linewidth=2, markersize=7, label="Classical log₂(p_max)")
    ax2.set_ylabel("log₂(max characteristic probability)", color=color_cl)
    ax2.tick_params(axis="y", labelcolor=color_cl)

    title = f"{cipher.upper()}32/64 — Neural distinguisher vs classical DDT bound"
    ax1.set_title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.95)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_compare(
    config_path: Path | None = None,
    *,
    ciphers: list[str] | None = None,
    force_classical: bool = False,
    results_dir: Path | str | None = None,
) -> list[Path]:
    cfg = load_config(config_path or DEFAULT_CONFIG)
    base = _REPO_ROOT
    results_path = _resolve_path(
        base,
        str(results_dir) if results_dir is not None else cfg.get("results_dir", "results/thesis"),
    )
    classical_cfg = cfg.get("classical", {})
    rounds = cfg.get("rounds", [3, 4, 5, 6, 7, 8, 9, 10])
    delta = tuple(cfg.get("input_delta", list(DEFAULT_INPUT_DELTA)))
    seed = int(cfg.get("seed", 1))
    top_k = int(classical_cfg.get("top_k_dp", 32))
    n_simon = classical_cfg.get("n_samples_simon", 250_000)
    n_speck = classical_cfg.get("n_samples_speck", 1_000_000)

    cipher_names = ciphers or cfg.get("ciphers") or ["simon", "speck"]
    if isinstance(cipher_names, str):
        cipher_names = [cipher_names]

    outputs: list[Path] = []
    for cipher in cipher_names:
        neural_path = results_path / f"{cipher}_round_sweep.csv"
        neural = load_neural_sweep(neural_path)
        require_neural_overlap(cipher, rounds, neural, neural_path)

        n_row = int(n_simon) if cipher == "simon" else int(n_speck)
        classical = ensure_classical_bounds(
            cipher,
            rounds,
            delta,
            results_path,
            n_samples_row=n_row,
            top_k=top_k,
            seed=seed,
            force=force_classical,
        )

        out_path = results_path / f"{cipher}_vs_classical.png"
        plot_cipher_comparison(cipher, rounds, neural, classical, out_path)
        outputs.append(out_path)
        print(f"[compare] saved {out_path}")

    return outputs


def resolve_config_path(
    config: Path | None = None,
    profile: str | None = None,
) -> Path:
    if config is not None:
        return config
    if profile in ("full", "quick", None):
        return config_path_for_profile(profile or "full")
    raise ValueError(f"unknown profile {profile!r}; use full or quick")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AI vs classical differential bounds")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--profile",
        choices=["full", "quick"],
        default=None,
        help="Use thesis.yaml (full) or thesis_quick.yaml (quick)",
    )
    parser.add_argument("--cipher", action="append", choices=["simon", "speck"])
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--force-classical", action="store_true")
    args = parser.parse_args()
    cfg_path = resolve_config_path(args.config, args.profile)
    run_compare(
        cfg_path,
        ciphers=args.cipher,
        force_classical=args.force_classical,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
