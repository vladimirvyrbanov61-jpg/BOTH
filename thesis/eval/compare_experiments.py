"""Compare two completed multi-seed experiments round by round."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from thesis.eval.aggregate import aggregate_csv

COMPARISON_FIELDS = [
    "cipher",
    "rounds",
    "metric",
    "baseline_mean",
    "baseline_ci95_low",
    "baseline_ci95_high",
    "candidate_mean",
    "candidate_ci95_low",
    "candidate_ci95_high",
    "mean_difference",
    "ci95_overlap",
]


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment manifest: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    expected_delta: list[int],
    expected_seeds: list[int],
    expected_rounds: list[int],
) -> None:
    parameters = manifest.get("parameters", {})
    actual_delta = parameters.get("input_delta")
    if actual_delta != expected_delta:
        raise ValueError(f"Expected input_delta={expected_delta}, found {actual_delta}")
    if parameters.get("seeds") != expected_seeds:
        raise ValueError(
            f"Expected seeds={expected_seeds}, found {parameters.get('seeds')}"
        )
    if parameters.get("rounds") != expected_rounds:
        raise ValueError(
            f"Expected rounds={expected_rounds}, found {parameters.get('rounds')}"
        )
    if manifest.get("status") != "completed":
        raise ValueError(f"Experiment status must be completed, found {manifest.get('status')}")


def _load_aggregates(
    run_dir: Path,
    ciphers: list[str],
    expected_seeds: int,
    expected_rounds: list[int],
) -> dict[str, dict[int, dict[str, Any]]]:
    output: dict[str, dict[int, dict[str, Any]]] = {}
    for cipher in ciphers:
        rows, _ = aggregate_csv(run_dir / f"{cipher}_multi_seed_raw.csv")
        by_round = {int(row["rounds"]): row for row in rows}
        if sorted(by_round) != expected_rounds:
            raise ValueError(
                f"{cipher} in {run_dir} has rounds {sorted(by_round)}, "
                f"expected {expected_rounds}"
            )
        for row in rows:
            if int(row["n_seeds"]) != expected_seeds:
                raise ValueError(
                    f"{cipher} round {row['rounds']} has {row['n_seeds']} seeds, "
                    f"expected {expected_seeds}"
                )
        output[cipher] = by_round
    return output


def build_comparison_rows(
    baseline: dict[str, dict[int, dict[str, Any]]],
    candidate: dict[str, dict[int, dict[str, Any]]],
    *,
    metrics: tuple[str, ...] = ("auc_roc", "advantage_abs"),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cipher in sorted(baseline):
        for rounds in sorted(baseline[cipher]):
            for metric in metrics:
                base = baseline[cipher][rounds]
                new = candidate[cipher][rounds]
                base_low = float(base[f"{metric}_ci95_low"])
                base_high = float(base[f"{metric}_ci95_high"])
                new_low = float(new[f"{metric}_ci95_low"])
                new_high = float(new[f"{metric}_ci95_high"])
                base_mean = float(base[f"{metric}_mean"])
                new_mean = float(new[f"{metric}_mean"])
                rows.append(
                    {
                        "cipher": cipher,
                        "rounds": rounds,
                        "metric": metric,
                        "baseline_mean": base_mean,
                        "baseline_ci95_low": base_low,
                        "baseline_ci95_high": base_high,
                        "candidate_mean": new_mean,
                        "candidate_ci95_low": new_low,
                        "candidate_ci95_high": new_high,
                        "mean_difference": new_mean - base_mean,
                        "ci95_overlap": max(base_low, new_low) <= min(base_high, new_high),
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(
    cipher: str,
    metric: str,
    rows: list[dict[str, Any]],
    output_path: Path,
    baseline_label: str,
    candidate_label: str,
) -> None:
    selected = [
        row for row in rows if row["cipher"] == cipher and row["metric"] == metric
    ]
    rounds = [int(row["rounds"]) for row in selected]
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8, 5))

    for prefix, label, color, offset in (
        ("baseline", baseline_label, "#1f77b4", -0.06),
        ("candidate", candidate_label, "#ff7f0e", 0.06),
    ):
        means = np.asarray([float(row[f"{prefix}_mean"]) for row in selected])
        lower = np.asarray([float(row[f"{prefix}_ci95_low"]) for row in selected])
        upper = np.asarray([float(row[f"{prefix}_ci95_high"]) for row in selected])
        ax.errorbar(
            np.asarray(rounds) + offset,
            means,
            yerr=np.vstack([means - lower, upper - means]),
            fmt="o-",
            capsize=4,
            linewidth=2,
            color=color,
            label=label,
        )

    ax.set_xticks(rounds)
    ax.set_xlabel("Round count R")
    ax.set_ylabel(metric)
    ax.set_title(f"{cipher.upper()}32/64 - Input-difference sensitivity ({metric})")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compare_experiments(
    baseline_dir: Path,
    candidate_dir: Path,
    output_dir: Path,
) -> list[Path]:
    ciphers = ["simon", "speck"]
    seeds = list(range(1, 11))
    rounds = list(range(3, 11))
    baseline_manifest = _load_manifest(baseline_dir)
    candidate_manifest = _load_manifest(candidate_dir)
    _validate_manifest(
        baseline_manifest,
        expected_delta=[1, 0],
        expected_seeds=seeds,
        expected_rounds=rounds,
    )
    _validate_manifest(
        candidate_manifest,
        expected_delta=[0x0040, 0],
        expected_seeds=seeds,
        expected_rounds=rounds,
    )
    baseline = _load_aggregates(baseline_dir, ciphers, len(seeds), rounds)
    candidate = _load_aggregates(candidate_dir, ciphers, len(seeds), rounds)
    rows = build_comparison_rows(baseline, candidate)

    csv_path = output_dir / "input_delta_comparison.csv"
    _write_csv(csv_path, rows)
    outputs = [csv_path]
    for cipher in ciphers:
        for metric in ("auc_roc", "advantage_abs"):
            plot_path = output_dir / f"{cipher}_{metric}_input_delta_comparison.png"
            _plot_metric(
                cipher,
                metric,
                rows,
                plot_path,
                "Delta=(0x0001, 0x0000)",
                "Delta=(0x0040, 0x0000)",
            )
            outputs.append(plot_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two input-difference experiments")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.candidate_dir / "baseline_comparison"
    outputs = compare_experiments(args.baseline_dir, args.candidate_dir, output_dir)
    for path in outputs:
        print(f"[experiment-compare] saved {path}")


if __name__ == "__main__":
    main()
