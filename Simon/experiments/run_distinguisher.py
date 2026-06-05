#!/usr/bin/env python3
"""Train and evaluate neural distinguishers per round count."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.config import CryptanalysisConfig, load_cryptanalysis_config
from experiments.distinguisher_data import generate_or_load_distinguisher
from experiments.distinguisher_model import NeuralDistinguisher
from experiments.metrics import binary_classification_metrics


def evaluate_distinguisher(
    model: NeuralDistinguisher,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    proba = model.predict_proba(X)
    preds = (proba >= 0.5).astype(np.int8)
    cls = binary_classification_metrics(y, preds)
    acc = cls["accuracy"]
    try:
        auc = float(roc_auc_score(y, proba))
    except ValueError:
        auc = float("nan")
    advantage = abs(acc - 0.5)
    return {
        "accuracy": acc,
        "auc": auc,
        "advantage": advantage,
        "tpr": cls["tpr"],
        "tnr": cls["tnr"],
        "mean_proba_real": float(proba[y == 1].mean()) if (y == 1).any() else 0.0,
        "mean_proba_random": float(proba[y == 0].mean()) if (y == 0).any() else 0.0,
    }


def run_distinguisher_experiment(
    cfg: CryptanalysisConfig,
    *,
    force_regen: bool = False,
    quick_rounds: Optional[list[int]] = None,
) -> dict[str, Any]:
    dc = cfg.distinguisher
    rounds_list = quick_rounds if quick_rounds is not None else dc.round_values
    model_dir = Path(dc.model_dir)
    results_dir = Path(dc.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, Any]] = []

    for r in rounds_list:
        print(f"\n[Distinguisher] rounds={r}")
        ds = generate_or_load_distinguisher(
            r, dc, cache_dir=Path(dc.data_dir), force_regen=force_regen
        )
        splits = ds["splits"]
        X, y = ds["X"], ds["y"]

        model = NeuralDistinguisher(dc)
        model.fit(
            X[splits["train"]],
            y[splits["train"]],
            X[splits["val"]],
            y[splits["val"]],
        )
        test_metrics = evaluate_distinguisher(model, X[splits["test"]], y[splits["test"]])
        test_metrics["rounds"] = r
        test_metrics["n_test"] = int(len(splits["test"]))

        model_path = model_dir / f"distinguisher_r{r}.pt"
        model.save(model_path)
        metrics_path = results_dir / f"metrics_r{r}.json"
        metrics_path.write_text(json.dumps(test_metrics, indent=2))
        all_metrics.append(test_metrics)
        print(
            f"  acc={test_metrics['accuracy']:.4f}  auc={test_metrics['auc']:.4f}  "
            f"advantage={test_metrics['advantage']:.4f}  "
            f"tpr={test_metrics['tpr']:.4f}  tnr={test_metrics['tnr']:.4f}"
        )

    csv_path = results_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rounds",
                "accuracy",
                "auc",
                "advantage",
                "tpr",
                "tnr",
                "n_test",
            ],
        )
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: m.get(k, "") for k in writer.fieldnames})

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(all_metrics, indent=2))

    try:
        import matplotlib.pyplot as plt

        rs = [m["rounds"] for m in all_metrics]
        adv = [m["advantage"] for m in all_metrics]
        acc = [m["accuracy"] for m in all_metrics]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(rs, adv, "o-")
        ax[0].set_xlabel("rounds")
        ax[0].set_ylabel("advantage |acc - 0.5|")
        ax[0].set_title("Neural distinguisher advantage vs rounds")
        ax[0].grid(True, alpha=0.3)
        ax[1].plot(rs, acc, "o-")
        ax[1].set_xlabel("rounds")
        ax[1].set_ylabel("accuracy")
        ax[1].set_title("Neural distinguisher accuracy vs rounds")
        ax[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(results_dir / "advantage_vs_rounds.png", dpi=120)
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed; skipping plot")

    return {"metrics": all_metrics, "csv_path": str(csv_path)}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Neural distinguisher sweep over rounds")
    parser.add_argument("--config", default=None)
    parser.add_argument("--force-regen", action="store_true")
    parser.add_argument(
        "--rounds",
        type=int,
        nargs="*",
        default=None,
        help="Override round list (e.g. --rounds 8 12 16)",
    )
    args = parser.parse_args(argv)
    cfg = load_cryptanalysis_config(args.config)
    run_distinguisher_experiment(
        cfg,
        force_regen=args.force_regen,
        quick_rounds=args.rounds,
    )


if __name__ == "__main__":
    main()
