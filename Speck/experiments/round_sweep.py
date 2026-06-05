#!/usr/bin/env python3
"""Track A: evaluate TorchAutoencoder scores across varying Speck round counts."""

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
from ml.features import build_feature_matrix
from ml.metrics import compute_metrics
from ml.models import TorchAutoencoder
from speck3264.cipher import ROUNDS, Speck3264
from speck3264.dataset import sample_keys, sample_plaintexts
from speck3264.faults import random_blocks


def generate_round_dataset(
    cipher: Speck3264,
    rounds: int,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    keys = sample_keys(n_samples, rng)
    pts = sample_plaintexts(n_samples, rng)
    blocks = np.empty((n_samples, 2), dtype=np.uint16)
    for i in range(n_samples):
        ct = cipher.encrypt_rounds(pts[i : i + 1], keys[i], rounds)
        blocks[i] = ct[0]
    label = 0 if rounds == ROUNDS else 1
    labels = np.full(n_samples, label, dtype=np.int8)
    return blocks, labels


def load_torch_model_and_threshold(
    model_dir: Path,
    reference_model: str,
) -> tuple[TorchAutoencoder, float]:
    model_path = model_dir / "torch_autoencoder.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Run: python ml/train.py --torch-only"
        )
    model = TorchAutoencoder.load(model_path)
    thresh_path = model_dir / "thresholds.json"
    if not thresh_path.exists():
        raise FileNotFoundError(f"Missing {thresh_path}")
    thresholds = json.loads(thresh_path.read_text(encoding="utf-8"))
    if reference_model not in thresholds:
        raise KeyError(
            f"threshold key {reference_model!r} not in {thresh_path}: {list(thresholds)}"
        )
    return model, float(thresholds[reference_model])


def evaluate_round(
    model: TorchAutoencoder,
    threshold: float,
    cipher: Speck3264,
    rounds: int,
    n_samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    blocks, _labels_trunc = generate_round_dataset(cipher, rounds, n_samples, rng)
    X = build_feature_matrix(blocks)
    scores = model.score_samples(X)

    rand_blocks = random_blocks(n_samples, rng)
    X_rand = build_feature_matrix(rand_blocks)
    scores_rand = model.score_samples(X_rand)

    labels_vs_rand = np.ones(n_samples * 2, dtype=np.int8)
    labels_vs_rand[:n_samples] = 0
    scores_vs_rand = np.concatenate([scores, scores_rand])

    flagged = scores > threshold
    rand_metrics = compute_metrics(scores_vs_rand, labels_vs_rand, threshold)

    if rounds == ROUNDS:
        detection_rate = float((~flagged).mean())
        false_positive_rate = float(flagged.mean())
    else:
        detection_rate = float(flagged.mean())
        false_positive_rate = float("nan")

    return {
        "rounds": rounds,
        "n_samples": n_samples,
        "mean_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
        "std_score": float(scores.std()) if n_samples > 1 else 0.0,
        "mean_score_random": float(scores_rand.mean()),
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
        "auc_vs_random": rand_metrics["auc_roc"],
        "threshold": threshold,
    }


def run_round_sweep(
    cfg: CryptanalysisConfig,
    *,
    quick_rounds: Optional[list[int]] = None,
    model_dir: Optional[Path] = None,
) -> dict[str, Any]:
    rc = cfg.round_sweep
    rounds_list = quick_rounds if quick_rounds is not None else rc.round_values
    model_dir = model_dir or Path(rc.model_dir)
    results_dir = Path(rc.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model, threshold = load_torch_model_and_threshold(model_dir, rc.reference_model)
    cipher = Speck3264()
    rng = np.random.default_rng(rc.seed)

    rows: list[dict[str, Any]] = []
    for r in rounds_list:
        if r > ROUNDS:
            print(f"\n[Round sweep] skipping rounds={r} (> {ROUNDS})")
            continue
        print(f"\n[Round sweep] rounds={r}")
        row = evaluate_round(model, threshold, cipher, r, rc.n_samples_per_round, rng)
        rows.append(row)
        print(
            f"  mean_score={row['mean_score']:.6f}  detection_rate={row['detection_rate']:.4f}  "
            f"auc_vs_random={row['auc_vs_random']:.4f}"
        )

    csv_path = results_dir / "round_sweep.csv"
    fieldnames = [
        "rounds",
        "mean_score",
        "median_score",
        "detection_rate",
        "false_positive_rate",
        "auc_vs_random",
        "n_samples",
        "mean_score_random",
        "threshold",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {"round_sweep": rows, "reference_model": rc.reference_model}
    json_path = results_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib.pyplot as plt

        rs = [row["rounds"] for row in rows]
        means = [row["mean_score"] for row in rows]
        rand_means = [row["mean_score_random"] for row in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(rs, means, "o-", label="Speck ciphertext")
        ax.plot(rs, rand_means, "s--", color="gray", label="random baseline")
        ax.set_xlabel("rounds")
        ax.set_ylabel("reconstruction error (AE score)")
        ax.set_title("Autoencoder score vs Speck encryption rounds")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(results_dir / "score_vs_rounds.png", dpi=120)
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed; skipping plot")

    return {"rows": rows, "csv_path": str(csv_path), "json_path": str(json_path)}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Round-count sweep for TorchAutoencoder")
    parser.add_argument("--config", default=None)
    parser.add_argument("--rounds", type=int, nargs="*", default=None)
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args(argv)
    cfg = load_cryptanalysis_config(args.config)
    run_round_sweep(
        cfg,
        quick_rounds=args.rounds,
        model_dir=Path(args.model_dir) if args.model_dir else None,
    )


if __name__ == "__main__":
    main()
