#!/usr/bin/env python3
"""Score SIMON 32/64 blocks for anomalies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.config import ExperimentConfig, load_config
from ml.data import DataSplit, generate_or_load_dataset
from ml.features import build_feature_matrix
from ml.metrics import (
    compute_metrics,
    fault_breakdown,
    find_threshold_at_fpr,
    format_fault_table,
    format_metrics_table,
)
from ml.models import IsolationForestModel, NumpyAutoencoder, TorchAutoencoder
from simon3264.io import blocks_from_file

MODEL_CHOICES = ("iso_forest", "autoencoder", "torch_autoencoder")
AnyModel = Union[IsolationForestModel, NumpyAutoencoder, TorchAutoencoder]


def load_model(model_name: str, model_dir: Path) -> AnyModel:
    if model_name == "iso_forest":
        path = model_dir / "iso_forest.pkl"
        return IsolationForestModel.load(path)
    if model_name == "autoencoder":
        path = model_dir / "autoencoder.pkl"
        return NumpyAutoencoder.load(path)
    if model_name == "torch_autoencoder":
        path = model_dir / "torch_autoencoder.pt"
        return TorchAutoencoder.load(path)
    raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_CHOICES}")


def load_threshold(model_name: str, model_dir: Path) -> Optional[float]:
    path = model_dir / "thresholds.json"
    if path.exists():
        return json.loads(path.read_text()).get(model_name)
    return None


def _feature_kwargs(cfg: ExperimentConfig) -> dict[str, bool]:
    return {
        "include_bits": cfg.features.include_bits,
        "include_stats": cfg.features.include_stats,
        "include_transitions": cfg.features.include_transitions,
        "include_entropy": cfg.features.include_entropy,
        "include_hw_chi2": cfg.features.include_hw_chi2,
        "include_recovery_error": False,
    }


def _load_hw_reference(model_dir: Path, data_dir: Path) -> np.ndarray | None:
    for path in (model_dir / "hw_reference.npy",):
        if path.exists():
            return np.load(path)
    return None


def _blocks_to_features(
    blocks: np.ndarray,
    cfg: ExperimentConfig,
    *,
    hw_reference: np.ndarray | None,
) -> np.ndarray:
    fk = _feature_kwargs(cfg)
    if fk.get("include_hw_chi2") and hw_reference is None:
        raise FileNotFoundError(
            "hw_reference.npy missing in model_dir; re-run ml/train.py with current config"
        )
    return build_feature_matrix(blocks, hw_reference=hw_reference, **fk)


def cmd_evaluate(args: argparse.Namespace, cfg: ExperimentConfig) -> None:
    model_dir = Path(cfg.paths.model_dir)
    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ds = generate_or_load_dataset(cfg, cache_dir=Path(cfg.paths.data_dir))
    split = DataSplit(ds)
    model = load_model(args.model, model_dir)
    threshold = args.threshold or load_threshold(args.model, model_dir)
    if threshold is None:
        val_scores = model.score_samples(split.X_val)
        threshold = find_threshold_at_fpr(val_scores, split.y_val, cfg.scoring.target_fpr)

    test_scores = model.score_samples(split.X_test)
    metrics = compute_metrics(test_scores, split.y_test, threshold)
    fb = fault_breakdown(test_scores, split.y_test, split.meta_test, threshold)

    print(format_metrics_table(metrics))
    print(format_fault_table(fb))

    out_path = results_dir / f"test_metrics_{args.model}.json"

    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    out_path.write_text(
        json.dumps(_clean({"model": args.model, "threshold": threshold, "metrics": metrics, "fault_breakdown": fb}), indent=2)
    )
    csv_path = results_dir / f"predictions_{args.model}.csv"
    preds = (test_scores > threshold).astype(int)
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("index,score,prediction,true_label,fault\n")
        for i, (sc, pr, lab, m) in enumerate(
            zip(test_scores, preds, split.y_test, split.meta_test)
        ):
            fh.write(f"{i},{sc:.8f},{pr},{int(lab)},{m.get('fault','')}\n")
    print(f"Saved {out_path} and {csv_path}")


def cmd_score_file(args: argparse.Namespace, cfg: ExperimentConfig) -> None:
    model_dir = Path(cfg.paths.model_dir)
    results_dir = Path(cfg.paths.results_dir)
    input_path = Path(args.input_file)
    blocks = blocks_from_file(input_path, format=args.format)
    hw_ref = _load_hw_reference(model_dir, Path(cfg.paths.data_dir))
    features = _blocks_to_features(blocks, cfg, hw_reference=hw_ref)
    model = load_model(args.model, model_dir)
    threshold = args.threshold or load_threshold(args.model, model_dir) or 0.5
    scores = model.score_samples(features)
    preds = (scores > threshold).astype(int)
    print(f"Scored {len(blocks)} blocks; {int(preds.sum())} anomalies (threshold={threshold:.6f})")
    csv_path = results_dir / f"scores_{input_path.stem}_{args.model}.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("index,score,prediction\n")
        for i, (sc, pr) in enumerate(zip(scores, preds)):
            fh.write(f"{i},{sc:.8f},{pr}\n")
    print(f"Saved {csv_path}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command")

    ev = sub.add_parser("evaluate")
    ev.add_argument("--model", default="torch_autoencoder", choices=MODEL_CHOICES)
    ev.add_argument("--threshold", type=float, default=None)

    sf = sub.add_parser("score-file")
    sf.add_argument("input_file")
    sf.add_argument("--format", default="hex", choices=["hex", "bin", "npz"])
    sf.add_argument("--model", default="torch_autoencoder", choices=MODEL_CHOICES)
    sf.add_argument("--threshold", type=float, default=None)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "evaluate":
        cmd_evaluate(args, cfg)
    elif args.command == "score-file":
        cmd_score_file(args, cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
