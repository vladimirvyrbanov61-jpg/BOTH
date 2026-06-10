#!/usr/bin/env python3
"""Train anomaly detection models on SIMON 32/64 ciphertext features."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.config import ExperimentConfig, load_config
from ml.data import DataSplit, generate_or_load_dataset
from ml.metrics import (
    compute_metrics,
    fault_breakdown,
    find_threshold_at_fpr,
    format_fault_table,
    format_metrics_table,
)
from ml.models import IsolationForestModel, NumpyAutoencoder, TorchAutoencoder


def lock_seeds(seed: int) -> None:
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def train_iso_forest(split: DataSplit, cfg: ExperimentConfig):
    print("\n[IsolationForest] Fitting …")
    t0 = time.perf_counter()
    model = IsolationForestModel(cfg=cfg.isolation_forest)
    model.fit(split.X_train_normal)
    print(f"  Fit complete in {time.perf_counter() - t0:.1f}s")
    val_scores = model.score_samples(split.X_val)
    threshold = find_threshold_at_fpr(val_scores, split.y_val, cfg.scoring.target_fpr)
    print(format_metrics_table(compute_metrics(val_scores, split.y_val, threshold)))
    print(format_fault_table(fault_breakdown(val_scores, split.y_val, split.meta_val, threshold)))
    return model, threshold


def train_numpy_ae(split: DataSplit, cfg: ExperimentConfig):
    print("\n[NumpyAutoencoder] Fitting …")
    t0 = time.perf_counter()
    model = NumpyAutoencoder(cfg=cfg.autoencoder)
    model.fit(split.X_train_normal, X_val=split.X_val, y_val=split.y_val)
    print(f"  Fit complete in {time.perf_counter() - t0:.1f}s")
    val_scores = model.score_samples(split.X_val)
    threshold = find_threshold_at_fpr(val_scores, split.y_val, cfg.scoring.target_fpr)
    print(format_metrics_table(compute_metrics(val_scores, split.y_val, threshold)))
    return model, threshold


def train_torch_ae(split: DataSplit, cfg: ExperimentConfig):
    print("\n[TorchAutoencoder] Fitting …")
    try:
        import torch
    except ImportError as e:
        raise SystemExit("PyTorch required. On Colab: pip install torch") from e

    print(f"  PyTorch {torch.__version__}  device={cfg.torch_autoencoder.device}")
    t0 = time.perf_counter()
    model = TorchAutoencoder(cfg=cfg.torch_autoencoder)
    model.fit(split.X_train_normal, X_val=split.X_val, y_val=split.y_val)
    print(f"  Fit complete in {time.perf_counter() - t0:.1f}s")
    val_scores = model.score_samples(split.X_val)
    threshold = find_threshold_at_fpr(val_scores, split.y_val, cfg.scoring.target_fpr)
    print(format_metrics_table(compute_metrics(val_scores, split.y_val, threshold)))
    print(format_fault_table(fault_breakdown(val_scores, split.y_val, split.meta_val, threshold)))
    return model, threshold


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train SIMON 32/64 anomaly models.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--anomaly-fraction", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="AE / torch epochs")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--force-regen", action="store_true")
    parser.add_argument("--no-autoencoder", action="store_true", help="Skip NumPy AE")
    parser.add_argument("--no-torch", action="store_true", help="Skip PyTorch AE")
    parser.add_argument("--torch-only", action="store_true", help="Only train PyTorch AE (+ skip IF)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.n_samples is not None:
        cfg.data.n_samples = args.n_samples
    if args.anomaly_fraction is not None:
        cfg.data.anomaly_fraction = args.anomaly_fraction
    if args.epochs is not None:
        cfg.autoencoder.epochs = args.epochs
        cfg.torch_autoencoder.epochs = args.epochs
    if args.model_dir:
        cfg.paths.model_dir = args.model_dir
    if args.results_dir:
        cfg.paths.results_dir = args.results_dir
    if args.seed is not None:
        cfg.data.seed = args.seed
        cfg.autoencoder.seed = args.seed
        cfg.torch_autoencoder.seed = args.seed
        cfg.isolation_forest.random_state = args.seed

    lock_seeds(cfg.data.seed)
    model_dir = Path(cfg.paths.model_dir)
    results_dir = Path(cfg.paths.results_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("SIMON 32/64 Anomaly Detection — Training")
    print(f"  n_samples={cfg.data.n_samples}  anomaly_fraction={cfg.data.anomaly_fraction:.1%}")

    data_dir = Path(cfg.paths.data_dir)
    ds = generate_or_load_dataset(cfg, cache_dir=data_dir, force_regen=args.force_regen)
    split = DataSplit(ds)
    print(split.summary())

    hw_ref = ds.get("hw_reference")
    if hw_ref is not None:
        np.save(model_dir / "hw_reference.npy", hw_ref)
        print(f"  HW reference histogram saved ({len(hw_ref)} bins)")

    thresholds: dict[str, float] = {}
    val_metrics: dict[str, Any] = {}

    if not args.torch_only:
        iso_model, iso_t = train_iso_forest(split, cfg)
        iso_model.save(model_dir / "iso_forest.pkl")
        thresholds["iso_forest"] = iso_t
        vs = iso_model.score_samples(split.X_val)
        val_metrics["iso_forest"] = compute_metrics(vs, split.y_val, iso_t)

        if not args.no_autoencoder:
            ae_model, ae_t = train_numpy_ae(split, cfg)
            ae_model.save(model_dir / "autoencoder.pkl")
            thresholds["autoencoder"] = ae_t
            vs = ae_model.score_samples(split.X_val)
            val_metrics["autoencoder"] = compute_metrics(vs, split.y_val, ae_t)

    if not args.no_torch:
        torch_model, torch_t = train_torch_ae(split, cfg)
        torch_model.save(model_dir / "torch_autoencoder.pt")
        ckpt = Path(cfg.torch_autoencoder.checkpoint_path or "")
        if ckpt.exists():
            print(f"  Best checkpoint (val loss): {ckpt}")
        thresholds["torch_autoencoder"] = torch_t
        vs = torch_model.score_samples(split.X_val)
        val_metrics["torch_autoencoder"] = compute_metrics(vs, split.y_val, torch_t)

    (model_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (results_dir / "val_metrics.json").write_text(json.dumps(_clean(val_metrics), indent=2))
    print(f"\nModels saved under {model_dir}")
    print("Training complete.")


if __name__ == "__main__":
    main()
