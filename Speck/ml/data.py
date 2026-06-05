"""ml/data.py — Data loading, splitting, and feature preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from speck3264.dataset import DatasetConfig, generate_labeled_dataset, stratified_split
from speck3264.io import load_blocks_npz, save_blocks_npz

from ml.config import ExperimentConfig
from ml.features import build_feature_matrix


def _feature_kwargs(cfg: ExperimentConfig) -> dict[str, bool]:
    return {
        "include_bits": cfg.features.include_bits,
        "include_stats": cfg.features.include_stats,
        "include_transitions": cfg.features.include_transitions,
        "include_entropy": cfg.features.include_entropy,
    }


def _config_hash(cfg: ExperimentConfig) -> str:
    spec = {
        "seed": cfg.data.seed,
        "n_samples": cfg.data.n_samples,
        "anomaly_fraction": cfg.data.anomaly_fraction,
        "fault_types": sorted(cfg.data.fault_types),
        "wrong_rounds_values": sorted(cfg.data.wrong_rounds_values),
        "flip_bits": cfg.data.flip_bits,
        "split_seed": cfg.split.split_seed,
        "train_ratio": cfg.split.train_ratio,
        "val_ratio": cfg.split.val_ratio,
        "features": _feature_kwargs(cfg),
    }
    return hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def generate_or_load_dataset(
    cfg: ExperimentConfig,
    *,
    cache_dir: Optional[Path] = None,
    force_regen: bool = False,
) -> dict[str, Any]:
    if cache_dir is None:
        cache_dir = Path(cfg.paths.data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = _config_hash(cfg)
    cache_path = cache_dir / f"dataset_{tag}.npz"
    feat_path = cache_dir / f"dataset_{tag}_features.npz"
    meta_path = cache_dir / f"dataset_{tag}.meta.json"

    fk = _feature_kwargs(cfg)

    if not force_regen and cache_path.exists() and feat_path.exists():
        raw = load_blocks_npz(cache_path)
        feat_raw = load_blocks_npz(feat_path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else []
        ds: dict[str, Any] = {
            "blocks": raw["blocks"],
            "labels": raw["labels"],
            "meta": meta,
            "config": cfg.data,
            "features": feat_raw["features"],
        }
        if "bits" in feat_raw:
            ds["bits"] = feat_raw["bits"]
        ds["split_indices"] = _compute_splits(ds["labels"], ds["meta"], cfg)
        return ds

    dc = DatasetConfig(
        seed=cfg.data.seed,
        n_samples=cfg.data.n_samples,
        anomaly_fraction=cfg.data.anomaly_fraction,
        fault_types=cfg.data.fault_types,
        wrong_rounds_values=cfg.data.wrong_rounds_values,
        flip_bits=cfg.data.flip_bits,
        feature_bits=cfg.data.feature_bits,
    )
    ds = generate_labeled_dataset(dc)
    ds["features"] = build_feature_matrix(ds["blocks"], **fk)
    ds["split_indices"] = _compute_splits(ds["labels"], ds["meta"], cfg)

    save_blocks_npz(cache_path, ds["blocks"], labels=ds["labels"])
    np.savez_compressed(
        feat_path,
        features=ds["features"],
        bits=ds.get("bits", np.empty(0)),
    )
    meta_path.write_text(json.dumps(ds["meta"]), encoding="utf-8")
    return ds


def _compute_splits(
    labels: np.ndarray,
    meta: list[dict[str, Any]],
    cfg: ExperimentConfig,
) -> dict[str, np.ndarray]:
    tr, va, te = stratified_split(
        labels,
        meta,
        train_ratio=cfg.split.train_ratio,
        val_ratio=cfg.split.val_ratio,
        seed=cfg.split.split_seed,
    )
    return {"train": tr, "val": va, "test": te}


class DataSplit:
    def __init__(self, ds: dict[str, Any]) -> None:
        idx = ds["split_indices"]
        tr, va, te = idx["train"], idx["val"], idx["test"]

        self.X_train = ds["features"][tr].astype(np.float64)
        self.X_val = ds["features"][va].astype(np.float64)
        self.X_test = ds["features"][te].astype(np.float64)

        self.y_train = ds["labels"][tr].astype(np.int8)
        self.y_val = ds["labels"][va].astype(np.int8)
        self.y_test = ds["labels"][te].astype(np.int8)

        self.meta_train = [ds["meta"][i] for i in tr]
        self.meta_val = [ds["meta"][i] for i in va]
        self.meta_test = [ds["meta"][i] for i in te]

        normal_mask = self.y_train == 0
        self.X_train_normal = self.X_train[normal_mask]

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def n_train(self) -> int:
        return len(self.y_train)

    @property
    def n_val(self) -> int:
        return len(self.y_val)

    @property
    def n_test(self) -> int:
        return len(self.y_test)

    def summary(self) -> str:
        lines = [
            f"  train : {self.n_train:>6} samples  ({int(self.y_train.sum()):>4} anomalies)",
            f"  val   : {self.n_val:>6} samples  ({int(self.y_val.sum()):>4} anomalies)",
            f"  test  : {self.n_test:>6} samples  ({int(self.y_test.sum()):>4} anomalies)",
            f"  normal-only train: {len(self.X_train_normal)} samples",
            f"  n_features: {self.n_features}",
        ]
        return "\n".join(lines)
