"""ml/config.py — Load and validate experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Union

try:
    import yaml as _yaml

    _YAML_OK = True
except ImportError:
    _YAML_OK = False


@dataclass
class DataConfig:
    seed: int = 42
    n_samples: int = 10_000
    """Generate ciphertext in chunks of this size to cap peak RAM during build."""
    chunk_size: int = 25_000
    anomaly_fraction: float = 0.20
    fault_types: list[str] = field(
        default_factory=lambda: ["random", "flip", "wrong_rounds"]
    )
    wrong_rounds_values: list[int] = field(default_factory=lambda: [8, 16])
    wrong_z_index: int = 1
    flip_bits: int = 3
    feature_bits: bool = True


@dataclass
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    split_seed: int = 0


@dataclass
class FeatureConfig:
    """Feature flags for build_feature_matrix.

    Raw 32-bit expansions often make random blocks indistinguishable from
    Simon ciphertext in Hamming-weight space; disable for anomaly training.
    """
    include_bits: bool = False
    include_stats: bool = True
    include_transitions: bool = True
    include_entropy: bool = True
    include_hw_chi2: bool = True
    """Requires plaintext+key arrays at dataset build (training/eval only)."""
    include_recovery_error: bool = True


@dataclass
class IsoForestConfig:
    n_estimators: int = 200
    max_samples: Union[str, int] = "auto"
    contamination: Union[str, float] = "auto"
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class AutoencoderConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32, 16])
    latent_dim: int = 8
    activation: str = "relu"
    dropout: float = 0.0
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    seed: int = 42


@dataclass
class TorchAutoencoderConfig:
    """PyTorch MLP autoencoder (recommended for Google Colab)."""

    hidden_dims: list[int] = field(default_factory=lambda: [128, 64, 32])
    latent_dim: int = 16
    dropout: float = 0.1
    epochs: int = 40
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 8
    seed: int = 42
    device: str = "auto"
    """Save state_dict here whenever validation loss improves (empty = no disk checkpoints)."""
    checkpoint_path: str = "models/torch_autoencoder_best.pt"


@dataclass
class ScoringConfig:
    target_fpr: float = 0.01
    score_agg: Literal["mean", "max"] = "mean"


@dataclass
class PathConfig:
    data_dir: str = "data/"
    model_dir: str = "models/"
    results_dir: str = "results/"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    isolation_forest: IsoForestConfig = field(default_factory=IsoForestConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    torch_autoencoder: TorchAutoencoderConfig = field(default_factory=TorchAutoencoderConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    paths: PathConfig = field(default_factory=PathConfig)


def _from_dict(cfg_dict: dict[str, Any]) -> ExperimentConfig:
    def _sub(cls: type, key: str) -> Any:
        d = cfg_dict.get(key, {})
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})

    return ExperimentConfig(
        data=_sub(DataConfig, "data"),
        split=_sub(SplitConfig, "split"),
        features=_sub(FeatureConfig, "features"),
        isolation_forest=_sub(IsoForestConfig, "isolation_forest"),
        autoencoder=_sub(AutoencoderConfig, "autoencoder"),
        torch_autoencoder=_sub(TorchAutoencoderConfig, "torch_autoencoder"),
        scoring=_sub(ScoringConfig, "scoring"),
        paths=_sub(PathConfig, "paths"),
    )


def load_config(path: Optional[Union[str, Path]] = None) -> ExperimentConfig:
    if path is None:
        here = Path(__file__).resolve().parent
        path = here.parent / "configs" / "default.yaml"

    path = Path(path)
    if not path.exists():
        return ExperimentConfig()

    if not _YAML_OK:
        import warnings

        warnings.warn(
            "PyYAML not installed; using default ExperimentConfig.",
            stacklevel=2,
        )
        return ExperimentConfig()

    with open(path, encoding="utf-8") as fh:
        raw = _yaml.safe_load(fh) or {}

    return _from_dict(raw)
