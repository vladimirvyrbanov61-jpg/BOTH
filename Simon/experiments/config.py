"""Cryptanalysis experiment configuration (separate from ml/config.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

try:
    import yaml as _yaml

    _YAML_OK = True
except ImportError:
    _YAML_OK = False


@dataclass
class RoundSweepConfig:
    round_values: list[int] = field(
        default_factory=lambda: [8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
    )
    n_samples_per_round: int = 2000
    seed: int = 42
    reference_model: str = "torch_autoencoder"
    model_dir: str = "models/"
    results_dir: str = "results/round_sweep/"


@dataclass
class DistinguisherConfig:
    round_values: list[int] = field(
        default_factory=lambda: [8, 10, 12, 14, 16, 18, 20]
    )
    n_samples_per_round: int = 10000
    input_delta: list[int] = field(default_factory=lambda: [1, 0])
    feature_mode: str = "xor_bits"
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 6
    seed: int = 42
    device: str = "auto"
    data_dir: str = "data/"
    model_dir: str = "models/"
    results_dir: str = "results/distinguisher/"


@dataclass
class CryptanalysisPaths:
    data_dir: str = "data/"
    model_dir: str = "models/"
    results_dir: str = "results/"


@dataclass
class CryptanalysisConfig:
    round_sweep: RoundSweepConfig = field(default_factory=RoundSweepConfig)
    distinguisher: DistinguisherConfig = field(default_factory=DistinguisherConfig)
    paths: CryptanalysisPaths = field(default_factory=CryptanalysisPaths)


def _sub(cls: type, d: dict[str, Any]) -> Any:
    fields = cls.__dataclass_fields__
    return cls(**{k: v for k, v in d.items() if k in fields})


def load_cryptanalysis_config(
    path: Optional[Union[str, Path]] = None,
) -> CryptanalysisConfig:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "cryptanalysis.yaml"
    path = Path(path)
    if not path.exists():
        return CryptanalysisConfig()
    if not _YAML_OK:
        import warnings

        warnings.warn("PyYAML not installed; using default CryptanalysisConfig.")
        return CryptanalysisConfig()
    with open(path, encoding="utf-8") as fh:
        raw = _yaml.safe_load(fh) or {}
    return CryptanalysisConfig(
        round_sweep=_sub(RoundSweepConfig, raw.get("round_sweep", {})),
        distinguisher=_sub(DistinguisherConfig, raw.get("distinguisher", {})),
        paths=_sub(CryptanalysisPaths, raw.get("paths", {})),
    )
