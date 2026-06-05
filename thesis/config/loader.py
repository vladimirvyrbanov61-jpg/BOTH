"""Load thesis YAML configs (requires PyYAML: pip install pyyaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

ProfileName = Literal["full", "quick"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = Path(__file__).resolve().parent

PROFILE_FILES: dict[ProfileName, str] = {
    "full": "thesis.yaml",
    "quick": "thesis_quick.yaml",
}


def config_path_for_profile(profile: ProfileName = "full") -> Path:
    """Resolve config file path for a named profile."""
    if profile not in PROFILE_FILES:
        raise ValueError(f"unknown profile {profile!r}; use {list(PROFILE_FILES)}")
    return _CONFIG_DIR / PROFILE_FILES[profile]


def load_config(path: Path | str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required for thesis config. Install with: pip install pyyaml"
        ) from e
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping, got {type(data)}")
    return data


def load_profile(profile: ProfileName = "full") -> dict[str, Any]:
    """Load thesis.yaml (full) or thesis_quick.yaml (smoke)."""
    return load_config(config_path_for_profile(profile))
