"""Validation tests for thesis experiment configurations."""

from __future__ import annotations

from copy import deepcopy

import pytest

from thesis.config.loader import config_path_for_profile, load_config, validate_config


def test_shipped_profiles_are_valid() -> None:
    load_config(config_path_for_profile("full"))
    load_config(config_path_for_profile("quick"))
    load_config("thesis/config/thesis_delta_0040.yaml")


def test_invalid_delta_fails_before_experiment() -> None:
    config = load_config(config_path_for_profile("quick"))
    invalid = deepcopy(config)
    invalid["input_delta"] = [1]

    with pytest.raises(ValueError, match="exactly two"):
        validate_config(invalid)


def test_invalid_split_ratios_fail_before_experiment() -> None:
    config = load_config(config_path_for_profile("quick"))
    invalid = deepcopy(config)
    invalid["train_ratio"] = 0.9
    invalid["val_ratio"] = 0.2

    with pytest.raises(ValueError, match="must be less than 1"):
        validate_config(invalid)


def test_odd_sample_count_is_rejected() -> None:
    config = load_config(config_path_for_profile("quick"))
    invalid = deepcopy(config)
    invalid["n_samples_per_round"] = 101

    with pytest.raises(ValueError, match="must be even"):
        validate_config(invalid)
