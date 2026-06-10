"""Validation tests for thesis experiment configurations."""

from __future__ import annotations

from copy import deepcopy
import math

import pytest

from thesis.config.loader import config_path_for_profile, load_config, validate_config
from scripts.multi_seed_sweep import run_seed_sweep


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


def test_zero_delta_fails_before_experiment() -> None:
    config = load_config(config_path_for_profile("quick"))
    invalid = deepcopy(config)
    invalid["input_delta"] = [0, 0]

    with pytest.raises(ValueError, match="must be nonzero"):
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


@pytest.mark.parametrize("seeds", [[], [1, 1], [-1]])
def test_seed_sweep_rejects_invalid_seed_lists(seeds) -> None:
    with pytest.raises(ValueError):
        run_seed_sweep(seeds, skip_tests=True)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_training_numbers_are_rejected(value) -> None:
    config = load_config(config_path_for_profile("quick"))
    invalid = deepcopy(config)
    invalid["training"]["lr"] = value

    with pytest.raises(ValueError, match="must be finite"):
        validate_config(invalid)
