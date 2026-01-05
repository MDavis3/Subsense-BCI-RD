"""
Configuration Management for Subsense BCI

Loads simulation parameters from YAML config files with fallback to
hardcoded defaults in physics.constants.

Usage:
    from subsense_bci.config import load_config, get_config_path

    cfg = load_config()  # Load default config
    cfg = load_config("configs/custom.yaml")  # Load custom config

    # Access parameters
    snr = cfg["temporal"]["snr_level"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Determine project root (works both installed and development mode)
# File is at: src/subsense_bci/config.py
# Project root: src/subsense_bci -> src -> project_root
_THIS_FILE = Path(__file__)
PROJECT_ROOT = _THIS_FILE.parent.parent.parent

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default_sim.yaml"


def get_config_path(config_name: str = "default_sim.yaml") -> Path:
    """
    Get the full path to a config file.

    Parameters
    ----------
    config_name : str
        Name of the config file (with or without .yaml extension).

    Returns
    -------
    Path
        Full path to the config file.
    """
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"
    return PROJECT_ROOT / "configs" / config_name


def get_default_config() -> dict[str, Any]:
    """
    Return hardcoded default configuration.

    Used as fallback when config file is missing.
    """
    # Import here to avoid circular imports
    from subsense_bci.physics.constants import (
        CLOUD_VOLUME_SIDE_MM,
        DEFAULT_SENSOR_COUNT,
        DEFAULT_RANDOM_SEED,
        SINGULARITY_THRESHOLD_MM,
        PARTICLE_RADIUS_NM,
        SAMPLING_RATE_HZ,
        DURATION_SEC,
        SNR_LEVEL,
        BRAIN_CONDUCTIVITY_S_M,
    )

    return {
        "cloud": {
            "volume_side_mm": CLOUD_VOLUME_SIDE_MM,
            "sensor_count": DEFAULT_SENSOR_COUNT,
            "random_seed": DEFAULT_RANDOM_SEED,
            "singularity_threshold_mm": SINGULARITY_THRESHOLD_MM,
            "particle_radius_nm": PARTICLE_RADIUS_NM,
        },
        "temporal": {
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "duration_sec": DURATION_SEC,
            "snr_level": SNR_LEVEL,
        },
        "unmixing": {
            "pca_variance_threshold": 0.999,
            "n_sources": 3,
            "ica_max_iter": 1000,
            "ica_random_state": 42,
        },
        "physics": {
            "brain_conductivity_s_m": BRAIN_CONDUCTIVITY_S_M,
            "source_frequencies_hz": {
                "alpha": 10.0,
                "beta": 20.0,
            },
        },
    }


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load configuration from YAML file with fallback to defaults.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to YAML config file. If None, uses default_sim.yaml.
        If file doesn't exist, falls back to hardcoded defaults.

    Returns
    -------
    dict
        Configuration dictionary.

    Examples
    --------
    >>> cfg = load_config()
    >>> cfg["temporal"]["snr_level"]
    5.0
    >>> cfg["cloud"]["sensor_count"]
    10000
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    else:
        # Fallback to hardcoded defaults
        return get_default_config()


def save_config(config: dict[str, Any], config_path: str | Path) -> None:
    """
    Save configuration to a YAML file.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    config_path : str or Path
        Output path for the YAML file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
