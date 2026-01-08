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
        BRAIN_CONDUCTIVITY_S_M,
        CARDIAC_FREQUENCY_HZ,
        CLOUD_VOLUME_SIDE_MM,
        DEFAULT_RANDOM_SEED,
        DEFAULT_SENSOR_COUNT,
        DURATION_SEC,
        HEMODYNAMIC_DRIFT_AMPLITUDE_MM,
        PARTICLE_RADIUS_NM,
        SAMPLING_RATE_HZ,
        SINGULARITY_THRESHOLD_MM,
        SNR_LEVEL,
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
        "realtime": {
            "chunk_size_ms": 100.0,
            "window_ms": 500.0,
            "display_sensors": 50,
            "animation_interval_ms": 50,
        },
        "biology": {
            "cardiac_pulse_active": True,
            "vascular_distribution": False,
            "hemodynamic": {
                "drift_amplitude_mm": HEMODYNAMIC_DRIFT_AMPLITUDE_MM,
                "drift_frequency_hz": CARDIAC_FREQUENCY_HZ,
                "phase_coherence": 0.2,
            },
            "artifact_rejection": {
                "enabled": False,
                "method": "rls",
                "n_taps": 32,
                "lambda_": 0.99,
                "mu": 0.01,
            },
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

    Raises
    ------
    None
        This function never raises; it gracefully falls back to defaults.

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
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if config is None:
                # Empty YAML file
                return get_default_config()
            return config
        except yaml.YAMLError:
            # Malformed YAML - fall back to defaults silently
            # For detailed error info, use validation.validate_config_file()
            return get_default_config()
        except IOError:
            # File read error - fall back to defaults
            return get_default_config()
    else:
        # Fallback to hardcoded defaults
        return get_default_config()


def load_config_safe(
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """
    Load configuration with detailed error reporting.

    Unlike load_config(), this function returns error messages
    for debugging and user feedback.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to YAML config file.

    Returns
    -------
    tuple[dict, list[str]]
        (config_dict, error_messages). Config is always valid (defaults used on error).
        error_messages is empty if load succeeded.

    Examples
    --------
    >>> cfg, errors = load_config_safe("bad_config.yaml")
    >>> if errors:
    ...     print("Warnings:", errors)
    """
    errors = []

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        errors.append(f"Config file not found: {config_path}. Using defaults.")
        return get_default_config(), errors

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config is None:
            errors.append(f"Config file is empty: {config_path}. Using defaults.")
            return get_default_config(), errors
        return config, errors
    except yaml.YAMLError as e:
        errors.append(
            f"YAML parse error in {config_path}: {e}. "
            "Check indentation and syntax. Using defaults."
        )
        return get_default_config(), errors
    except IOError as e:
        errors.append(f"Cannot read {config_path}: {e}. Using defaults.")
        return get_default_config(), errors


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
