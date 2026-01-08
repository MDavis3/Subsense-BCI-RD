"""
Input Validators for Subsense R&D Signal Bench

Provides robust validation for:
- Nyquist theorem compliance
- Real-time latency budget enforcement
- YAML configuration file parsing

These validators ensure non-expert researchers receive helpful feedback
rather than cryptic errors when parameters are invalid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from subsense_bci.physics.constants import (
    REALTIME_LATENCY_BUDGET_MS,
    SAMPLING_RATE_HZ,
)


# =============================================================================
# Validation Result Types
# =============================================================================


@dataclass
class NyquistResult:
    """Result of Nyquist theorem validation.

    Attributes
    ----------
    is_valid : bool
        True if sampling rate satisfies Nyquist criterion.
    sampling_rate_hz : float
        The sampling rate being validated.
    max_frequency_hz : float
        Maximum frequency in the signal.
    nyquist_frequency_hz : float
        Nyquist frequency (sampling_rate / 2).
    oversampling_factor : float
        Ratio of sampling rate to twice the max frequency.
    warnings : list[str]
        Non-fatal warnings (e.g., marginal oversampling).
    errors : list[str]
        Fatal errors (e.g., Nyquist violation).
    recovery_suggestions : list[str]
        Actionable suggestions for fixing issues.
    """

    is_valid: bool
    sampling_rate_hz: float
    max_frequency_hz: float
    nyquist_frequency_hz: float
    oversampling_factor: float
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    recovery_suggestions: list[str] = field(default_factory=list)


@dataclass
class BudgetResult:
    """Result of real-time budget validation.

    Attributes
    ----------
    is_within_budget : bool
        True if estimated latency is within budget.
    estimated_latency_ms : float
        Estimated processing latency.
    budget_ms : float
        The latency budget constraint.
    budget_utilization : float
        Ratio of estimated to budget (1.0 = exactly at budget).
    filter_type : str
        The filter type being validated.
    n_sensors : int
        Number of sensors in configuration.
    n_taps : int
        Number of filter taps.
    warnings : list[str]
        Non-fatal warnings (e.g., approaching budget limit).
    errors : list[str]
        Fatal errors (e.g., budget exceeded).
    recovery_suggestions : list[str]
        Actionable suggestions for optimization.
    """

    is_within_budget: bool
    estimated_latency_ms: float
    budget_ms: float
    budget_utilization: float
    filter_type: str
    n_sensors: int
    n_taps: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    recovery_suggestions: list[str] = field(default_factory=list)


@dataclass
class ConfigValidationResult:
    """Result of YAML configuration file validation.

    Attributes
    ----------
    is_valid : bool
        True if config loaded and validated successfully.
    config : dict | None
        Loaded configuration (None if load failed).
    file_path : Path | None
        Path to the config file (None if using defaults).
    warnings : list[str]
        Non-fatal warnings (e.g., missing optional fields).
    errors : list[str]
        Fatal errors (e.g., parse failures).
    recovery_suggestions : list[str]
        Actionable suggestions for fixing issues.
    """

    is_valid: bool
    config: dict[str, Any] | None
    file_path: Path | None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    recovery_suggestions: list[str] = field(default_factory=list)


# =============================================================================
# Nyquist Validation
# =============================================================================


def validate_nyquist(
    sampling_rate_hz: float,
    source_frequencies: dict[str, float] | list[float] | None = None,
    min_oversampling_factor: float = 2.5,
) -> NyquistResult:
    """
    Validate that sampling rate satisfies Nyquist criterion.

    The Nyquist theorem requires: fs > 2 * f_max
    For practical signal processing, we recommend fs >= 2.5 * f_max.

    Parameters
    ----------
    sampling_rate_hz : float
        Sampling rate of the system.
    source_frequencies : dict or list, optional
        Frequencies present in the signal. If dict, uses values.
        Default assumes alpha (10 Hz) and beta (20 Hz) bands.
    min_oversampling_factor : float, optional
        Minimum recommended oversampling factor. Default 2.5.

    Returns
    -------
    NyquistResult
        Validation result with is_valid status and diagnostic info.

    Examples
    --------
    >>> result = validate_nyquist(1000, {"alpha": 10, "beta": 20})
    >>> result.is_valid
    True
    >>> result.oversampling_factor
    25.0

    >>> result = validate_nyquist(30, {"alpha": 10, "beta": 20})
    >>> result.is_valid
    False
    >>> result.errors
    ['NYQUIST VIOLATION: ...']
    """
    # Default frequencies if not provided
    if source_frequencies is None:
        source_frequencies = {"alpha": 10.0, "beta": 20.0}

    # Extract max frequency
    if isinstance(source_frequencies, dict):
        freqs = list(source_frequencies.values())
    else:
        freqs = list(source_frequencies)

    max_freq = max(freqs) if freqs else 0.0
    nyquist_freq = sampling_rate_hz / 2.0

    # Calculate oversampling
    if max_freq > 0:
        oversampling = sampling_rate_hz / (2.0 * max_freq)
    else:
        oversampling = float("inf")

    warnings = []
    errors = []
    suggestions = []

    # Check Nyquist criterion
    is_valid = True

    if sampling_rate_hz <= 2 * max_freq:
        is_valid = False
        errors.append(
            f"NYQUIST VIOLATION: Sampling rate ({sampling_rate_hz} Hz) must be "
            f"> 2 * max_frequency ({2 * max_freq} Hz). "
            f"Current: {sampling_rate_hz} Hz <= {2 * max_freq} Hz."
        )
        suggestions.append(
            f"Increase sampling rate to at least {2.5 * max_freq:.1f} Hz, "
            f"or reduce max source frequency below {nyquist_freq:.1f} Hz."
        )

    elif oversampling < min_oversampling_factor:
        warnings.append(
            f"MARGINAL OVERSAMPLING: Sampling rate ({sampling_rate_hz} Hz) "
            f"provides only {oversampling:.2f}x oversampling. "
            f"Recommended minimum is {min_oversampling_factor}x for robust filtering."
        )
        suggestions.append(
            f"Consider increasing sampling rate to {min_oversampling_factor * 2 * max_freq:.1f} Hz "
            "for better anti-aliasing margin."
        )

    return NyquistResult(
        is_valid=is_valid,
        sampling_rate_hz=sampling_rate_hz,
        max_frequency_hz=max_freq,
        nyquist_frequency_hz=nyquist_freq,
        oversampling_factor=oversampling,
        warnings=warnings,
        errors=errors,
        recovery_suggestions=suggestions,
    )


# =============================================================================
# Real-Time Budget Validation
# =============================================================================


def estimate_processing_time_ms(
    n_sensors: int,
    n_taps: int,
    chunk_samples: int = 100,
    filter_type: str = "rls",
) -> float:
    """
    Estimate processing time for adaptive filtering.

    Complexity analysis:
    - RLS: O(n_taps^2) per sample (matrix operations)
    - LMS: O(n_taps) per sample (vector operations)

    Parameters
    ----------
    n_sensors : int
        Number of sensors to process.
    n_taps : int
        Number of filter taps.
    chunk_samples : int
        Samples per chunk (default 100 for 100ms at 1kHz).
    filter_type : str
        Filter type ("lms", "rls", or "phase_aware_rls").

    Returns
    -------
    float
        Estimated processing time in milliseconds.
    """
    filter_type = filter_type.lower()

    if filter_type in ["rls", "phase_aware_rls"]:
        ops_per_sample = n_taps**2  # Matrix update
    else:  # LMS
        ops_per_sample = n_taps  # Vector update

    total_ops = n_sensors * chunk_samples * ops_per_sample

    # Empirical conversion: ~66 GFLOP/s typical single-thread NumPy
    # Conservative estimate accounts for memory bandwidth
    gflops = total_ops / 1e9
    estimated_ms = gflops * 15.0  # ~15ms per GFLOP

    return estimated_ms


def validate_realtime_budget(
    filter_type: str,
    n_sensors: int,
    n_taps: int,
    chunk_samples: int = 100,
    budget_ms: float | None = None,
    warning_threshold: float = 0.8,
) -> BudgetResult:
    """
    Validate that filter configuration meets real-time latency budget.

    The SubSense BCI system requires processing within 43ms for real-time
    neural control. This validator checks if the proposed configuration
    can meet that constraint.

    Parameters
    ----------
    filter_type : str
        Filter type ("lms", "rls", or "phase_aware_rls").
    n_sensors : int
        Number of sensors to process.
    n_taps : int
        Number of filter taps.
    chunk_samples : int
        Samples per chunk. Default 100 (100ms at 1kHz).
    budget_ms : float, optional
        Latency budget. Default is REALTIME_LATENCY_BUDGET_MS (43ms).
    warning_threshold : float
        Fraction of budget that triggers warning. Default 0.8 (80%).

    Returns
    -------
    BudgetResult
        Validation result with budget status and optimization suggestions.

    Examples
    --------
    >>> result = validate_realtime_budget("rls", 1000, 8)
    >>> result.is_within_budget
    True

    >>> result = validate_realtime_budget("rls", 10000, 32)
    >>> result.is_within_budget
    False
    >>> result.errors
    ['BUDGET EXCEEDED: ...']
    """
    if budget_ms is None:
        budget_ms = REALTIME_LATENCY_BUDGET_MS

    estimated = estimate_processing_time_ms(
        n_sensors=n_sensors,
        n_taps=n_taps,
        chunk_samples=chunk_samples,
        filter_type=filter_type,
    )

    utilization = estimated / budget_ms

    warnings = []
    errors = []
    suggestions = []

    is_within_budget = estimated < budget_ms

    if not is_within_budget:
        errors.append(
            f"BUDGET EXCEEDED: Estimated latency ({estimated:.1f}ms) exceeds "
            f"real-time budget ({budget_ms}ms) by {estimated - budget_ms:.1f}ms. "
            f"Config: {filter_type.upper()}, {n_sensors:,} sensors, {n_taps} taps."
        )

        # Calculate what would fit in budget
        if filter_type.lower() in ["rls", "phase_aware_rls"]:
            # For RLS: latency ~ n_taps^2, so reduce taps
            max_taps = int((n_taps**2 * budget_ms * 0.8 / estimated) ** 0.5)
            suggestions.append(
                f"Reduce n_taps from {n_taps} to {max_taps} or fewer."
            )
            suggestions.append(
                f"Or switch to LMS filter (O(n_taps) vs O(n_taps^2))."
            )
        else:
            # For LMS: latency ~ n_taps
            max_taps = int(n_taps * budget_ms * 0.8 / estimated)
            suggestions.append(f"Reduce n_taps from {n_taps} to {max_taps}.")

        # Suggest sensor reduction as last resort
        max_sensors = int(n_sensors * budget_ms * 0.8 / estimated)
        suggestions.append(
            f"Or reduce sensors from {n_sensors:,} to {max_sensors:,}."
        )

    elif utilization > warning_threshold:
        warnings.append(
            f"APPROACHING BUDGET LIMIT: Estimated latency ({estimated:.1f}ms) "
            f"uses {utilization * 100:.0f}% of {budget_ms}ms budget. "
            f"Headroom: {budget_ms - estimated:.1f}ms."
        )
        suggestions.append(
            "Consider reducing n_taps or sensor count for safety margin."
        )

    return BudgetResult(
        is_within_budget=is_within_budget,
        estimated_latency_ms=estimated,
        budget_ms=budget_ms,
        budget_utilization=utilization,
        filter_type=filter_type,
        n_sensors=n_sensors,
        n_taps=n_taps,
        warnings=warnings,
        errors=errors,
        recovery_suggestions=suggestions,
    )


# =============================================================================
# Configuration File Validation
# =============================================================================

# Required sections in config
REQUIRED_CONFIG_SECTIONS = ["cloud", "temporal", "physics"]

# Type specifications for validation
CONFIG_TYPE_SPECS = {
    "cloud": {
        "sensor_count": (int, 1, 100_000),
        "volume_side_mm": (float, 0.1, 100.0),
    },
    "temporal": {
        "sampling_rate_hz": (float, 1.0, 100_000.0),
        "duration_sec": (float, 0.1, 3600.0),
        "snr_level": (float, 0.1, 1000.0),
    },
    "physics": {
        "brain_conductivity_s_m": (float, 0.01, 10.0),
    },
}


def validate_config_file(
    config_path: Path | str | None = None,
    strict: bool = False,
) -> ConfigValidationResult:
    """
    Load and validate YAML configuration file.

    Provides graceful error handling with helpful messages for:
    - Missing files (falls back to defaults)
    - Malformed YAML (syntax errors)
    - Invalid parameter values (type/range checks)

    Parameters
    ----------
    config_path : Path or str, optional
        Path to YAML config file. If None, uses default_sim.yaml.
    strict : bool
        If True, treat warnings as errors. Default False.

    Returns
    -------
    ConfigValidationResult
        Validation result with loaded config and any issues found.

    Examples
    --------
    >>> result = validate_config_file("configs/default_sim.yaml")
    >>> result.is_valid
    True
    >>> result.config["temporal"]["sampling_rate_hz"]
    1000.0

    >>> result = validate_config_file("nonexistent.yaml")
    >>> result.is_valid
    True  # Falls back to defaults
    >>> len(result.warnings) > 0
    True
    """
    from subsense_bci.config import DEFAULT_CONFIG_PATH, get_default_config

    warnings = []
    errors = []
    suggestions = []

    # Determine file path
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path)

    # Attempt to load file
    config = None
    file_exists = config_path.exists()

    if not file_exists:
        warnings.append(
            f"CONFIG FILE NOT FOUND: '{config_path}' does not exist. "
            "Using built-in defaults."
        )
        suggestions.append(
            f"Create config file at '{config_path}' or use load_config() without path."
        )
        config = get_default_config()
    else:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            # Handle empty YAML file (returns None)
            if config is None:
                warnings.append(
                    f"CONFIG FILE EMPTY: '{config_path}' contains no data. "
                    "Using built-in defaults."
                )
                config = get_default_config()
        except yaml.YAMLError as e:
            errors.append(
                f"YAML PARSE ERROR in '{config_path}': {str(e)}"
            )
            suggestions.append(
                "Check YAML syntax: proper indentation (2 spaces), "
                "colons after keys, no tabs."
            )
            suggestions.append(
                "Use a YAML validator: https://www.yamllint.com/"
            )
            config = get_default_config()
        except IOError as e:
            errors.append(
                f"FILE READ ERROR for '{config_path}': {str(e)}"
            )
            suggestions.append("Check file permissions and path.")
            config = get_default_config()

    # Validate structure if we have a config
    if config is not None:
        # Check required sections
        for section in REQUIRED_CONFIG_SECTIONS:
            if section not in config:
                if strict:
                    errors.append(
                        f"MISSING REQUIRED SECTION: '{section}' not found in config."
                    )
                else:
                    warnings.append(
                        f"MISSING SECTION: '{section}' not found. Using defaults."
                    )
                # Merge in defaults for missing section
                defaults = get_default_config()
                if section in defaults:
                    config[section] = defaults[section]

        # Type and range validation
        for section, specs in CONFIG_TYPE_SPECS.items():
            if section not in config:
                continue
            for param, (expected_type, min_val, max_val) in specs.items():
                if param not in config[section]:
                    continue
                value = config[section][param]

                # Type check
                if not isinstance(value, (expected_type, int if expected_type == float else type(None))):
                    if strict:
                        errors.append(
                            f"TYPE ERROR: {section}.{param} should be {expected_type.__name__}, "
                            f"got {type(value).__name__}."
                        )
                    else:
                        warnings.append(
                            f"TYPE WARNING: {section}.{param} should be {expected_type.__name__}, "
                            f"got {type(value).__name__}. Attempting conversion."
                        )
                        try:
                            config[section][param] = expected_type(value)
                        except (ValueError, TypeError):
                            errors.append(
                                f"CONVERSION FAILED: Cannot convert {section}.{param} "
                                f"value '{value}' to {expected_type.__name__}."
                            )

                # Range check (only for numeric types)
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        warnings.append(
                            f"RANGE WARNING: {section}.{param}={value} is outside "
                            f"expected range [{min_val}, {max_val}]."
                        )

    # Determine overall validity
    is_valid = len(errors) == 0
    if strict:
        is_valid = is_valid and len(warnings) == 0

    return ConfigValidationResult(
        is_valid=is_valid,
        config=config,
        file_path=config_path if file_exists else None,
        warnings=warnings,
        errors=errors,
        recovery_suggestions=suggestions,
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_all(
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
    source_frequencies: dict[str, float] | None = None,
    filter_type: str = "rls",
    n_sensors: int = 1000,
    n_taps: int = 8,
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Run all validators and return combined results.

    Convenience function for the Streamlit dashboard to validate
    all parameters at once.

    Parameters
    ----------
    sampling_rate_hz : float
        Sampling rate.
    source_frequencies : dict, optional
        Signal frequencies.
    filter_type : str
        Filter type.
    n_sensors : int
        Number of sensors.
    n_taps : int
        Number of filter taps.
    config_path : Path or str, optional
        Config file path.

    Returns
    -------
    dict
        Dictionary with "nyquist", "budget", "config" results and
        "all_valid" boolean.
    """
    nyquist_result = validate_nyquist(sampling_rate_hz, source_frequencies)
    budget_result = validate_realtime_budget(filter_type, n_sensors, n_taps)
    config_result = validate_config_file(config_path)

    all_valid = (
        nyquist_result.is_valid
        and budget_result.is_within_budget
        and config_result.is_valid
    )

    return {
        "nyquist": nyquist_result,
        "budget": budget_result,
        "config": config_result,
        "all_valid": all_valid,
    }
