"""
Demo Presets for Subsense R&D Signal Bench

Pre-configured simulation scenarios for common research use cases.
Each preset defines a complete parameter set for the Signal Bench dashboard.

Usage:
    from subsense_bci.presets import STANDARD_CARDIAC, get_preset

    # Use preset directly
    params = STANDARD_CARDIAC

    # Or load by name
    params = get_preset("standard_cardiac")
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# Preset Definitions
# =============================================================================


STANDARD_CARDIAC: dict[str, Any] = {
    "name": "Standard Cardiac Interference",
    "description": (
        "Typical resting conditions with moderate cardiac artifact. "
        "Demonstrates PhaseAwareRLS filter effectiveness at 72 BPM."
    ),
    # Sensor configuration
    "n_sensors": 1000,
    # Adaptive filter settings
    "filter_type": "PhaseAwareRLS",
    "n_taps": 8,
    "lambda_": 0.95,
    "mu": 0.01,  # For LMS fallback
    # Cardiac parameters
    "pulse_wave_velocity_m_s": 7.5,
    "drift_amplitude_mm": 0.05,  # 50 microns
    "cardiac_freq_hz": 1.2,  # ~72 BPM
    # Temporal parameters
    "sampling_rate_hz": 1000.0,
    "duration_sec": 2.0,
    "chunk_size_ms": 100.0,
    # Signal parameters
    "snr_level": 5.0,
    "source_frequencies": {"alpha": 10.0, "beta": 20.0},
    # Display
    "display_sensors": 50,
    "window_ms": 500.0,
    # Experimental features
    "nanoparticle_drift_enabled": False,
}


EXERCISE_STRESS: dict[str, Any] = {
    "name": "Exercise Stress Test",
    "description": (
        "High heart rate scenario (150 BPM) with increased motion artifacts. "
        "Tests filter adaptation speed under non-stationary conditions."
    ),
    # Sensor configuration
    "n_sensors": 500,
    # Adaptive filter settings (faster adaptation)
    "filter_type": "PhaseAwareRLS",
    "n_taps": 8,
    "lambda_": 0.90,  # Faster forgetting for rapid changes
    "mu": 0.02,
    # Cardiac parameters (elevated)
    "pulse_wave_velocity_m_s": 10.0,  # Higher during exercise
    "drift_amplitude_mm": 0.10,  # Doubled motion
    "cardiac_freq_hz": 2.5,  # ~150 BPM
    # Temporal parameters
    "sampling_rate_hz": 1000.0,
    "duration_sec": 1.0,  # Shorter for stress test
    "chunk_size_ms": 100.0,
    # Signal parameters
    "snr_level": 3.0,  # More noise during exercise
    "source_frequencies": {"alpha": 10.0, "beta": 20.0},
    # Display
    "display_sensors": 50,
    "window_ms": 500.0,
    # Experimental features
    "nanoparticle_drift_enabled": False,
}


HIGH_DENSITY: dict[str, Any] = {
    "name": "High-Density Array (10k Sensors)",
    "description": (
        "Maximum sensor count stress test. Demonstrates real-time budget "
        "management with 10,000 simultaneous channels."
    ),
    # Sensor configuration
    "n_sensors": 10000,
    # Adaptive filter settings (optimized for throughput)
    "filter_type": "PhaseAwareRLS",
    "n_taps": 8,  # Minimal taps for budget compliance
    "lambda_": 0.95,
    "mu": 0.01,
    # Cardiac parameters
    "pulse_wave_velocity_m_s": 7.5,
    "drift_amplitude_mm": 0.05,
    "cardiac_freq_hz": 1.2,
    # Temporal parameters
    "sampling_rate_hz": 1000.0,
    "duration_sec": 2.0,
    "chunk_size_ms": 100.0,
    # Signal parameters
    "snr_level": 5.0,
    "source_frequencies": {"alpha": 10.0, "beta": 20.0},
    # Display
    "display_sensors": 100,  # Show more sensors
    "window_ms": 500.0,
    # Experimental features
    "nanoparticle_drift_enabled": False,
}


LOW_LATENCY: dict[str, Any] = {
    "name": "Low-Latency Mode",
    "description": (
        "Optimized for minimum processing delay. Uses LMS filter for "
        "O(n) complexity at the cost of slower convergence."
    ),
    # Sensor configuration
    "n_sensors": 2000,
    # Adaptive filter settings (LMS for speed)
    "filter_type": "LMS",
    "n_taps": 16,
    "lambda_": 0.99,
    "mu": 0.005,  # Small step size for stability
    # Cardiac parameters
    "pulse_wave_velocity_m_s": 7.5,
    "drift_amplitude_mm": 0.05,
    "cardiac_freq_hz": 1.2,
    # Temporal parameters
    "sampling_rate_hz": 1000.0,
    "duration_sec": 2.0,
    "chunk_size_ms": 50.0,  # Smaller chunks for lower latency
    # Signal parameters
    "snr_level": 5.0,
    "source_frequencies": {"alpha": 10.0, "beta": 20.0},
    # Display
    "display_sensors": 50,
    "window_ms": 250.0,  # Shorter window
    # Experimental features
    "nanoparticle_drift_enabled": False,
}


NANOPARTICLE_RESEARCH: dict[str, Any] = {
    "name": "Nanoparticle Drift Research (Experimental)",
    "description": (
        "Experimental mode with Brownian drift modeling enabled. "
        "For SubSense 2026 R&D roadmap research."
    ),
    # Sensor configuration
    "n_sensors": 1000,
    # Adaptive filter settings
    "filter_type": "PhaseAwareRLS",
    "n_taps": 8,
    "lambda_": 0.95,
    "mu": 0.01,
    # Cardiac parameters
    "pulse_wave_velocity_m_s": 7.5,
    "drift_amplitude_mm": 0.05,
    "cardiac_freq_hz": 1.2,
    # Temporal parameters
    "sampling_rate_hz": 1000.0,
    "duration_sec": 2.0,
    "chunk_size_ms": 100.0,
    # Signal parameters
    "snr_level": 5.0,
    "source_frequencies": {"alpha": 10.0, "beta": 20.0},
    # Display
    "display_sensors": 50,
    "window_ms": 500.0,
    # Experimental features
    "nanoparticle_drift_enabled": True,
    "nanoparticle_diffusion_coeff_mm2_s": 4.4e-6,
}


# =============================================================================
# Preset Registry
# =============================================================================


PRESETS: dict[str, dict[str, Any]] = {
    "standard_cardiac": STANDARD_CARDIAC,
    "exercise_stress": EXERCISE_STRESS,
    "high_density": HIGH_DENSITY,
    "low_latency": LOW_LATENCY,
    "nanoparticle_research": NANOPARTICLE_RESEARCH,
}


def list_presets() -> list[str]:
    """
    List all available preset names.

    Returns
    -------
    list[str]
        Names of available presets.

    Examples
    --------
    >>> list_presets()
    ['standard_cardiac', 'exercise_stress', 'high_density', 'low_latency', 'nanoparticle_research']
    """
    return list(PRESETS.keys())


def get_preset(name: str) -> dict[str, Any]:
    """
    Get a preset by name.

    Parameters
    ----------
    name : str
        Preset name (case-insensitive, underscores optional).

    Returns
    -------
    dict
        Preset parameter dictionary.

    Raises
    ------
    KeyError
        If preset name not found.

    Examples
    --------
    >>> preset = get_preset("standard_cardiac")
    >>> preset["n_sensors"]
    1000

    >>> preset = get_preset("Standard Cardiac")  # Also works
    >>> preset["filter_type"]
    'PhaseAwareRLS'
    """
    # Normalize name: lowercase, replace spaces with underscores
    normalized = name.lower().replace(" ", "_").replace("-", "_")

    if normalized not in PRESETS:
        available = ", ".join(list_presets())
        raise KeyError(
            f"Preset '{name}' not found. Available presets: {available}"
        )

    # Return a copy to prevent modification of original
    return dict(PRESETS[normalized])


def get_preset_names_and_descriptions() -> list[tuple[str, str, str]]:
    """
    Get all preset names with their display names and descriptions.

    Returns
    -------
    list[tuple[str, str, str]]
        List of (key, display_name, description) tuples.

    Examples
    --------
    >>> presets = get_preset_names_and_descriptions()
    >>> presets[0]
    ('standard_cardiac', 'Standard Cardiac Interference', 'Typical resting...')
    """
    result = []
    for key, preset in PRESETS.items():
        result.append((key, preset["name"], preset["description"]))
    return result
