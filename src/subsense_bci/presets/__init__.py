"""
Demo Presets for Subsense R&D Signal Bench

Pre-configured simulation scenarios for common research use cases.
"""

from __future__ import annotations

from subsense_bci.presets.demo_presets import (
    EXERCISE_STRESS,
    HIGH_DENSITY,
    LOW_LATENCY,
    NANOPARTICLE_RESEARCH,
    PRESETS,
    STANDARD_CARDIAC,
    STRESS_TEST_10K,
    get_preset,
    get_preset_names_and_descriptions,
    list_presets,
)

__all__ = [
    "STANDARD_CARDIAC",
    "EXERCISE_STRESS",
    "HIGH_DENSITY",
    "LOW_LATENCY",
    "NANOPARTICLE_RESEARCH",
    "STRESS_TEST_10K",
    "PRESETS",
    "get_preset",
    "get_preset_names_and_descriptions",
    "list_presets",
]
