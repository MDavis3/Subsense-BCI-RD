"""
Demo Presets for Subsense R&D Signal Bench

Pre-configured simulation scenarios for common research use cases.
"""

from __future__ import annotations

from subsense_bci.presets.demo_presets import (
    EXERCISE_STRESS,
    HIGH_DENSITY,
    PRESETS,
    STANDARD_CARDIAC,
    get_preset,
    get_preset_names_and_descriptions,
    list_presets,
)

__all__ = [
    "STANDARD_CARDIAC",
    "EXERCISE_STRESS",
    "HIGH_DENSITY",
    "PRESETS",
    "get_preset",
    "get_preset_names_and_descriptions",
    "list_presets",
]
