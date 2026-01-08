"""
Validation Module for Subsense BCI R&D Signal Bench

Provides input validation, Nyquist checks, and real-time budget enforcement
for production-grade signal processing.
"""

from __future__ import annotations

from subsense_bci.validation.input_validators import (
    BudgetResult,
    ConfigValidationResult,
    NyquistResult,
    validate_config_file,
    validate_nyquist,
    validate_realtime_budget,
)

__all__ = [
    "NyquistResult",
    "BudgetResult",
    "ConfigValidationResult",
    "validate_nyquist",
    "validate_realtime_budget",
    "validate_config_file",
]
