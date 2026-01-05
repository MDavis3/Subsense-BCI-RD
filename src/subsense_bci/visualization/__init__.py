"""
Visualization Module

Contains shared theming, color palettes, and matplotlib styling
for consistent dark-mode dashboards across all notebooks.
"""

from .theme import COLORS, apply_dark_theme, setup_axis_style

__all__ = ["COLORS", "apply_dark_theme", "setup_axis_style"]

