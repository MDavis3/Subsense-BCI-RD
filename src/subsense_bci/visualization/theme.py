"""
Subsense BCI Dark Lab Theme

Consolidated color palette and matplotlib styling for all visualization
dashboards. This module provides a consistent "dark lab" aesthetic across
Phase 1, 2, and 3 visualizations.

Usage:
    from subsense_bci.visualization.theme import COLORS, apply_dark_theme, setup_axis_style
    
    apply_dark_theme()
    fig, ax = plt.subplots()
    setup_axis_style(ax, "My Title")
"""

from __future__ import annotations

import matplotlib.pyplot as plt


# =============================================================================
# SUBSENSE COLOR PALETTE
# Consolidated from: visualize_cloud.py, visualize_signals.py, validate_unmixing.py
# =============================================================================

COLORS: dict[str, str] = {
    # Base theme - dark backgrounds
    "background": "#0f0f0f",
    "panel_bg": "#12121a",
    "grid_line": "#1a1a2e",
    
    # Primary accent - nanotech cyan
    "nanotech_cyan": "#00FFFF",
    "text_accent": "#00FFFF",
    
    # Status indicators
    "success_green": "#00FF88",
    "warning_red": "#FF3333",
    "safety_yellow": "#FFD700",
    
    # Text hierarchy
    "text_primary": "#E0E0E0",
    "text_secondary": "#808080",
    
    # Source waveform colors (Phase 2+)
    "source_a": "#FF6B6B",  # Alpha 10Hz - warm red
    "source_b": "#4ECDC4",  # Beta 20Hz - teal
    "source_c": "#95E1D3",  # Pink noise - soft green
    
    # ICA recovered signal (Phase 3)
    "recovered": "#FFD93D",  # Gold overlay
    
    # Legacy aliases for backward compatibility
    "glow_red": "#FF6666",
}


def apply_dark_theme() -> None:
    """
    Configure matplotlib for the dark lab aesthetic.
    
    Call this at the start of any visualization script to ensure
    consistent styling across all dashboards.
    
    Example
    -------
    >>> from subsense_bci.visualization.theme import apply_dark_theme
    >>> apply_dark_theme()
    >>> fig, ax = plt.subplots()
    """
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = COLORS["background"]
    plt.rcParams["axes.facecolor"] = COLORS["panel_bg"]
    plt.rcParams["savefig.facecolor"] = COLORS["background"]
    
    # Additional dark theme refinements
    plt.rcParams["axes.edgecolor"] = COLORS["grid_line"]
    plt.rcParams["axes.labelcolor"] = COLORS["text_secondary"]
    plt.rcParams["xtick.color"] = COLORS["text_secondary"]
    plt.rcParams["ytick.color"] = COLORS["text_secondary"]
    plt.rcParams["grid.color"] = COLORS["grid_line"]
    plt.rcParams["text.color"] = COLORS["text_primary"]


def setup_axis_style(ax, title: str) -> None:
    """
    Apply consistent dark styling to a matplotlib axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to style.
    title : str
        Title text for the axis.
        
    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> setup_axis_style(ax, "LEAD FIELD ANALYSIS")
    """
    ax.set_facecolor(COLORS["panel_bg"])
    ax.set_title(
        title,
        color=COLORS["text_accent"],
        fontsize=11,
        fontweight="bold",
        fontfamily="monospace",
        pad=10,
    )
    ax.tick_params(colors=COLORS["text_secondary"], labelsize=8)
    ax.grid(True, alpha=0.2, color=COLORS["grid_line"])
    
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid_line"])

