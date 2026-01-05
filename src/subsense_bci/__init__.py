"""
Subsense BCI R&D - Core Production Code

This package contains the production-ready implementations for:
- Physics: Transfer functions and 3D coordinate mathematics
- Filtering: DSP, ICA, and signal unmixing logic
- Simulation: Cloud generators and lead-field creators
- Visualization: Dark lab theme and styling utilities

Usage:
    # After installing with: pip install -e .
    from subsense_bci.physics.constants import SNR_LEVEL
    from subsense_bci.physics.transfer_function import compute_lead_field
    from subsense_bci.filtering.unmixing import unmix_sources
    from subsense_bci.visualization.theme import COLORS, apply_dark_theme
    from subsense_bci.config import load_config
"""

__version__ = "0.1.0"
__all__ = ["physics", "filtering", "simulation", "visualization", "config"]

