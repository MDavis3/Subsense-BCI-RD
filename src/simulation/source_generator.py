"""
Source Generator Module - Phase 1 Neural Point Sources

Defines fixed neural point source locations for controlled forward modeling
experiments. These sources represent idealized current dipoles in the
volume conductor.

All coordinates are in mm, centered at origin within the 1mm^3 domain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Import constants from physics module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from physics.constants import CLOUD_VOLUME_SIDE_MM


# =============================================================================
# Fixed Source Configurations
# =============================================================================

# Three-source configuration for Phase 1 validation
# Coordinates chosen to be well-separated and within bounds
SOURCES_3FIXED: np.ndarray = np.array([
    [0.2,  0.0, 0.0],   # Source A: Right of center
    [-0.2, 0.2, 0.0],   # Source B: Left-anterior
    [0.0, -0.2, 0.1],   # Source C: Posterior-superior
], dtype=np.float64)


def get_fixed_sources(config: str = "3fixed") -> np.ndarray:
    """
    Get a predefined source configuration.

    Parameters
    ----------
    config : str, optional
        Configuration name. Currently supported: "3fixed".
        Default is "3fixed".

    Returns
    -------
    np.ndarray
        Source coordinates with shape (n_sources, 3) in mm.

    Raises
    ------
    ValueError
        If the configuration name is not recognized.

    Examples
    --------
    >>> sources = get_fixed_sources("3fixed")
    >>> sources.shape
    (3, 3)
    """
    configs = {
        "3fixed": SOURCES_3FIXED,
    }

    if config not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown config '{config}'. Available: {available}")

    return configs[config].copy()


def validate_sources(
    sources: np.ndarray,
    volume_side_mm: float = CLOUD_VOLUME_SIDE_MM,
) -> bool:
    """
    Validate that all sources are within the cubic domain.

    Parameters
    ----------
    sources : np.ndarray
        Source coordinates with shape (n_sources, 3) in mm.
    volume_side_mm : float, optional
        Side length of the cubic volume in mm. Default is 1.0 mm.

    Returns
    -------
    bool
        True if all sources are within bounds.

    Raises
    ------
    ValueError
        If any source is outside the domain.
    """
    half_side = volume_side_mm / 2.0

    if sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {sources.shape}")

    out_of_bounds = np.any(np.abs(sources) > half_side, axis=1)
    if np.any(out_of_bounds):
        bad_indices = np.where(out_of_bounds)[0]
        raise ValueError(
            f"Sources at indices {bad_indices.tolist()} are outside "
            f"the [{-half_side}, {half_side}] mm domain."
        )

    return True


def save_sources(
    sources: np.ndarray,
    output_dir: Path | str = "data/raw",
    name: str = "sources",
) -> Path:
    """
    Save source coordinates to a NumPy binary file.

    Parameters
    ----------
    sources : np.ndarray
        Source coordinates with shape (n_sources, 3).
    output_dir : Path or str, optional
        Directory to save the file. Default is "data/raw".
    name : str, optional
        Base name for the file. Default is "sources".

    Returns
    -------
    Path
        Path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{name}.npy"
    filepath = output_dir / filename

    np.save(filepath, sources)

    return filepath


def load_sources(filepath: Path | str) -> np.ndarray:
    """
    Load source coordinates from a NumPy binary file.

    Parameters
    ----------
    filepath : Path or str
        Path to the .npy file.

    Returns
    -------
    np.ndarray
        Source coordinates with shape (n_sources, 3).
    """
    return np.load(filepath)


def main() -> None:
    """Generate and save the default 3-source configuration."""
    print("Generating fixed source configuration: 3fixed")

    sources = get_fixed_sources("3fixed")

    print(f"  Shape: {sources.shape}")
    print("  Source locations (mm):")
    labels = ["A", "B", "C"]
    for i, (label, coord) in enumerate(zip(labels, sources)):
        print(f"    Source {label}: [{coord[0]:+.2f}, {coord[1]:+.2f}, {coord[2]:+.2f}]")

    # Validate
    validate_sources(sources)
    print("  Validation: All sources within bounds")

    # Determine output directory relative to project root
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "data" / "raw"

    filepath = save_sources(
        sources,
        output_dir=output_dir,
        name="sources_3fixed",
    )

    print(f"  Saved to: {filepath}")


if __name__ == "__main__":
    main()

