"""
Cloud Generator Module - Phase 1 Stochastic Nanoparticle Cloud

Generates uniformly distributed sensor coordinates within a cubic volume
for Subsense volumetric BCI forward modeling.

Physics: Uniform random distribution within a 1mm^3 cube centered at origin.

Note on collision detection:
    With 10,000 particles (r=100nm) in 1mm^3, the volume fraction is ~0.004%.
    At this density, particle collisions are statistically negligible and
    collision checking is omitted for computational efficiency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from subsense_bci.physics.constants import (
    CLOUD_VOLUME_SIDE_MM,
    DEFAULT_SENSOR_COUNT,
    DEFAULT_RANDOM_SEED,
)


def generate_sensor_cloud(
    n_sensors: int = DEFAULT_SENSOR_COUNT,
    volume_side_mm: float = CLOUD_VOLUME_SIDE_MM,
    seed: int = DEFAULT_RANDOM_SEED,
) -> np.ndarray:
    """
    Generate uniformly distributed sensor coordinates within a cubic volume.

    Parameters
    ----------
    n_sensors : int, optional
        Number of sensors to generate. Default is 10,000.
    volume_side_mm : float, optional
        Side length of the cubic volume in mm. Default is 1.0 mm.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    np.ndarray
        Sensor coordinates with shape (n_sensors, 3).
        Each row is [x, y, z] in mm, centered at origin.
        Coordinates range from [-volume_side_mm/2, +volume_side_mm/2].

    Examples
    --------
    >>> sensors = generate_sensor_cloud(n_sensors=1000, seed=42)
    >>> sensors.shape
    (1000, 3)
    >>> np.all(np.abs(sensors) <= 0.5)
    True
    """
    # Set seed for reproducibility (per .cursorrules mandate)
    np.random.seed(seed)

    # Generate uniform random coordinates in [-0.5, 0.5] * volume_side_mm
    half_side = volume_side_mm / 2.0
    sensors = np.random.uniform(
        low=-half_side,
        high=half_side,
        size=(n_sensors, 3)
    )

    # Validate bounds (should always pass, but explicit check for safety)
    assert np.all(np.abs(sensors) <= half_side), "Sensor coordinates out of bounds!"

    return sensors.astype(np.float64)


def save_sensor_cloud(
    sensors: np.ndarray,
    output_dir: Path | str = "data/raw",
    n_sensors: int | None = None,
    seed: int | None = None,
) -> Path:
    """
    Save sensor coordinates to a NumPy binary file.

    Parameters
    ----------
    sensors : np.ndarray
        Sensor coordinates with shape (n_sensors, 3).
    output_dir : Path or str, optional
        Directory to save the file. Default is "data/raw".
    n_sensors : int, optional
        Number of sensors (for filename). Inferred from array if not provided.
    seed : int, optional
        Random seed used (for filename). Default is None (not included in name).

    Returns
    -------
    Path
        Path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename with parameters
    n = n_sensors if n_sensors is not None else sensors.shape[0]
    if seed is not None:
        filename = f"sensors_N{n}_seed{seed}.npy"
    else:
        filename = f"sensors_N{n}.npy"

    filepath = output_dir / filename
    np.save(filepath, sensors)

    return filepath


def load_sensor_cloud(filepath: Path | str) -> np.ndarray:
    """
    Load sensor coordinates from a NumPy binary file.

    Parameters
    ----------
    filepath : Path or str
        Path to the .npy file.

    Returns
    -------
    np.ndarray
        Sensor coordinates with shape (n_sensors, 3).
    """
    return np.load(filepath)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent.parent


def main() -> None:
    """Generate and save the default sensor cloud."""
    print(f"Generating sensor cloud: N={DEFAULT_SENSOR_COUNT}, seed={DEFAULT_RANDOM_SEED}")

    sensors = generate_sensor_cloud()

    print(f"  Shape: {sensors.shape}")
    print(f"  Bounds: [{sensors.min():.4f}, {sensors.max():.4f}] mm")
    print(f"  Mean: {sensors.mean(axis=0)}")

    # Determine output directory relative to project root
    output_dir = get_project_root() / "data" / "raw"

    filepath = save_sensor_cloud(
        sensors,
        output_dir=output_dir,
        n_sensors=DEFAULT_SENSOR_COUNT,
        seed=DEFAULT_RANDOM_SEED,
    )

    print(f"  Saved to: {filepath}")


if __name__ == "__main__":
    main()

