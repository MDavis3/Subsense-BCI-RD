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

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from subsense_bci.physics.constants import (
    CLOUD_VOLUME_SIDE_MM,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SENSOR_COUNT,
    VESSEL_AXIAL_DENSITY,
    VESSEL_BRANCH_ANGLE_DEG,
    VESSEL_N_BRANCHES,
    VESSEL_RADIAL_STD_MM,
    VESSEL_SEGMENT_LENGTH_MM,
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


# Backward-compatible alias
generate_uniform_cloud = generate_sensor_cloud


# =============================================================================
# Vascular Sensor Distribution - Subsense Intravascular Delivery Model
# =============================================================================


@dataclass
class VesselSegment:
    """
    A single vessel segment defined by start and end points.

    Attributes
    ----------
    start : np.ndarray
        Start position (3,) in mm.
    end : np.ndarray
        End position (3,) in mm.
    radius_mm : float
        Vessel radius in mm (for visualization, not sampling).
    branch_id : int
        Identifier for hierarchical tracking (0 = root vessel).
    parent_id : int
        Parent branch ID (-1 for root vessel).

    Properties
    ----------
    length : float
        Segment length in mm.
    direction : np.ndarray
        Unit vector along segment axis.
    """

    start: np.ndarray
    end: np.ndarray
    radius_mm: float = 0.1
    branch_id: int = 0
    parent_id: int = -1

    def __post_init__(self) -> None:
        """Convert to numpy arrays."""
        self.start = np.asarray(self.start, dtype=np.float64)
        self.end = np.asarray(self.end, dtype=np.float64)

    @property
    def length(self) -> float:
        """Segment length in mm."""
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        """Unit vector along segment axis."""
        length = self.length
        if length < 1e-10:
            return np.array([0.0, 0.0, 1.0])
        return (self.end - self.start) / length

    def point_at(self, t: float) -> np.ndarray:
        """
        Get point along segment at parameter t in [0, 1].

        Parameters
        ----------
        t : float
            Parameter in [0, 1], where 0 is start and 1 is end.

        Returns
        -------
        np.ndarray
            Position (3,) in mm.
        """
        return self.start + t * (self.end - self.start)


def generate_vessel_tree(
    root_position: np.ndarray | None = None,
    root_direction: np.ndarray | None = None,
    n_branches: int = VESSEL_N_BRANCHES,
    branch_angle_deg: float = VESSEL_BRANCH_ANGLE_DEG,
    segment_length_mm: float = VESSEL_SEGMENT_LENGTH_MM,
    vessel_radius_mm: float = 0.1,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[VesselSegment]:
    """
    Generate a branching vessel tree using L-system-like recursion.

    Creates a simplified vascular structure for simulating intravascular
    sensor delivery. The tree starts from a root position and branches
    with randomized angles.

    Parameters
    ----------
    root_position : np.ndarray, optional
        Starting position (3,) in mm. Default is [0, 0, 0.4] (near top of volume).
    root_direction : np.ndarray, optional
        Initial direction (3,). Default is [0, 0, -1] (downward).
    n_branches : int, optional
        Number of branch generations. Default is 5.
    branch_angle_deg : float, optional
        Branch angle from parent direction in degrees. Default is 30.
    segment_length_mm : float, optional
        Length of each vessel segment in mm. Default is 0.3.
    vessel_radius_mm : float, optional
        Vessel radius for visualization. Default is 0.1.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    list[VesselSegment]
        List of vessel segments forming the tree.

    Notes
    -----
    The tree uses a binary branching model where each segment can split
    into two children with randomized angular offsets. This creates a
    simplified representation of nasal/vascular anatomy.

    Examples
    --------
    >>> vessels = generate_vessel_tree(n_branches=3, seed=42)
    >>> len(vessels)  # Root + children
    7
    >>> vessels[0].branch_id
    0
    """
    np.random.seed(seed)

    if root_position is None:
        root_position = np.array([0.0, 0.0, 0.4])
    else:
        root_position = np.asarray(root_position, dtype=np.float64)

    if root_direction is None:
        root_direction = np.array([0.0, 0.0, -1.0])
    else:
        root_direction = np.asarray(root_direction, dtype=np.float64)
        root_direction = root_direction / np.linalg.norm(root_direction)

    vessels: list[VesselSegment] = []
    branch_counter = 0

    def _create_perpendicular_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create two perpendicular unit vectors to the given direction."""
        # Find a vector not parallel to direction
        if abs(direction[0]) < 0.9:
            perp1 = np.cross(direction, np.array([1.0, 0.0, 0.0]))
        else:
            perp1 = np.cross(direction, np.array([0.0, 1.0, 0.0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)
        return perp1, perp2

    def _rotate_direction(
        direction: np.ndarray, angle_rad: float, axis: np.ndarray
    ) -> np.ndarray:
        """Rotate direction around axis by angle using Rodrigues' formula."""
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        return (
            direction * cos_a
            + np.cross(axis, direction) * sin_a
            + axis * np.dot(axis, direction) * (1 - cos_a)
        )

    def _add_branch(
        start: np.ndarray,
        direction: np.ndarray,
        depth: int,
        parent_id: int,
    ) -> None:
        """Recursively add vessel branches."""
        nonlocal branch_counter

        if depth > n_branches:
            return

        # Create this segment
        end = start + direction * segment_length_mm

        # Keep within bounds (optional, for bounded volumes)
        # end = np.clip(end, -0.5, 0.5)

        segment = VesselSegment(
            start=start.copy(),
            end=end.copy(),
            radius_mm=vessel_radius_mm * (0.9**depth),  # Taper with depth
            branch_id=branch_counter,
            parent_id=parent_id,
        )
        vessels.append(segment)
        current_id = branch_counter
        branch_counter += 1

        # Branch into two children with angular offset
        perp1, perp2 = _create_perpendicular_basis(direction)
        angle_rad = np.deg2rad(branch_angle_deg)

        # Random azimuthal angles for branch diversity
        phi1 = np.random.uniform(0, 2 * np.pi)
        phi2 = phi1 + np.pi + np.random.uniform(-0.3, 0.3)  # Roughly opposite

        # First child
        axis1 = np.cos(phi1) * perp1 + np.sin(phi1) * perp2
        dir1 = _rotate_direction(direction, angle_rad, axis1)
        _add_branch(end, dir1, depth + 1, current_id)

        # Second child (only at some depths for asymmetry)
        if depth < n_branches - 1 and np.random.random() > 0.3:
            axis2 = np.cos(phi2) * perp1 + np.sin(phi2) * perp2
            dir2 = _rotate_direction(direction, angle_rad * 0.8, axis2)
            _add_branch(end, dir2, depth + 1, current_id)

    # Start tree generation
    _add_branch(root_position, root_direction, depth=0, parent_id=-1)

    return vessels


def sample_sensors_along_vessels(
    vessels: list[VesselSegment],
    n_sensors: int = DEFAULT_SENSOR_COUNT,
    radial_std_mm: float = VESSEL_RADIAL_STD_MM,
    axial_density: float = VESSEL_AXIAL_DENSITY,
    seed: int = DEFAULT_RANDOM_SEED,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample sensor positions with higher density along vessel axes.

    Distributes sensors along the vessel tree centerlines with Gaussian
    radial spread perpendicular to each vessel axis. This simulates
    intravascular delivery where sensors follow blood flow paths.

    Parameters
    ----------
    vessels : list[VesselSegment]
        Vessel segments from generate_vessel_tree().
    n_sensors : int, optional
        Target number of sensors. Default is 10,000.
    radial_std_mm : float, optional
        Gaussian spread from vessel axis in mm. Default is 0.05.
    axial_density : float, optional
        Target sensors per mm along vessel. Default is 100.
        Actual density may vary based on n_sensors constraint.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    return_metadata : bool, optional
        If True, return vessel_ids and phase_offsets. Default is False.

    Returns
    -------
    positions : np.ndarray
        Sensor positions with shape (n_sensors, 3) in mm.
    vessel_ids : np.ndarray (if return_metadata=True)
        Vessel assignment for each sensor, shape (n_sensors,).
    phase_offsets : np.ndarray (if return_metadata=True)
        Cardiac phase offset for each sensor, shape (n_sensors,).
        Sensors in the same vessel share similar phases.

    Examples
    --------
    >>> vessels = generate_vessel_tree(n_branches=3)
    >>> positions = sample_sensors_along_vessels(vessels, n_sensors=1000)
    >>> positions.shape
    (1000, 3)
    """
    np.random.seed(seed)

    # Calculate total vessel length and sensors per segment
    total_length = sum(v.length for v in vessels)
    if total_length < 1e-10:
        raise ValueError("Vessel tree has zero total length")

    # Distribute sensors proportionally to segment length
    all_positions = []
    all_vessel_ids = []
    all_phases = []

    # First pass: generate all positions
    for vessel in vessels:
        segment_length = vessel.length
        # Number of sensors proportional to length
        n_on_segment = max(1, int(segment_length * axial_density))

        # Axial positions (uniform along segment)
        t_values = np.linspace(0, 1, n_on_segment, endpoint=False)
        # Add small jitter to avoid perfectly regular spacing
        t_values += np.random.uniform(0, 0.5 / n_on_segment, n_on_segment)
        t_values = np.clip(t_values, 0, 1)

        axial_positions = np.array([vessel.point_at(t) for t in t_values])

        # Create perpendicular basis for radial offsets
        direction = vessel.direction
        if abs(direction[0]) < 0.9:
            perp1 = np.cross(direction, np.array([1.0, 0.0, 0.0]))
        else:
            perp1 = np.cross(direction, np.array([0.0, 1.0, 0.0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)

        # Radial Gaussian offsets
        r = np.abs(np.random.randn(n_on_segment) * radial_std_mm)
        theta = np.random.uniform(0, 2 * np.pi, n_on_segment)

        offsets = (
            r[:, np.newaxis]
            * (np.cos(theta)[:, np.newaxis] * perp1 + np.sin(theta)[:, np.newaxis] * perp2)
        )

        positions = axial_positions + offsets
        all_positions.append(positions)
        all_vessel_ids.append(np.full(n_on_segment, vessel.branch_id))

        # Phase offset: sensors in same vessel have similar cardiac phase
        # Base phase for this vessel + small per-sensor variation
        vessel_base_phase = np.random.uniform(0, 2 * np.pi)
        sensor_phases = vessel_base_phase + np.random.randn(n_on_segment) * 0.2
        all_phases.append(sensor_phases)

    # Concatenate all segments
    all_positions = np.vstack(all_positions)
    all_vessel_ids = np.concatenate(all_vessel_ids)
    all_phases = np.concatenate(all_phases)

    # Subsample or pad to exactly n_sensors
    n_generated = len(all_positions)
    if n_generated > n_sensors:
        # Random subsample
        indices = np.random.choice(n_generated, n_sensors, replace=False)
        all_positions = all_positions[indices]
        all_vessel_ids = all_vessel_ids[indices]
        all_phases = all_phases[indices]
    elif n_generated < n_sensors:
        # Upsample by adding duplicates with extra noise
        n_extra = n_sensors - n_generated
        extra_indices = np.random.choice(n_generated, n_extra, replace=True)
        extra_positions = all_positions[extra_indices] + np.random.randn(n_extra, 3) * radial_std_mm * 0.5
        extra_vessel_ids = all_vessel_ids[extra_indices]
        extra_phases = all_phases[extra_indices] + np.random.randn(n_extra) * 0.1

        all_positions = np.vstack([all_positions, extra_positions])
        all_vessel_ids = np.concatenate([all_vessel_ids, extra_vessel_ids])
        all_phases = np.concatenate([all_phases, extra_phases])

    all_positions = all_positions.astype(np.float64)
    all_vessel_ids = all_vessel_ids.astype(np.int32)
    all_phases = all_phases.astype(np.float64)

    if return_metadata:
        return all_positions, all_vessel_ids, all_phases
    return all_positions


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

