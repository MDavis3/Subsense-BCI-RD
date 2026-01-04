"""
Transfer Function Module - Lead Field Computation for Volume Conductors

Implements the forward model for point current sources in a homogeneous
isotropic volume conductor. This is the core physics engine for Subsense
BCI signal simulation.

Mathematical Foundation
-----------------------
For a point current source in a homogeneous isotropic conductor, the
electrical potential at distance r is:

    V = I / (4 * pi * sigma * r)

For a unit current source (I = 1 A), the lead field entry is:

    L_ij = 1 / (4 * pi * sigma * r_ij)

where r_ij is the Euclidean distance from sensor i to source j.

Unit Convention
---------------
- Coordinates: mm (at API boundary)
- Conductivity: S/m
- Distances: Converted to meters internally for physical consistency
- Lead Field: V/A (volts per ampere)

Singularity Handling
--------------------
When r approaches zero, V approaches infinity. We use distance clamping
rather than sensor pruning to maintain consistent array shapes:

    r_safe = max(r, SINGULARITY_THRESHOLD_MM)

At 0.05mm with sigma=0.33 S/m: V_max ≈ 4823 V/A
"""

from __future__ import annotations

import numpy as np

from .constants import (
    BRAIN_CONDUCTIVITY_S_M,
    SINGULARITY_THRESHOLD_MM,
)


def compute_lead_field(
    sensors: np.ndarray,
    sources: np.ndarray,
    conductivity: float = BRAIN_CONDUCTIVITY_S_M,
    min_distance_mm: float = SINGULARITY_THRESHOLD_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the lead field matrix for point sources in a homogeneous conductor.

    The lead field L maps source currents to sensor potentials: V = L @ I

    Parameters
    ----------
    sensors : np.ndarray
        Sensor coordinates with shape (n_sensors, 3) in mm.
    sources : np.ndarray
        Source coordinates with shape (n_sources, 3) in mm.
    conductivity : float, optional
        Volume conductor conductivity in S/m. Default is 0.33 S/m (brain tissue).
    min_distance_mm : float, optional
        Minimum distance threshold for singularity clamping in mm.
        Default is 0.05 mm.

    Returns
    -------
    lead_field : np.ndarray
        Lead field matrix with shape (n_sensors, n_sources) in V/A.
    singularity_mask : np.ndarray
        Boolean array with shape (n_sensors, n_sources).
        True where distance clamping was applied.

    Raises
    ------
    ValueError
        If input arrays have incorrect shapes.

    Notes
    -----
    Physics: V = I / (4 * pi * sigma * r)

    The formula uses r in meters for dimensional consistency with
    conductivity in S/m. Coordinates are input in mm and converted internally.

    Examples
    --------
    >>> sensors = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    >>> sources = np.array([[0.5, 0.0, 0.0]])
    >>> L, mask = compute_lead_field(sensors, sources)
    >>> L.shape
    (2, 1)
    >>> np.any(mask)  # No singularities at these distances
    False
    """
    # Validate inputs
    sensors = np.asarray(sensors, dtype=np.float64)
    sources = np.asarray(sources, dtype=np.float64)

    if sensors.ndim != 2 or sensors.shape[1] != 3:
        raise ValueError(f"sensors must have shape (N, 3), got {sensors.shape}")
    if sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError(f"sources must have shape (M, 3), got {sources.shape}")

    n_sensors = sensors.shape[0]
    n_sources = sources.shape[0]

    # Compute pairwise distances using broadcasting
    # sensors: (N, 3) -> (N, 1, 3)
    # sources: (M, 3) -> (1, M, 3)
    # diff: (N, M, 3)
    diff = sensors[:, np.newaxis, :] - sources[np.newaxis, :, :]
    distances_mm = np.linalg.norm(diff, axis=2)  # Shape: (N, M)

    # Identify singularities (sensors too close to sources)
    singularity_mask = distances_mm < min_distance_mm

    # Clamp distances to prevent division by zero
    distances_clamped_mm = np.maximum(distances_mm, min_distance_mm)

    # Convert to meters for physical calculation
    distances_m = distances_clamped_mm * 1e-3

    # Compute lead field: L = 1 / (4 * pi * sigma * r)
    # Using the point source potential formula for homogeneous conductor
    lead_field = 1.0 / (4.0 * np.pi * conductivity * distances_m)

    return lead_field, singularity_mask


def compute_forward_solution(
    lead_field: np.ndarray,
    source_amplitudes: np.ndarray,
) -> np.ndarray:
    """
    Compute sensor potentials given source amplitudes.

    Parameters
    ----------
    lead_field : np.ndarray
        Lead field matrix with shape (n_sensors, n_sources) in V/A.
    source_amplitudes : np.ndarray
        Source current amplitudes with shape (n_sources,) or (n_sources, n_times)
        in Amperes.

    Returns
    -------
    np.ndarray
        Sensor potentials with shape (n_sensors,) or (n_sensors, n_times) in Volts.

    Examples
    --------
    >>> L = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 sensors, 2 sources
    >>> I = np.array([1e-9, 2e-9])  # nanoampere currents
    >>> V = compute_forward_solution(L, I)
    >>> V.shape
    (2,)
    """
    return lead_field @ source_amplitudes


def get_sensors_in_exclusion_zone(
    singularity_mask: np.ndarray,
    source_index: int | None = None,
) -> np.ndarray:
    """
    Get indices of sensors that are within exclusion zones of sources.

    Parameters
    ----------
    singularity_mask : np.ndarray
        Boolean array with shape (n_sensors, n_sources) from compute_lead_field.
    source_index : int, optional
        If provided, return sensors in exclusion zone of this specific source.
        If None, return sensors in exclusion zone of ANY source.

    Returns
    -------
    np.ndarray
        Array of sensor indices in exclusion zone(s).

    Examples
    --------
    >>> mask = np.array([[True, False], [False, False], [True, True]])
    >>> get_sensors_in_exclusion_zone(mask)  # Any source
    array([0, 2])
    >>> get_sensors_in_exclusion_zone(mask, source_index=1)  # Source 1 only
    array([2])
    """
    if source_index is not None:
        return np.where(singularity_mask[:, source_index])[0]
    else:
        # Sensors in exclusion zone of any source
        return np.where(np.any(singularity_mask, axis=1))[0]


def compute_distance_matrix(
    sensors: np.ndarray,
    sources: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise distance matrix between sensors and sources.

    Parameters
    ----------
    sensors : np.ndarray
        Sensor coordinates with shape (n_sensors, 3) in mm.
    sources : np.ndarray
        Source coordinates with shape (n_sources, 3) in mm.

    Returns
    -------
    np.ndarray
        Distance matrix with shape (n_sensors, n_sources) in mm.
    """
    sensors = np.asarray(sensors, dtype=np.float64)
    sources = np.asarray(sources, dtype=np.float64)

    diff = sensors[:, np.newaxis, :] - sources[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)


def validate_lead_field(
    lead_field: np.ndarray,
    expected_shape: tuple[int, int] | None = None,
) -> dict[str, any]:
    """
    Validate a lead field matrix and return diagnostic information.

    Parameters
    ----------
    lead_field : np.ndarray
        Lead field matrix to validate.
    expected_shape : tuple of int, optional
        Expected shape (n_sensors, n_sources).

    Returns
    -------
    dict
        Validation results including:
        - is_valid: bool
        - shape: tuple
        - has_infinities: bool
        - has_nans: bool
        - min_value: float
        - max_value: float
        - errors: list of str

    Examples
    --------
    >>> L = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> info = validate_lead_field(L, expected_shape=(2, 2))
    >>> info['is_valid']
    True
    """
    errors = []

    # Check shape
    if expected_shape is not None and lead_field.shape != expected_shape:
        errors.append(f"Shape mismatch: expected {expected_shape}, got {lead_field.shape}")

    # Check for numerical issues
    has_infinities = np.any(np.isinf(lead_field))
    has_nans = np.any(np.isnan(lead_field))

    if has_infinities:
        errors.append("Lead field contains infinite values")
    if has_nans:
        errors.append("Lead field contains NaN values")

    return {
        "is_valid": len(errors) == 0,
        "shape": lead_field.shape,
        "has_infinities": has_infinities,
        "has_nans": has_nans,
        "min_value": float(np.min(lead_field)) if lead_field.size > 0 else None,
        "max_value": float(np.max(lead_field)) if lead_field.size > 0 else None,
        "errors": errors,
    }

