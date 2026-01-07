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
    ME_ALPHA_MAX,
    ME_Q_FACTOR,
    ME_RESONANT_FREQ_KHZ,
    SINGULARITY_THRESHOLD_MM,
    TX_DEFAULT_POSITION_MM,
    VACUUM_PERMEABILITY_H_M,
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


# =============================================================================
# Magnetoelectric (ME) Backscatter Physics - Subsense Round-Trip Model
# =============================================================================


def compute_alpha_ME(
    frequency_khz: float,
    f0_khz: float = ME_RESONANT_FREQ_KHZ,
    alpha_max: float = ME_ALPHA_MAX,
    Q_factor: float = ME_Q_FACTOR,
) -> float:
    """
    Compute the frequency-dependent ME coupling coefficient.

    Uses a Lorentzian resonance profile centered at the ME transducer's
    resonant frequency:

        alpha_ME(f) = alpha_max / sqrt(1 + Q^2 * (f/f0 - f0/f)^2)

    At resonance (f = f0): alpha_ME = alpha_max
    Off resonance: alpha_ME decays with Q-dependent bandwidth.

    Parameters
    ----------
    frequency_khz : float
        Operating frequency in kHz.
    f0_khz : float, optional
        ME transducer resonant frequency in kHz. Default is 340 kHz.
    alpha_max : float, optional
        Peak coupling coefficient (normalized). Default is 1.0.
    Q_factor : float, optional
        Quality factor of the ME resonator. Default is 50.
        Higher Q = narrower bandwidth, sharper resonance.

    Returns
    -------
    float
        ME coupling coefficient at the specified frequency.

    Notes
    -----
    The Lorentzian profile is derived from the mechanical resonance of the
    piezoelectric-magnetostrictive bilayer. Q factors of 50-100 are typical
    for ME composite films.

    Examples
    --------
    >>> alpha_at_resonance = compute_alpha_ME(340.0)  # At f0
    >>> np.isclose(alpha_at_resonance, 1.0)
    True
    >>> alpha_off_resonance = compute_alpha_ME(300.0)  # 40 kHz below f0
    >>> alpha_off_resonance < alpha_at_resonance
    True
    """
    # Avoid division by zero
    if frequency_khz <= 0 or f0_khz <= 0:
        raise ValueError("Frequencies must be positive")

    f_ratio = frequency_khz / f0_khz
    # Detuning term: (f/f0 - f0/f) = 0 at resonance
    detuning = f_ratio - 1.0 / f_ratio
    denominator = np.sqrt(1.0 + Q_factor**2 * detuning**2)

    return alpha_max / denominator


def compute_magnetic_field_decay(
    distances_m: np.ndarray,
    min_distance_m: float = SINGULARITY_THRESHOLD_MM * 1e-3,
) -> np.ndarray:
    """
    Compute magnetic dipole field decay (far-field approximation).

    For a magnetic dipole, the field magnitude decays as 1/r^3:

        H_mag = mu_0 / (4 * pi * r^3)

    This represents the magnetic field from the TX coil at the sensor location.

    Parameters
    ----------
    distances_m : np.ndarray
        Distances from TX to sensors in meters.
    min_distance_m : float, optional
        Minimum distance for singularity clamping in meters.

    Returns
    -------
    np.ndarray
        Magnetic field decay factor with same shape as distances_m.
        Units: H/m (field magnitude per unit dipole moment).
    """
    distances_clamped = np.maximum(distances_m, min_distance_m)
    return VACUUM_PERMEABILITY_H_M / (4.0 * np.pi * distances_clamped**3)


def compute_me_lead_field(
    sensors: np.ndarray,
    sources: np.ndarray,
    tx_position: np.ndarray | None = None,
    frequency_khz: float = ME_RESONANT_FREQ_KHZ,
    conductivity: float = BRAIN_CONDUCTIVITY_S_M,
    f0_khz: float = ME_RESONANT_FREQ_KHZ,
    alpha_max: float = ME_ALPHA_MAX,
    Q_factor: float = ME_Q_FACTOR,
    min_distance_mm: float = SINGULARITY_THRESHOLD_MM,
) -> tuple[np.ndarray, dict]:
    """
    Compute the round-trip ME backscatter lead field matrix.

    This implements the Subsense signal model where:
    1. TX coil generates magnetic field (1/r^3 dipole decay)
    2. ME sensor converts magnetic field to electric voltage (alpha_ME coupling)
    3. Electric signal propagates through tissue (1/r volume conductor)

    Combined transfer function:
        H_total = H_mag(r_tx) * alpha_ME(f) * H_elec(r_rx)

    where:
        - H_mag = mu_0 / (4*pi*r_tx^3)  [magnetic dipole decay]
        - alpha_ME(f) = Lorentzian resonance coupling
        - H_elec = 1 / (4*pi*sigma*r_rx)  [volume conductor]

    Parameters
    ----------
    sensors : np.ndarray
        Sensor coordinates with shape (n_sensors, 3) in mm.
    sources : np.ndarray
        Neural source coordinates with shape (n_sources, 3) in mm.
    tx_position : np.ndarray, optional
        TX coil position with shape (3,) in mm.
        Default is [0, 0, 10] (10mm above origin).
    frequency_khz : float, optional
        Operating frequency in kHz. Default is 340 kHz (resonance).
    conductivity : float, optional
        Tissue conductivity in S/m. Default is 0.33 (brain).
    f0_khz : float, optional
        ME resonant frequency in kHz. Default is 340 kHz.
    alpha_max : float, optional
        Peak ME coupling coefficient. Default is 1.0.
    Q_factor : float, optional
        ME resonator Q factor. Default is 50.
    min_distance_mm : float, optional
        Singularity threshold in mm. Default is 0.05.

    Returns
    -------
    lead_field : np.ndarray
        Round-trip lead field matrix with shape (n_sensors, n_sources).
        Units are in normalized voltage per unit source current.
    metadata : dict
        Diagnostic information including:
        - H_mag: Magnetic field decay factors (n_sensors,)
        - alpha_ME: ME coupling coefficient (scalar)
        - H_elec: Electric lead field (n_sensors, n_sources)
        - frequency_khz: Operating frequency
        - singularity_mask: Boolean mask for clamped distances

    Examples
    --------
    >>> sensors = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    >>> sources = np.array([[0.2, 0.0, 0.0]])
    >>> tx_pos = np.array([0.0, 0.0, 10.0])
    >>> L_me, meta = compute_me_lead_field(sensors, sources, tx_pos)
    >>> L_me.shape
    (2, 1)
    >>> meta['alpha_ME']  # At resonance
    1.0
    """
    # Validate inputs
    sensors = np.asarray(sensors, dtype=np.float64)
    sources = np.asarray(sources, dtype=np.float64)

    if sensors.ndim != 2 or sensors.shape[1] != 3:
        raise ValueError(f"sensors must have shape (N, 3), got {sensors.shape}")
    if sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError(f"sources must have shape (M, 3), got {sources.shape}")

    # Default TX position
    if tx_position is None:
        tx_position = TX_DEFAULT_POSITION_MM.copy()
    else:
        tx_position = np.asarray(tx_position, dtype=np.float64)

    if tx_position.shape != (3,):
        raise ValueError(f"tx_position must have shape (3,), got {tx_position.shape}")

    min_distance_m = min_distance_mm * 1e-3

    # Step 1: Compute magnetic field decay (TX -> sensors)
    # Distance from TX to each sensor
    tx_to_sensors = sensors - tx_position  # (n_sensors, 3)
    r_tx_mm = np.linalg.norm(tx_to_sensors, axis=1)  # (n_sensors,)
    r_tx_m = r_tx_mm * 1e-3

    H_mag = compute_magnetic_field_decay(r_tx_m, min_distance_m)  # (n_sensors,)

    # Step 2: Compute ME coupling at operating frequency
    alpha_ME = compute_alpha_ME(frequency_khz, f0_khz, alpha_max, Q_factor)

    # Step 3: Compute electric lead field (sensors -> sources)
    # This is the standard volume conductor model
    diff = sensors[:, np.newaxis, :] - sources[np.newaxis, :, :]  # (N, M, 3)
    r_rx_mm = np.linalg.norm(diff, axis=2)  # (N, M)

    # Track singularities
    singularity_mask = r_rx_mm < min_distance_mm

    # Clamp distances
    r_rx_clamped_mm = np.maximum(r_rx_mm, min_distance_mm)
    r_rx_m = r_rx_clamped_mm * 1e-3

    # Electric field transfer (1/r volume conductor)
    H_elec = 1.0 / (4.0 * np.pi * conductivity * r_rx_m)  # (N, M)

    # Step 4: Combine into round-trip transfer function
    # H_total[i,j] = H_mag[i] * alpha_ME * H_elec[i,j]
    lead_field = H_mag[:, np.newaxis] * alpha_ME * H_elec  # (N, M)

    metadata = {
        "H_mag": H_mag,
        "alpha_ME": alpha_ME,
        "H_elec": H_elec,
        "frequency_khz": frequency_khz,
        "singularity_mask": singularity_mask,
        "tx_position": tx_position,
    }

    return lead_field, metadata


def compute_electric_lead_field(
    sensors: np.ndarray,
    sources: np.ndarray,
    conductivity: float = BRAIN_CONDUCTIVITY_S_M,
    min_distance_mm: float = SINGULARITY_THRESHOLD_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible alias for compute_lead_field.

    This function is provided for API compatibility with code that
    explicitly wants the electric-only (one-way) lead field model.

    See compute_lead_field for full documentation.
    """
    return compute_lead_field(sensors, sources, conductivity, min_distance_mm)


# =============================================================================
# LeadFieldManager - Cached and Incremental Lead Field Computation
# =============================================================================

from dataclasses import dataclass, field


@dataclass
class LeadFieldManager:
    """
    Manages lead field computation with optional caching and incremental updates.

    Wraps the pure lead field functions to provide:
    1. Full recomputation (stateless)
    2. Cached computation (returns cached if positions unchanged)
    3. Incremental updates (recomputes only changed rows)

    This is essential for Phase 5 hemodynamic simulation where sensors
    drift due to cardiac pulsation, requiring efficient lead field updates.

    Attributes
    ----------
    sources : np.ndarray
        Neural source positions, shape (n_sources, 3) in mm.
    use_me_physics : bool
        If True, use ME round-trip model. If False, use simple 1/r.
    tx_position : np.ndarray, optional
        TX coil position for ME physics, shape (3,) in mm.
    frequency_khz : float
        Operating frequency for ME physics in kHz.
    conductivity : float
        Tissue conductivity in S/m.
    min_distance_mm : float
        Singularity threshold for distance clamping.

    Examples
    --------
    >>> sources = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    >>> manager = LeadFieldManager(sources=sources, use_me_physics=False)
    >>> sensors = np.random.randn(100, 3) * 0.5
    >>> L1 = manager.compute(sensors)
    >>> L2 = manager.compute_with_cache(sensors)  # Returns cached
    >>> np.allclose(L1, L2)
    True
    """

    sources: np.ndarray
    use_me_physics: bool = True
    tx_position: np.ndarray | None = None
    frequency_khz: float = ME_RESONANT_FREQ_KHZ
    conductivity: float = BRAIN_CONDUCTIVITY_S_M
    min_distance_mm: float = SINGULARITY_THRESHOLD_MM

    # Private cache fields (not exposed in __init__)
    _cached_L: np.ndarray | None = field(default=None, init=False, repr=False)
    _cached_positions: np.ndarray | None = field(default=None, init=False, repr=False)
    _cached_metadata: dict | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate inputs."""
        self.sources = np.asarray(self.sources, dtype=np.float64)
        if self.sources.ndim != 2 or self.sources.shape[1] != 3:
            raise ValueError(f"sources must have shape (M, 3), got {self.sources.shape}")

        if self.tx_position is not None:
            self.tx_position = np.asarray(self.tx_position, dtype=np.float64)

    @property
    def n_sources(self) -> int:
        """Number of neural sources."""
        return self.sources.shape[0]

    @property
    def is_cached(self) -> bool:
        """Whether a cached lead field exists."""
        return self._cached_L is not None

    def compute(self, sensor_positions: np.ndarray) -> np.ndarray:
        """
        Compute lead field matrix (full recomputation, no caching).

        Parameters
        ----------
        sensor_positions : np.ndarray
            Sensor positions, shape (n_sensors, 3) in mm.

        Returns
        -------
        np.ndarray
            Lead field matrix, shape (n_sensors, n_sources).
        """
        sensor_positions = np.asarray(sensor_positions, dtype=np.float64)

        if self.use_me_physics:
            L, _ = compute_me_lead_field(
                sensors=sensor_positions,
                sources=self.sources,
                tx_position=self.tx_position,
                frequency_khz=self.frequency_khz,
                conductivity=self.conductivity,
                min_distance_mm=self.min_distance_mm,
            )
        else:
            L, _ = compute_lead_field(
                sensors=sensor_positions,
                sources=self.sources,
                conductivity=self.conductivity,
                min_distance_mm=self.min_distance_mm,
            )
        return L

    def compute_with_cache(
        self,
        sensor_positions: np.ndarray,
        tolerance_mm: float = 1e-6,
    ) -> np.ndarray:
        """
        Return cached lead field if positions unchanged, else recompute.

        Useful when sensors are static or move infrequently. Avoids
        redundant computation by comparing positions to cached values.

        Parameters
        ----------
        sensor_positions : np.ndarray
            Sensor positions, shape (n_sensors, 3) in mm.
        tolerance_mm : float
            Position tolerance for cache hit. Default is 1e-6 mm.

        Returns
        -------
        np.ndarray
            Lead field matrix, shape (n_sensors, n_sources).
        """
        sensor_positions = np.asarray(sensor_positions, dtype=np.float64)

        # Check for cache hit
        if (
            self._cached_L is not None
            and self._cached_positions is not None
            and self._cached_positions.shape == sensor_positions.shape
            and np.allclose(sensor_positions, self._cached_positions, atol=tolerance_mm)
        ):
            return self._cached_L

        # Cache miss - recompute
        self._cached_L = self.compute(sensor_positions)
        self._cached_positions = sensor_positions.copy()
        return self._cached_L

    def update_rows(
        self,
        sensor_positions: np.ndarray,
        changed_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Incrementally update only changed rows of the lead field.

        For hemodynamic drift where only some sensors move significantly,
        this avoids recomputing the entire matrix. Only rows corresponding
        to changed sensors are recomputed.

        Parameters
        ----------
        sensor_positions : np.ndarray
            Full sensor positions, shape (n_sensors, 3) in mm.
        changed_indices : np.ndarray
            Indices of sensors that have moved, shape (n_changed,).

        Returns
        -------
        np.ndarray
            Updated lead field matrix, shape (n_sensors, n_sources).

        Notes
        -----
        If no cache exists, falls back to full computation.
        For large numbers of changed sensors, full recompute may be faster.
        """
        sensor_positions = np.asarray(sensor_positions, dtype=np.float64)
        changed_indices = np.asarray(changed_indices, dtype=np.intp)

        # If no cache, do full computation
        if self._cached_L is None or self._cached_positions is None:
            return self.compute_with_cache(sensor_positions)

        # If all sensors changed, do full recompute
        if len(changed_indices) >= sensor_positions.shape[0]:
            return self.compute_with_cache(sensor_positions)

        # Incremental update: recompute only changed rows
        for idx in changed_indices:
            single_sensor = sensor_positions[idx : idx + 1]  # Shape (1, 3)

            if self.use_me_physics:
                L_row, _ = compute_me_lead_field(
                    sensors=single_sensor,
                    sources=self.sources,
                    tx_position=self.tx_position,
                    frequency_khz=self.frequency_khz,
                    conductivity=self.conductivity,
                    min_distance_mm=self.min_distance_mm,
                )
            else:
                L_row, _ = compute_lead_field(
                    sensors=single_sensor,
                    sources=self.sources,
                    conductivity=self.conductivity,
                    min_distance_mm=self.min_distance_mm,
                )
            self._cached_L[idx] = L_row[0]

        # Update cached positions
        self._cached_positions = sensor_positions.copy()
        return self._cached_L

    def invalidate_cache(self) -> None:
        """Clear cached lead field and positions."""
        self._cached_L = None
        self._cached_positions = None
        self._cached_metadata = None

    def get_cache_info(self) -> dict:
        """
        Get information about the current cache state.

        Returns
        -------
        dict
            Cache information including:
            - is_cached: bool
            - n_sensors: int or None
            - n_sources: int
        """
        return {
            "is_cached": self.is_cached,
            "n_sensors": self._cached_L.shape[0] if self._cached_L is not None else None,
            "n_sources": self.n_sources,
        }

    def __repr__(self) -> str:
        return (
            f"LeadFieldManager(n_sources={self.n_sources}, "
            f"use_me_physics={self.use_me_physics}, "
            f"is_cached={self.is_cached})"
        )

