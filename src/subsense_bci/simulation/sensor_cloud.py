"""
SensorCloud Module - Encapsulated Sensor State for Dynamic BCI Simulation

Provides a dataclass that encapsulates sensor positions, uncertainties,
and hemodynamic drift parameters for time-varying lead field computation.

This module supports both static (uniform cube) and vascular (vessel-following)
sensor distributions, with optional cardiac-driven position oscillation.

Physics: r(t) = r0 + delta_r * sin(2*pi*f_cardiac*t + phi)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from subsense_bci.physics.constants import (
    CARDIAC_FREQUENCY_HZ,
    CLOUD_VOLUME_SIDE_MM,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SENSOR_COUNT,
    HEMODYNAMIC_DRIFT_AMPLITUDE_MM,
)

if TYPE_CHECKING:
    from subsense_bci.simulation.cloud_generator import VesselSegment


@dataclass
class SensorCloud:
    """
    Encapsulates sensor positions and their dynamic properties.

    This class provides a unified interface for sensor state management,
    supporting both static and time-varying sensor positions due to
    hemodynamic (cardiac) drift.

    Attributes
    ----------
    positions : np.ndarray
        Baseline sensor positions, shape (n_sensors, 3) in mm.
    covariance : np.ndarray, optional
        Position uncertainty matrix. Shape (n_sensors, 3, 3) for full covariance
        or (n_sensors, 3) for diagonal covariance. None for no uncertainty.
    drift_amplitude : np.ndarray
        Hemodynamic drift amplitude. Shape (n_sensors,) for scalar amplitude
        (radial direction) or (n_sensors, 3) for vector amplitude.
    drift_frequency_hz : float
        Cardiac frequency for drift oscillation (default 1.2 Hz = 72 BPM).
    phase_offsets : np.ndarray
        Per-sensor phase offset for drift, shape (n_sensors,).
        Sensors in the same vessel may share similar phases.
    vessel_ids : np.ndarray, optional
        Vessel assignment for each sensor, shape (n_sensors,).
        -1 indicates no vessel assignment (uniform cloud).

    Examples
    --------
    >>> cloud = SensorCloud.from_uniform_cloud(n_sensors=1000, seed=42)
    >>> cloud.n_sensors
    1000
    >>> positions_t0 = cloud.get_positions_at_time(0.0)
    >>> positions_t1 = cloud.get_positions_at_time(0.5)  # Half cardiac cycle
    >>> np.allclose(positions_t0, positions_t1)  # Different due to drift
    False
    """

    positions: np.ndarray
    covariance: np.ndarray | None = None
    drift_amplitude: np.ndarray = field(default_factory=lambda: np.array([]))
    drift_frequency_hz: float = CARDIAC_FREQUENCY_HZ
    phase_offsets: np.ndarray = field(default_factory=lambda: np.array([]))
    vessel_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Initialize derived attributes and validate inputs."""
        self.positions = np.asarray(self.positions, dtype=np.float64)

        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(
                f"positions must have shape (n_sensors, 3), got {self.positions.shape}"
            )

        self._n_sensors = self.positions.shape[0]

        # Initialize drift amplitude if empty
        if self.drift_amplitude.size == 0:
            self.drift_amplitude = np.zeros(self._n_sensors)
        else:
            self.drift_amplitude = np.asarray(self.drift_amplitude, dtype=np.float64)

        # Initialize phase offsets if empty
        if self.phase_offsets.size == 0:
            self.phase_offsets = np.zeros(self._n_sensors)
        else:
            self.phase_offsets = np.asarray(self.phase_offsets, dtype=np.float64)

        # Validate array shapes
        if self.drift_amplitude.shape[0] != self._n_sensors:
            raise ValueError(
                f"drift_amplitude must have {self._n_sensors} entries, "
                f"got {self.drift_amplitude.shape[0]}"
            )

        if self.phase_offsets.shape[0] != self._n_sensors:
            raise ValueError(
                f"phase_offsets must have {self._n_sensors} entries, "
                f"got {self.phase_offsets.shape[0]}"
            )

    @property
    def n_sensors(self) -> int:
        """Number of sensors in the cloud."""
        return self._n_sensors

    @classmethod
    def from_uniform_cloud(
        cls,
        n_sensors: int = DEFAULT_SENSOR_COUNT,
        volume_side_mm: float = CLOUD_VOLUME_SIDE_MM,
        drift_amplitude_mm: float = HEMODYNAMIC_DRIFT_AMPLITUDE_MM,
        drift_frequency_hz: float = CARDIAC_FREQUENCY_HZ,
        seed: int = DEFAULT_RANDOM_SEED,
    ) -> "SensorCloud":
        """
        Create a SensorCloud from uniform cube distribution.

        This is the legacy generation mode, producing uniformly distributed
        sensors within a cubic volume centered at the origin.

        Parameters
        ----------
        n_sensors : int, optional
            Number of sensors. Default is 10,000.
        volume_side_mm : float, optional
            Cube side length in mm. Default is 1.0.
        drift_amplitude_mm : float, optional
            Hemodynamic drift amplitude in mm. Default is 0.05 (50 microns).
        drift_frequency_hz : float, optional
            Cardiac frequency in Hz. Default is 1.2.
        seed : int, optional
            Random seed for reproducibility. Default is 42.

        Returns
        -------
        SensorCloud
            Sensor cloud with uniform distribution and hemodynamic drift.
        """
        np.random.seed(seed)

        # Generate uniform positions
        half_side = volume_side_mm / 2.0
        positions = np.random.uniform(
            low=-half_side, high=half_side, size=(n_sensors, 3)
        ).astype(np.float64)

        # Uniform drift amplitude for all sensors
        drift_amplitude = np.full(n_sensors, drift_amplitude_mm)

        # Random phase offsets (uniform distribution)
        phase_offsets = np.random.uniform(0, 2 * np.pi, n_sensors)

        return cls(
            positions=positions,
            drift_amplitude=drift_amplitude,
            drift_frequency_hz=drift_frequency_hz,
            phase_offsets=phase_offsets,
            vessel_ids=None,  # No vessel assignment for uniform cloud
        )

    @classmethod
    def from_vascular_tree(
        cls,
        vessels: list["VesselSegment"],
        n_sensors: int = DEFAULT_SENSOR_COUNT,
        radial_std_mm: float = 0.05,
        drift_amplitude_mm: float = HEMODYNAMIC_DRIFT_AMPLITUDE_MM,
        drift_frequency_hz: float = CARDIAC_FREQUENCY_HZ,
        seed: int = DEFAULT_RANDOM_SEED,
    ) -> "SensorCloud":
        """
        Create a SensorCloud from vascular vessel distribution.

        Sensors are distributed along vessel centerlines with Gaussian
        radial spread, simulating intravascular delivery.

        Parameters
        ----------
        vessels : list[VesselSegment]
            List of vessel segments defining the vascular tree.
        n_sensors : int, optional
            Number of sensors. Default is 10,000.
        radial_std_mm : float, optional
            Gaussian spread from vessel axis in mm. Default is 0.05.
        drift_amplitude_mm : float, optional
            Hemodynamic drift amplitude in mm. Default is 0.05.
        drift_frequency_hz : float, optional
            Cardiac frequency in Hz. Default is 1.2.
        seed : int, optional
            Random seed for reproducibility. Default is 42.

        Returns
        -------
        SensorCloud
            Sensor cloud with vascular distribution and synchronized drift.
        """
        # Import here to avoid circular dependency
        from subsense_bci.simulation.cloud_generator import sample_sensors_along_vessels

        positions, vessel_ids, phase_offsets = sample_sensors_along_vessels(
            vessels=vessels,
            n_sensors=n_sensors,
            radial_std_mm=radial_std_mm,
            seed=seed,
            return_metadata=True,
        )

        # Uniform drift amplitude for all sensors
        drift_amplitude = np.full(len(positions), drift_amplitude_mm)

        return cls(
            positions=positions,
            drift_amplitude=drift_amplitude,
            drift_frequency_hz=drift_frequency_hz,
            phase_offsets=phase_offsets,
            vessel_ids=vessel_ids,
        )

    def get_positions_at_time(self, t: float) -> np.ndarray:
        """
        Get instantaneous sensor positions at time t.

        Applies hemodynamic drift model:
            r_i(t) = r0_i + delta_r_i * sin(2*pi*f*t + phi_i)

        For scalar drift amplitude, displacement is in the radial direction
        (away from origin). For vector amplitude, displacement follows the
        amplitude vector.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        np.ndarray
            Instantaneous positions with shape (n_sensors, 3) in mm.
        """
        phase = 2 * np.pi * self.drift_frequency_hz * t + self.phase_offsets
        sin_phase = np.sin(phase)

        if self.drift_amplitude.ndim == 1:
            # Scalar amplitude: displacement in radial direction from origin
            norms = np.linalg.norm(self.positions, axis=1, keepdims=True)
            # Avoid division by zero for sensors at origin
            safe_norms = np.maximum(norms, 1e-10)
            radial_unit = self.positions / safe_norms

            displacement = (
                self.drift_amplitude[:, np.newaxis]
                * radial_unit
                * sin_phase[:, np.newaxis]
            )
        else:
            # Vector amplitude: displacement in specified direction
            displacement = self.drift_amplitude * sin_phase[:, np.newaxis]

        return self.positions + displacement

    def get_covariance_at_time(self, t: float) -> np.ndarray | None:
        """
        Get position uncertainty at time t.

        Currently returns static covariance (no time dependence).
        Future versions may model time-varying uncertainty.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        np.ndarray or None
            Covariance matrix if available, None otherwise.
        """
        return self.covariance

    def get_baseline_positions(self) -> np.ndarray:
        """
        Get baseline (static) sensor positions without drift.

        Returns
        -------
        np.ndarray
            Baseline positions with shape (n_sensors, 3) in mm.
        """
        return self.positions.copy()

    def save(self, filepath: Path | str) -> Path:
        """
        Save SensorCloud to a NumPy archive.

        Parameters
        ----------
        filepath : Path or str
            Output file path (.npz extension recommended).

        Returns
        -------
        Path
            Path to saved file.
        """
        filepath = Path(filepath)
        np.savez(
            filepath,
            positions=self.positions,
            covariance=self.covariance if self.covariance is not None else np.array([]),
            drift_amplitude=self.drift_amplitude,
            drift_frequency_hz=np.array([self.drift_frequency_hz]),
            phase_offsets=self.phase_offsets,
            vessel_ids=self.vessel_ids if self.vessel_ids is not None else np.array([]),
        )
        return filepath

    @classmethod
    def load(cls, filepath: Path | str) -> "SensorCloud":
        """
        Load SensorCloud from a NumPy archive.

        Parameters
        ----------
        filepath : Path or str
            Path to .npz file.

        Returns
        -------
        SensorCloud
            Loaded sensor cloud.
        """
        data = np.load(filepath)

        covariance = data["covariance"]
        if covariance.size == 0:
            covariance = None

        vessel_ids = data["vessel_ids"]
        if vessel_ids.size == 0:
            vessel_ids = None

        return cls(
            positions=data["positions"],
            covariance=covariance,
            drift_amplitude=data["drift_amplitude"],
            drift_frequency_hz=float(data["drift_frequency_hz"][0]),
            phase_offsets=data["phase_offsets"],
            vessel_ids=vessel_ids,
        )

    def move(self, displacement: np.ndarray) -> None:
        """
        Apply instantaneous displacement to baseline positions.

        This modifies the baseline positions in-place, which affects all
        future calls to get_positions_at_time(). Use for explicit position
        manipulation beyond the sinusoidal drift model.

        Parameters
        ----------
        displacement : np.ndarray
            Displacement vector. Shape (3,) for uniform displacement applied
            to all sensors, or (n_sensors, 3) for per-sensor displacement.

        Raises
        ------
        ValueError
            If displacement shape is incompatible with sensor cloud.

        Examples
        --------
        >>> cloud = SensorCloud.from_uniform_cloud(n_sensors=100)
        >>> cloud.move(np.array([0.1, 0.0, 0.0]))  # Shift all sensors +0.1mm in X
        >>> cloud.positions[0, 0]  # X coordinate increased
        """
        displacement = np.asarray(displacement, dtype=np.float64)

        if displacement.shape == (3,):
            # Uniform displacement for all sensors
            self.positions += displacement
        elif displacement.shape == (self._n_sensors, 3):
            # Per-sensor displacement
            self.positions += displacement
        else:
            raise ValueError(
                f"displacement must have shape (3,) or ({self._n_sensors}, 3), "
                f"got {displacement.shape}"
            )

    def set_drift_parameters(
        self,
        drift_amplitude: float | np.ndarray | None = None,
        drift_frequency_hz: float | None = None,
        phase_offsets: np.ndarray | None = None,
    ) -> None:
        """
        Update drift parameters dynamically without recreating the cloud.

        Allows runtime modification of hemodynamic drift behavior for
        adaptive simulation scenarios (e.g., varying heart rate).

        Parameters
        ----------
        drift_amplitude : float or np.ndarray, optional
            New drift amplitude. Scalar applies to all sensors uniformly.
            Array must have shape (n_sensors,) or (n_sensors, 3).
        drift_frequency_hz : float, optional
            New cardiac frequency in Hz.
        phase_offsets : np.ndarray, optional
            New phase offsets, shape (n_sensors,).

        Examples
        --------
        >>> cloud = SensorCloud.from_uniform_cloud(n_sensors=100)
        >>> cloud.set_drift_parameters(drift_frequency_hz=1.5)  # Increase HR
        >>> cloud.drift_frequency_hz
        1.5
        """
        if drift_amplitude is not None:
            if np.isscalar(drift_amplitude):
                self.drift_amplitude = np.full(self._n_sensors, float(drift_amplitude))
            else:
                drift_amplitude = np.asarray(drift_amplitude, dtype=np.float64)
                if drift_amplitude.shape[0] != self._n_sensors:
                    raise ValueError(
                        f"drift_amplitude must have {self._n_sensors} entries, "
                        f"got {drift_amplitude.shape[0]}"
                    )
                self.drift_amplitude = drift_amplitude

        if drift_frequency_hz is not None:
            self.drift_frequency_hz = float(drift_frequency_hz)

        if phase_offsets is not None:
            phase_offsets = np.asarray(phase_offsets, dtype=np.float64)
            if phase_offsets.shape[0] != self._n_sensors:
                raise ValueError(
                    f"phase_offsets must have {self._n_sensors} entries, "
                    f"got {phase_offsets.shape[0]}"
                )
            self.phase_offsets = phase_offsets

    def copy(self) -> "SensorCloud":
        """
        Create a deep copy of the sensor cloud.

        Useful for creating modified variants without affecting the original,
        e.g., testing different drift parameters or perturbations.

        Returns
        -------
        SensorCloud
            Deep copy with independent arrays.

        Examples
        --------
        >>> cloud = SensorCloud.from_uniform_cloud(n_sensors=100)
        >>> cloud_copy = cloud.copy()
        >>> cloud_copy.move(np.array([0.1, 0.0, 0.0]))
        >>> np.allclose(cloud.positions, cloud_copy.positions)
        False
        """
        return SensorCloud(
            positions=self.positions.copy(),
            covariance=self.covariance.copy() if self.covariance is not None else None,
            drift_amplitude=self.drift_amplitude.copy(),
            drift_frequency_hz=self.drift_frequency_hz,
            phase_offsets=self.phase_offsets.copy(),
            vessel_ids=self.vessel_ids.copy() if self.vessel_ids is not None else None,
        )

    def __len__(self) -> int:
        """Return number of sensors."""
        return self._n_sensors

    def __repr__(self) -> str:
        """String representation."""
        has_drift = np.any(self.drift_amplitude != 0)
        has_vessels = self.vessel_ids is not None
        return (
            f"SensorCloud(n_sensors={self._n_sensors}, "
            f"drift_freq={self.drift_frequency_hz}Hz, "
            f"has_drift={has_drift}, has_vessels={has_vessels})"
        )
