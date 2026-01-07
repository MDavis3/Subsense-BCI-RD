"""
Physical Constants for Subsense BCI Simulations

All constants include units in their names and reference literature sources.
"""

from __future__ import annotations

import numpy as np

# Electromagnetic Constants
SPEED_OF_LIGHT_M_S: float = 299_792_458.0  # m/s, exact by definition
VACUUM_PERMEABILITY_H_M: float = 1.25663706212e-6  # H/m (henries per meter)
VACUUM_PERMITTIVITY_F_M: float = 8.8541878128e-12  # F/m (farads per meter)

# Tissue Conductivity (S/m) - Reference: Gabriel et al., 1996
BRAIN_CONDUCTIVITY_S_M: float = 0.33  # Gray matter at low frequency
SKULL_CONDUCTIVITY_S_M: float = 0.0042  # Compact bone
SCALP_CONDUCTIVITY_S_M: float = 0.43  # Skin
CSF_CONDUCTIVITY_S_M: float = 1.79  # Cerebrospinal fluid

# Magnetoelectric Transducer Parameters (ActiveEcho System)
# Reference: Based on ME film specifications
ME_FILM_LENGTH_MM: float = 5.0
ME_FILM_WIDTH_MM: float = 2.0
ME_FILM_THICKNESS_MM: float = 0.18
ME_RESONANT_FREQ_KHZ: float = 340.0
AUX_COIL_INDUCTANCE_NH: float = 250.0

# ME Coupling Physics (Subsense Round-Trip Model)
# Reference: Lorentzian resonance model for ME transducers
ME_ALPHA_MAX: float = 1.0  # Normalized peak ME coupling coefficient (V/T or dimensionless)
ME_Q_FACTOR: float = 50.0  # Quality factor (typical 50-100 for piezo films)

# TX Array Parameters
TX_COIL_DIAMETER_MM: float = 42.0
TX_COIL_SPACING_MM: float = 24.0  # Center-to-center for mutual inductance cancellation

# TX Default Position for Round-Trip Model
TX_DEFAULT_POSITION_MM: np.ndarray = np.array([0.0, 0.0, 10.0])  # 10mm above origin

# Typical BCI Frequency Bands (Hz)
DELTA_BAND_HZ: tuple[float, float] = (0.5, 4.0)
THETA_BAND_HZ: tuple[float, float] = (4.0, 8.0)
ALPHA_BAND_HZ: tuple[float, float] = (8.0, 13.0)
BETA_BAND_HZ: tuple[float, float] = (13.0, 30.0)
GAMMA_BAND_HZ: tuple[float, float] = (30.0, 100.0)
HIGH_GAMMA_BAND_HZ: tuple[float, float] = (100.0, 200.0)

# =============================================================================
# Phase 1 Simulation Parameters - Stochastic Nanoparticle Cloud
# =============================================================================

# Domain geometry
CLOUD_VOLUME_SIDE_MM: float = 1.0  # Cubic domain extent (centered at origin)

# Singularity handling for lead field computation
# At r < this threshold, distance is clamped to prevent infinite voltage
SINGULARITY_THRESHOLD_MM: float = 0.05

# Default simulation parameters (per .cursorrules reproducibility mandate)
DEFAULT_SENSOR_COUNT: int = 10_000
DEFAULT_RANDOM_SEED: int = 42

# Particle parameters (for collision checking if needed)
PARTICLE_RADIUS_NM: float = 100.0  # Assumed nanoparticle radius

# =============================================================================
# Phase 2 Simulation Parameters - Temporal Dynamics
# =============================================================================

# Sampling and duration
SAMPLING_RATE_HZ: float = 1000.0  # 1 kHz sampling rate
DURATION_SEC: float = 2.0  # 2 seconds of simulated data

# Noise parameters
SNR_LEVEL: float = 5.0  # Signal-to-Noise Ratio (linear, not dB)

# =============================================================================
# Hemodynamic Drift Parameters - Cardiac Pulsatility Model
# =============================================================================

# Cardiac oscillation causes sensor position drift in intravascular delivery
HEMODYNAMIC_DRIFT_AMPLITUDE_MM: float = 0.05  # 50 microns typical vessel pulsatility
CARDIAC_FREQUENCY_HZ: float = 1.2  # ~72 BPM resting heart rate

# =============================================================================
# Vascular Geometry Parameters
# =============================================================================

# Vessel tree generation defaults
VESSEL_N_BRANCHES: int = 5
VESSEL_BRANCH_ANGLE_DEG: float = 30.0
VESSEL_SEGMENT_LENGTH_MM: float = 0.3

# Sensor distribution around vessels
VESSEL_RADIAL_STD_MM: float = 0.05  # Gaussian spread from vessel axis
VESSEL_AXIAL_DENSITY: float = 100.0  # Sensors per mm along vessel

