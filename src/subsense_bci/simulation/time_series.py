"""
Time Series Generator - Phase 2 Temporal Dynamics

Generates time-domain simulations of the nanoparticle cloud response
to neural sources with distinct frequency characteristics.

Source Model:
- Source A: 10 Hz sine wave (Alpha band) - "Relaxation"
- Source B: 20 Hz sine wave (Beta band) - "Motor Planning"
- Source C: Pink noise (1/f) - "Background Neural Activity"

Forward Model:
- X(t) = L @ S(t) + N(t)
- where L is the lead field, S(t) are source waveforms, N(t) is sensor noise
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from subsense_bci.physics.constants import (
    CARDIAC_DICROTIC_AMPLITUDE,
    CARDIAC_DICROTIC_POSITION,
    CARDIAC_FREQUENCY_HZ,
    CARDIAC_ORIGIN_MM,
    CARDIAC_SYSTOLE_FRACTION,
    DEFAULT_RANDOM_SEED,
    DURATION_SEC,
    HEMODYNAMIC_DRIFT_AMPLITUDE_MM,
    ME_RESONANT_FREQ_KHZ,
    PULSE_WAVE_VELOCITY_M_S,
    SAMPLING_RATE_HZ,
    SNR_LEVEL,
)
from subsense_bci.physics.transfer_function import compute_lead_field, compute_me_lead_field


@dataclass
class CardiacPulseGenerator:
    """
    Generates realistic cardiac pressure waveforms with physiological accuracy.

    Models the arterial pressure waveform with:
    - Asymmetric systole/diastole (30%/70% duty cycle)
    - Steep systolic rise (rapid ventricular ejection)
    - Gradual diastolic decay (aortic runoff)
    - Dicrotic notch (aortic valve closure)
    - Sensor-specific phase delays from pulse wave propagation

    The waveform is constructed piecewise:
    - Phase 0-30%: Sigmoid systolic rise
    - Phase 30-60%: Gradual decay toward dicrotic notch
    - Phase 60-70%: Dicrotic notch (secondary bump)
    - Phase 70-100%: Exponential diastolic decay

    Parameters
    ----------
    cardiac_freq_hz : float
        Fundamental cardiac frequency in Hz. Default 1.2 (~72 BPM).
    systole_fraction : float
        Fraction of cycle occupied by systole. Default 0.30 (30%).
    dicrotic_position : float
        Position of dicrotic notch as fraction of cycle. Default 0.60.
    dicrotic_amplitude : float
        Amplitude of dicrotic notch as fraction of peak. Default 0.15.
    propagation_velocity_m_s : float
        Pulse wave velocity for phase delay calculation. Default 7.5 m/s.
    cardiac_origin : np.ndarray
        Origin point for phase propagation in mm. Default [0, 0, 0.5].

    Examples
    --------
    >>> generator = CardiacPulseGenerator()
    >>> t = np.linspace(0, 2, 2000)
    >>> waveform = generator.generate_waveform_vectorized(t)
    >>> waveform.shape
    (2000,)
    """

    cardiac_freq_hz: float = CARDIAC_FREQUENCY_HZ
    systole_fraction: float = CARDIAC_SYSTOLE_FRACTION
    dicrotic_position: float = CARDIAC_DICROTIC_POSITION
    dicrotic_amplitude: float = CARDIAC_DICROTIC_AMPLITUDE
    propagation_velocity_m_s: float = PULSE_WAVE_VELOCITY_M_S
    cardiac_origin: np.ndarray = field(
        default_factory=lambda: CARDIAC_ORIGIN_MM.copy()
    )

    def compute_phase_offsets(self, sensor_positions: np.ndarray) -> np.ndarray:
        """
        Compute per-sensor phase offsets based on distance from cardiac origin.

        Phase offset = 2π × freq × distance / velocity

        This models the time delay of the pressure wave propagating from the
        heart through the arterial tree.

        Parameters
        ----------
        sensor_positions : np.ndarray
            Sensor coordinates with shape (n_sensors, 3) in mm.

        Returns
        -------
        np.ndarray
            Phase offsets in radians with shape (n_sensors,).
        """
        sensor_positions = np.asarray(sensor_positions)
        origin = np.asarray(self.cardiac_origin)

        # Compute Euclidean distance from cardiac origin (mm)
        distances_mm = np.linalg.norm(sensor_positions - origin, axis=1)

        # Convert mm to meters for velocity calculation
        distances_m = distances_mm / 1000.0

        # Time delay = distance / velocity
        time_delays_s = distances_m / self.propagation_velocity_m_s

        # Phase offset = 2π × freq × time_delay
        phase_offsets = 2 * np.pi * self.cardiac_freq_hz * time_delays_s

        return phase_offsets

    def generate_waveform(self, t: float, phase_offset: float = 0.0) -> float:
        """
        Generate cardiac waveform value at a single time point.

        Parameters
        ----------
        t : float
            Time in seconds.
        phase_offset : float
            Phase offset in radians (for pulse wave propagation).

        Returns
        -------
        float
            Waveform amplitude at time t (normalized to [0, 1]).
        """
        # Compute phase within cardiac cycle [0, 1)
        phase = (self.cardiac_freq_hz * t + phase_offset / (2 * np.pi)) % 1.0

        # Piecewise waveform construction
        if phase < self.systole_fraction:
            # Systolic phase: steep sigmoid rise
            # Normalize to [0, 1] within systole
            x = phase / self.systole_fraction
            # Sigmoid: 1 / (1 + exp(-k*(x-0.5))) scaled to [0, 1]
            k = 12.0  # Steepness parameter
            value = 1.0 / (1.0 + np.exp(-k * (x - 0.5)))
            # Rescale sigmoid from ~[0.002, 0.998] to [0, 1]
            sig_min = 1.0 / (1.0 + np.exp(k * 0.5))
            sig_max = 1.0 / (1.0 + np.exp(-k * 0.5))
            value = (value - sig_min) / (sig_max - sig_min)

        elif phase < self.dicrotic_position:
            # Post-systole: gradual decay toward dicrotic notch
            # Decay from 1.0 to ~0.4 (pre-notch level)
            x = (phase - self.systole_fraction) / (
                self.dicrotic_position - self.systole_fraction
            )
            pre_notch_level = 0.4
            value = 1.0 - (1.0 - pre_notch_level) * x

        elif phase < self.dicrotic_position + 0.10:
            # Dicrotic notch: secondary bump (10% of cycle)
            # Creates the characteristic "notch" from aortic valve closure
            x = (phase - self.dicrotic_position) / 0.10
            pre_notch_level = 0.4
            # Gaussian bump centered at x=0.5
            notch_bump = self.dicrotic_amplitude * np.exp(-((x - 0.5) ** 2) / 0.08)
            value = pre_notch_level + notch_bump

        else:
            # Diastolic phase: exponential decay to baseline
            x = (phase - self.dicrotic_position - 0.10) / (
                1.0 - self.dicrotic_position - 0.10
            )
            post_notch_level = 0.4 + self.dicrotic_amplitude * np.exp(-0.5**2 / 0.08)
            # Decay from post-notch level to ~0.1 (diastolic baseline)
            baseline = 0.1
            value = baseline + (post_notch_level - baseline) * np.exp(-3.0 * x)

        return value

    def generate_waveform_vectorized(
        self,
        time_vector: np.ndarray,
        phase_offsets: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """
        Generate cardiac waveform for a time vector with optional phase offsets.

        Parameters
        ----------
        time_vector : np.ndarray
            Time vector in seconds with shape (n_samples,).
        phase_offsets : np.ndarray or float
            Phase offsets in radians. Can be:
            - Scalar: Same offset for all samples
            - 1D array shape (n_sensors,): Per-sensor offset (returns 2D)

        Returns
        -------
        np.ndarray
            Waveform values. Shape depends on phase_offsets:
            - If scalar: shape (n_samples,)
            - If (n_sensors,): shape (n_sensors, n_samples)
        """
        time_vector = np.asarray(time_vector)
        phase_offsets = np.asarray(phase_offsets)

        # Handle per-sensor phase offsets
        if phase_offsets.ndim == 0:
            # Scalar phase offset
            return np.array(
                [self.generate_waveform(t, float(phase_offsets)) for t in time_vector]
            )
        elif phase_offsets.ndim == 1:
            # Per-sensor phase offsets: (n_sensors,)
            n_sensors = len(phase_offsets)
            n_samples = len(time_vector)
            result = np.zeros((n_sensors, n_samples))
            for i, offset in enumerate(phase_offsets):
                for j, t in enumerate(time_vector):
                    result[i, j] = self.generate_waveform(t, offset)
            return result
        else:
            raise ValueError(
                f"phase_offsets must be scalar or 1D, got shape {phase_offsets.shape}"
            )

    def generate_displacement_field(
        self,
        time_vector: np.ndarray,
        sensor_positions: np.ndarray,
        drift_amplitude_mm: float = HEMODYNAMIC_DRIFT_AMPLITUDE_MM,
        drift_direction: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Generate 3D displacement field for all sensors over time.

        Parameters
        ----------
        time_vector : np.ndarray
            Time vector in seconds with shape (n_samples,).
        sensor_positions : np.ndarray
            Sensor positions with shape (n_sensors, 3) in mm.
        drift_amplitude_mm : float
            Maximum displacement amplitude in mm.
        drift_direction : np.ndarray, optional
            Unit vector for drift direction. If None, uses radial direction
            from cardiac origin.

        Returns
        -------
        np.ndarray
            Displacement field with shape (n_samples, n_sensors, 3) in mm.
            displacement[t, i, :] is the 3D displacement of sensor i at time t.
        """
        sensor_positions = np.asarray(sensor_positions)
        n_sensors = sensor_positions.shape[0]
        n_samples = len(time_vector)

        # Compute per-sensor phase offsets
        phase_offsets = self.compute_phase_offsets(sensor_positions)

        # Generate waveform for each sensor
        waveforms = self.generate_waveform_vectorized(time_vector, phase_offsets)
        # waveforms shape: (n_sensors, n_samples)

        # Compute drift directions (radial from cardiac origin if not specified)
        if drift_direction is None:
            origin = np.asarray(self.cardiac_origin)
            directions = sensor_positions - origin
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)  # Avoid division by zero
            directions = directions / norms  # (n_sensors, 3)
        else:
            drift_direction = np.asarray(drift_direction)
            drift_direction = drift_direction / np.linalg.norm(drift_direction)
            directions = np.tile(drift_direction, (n_sensors, 1))  # (n_sensors, 3)

        # Compute displacement: amplitude * waveform * direction
        # waveforms: (n_sensors, n_samples) -> transpose to (n_samples, n_sensors)
        # directions: (n_sensors, 3)
        displacement = np.zeros((n_samples, n_sensors, 3))
        for t_idx in range(n_samples):
            # Scale each sensor's direction by its waveform value
            displacement[t_idx] = (
                drift_amplitude_mm * waveforms[:, t_idx, np.newaxis] * directions
            )

        return displacement


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent.parent


def generate_time_vector(
    duration_sec: float = DURATION_SEC,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
) -> np.ndarray:
    """
    Generate time vector for simulation.

    Parameters
    ----------
    duration_sec : float
        Duration in seconds.
    sampling_rate_hz : float
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        Time vector with shape (n_samples,) in seconds.
    """
    n_samples = int(duration_sec * sampling_rate_hz)
    return np.linspace(0, duration_sec, n_samples, endpoint=False)


def generate_pink_noise(n_samples: int, seed: int | None = None) -> np.ndarray:
    """
    Generate pink noise (1/f noise) using FFT-based spectral shaping.

    Pink noise has equal energy per octave, with power spectral density
    proportional to 1/f. This implementation creates true 1/f spectrum
    by shaping white noise in the frequency domain.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Pink noise signal (NOT normalized - caller should standardize).
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate white noise in time domain
    white_noise = np.random.randn(n_samples)

    # Transform to frequency domain
    fft_white = np.fft.rfft(white_noise)

    # Create frequency bins (Hz normalized by sample rate)
    freqs = np.fft.rfftfreq(n_samples)

    # Create 1/f filter: PSD ∝ 1/f means amplitude ∝ 1/sqrt(f)
    # Handle DC component (f=0) by setting to 0 to remove DC offset
    pink_filter = np.zeros_like(freqs)
    nonzero_mask = freqs > 0
    pink_filter[nonzero_mask] = 1.0 / np.sqrt(freqs[nonzero_mask])

    # Apply filter in frequency domain
    fft_pink = fft_white * pink_filter

    # Transform back to time domain
    pink = np.fft.irfft(fft_pink, n=n_samples)

    return pink


def standardize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Standardize a signal to have zero mean and unit variance.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.

    Returns
    -------
    np.ndarray
        Standardized signal with mean=0 and std=1.
    """
    return (signal - np.mean(signal)) / np.std(signal)


def generate_source_waveforms(
    time_vector: np.ndarray,
    seed: int = DEFAULT_RANDOM_SEED,
    source_frequencies: dict | None = None,
) -> np.ndarray:
    """
    Generate source waveforms for all 3 neural sources.

    All sources are standardized to unit variance (sigma=1) before mixing
    to ensure equal contribution to the lead field mixing and enable
    proper SNR calculation.

    Parameters
    ----------
    time_vector : np.ndarray
        Time vector in seconds.
    seed : int
        Random seed for pink noise generation.
    source_frequencies : dict, optional
        Dictionary with source frequencies in Hz. Expected keys:
        - "alpha": Frequency for source A (default: 10.0 Hz)
        - "beta": Frequency for source B (default: 20.0 Hz)
        If None, loads from config["physics"]["source_frequencies_hz"].

    Returns
    -------
    np.ndarray
        Source waveforms with shape (n_sources=3, n_samples).
        Row 0: Source A - Alpha sine wave, standardized
        Row 1: Source B - Beta sine wave, standardized
        Row 2: Source C - Pink noise (1/f), standardized

        All sources have mean=0 and std=1.
    """
    # Load frequencies from config if not provided
    if source_frequencies is None:
        from subsense_bci.config import load_config
        config = load_config()
        source_frequencies = config.get("physics", {}).get(
            "source_frequencies_hz",
            {"alpha": 10.0, "beta": 20.0}
        )

    alpha_freq = source_frequencies.get("alpha", 10.0)
    beta_freq = source_frequencies.get("beta", 20.0)

    n_samples = len(time_vector)

    # Source A: Alpha wave
    source_a = np.sin(2 * np.pi * alpha_freq * time_vector)

    # Source B: Beta wave
    source_b = np.sin(2 * np.pi * beta_freq * time_vector)

    # Source C: Pink noise (1/f background activity)
    source_c = generate_pink_noise(n_samples, seed=seed + 100)

    # CRITICAL: Standardize ALL sources to unit variance (mean=0, std=1)
    # This ensures:
    # 1. Equal "power" contribution from each source before lead field mixing
    # 2. Pink noise is visible alongside sinusoidal sources
    # 3. SNR calculation is mathematically well-defined
    source_a = standardize_signal(source_a)
    source_b = standardize_signal(source_b)
    source_c = standardize_signal(source_c)

    # Stack into (n_sources, n_samples) matrix
    source_waveforms = np.vstack([source_a, source_b, source_c])

    return source_waveforms


def add_sensor_noise(
    clean_data: np.ndarray,
    snr: float = SNR_LEVEL,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add Gaussian white noise to sensor data based on SNR.

    The noise level is scaled relative to the RMS power of the clean signal.

    SNR Definition (power ratio, linear scale):
        SNR = signal_power / noise_power = signal_rms² / noise_rms²

    For SNR=5.0:
        noise_power = signal_power / 5
        noise_rms = signal_rms / sqrt(5) ≈ signal_rms / 2.236

    Parameters
    ----------
    clean_data : np.ndarray
        Clean sensor data with shape (n_sensors, n_samples).
    snr : float
        Signal-to-noise ratio (linear power ratio, not dB).
        SNR=5 means signal power is 5× noise power.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    noisy_data : np.ndarray
        Noisy sensor data with shape (n_sensors, n_samples).
    noise : np.ndarray
        The noise component (for SNR verification).
    """
    np.random.seed(seed + 200)

    # Compute signal RMS across all sensors and time
    signal_rms = np.sqrt(np.mean(clean_data ** 2))

    # Compute noise standard deviation from SNR
    # SNR = signal_power / noise_power = signal_rms² / noise_std²
    # Solving for noise_std: noise_std = signal_rms / sqrt(SNR)
    # This ensures: noise_power = noise_std² = signal_rms² / SNR = signal_power / SNR
    noise_std = signal_rms / np.sqrt(snr)

    # Generate Gaussian white noise with target standard deviation
    noise = np.random.randn(*clean_data.shape) * noise_std

    return clean_data + noise, noise


def simulate_recording(
    sensors: np.ndarray,
    sources: np.ndarray,
    duration_sec: float = DURATION_SEC,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
    snr: float = SNR_LEVEL,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the full time-domain simulation.

    Forward model: X(t) = L @ S(t) + N(t)

    Parameters
    ----------
    sensors : np.ndarray
        Sensor coordinates with shape (n_sensors, 3) in mm.
    sources : np.ndarray
        Source coordinates with shape (n_sources, 3) in mm.
    duration_sec : float
        Simulation duration in seconds.
    sampling_rate_hz : float
        Sampling rate in Hz.
    snr : float
        Signal-to-noise ratio (linear).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    time_vector : np.ndarray
        Time vector with shape (n_samples,) in seconds.
    source_waveforms : np.ndarray
        Source signals with shape (n_sources, n_samples).
    clean_data : np.ndarray
        Clean sensor data with shape (n_sensors, n_samples).
    noisy_data : np.ndarray
        Noisy sensor data with shape (n_sensors, n_samples).
    """
    # Step 1: Generate time vector
    time_vector = generate_time_vector(duration_sec, sampling_rate_hz)
    n_samples = len(time_vector)

    print(f"  Time vector: {n_samples} samples @ {sampling_rate_hz} Hz")

    # Step 2: Generate source waveforms (all standardized to unit variance)
    source_waveforms = generate_source_waveforms(time_vector, seed=seed)
    print(f"  Source waveforms: {source_waveforms.shape}")

    # Verify source standardization
    for i, name in enumerate(["A (10Hz)", "B (20Hz)", "C (Pink)"]):
        src_std = np.std(source_waveforms[i])
        src_mean = np.mean(source_waveforms[i])
        print(f"    Source {name}: mean={src_mean:.2e}, std={src_std:.4f}")

    # Step 3: Compute lead field matrix
    lead_field, _ = compute_lead_field(sensors, sources)
    print(f"  Lead field: {lead_field.shape}")

    # Step 4: Forward model - mix sources to sensors
    # clean_data = L @ S(t)
    # L: (n_sensors, n_sources), S: (n_sources, n_samples)
    # Result: (n_sensors, n_samples)
    clean_data = lead_field @ source_waveforms
    print(f"  Clean data: {clean_data.shape}")

    # Step 5: Add sensor noise
    noisy_data, noise = add_sensor_noise(clean_data, snr=snr, seed=seed)

    # Verify actual SNR
    signal_power = np.mean(clean_data ** 2)
    noise_power = np.mean(noise ** 2)
    actual_snr = signal_power / noise_power
    print(f"  Noisy data: {noisy_data.shape}")
    print(f"  SNR verification: target={snr:.2f}, actual={actual_snr:.4f}")

    return time_vector, source_waveforms, clean_data, noisy_data


def generate_ppg_reference(
    time_vector: np.ndarray,
    cardiac_freq_hz: float = CARDIAC_FREQUENCY_HZ,
    seed: int = DEFAULT_RANDOM_SEED,
    use_realistic_waveform: bool = False,
    cardiac_generator: CardiacPulseGenerator | None = None,
) -> np.ndarray:
    """
    Generate a simulated PPG (photoplethysmography) reference signal.

    Two modes are available:
    1. Legacy mode (use_realistic_waveform=False): Sine + harmonic approximation
    2. Realistic mode (use_realistic_waveform=True): CardiacPulseGenerator waveform

    The legacy PPG waveform approximates the cardiac pulse wave with:
    - Fundamental at cardiac frequency
    - Second harmonic for dicrotic notch characteristic
    - Small phase noise for physiological variability

    The realistic waveform provides:
    - Asymmetric systole/diastole
    - Proper dicrotic notch timing
    - Physiologically accurate pressure profile

    Parameters
    ----------
    time_vector : np.ndarray
        Time vector in seconds.
    cardiac_freq_hz : float, optional
        Cardiac frequency in Hz. Default is 1.2 (~72 BPM).
    seed : int, optional
        Random seed for phase noise. Default is 42.
    use_realistic_waveform : bool, optional
        If True, use CardiacPulseGenerator for realistic waveform.
        If False, use legacy sine + harmonic model. Default is False.
    cardiac_generator : CardiacPulseGenerator, optional
        Pre-configured generator. If None and use_realistic_waveform=True,
        creates a new generator with default parameters.

    Returns
    -------
    np.ndarray
        PPG reference signal with shape (n_samples,).
        Normalized to have unit amplitude.
    """
    if use_realistic_waveform:
        # Use CardiacPulseGenerator for realistic waveform
        if cardiac_generator is None:
            cardiac_generator = CardiacPulseGenerator(cardiac_freq_hz=cardiac_freq_hz)

        # Generate waveform (no phase offset for reference signal)
        ppg = cardiac_generator.generate_waveform_vectorized(time_vector, phase_offsets=0.0)

        # Normalize to [-1, 1] range
        ppg = 2.0 * (ppg - ppg.min()) / (ppg.max() - ppg.min()) - 1.0

        return ppg

    # Legacy mode: sine + harmonic
    np.random.seed(seed + 500)

    # Small phase jitter for physiological variability
    phase_noise = np.cumsum(np.random.randn(len(time_vector)) * 0.002)

    # Fundamental frequency
    fundamental = np.sin(2 * np.pi * cardiac_freq_hz * time_vector + phase_noise)

    # Second harmonic (creates dicrotic notch characteristic)
    harmonic = 0.3 * np.sin(4 * np.pi * cardiac_freq_hz * time_vector + 0.5 + phase_noise * 2)

    ppg = fundamental + harmonic

    # Normalize
    ppg = ppg / np.max(np.abs(ppg))

    return ppg


def simulate_recording_dynamic(
    sensor_cloud: "SensorCloud",
    sources: np.ndarray,
    tx_position: np.ndarray | None = None,
    duration_sec: float = DURATION_SEC,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
    frequency_khz: float = ME_RESONANT_FREQ_KHZ,
    snr: float = SNR_LEVEL,
    seed: int = DEFAULT_RANDOM_SEED,
    use_me_physics: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run time-domain simulation with dynamic (time-varying) lead field.

    This is the Subsense-specific simulation that accounts for:
    1. ME backscatter physics (round-trip transfer function)
    2. Hemodynamic drift (cardiac-driven sensor position oscillation)
    3. Time-varying lead field L(t) computed per timestep

    Forward model: X(t) = L(t) @ S(t) + N(t)

    where L(t) depends on sensor positions at time t.

    Parameters
    ----------
    sensor_cloud : SensorCloud
        Sensor cloud with positions and drift parameters.
    sources : np.ndarray
        Source coordinates with shape (n_sources, 3) in mm.
    tx_position : np.ndarray, optional
        TX coil position (3,) in mm. Default is [0, 0, 10].
    duration_sec : float, optional
        Simulation duration in seconds. Default is 2.0.
    sampling_rate_hz : float, optional
        Sampling rate in Hz. Default is 1000.
    frequency_khz : float, optional
        ME operating frequency in kHz. Default is 340 (resonance).
    snr : float, optional
        Signal-to-noise ratio (linear). Default is 5.0.
    seed : int, optional
        Random seed. Default is 42.
    use_me_physics : bool, optional
        If True, use ME round-trip model. If False, use electric-only.
        Default is True.
    verbose : bool, optional
        Print progress messages. Default is True.

    Returns
    -------
    time_vector : np.ndarray
        Time vector with shape (n_samples,) in seconds.
    source_waveforms : np.ndarray
        Source signals with shape (n_sources, n_samples).
    clean_data : np.ndarray
        Clean sensor data with shape (n_sensors, n_samples).
    noisy_data : np.ndarray
        Noisy sensor data with shape (n_sensors, n_samples).
    ppg_reference : np.ndarray
        PPG reference signal with shape (n_samples,).

    Notes
    -----
    This function is computationally expensive as it recomputes the lead
    field at each timestep. For 2 seconds at 1 kHz with 10,000 sensors,
    expect ~2000 lead field computations.

    Examples
    --------
    >>> from subsense_bci.simulation.sensor_cloud import SensorCloud
    >>> cloud = SensorCloud.from_uniform_cloud(n_sensors=1000)
    >>> sources = np.array([[0.2, 0.0, 0.0], [-0.2, 0.2, 0.0], [0.0, -0.2, 0.1]])
    >>> t, S, X_clean, X_noisy, ppg = simulate_recording_dynamic(cloud, sources)
    """
    # Import here to avoid circular dependency at module load
    from subsense_bci.simulation.sensor_cloud import SensorCloud

    # Type hint for IDE (actual check is implicit from method calls)
    if not isinstance(sensor_cloud, SensorCloud):
        raise TypeError("sensor_cloud must be a SensorCloud instance")

    sources = np.asarray(sources, dtype=np.float64)
    if sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError(f"sources must have shape (n_sources, 3), got {sources.shape}")

    # Step 1: Generate time vector
    time_vector = generate_time_vector(duration_sec, sampling_rate_hz)
    n_samples = len(time_vector)
    n_sensors = sensor_cloud.n_sensors
    n_sources = sources.shape[0]

    if verbose:
        print(f"  Time vector: {n_samples} samples @ {sampling_rate_hz} Hz")
        print(f"  Sensors: {n_sensors}, Sources: {n_sources}")
        print(f"  Using {'ME round-trip' if use_me_physics else 'electric-only'} physics")

    # Step 2: Generate source waveforms
    source_waveforms = generate_source_waveforms(time_vector, seed=seed)

    if verbose:
        print(f"  Source waveforms: {source_waveforms.shape}")

    # Step 3: Generate PPG reference (for artifact rejection later)
    ppg_reference = generate_ppg_reference(
        time_vector,
        cardiac_freq_hz=sensor_cloud.drift_frequency_hz,
        seed=seed,
    )

    # Step 4: Compute time-varying forward model
    # Pre-allocate output
    clean_data = np.zeros((n_sensors, n_samples), dtype=np.float64)

    if verbose:
        print(f"  Computing dynamic lead field ({n_samples} timesteps)...")

    # Progress tracking
    progress_interval = max(1, n_samples // 10)

    for i, t in enumerate(time_vector):
        # Get sensor positions at this time
        positions_t = sensor_cloud.get_positions_at_time(t)

        # Compute lead field
        if use_me_physics:
            L_t, _ = compute_me_lead_field(
                positions_t,
                sources,
                tx_position=tx_position,
                frequency_khz=frequency_khz,
            )
        else:
            L_t, _ = compute_lead_field(positions_t, sources)

        # Forward model: X_t = L_t @ S_t
        clean_data[:, i] = L_t @ source_waveforms[:, i]

        # Progress reporting
        if verbose and (i + 1) % progress_interval == 0:
            pct = 100 * (i + 1) / n_samples
            print(f"    Progress: {pct:.0f}%")

    if verbose:
        print(f"  Clean data: {clean_data.shape}")

    # Step 5: Add sensor noise
    noisy_data, noise = add_sensor_noise(clean_data, snr=snr, seed=seed)

    if verbose:
        signal_power = np.mean(clean_data ** 2)
        noise_power = np.mean(noise ** 2)
        actual_snr = signal_power / noise_power
        print(f"  Noisy data: {noisy_data.shape}")
        print(f"  SNR verification: target={snr:.2f}, actual={actual_snr:.4f}")

    return time_vector, source_waveforms, clean_data, noisy_data, ppg_reference


def load_phase1_data() -> tuple[np.ndarray, np.ndarray]:
    """Load sensor and source data from Phase 1."""
    data_dir = get_project_root() / "data" / "raw"
    sensors = np.load(data_dir / "sensors_N10000_seed42.npy")
    sources = np.load(data_dir / "sources_3fixed.npy")
    return sensors, sources


def save_simulation_results(
    time_vector: np.ndarray,
    source_waveforms: np.ndarray,
    noisy_data: np.ndarray,
    output_dir: Path | str = None,
) -> dict[str, Path]:
    """
    Save simulation results to disk.

    Parameters
    ----------
    time_vector : np.ndarray
        Time vector.
    source_waveforms : np.ndarray
        Source waveforms.
    noisy_data : np.ndarray
        Noisy sensor recording.
    output_dir : Path, optional
        Output directory. Defaults to data/raw.

    Returns
    -------
    dict
        Paths to saved files.
    """
    if output_dir is None:
        output_dir = get_project_root() / "data" / "raw"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save time vector
    path_time = output_dir / "time_vector.npy"
    np.save(path_time, time_vector)
    paths["time_vector"] = path_time

    # Save source waveforms (ground truth)
    path_sources = output_dir / "source_waveforms.npy"
    np.save(path_sources, source_waveforms)
    paths["source_waveforms"] = path_sources

    # Save noisy recording
    path_recording = output_dir / "recording_simulation.npy"
    np.save(path_recording, noisy_data)
    paths["recording_simulation"] = path_recording

    return paths


def main() -> None:
    """Run Phase 2 time-domain simulation."""
    print("=" * 60)
    print("  PHASE 2: Temporal Dynamics Simulation")
    print("=" * 60)

    # Load Phase 1 data
    print("\n[1/4] Loading Phase 1 data...")
    sensors, sources = load_phase1_data()
    print(f"  Sensors: {sensors.shape}")
    print(f"  Sources: {sources.shape}")

    # Run simulation
    print("\n[2/4] Running forward simulation...")
    time_vector, source_waveforms, clean_data, noisy_data = simulate_recording(
        sensors, sources
    )

    # Validate
    print("\n[3/4] Validation checks...")
    print(f"  Time range: [{time_vector[0]:.3f}, {time_vector[-1]:.3f}] sec")
    print(f"  Clean data range: [{clean_data.min():.2e}, {clean_data.max():.2e}]")
    print(f"  Noisy data range: [{noisy_data.min():.2e}, {noisy_data.max():.2e}]")

    # Verify noise was added
    noise_power = np.mean((noisy_data - clean_data) ** 2)
    signal_power = np.mean(clean_data ** 2)
    measured_snr = signal_power / noise_power
    print(f"  Measured SNR: {measured_snr:.2f} (target: {SNR_LEVEL})")

    # Save results
    print("\n[4/4] Saving results...")
    paths = save_simulation_results(time_vector, source_waveforms, noisy_data)
    for name, path in paths.items():
        print(f"  {name}: {path.name}")

    print("\n" + "=" * 60)
    print("  Phase 2 simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
