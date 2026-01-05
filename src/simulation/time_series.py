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

import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from physics.constants import (
    SAMPLING_RATE_HZ,
    DURATION_SEC,
    SNR_LEVEL,
    DEFAULT_RANDOM_SEED,
)
from physics.transfer_function import compute_lead_field


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
    Generate pink noise (1/f noise) using the Voss-McCartney algorithm.

    Pink noise has equal energy per octave, with power spectral density
    proportional to 1/f.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Pink noise signal normalized to unit amplitude.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate white noise in frequency domain
    white = np.fft.rfft(np.random.randn(n_samples))

    # Create 1/f filter (avoid division by zero at DC)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1e-10  # Avoid division by zero
    pink_filter = 1.0 / np.sqrt(freqs)

    # Apply filter and transform back
    pink = np.fft.irfft(white * pink_filter, n=n_samples)

    # Normalize to unit amplitude
    pink = pink / np.max(np.abs(pink))

    return pink


def generate_source_waveforms(
    time_vector: np.ndarray,
    seed: int = DEFAULT_RANDOM_SEED,
) -> np.ndarray:
    """
    Generate source waveforms for all 3 neural sources.

    Parameters
    ----------
    time_vector : np.ndarray
        Time vector in seconds.
    seed : int
        Random seed for pink noise generation.

    Returns
    -------
    np.ndarray
        Source waveforms with shape (n_sources=3, n_samples).
        Row 0: Source A - 10 Hz sine (Alpha)
        Row 1: Source B - 20 Hz sine (Beta)
        Row 2: Source C - Pink noise (1/f)
    """
    n_samples = len(time_vector)

    # Source A: 10 Hz Alpha wave
    source_a = np.sin(2 * np.pi * 10.0 * time_vector)

    # Source B: 20 Hz Beta wave
    source_b = np.sin(2 * np.pi * 20.0 * time_vector)

    # Source C: Pink noise (1/f background activity)
    source_c = generate_pink_noise(n_samples, seed=seed + 100)

    # Stack into (n_sources, n_samples) matrix
    # All are already normalized to unit amplitude (sine waves are [-1, 1])
    source_waveforms = np.vstack([source_a, source_b, source_c])

    return source_waveforms


def add_sensor_noise(
    clean_data: np.ndarray,
    snr: float = SNR_LEVEL,
    seed: int = DEFAULT_RANDOM_SEED,
) -> np.ndarray:
    """
    Add Gaussian white noise to sensor data based on SNR.

    The noise level is scaled relative to the RMS power of the clean signal.

    SNR Definition (linear):
        SNR = signal_power / noise_power
        noise_std = signal_rms / sqrt(SNR)

    Parameters
    ----------
    clean_data : np.ndarray
        Clean sensor data with shape (n_sensors, n_samples).
    snr : float
        Signal-to-noise ratio (linear scale, not dB).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy sensor data with shape (n_sensors, n_samples).
    """
    np.random.seed(seed + 200)

    # Compute signal RMS across all sensors and time
    signal_rms = np.sqrt(np.mean(clean_data ** 2))

    # Compute noise standard deviation from SNR
    # SNR = (signal_rms^2) / (noise_std^2)
    # noise_std = signal_rms / sqrt(SNR)
    noise_std = signal_rms / np.sqrt(snr)

    # Generate Gaussian white noise
    noise = np.random.randn(*clean_data.shape) * noise_std

    return clean_data + noise


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

    # Step 2: Generate source waveforms
    source_waveforms = generate_source_waveforms(time_vector, seed=seed)
    print(f"  Source waveforms: {source_waveforms.shape}")

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
    noisy_data = add_sensor_noise(clean_data, snr=snr, seed=seed)
    print(f"  Noisy data: {noisy_data.shape}, SNR={snr}")

    return time_vector, source_waveforms, clean_data, noisy_data


def load_phase1_data() -> tuple[np.ndarray, np.ndarray]:
    """Load sensor and source data from Phase 1."""
    data_dir = project_root / "data" / "raw"
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
        output_dir = project_root / "data" / "raw"
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
