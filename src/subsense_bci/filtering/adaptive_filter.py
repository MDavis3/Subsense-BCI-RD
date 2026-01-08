"""
Adaptive Filtering Module - Hemodynamic Artifact Rejection

Implements adaptive filters (LMS and RLS) for canceling cardiac-correlated
artifacts from sensor recordings. These artifacts arise from hemodynamic
drift causing time-varying lead fields.

Theory:
    The cardiac pulsatility at ~1.2 Hz creates a periodic modulation in the
    sensor signals that correlates with the PPG reference. Adaptive filtering
    uses this reference to estimate and subtract the artifact component.

Filter Model:
    y[n] = d[n] - w^T @ x[n]

    where:
    - d[n]: Corrupted sensor signal (desired signal + artifact)
    - x[n]: Reference signal vector (PPG and its delays)
    - w: Adaptive filter weights
    - y[n]: Cleaned output (error signal)

References:
    - Haykin, S. (2002). Adaptive Filter Theory. Prentice Hall.
    - Widrow, B., & Stearns, S. (1985). Adaptive Signal Processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from subsense_bci.physics.constants import (
    ALPHA_BAND_HZ,
    BETA_BAND_HZ,
    CARDIAC_FREQUENCY_HZ,
    DEFAULT_RANDOM_SEED,
    REALTIME_LATENCY_BUDGET_MS,
    SAMPLING_RATE_HZ,
)


@dataclass
class LMSFilter:
    """
    Least Mean Squares (LMS) adaptive filter.

    The LMS algorithm minimizes the mean squared error using stochastic
    gradient descent with instantaneous gradient estimates.

    Update rule:
        w[n+1] = w[n] + mu * e[n] * x[n]

    where e[n] = d[n] - y_hat[n] is the error signal.

    Parameters
    ----------
    n_taps : int
        Number of filter taps (delay line length). Default is 32.
    mu : float
        Step size / learning rate. Default is 0.01.
        Must be 0 < mu < 2/lambda_max where lambda_max is the
        largest eigenvalue of the input autocorrelation matrix.

    Attributes
    ----------
    w : np.ndarray
        Filter weight vector, shape (n_taps,).
    x_buffer : np.ndarray
        Delay line buffer, shape (n_taps,).

    Notes
    -----
    Convergence speed: O(1/(mu * eigenvalue_spread))
    Stability: Requires mu < 2/trace(R_xx) for convergence

    Examples
    --------
    >>> filt = LMSFilter(n_taps=16, mu=0.01)
    >>> for i in range(100):
    ...     y = filt.update(corrupted_signal[i], reference[i])
    """

    n_taps: int = 32
    mu: float = 0.01
    w: np.ndarray = field(default_factory=lambda: np.array([]))
    x_buffer: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        """Initialize filter state."""
        if self.w.size == 0:
            self.w = np.zeros(self.n_taps, dtype=np.float64)
        if self.x_buffer.size == 0:
            self.x_buffer = np.zeros(self.n_taps, dtype=np.float64)

    def reset(self) -> None:
        """Reset filter to initial state."""
        self.w = np.zeros(self.n_taps, dtype=np.float64)
        self.x_buffer = np.zeros(self.n_taps, dtype=np.float64)

    def update(self, d: float, x: float) -> float:
        """
        Process one sample and update filter weights.

        Parameters
        ----------
        d : float
            Corrupted signal sample (contains desired signal + artifact).
        x : float
            Reference signal sample (correlated with artifact only).

        Returns
        -------
        float
            Cleaned output (error signal = d - estimated_artifact).
        """
        # Shift delay line and insert new sample
        self.x_buffer = np.roll(self.x_buffer, 1)
        self.x_buffer[0] = x

        # Compute filter output (estimated artifact)
        y_hat = np.dot(self.w, self.x_buffer)

        # Compute error (cleaned signal)
        e = d - y_hat

        # Update weights using LMS rule
        self.w = self.w + self.mu * e * self.x_buffer

        return e

    def filter_batch(self, d: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Filter an entire signal batch.

        Parameters
        ----------
        d : np.ndarray
            Corrupted signal, shape (n_samples,).
        x : np.ndarray
            Reference signal, shape (n_samples,).

        Returns
        -------
        np.ndarray
            Cleaned signal, shape (n_samples,).
        """
        if len(d) != len(x):
            raise ValueError("d and x must have same length")

        output = np.zeros_like(d)
        for i in range(len(d)):
            output[i] = self.update(d[i], x[i])

        return output


@dataclass
class RLSFilter:
    """
    Recursive Least Squares (RLS) adaptive filter.

    The RLS algorithm minimizes the exponentially weighted sum of squared
    errors, providing faster convergence than LMS at higher computational
    cost.

    Update rules:
        k[n] = P[n-1] @ x[n] / (lambda + x[n]^T @ P[n-1] @ x[n])
        e[n] = d[n] - w[n-1]^T @ x[n]
        w[n] = w[n-1] + k[n] * e[n]
        P[n] = (P[n-1] - k[n] @ x[n]^T @ P[n-1]) / lambda

    Parameters
    ----------
    n_taps : int
        Number of filter taps. Default is 32.
    lambda_ : float
        Forgetting factor in (0, 1]. Default is 0.99.
        lambda = 1: infinite memory (all samples weighted equally)
        lambda < 1: exponential forgetting of past samples
    delta : float
        Regularization for initial P matrix. Default is 0.01.
        P[0] = I / delta

    Attributes
    ----------
    w : np.ndarray
        Filter weight vector, shape (n_taps,).
    P : np.ndarray
        Inverse correlation matrix estimate, shape (n_taps, n_taps).
    x_buffer : np.ndarray
        Delay line buffer, shape (n_taps,).

    Notes
    -----
    Convergence speed: O(n_taps) samples (very fast)
    Complexity: O(n_taps^2) per sample (matrix updates)
    Numerical stability: May require periodic re-initialization of P

    Examples
    --------
    >>> filt = RLSFilter(n_taps=16, lambda_=0.99)
    >>> for i in range(100):
    ...     y = filt.update(corrupted_signal[i], reference[i])
    """

    n_taps: int = 32
    lambda_: float = 0.99
    delta: float = 0.01
    w: np.ndarray = field(default_factory=lambda: np.array([]))
    P: np.ndarray = field(default_factory=lambda: np.array([]))
    x_buffer: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        """Initialize filter state."""
        if self.w.size == 0:
            self.w = np.zeros(self.n_taps, dtype=np.float64)
        if self.P.size == 0:
            self.P = np.eye(self.n_taps, dtype=np.float64) / self.delta
        if self.x_buffer.size == 0:
            self.x_buffer = np.zeros(self.n_taps, dtype=np.float64)

    def reset(self) -> None:
        """Reset filter to initial state."""
        self.w = np.zeros(self.n_taps, dtype=np.float64)
        self.P = np.eye(self.n_taps, dtype=np.float64) / self.delta
        self.x_buffer = np.zeros(self.n_taps, dtype=np.float64)

    def update(self, d: float, x: float) -> float:
        """
        Process one sample and update filter weights.

        Parameters
        ----------
        d : float
            Corrupted signal sample (contains desired signal + artifact).
        x : float
            Reference signal sample (correlated with artifact only).

        Returns
        -------
        float
            Cleaned output (error signal = d - estimated_artifact).
        """
        # Shift delay line and insert new sample
        self.x_buffer = np.roll(self.x_buffer, 1)
        self.x_buffer[0] = x

        # Compute gain vector k
        Px = self.P @ self.x_buffer
        denominator = self.lambda_ + self.x_buffer @ Px
        k = Px / denominator

        # Compute a priori error
        y_hat = np.dot(self.w, self.x_buffer)
        e = d - y_hat

        # Update weights
        self.w = self.w + k * e

        # Update inverse correlation matrix
        self.P = (self.P - np.outer(k, self.x_buffer @ self.P)) / self.lambda_

        return e

    def filter_batch(self, d: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Filter an entire signal batch.

        Parameters
        ----------
        d : np.ndarray
            Corrupted signal, shape (n_samples,).
        x : np.ndarray
            Reference signal, shape (n_samples,).

        Returns
        -------
        np.ndarray
            Cleaned signal, shape (n_samples,).
        """
        if len(d) != len(x):
            raise ValueError("d and x must have same length")

        output = np.zeros_like(d)
        for i in range(len(d)):
            output[i] = self.update(d[i], x[i])

        return output


@dataclass
class PhaseAwareRLSFilter:
    """
    RLS filter with per-sensor phase compensation for cardiac artifact rejection.

    This filter extends RLSFilter to handle phase-delayed cardiac artifacts
    where different sensors experience the pulse wave at different times due
    to propagation through the arterial system.

    Instead of using the current reference sample, this filter indexes into
    a reference buffer at a phase-compensated position, aligning the artifact
    template with each sensor's actual artifact timing.

    Parameters
    ----------
    n_taps : int
        Number of filter taps. Reduced from default 32 to 8 to meet
        real-time latency budget (~43ms for BCI applications).
    lambda_ : float
        Forgetting factor. Default 0.95 for faster adaptation than
        standard RLS (0.99), needed to track non-stationary artifacts.
    delta : float
        Regularization for initial P matrix. Default is 0.01.
    phase_offset : float
        Phase offset for this sensor in radians.
    sampling_rate_hz : float
        Sampling rate for delay calculation. Default is 1000 Hz.
    cardiac_freq_hz : float
        Cardiac frequency for phase-to-sample conversion. Default is 1.2 Hz.

    Notes
    -----
    Phase delay in samples = (phase_offset / (2π)) × (sampling_rate / cardiac_freq)

    At 1 kHz sampling and 1.2 Hz cardiac:
    - One cardiac cycle = 833 samples
    - Phase offset of π = 417 samples delay

    The reference buffer must be large enough to accommodate the maximum
    expected delay plus filter tap length.
    """

    n_taps: int = 8  # Reduced for 43ms latency budget
    lambda_: float = 0.95  # Faster adaptation for non-stationary artifacts
    delta: float = 0.01
    phase_offset: float = 0.0
    sampling_rate_hz: float = SAMPLING_RATE_HZ
    cardiac_freq_hz: float = CARDIAC_FREQUENCY_HZ

    # Internal RLS filter
    _rls_filter: RLSFilter | None = field(default=None, init=False, repr=False)
    _reference_buffer: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    _buffer_size: int = field(default=0, init=False, repr=False)
    _sample_delay: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize internal RLS filter and compute sample delay."""
        # Create internal RLS filter
        self._rls_filter = RLSFilter(
            n_taps=self.n_taps,
            lambda_=self.lambda_,
            delta=self.delta,
        )

        # Compute phase delay in samples
        # delay = (phase / 2π) × (samples_per_cycle)
        samples_per_cycle = self.sampling_rate_hz / self.cardiac_freq_hz
        self._sample_delay = int((self.phase_offset / (2 * np.pi)) * samples_per_cycle)

        # Buffer size: enough for max delay + filter taps + margin
        # Cap at 2 cardiac cycles to bound memory
        max_delay = int(2 * samples_per_cycle)
        self._buffer_size = min(max_delay, abs(self._sample_delay) + self.n_taps + 50)
        self._buffer_size = max(self._buffer_size, self.n_taps + 10)  # Minimum buffer

        # Initialize circular buffer
        self._reference_buffer = np.zeros(self._buffer_size, dtype=np.float64)
        self._buffer_idx = 0

    def reset(self) -> None:
        """Reset filter and buffer to initial state."""
        if self._rls_filter is not None:
            self._rls_filter.reset()
        self._reference_buffer = np.zeros(self._buffer_size, dtype=np.float64)
        self._buffer_idx = 0

    def update_with_phase_compensation(
        self,
        d: float,
        reference_sample: float,
    ) -> float:
        """
        Process one sample with phase-compensated reference.

        Updates the reference buffer and retrieves the phase-delayed
        reference value for artifact estimation.

        Parameters
        ----------
        d : float
            Corrupted signal sample.
        reference_sample : float
            Current reference (e.g., PPG) sample.

        Returns
        -------
        float
            Cleaned output sample.
        """
        # Add current reference to circular buffer
        self._reference_buffer[self._buffer_idx] = reference_sample

        # Compute index for phase-delayed reference
        # Negative delay means artifact arrives earlier at this sensor
        delayed_idx = (self._buffer_idx - self._sample_delay) % self._buffer_size

        # Get phase-compensated reference
        x_compensated = self._reference_buffer[delayed_idx]

        # Update buffer index
        self._buffer_idx = (self._buffer_idx + 1) % self._buffer_size

        # Apply RLS with compensated reference
        return self._rls_filter.update(d, x_compensated)

    def filter_batch(self, d: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Filter an entire signal batch with phase compensation.

        Parameters
        ----------
        d : np.ndarray
            Corrupted signal, shape (n_samples,).
        x : np.ndarray
            Reference signal, shape (n_samples,).

        Returns
        -------
        np.ndarray
            Cleaned signal, shape (n_samples,).
        """
        if len(d) != len(x):
            raise ValueError("d and x must have same length")

        output = np.zeros_like(d)
        for i in range(len(d)):
            output[i] = self.update_with_phase_compensation(d[i], x[i])

        return output


def detect_harmonic_aliasing(
    cardiac_freq_hz: float = CARDIAC_FREQUENCY_HZ,
    neural_bands: dict | None = None,
    n_harmonics: int = 10,
) -> dict[str, list[float]]:
    """
    Detect cardiac harmonics that overlap with neural frequency bands.

    Cardiac artifacts contain harmonics at integer multiples of the
    fundamental frequency. Some of these may fall within neural bands
    of interest (e.g., alpha, beta), causing aliasing issues that
    adaptive filtering alone cannot resolve.

    Parameters
    ----------
    cardiac_freq_hz : float
        Fundamental cardiac frequency in Hz. Default is 1.2 (~72 BPM).
    neural_bands : dict, optional
        Dictionary mapping band names to (low, high) frequency tuples.
        Default is {"alpha": (8, 13), "beta": (13, 30)}.
    n_harmonics : int
        Number of harmonics to check. Default is 10.

    Returns
    -------
    dict[str, list[float]]
        Dictionary mapping each neural band to list of overlapping
        cardiac harmonic frequencies.

    Notes
    -----
    Example: For cardiac_freq=1.2 Hz:
    - 8th harmonic = 9.6 Hz (falls in alpha band 8-13 Hz)
    - 10th harmonic = 12.0 Hz (falls in alpha band 8-13 Hz)

    These require additional handling (e.g., notch filtering) beyond
    adaptive artifact rejection.

    Examples
    --------
    >>> overlaps = detect_harmonic_aliasing(cardiac_freq_hz=1.2)
    >>> overlaps['alpha']
    [9.6, 10.8, 12.0]
    """
    if neural_bands is None:
        neural_bands = {
            "alpha": ALPHA_BAND_HZ,
            "beta": BETA_BAND_HZ,
        }

    # Compute harmonics
    harmonics = [cardiac_freq_hz * n for n in range(1, n_harmonics + 1)]

    # Check overlap with each band
    overlaps = {}
    for band_name, (low, high) in neural_bands.items():
        overlapping_harmonics = [h for h in harmonics if low <= h <= high]
        overlaps[band_name] = overlapping_harmonics

    return overlaps


def recommend_filter_strategy(
    cardiac_freq_hz: float = CARDIAC_FREQUENCY_HZ,
    target_band: str = "alpha",
    n_sensors: int = 10000,
    latency_budget_ms: float = REALTIME_LATENCY_BUDGET_MS,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
) -> dict:
    """
    Recommend adaptive filter parameters for given constraints.

    Analyzes the trade-offs between filter complexity (n_taps), latency,
    and artifact rejection capability to suggest optimal parameters.

    Parameters
    ----------
    cardiac_freq_hz : float
        Cardiac frequency in Hz. Default is 1.2.
    target_band : str
        Neural band to preserve. Default is "alpha".
    n_sensors : int
        Number of sensors to process. Default is 10,000.
    latency_budget_ms : float
        Maximum allowed processing latency in ms. Default is 43.
    sampling_rate_hz : float
        Sampling rate in Hz. Default is 1000.

    Returns
    -------
    dict
        Recommendations including:
        - recommended_method: 'rls' or 'phase_aware_rls'
        - recommended_n_taps: Suggested number of taps
        - recommended_lambda: Suggested forgetting factor
        - harmonic_warnings: List of problematic harmonics
        - suggested_notch_frequencies: Frequencies to notch filter
        - estimated_latency_ms: Estimated processing latency

    Notes
    -----
    RLS complexity: O(n_taps²) per sample
    For n_sensors=10000, n_taps=8, 100 samples/chunk:
    - Operations: 10000 × 100 × 64 = 64M ops
    - At ~1 GFLOP/s: ~64ms (exceeds budget)
    - With n_taps=8 and optimizations: ~12-15ms (within budget)
    """
    # Detect harmonic aliasing
    overlaps = detect_harmonic_aliasing(
        cardiac_freq_hz=cardiac_freq_hz,
        n_harmonics=15,  # Check more harmonics
    )

    # Estimate complexity and latency
    # RLS: O(n_taps²) per sample
    # Assuming ~100 samples per 100ms chunk, 1e9 ops/sec processing speed
    def estimate_latency(n_taps: int, n_sensors: int, chunk_samples: int) -> float:
        ops_per_sample = n_taps * n_taps  # Matrix operations
        total_ops = n_sensors * chunk_samples * ops_per_sample
        gflops = total_ops / 1e9
        return gflops * 15  # Rough conversion to ms (assuming ~66 GFLOP/s)

    chunk_samples = 100  # Assuming 100ms chunks at 1kHz

    # Try different tap counts
    candidate_taps = [4, 8, 16, 32]
    best_taps = 8  # Default
    for n_taps in candidate_taps:
        est_latency = estimate_latency(n_taps, n_sensors, chunk_samples)
        if est_latency < latency_budget_ms * 0.8:  # Leave 20% margin
            best_taps = n_taps

    # Get harmonic warnings for target band
    harmonic_warnings = overlaps.get(target_band, [])

    # Suggest notch frequencies (harmonics in target band)
    suggested_notch = harmonic_warnings.copy()

    # Determine method
    recommended_method = "phase_aware_rls"  # Default for cardiac artifacts

    # Compute estimated latency
    estimated_latency = estimate_latency(best_taps, n_sensors, chunk_samples)

    return {
        "recommended_method": recommended_method,
        "recommended_n_taps": best_taps,
        "recommended_lambda": 0.95,  # Fast adaptation for non-stationary
        "harmonic_warnings": harmonic_warnings,
        "suggested_notch_frequencies": suggested_notch,
        "estimated_latency_ms": estimated_latency,
        "latency_budget_ms": latency_budget_ms,
        "within_budget": estimated_latency < latency_budget_ms,
    }


def apply_adaptive_cancellation(
    recording: np.ndarray,
    reference: np.ndarray,
    method: str = "rls",
    n_taps: int = 32,
    mu: float = 0.01,
    lambda_: float = 0.99,
    verbose: bool = False,
) -> np.ndarray:
    """
    Apply adaptive artifact cancellation to multi-channel recording.

    Processes each sensor channel independently using the shared reference
    signal to estimate and remove cardiac-correlated artifacts.

    Parameters
    ----------
    recording : np.ndarray
        Sensor recording with shape (n_sensors, n_samples).
    reference : np.ndarray
        PPG or cardiac reference with shape (n_samples,).
    method : str, optional
        Filter method: 'lms' or 'rls'. Default is 'rls'.
    n_taps : int, optional
        Number of filter taps. Default is 32.
    mu : float, optional
        LMS step size. Default is 0.01.
    lambda_ : float, optional
        RLS forgetting factor. Default is 0.99.
    verbose : bool, optional
        Print progress. Default is False.

    Returns
    -------
    np.ndarray
        Cleaned recording with shape (n_sensors, n_samples).

    Examples
    --------
    >>> cleaned = apply_adaptive_cancellation(recording, ppg_ref, method='rls')
    >>> artifact_power = np.mean((recording - cleaned) ** 2)
    """
    recording = np.asarray(recording, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if recording.ndim != 2:
        raise ValueError(f"recording must be 2D, got shape {recording.shape}")
    if reference.ndim != 1:
        raise ValueError(f"reference must be 1D, got shape {reference.shape}")
    if recording.shape[1] != len(reference):
        raise ValueError(
            f"recording has {recording.shape[1]} samples but reference has {len(reference)}"
        )

    n_sensors, n_samples = recording.shape
    cleaned = np.zeros_like(recording)

    if verbose:
        print(f"Applying {method.upper()} adaptive cancellation to {n_sensors} sensors...")

    progress_interval = max(1, n_sensors // 10)

    for i in range(n_sensors):
        # Create filter for this channel
        if method.lower() == "lms":
            filt = LMSFilter(n_taps=n_taps, mu=mu)
        elif method.lower() == "rls":
            filt = RLSFilter(n_taps=n_taps, lambda_=lambda_)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'lms' or 'rls'.")

        # Filter this channel
        cleaned[i] = filt.filter_batch(recording[i], reference)

        if verbose and (i + 1) % progress_interval == 0:
            print(f"  Progress: {100 * (i + 1) / n_sensors:.0f}%")

    return cleaned


def compute_artifact_rejection_snr(
    original: np.ndarray,
    cleaned: np.ndarray,
    reference: np.ndarray,
) -> dict:
    """
    Compute metrics for artifact rejection quality.

    Parameters
    ----------
    original : np.ndarray
        Original recording with shape (n_sensors, n_samples).
    cleaned : np.ndarray
        Cleaned recording with shape (n_sensors, n_samples).
    reference : np.ndarray
        PPG reference with shape (n_samples,).

    Returns
    -------
    dict
        Metrics including:
        - artifact_power_before: Power correlated with reference (before)
        - artifact_power_after: Power correlated with reference (after)
        - rejection_db: Artifact reduction in dB
        - signal_distortion: Power change in cleaned signal
    """
    # Compute artifact as removed component
    artifact = original - cleaned

    # Compute power metrics
    artifact_power = np.mean(artifact**2)
    original_power = np.mean(original**2)
    cleaned_power = np.mean(cleaned**2)

    # Compute correlation with reference
    # High correlation before, low after indicates good rejection
    def channel_ref_correlation(data: np.ndarray, ref: np.ndarray) -> float:
        """Mean absolute correlation across channels."""
        correlations = []
        for i in range(data.shape[0]):
            r = np.corrcoef(data[i], ref)[0, 1]
            correlations.append(abs(r) if not np.isnan(r) else 0)
        return np.mean(correlations)

    corr_before = channel_ref_correlation(original, reference)
    corr_after = channel_ref_correlation(cleaned, reference)

    # Rejection ratio in dB
    if corr_after > 1e-10:
        rejection_db = 20 * np.log10(corr_before / corr_after)
    else:
        rejection_db = float("inf")

    return {
        "artifact_power": artifact_power,
        "original_power": original_power,
        "cleaned_power": cleaned_power,
        "correlation_before": corr_before,
        "correlation_after": corr_after,
        "rejection_db": rejection_db,
    }


@dataclass
class AdaptiveFilterHook:
    """
    Pre-processing hook for online adaptive artifact rejection.

    This class wraps LMS/RLS/PhaseAwareRLS filters to provide chunk-by-chunk
    artifact rejection that can be integrated into the OnlineDecoder pipeline.
    Implements the PreProcessingHook protocol.

    Filter states are maintained across chunks, allowing the adaptive
    filters to continuously track and cancel time-varying artifacts.

    Parameters
    ----------
    method : str
        Filter method: 'lms', 'rls', or 'phase_aware_rls'. Default is 'rls'.
        Use 'phase_aware_rls' for sensors with spatially-varying phase delays.
    n_taps : int
        Number of filter taps. Default is 32 for standard RLS, 8 for phase_aware_rls
        to meet real-time latency constraints.
    mu : float
        LMS step size. Default is 0.01.
    lambda_ : float
        RLS forgetting factor. Default is 0.99 for standard RLS, 0.95 for
        phase_aware_rls (faster adaptation for non-stationary artifacts).
    phase_offsets : np.ndarray, optional
        Per-sensor phase offsets in radians for phase_aware_rls method.
        Shape (n_sensors,). Required for phase_aware_rls.
    sampling_rate_hz : float
        Sampling rate for phase delay calculation. Default is 1000 Hz.
    cardiac_freq_hz : float
        Cardiac frequency for phase delay calculation. Default is 1.2 Hz.

    Attributes
    ----------
    filters : list
        Per-channel filter instances (created lazily on first call).
    n_channels : int
        Number of channels (set on first call).

    Examples
    --------
    >>> from subsense_bci.filtering.online_decoder import OnlineDecoder
    >>> hook = AdaptiveFilterHook(method='rls', n_taps=32)
    >>> decoder = OnlineDecoder.from_phase3_data()
    >>> decoder.pre_processor = hook
    >>> # Now decoder.decode() will apply artifact rejection automatically

    >>> # For phase-aware filtering with cardiac propagation:
    >>> from subsense_bci.simulation.sensor_cloud import SensorCloud
    >>> cloud = SensorCloud.from_uniform_cloud_with_cardiac_propagation(n_sensors=100)
    >>> hook = AdaptiveFilterHook.from_sensor_cloud(cloud, method='phase_aware_rls')
    """

    method: str = "rls"
    n_taps: int = 32
    mu: float = 0.01
    lambda_: float = 0.99
    phase_offsets: np.ndarray | None = None
    sampling_rate_hz: float = SAMPLING_RATE_HZ
    cardiac_freq_hz: float = CARDIAC_FREQUENCY_HZ
    _filters: list = field(default_factory=list, repr=False)
    _n_channels: int = field(default=0, repr=False)

    def set_phase_offsets(self, phase_offsets: np.ndarray) -> None:
        """
        Set per-sensor phase offsets for phase_aware_rls method.

        Must be called before process() if using phase_aware_rls without
        using from_sensor_cloud() factory.

        Parameters
        ----------
        phase_offsets : np.ndarray
            Per-sensor phase offsets in radians, shape (n_sensors,).
        """
        self.phase_offsets = np.asarray(phase_offsets, dtype=np.float64)
        # Reset filters so they will be reinitialized with new phase offsets
        self.reset()

    @classmethod
    def from_sensor_cloud(
        cls,
        sensor_cloud,
        method: str = "phase_aware_rls",
        n_taps: int = 8,
        lambda_: float = 0.95,
    ) -> "AdaptiveFilterHook":
        """
        Create an AdaptiveFilterHook from a SensorCloud with cardiac propagation.

        Extracts phase offsets from the sensor cloud for phase-aware filtering.

        Parameters
        ----------
        sensor_cloud : SensorCloud
            Sensor cloud with phase_offsets attribute.
        method : str
            Filter method. Default is 'phase_aware_rls'.
        n_taps : int
            Number of filter taps. Default is 8 for latency budget.
        lambda_ : float
            Forgetting factor. Default is 0.95 for fast adaptation.

        Returns
        -------
        AdaptiveFilterHook
            Hook configured with phase offsets from sensor cloud.
        """
        phase_offsets = sensor_cloud.phase_offsets

        return cls(
            method=method,
            n_taps=n_taps,
            lambda_=lambda_,
            phase_offsets=phase_offsets,
            sampling_rate_hz=SAMPLING_RATE_HZ,
            cardiac_freq_hz=sensor_cloud.drift_frequency_hz,
        )

    def process(
        self,
        chunk: np.ndarray,
        timestamp: float,
        reference: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply adaptive filtering to a chunk.

        If no reference signal is provided, returns the chunk unchanged.
        On the first call, creates per-channel filter instances.

        Parameters
        ----------
        chunk : np.ndarray
            Input sensor data, shape (n_sensors, chunk_samples).
        timestamp : float
            Timestamp of chunk start in seconds (unused, for protocol compatibility).
        reference : np.ndarray, optional
            Reference signal (e.g., PPG/ECG), shape (chunk_samples,).

        Returns
        -------
        np.ndarray
            Filtered chunk with same shape as input.
        """
        # If no reference, cannot filter - return unchanged
        if reference is None:
            return chunk

        n_sensors, chunk_samples = chunk.shape

        # Validate reference length
        if len(reference) != chunk_samples:
            raise ValueError(
                f"Reference length ({len(reference)}) must match "
                f"chunk samples ({chunk_samples})"
            )

        # Initialize filters on first call
        if self._n_channels == 0:
            self._n_channels = n_sensors
            self._filters = []

            method_lower = self.method.lower()

            for i in range(n_sensors):
                if method_lower == "lms":
                    self._filters.append(LMSFilter(n_taps=self.n_taps, mu=self.mu))
                elif method_lower == "rls":
                    self._filters.append(RLSFilter(n_taps=self.n_taps, lambda_=self.lambda_))
                elif method_lower == "phase_aware_rls":
                    # Validate phase offsets for phase-aware filtering
                    if self.phase_offsets is None:
                        raise ValueError(
                            "phase_offsets must be set for phase_aware_rls method. "
                            "Use set_phase_offsets() or from_sensor_cloud()."
                        )
                    if len(self.phase_offsets) != n_sensors:
                        raise ValueError(
                            f"phase_offsets has {len(self.phase_offsets)} entries "
                            f"but chunk has {n_sensors} channels."
                        )

                    self._filters.append(
                        PhaseAwareRLSFilter(
                            n_taps=self.n_taps,
                            lambda_=self.lambda_,
                            phase_offset=self.phase_offsets[i],
                            sampling_rate_hz=self.sampling_rate_hz,
                            cardiac_freq_hz=self.cardiac_freq_hz,
                        )
                    )
                else:
                    raise ValueError(
                        f"Unknown method: {self.method}. "
                        f"Use 'lms', 'rls', or 'phase_aware_rls'."
                    )

        # Check channel count matches
        if n_sensors != self._n_channels:
            raise ValueError(
                f"Chunk has {n_sensors} channels but filter was initialized "
                f"for {self._n_channels} channels. Call reset() to reinitialize."
            )

        # Apply per-channel filtering
        output = np.zeros_like(chunk)
        for i in range(n_sensors):
            output[i] = self._filters[i].filter_batch(chunk[i], reference)

        return output

    def reset(self) -> None:
        """
        Reset all filter states.

        Clears filter weights and buffers, and resets channel count
        so filters will be reinitialized on next process() call.
        """
        for filt in self._filters:
            filt.reset()
        self._filters = []
        self._n_channels = 0

    @classmethod
    def from_config(cls) -> "AdaptiveFilterHook":
        """
        Create an AdaptiveFilterHook from the default config.

        Returns
        -------
        AdaptiveFilterHook
            Hook configured from biology.artifact_rejection settings.
        """
        from subsense_bci.config import load_config

        config = load_config()
        ar_config = config.get("biology", {}).get("artifact_rejection", {})

        return cls(
            method=ar_config.get("method", "rls"),
            n_taps=ar_config.get("n_taps", 32),
            mu=ar_config.get("mu", 0.01),
            lambda_=ar_config.get("lambda_", 0.99),
        )
