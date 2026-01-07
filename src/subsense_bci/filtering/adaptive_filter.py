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

from subsense_bci.physics.constants import CARDIAC_FREQUENCY_HZ, DEFAULT_RANDOM_SEED


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
