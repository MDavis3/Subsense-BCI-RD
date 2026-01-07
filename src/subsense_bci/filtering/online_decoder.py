"""
Online Decoder - Phase 4 Real-Time Source Recovery

Applies pre-trained PCA and ICA transformations to incoming sensor data
chunks in real-time, recovering the 3 independent neural source signals.

This is a "Static Decoder" approach: the PCA/ICA matrices are trained
once on the full recording (Phase 3) and then applied to streaming chunks.

Usage:
    from subsense_bci.filtering.online_decoder import OnlineDecoder
    
    decoder = OnlineDecoder.from_training_data(recording, ground_truth)
    
    for chunk, timestamp in streamer.get_chunks():
        result = decoder.decode(chunk)
        print(f"Latency: {result.latency_ms:.2f}ms")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler

from subsense_bci.filtering.unmixing import (
    load_phase2_data,
    match_sources,
)
from subsense_bci.config import load_config


@runtime_checkable
class PreProcessingHook(Protocol):
    """
    Protocol for pre-processing hooks in the decoding pipeline.

    Pre-processing hooks are applied to each chunk before PCA/ICA decoding.
    This enables runtime artifact rejection, filtering, or other signal
    conditioning without modifying the core decoder logic.

    Implementations must provide:
    - process(): Apply the pre-processing to a chunk
    - reset(): Reset any internal state (e.g., filter memory)

    Examples
    --------
    >>> class MyFilter:
    ...     def process(self, chunk, timestamp, reference=None):
    ...         return chunk * 0.5  # Example: attenuate signal
    ...     def reset(self):
    ...         pass
    >>> decoder = OnlineDecoder.from_phase3_data()
    >>> decoder.pre_processor = MyFilter()
    """

    def process(
        self,
        chunk: np.ndarray,
        timestamp: float,
        reference: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply pre-processing to a chunk.

        Parameters
        ----------
        chunk : np.ndarray
            Input sensor data, shape (n_sensors, chunk_samples).
        timestamp : float
            Timestamp of chunk start in seconds.
        reference : np.ndarray, optional
            Reference signal for artifact rejection (e.g., PPG/ECG).
            Shape (chunk_samples,) if provided.

        Returns
        -------
        np.ndarray
            Processed chunk with same shape as input.
        """
        ...

    def reset(self) -> None:
        """Reset internal state (e.g., filter memory)."""
        ...


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent.parent


@dataclass
class DecodingResult:
    """Container for real-time decoding results."""
    
    # Recovered source signals for this chunk
    sources: np.ndarray  # (n_sources, chunk_samples)
    
    # Timing metrics
    latency_ms: float  # Processing time for this chunk
    timestamp_sec: float  # Timestamp of chunk start
    
    # Chunk info
    chunk_samples: int
    chunk_duration_ms: float


class OnlineDecoder:
    """
    Real-time decoder using pre-trained PCA/ICA transformations.
    
    The decoder is trained once on the full recording, extracting:
    1. StandardScaler mean (for centering)
    2. PCA components (for dimensionality reduction)
    3. ICA unmixing matrix (for source separation)
    4. Source ordering and sign corrections (for consistent output)
    
    These transformations are then applied to each incoming chunk
    in a stateless manner (no temporal dependencies between chunks).
    
    Parameters
    ----------
    scaler : StandardScaler
        Fitted scaler for centering input data.
    pca : PCA
        Fitted PCA for dimensionality reduction.
    ica : FastICA
        Fitted ICA for source separation.
    source_order : np.ndarray
        Permutation to reorder ICA outputs to match ground truth.
    sign_flips : np.ndarray
        Sign corrections for each source.
    sampling_rate_hz : float
        Sampling rate for timing calculations.
    """
    
    def __init__(
        self,
        scaler: StandardScaler,
        pca: PCA,
        ica: FastICA,
        source_order: np.ndarray,
        sign_flips: np.ndarray,
        sampling_rate_hz: float = 1000.0,
        pre_processor: PreProcessingHook | None = None,
    ) -> None:
        self.scaler = scaler
        self.pca = pca
        self.ica = ica
        self.source_order = source_order
        self.sign_flips = sign_flips
        self.sampling_rate_hz = sampling_rate_hz
        self.pre_processor = pre_processor

        # Precompute the combined sensor-to-source transformation
        # Full transform: sources = ((X - sensor_mean) @ PCA.T - ica_mean) @ ICA.T
        # We can precompute: combined = ICA.T @ PCA.T so sources = (X - means) @ combined
        # But ICA has its own mean subtraction, so we keep them separate

        # Cache ICA mean for efficiency
        self._ica_mean = self.ica.mean_

        # Statistics
        self._decode_count = 0
        self._total_latency_ms = 0.0
    
    @classmethod
    def from_training_data(
        cls,
        recording: np.ndarray,
        ground_truth: np.ndarray,
        variance_threshold: float = 0.999,
        n_sources: int = 3,
        random_state: int = 42,
    ) -> "OnlineDecoder":
        """
        Train an OnlineDecoder from full recording data.
        
        Parameters
        ----------
        recording : np.ndarray
            Full sensor recording, shape (n_sensors, n_samples).
        ground_truth : np.ndarray
            Ground truth sources for determining order/sign.
        variance_threshold : float
            PCA variance threshold.
        n_sources : int
            Number of sources to recover.
        random_state : int
            Random seed for ICA.
            
        Returns
        -------
        OnlineDecoder
            Trained decoder ready for real-time use.
        """
        config = load_config()
        sampling_rate_hz = config["temporal"]["sampling_rate_hz"]
        
        print("Training OnlineDecoder...")
        
        # Step 1: Fit StandardScaler (center only, no variance scaling)
        X = recording.T  # (n_samples, n_sensors)
        scaler = StandardScaler(with_std=False)
        X_centered = scaler.fit_transform(X)
        print(f"  Scaler: centering {recording.shape[0]} sensors")
        
        # Step 2: Fit PCA
        # First determine number of components needed
        pca_full = PCA(n_components=min(X_centered.shape))
        pca_full.fit(X_centered)
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.searchsorted(cumsum, variance_threshold) + 1
        n_components = max(n_sources, n_components)
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_centered)
        print(f"  PCA: {recording.shape[0]} -> {n_components} components ({cumsum[n_components-1]*100:.1f}% var)")
        
        # Step 3: Fit FastICA
        ica = FastICA(
            n_components=n_sources,
            algorithm="parallel",
            whiten="unit-variance",
            fun="logcosh",
            max_iter=1000,
            random_state=random_state,
            tol=1e-6,
        )
        sources = ica.fit_transform(X_pca)
        print(f"  ICA: {n_components} -> {n_sources} sources (converged in {ica.n_iter_} iter)")
        
        # Step 4: Determine source ordering and sign flips
        # Match recovered sources to ground truth
        _, _, source_order, sign_flips = match_sources(sources.T, ground_truth)
        print(f"  Matching: order={source_order}, signs={sign_flips}")
        
        return cls(
            scaler=scaler,
            pca=pca,
            ica=ica,
            source_order=source_order,
            sign_flips=sign_flips,
            sampling_rate_hz=sampling_rate_hz,
        )
    
    @classmethod
    def from_phase3_data(cls) -> "OnlineDecoder":
        """
        Create decoder from Phase 2/3 data files.

        Convenience method that loads the standard simulation data
        and trains the decoder.

        Returns
        -------
        OnlineDecoder
            Trained decoder.
        """
        recording, ground_truth, _ = load_phase2_data()
        return cls.from_training_data(recording, ground_truth)

    @classmethod
    def from_training_data_with_artifact_rejection(
        cls,
        recording: np.ndarray,
        ground_truth: np.ndarray,
        ppg_reference: np.ndarray,
        artifact_method: str = "rls",
        n_taps: int = 32,
        variance_threshold: float = 0.999,
        n_sources: int = 3,
        random_state: int = 42,
    ) -> "OnlineDecoder":
        """
        Train an OnlineDecoder with adaptive artifact rejection.

        This method first applies adaptive filtering (RLS or LMS) to remove
        cardiac artifacts from the training data, then trains the PCA/ICA
        transformations on the cleaned data.

        Parameters
        ----------
        recording : np.ndarray
            Full sensor recording, shape (n_sensors, n_samples).
        ground_truth : np.ndarray
            Ground truth sources for determining order/sign.
        ppg_reference : np.ndarray
            PPG or cardiac reference with shape (n_samples,).
        artifact_method : str, optional
            Adaptive filter method: 'rls' or 'lms'. Default is 'rls'.
        n_taps : int, optional
            Number of filter taps. Default is 32.
        variance_threshold : float
            PCA variance threshold.
        n_sources : int
            Number of sources to recover.
        random_state : int
            Random seed for ICA.

        Returns
        -------
        OnlineDecoder
            Trained decoder ready for real-time use.

        Notes
        -----
        The artifact rejection is applied ONLY during training to learn
        cleaner transformations. For online decoding, you should either:
        1. Apply artifact rejection to each chunk before calling decode()
        2. Use the OnlineDecoderWithArtifactRejection class (if implemented)
        """
        from subsense_bci.filtering.adaptive_filter import apply_adaptive_cancellation

        config = load_config()
        sampling_rate_hz = config["temporal"]["sampling_rate_hz"]

        print("Training OnlineDecoder with artifact rejection...")
        print(f"  Artifact method: {artifact_method.upper()}, Taps: {n_taps}")

        # Step 0: Apply adaptive artifact rejection
        cleaned_recording = apply_adaptive_cancellation(
            recording=recording,
            reference=ppg_reference,
            method=artifact_method,
            n_taps=n_taps,
            verbose=True,
        )
        print(f"  Artifact rejection applied to {recording.shape[0]} sensors")

        # Step 1: Fit StandardScaler on CLEANED data
        X = cleaned_recording.T  # (n_samples, n_sensors)
        scaler = StandardScaler(with_std=False)
        X_centered = scaler.fit_transform(X)
        print(f"  Scaler: centering {recording.shape[0]} sensors")

        # Step 2: Fit PCA on cleaned data
        pca_full = PCA(n_components=min(X_centered.shape))
        pca_full.fit(X_centered)
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.searchsorted(cumsum, variance_threshold) + 1
        n_components = max(n_sources, n_components)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_centered)
        print(f"  PCA: {recording.shape[0]} -> {n_components} components ({cumsum[n_components-1]*100:.1f}% var)")

        # Step 3: Fit FastICA on cleaned data
        ica = FastICA(
            n_components=n_sources,
            algorithm="parallel",
            whiten="unit-variance",
            fun="logcosh",
            max_iter=1000,
            random_state=random_state,
            tol=1e-6,
        )
        sources = ica.fit_transform(X_pca)
        print(f"  ICA: {n_components} -> {n_sources} sources (converged in {ica.n_iter_} iter)")

        # Step 4: Determine source ordering and sign flips
        _, _, source_order, sign_flips = match_sources(sources.T, ground_truth)
        print(f"  Matching: order={source_order}, signs={sign_flips}")

        return cls(
            scaler=scaler,
            pca=pca,
            ica=ica,
            source_order=source_order,
            sign_flips=sign_flips,
            sampling_rate_hz=sampling_rate_hz,
        )

    @classmethod
    def from_dynamic_simulation(
        cls,
        sensor_cloud: "SensorCloud",
        sources: np.ndarray,
        ground_truth: np.ndarray | None = None,
        variance_threshold: float = 0.999,
        n_sources: int = 3,
        random_state: int = 42,
        use_artifact_rejection: bool = True,
        artifact_method: str = "rls",
    ) -> "OnlineDecoder":
        """
        Train decoder from a dynamic simulation with SensorCloud.

        This is a convenience method that runs the dynamic simulation
        and trains the decoder in one call.

        Parameters
        ----------
        sensor_cloud : SensorCloud
            Sensor cloud with hemodynamic drift parameters.
        sources : np.ndarray
            Source coordinates, shape (n_sources, 3) in mm.
        ground_truth : np.ndarray, optional
            Ground truth waveforms. If None, generated internally.
        variance_threshold : float
            PCA variance threshold.
        n_sources : int
            Number of sources to recover.
        random_state : int
            Random seed.
        use_artifact_rejection : bool
            Whether to apply adaptive artifact rejection.
        artifact_method : str
            Artifact rejection method if use_artifact_rejection=True.

        Returns
        -------
        OnlineDecoder
            Trained decoder.
        """
        from subsense_bci.simulation.sensor_cloud import SensorCloud
        from subsense_bci.simulation.time_series import simulate_recording_dynamic

        print("Running dynamic simulation for decoder training...")

        # Run simulation
        time_vec, source_waveforms, clean_data, noisy_data, ppg_reference = simulate_recording_dynamic(
            sensor_cloud=sensor_cloud,
            sources=sources,
            verbose=True,
        )

        if ground_truth is None:
            ground_truth = source_waveforms

        # Train decoder
        if use_artifact_rejection:
            return cls.from_training_data_with_artifact_rejection(
                recording=noisy_data,
                ground_truth=ground_truth,
                ppg_reference=ppg_reference,
                artifact_method=artifact_method,
                variance_threshold=variance_threshold,
                n_sources=n_sources,
                random_state=random_state,
            )
        else:
            return cls.from_training_data(
                recording=noisy_data,
                ground_truth=ground_truth,
                variance_threshold=variance_threshold,
                n_sources=n_sources,
                random_state=random_state,
            )
    
    def decode(
        self,
        chunk: np.ndarray,
        timestamp: float = 0.0,
        reference: np.ndarray | None = None,
    ) -> DecodingResult:
        """
        Decode a single chunk of sensor data.

        Parameters
        ----------
        chunk : np.ndarray
            Sensor data chunk, shape (n_sensors, chunk_samples).
        timestamp : float
            Timestamp of chunk start in seconds.
        reference : np.ndarray, optional
            Reference signal for pre-processing (e.g., PPG/ECG).
            Shape (chunk_samples,) if provided. Only used if pre_processor
            is configured.

        Returns
        -------
        DecodingResult
            Decoded sources and timing metrics.
        """
        start_time = time.perf_counter()

        chunk_samples = chunk.shape[1]

        # Apply pre-processing if configured
        if self.pre_processor is not None:
            chunk = self.pre_processor.process(chunk, timestamp, reference)

        # Transpose for sklearn: (chunk_samples, n_sensors)
        X = chunk.T
        
        # Apply centering using fitted scaler
        X_centered = X - self.scaler.mean_
        
        # Apply PCA transformation
        X_pca = X_centered @ self.pca.components_.T
        
        # Apply ICA transformation
        # ICA transform: (X_pca - mean) @ components.T
        X_pca_centered = X_pca - self._ica_mean
        sources_raw = X_pca_centered @ self.ica.components_.T
        
        # Transpose back: (chunk_samples, n_sources) -> (n_sources, chunk_samples)
        sources_raw = sources_raw.T
        
        # Apply source reordering and sign corrections
        n_sources = len(self.source_order)
        sources = np.zeros((n_sources, chunk_samples))
        
        # Reorder to match ground truth indices
        for ica_idx in range(n_sources):
            gt_idx = self.source_order[ica_idx]
            sources[gt_idx] = sources_raw[ica_idx] * self.sign_flips[gt_idx]
        
        # Calculate timing
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        
        # Update statistics
        self._decode_count += 1
        self._total_latency_ms += latency_ms
        
        chunk_duration_ms = chunk_samples * 1000.0 / self.sampling_rate_hz
        
        return DecodingResult(
            sources=sources,
            latency_ms=latency_ms,
            timestamp_sec=timestamp,
            chunk_samples=chunk_samples,
            chunk_duration_ms=chunk_duration_ms,
        )
    
    @property
    def n_sources(self) -> int:
        """Number of sources recovered by decoder."""
        return self.ica.n_components
    
    @property
    def average_latency_ms(self) -> float:
        """Average decoding latency across all calls."""
        if self._decode_count == 0:
            return 0.0
        return self._total_latency_ms / self._decode_count
    
    def reset_statistics(self) -> None:
        """Reset latency statistics."""
        self._decode_count = 0
        self._total_latency_ms = 0.0
    
    def __repr__(self) -> str:
        return (
            f"OnlineDecoder("
            f"n_sources={self.n_sources}, "
            f"pca_components={self.pca.n_components_}, "
            f"avg_latency={self.average_latency_ms:.2f}ms)"
        )


def main() -> None:
    """Demo the OnlineDecoder with streaming data."""
    from subsense_bci.simulation.streamer import DataStreamer

    print("=" * 60)
    print("  PHASE 4: OnlineDecoder Demo")
    print("=" * 60)

    # Train decoder
    print("\n[1/2] Training decoder from Phase 3 data...")
    decoder = OnlineDecoder.from_phase3_data()
    print(f"\nDecoder ready: {decoder}")

    # Stream and decode
    print("\n[2/2] Streaming and decoding...")
    streamer = DataStreamer()

    chunk_size_ms = 100.0
    n_chunks = streamer.get_chunk_count(chunk_size_ms)

    latencies = []

    for i, packet in enumerate(streamer.get_chunks(chunk_size_ms)):
        result = decoder.decode(packet.chunk, packet.timestamp, packet.reference)
        latencies.append(result.latency_ms)

        if i < 5 or i == n_chunks - 1:
            print(
                f"  [{i+1:3d}/{n_chunks}] "
                f"t={packet.timestamp:.3f}s | "
                f"sources={result.sources.shape} | "
                f"latency={result.latency_ms:.2f}ms"
            )
        elif i == 5:
            print("  ...")

    # Summary
    print("\n" + "-" * 40)
    print("DECODING SUMMARY")
    print("-" * 40)
    print(f"  Chunks processed: {n_chunks}")
    print(f"  Chunk size: {chunk_size_ms:.0f}ms")
    print(f"  Average latency: {np.mean(latencies):.2f}ms")
    print(f"  Max latency: {np.max(latencies):.2f}ms")
    print(f"  Min latency: {np.min(latencies):.2f}ms")

    realtime_ratio = chunk_size_ms / np.mean(latencies)
    print(f"  Real-time factor: {realtime_ratio:.1f}x (>{1.0:.1f}x required)")

    if realtime_ratio > 1.0:
        print(f"\n  [OK] REAL-TIME CAPABLE")
    else:
        print(f"\n  [!] TOO SLOW FOR REAL-TIME (consider reducing PCA components)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

