"""
Unmixing Pipeline Unit Tests

Validates the PCA/ICA source separation pipeline, ensuring that
independent sources can be recovered from mixed sensor recordings
with high fidelity.
"""

from __future__ import annotations

import numpy as np
import pytest

from subsense_bci.filtering.unmixing import (
    UnmixingResult,
    pca_denoise,
    run_fastica,
    match_sources,
    unmix_sources,
)
from subsense_bci.physics.transfer_function import compute_lead_field


class TestPCADenoise:
    """Test PCA dimensionality reduction."""

    def test_pca_preserves_variance(self) -> None:
        """PCA should preserve at least the target variance threshold."""
        np.random.seed(42)
        
        # Create data with clear low-rank structure (3 sources, 100 sensors)
        n_sensors = 100
        n_samples = 500
        n_sources = 3
        
        # Generate independent sources
        sources = np.random.randn(n_sources, n_samples)
        
        # Create mixing matrix
        mixing = np.random.randn(n_sensors, n_sources)
        
        # Mixed data
        data = mixing @ sources
        
        # Run PCA with 99.9% variance threshold
        reduced, pca = pca_denoise(data, variance_threshold=0.999)
        
        # Variance explained should be >= 99.9%
        assert pca.explained_variance_ratio_.sum() >= 0.999
        
        # Should have at least n_sources components
        assert reduced.shape[0] >= n_sources

    def test_pca_reduces_dimensionality(self) -> None:
        """PCA should reduce high-dimensional data to fewer components."""
        np.random.seed(42)
        
        # High-dimensional data with low-rank structure
        n_sensors = 1000
        n_samples = 500
        n_sources = 5
        
        sources = np.random.randn(n_sources, n_samples)
        mixing = np.random.randn(n_sensors, n_sources)
        data = mixing @ sources + 0.01 * np.random.randn(n_sensors, n_samples)
        
        reduced, pca = pca_denoise(data, variance_threshold=0.99)
        
        # Should have far fewer components than original sensors
        assert reduced.shape[0] < n_sensors // 10


class TestFastICA:
    """Test FastICA source separation."""

    def test_fastica_extracts_correct_number_of_sources(self) -> None:
        """FastICA should return the requested number of components."""
        np.random.seed(42)
        
        n_components = 10
        n_samples = 500
        
        data = np.random.randn(n_components, n_samples)
        
        sources, ica = run_fastica(data, n_components=3, random_state=42)
        
        assert sources.shape == (3, n_samples)

    def test_fastica_converges(self) -> None:
        """FastICA should converge within max iterations."""
        np.random.seed(42)
        
        # Create clearly separable sources
        t = np.linspace(0, 1, 1000)
        s1 = np.sin(2 * np.pi * 5 * t)
        s2 = np.sign(np.sin(2 * np.pi * 11 * t))  # Square wave
        s3 = np.random.randn(1000)
        
        sources = np.vstack([s1, s2, s3])
        mixing = np.random.randn(10, 3)
        data = mixing @ sources
        
        recovered, ica = run_fastica(data, n_components=3, random_state=42)
        
        # Should converge (n_iter_ < max_iter)
        assert ica.n_iter_ < 1000


class TestSourceMatching:
    """Test Hungarian algorithm source matching."""

    def test_perfect_match_gives_high_correlation(self) -> None:
        """Identical sources should have correlation ~1.0."""
        np.random.seed(42)
        
        # Create sources
        sources = np.random.randn(3, 500)
        
        # Match with self (with shuffled order)
        shuffled = sources[[2, 0, 1], :]  # Permute rows
        
        matched, corr_matrix, order, signs = match_sources(shuffled, sources)
        
        # After matching, correlation should be very high
        for i in range(3):
            corr = np.corrcoef(matched[i], sources[i])[0, 1]
            assert abs(corr) > 0.99

    def test_sign_flip_correction(self) -> None:
        """Matching should correct for sign flips."""
        np.random.seed(42)
        
        sources = np.random.randn(3, 500)
        
        # Flip signs
        flipped = sources.copy()
        flipped[0] *= -1
        flipped[2] *= -1
        
        matched, corr_matrix, order, signs = match_sources(flipped, sources)
        
        # After matching, signs should be corrected
        for i in range(3):
            corr = np.corrcoef(matched[i], sources[i])[0, 1]
            assert corr > 0.99  # Positive correlation after correction


class TestUnmixingPipeline:
    """Integration tests for the full unmixing pipeline."""

    def test_unmix_synthetic_data_high_snr(self) -> None:
        """
        Source recovery correlation should exceed 0.90 on clean synthetic data.
        
        This is the critical validation test: if ICA cannot recover sources
        from controlled synthetic data, it won't work on real recordings.
        """
        np.random.seed(42)
        
        # Create synthetic recording with known sources
        n_sensors = 500
        n_samples = 2000
        n_sources = 3
        
        # Generate time vector
        t = np.linspace(0, 2, n_samples)
        
        # Create clearly distinguishable sources
        source_a = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine
        source_b = np.sin(2 * np.pi * 25 * t)  # 25 Hz sine (different freq)
        source_c = np.random.randn(n_samples)  # Noise
        source_c = (source_c - source_c.mean()) / source_c.std()  # Standardize
        
        ground_truth = np.vstack([source_a, source_b, source_c])
        
        # Standardize all sources
        for i in range(n_sources):
            ground_truth[i] = (ground_truth[i] - ground_truth[i].mean()) / ground_truth[i].std()
        
        # Create random mixing matrix (simulates lead field)
        np.random.seed(123)
        mixing = np.random.randn(n_sensors, n_sources)
        
        # Mix sources
        clean_recording = mixing @ ground_truth
        
        # Add small noise (high SNR = 20)
        noise_std = np.std(clean_recording) / np.sqrt(20)
        recording = clean_recording + noise_std * np.random.randn(n_sensors, n_samples)
        
        # Run unmixing pipeline
        result = unmix_sources(
            recording,
            ground_truth,
            variance_threshold=0.999,
            n_sources=3,
            random_state=42,
        )
        
        # All correlations should exceed 0.90
        assert isinstance(result, UnmixingResult)
        for i, corr in enumerate(result.matched_correlations):
            assert corr > 0.90, f"Source {i} correlation {corr:.3f} < 0.90"

    def test_unmix_with_realistic_lead_field(self) -> None:
        """
        Test unmixing with a physics-based lead field matrix.
        
        This validates the pipeline against the actual Subsense forward model.
        """
        np.random.seed(42)
        
        # Generate sensor cloud
        n_sensors = 200
        sensors = np.random.uniform(-0.5, 0.5, (n_sensors, 3))
        
        # Fixed source locations
        sources_pos = np.array([
            [0.2, 0.0, 0.0],
            [-0.2, 0.2, 0.0],
            [0.0, -0.2, 0.1],
        ])
        
        # Compute lead field using actual physics
        lead_field, _ = compute_lead_field(sensors, sources_pos)
        
        # Generate source waveforms
        n_samples = 1000
        t = np.linspace(0, 1, n_samples)
        
        source_a = np.sin(2 * np.pi * 10 * t)
        source_b = np.sin(2 * np.pi * 23 * t)
        source_c = np.random.randn(n_samples)
        
        ground_truth = np.vstack([source_a, source_b, source_c])
        
        # Standardize
        for i in range(3):
            ground_truth[i] = (ground_truth[i] - ground_truth[i].mean()) / ground_truth[i].std()
        
        # Forward model: X = L @ S + N
        clean_recording = lead_field @ ground_truth
        noise_std = np.std(clean_recording) / np.sqrt(10)  # SNR = 10
        recording = clean_recording + noise_std * np.random.randn(n_sensors, n_samples)
        
        # Run unmixing
        result = unmix_sources(
            recording,
            ground_truth,
            variance_threshold=0.99,
            n_sources=3,
            random_state=42,
        )
        
        # Should achieve reasonable recovery (>0.85 for physics-based mixing)
        avg_corr = np.mean(result.matched_correlations)
        assert avg_corr > 0.85, f"Average correlation {avg_corr:.3f} < 0.85"

    def test_unmixing_result_structure(self) -> None:
        """UnmixingResult should have all expected fields populated."""
        np.random.seed(42)
        
        # Minimal synthetic data
        n_sensors = 50
        n_samples = 500
        
        sources = np.random.randn(3, n_samples)
        mixing = np.random.randn(n_sensors, 3)
        recording = mixing @ sources
        
        result = unmix_sources(recording, sources, n_sources=3, random_state=42)
        
        # Check all fields exist and have correct shapes
        assert result.recovered_sources.shape == (3, n_samples)
        assert result.matched_sources.shape == (3, n_samples)
        assert result.correlation_matrix.shape == (3, 3)
        assert result.matched_correlations.shape == (3,)
        assert result.source_order.shape == (3,)
        assert result.sign_flips.shape == (3,)
        assert isinstance(result.n_pca_components, int)
        assert 0 < result.variance_explained <= 1.0
        assert isinstance(result.ica_n_iter, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

