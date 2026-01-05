"""
Physics Module Unit Tests

Validates mathematical correctness of transfer functions,
coordinate transformations, and physical models.
"""

from __future__ import annotations

import numpy as np
import pytest

from subsense_bci.physics.constants import (
    BRAIN_CONDUCTIVITY_S_M,
    VACUUM_PERMEABILITY_H_M,
    ME_RESONANT_FREQ_KHZ,
    SINGULARITY_THRESHOLD_MM,
)
from subsense_bci.physics.transfer_function import (
    compute_lead_field,
    compute_distance_matrix,
    validate_lead_field,
)


class TestPhysicsConstants:
    """Test that physics constants are correctly defined."""

    def test_conductivity_ranges(self) -> None:
        """Verify tissue conductivities are in physiologically valid ranges."""
        # Brain conductivity should be between 0.1 and 0.5 S/m
        assert 0.1 <= BRAIN_CONDUCTIVITY_S_M <= 0.5

    def test_permeability_positive(self) -> None:
        """Vacuum permeability must be positive."""
        assert VACUUM_PERMEABILITY_H_M > 0

    def test_me_resonant_frequency(self) -> None:
        """ME film resonant frequency should be in expected range."""
        # ActiveEcho operates around 340 kHz
        assert 300.0 <= ME_RESONANT_FREQ_KHZ <= 400.0

    def test_singularity_threshold_positive(self) -> None:
        """Singularity threshold must be positive and small."""
        assert 0 < SINGULARITY_THRESHOLD_MM < 1.0


class TestLeadFieldComputation:
    """Test the lead field computation for volume conductor model."""

    def test_lead_field_shape(self) -> None:
        """Lead field should have shape (n_sensors, n_sources)."""
        sensors = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
        sources = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])

        lead_field, mask = compute_lead_field(sensors, sources)

        assert lead_field.shape == (3, 2)
        assert mask.shape == (3, 2)

    def test_lead_field_1_over_r_decay(self) -> None:
        """
        Lead field should follow 1/r decay law.

        Physics: V = I / (4 * pi * sigma * r)

        For unit current at distance r (in meters):
            V = 1 / (4 * pi * 0.33 * r_m)
        """
        # Single source at origin
        sources = np.array([[0.0, 0.0, 0.0]])

        # Sensors at known distances along x-axis (in mm)
        distances_mm = np.array([0.1, 0.2, 0.5, 1.0])
        sensors = np.column_stack([distances_mm, np.zeros(4), np.zeros(4)])

        lead_field, _ = compute_lead_field(sensors, sources)

        # Convert distances to meters for expected calculation
        distances_m = distances_mm * 1e-3

        # Expected values from analytical formula
        expected = 1.0 / (4.0 * np.pi * BRAIN_CONDUCTIVITY_S_M * distances_m)

        # Check relative error is negligible
        relative_error = np.abs(lead_field.flatten() - expected) / expected
        assert np.all(relative_error < 1e-10), f"Max error: {np.max(relative_error)}"

    def test_singularity_clamping(self) -> None:
        """Sensors very close to sources should have clamped distances."""
        # Source at origin
        sources = np.array([[0.0, 0.0, 0.0]])

        # Sensor inside singularity threshold
        sensors = np.array([[0.01, 0.0, 0.0]])  # 0.01 mm < 0.05 mm threshold

        lead_field, mask = compute_lead_field(sensors, sources)

        # Should be marked as in singularity zone
        assert mask[0, 0] == True

        # Value should be clamped (finite, not infinite)
        assert np.isfinite(lead_field[0, 0])

        # Value should equal 1/(4*pi*sigma*r_min)
        r_min_m = SINGULARITY_THRESHOLD_MM * 1e-3
        expected_max = 1.0 / (4.0 * np.pi * BRAIN_CONDUCTIVITY_S_M * r_min_m)
        assert np.isclose(lead_field[0, 0], expected_max)

    def test_no_infinities_or_nans(self) -> None:
        """Lead field should never contain infinities or NaNs."""
        # Generate random sensors including some very close to sources
        np.random.seed(42)
        sensors = np.random.uniform(-0.5, 0.5, (100, 3))
        sources = np.array([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]])

        lead_field, _ = compute_lead_field(sensors, sources)

        assert not np.any(np.isinf(lead_field))
        assert not np.any(np.isnan(lead_field))

    def test_symmetry_equal_distances(self) -> None:
        """Sensors equidistant from a source should have equal lead field values."""
        sources = np.array([[0.0, 0.0, 0.0]])

        # 4 sensors at same distance but different directions
        d = 0.5  # mm
        sensors = np.array([
            [d, 0, 0],
            [-d, 0, 0],
            [0, d, 0],
            [0, 0, d],
        ])

        lead_field, _ = compute_lead_field(sensors, sources)

        # All should be equal
        assert np.allclose(lead_field, lead_field[0, 0])


class TestDistanceMatrix:
    """Test distance matrix computation."""

    def test_distance_matrix_shape(self) -> None:
        """Distance matrix should have shape (n_sensors, n_sources)."""
        sensors = np.random.randn(10, 3)
        sources = np.random.randn(3, 3)

        distances = compute_distance_matrix(sensors, sources)

        assert distances.shape == (10, 3)

    def test_distance_matrix_values(self) -> None:
        """Distance matrix should contain correct Euclidean distances."""
        sensors = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        sources = np.array([[0.0, 0.0, 0.0]])

        distances = compute_distance_matrix(sensors, sources)

        # First sensor at origin -> distance = 0
        assert np.isclose(distances[0, 0], 0.0)

        # Second sensor at (3, 4, 0) -> distance = 5 (3-4-5 triangle)
        assert np.isclose(distances[1, 0], 5.0)

    def test_distance_matrix_symmetric_sources(self) -> None:
        """Distance from sensor to source equals distance from source to sensor."""
        sensors = np.array([[1.0, 2.0, 3.0]])
        sources = np.array([[4.0, 5.0, 6.0]])

        d1 = compute_distance_matrix(sensors, sources)
        d2 = compute_distance_matrix(sources, sensors)

        assert np.isclose(d1[0, 0], d2[0, 0])


class TestLeadFieldValidation:
    """Test the lead field validation utility."""

    def test_valid_lead_field(self) -> None:
        """A normal lead field should pass validation."""
        lead_field = np.random.rand(100, 3) * 1000

        info = validate_lead_field(lead_field, expected_shape=(100, 3))

        assert info["is_valid"]
        assert info["shape"] == (100, 3)
        assert not info["has_infinities"]
        assert not info["has_nans"]

    def test_invalid_shape(self) -> None:
        """Wrong shape should fail validation."""
        lead_field = np.random.rand(100, 3)

        info = validate_lead_field(lead_field, expected_shape=(50, 3))

        assert not info["is_valid"]
        assert len(info["errors"]) > 0

    def test_infinity_detection(self) -> None:
        """Infinities should be detected."""
        lead_field = np.array([[1.0, np.inf], [2.0, 3.0]])

        info = validate_lead_field(lead_field)

        assert not info["is_valid"]
        assert info["has_infinities"]

    def test_nan_detection(self) -> None:
        """NaNs should be detected."""
        lead_field = np.array([[1.0, np.nan], [2.0, 3.0]])

        info = validate_lead_field(lead_field)

        assert not info["is_valid"]
        assert info["has_nans"]


class TestCoordinateTransforms:
    """Placeholder tests for coordinate transformation functions."""

    def test_identity_transform(self) -> None:
        """Identity transform should preserve coordinates."""
        coords = np.array([[1.0, 2.0, 3.0]])
        identity = np.eye(3)
        transformed = coords @ identity.T
        np.testing.assert_array_almost_equal(coords, transformed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
