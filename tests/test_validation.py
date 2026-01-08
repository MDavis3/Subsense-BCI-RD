"""
Tests for Subsense R&D Signal Bench Validation Module

Tests input validation, Nyquist checks, and real-time budget enforcement.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from subsense_bci.validation import (
    NyquistResult,
    BudgetResult,
    ConfigValidationResult,
    validate_nyquist,
    validate_realtime_budget,
    validate_config_file,
)
from subsense_bci.validation.input_validators import (
    estimate_processing_time_ms,
    validate_all,
)


# =============================================================================
# Nyquist Validation Tests
# =============================================================================


class TestNyquistValidation:
    """Tests for Nyquist theorem validation."""

    def test_valid_sampling_rate(self):
        """Test that valid sampling rate passes validation."""
        result = validate_nyquist(
            sampling_rate_hz=1000.0,
            source_frequencies={"alpha": 10.0, "beta": 20.0},
        )

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.max_frequency_hz == 20.0
        assert result.nyquist_frequency_hz == 500.0
        assert result.oversampling_factor == 25.0

    def test_nyquist_violation(self):
        """Test that Nyquist violation is detected."""
        result = validate_nyquist(
            sampling_rate_hz=30.0,  # Too low! max_freq=20 -> need >40
            source_frequencies={"alpha": 10.0, "beta": 20.0},
        )

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "NYQUIST VIOLATION" in result.errors[0]
        assert len(result.recovery_suggestions) > 0

    def test_edge_case_exact_nyquist(self):
        """Test edge case at exactly Nyquist limit."""
        # Exactly 2x should still fail (need > 2x, not >=)
        result = validate_nyquist(
            sampling_rate_hz=40.0,
            source_frequencies={"beta": 20.0},
        )

        # At exactly 2x, it should fail (need strictly greater)
        assert result.is_valid is False

    def test_marginal_oversampling_warning(self):
        """Test warning for marginal oversampling."""
        result = validate_nyquist(
            sampling_rate_hz=50.0,  # Only 1.25x oversampling (below 2.5x threshold)
            source_frequencies={"beta": 20.0},
            min_oversampling_factor=2.5,
        )

        assert result.is_valid is True  # Not a hard failure
        assert len(result.warnings) > 0
        assert "MARGINAL OVERSAMPLING" in result.warnings[0]

    def test_default_frequencies(self):
        """Test with default source frequencies."""
        result = validate_nyquist(sampling_rate_hz=1000.0)

        assert result.is_valid is True
        assert result.max_frequency_hz == 20.0  # Default beta

    def test_list_frequencies(self):
        """Test with list of frequencies instead of dict."""
        result = validate_nyquist(
            sampling_rate_hz=1000.0,
            source_frequencies=[10.0, 20.0, 30.0],
        )

        assert result.is_valid is True
        assert result.max_frequency_hz == 30.0


# =============================================================================
# Budget Validation Tests
# =============================================================================


class TestBudgetValidation:
    """Tests for real-time budget validation."""

    def test_rls_within_budget(self):
        """Test RLS with low sensor count is within budget."""
        result = validate_realtime_budget(
            filter_type="rls",
            n_sensors=1000,
            n_taps=8,
        )

        assert result.is_within_budget is True
        assert len(result.errors) == 0
        assert result.budget_utilization < 1.0

    def test_rls_exceeds_budget(self):
        """Test RLS with high sensor count and many taps exceeds budget."""
        result = validate_realtime_budget(
            filter_type="rls",
            n_sensors=10000,
            n_taps=64,  # Very high tap count to ensure budget exceeded
        )

        assert result.is_within_budget is False
        assert len(result.errors) > 0
        assert "BUDGET EXCEEDED" in result.errors[0]
        assert len(result.recovery_suggestions) > 0

    def test_lms_always_faster(self):
        """Test LMS is faster than RLS for same config."""
        rls_result = validate_realtime_budget("rls", 5000, 16)
        lms_result = validate_realtime_budget("lms", 5000, 16)

        assert lms_result.estimated_latency_ms < rls_result.estimated_latency_ms

    def test_phase_aware_rls(self):
        """Test PhaseAwareRLS filter type."""
        result = validate_realtime_budget(
            filter_type="PhaseAwareRLS",
            n_sensors=1000,
            n_taps=8,
        )

        assert result.filter_type == "PhaseAwareRLS"
        assert result.is_within_budget is True

    def test_approaching_budget_warning(self):
        """Test warning when approaching budget limit."""
        result = validate_realtime_budget(
            filter_type="rls",
            n_sensors=8000,
            n_taps=8,
            warning_threshold=0.5,  # Warn at 50% utilization
        )

        # Should be within budget but trigger warning
        if result.budget_utilization > 0.5:
            assert len(result.warnings) > 0


class TestProcessingTimeEstimation:
    """Tests for processing time estimation helper."""

    def test_rls_complexity_quadratic(self):
        """Test RLS has quadratic complexity in n_taps."""
        time_8_taps = estimate_processing_time_ms(1000, 8, 100, "rls")
        time_16_taps = estimate_processing_time_ms(1000, 16, 100, "rls")

        # Should be roughly 4x (16^2 / 8^2 = 4)
        ratio = time_16_taps / time_8_taps
        assert 3.5 < ratio < 4.5

    def test_lms_complexity_linear(self):
        """Test LMS has linear complexity in n_taps."""
        time_8_taps = estimate_processing_time_ms(1000, 8, 100, "lms")
        time_16_taps = estimate_processing_time_ms(1000, 16, 100, "lms")

        # Should be roughly 2x (16 / 8 = 2)
        ratio = time_16_taps / time_8_taps
        assert 1.8 < ratio < 2.2


# =============================================================================
# Config Validation Tests
# =============================================================================


class TestConfigValidation:
    """Tests for configuration file validation."""

    def test_valid_default_config(self):
        """Test loading valid default config."""
        result = validate_config_file()

        assert result.is_valid is True
        assert result.config is not None
        assert "temporal" in result.config
        assert "cloud" in result.config

    def test_nonexistent_file_fallback(self):
        """Test fallback to defaults for nonexistent file."""
        result = validate_config_file("/nonexistent/path/config.yaml")

        assert result.is_valid is True  # Falls back gracefully
        assert result.config is not None
        assert len(result.warnings) > 0
        assert "not found" in result.warnings[0].lower()

    def test_malformed_yaml(self):
        """Test handling of malformed YAML."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("invalid: yaml: content: [unclosed")
            temp_path = f.name

        try:
            result = validate_config_file(temp_path)

            assert result.config is not None  # Falls back to defaults
            assert len(result.errors) > 0
            assert "YAML" in result.errors[0].upper()
        finally:
            Path(temp_path).unlink()

    def test_empty_yaml_file(self):
        """Test handling of empty YAML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")  # Empty file
            temp_path = f.name

        try:
            result = validate_config_file(temp_path)

            assert result.config is not None  # Falls back to defaults
            assert len(result.warnings) > 0
        finally:
            Path(temp_path).unlink()

    def test_valid_custom_config(self):
        """Test loading valid custom config."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("""
cloud:
  sensor_count: 5000
  volume_side_mm: 2.0
temporal:
  sampling_rate_hz: 2000.0
  duration_sec: 1.0
  snr_level: 10.0
physics:
  brain_conductivity_s_m: 0.33
""")
            temp_path = f.name

        try:
            result = validate_config_file(temp_path)

            assert result.is_valid is True
            assert result.config["cloud"]["sensor_count"] == 5000
            assert result.config["temporal"]["sampling_rate_hz"] == 2000.0
        finally:
            Path(temp_path).unlink()


# =============================================================================
# Combined Validation Tests
# =============================================================================


class TestCombinedValidation:
    """Tests for validate_all convenience function."""

    def test_all_valid(self):
        """Test all validators pass with good config."""
        result = validate_all(
            sampling_rate_hz=1000.0,
            source_frequencies={"alpha": 10.0, "beta": 20.0},
            filter_type="PhaseAwareRLS",
            n_sensors=1000,
            n_taps=8,
        )

        assert result["all_valid"] is True
        assert result["nyquist"].is_valid is True
        assert result["budget"].is_within_budget is True

    def test_nyquist_failure_propagates(self):
        """Test Nyquist failure makes all_valid False."""
        result = validate_all(
            sampling_rate_hz=30.0,  # Nyquist violation
            source_frequencies={"beta": 20.0},
            filter_type="lms",
            n_sensors=100,
            n_taps=4,
        )

        assert result["all_valid"] is False
        assert result["nyquist"].is_valid is False

    def test_budget_failure_propagates(self):
        """Test budget failure makes all_valid False."""
        result = validate_all(
            sampling_rate_hz=1000.0,
            filter_type="rls",
            n_sensors=10000,
            n_taps=64,  # Way too many taps
        )

        assert result["all_valid"] is False
        assert result["budget"].is_within_budget is False


# =============================================================================
# Preset Tests
# =============================================================================


class TestPresets:
    """Tests for demo presets module."""

    def test_standard_cardiac_preset_exists(self):
        """Test Standard Cardiac preset is available."""
        from subsense_bci.presets import STANDARD_CARDIAC, get_preset

        assert STANDARD_CARDIAC is not None
        assert STANDARD_CARDIAC["name"] == "Standard Cardiac Interference"

        # Should also be retrievable by name
        preset = get_preset("standard_cardiac")
        assert preset["n_sensors"] == 1000
        assert preset["filter_type"] == "PhaseAwareRLS"

    def test_all_presets_have_required_keys(self):
        """Test all presets have required parameters."""
        from subsense_bci.presets import PRESETS

        required_keys = [
            "name",
            "n_sensors",
            "filter_type",
            "n_taps",
            "pulse_wave_velocity_m_s",
            "drift_amplitude_mm",
            "cardiac_freq_hz",
        ]

        for preset_name, preset in PRESETS.items():
            for key in required_keys:
                assert key in preset, f"Preset '{preset_name}' missing key '{key}'"

    def test_preset_name_normalization(self):
        """Test preset lookup handles various name formats."""
        from subsense_bci.presets import get_preset

        # All these should work
        preset1 = get_preset("standard_cardiac")
        preset2 = get_preset("Standard Cardiac")
        preset3 = get_preset("STANDARD-CARDIAC")

        assert preset1["name"] == preset2["name"] == preset3["name"]

    def test_invalid_preset_raises(self):
        """Test invalid preset name raises KeyError."""
        from subsense_bci.presets import get_preset

        with pytest.raises(KeyError):
            get_preset("nonexistent_preset")

    def test_list_presets(self):
        """Test listing available presets."""
        from subsense_bci.presets import list_presets

        presets = list_presets()

        assert len(presets) >= 4  # At least our 4 main presets
        assert "standard_cardiac" in presets
        assert "exercise_stress" in presets


# =============================================================================
# Nanoparticle Drift Placeholder Tests
# =============================================================================


class TestNanoparticleDriftPlaceholder:
    """Tests for nanoparticle drift modeling placeholder."""

    def test_enable_nanoparticle_drift(self):
        """Test enabling nanoparticle drift placeholder."""
        from subsense_bci.simulation.sensor_cloud import SensorCloud

        cloud = SensorCloud.from_uniform_cloud(n_sensors=100)

        # Initially disabled
        assert cloud.nanoparticle_drift_enabled is False

        # Enable it
        cloud.enable_nanoparticle_drift(enabled=True)
        assert cloud.nanoparticle_drift_enabled is True

        # Disable it
        cloud.enable_nanoparticle_drift(enabled=False)
        assert cloud.nanoparticle_drift_enabled is False

    def test_diffusion_coefficient(self):
        """Test diffusion coefficient property."""
        from subsense_bci.simulation.sensor_cloud import SensorCloud

        cloud = SensorCloud.from_uniform_cloud(n_sensors=100)

        # Default value
        assert cloud.diffusion_coefficient_mm2_s == pytest.approx(4.4e-6)

        # Custom value
        cloud.enable_nanoparticle_drift(diffusion_coefficient_mm2_s=1e-5)
        assert cloud.diffusion_coefficient_mm2_s == pytest.approx(1e-5)
