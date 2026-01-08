"""
Subsense R&D Signal Bench - Interactive Streamlit Dashboard

A professional tool for exploring cardiac artifact rejection in BCI systems.
Designed for non-software researchers (biologists, PIs) with:
- Interactive parameter controls
- Real-time signal visualization
- Automatic validation and budget warnings

Run directly:
    streamlit run src/subsense_bci/dashboard/streamlit_app.py

Or via demo script:
    python run_demo.py
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subsense_bci.config import load_config
from subsense_bci.presets import get_preset, get_preset_names_and_descriptions, STANDARD_CARDIAC
from subsense_bci.validation import (
    validate_nyquist,
    validate_realtime_budget,
    validate_config_file,
)
from subsense_bci.physics.constants import (
    REALTIME_LATENCY_BUDGET_MS,
    SAMPLING_RATE_HZ,
)


# =============================================================================
# Theme Colors (matching visualization/theme.py)
# =============================================================================

COLORS = {
    "background": "#0f0f0f",
    "panel_bg": "#12121a",
    "grid_line": "#1a1a2e",
    "nanotech_cyan": "#00FFFF",
    "success_green": "#00FF88",
    "warning_red": "#FF3333",
    "safety_yellow": "#FFD700",
    "text_primary": "#E0E0E0",
    "text_secondary": "#808080",
    "source_a": "#FF6B6B",  # Alpha
    "source_b": "#4ECDC4",  # Beta
    "source_c": "#95E1D3",  # Pink noise
}


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="SubSense R&D Signal Bench",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme
st.markdown(f"""
<style>
    .stApp {{
        background-color: {COLORS['background']};
    }}
    .stSidebar {{
        background-color: {COLORS['panel_bg']};
    }}
    .metric-card {{
        background-color: {COLORS['panel_bg']};
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid {COLORS['grid_line']};
    }}
    .success {{
        color: {COLORS['success_green']};
    }}
    .warning {{
        color: {COLORS['warning_red']};
    }}
    .info {{
        color: {COLORS['nanotech_cyan']};
    }}
    h1, h2, h3 {{
        color: {COLORS['nanotech_cyan']} !important;
    }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Signal Generation Functions
# =============================================================================

def generate_source_signals(
    duration_sec: float,
    sampling_rate_hz: float,
    source_frequencies: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Generate clean source signals (alpha, beta, pink noise)."""
    n_samples = int(duration_sec * sampling_rate_hz)
    t = np.linspace(0, duration_sec, n_samples)

    # Generate sources
    alpha = np.sin(2 * np.pi * source_frequencies.get("alpha", 10.0) * t)
    beta = np.sin(2 * np.pi * source_frequencies.get("beta", 20.0) * t)

    # Pink noise (1/f spectrum)
    np.random.seed(42)
    white = np.random.randn(n_samples)
    # Simple 1/f filter via cumsum
    pink = np.cumsum(white)
    pink = pink - np.mean(pink)
    pink = pink / np.std(pink)

    sources = {
        "alpha": alpha,
        "beta": beta,
        "pink": pink,
    }

    return t, np.column_stack([alpha, beta, pink]), sources


def generate_cardiac_artifact(
    t: np.ndarray,
    cardiac_freq_hz: float,
    drift_amplitude_mm: float,
    n_sensors: int,
    pwv_m_s: float,
) -> np.ndarray:
    """Generate cardiac artifact signal based on hemodynamic drift."""
    n_samples = len(t)

    # Cardiac waveform (simplified realistic shape)
    cardiac_phase = 2 * np.pi * cardiac_freq_hz * t
    # Systolic rise + dicrotic notch approximation
    cardiac = (
        0.6 * np.sin(cardiac_phase)
        + 0.3 * np.sin(2 * cardiac_phase)
        + 0.1 * np.sin(3 * cardiac_phase)
    )

    # Scale by drift amplitude (artifact ~ gradient * displacement)
    # Simplified: artifact proportional to displacement
    artifact = cardiac * drift_amplitude_mm * 10.0  # Arbitrary scale factor

    return artifact


def simulate_filtering(
    raw_signal: np.ndarray,
    artifact: np.ndarray,
    filter_type: str,
    n_taps: int,
    lambda_: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate adaptive filtering to remove artifact.

    This is a simplified simulation for the dashboard.
    Real filtering uses the full AdaptiveFilterHook.
    """
    n_samples = len(raw_signal)

    # Simulated filter performance based on type and parameters
    if filter_type.lower() == "lms":
        # LMS: slower convergence
        convergence_samples = int(n_taps / mu * 10)
        rejection_factor = 0.7
    elif filter_type.lower() == "rls":
        # RLS: faster convergence
        convergence_samples = int(n_taps * (1 - lambda_) * 100)
        rejection_factor = 0.85
    else:  # PhaseAwareRLS
        # PhaseAwareRLS: best for cardiac
        convergence_samples = int(n_taps * (1 - lambda_) * 50)
        rejection_factor = 0.95

    # Simulate convergence curve
    convergence_curve = 1 - np.exp(-np.arange(n_samples) / max(convergence_samples, 1))

    # Apply simulated artifact rejection
    rejected_artifact = artifact * (1 - convergence_curve * rejection_factor)
    cleaned = raw_signal - artifact + rejected_artifact
    residual = artifact - rejected_artifact

    return cleaned, residual


def compute_metrics(
    clean_source: np.ndarray,
    noisy_signal: np.ndarray,
    cleaned_signal: np.ndarray,
    artifact: np.ndarray,
    residual: np.ndarray,
) -> dict:
    """Compute signal quality metrics."""
    # MSE
    mse_before = np.mean((noisy_signal - clean_source) ** 2)
    mse_after = np.mean((cleaned_signal - clean_source) ** 2)

    # SNR (in dB)
    signal_power = np.mean(clean_source ** 2)
    noise_before = np.mean((noisy_signal - clean_source) ** 2)
    noise_after = np.mean((cleaned_signal - clean_source) ** 2)

    snr_before = 10 * np.log10(signal_power / noise_before) if noise_before > 0 else float("inf")
    snr_after = 10 * np.log10(signal_power / noise_after) if noise_after > 0 else float("inf")

    # Artifact rejection ratio (dB)
    artifact_power = np.mean(artifact ** 2)
    residual_power = np.mean(residual ** 2)
    rejection_db = 10 * np.log10(artifact_power / residual_power) if residual_power > 0 else float("inf")

    # Correlation with clean source
    correlation = np.corrcoef(clean_source, cleaned_signal)[0, 1]

    return {
        "mse_before": mse_before,
        "mse_after": mse_after,
        "mse_improvement": (mse_before - mse_after) / mse_before * 100 if mse_before > 0 else 0,
        "snr_before_db": snr_before,
        "snr_after_db": snr_after,
        "snr_improvement_db": snr_after - snr_before,
        "rejection_db": rejection_db,
        "correlation": correlation,
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def create_signal_comparison_plot(
    t: np.ndarray,
    raw_signal: np.ndarray,
    cleaned_signal: np.ndarray,
    residual: np.ndarray,
    clean_source: np.ndarray,
) -> go.Figure:
    """Create side-by-side signal comparison plot."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Raw Signal (with artifact)", "Cleaned Signal", "Residual Error"),
        horizontal_spacing=0.08,
    )

    # Raw signal
    fig.add_trace(
        go.Scatter(
            x=t, y=raw_signal,
            mode="lines",
            name="Raw",
            line=dict(color=COLORS["warning_red"], width=1),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=clean_source,
            mode="lines",
            name="Ground Truth",
            line=dict(color=COLORS["nanotech_cyan"], width=1, dash="dot"),
            opacity=0.5,
        ),
        row=1, col=1,
    )

    # Cleaned signal
    fig.add_trace(
        go.Scatter(
            x=t, y=cleaned_signal,
            mode="lines",
            name="Cleaned",
            line=dict(color=COLORS["success_green"], width=1),
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=clean_source,
            mode="lines",
            name="Ground Truth",
            line=dict(color=COLORS["nanotech_cyan"], width=1, dash="dot"),
            opacity=0.5,
        ),
        row=1, col=2,
    )

    # Residual
    fig.add_trace(
        go.Scatter(
            x=t, y=residual,
            mode="lines",
            name="Residual",
            line=dict(color=COLORS["safety_yellow"], width=1),
        ),
        row=1, col=3,
    )
    # Zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["text_secondary"], row=1, col=3)

    fig.update_layout(
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["panel_bg"],
        font=dict(color=COLORS["text_primary"]),
        margin=dict(l=50, r=50, t=60, b=50),
    )

    fig.update_xaxes(
        title_text="Time (s)",
        gridcolor=COLORS["grid_line"],
        showgrid=True,
    )
    fig.update_yaxes(
        title_text="Amplitude",
        gridcolor=COLORS["grid_line"],
        showgrid=True,
    )

    return fig


def create_source_plot(
    t: np.ndarray,
    sources: dict[str, np.ndarray],
) -> go.Figure:
    """Create plot showing all source signals."""
    fig = go.Figure()

    colors = [COLORS["source_a"], COLORS["source_b"], COLORS["source_c"]]
    names = ["Alpha (10 Hz)", "Beta (20 Hz)", "Pink Noise"]

    for i, (key, color, name) in enumerate(zip(sources.keys(), colors, names)):
        offset = (1 - i) * 3  # Stack vertically
        fig.add_trace(
            go.Scatter(
                x=t,
                y=sources[key] + offset,
                mode="lines",
                name=name,
                line=dict(color=color, width=1),
            )
        )

    fig.update_layout(
        title="Neural Source Signals",
        height=250,
        showlegend=True,
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["panel_bg"],
        font=dict(color=COLORS["text_primary"]),
        margin=dict(l=50, r=50, t=50, b=50),
    )

    fig.update_xaxes(title_text="Time (s)", gridcolor=COLORS["grid_line"])
    fig.update_yaxes(title_text="Amplitude (stacked)", gridcolor=COLORS["grid_line"])

    return fig


# =============================================================================
# Sidebar Controls
# =============================================================================

def render_sidebar() -> dict:
    """Render sidebar controls and return parameters."""
    st.sidebar.markdown(
        f"<h2 style='color: {COLORS['nanotech_cyan']}'>Signal Bench Parameters</h2>",
        unsafe_allow_html=True,
    )

    # Preset selector
    st.sidebar.markdown("### Presets")
    presets = get_preset_names_and_descriptions()
    preset_names = ["Custom"] + [p[1] for p in presets]
    preset_keys = [None] + [p[0] for p in presets]

    selected_preset_idx = st.sidebar.selectbox(
        "Load Preset",
        range(len(preset_names)),
        format_func=lambda i: preset_names[i],
        help="Load pre-configured parameters for common scenarios",
    )

    # Load preset if selected
    if selected_preset_idx > 0:
        preset = get_preset(preset_keys[selected_preset_idx])
        st.sidebar.success(f"Loaded: {preset['name']}")
    else:
        preset = None

    st.sidebar.markdown("---")

    # Sensor configuration
    st.sidebar.markdown("### Sensor Configuration")
    n_sensors = st.sidebar.slider(
        "Number of Sensors",
        min_value=100,
        max_value=10000,
        value=preset["n_sensors"] if preset else 1000,
        step=100,
        help="Total ME nanoparticle sensors (up to 10k)",
    )

    st.sidebar.markdown("---")

    # Filter configuration
    st.sidebar.markdown("### Adaptive Filter")
    filter_type = st.sidebar.selectbox(
        "Filter Type",
        ["LMS", "RLS", "PhaseAwareRLS"],
        index=["LMS", "RLS", "PhaseAwareRLS"].index(preset["filter_type"]) if preset else 2,
        help="LMS: simple, slow convergence. RLS: fast, O(n^2). PhaseAwareRLS: optimized for cardiac.",
    )

    n_taps = st.sidebar.slider(
        "Filter Taps (n_taps)",
        min_value=4,
        max_value=64,
        value=preset["n_taps"] if preset else 8,
        step=4,
        help="More taps = better rejection but higher latency",
    )

    if filter_type == "LMS":
        mu = st.sidebar.slider(
            "Step Size (mu)",
            min_value=0.001,
            max_value=0.1,
            value=preset.get("mu", 0.01) if preset else 0.01,
            step=0.001,
            format="%.3f",
            help="Learning rate for LMS adaptation",
        )
        lambda_ = 0.99
    else:
        lambda_ = st.sidebar.slider(
            "Forgetting Factor (lambda)",
            min_value=0.85,
            max_value=0.999,
            value=preset.get("lambda_", 0.95) if preset else 0.95,
            step=0.01,
            format="%.2f",
            help="0.95 for non-stationary, 0.99 for stationary artifacts",
        )
        mu = 0.01

    st.sidebar.markdown("---")

    # Cardiac parameters
    st.sidebar.markdown("### Cardiac Physics")
    pulse_wave_velocity = st.sidebar.slider(
        "Pulse Wave Velocity (m/s)",
        min_value=3.0,
        max_value=15.0,
        value=preset["pulse_wave_velocity_m_s"] if preset else 7.5,
        step=0.5,
        help="5-7 m/s young, 10-15 m/s elderly. Affects phase delay.",
    )

    drift_amplitude = st.sidebar.slider(
        "Movement Noise Scale (mm)",
        min_value=0.01,
        max_value=0.20,
        value=preset["drift_amplitude_mm"] if preset else 0.05,
        step=0.01,
        format="%.2f",
        help="Cardiac pulsation amplitude (~50 microns typical)",
    )

    cardiac_freq = st.sidebar.slider(
        "Heart Rate (BPM)",
        min_value=40,
        max_value=180,
        value=int(preset["cardiac_freq_hz"] * 60) if preset else 72,
        step=4,
        help="Resting: 60-80 BPM, Exercise: 120-180 BPM",
    )
    cardiac_freq_hz = cardiac_freq / 60.0

    st.sidebar.markdown("---")

    # Experimental features
    st.sidebar.markdown("### Experimental Features")
    nanoparticle_drift = st.sidebar.checkbox(
        "Nanoparticle Drift Modeling",
        value=preset.get("nanoparticle_drift_enabled", False) if preset else False,
        help="PLACEHOLDER: Brownian motion modeling for SubSense 2026",
    )

    if nanoparticle_drift:
        st.sidebar.info(
            "Nanoparticle drift modeling is a placeholder for future Phase 6 development. "
            "Enable to see planned feature integration."
        )

    return {
        "n_sensors": n_sensors,
        "filter_type": filter_type,
        "n_taps": n_taps,
        "lambda_": lambda_,
        "mu": mu,
        "pulse_wave_velocity_m_s": pulse_wave_velocity,
        "drift_amplitude_mm": drift_amplitude,
        "cardiac_freq_hz": cardiac_freq_hz,
        "nanoparticle_drift_enabled": nanoparticle_drift,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "duration_sec": 2.0,
        "source_frequencies": {"alpha": 10.0, "beta": 20.0},
    }


# =============================================================================
# Validation Display
# =============================================================================

def render_validation_status(params: dict) -> None:
    """Render validation status panel."""
    st.markdown("### System Validation")

    col1, col2, col3 = st.columns(3)

    # Nyquist validation
    nyquist = validate_nyquist(
        params["sampling_rate_hz"],
        params["source_frequencies"],
    )

    with col1:
        if nyquist.is_valid:
            st.success(f"Nyquist: {nyquist.oversampling_factor:.1f}x oversampling")
        else:
            st.error("Nyquist Violation!")
            for err in nyquist.errors:
                st.error(err)
            for sug in nyquist.recovery_suggestions:
                st.info(f"Suggestion: {sug}")

    # Budget validation
    budget = validate_realtime_budget(
        params["filter_type"],
        params["n_sensors"],
        params["n_taps"],
    )

    with col2:
        if budget.is_within_budget:
            color = "success" if budget.budget_utilization < 0.8 else "warning"
            st.markdown(
                f"<div class='{color}'>Budget: {budget.estimated_latency_ms:.1f}ms / {budget.budget_ms}ms "
                f"({budget.budget_utilization * 100:.0f}%)</div>",
                unsafe_allow_html=True,
            )
            if budget.warnings:
                for w in budget.warnings:
                    st.warning(w)
        else:
            st.error(f"Budget Exceeded: {budget.estimated_latency_ms:.1f}ms > {budget.budget_ms}ms")
            for err in budget.errors:
                st.error(err)
            for sug in budget.recovery_suggestions:
                st.info(f"Suggestion: {sug}")

    # Config status
    with col3:
        config_result = validate_config_file()
        if config_result.is_valid:
            st.success("Config: Valid")
        else:
            st.warning("Config: Using defaults")
            for w in config_result.warnings:
                st.caption(w)


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main Streamlit application."""
    # Header
    st.markdown(
        f"""
        <h1 style='text-align: center; color: {COLORS["nanotech_cyan"]}'>
            SUBSENSE R&D SIGNAL BENCH
        </h1>
        <p style='text-align: center; color: {COLORS["text_secondary"]}'>
            Interactive Cardiac Artifact Rejection Explorer
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar parameters
    params = render_sidebar()

    # Validation status
    render_validation_status(params)

    st.markdown("---")

    # Generate signals
    with st.spinner("Generating signals..."):
        t, source_matrix, sources = generate_source_signals(
            params["duration_sec"],
            params["sampling_rate_hz"],
            params["source_frequencies"],
        )

        # Use alpha as primary signal for demonstration
        clean_source = sources["alpha"]

        # Generate cardiac artifact
        artifact = generate_cardiac_artifact(
            t,
            params["cardiac_freq_hz"],
            params["drift_amplitude_mm"],
            params["n_sensors"],
            params["pulse_wave_velocity_m_s"],
        )

        # Add noise and artifact to clean source
        np.random.seed(123)
        noise = np.random.randn(len(t)) * 0.1
        raw_signal = clean_source + artifact + noise

        # Apply simulated filtering
        cleaned_signal, residual = simulate_filtering(
            raw_signal,
            artifact,
            params["filter_type"],
            params["n_taps"],
            params["lambda_"],
            params["mu"],
        )

    # Metrics
    metrics = compute_metrics(clean_source, raw_signal, cleaned_signal, artifact, residual)

    # Metrics display
    st.markdown("### Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "MSE Improvement",
            f"{metrics['mse_improvement']:.1f}%",
            delta=f"{metrics['mse_improvement']:.1f}%",
        )

    with m2:
        st.metric(
            "SNR After",
            f"{metrics['snr_after_db']:.1f} dB",
            delta=f"+{metrics['snr_improvement_db']:.1f} dB",
        )

    with m3:
        st.metric(
            "Artifact Rejection",
            f"{metrics['rejection_db']:.1f} dB",
        )

    with m4:
        st.metric(
            "Correlation",
            f"{metrics['correlation']:.4f}",
        )

    st.markdown("---")

    # Signal comparison plot
    st.markdown("### Signal Comparison")
    fig_comparison = create_signal_comparison_plot(
        t, raw_signal, cleaned_signal, residual, clean_source
    )
    st.plotly_chart(fig_comparison, use_container_width=True)

    # Source signals
    st.markdown("### Neural Source Signals")
    fig_sources = create_source_plot(t, sources)
    st.plotly_chart(fig_sources, use_container_width=True)

    # Nanoparticle drift notice
    if params["nanoparticle_drift_enabled"]:
        st.markdown("---")
        st.markdown(
            f"""
            <div style='background-color: {COLORS["panel_bg"]}; padding: 1rem;
                        border-radius: 0.5rem; border-left: 4px solid {COLORS["safety_yellow"]}'>
                <h4 style='color: {COLORS["safety_yellow"]}'>Nanoparticle Drift Modeling (Experimental)</h4>
                <p style='color: {COLORS["text_secondary"]}'>
                    This feature is a <strong>placeholder</strong> for SubSense 2026 R&D roadmap.
                    Full Brownian drift modeling will include:
                </p>
                <ul style='color: {COLORS["text_secondary"]}'>
                    <li>Einstein-Stokes diffusion: D = k_B T / (6 pi eta r)</li>
                    <li>Wiener process integration for stochastic motion</li>
                    <li>Vessel wall boundary conditions</li>
                    <li>Combined cardiac + Brownian artifact rejection</li>
                </ul>
                <p style='color: {COLORS["text_secondary"]}'>
                    Estimated diffusion coefficient (100nm in blood @ 37C):
                    <code>D = 4.4 x 10^-6 mm^2/s</code>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <p style='text-align: center; color: {COLORS["text_secondary"]}; font-size: 0.8rem;'>
            SubSense R&D Signal Bench | Cardiac Artifact Rejection for ME Nanoparticle BCI<br>
            Real-time budget: {REALTIME_LATENCY_BUDGET_MS}ms | Sampling: {SAMPLING_RATE_HZ}Hz
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    # Parse command line arguments for demo mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default=None, help="Preset to load")
    parser.add_argument("--demo-mode", action="store_true", help="Run in demo mode")

    # Streamlit passes its own args, so we need to handle this carefully
    # Just run main() - preset handling is done via session state in sidebar
    main()
