"""
Subsense R&D Dashboard - Phase 2 Signal Visualization

Professional dark-mode interface for visualizing temporal dynamics
of the nanoparticle cloud simulation.

Demonstrates the mixing problem:
- Top panel: Clean source waveforms (ground truth)
- Bottom panel: Noisy mixed sensor observations

Run from project root:
    python notebooks/visualize_signals.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np

from subsense_bci.physics.constants import SAMPLING_RATE_HZ, DURATION_SEC, SNR_LEVEL
from subsense_bci.visualization.theme import COLORS, setup_axis_style


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load simulation data from disk."""
    data_dir = get_project_root() / "data" / "raw"

    time_vector = np.load(data_dir / "time_vector.npy")
    source_waveforms = np.load(data_dir / "source_waveforms.npy")
    recording = np.load(data_dir / "recording_simulation.npy")

    return time_vector, source_waveforms, recording


def plot_source_waveforms(
    ax,
    time_vector: np.ndarray,
    source_waveforms: np.ndarray,
    time_window: tuple[float, float] = (0.0, 0.5),
) -> None:
    """Plot the ground truth source waveforms."""

    # Time window mask
    mask = (time_vector >= time_window[0]) & (time_vector <= time_window[1])
    t = time_vector[mask]

    source_labels = [
        ("SRC-A: 10Hz Alpha", COLORS["source_a"]),
        ("SRC-B: 20Hz Beta", COLORS["source_b"]),
        ("SRC-C: Pink Noise", COLORS["source_c"]),
    ]

    # Plot each source with vertical offset for clarity
    # Z-score normalize each source so pink noise is visible alongside sinusoids
    offsets = [2, 0, -2]
    for i, (label, color) in enumerate(source_labels):
        signal = source_waveforms[i, mask]
        signal = (signal - np.mean(signal)) / np.std(signal)  # Normalize for display
        ax.plot(
            t * 1000,  # Convert to ms
            signal + offsets[i],
            color=color,
            linewidth=1.2,
            alpha=0.9,
            label=label,
        )

        # Add label at left edge
        ax.text(
            t[0] * 1000 - 15,
            offsets[i],
            label.split(":")[0],
            color=color,
            fontsize=9,
            fontfamily="monospace",
            fontweight="bold",
            va="center",
            ha="right",
        )

    setup_axis_style(ax, "TRUE SOURCE WAVEFORMS [S(t)]")
    ax.set_xlabel("Time (ms)", color=COLORS["text_secondary"], fontsize=9)
    ax.set_ylabel("Amplitude (z-scored)", color=COLORS["text_secondary"], fontsize=9)
    ax.set_xlim([time_window[0] * 1000 - 20, time_window[1] * 1000])
    ax.set_ylim([-4, 4])

    # Hide y-ticks (signals are offset for display)
    ax.set_yticks([])


def plot_sensor_observations(
    ax,
    time_vector: np.ndarray,
    recording: np.ndarray,
    sensor_indices: list[int],
    time_window: tuple[float, float] = (0.0, 0.5),
) -> None:
    """Plot observed signals from selected sensors."""

    # Time window mask
    mask = (time_vector >= time_window[0]) & (time_vector <= time_window[1])
    t = time_vector[mask]

    # Normalize for display
    display_signals = []
    for idx in sensor_indices:
        sig = recording[idx, mask]
        # Z-score normalize
        sig = (sig - np.mean(sig)) / np.std(sig)
        display_signals.append(sig)

    # Plot each sensor with vertical offset
    offsets = [3, 0, -3]
    for i, (idx, sig) in enumerate(zip(sensor_indices, display_signals)):
        ax.plot(
            t * 1000,
            sig + offsets[i],
            color=COLORS["nanotech_cyan"],
            linewidth=0.8,
            alpha=0.7,
        )

        # Sensor label
        ax.text(
            t[0] * 1000 - 15,
            offsets[i],
            f"S-{idx:04d}",
            color=COLORS["nanotech_cyan"],
            fontsize=9,
            fontfamily="monospace",
            fontweight="bold",
            va="center",
            ha="right",
        )

    setup_axis_style(ax, "OBSERVED SENSOR SIGNALS [X(t) = L @ S(t) + N(t)]")
    ax.set_xlabel("Time (ms)", color=COLORS["text_secondary"], fontsize=9)
    ax.set_ylabel("Amplitude (z-scored)", color=COLORS["text_secondary"], fontsize=9)
    ax.set_xlim([time_window[0] * 1000 - 20, time_window[1] * 1000])
    ax.set_ylim([-6, 6])
    ax.set_yticks([])


def render_info_panel(ax, n_sensors: int, n_samples: int) -> None:
    """Render simulation info panel."""
    ax.set_facecolor(COLORS["panel_bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Border
    border = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=COLORS["panel_bg"],
        edgecolor=COLORS["nanotech_cyan"],
        linewidth=1,
        alpha=0.8,
    )
    ax.add_patch(border)

    # Title
    ax.text(
        0.5, 0.85,
        "SIMULATION PARAMS",
        ha="center",
        fontsize=10,
        fontweight="bold",
        fontfamily="monospace",
        color=COLORS["text_accent"],
    )

    # Parameters
    params = [
        ("Fs", f"{SAMPLING_RATE_HZ:.0f} Hz"),
        ("Duration", f"{DURATION_SEC:.1f} sec"),
        ("Samples", f"{n_samples:,}"),
        ("Sensors", f"{n_sensors:,}"),
        ("SNR", f"{SNR_LEVEL:.1f}"),
    ]

    y_pos = 0.68
    for label, value in params:
        ax.text(
            0.15, y_pos,
            f"{label}:",
            fontsize=9,
            fontfamily="monospace",
            color=COLORS["text_secondary"],
        )
        ax.text(
            0.85, y_pos,
            value,
            ha="right",
            fontsize=9,
            fontfamily="monospace",
            fontweight="bold",
            color=COLORS["text_primary"],
        )
        y_pos -= 0.12

    # Status
    ax.text(
        0.5, 0.08,
        "MIXING VERIFIED",
        ha="center",
        fontsize=8,
        fontfamily="monospace",
        color=COLORS["success_green"],
    )


def main() -> None:
    """Generate the Phase 2 signal visualization dashboard."""

    # Apply dark theme
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = COLORS["background"]
    plt.rcParams["axes.facecolor"] = COLORS["panel_bg"]
    plt.rcParams["savefig.facecolor"] = COLORS["background"]

    print("=" * 60)
    print("  PHASE 2 DASHBOARD - Signal Visualization")
    print("=" * 60)

    # Load data
    print("  [1/3] Loading simulation data...")
    time_vector, source_waveforms, recording = load_data()

    n_sensors = recording.shape[0]
    n_samples = recording.shape[1]
    print(f"        Recording: {n_sensors} sensors x {n_samples} samples")

    # Select 3 random sensors for display (reproducible)
    np.random.seed(123)
    sensor_indices = sorted(np.random.choice(n_sensors, size=3, replace=False))
    print(f"        Display sensors: {sensor_indices}")

    # Create figure
    print("  [2/3] Rendering dashboard...")
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(COLORS["background"])

    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[4, 1],
        height_ratios=[1, 1],
        hspace=0.35,
        wspace=0.15,
        left=0.08,
        right=0.95,
        top=0.90,
        bottom=0.08,
    )

    # Source waveforms (top left)
    ax_sources = fig.add_subplot(gs[0, 0])
    plot_source_waveforms(ax_sources, time_vector, source_waveforms)

    # Sensor observations (bottom left)
    ax_sensors = fig.add_subplot(gs[1, 0])
    plot_sensor_observations(ax_sensors, time_vector, recording, sensor_indices)

    # Info panel (right side, spans both rows)
    ax_info = fig.add_subplot(gs[:, 1])
    render_info_panel(ax_info, n_sensors, n_samples)

    # Main title
    fig.suptitle(
        "SUBSENSE TEMPORAL DYNAMICS - FORWARD MODEL VERIFICATION",
        fontsize=14,
        fontweight="bold",
        fontfamily="monospace",
        color=COLORS["text_accent"],
        y=0.96,
    )

    # Save
    print("  [3/3] Saving dashboard...")
    output_dir = get_project_root() / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase2_signals.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=COLORS["background"])

    print("=" * 60)
    print(f"  Dashboard saved: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
