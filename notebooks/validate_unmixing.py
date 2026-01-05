"""
Subsense R&D Dashboard - Phase 3 Source Unmixing Validation

Professional dark-mode interface for validating ICA source recovery.

Dashboard Layout:
- Left column: Source waveform comparison (Ground Truth vs Recovered)
- Top right: Correlation matrix heatmap
- Bottom right: Recovery metrics panel

Run from project root:
    python notebooks/validate_unmixing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy import signal as scipy_signal

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from physics.constants import SAMPLING_RATE_HZ  # type: ignore[import-not-found]
    from filtering.unmixing import load_phase2_data, unmix_sources, UnmixingResult  # type: ignore[import-not-found]
except ImportError:
    from src.physics.constants import SAMPLING_RATE_HZ
    from src.filtering.unmixing import load_phase2_data, unmix_sources, UnmixingResult

# =============================================================================
# SUBSENSE COLOR PALETTE (consistent with Phase 1 & 2)
# =============================================================================
COLORS = {
    "background": "#0f0f0f",
    "panel_bg": "#12121a",
    "nanotech_cyan": "#00FFFF",
    "warning_red": "#FF3333",
    "safety_yellow": "#FFD700",
    "grid_line": "#1a1a2e",
    "text_primary": "#E0E0E0",
    "text_secondary": "#808080",
    "text_accent": "#00FFFF",
    "success_green": "#00FF88",
    "source_a": "#FF6B6B",  # Alpha - warm red
    "source_b": "#4ECDC4",  # Beta - teal
    "source_c": "#95E1D3",  # Pink noise - soft green
    "recovered": "#FFD93D",  # Recovered signal - gold
}


def setup_axis_style(ax, title: str) -> None:
    """Apply consistent dark styling to an axis."""
    ax.set_facecolor(COLORS["panel_bg"])
    ax.set_title(
        title,
        color=COLORS["text_accent"],
        fontsize=10,
        fontweight="bold",
        fontfamily="monospace",
        pad=8,
    )
    ax.tick_params(colors=COLORS["text_secondary"], labelsize=8)
    ax.grid(True, alpha=0.2, color=COLORS["grid_line"])

    for spine in ax.spines.values():
        spine.set_color(COLORS["grid_line"])


def plot_source_comparison(
    axes: list,
    time_vector: np.ndarray,
    ground_truth: np.ndarray,
    recovered: np.ndarray,
    correlations: np.ndarray,
    time_window: tuple[float, float] = (0.0, 0.5),
) -> None:
    """Plot ground truth vs recovered sources side by side."""

    source_info = [
        ("SOURCE A: 10Hz Alpha", COLORS["source_a"]),
        ("SOURCE B: 20Hz Beta", COLORS["source_b"]),
        ("SOURCE C: Pink Noise", COLORS["source_c"]),
    ]

    # Time window mask
    mask = (time_vector >= time_window[0]) & (time_vector <= time_window[1])
    t = time_vector[mask] * 1000  # Convert to ms

    for i, (ax, (label, color)) in enumerate(zip(axes, source_info)):
        gt = ground_truth[i, mask]
        rec = recovered[i, mask]

        # Normalize both to same scale for visual comparison
        gt_norm = (gt - np.mean(gt)) / np.std(gt)
        rec_norm = (rec - np.mean(rec)) / np.std(rec)

        # Plot ground truth (solid)
        ax.plot(
            t, gt_norm,
            color=color,
            linewidth=1.5,
            alpha=0.9,
            label="Ground Truth",
        )

        # Plot recovered (dashed overlay)
        ax.plot(
            t, rec_norm,
            color=COLORS["recovered"],
            linewidth=1.2,
            alpha=0.8,
            linestyle="--",
            label="Recovered (ICA)",
        )

        # Title with correlation
        corr = correlations[i]
        quality_color = COLORS["success_green"] if corr > 0.95 else COLORS["safety_yellow"] if corr > 0.85 else COLORS["warning_red"]
        setup_axis_style(ax, f"{label}")

        # Correlation annotation
        ax.text(
            0.98, 0.95,
            f"r = {corr:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            fontfamily="monospace",
            fontweight="bold",
            color=quality_color,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["panel_bg"], edgecolor=quality_color, alpha=0.9),
        )

        ax.set_xlabel("Time (ms)", color=COLORS["text_secondary"], fontsize=8)
        ax.set_xlim([t[0], t[-1]])
        ax.set_ylim([-3.5, 3.5])

        if i == 0:
            ax.legend(
                loc="upper left",
                fontsize=8,
                framealpha=0.8,
                facecolor=COLORS["panel_bg"],
                edgecolor=COLORS["grid_line"],
                labelcolor=COLORS["text_primary"],
            )


def plot_correlation_matrix(
    ax,
    corr_matrix: np.ndarray,
) -> None:
    """Plot correlation heatmap between recovered and ground truth."""

    # Take absolute values (ICA has sign ambiguity)
    abs_corr = np.abs(corr_matrix)

    # Create heatmap
    im = ax.imshow(
        abs_corr,
        cmap="viridis",
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(colors=COLORS["text_secondary"], labelsize=8)
    cbar.set_label("|Correlation|", color=COLORS["text_secondary"], fontsize=9)

    # Labels
    source_labels = ["A", "B", "C"]
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(source_labels, fontsize=9, fontfamily="monospace")
    ax.set_yticklabels(source_labels, fontsize=9, fontfamily="monospace")
    ax.set_xlabel("Ground Truth", color=COLORS["text_secondary"], fontsize=9)
    ax.set_ylabel("Recovered (ICA)", color=COLORS["text_secondary"], fontsize=9)

    # Annotate cells with correlation values
    for i in range(3):
        for j in range(3):
            val = abs_corr[i, j]
            text_color = COLORS["background"] if val > 0.5 else COLORS["text_primary"]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                fontfamily="monospace",
                color=text_color,
            )

    setup_axis_style(ax, "CORRELATION MATRIX")
    ax.grid(False)


def plot_psd_comparison(
    ax,
    ground_truth: np.ndarray,
    recovered: np.ndarray,
    fs: float = SAMPLING_RATE_HZ,
) -> None:
    """Plot power spectral density comparison."""

    source_colors = [COLORS["source_a"], COLORS["source_b"], COLORS["source_c"]]

    for i in range(3):
        # Compute PSD for ground truth
        f_gt, psd_gt = scipy_signal.welch(ground_truth[i], fs=fs, nperseg=512)

        # Compute PSD for recovered
        f_rec, psd_rec = scipy_signal.welch(recovered[i], fs=fs, nperseg=512)

        # Normalize PSDs
        psd_gt = psd_gt / np.max(psd_gt)
        psd_rec = psd_rec / np.max(psd_rec)

        # Plot (offset for clarity)
        offset = (2 - i) * 0.3
        ax.semilogy(
            f_gt, psd_gt + offset,
            color=source_colors[i],
            linewidth=1.2,
            alpha=0.9,
        )
        ax.semilogy(
            f_rec, psd_rec + offset,
            color=COLORS["recovered"],
            linewidth=1.0,
            alpha=0.7,
            linestyle="--",
        )

    setup_axis_style(ax, "POWER SPECTRAL DENSITY")
    ax.set_xlabel("Frequency (Hz)", color=COLORS["text_secondary"], fontsize=9)
    ax.set_ylabel("PSD (normalized)", color=COLORS["text_secondary"], fontsize=9)
    ax.set_xlim([0, 50])
    ax.set_ylim([1e-4, 10])


def render_metrics_panel(
    ax,
    result: UnmixingResult,
) -> None:
    """Render recovery metrics panel."""
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
        0.5, 0.92,
        "RECOVERY METRICS",
        ha="center",
        fontsize=10,
        fontweight="bold",
        fontfamily="monospace",
        color=COLORS["text_accent"],
    )

    # PCA section
    ax.text(0.1, 0.82, "PCA:", fontsize=9, fontfamily="monospace", color=COLORS["text_accent"])
    ax.text(0.15, 0.74, f"Components: {result.n_pca_components}", fontsize=8, fontfamily="monospace", color=COLORS["text_primary"])
    ax.text(0.15, 0.67, f"Var. Explained: {result.variance_explained*100:.1f}%", fontsize=8, fontfamily="monospace", color=COLORS["text_primary"])

    # ICA section
    ax.text(0.1, 0.56, "FastICA:", fontsize=9, fontfamily="monospace", color=COLORS["text_accent"])
    ax.text(0.15, 0.48, f"Iterations: {result.ica_n_iter}", fontsize=8, fontfamily="monospace", color=COLORS["text_primary"])

    # Correlation section
    ax.text(0.1, 0.37, "Recovery:", fontsize=9, fontfamily="monospace", color=COLORS["text_accent"])

    source_info = [
        ("A (Alpha)", result.matched_correlations[0], COLORS["source_a"]),
        ("B (Beta)", result.matched_correlations[1], COLORS["source_b"]),
        ("C (Pink)", result.matched_correlations[2], COLORS["source_c"]),
    ]

    y_pos = 0.29
    for name, corr, color in source_info:
        quality = "Excellent" if corr > 0.95 else "Good" if corr > 0.85 else "Fair"
        q_color = COLORS["success_green"] if corr > 0.95 else COLORS["safety_yellow"] if corr > 0.85 else COLORS["warning_red"]
        ax.text(0.15, y_pos, f"{name}:", fontsize=8, fontfamily="monospace", color=color)
        ax.text(0.55, y_pos, f"r={corr:.3f}", fontsize=8, fontfamily="monospace", color=COLORS["text_primary"])
        ax.text(0.85, y_pos, quality, fontsize=8, fontfamily="monospace", color=q_color, ha="right")
        y_pos -= 0.07

    # Overall status
    avg_corr = np.mean(result.matched_correlations)
    if avg_corr > 0.95:
        status = "UNMIXING SUCCESS"
        status_color = COLORS["success_green"]
    elif avg_corr > 0.85:
        status = "GOOD RECOVERY"
        status_color = COLORS["safety_yellow"]
    else:
        status = "PARTIAL RECOVERY"
        status_color = COLORS["warning_red"]

    ax.text(
        0.5, 0.05,
        status,
        ha="center",
        fontsize=9,
        fontfamily="monospace",
        fontweight="bold",
        color=status_color,
    )


def main() -> None:
    """Generate the Phase 3 unmixing validation dashboard."""

    # Apply dark theme
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = COLORS["background"]
    plt.rcParams["axes.facecolor"] = COLORS["panel_bg"]
    plt.rcParams["savefig.facecolor"] = COLORS["background"]

    print("=" * 60)
    print("  PHASE 3 DASHBOARD - Source Unmixing Validation")
    print("=" * 60)

    # Load data and run unmixing
    print("\n  [1/4] Loading data and running unmixing pipeline...")
    recording, ground_truth, time_vector = load_phase2_data()
    result = unmix_sources(recording, ground_truth)

    # Create figure
    print("\n  [2/4] Rendering dashboard...")
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(COLORS["background"])

    # Create grid: 3 rows for sources (left), 2 rows for metrics (right)
    gs = gridspec.GridSpec(
        3, 3,
        width_ratios=[2, 2, 1.2],
        height_ratios=[1, 1, 1],
        hspace=0.35,
        wspace=0.25,
        left=0.06,
        right=0.95,
        top=0.90,
        bottom=0.08,
    )

    # Source comparison plots (left two columns, each row)
    ax_sources = []
    for i in range(3):
        ax = fig.add_subplot(gs[i, 0:2])
        ax_sources.append(ax)

    plot_source_comparison(
        ax_sources,
        time_vector,
        ground_truth,
        result.matched_sources,
        result.matched_correlations,
    )

    # Correlation matrix (top right)
    ax_corr = fig.add_subplot(gs[0, 2])
    plot_correlation_matrix(ax_corr, result.correlation_matrix)

    # PSD comparison (middle right)
    ax_psd = fig.add_subplot(gs[1, 2])
    plot_psd_comparison(ax_psd, ground_truth, result.matched_sources)

    # Metrics panel (bottom right)
    ax_metrics = fig.add_subplot(gs[2, 2])
    render_metrics_panel(ax_metrics, result)

    # Main title
    fig.suptitle(
        "SUBSENSE SOURCE RECOVERY - PCA/ICA UNMIXING VALIDATION",
        fontsize=14,
        fontweight="bold",
        fontfamily="monospace",
        color=COLORS["text_accent"],
        y=0.96,
    )

    # Legend note
    fig.text(
        0.5, 0.02,
        "Solid = Ground Truth  |  Dashed Gold = Recovered (ICA)",
        ha="center",
        fontsize=9,
        fontfamily="monospace",
        color=COLORS["text_secondary"],
    )

    # Save
    print("  [3/4] Saving dashboard...")
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase3_unmixing.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=COLORS["background"])

    # Print summary
    print("\n  [4/4] Summary:")
    print(f"        PCA components: {result.n_pca_components}")
    print(f"        Variance explained: {result.variance_explained*100:.1f}%")
    for i, name in enumerate(["A (Alpha)", "B (Beta)", "C (Pink)"]):
        print(f"        Source {name}: r = {result.matched_correlations[i]:.4f}")

    print("\n" + "=" * 60)
    print(f"  Dashboard saved: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
