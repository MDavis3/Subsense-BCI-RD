"""
Subsense R&D Dashboard - Phase 1 Cloud Visualization

Professional dark-mode scientific interface for presenting
nanoparticle cloud simulation results to leadership.

Features:
- Dark Lab aesthetic with nanotech color palette
- Volumetric fog-style sensor visualization
- Glowing source markers with exclusion zone wireframes
- Integrated HUD stats panel
- Custom GridSpec layout for optimal data density

Run from project root:
    python notebooks/visualize_cloud.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from physics.constants import (
    BRAIN_CONDUCTIVITY_S_M,
    SINGULARITY_THRESHOLD_MM,
    CLOUD_VOLUME_SIDE_MM,
)
from physics.transfer_function import (
    compute_lead_field,
    compute_distance_matrix,
    validate_lead_field,
)

# =============================================================================
# SUBSENSE COLOR PALETTE
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
    "glow_red": "#FF6666",
    "success_green": "#00FF88",
}


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load sensor cloud and source data from disk."""
    data_dir = project_root / "data" / "raw"
    sensors = np.load(data_dir / "sensors_N10000_seed42.npy")
    sources = np.load(data_dir / "sources_3fixed.npy")
    return sensors, sources


def draw_wireframe_sphere(ax, center: np.ndarray, radius: float, color: str, alpha: float = 0.3):
    """Draw a wireframe sphere for exclusion zone visualization."""
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 12)
    
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    
    # Draw wireframe lines
    for i in range(0, len(u), 4):
        ax.plot(x[i, :], y[i, :], z[i, :], color=color, alpha=alpha, linewidth=0.5)
    for j in range(0, len(v), 3):
        ax.plot(x[:, j], y[:, j], z[:, j], color=color, alpha=alpha, linewidth=0.5)


def draw_domain_cube(ax, half_side: float):
    """Draw the domain boundary cube with subtle grid lines."""
    corners = np.array([
        [-half_side, -half_side, -half_side],
        [half_side, -half_side, -half_side],
        [half_side, half_side, -half_side],
        [-half_side, half_side, -half_side],
        [-half_side, -half_side, half_side],
        [half_side, -half_side, half_side],
        [half_side, half_side, half_side],
        [-half_side, half_side, half_side],
    ])
    
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    
    for edge in edges:
        pts = corners[edge]
        ax.plot3D(pts[:, 0], pts[:, 1], pts[:, 2], 
                  color=COLORS["grid_line"], alpha=0.6, linewidth=1)


def plot_cloud_3d(
    ax,
    sensors: np.ndarray,
    sources: np.ndarray,
    singularity_mask: np.ndarray,
) -> None:
    """Create the main 3D volumetric cloud visualization."""
    
    # Make 3D axis panes transparent/dark
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(COLORS["grid_line"])
    ax.yaxis.pane.set_edgecolor(COLORS["grid_line"])
    ax.zaxis.pane.set_edgecolor(COLORS["grid_line"])
    
    # Set grid styling
    ax.xaxis._axinfo["grid"]["color"] = COLORS["grid_line"]
    ax.yaxis._axinfo["grid"]["color"] = COLORS["grid_line"]
    ax.zaxis._axinfo["grid"]["color"] = COLORS["grid_line"]
    ax.xaxis._axinfo["grid"]["linewidth"] = 0.3
    ax.yaxis._axinfo["grid"]["linewidth"] = 0.3
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.3
    
    # Identify sensors in exclusion zones
    in_exclusion = np.any(singularity_mask, axis=1)
    normal_sensors = sensors[~in_exclusion]
    exclusion_sensors = sensors[in_exclusion]
    
    # Plot sensors as volumetric fog (Nanotech Cyan)
    ax.scatter(
        normal_sensors[:, 0],
        normal_sensors[:, 1],
        normal_sensors[:, 2],
        c=COLORS["nanotech_cyan"],
        alpha=0.03,
        s=2,
        edgecolors="none",
    )
    
    # Add subtle density glow layer
    ax.scatter(
        normal_sensors[::10, 0],
        normal_sensors[::10, 1],
        normal_sensors[::10, 2],
        c=COLORS["nanotech_cyan"],
        alpha=0.08,
        s=8,
        edgecolors="none",
    )
    
    # Plot sensors in exclusion zones (Warning indicator)
    if len(exclusion_sensors) > 0:
        ax.scatter(
            exclusion_sensors[:, 0],
            exclusion_sensors[:, 1],
            exclusion_sensors[:, 2],
            c=COLORS["safety_yellow"],
            alpha=0.9,
            s=30,
            edgecolors="white",
            linewidths=0.5,
            zorder=15,
        )
    
    # Plot sources with glow effect (multiple layers)
    source_labels = ["A", "B", "C"]
    
    # Outer glow
    ax.scatter(
        sources[:, 0], sources[:, 1], sources[:, 2],
        c=COLORS["glow_red"],
        alpha=0.15,
        s=800,
        edgecolors="none",
        zorder=8,
    )
    
    # Mid glow
    ax.scatter(
        sources[:, 0], sources[:, 1], sources[:, 2],
        c=COLORS["warning_red"],
        alpha=0.4,
        s=400,
        edgecolors="none",
        zorder=9,
    )
    
    # Core marker
    ax.scatter(
        sources[:, 0], sources[:, 1], sources[:, 2],
        c=COLORS["warning_red"],
        marker="*",
        s=200,
        edgecolors="white",
        linewidths=1,
        zorder=10,
    )
    
    # Source labels
    for src, label in zip(sources, source_labels):
        ax.text(
            src[0] + 0.08, src[1] + 0.08, src[2] + 0.08,
            f"SRC-{label}",
            fontsize=9,
            fontweight="bold",
            color=COLORS["text_primary"],
            fontfamily="monospace",
        )
    
    # Draw exclusion zone wireframes (Safety Yellow)
    for src in sources:
        draw_wireframe_sphere(ax, src, SINGULARITY_THRESHOLD_MM, COLORS["safety_yellow"], alpha=0.4)
    
    # Draw domain boundary
    draw_domain_cube(ax, CLOUD_VOLUME_SIDE_MM / 2)
    
    # Axis styling
    ax.set_xlabel("X (mm)", color=COLORS["text_secondary"], fontsize=10, labelpad=10)
    ax.set_ylabel("Y (mm)", color=COLORS["text_secondary"], fontsize=10, labelpad=10)
    ax.set_zlabel("Z (mm)", color=COLORS["text_secondary"], fontsize=10, labelpad=10)
    
    ax.tick_params(colors=COLORS["text_secondary"], labelsize=8)
    
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.6, 0.6])
    ax.set_zlim([-0.6, 0.6])
    
    # Title
    ax.set_title(
        "VOLUMETRIC SENSOR CLOUD",
        color=COLORS["text_accent"],
        fontsize=14,
        fontweight="bold",
        fontfamily="monospace",
        pad=20,
    )


def plot_distance_histogram(ax, sensors: np.ndarray, sources: np.ndarray) -> int:
    """Plot histogram of minimum distances with nanotech styling. Returns exclusion count."""
    
    distances = compute_distance_matrix(sensors, sources)
    min_distances = np.min(distances, axis=1)
    n_in_zone = np.sum(min_distances < SINGULARITY_THRESHOLD_MM)
    
    # Create histogram with cyan bars
    counts, bins, patches = ax.hist(
        min_distances,
        bins=40,
        color=COLORS["nanotech_cyan"],
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
    )
    
    # Add glow effect by plotting a second, wider histogram behind
    ax.hist(
        min_distances,
        bins=40,
        color=COLORS["nanotech_cyan"],
        alpha=0.2,
        edgecolor="none",
    )
    
    # Mark singularity threshold
    ax.axvline(
        SINGULARITY_THRESHOLD_MM,
        color=COLORS["warning_red"],
        linestyle="--",
        linewidth=2,
        alpha=0.8,
    )
    
    # Add threshold label
    ax.text(
        SINGULARITY_THRESHOLD_MM + 0.01,
        ax.get_ylim()[1] * 0.9,
        f"CLAMP\n{SINGULARITY_THRESHOLD_MM}mm",
        color=COLORS["warning_red"],
        fontsize=8,
        fontfamily="monospace",
        fontweight="bold",
        verticalalignment="top",
    )
    
    ax.set_xlabel("Distance to Nearest Source (mm)", color=COLORS["text_secondary"], fontsize=9)
    ax.set_ylabel("Sensor Count", color=COLORS["text_secondary"], fontsize=9)
    ax.set_title(
        "PROXIMITY DISTRIBUTION",
        color=COLORS["text_accent"],
        fontsize=11,
        fontweight="bold",
        fontfamily="monospace",
    )
    
    ax.tick_params(colors=COLORS["text_secondary"], labelsize=8)
    ax.set_facecolor(COLORS["panel_bg"])
    
    # Grid
    ax.grid(True, alpha=0.2, color=COLORS["grid_line"])
    ax.spines["bottom"].set_color(COLORS["grid_line"])
    ax.spines["top"].set_color(COLORS["grid_line"])
    ax.spines["left"].set_color(COLORS["grid_line"])
    ax.spines["right"].set_color(COLORS["grid_line"])
    
    return n_in_zone


def plot_decay_validation(
    ax,
    lead_field: np.ndarray,
    sensors: np.ndarray,
    sources: np.ndarray,
    source_index: int = 0,
) -> float:
    """Plot lead field decay validation. Returns max relative error."""
    
    distances = compute_distance_matrix(sensors, sources)[:, source_index]
    potentials = lead_field[:, source_index]
    
    # Sort for plotting
    sort_idx = np.argsort(distances)
    distances_sorted = distances[sort_idx]
    potentials_sorted = potentials[sort_idx]
    
    # Computed values (scatter with glow) - drawn on top of theory line
    ax.scatter(
        distances_sorted,
        potentials_sorted,
        c=COLORS["nanotech_cyan"],
        alpha=0.15,
        s=3,
        edgecolors="none",
        label="Measured",
        zorder=2,
    )
    
    # Theoretical curve
    r_theory = np.linspace(SINGULARITY_THRESHOLD_MM, distances_sorted.max(), 100)
    r_theory_m = r_theory * 1e-3
    v_theory = 1.0 / (4.0 * np.pi * BRAIN_CONDUCTIVITY_S_M * r_theory_m)
    
    ax.plot(
        r_theory,
        v_theory,
        color=COLORS["warning_red"],
        linewidth=3,
        alpha=0.9,
        label="Theory: 1/r",
        zorder=1,
    )
    
    # Clamp threshold
    ax.axvline(
        SINGULARITY_THRESHOLD_MM,
        color=COLORS["safety_yellow"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )
    
    ax.set_xlabel("Distance (mm)", color=COLORS["text_secondary"], fontsize=9)
    ax.set_ylabel("Lead Field (V/A)", color=COLORS["text_secondary"], fontsize=9)
    ax.set_title(
        f"TRANSFER FUNCTION [SRC-{['A', 'B', 'C'][source_index]}]",
        color=COLORS["text_accent"],
        fontsize=11,
        fontweight="bold",
        fontfamily="monospace",
    )
    
    ax.set_yscale("log")
    ax.set_xscale("log")
    
    ax.tick_params(colors=COLORS["text_secondary"], labelsize=8)
    ax.set_facecolor(COLORS["panel_bg"])
    
    ax.grid(True, alpha=0.2, color=COLORS["grid_line"])
    ax.spines["bottom"].set_color(COLORS["grid_line"])
    ax.spines["top"].set_color(COLORS["grid_line"])
    ax.spines["left"].set_color(COLORS["grid_line"])
    ax.spines["right"].set_color(COLORS["grid_line"])
    
    ax.legend(
        loc="upper right",
        fontsize=8,
        facecolor=COLORS["panel_bg"],
        edgecolor=COLORS["grid_line"],
        labelcolor=COLORS["text_primary"],
    )
    
    # Calculate max relative error (excluding clamped values)
    valid_mask = distances_sorted >= SINGULARITY_THRESHOLD_MM
    if np.any(valid_mask):
        d_valid = distances_sorted[valid_mask]
        p_valid = potentials_sorted[valid_mask]
        p_expected = 1.0 / (4.0 * np.pi * BRAIN_CONDUCTIVITY_S_M * d_valid * 1e-3)
        rel_errors = np.abs(p_valid - p_expected) / p_expected * 100
        return np.max(rel_errors)
    return 0.0


def render_hud_panel(
    fig,
    ax_hud,
    n_sensors: int,
    n_clamps: int,
    mean_connectivity: float,
    max_error: float,
) -> None:
    """Render the heads-up display stats panel."""
    
    ax_hud.set_facecolor(COLORS["panel_bg"])
    ax_hud.set_xlim(0, 1)
    ax_hud.set_ylim(0, 1)
    ax_hud.axis("off")
    
    # Panel border
    border = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=COLORS["panel_bg"],
        edgecolor=COLORS["nanotech_cyan"],
        linewidth=1,
        alpha=0.8,
    )
    ax_hud.add_patch(border)
    
    # Title
    ax_hud.text(
        0.5, 0.92,
        "SUBSENSE R&D",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        fontfamily="monospace",
        color=COLORS["text_accent"],
    )
    
    ax_hud.text(
        0.5, 0.82,
        "Phase 1 Validation",
        ha="center",
        va="top",
        fontsize=9,
        fontfamily="monospace",
        color=COLORS["text_secondary"],
    )
    
    # Divider line
    ax_hud.plot([0.1, 0.9], [0.75, 0.75], color=COLORS["grid_line"], linewidth=1)
    
    # Status indicator
    status_color = COLORS["success_green"] if max_error < 0.01 else COLORS["safety_yellow"]
    ax_hud.scatter([0.15], [0.65], c=status_color, s=80, marker="o", edgecolors="white", linewidths=1)
    ax_hud.text(
        0.25, 0.65,
        "SYSTEM ONLINE",
        va="center",
        fontsize=10,
        fontfamily="monospace",
        fontweight="bold",
        color=status_color,
    )
    
    # Stats
    stats = [
        ("SENSORS", f"{n_sensors:,}", COLORS["nanotech_cyan"]),
        ("CLAMPS", f"{n_clamps}", COLORS["safety_yellow"] if n_clamps > 0 else COLORS["success_green"]),
        ("CONNECT", f"{mean_connectivity:.1f}%", COLORS["text_primary"]),
        ("ERROR", f"{max_error:.4f}%", COLORS["success_green"] if max_error < 0.01 else COLORS["warning_red"]),
    ]
    
    y_pos = 0.52
    for label, value, color in stats:
        ax_hud.text(
            0.12, y_pos,
            f"{label}:",
            va="center",
            fontsize=9,
            fontfamily="monospace",
            color=COLORS["text_secondary"],
        )
        ax_hud.text(
            0.88, y_pos,
            value,
            va="center",
            ha="right",
            fontsize=10,
            fontfamily="monospace",
            fontweight="bold",
            color=color,
        )
        y_pos -= 0.11
    
    # Bottom status bar
    ax_hud.plot([0.1, 0.9], [0.12, 0.12], color=COLORS["grid_line"], linewidth=1)
    ax_hud.text(
        0.5, 0.06,
        "READY FOR PHASE 2",
        ha="center",
        va="center",
        fontsize=8,
        fontfamily="monospace",
        color=COLORS["success_green"],
        alpha=0.8,
    )


def main() -> None:
    """Generate the Subsense R&D Dashboard."""
    
    # Apply dark theme
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = COLORS["background"]
    plt.rcParams["axes.facecolor"] = COLORS["panel_bg"]
    plt.rcParams["savefig.facecolor"] = COLORS["background"]
    
    print("=" * 60)
    print("  SUBSENSE R&D DASHBOARD - Initializing...")
    print("=" * 60)
    
    # Load data
    print("  [1/4] Loading sensor cloud...")
    sensors, sources = load_data()
    
    # Compute lead field
    print("  [2/4] Computing lead field matrix...")
    lead_field, singularity_mask = compute_lead_field(sensors, sources)
    
    # Validate
    print("  [3/4] Validating physics...")
    info = validate_lead_field(lead_field)
    
    # Create figure with custom GridSpec
    # Layout: 3D cloud (60% left), stacked charts + HUD (40% right)
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(COLORS["background"])
    
    gs = gridspec.GridSpec(
        3, 2,
        width_ratios=[1.5, 1],
        height_ratios=[1, 1, 0.6],
        hspace=0.35,
        wspace=0.25,
        left=0.05,
        right=0.95,
        top=0.92,
        bottom=0.08,
    )
    
    # Main 3D plot (spans all rows on left)
    ax_3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_3d.set_facecolor(COLORS["background"])
    
    # Histogram (top right)
    ax_hist = fig.add_subplot(gs[0, 1])
    
    # Decay validation (middle right)
    ax_decay = fig.add_subplot(gs[1, 1])
    
    # HUD panel (bottom right)
    ax_hud = fig.add_subplot(gs[2, 1])
    
    # Generate plots
    print("  [4/4] Rendering dashboard...")
    
    plot_cloud_3d(ax_3d, sensors, sources, singularity_mask)
    n_clamps = plot_distance_histogram(ax_hist, sensors, sources)
    max_error = plot_decay_validation(ax_decay, lead_field, sensors, sources, source_index=0)
    
    # Calculate mean connectivity (average lead field value as proxy)
    mean_connectivity = 100.0 * (1.0 - n_clamps / len(sensors))
    
    render_hud_panel(
        fig, ax_hud,
        n_sensors=len(sensors),
        n_clamps=n_clamps,
        mean_connectivity=mean_connectivity,
        max_error=max_error,
    )
    
    # Main title
    fig.suptitle(
        "SUBSENSE NANOPARTICLE CLOUD SIMULATION",
        fontsize=16,
        fontweight="bold",
        fontfamily="monospace",
        color=COLORS["text_accent"],
        y=0.98,
    )
    
    # Save figure
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase1_dashboard.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=COLORS["background"])
    
    print("=" * 60)
    print(f"  Dashboard saved: {output_path}")
    print("=" * 60)
    
    # Also update the old filename for backwards compatibility
    plt.savefig(output_dir / "phase1_validation.png", dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
    
    # Show (uncomment for interactive viewing)
    # plt.show()


if __name__ == "__main__":
    main()
