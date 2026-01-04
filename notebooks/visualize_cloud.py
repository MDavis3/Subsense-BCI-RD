"""
Visualization Script - Phase 1 Cloud and Source Validation

Generates 3D visualizations to validate:
1. Sensor cloud distribution within the 1mm^3 domain
2. Source placement
3. Exclusion zones (singularity handling)
4. Lead field computation

Run from project root:
    python notebooks/visualize_cloud.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3D projection)

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
    get_sensors_in_exclusion_zone,
    validate_lead_field,
)


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load sensor cloud and source data from disk."""
    data_dir = project_root / "data" / "raw"
    
    sensors = np.load(data_dir / "sensors_N10000_seed42.npy")
    sources = np.load(data_dir / "sources_3fixed.npy")
    
    return sensors, sources


def plot_cloud_and_sources(
    sensors: np.ndarray,
    sources: np.ndarray,
    singularity_mask: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Create 3D scatter plot of sensor cloud with sources and exclusion zones.
    
    Parameters
    ----------
    sensors : np.ndarray
        Sensor coordinates (N, 3) in mm.
    sources : np.ndarray
        Source coordinates (M, 3) in mm.
    singularity_mask : np.ndarray
        Boolean mask (N, M) indicating sensors in exclusion zones.
    ax : plt.Axes, optional
        Matplotlib 3D axes to plot on.
        
    Returns
    -------
    plt.Axes
        The axes with the plot.
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
    
    # Identify sensors in any exclusion zone
    in_exclusion = np.any(singularity_mask, axis=1)
    normal_sensors = sensors[~in_exclusion]
    exclusion_sensors = sensors[in_exclusion]
    
    # Plot normal sensors (blue, low alpha for density)
    ax.scatter(
        normal_sensors[:, 0],
        normal_sensors[:, 1],
        normal_sensors[:, 2],
        c="steelblue",
        alpha=0.05,
        s=1,
        label=f"Sensors (N={len(normal_sensors)})",
    )
    
    # Plot sensors in exclusion zones (orange, highlighted)
    if len(exclusion_sensors) > 0:
        ax.scatter(
            exclusion_sensors[:, 0],
            exclusion_sensors[:, 1],
            exclusion_sensors[:, 2],
            c="orange",
            alpha=0.8,
            s=20,
            label=f"In exclusion zone (N={len(exclusion_sensors)})",
        )
    
    # Plot sources (red stars)
    source_labels = ["A", "B", "C"]
    ax.scatter(
        sources[:, 0],
        sources[:, 1],
        sources[:, 2],
        c="red",
        marker="*",
        s=300,
        edgecolors="black",
        linewidths=1,
        label="Neural sources",
        zorder=10,
    )
    
    # Annotate sources
    for i, (src, label) in enumerate(zip(sources, source_labels)):
        ax.text(
            src[0] + 0.05,
            src[1] + 0.05,
            src[2] + 0.05,
            f"Source {label}",
            fontsize=10,
            fontweight="bold",
        )
    
    # Draw exclusion zone spheres (wireframe)
    for src in sources:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        r = SINGULARITY_THRESHOLD_MM
        
        x = r * np.outer(np.cos(u), np.sin(v)) + src[0]
        y = r * np.outer(np.sin(u), np.sin(v)) + src[1]
        z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + src[2]
        
        ax.plot_surface(x, y, z, alpha=0.1, color="red", linewidth=0)
    
    # Draw domain boundary (wireframe cube)
    half = CLOUD_VOLUME_SIDE_MM / 2
    corners = np.array([
        [-half, -half, -half],
        [half, -half, -half],
        [half, half, -half],
        [-half, half, -half],
        [-half, -half, half],
        [half, -half, half],
        [half, half, half],
        [-half, half, half],
    ])
    
    # Draw cube edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical
    ]
    for edge in edges:
        pts = corners[edge]
        ax.plot3D(pts[:, 0], pts[:, 1], pts[:, 2], "k-", alpha=0.3, linewidth=1)
    
    # Labels and formatting
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Phase 1: Stochastic Nanoparticle Cloud\n(10,000 sensors, 3 sources)")
    ax.legend(loc="upper left")
    
    # Equal aspect ratio
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.6, 0.6])
    ax.set_zlim([-0.6, 0.6])
    
    return ax


def plot_distance_histogram(
    sensors: np.ndarray,
    sources: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot histogram of minimum distances from sensors to any source.
    
    Parameters
    ----------
    sensors : np.ndarray
        Sensor coordinates (N, 3) in mm.
    sources : np.ndarray
        Source coordinates (M, 3) in mm.
    ax : plt.Axes, optional
        Matplotlib axes to plot on.
        
    Returns
    -------
    plt.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute distance matrix and find minimum distance per sensor
    distances = compute_distance_matrix(sensors, sources)
    min_distances = np.min(distances, axis=1)
    
    # Plot histogram
    ax.hist(min_distances, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
    
    # Mark singularity threshold
    ax.axvline(
        SINGULARITY_THRESHOLD_MM,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Singularity threshold ({SINGULARITY_THRESHOLD_MM} mm)",
    )
    
    # Count sensors in exclusion zone
    n_in_zone = np.sum(min_distances < SINGULARITY_THRESHOLD_MM)
    
    ax.set_xlabel("Distance to nearest source (mm)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Minimum Distances\n({n_in_zone} sensors in exclusion zone)")
    ax.legend()
    
    return ax


def plot_lead_field_validation(
    lead_field: np.ndarray,
    sensors: np.ndarray,
    sources: np.ndarray,
    source_index: int = 0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Validate lead field by plotting potential vs distance for one source.
    
    Parameters
    ----------
    lead_field : np.ndarray
        Lead field matrix (N, M) in V/A.
    sensors : np.ndarray
        Sensor coordinates (N, 3) in mm.
    sources : np.ndarray
        Source coordinates (M, 3) in mm.
    source_index : int, optional
        Which source to visualize. Default is 0 (Source A).
    ax : plt.Axes, optional
        Matplotlib axes to plot on.
        
    Returns
    -------
    plt.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get distances and potentials for this source
    distances = compute_distance_matrix(sensors, sources)[:, source_index]
    potentials = lead_field[:, source_index]
    
    # Sort by distance for cleaner plotting
    sort_idx = np.argsort(distances)
    distances_sorted = distances[sort_idx]
    potentials_sorted = potentials[sort_idx]
    
    # Plot measured values
    ax.scatter(
        distances_sorted,
        potentials_sorted,
        c="steelblue",
        alpha=0.3,
        s=1,
        label="Computed lead field",
    )
    
    # Plot theoretical curve (1/r decay)
    r_theory = np.linspace(SINGULARITY_THRESHOLD_MM, distances_sorted.max(), 100)
    r_theory_m = r_theory * 1e-3  # Convert to meters
    v_theory = 1.0 / (4.0 * np.pi * BRAIN_CONDUCTIVITY_S_M * r_theory_m)
    
    ax.plot(
        r_theory,
        v_theory,
        "r-",
        linewidth=2,
        label=r"Theory: $V = \frac{1}{4\pi\sigma r}$",
    )
    
    # Mark singularity threshold
    ax.axvline(
        SINGULARITY_THRESHOLD_MM,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Clamping threshold ({SINGULARITY_THRESHOLD_MM} mm)",
    )
    
    ax.set_xlabel("Distance to source (mm)")
    ax.set_ylabel("Lead field value (V/A)")
    ax.set_title(f"Lead Field Validation - Source {['A', 'B', 'C'][source_index]}\n(1/r decay verification)")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    
    return ax


def print_validation_summary(
    sensors: np.ndarray,
    sources: np.ndarray,
    lead_field: np.ndarray,
    singularity_mask: np.ndarray,
) -> None:
    """Print validation summary to console."""
    print("=" * 60)
    print("PHASE 1 VALIDATION SUMMARY")
    print("=" * 60)
    
    # Sensor bounds
    print("\n[Sensor Cloud]")
    print(f"  Count: {len(sensors)}")
    print(f"  X range: [{sensors[:, 0].min():.4f}, {sensors[:, 0].max():.4f}] mm")
    print(f"  Y range: [{sensors[:, 1].min():.4f}, {sensors[:, 1].max():.4f}] mm")
    print(f"  Z range: [{sensors[:, 2].min():.4f}, {sensors[:, 2].max():.4f}] mm")
    print(f"  Mean: {sensors.mean(axis=0)}")
    
    # Source info
    print("\n[Neural Sources]")
    source_labels = ["A", "B", "C"]
    for i, (src, label) in enumerate(zip(sources, source_labels)):
        print(f"  Source {label}: [{src[0]:+.2f}, {src[1]:+.2f}, {src[2]:+.2f}] mm")
    
    # Singularity analysis
    print("\n[Singularity Analysis]")
    n_in_zone = np.sum(np.any(singularity_mask, axis=1))
    print(f"  Threshold: {SINGULARITY_THRESHOLD_MM} mm")
    print(f"  Sensors in ANY exclusion zone: {n_in_zone}")
    for i, label in enumerate(source_labels):
        n_per_source = np.sum(singularity_mask[:, i])
        print(f"  Sensors near Source {label}: {n_per_source}")
    
    # Lead field validation
    print("\n[Lead Field]")
    info = validate_lead_field(lead_field, expected_shape=(len(sensors), len(sources)))
    print(f"  Shape: {info['shape']}")
    print(f"  Valid: {info['is_valid']}")
    print(f"  Has infinities: {info['has_infinities']}")
    print(f"  Has NaNs: {info['has_nans']}")
    print(f"  Min value: {info['min_value']:.2e} V/A")
    print(f"  Max value: {info['max_value']:.2e} V/A")
    
    if info["errors"]:
        print(f"  ERRORS: {info['errors']}")
    
    # Analytical validation: check one point manually
    print("\n[Analytical Spot Check]")
    # Pick a sensor far from sources for clean comparison
    distances = compute_distance_matrix(sensors, sources)
    far_idx = np.argmax(distances[:, 0])  # Farthest from Source A
    r_mm = distances[far_idx, 0]
    r_m = r_mm * 1e-3
    v_computed = lead_field[far_idx, 0]
    v_expected = 1.0 / (4.0 * np.pi * BRAIN_CONDUCTIVITY_S_M * r_m)
    rel_error = abs(v_computed - v_expected) / v_expected * 100
    print(f"  Sensor {far_idx} (r={r_mm:.3f} mm from Source A):")
    print(f"    Computed: {v_computed:.4e} V/A")
    print(f"    Expected: {v_expected:.4e} V/A")
    print(f"    Relative error: {rel_error:.4f}%")
    
    print("\n" + "=" * 60)


def main() -> None:
    """Run all visualizations and validation."""
    print("Loading data...")
    sensors, sources = load_data()
    
    print("Computing lead field...")
    lead_field, singularity_mask = compute_lead_field(sensors, sources)
    
    # Print validation summary
    print_validation_summary(sensors, sources, lead_field, singularity_mask)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    plot_cloud_and_sources(sensors, sources, singularity_mask, ax=ax1)
    
    # Distance histogram
    ax2 = fig.add_subplot(2, 2, 2)
    plot_distance_histogram(sensors, sources, ax=ax2)
    
    # Lead field validation for Source A
    ax3 = fig.add_subplot(2, 2, 3)
    plot_lead_field_validation(lead_field, sensors, sources, source_index=0, ax=ax3)
    
    # Lead field validation for Source B
    ax4 = fig.add_subplot(2, 2, 4)
    plot_lead_field_validation(lead_field, sensors, sources, source_index=1, ax=ax4)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase1_validation.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")
    
    # Show plot (non-blocking in script mode)
    # Comment out for headless execution
    # plt.show()


if __name__ == "__main__":
    main()

