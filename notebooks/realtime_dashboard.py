"""
Subsense Real-Time BCI Dashboard - Phase 4 Online Decoding

Animated HUD displaying real-time neural decoding from the 10,000-sensor
nanoparticle cloud. Uses matplotlib animation to show:

- Top: Rolling window of raw sensor signals (subset for performance)
- Bottom: Recovered neural intent (Alpha, Beta, Pink Noise sources)
- HUD: Current latency, frame rate, and decoding status

Performance optimizations:
- Blitting enabled for GPU-accelerated rendering
- Pre-allocated numpy arrays instead of deques
- Batch buffer updates instead of sample-by-sample
- Reduced HUD update frequency

Run from project root:
    python notebooks/realtime_dashboard.py
"""

from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy import stats

from subsense_bci.visualization.theme import COLORS, apply_dark_theme, setup_axis_style
from subsense_bci.simulation.streamer import DataStreamer
from subsense_bci.filtering.online_decoder import OnlineDecoder
from subsense_bci.config import load_config


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


class RealtimeDashboard:
    """
    Animated real-time BCI dashboard with optimized rendering.

    Parameters
    ----------
    chunk_size_ms : float
        Size of each processing chunk in milliseconds.
    window_ms : float
        Width of the rolling display window in milliseconds.
    display_sensors : int
        Number of sensors to display (subset for performance).
    interval_ms : int
        Animation update interval in milliseconds.
    """

    def __init__(
        self,
        chunk_size_ms: float = 100.0,
        window_ms: float = 500.0,
        display_sensors: int = 50,
        interval_ms: int = 50,
    ) -> None:
        self.chunk_size_ms = chunk_size_ms
        self.window_ms = window_ms
        self.display_sensors = display_sensors
        self.interval_ms = interval_ms

        # Load config
        config = load_config()
        self.sampling_rate_hz = config["temporal"]["sampling_rate_hz"]

        # Calculate buffer sizes
        self.window_samples = int(window_ms * self.sampling_rate_hz / 1000.0)
        self.chunk_samples = int(chunk_size_ms * self.sampling_rate_hz / 1000.0)

        # Initialize streamer and decoder
        print("Initializing components...")
        self.streamer = DataStreamer()
        self.decoder = OnlineDecoder.from_phase3_data()

        # Load ground truth sources for correlation calculation
        data_dir = get_project_root() / "data" / "raw"
        self.ground_truth_sources = np.load(data_dir / "source_waveforms.npy")  # (3, n_samples)
        print(f"  Ground truth loaded: {self.ground_truth_sources.shape}")

        # Select subset of sensors for display
        np.random.seed(999)
        self.sensor_indices = np.sort(
            np.random.choice(self.streamer.n_sensors, size=display_sensors, replace=False)
        )

        # Pre-allocated rolling buffers (much faster than deques)
        self.sensor_buffer = np.zeros((self.window_samples, display_sensors))
        self.source_buffer = np.zeros((self.window_samples, 3))
        self.ground_truth_buffer = np.zeros((self.window_samples, 3))  # For correlation
        self.time_buffer = np.zeros(self.window_samples)
        self.buffer_idx = 0  # Current write position

        # Correlation tracking
        self.correlations = np.array([0.0, 0.0, 0.0])  # Alpha, Beta, Pink

        # Performance metrics (use simple arrays)
        self.latency_history = np.zeros(50)
        self.latency_idx = 0
        self.frame_time_history = np.zeros(30)
        self.frame_time_idx = 0
        self.last_frame_time = time.perf_counter()
        self.current_timestamp = 0.0
        self.frame_count = 0

        # Pre-compute sensor offsets for stacking
        self.sensor_offsets = np.linspace(2, -2, display_sensors)
        self.source_offsets = np.array([2.0, 0.0, -2.0])

    def setup_figure(self) -> tuple:
        """Create and configure the dashboard figure."""
        apply_dark_theme()

        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor(COLORS["background"])

        # Create grid with MORE vertical space at top for title
        gs = self.fig.add_gridspec(
            2, 2,
            width_ratios=[4, 1],
            height_ratios=[1, 1],
            hspace=0.30,       # Increased from 0.25
            wspace=0.15,
            left=0.06,
            right=0.95,
            top=0.88,          # Lowered from 0.92 to give title more room
            bottom=0.08,
        )

        # Sensor display (top left)
        self.ax_sensors = self.fig.add_subplot(gs[0, 0])
        setup_axis_style(self.ax_sensors, "RAW SENSOR STREAM")

        # Source display (bottom left)
        self.ax_sources = self.fig.add_subplot(gs[1, 0])
        setup_axis_style(self.ax_sources, "DECODED NEURAL INTENT")

        # HUD panel (right, spans both rows)
        self.ax_hud = self.fig.add_subplot(gs[:, 1])
        self.ax_hud.set_facecolor(COLORS["panel_bg"])
        self.ax_hud.axis("off")

        # Main title - moved up with more padding
        self.fig.suptitle(
            "SUBSENSE REAL-TIME NEURAL DECODER",
            fontsize=14,
            fontweight="bold",
            fontfamily="monospace",
            color=COLORS["text_accent"],
            y=0.96,            # Moved up from 0.97
        )

        # Initialize plot elements
        self._init_sensor_plot()
        self._init_source_plot()
        self._init_hud()

        # Collect all artists for blitting
        self.blit_artists = []
        self.blit_artists.extend([line for line, _ in self.sensor_lines])
        self.blit_artists.extend([line for line, _ in self.source_lines])
        self.blit_artists.extend(self.corr_labels)  # Correlation labels

        return self.fig,

    def _init_sensor_plot(self) -> None:
        """Initialize sensor waveform display."""
        self.sensor_lines = []

        # Pre-compute x-axis data (will be updated per frame)
        self.time_xdata = np.linspace(0, self.window_ms, self.window_samples)

        for i, offset in enumerate(self.sensor_offsets):
            line, = self.ax_sensors.plot(
                self.time_xdata,
                np.zeros(self.window_samples) + offset,
                color=COLORS["nanotech_cyan"],
                linewidth=0.4,
                alpha=0.6,
                animated=True,  # Enable for blitting
            )
            self.sensor_lines.append((line, offset))

        self.ax_sensors.set_xlim([0, self.window_ms])
        self.ax_sensors.set_ylim([-3, 3])
        self.ax_sensors.set_xlabel("Time (ms)", color=COLORS["text_secondary"], fontsize=9)
        self.ax_sensors.set_ylabel("Sensors (stacked)", color=COLORS["text_secondary"], fontsize=9)
        self.ax_sensors.set_yticks([])

    def _init_source_plot(self) -> None:
        """Initialize source waveform display with correlation labels."""
        source_colors = [COLORS["source_a"], COLORS["source_b"], COLORS["source_c"]]
        source_labels = ["Alpha (10Hz)", "Beta (20Hz)", "Pink Noise"]

        self.source_lines = []
        self.corr_labels = []  # Correlation text labels

        for i, (color, label, offset) in enumerate(zip(source_colors, source_labels, self.source_offsets)):
            line, = self.ax_sources.plot(
                self.time_xdata,
                np.zeros(self.window_samples) + offset,
                color=color,
                linewidth=1.5,
                alpha=0.9,
                label=label,
                animated=True,  # Enable for blitting
            )
            self.source_lines.append((line, offset))

            # Label on left
            self.ax_sources.text(
                -self.window_ms * 0.02, offset,
                label.split()[0],
                color=color,
                fontsize=9,
                fontfamily="monospace",
                fontweight="bold",
                va="center",
                ha="right",
            )

            # Correlation label on right (updated in real-time)
            corr_text = self.ax_sources.text(
                self.window_ms * 1.01, offset,
                "r=---",
                color=color,
                fontsize=9,
                fontfamily="monospace",
                fontweight="bold",
                va="center",
                ha="left",
                animated=True,
            )
            self.corr_labels.append(corr_text)

        self.ax_sources.set_xlim([0, self.window_ms])
        self.ax_sources.set_ylim([-4, 4])
        self.ax_sources.set_xlabel("Time (ms)", color=COLORS["text_secondary"], fontsize=9)
        self.ax_sources.set_yticks([])

    def _init_hud(self) -> None:
        """Initialize HUD panel."""
        ax = self.ax_hud
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

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
            0.5, 0.95,
            "SYSTEM STATUS",
            ha="center",
            fontsize=12,
            fontweight="bold",
            fontfamily="monospace",
            color=COLORS["text_accent"],
        )

        # Status indicator
        self.status_dot = ax.scatter(
            [0.15], [0.85],
            c=COLORS["success_green"],
            s=100,
            marker="o",
            edgecolors="white",
            linewidths=1,
            zorder=10,
        )
        self.status_text = ax.text(
            0.25, 0.85,
            "DECODING",
            va="center",
            fontsize=10,
            fontfamily="monospace",
            fontweight="bold",
            color=COLORS["success_green"],
        )

        # Divider
        ax.plot([0.1, 0.9], [0.78, 0.78], color=COLORS["grid_line"], linewidth=1)

        # Metric labels (static) - Performance section
        metrics_y = [0.68, 0.58, 0.48, 0.38]
        metric_labels = ["LATENCY", "FPS", "TIMESTAMP", "RT FACTOR"]

        for y, label in zip(metrics_y, metric_labels):
            ax.text(
                0.12, y,
                f"{label}:",
                fontsize=9,
                fontfamily="monospace",
                color=COLORS["text_secondary"],
            )

        # Metric values (will be updated)
        self.metric_texts = {}
        for y, key in zip(metrics_y, ["latency", "fps", "timestamp", "rt_factor"]):
            self.metric_texts[key] = ax.text(
                0.88, y,
                "---",
                ha="right",
                fontsize=10,
                fontfamily="monospace",
                fontweight="bold",
                color=COLORS["text_primary"],
                animated=True,
            )

        # Divider for source recovery section
        ax.plot([0.1, 0.9], [0.30, 0.30], color=COLORS["grid_line"], linewidth=1)

        # Source recovery quality section title
        ax.text(
            0.5, 0.25,
            "SOURCE RECOVERY",
            ha="center",
            fontsize=9,
            fontweight="bold",
            fontfamily="monospace",
            color=COLORS["text_accent"],
        )

        # Average correlation metric
        ax.text(
            0.12, 0.17,
            "AVG CORR:",
            fontsize=9,
            fontfamily="monospace",
            color=COLORS["text_secondary"],
        )
        self.metric_texts["avg_corr"] = ax.text(
            0.88, 0.17,
            "---",
            ha="right",
            fontsize=10,
            fontfamily="monospace",
            fontweight="bold",
            color=COLORS["text_primary"],
            animated=True,
        )

        # Progress bar
        ax.plot([0.1, 0.9], [0.08, 0.08], color=COLORS["grid_line"], linewidth=4)
        self.progress_line, = ax.plot(
            [0.1, 0.1], [0.08, 0.08],
            color=COLORS["nanotech_cyan"],
            linewidth=4,
            animated=True,
        )

        # Add HUD elements to blit list
        self.hud_artists = list(self.metric_texts.values()) + [self.progress_line]

    def _update_hud(self, timestamp: float, latency_ms: float) -> None:
        """Update HUD metric displays (called every few frames for performance)."""
        # Calculate FPS from history
        avg_frame_time = np.mean(self.frame_time_history)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        # Calculate real-time factor
        avg_latency = np.mean(self.latency_history)
        rt_factor = self.chunk_size_ms / avg_latency if avg_latency > 0 else 0

        # Update text
        self.metric_texts["latency"].set_text(f"{latency_ms:.1f} ms")
        self.metric_texts["latency"].set_color(
            COLORS["success_green"] if latency_ms < self.chunk_size_ms else COLORS["warning_red"]
        )

        self.metric_texts["fps"].set_text(f"{fps:.1f}")
        self.metric_texts["timestamp"].set_text(f"{timestamp:.2f} s")

        self.metric_texts["rt_factor"].set_text(f"{rt_factor:.1f}x")
        self.metric_texts["rt_factor"].set_color(
            COLORS["success_green"] if rt_factor > 1.0 else COLORS["warning_red"]
        )

        # Update average correlation
        avg_corr = np.mean(self.correlations)
        self.metric_texts["avg_corr"].set_text(f"r={avg_corr:.3f}")
        self.metric_texts["avg_corr"].set_color(
            COLORS["success_green"] if avg_corr > 0.9 else
            COLORS["text_accent"] if avg_corr > 0.7 else COLORS["warning_red"]
        )

        # Update progress bar
        progress = timestamp / self.streamer.duration_sec
        progress = min(1.0, max(0.0, progress))
        self.progress_line.set_xdata([0.1, 0.1 + 0.8 * progress])

    def animate(self, frame: int) -> list:
        """Animation update function - optimized for performance."""
        # Track frame timing
        current_time = time.perf_counter()
        frame_delta = current_time - self.last_frame_time
        self.last_frame_time = current_time
        self.frame_time_history[self.frame_time_idx % len(self.frame_time_history)] = frame_delta
        self.frame_time_idx += 1
        self.frame_count += 1

        # Get next chunk
        result = self.streamer.get_next_chunk(self.chunk_size_ms)

        if result is None:
            # End of stream - restart
            self.streamer.reset()
            result = self.streamer.get_next_chunk(self.chunk_size_ms)

        chunk, timestamp = result
        self.current_timestamp = timestamp

        # Decode chunk
        decode_result = self.decoder.decode(chunk, timestamp)

        # Store latency
        self.latency_history[self.latency_idx % len(self.latency_history)] = decode_result.latency_ms
        self.latency_idx += 1

        # ===== OPTIMIZED BUFFER UPDATE (batch operations) =====
        chunk_samples = chunk.shape[1]

        # Get sensor data for display (batch operation)
        sensor_chunk = chunk[self.sensor_indices, :].T  # (chunk_samples, display_sensors)
        sensor_std = np.std(sensor_chunk)
        if sensor_std > 1e-6:
            sensor_chunk = sensor_chunk / sensor_std * 0.3

        # Get source data (batch operation)
        source_chunk = decode_result.sources.T  # (chunk_samples, 3)
        source_std = np.std(source_chunk)
        if source_std > 1e-6:
            source_chunk = source_chunk / source_std

        # Get corresponding ground truth samples for correlation
        start_sample = int(timestamp * self.sampling_rate_hz)
        end_sample = min(start_sample + chunk_samples, self.ground_truth_sources.shape[1])
        actual_samples = end_sample - start_sample

        if actual_samples > 0:
            gt_chunk = self.ground_truth_sources[:, start_sample:end_sample].T  # (chunk_samples, 3)
        else:
            gt_chunk = np.zeros((chunk_samples, 3))

        # Roll buffers left and insert new data at end (vectorized)
        self.sensor_buffer = np.roll(self.sensor_buffer, -chunk_samples, axis=0)
        self.sensor_buffer[-chunk_samples:] = sensor_chunk

        self.source_buffer = np.roll(self.source_buffer, -chunk_samples, axis=0)
        self.source_buffer[-chunk_samples:] = source_chunk

        self.ground_truth_buffer = np.roll(self.ground_truth_buffer, -chunk_samples, axis=0)
        if actual_samples > 0:
            self.ground_truth_buffer[-actual_samples:] = gt_chunk[:actual_samples]

        # ===== COMPUTE CORRELATIONS (every frame for accuracy) =====
        for i in range(3):
            decoded = self.source_buffer[:, i]
            gt = self.ground_truth_buffer[:, i]
            # Only compute if there's variance in both signals
            if np.std(decoded) > 1e-6 and np.std(gt) > 1e-6:
                r, _ = stats.pearsonr(decoded, gt)
                self.correlations[i] = abs(r)  # Use absolute value (sign ambiguity in ICA)

        # ===== UPDATE PLOT LINES (vectorized) =====
        # Update all sensor lines
        for i, (line, offset) in enumerate(self.sensor_lines):
            line.set_ydata(self.sensor_buffer[:, i] + offset)

        # Update all source lines
        for i, (line, offset) in enumerate(self.source_lines):
            line.set_ydata(self.source_buffer[:, i] + offset)

        # Update correlation labels
        source_names = ["Alpha", "Beta", "Pink"]
        for i, (corr_label, name) in enumerate(zip(self.corr_labels, source_names)):
            corr_label.set_text(f"r={self.correlations[i]:.2f}")

        # Update HUD every 3 frames (reduces overhead)
        if self.frame_count % 3 == 0:
            self._update_hud(timestamp, decode_result.latency_ms)

        # Return artists for blitting
        return self.blit_artists + self.hud_artists

    def run(self, save_path: Path | str | None = None) -> None:
        """
        Run the real-time dashboard.

        Parameters
        ----------
        save_path : Path, optional
            If provided, saves animation to this path instead of displaying.
        """
        print("\n" + "=" * 60)
        print("  SUBSENSE REAL-TIME DASHBOARD")
        print("=" * 60)
        print(f"\n  Chunk size: {self.chunk_size_ms}ms")
        print(f"  Window: {self.window_ms}ms")
        print(f"  Display sensors: {self.display_sensors}")
        print(f"  Target FPS: {1000 / self.interval_ms:.0f}")
        print(f"\n  Starting animation (with blitting for performance)...")

        self.setup_figure()

        # Create animation with blitting ENABLED for smooth performance
        anim = animation.FuncAnimation(
            self.fig,
            self.animate,
            interval=self.interval_ms,
            blit=True,           # ENABLED - major performance boost
            cache_frame_data=False,
            save_count=200,
        )

        if save_path is not None:
            # Save animation
            print(f"  Saving to {save_path}...")
            writer = animation.PillowWriter(fps=20)
            anim.save(save_path, writer=writer)
            print("  Saved!")
        else:
            # Show interactive
            plt.show()

        print("\n" + "=" * 60)


def main() -> None:
    """Run the real-time dashboard."""
    # Load config for Phase 4 parameters
    config = load_config()
    rt_config = config.get("realtime", {})
    chunk_size_ms = rt_config.get("chunk_size_ms", 100.0)
    window_ms = rt_config.get("window_ms", 500.0)
    display_sensors = rt_config.get("display_sensors", 50)
    interval_ms = rt_config.get("animation_interval_ms", 50)

    dashboard = RealtimeDashboard(
        chunk_size_ms=chunk_size_ms,
        window_ms=window_ms,
        display_sensors=display_sensors,
        interval_ms=interval_ms,
    )

    # Run interactively
    dashboard.run()

    # Or save a preview GIF (uncomment to use):
    # output_dir = get_project_root() / "data" / "processed"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # dashboard.run(save_path=output_dir / "phase4_realtime_preview.gif")


if __name__ == "__main__":
    main()
