"""
Subsense Real-Time BCI Dashboard - Phase 4 Online Decoding

Animated HUD displaying real-time neural decoding from the 10,000-sensor
nanoparticle cloud. Uses matplotlib animation to show:

- Top: Rolling window of raw sensor signals (subset for performance)
- Bottom: Recovered neural intent (Alpha, Beta, Pink Noise sources)
- HUD: Current latency, frame rate, and decoding status

Run from project root:
    python notebooks/realtime_dashboard.py
"""

from __future__ import annotations

from pathlib import Path
from collections import deque
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
import numpy as np

from subsense_bci.visualization.theme import COLORS, apply_dark_theme, setup_axis_style
from subsense_bci.simulation.streamer import DataStreamer
from subsense_bci.filtering.online_decoder import OnlineDecoder
from subsense_bci.config import load_config


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


class RealtimeDashboard:
    """
    Animated real-time BCI dashboard.
    
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
        
        # Select subset of sensors for display
        np.random.seed(999)
        self.sensor_indices = sorted(
            np.random.choice(self.streamer.n_sensors, size=display_sensors, replace=False)
        )
        
        # Rolling buffers for display
        self.sensor_buffer = deque(maxlen=self.window_samples)
        self.source_buffer = deque(maxlen=self.window_samples)
        self.time_buffer = deque(maxlen=self.window_samples)
        
        # Initialize with zeros
        for _ in range(self.window_samples):
            self.sensor_buffer.append(np.zeros(display_sensors))
            self.source_buffer.append(np.zeros(3))
            self.time_buffer.append(0.0)
        
        # Performance metrics
        self.latencies = deque(maxlen=100)
        self.frame_times = deque(maxlen=100)
        self.last_frame_time = time.perf_counter()
        self.current_timestamp = 0.0
        
        # Animation state
        self.chunk_generator = None
        self.is_running = False
        
    def setup_figure(self) -> tuple:
        """Create and configure the dashboard figure."""
        apply_dark_theme()
        
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor(COLORS["background"])
        
        # Create grid: sensors (top), sources (bottom), HUD (right)
        gs = self.fig.add_gridspec(
            2, 2,
            width_ratios=[4, 1],
            height_ratios=[1, 1],
            hspace=0.25,
            wspace=0.15,
            left=0.06,
            right=0.95,
            top=0.92,
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
        
        # Main title
        self.fig.suptitle(
            "SUBSENSE REAL-TIME NEURAL DECODER",
            fontsize=14,
            fontweight="bold",
            fontfamily="monospace",
            color=COLORS["text_accent"],
            y=0.97,
        )
        
        # Initialize plot elements
        self._init_sensor_plot()
        self._init_source_plot()
        self._init_hud()
        
        return self.fig,
    
    def _init_sensor_plot(self) -> None:
        """Initialize sensor waveform display."""
        # Create line for each displayed sensor
        self.sensor_lines = []
        
        # Offset each sensor for visibility
        offsets = np.linspace(2, -2, self.display_sensors)
        
        t = np.zeros(self.window_samples)
        for i, offset in enumerate(offsets):
            line, = self.ax_sensors.plot(
                t, np.zeros(self.window_samples) + offset,
                color=COLORS["nanotech_cyan"],
                linewidth=0.3,
                alpha=0.5,
            )
            self.sensor_lines.append((line, offset))
        
        self.ax_sensors.set_xlim([0, self.window_ms])
        self.ax_sensors.set_ylim([-3, 3])
        self.ax_sensors.set_xlabel("Time (ms)", color=COLORS["text_secondary"], fontsize=9)
        self.ax_sensors.set_ylabel("Sensors (stacked)", color=COLORS["text_secondary"], fontsize=9)
        self.ax_sensors.set_yticks([])
    
    def _init_source_plot(self) -> None:
        """Initialize source waveform display."""
        source_colors = [COLORS["source_a"], COLORS["source_b"], COLORS["source_c"]]
        source_labels = ["Alpha (10Hz)", "Beta (20Hz)", "Pink Noise"]
        offsets = [2, 0, -2]
        
        self.source_lines = []
        t = np.zeros(self.window_samples)
        
        for i, (color, label, offset) in enumerate(zip(source_colors, source_labels, offsets)):
            line, = self.ax_sources.plot(
                t, np.zeros(self.window_samples) + offset,
                color=color,
                linewidth=1.5,
                alpha=0.9,
                label=label,
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
        
        # Metric labels (static)
        metrics_y = [0.68, 0.56, 0.44, 0.32, 0.20]
        metric_labels = ["LATENCY", "FPS", "TIMESTAMP", "CHUNK", "RT FACTOR"]
        
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
        for y, key in zip(metrics_y, ["latency", "fps", "timestamp", "chunk", "rt_factor"]):
            self.metric_texts[key] = ax.text(
                0.88, y,
                "---",
                ha="right",
                fontsize=10,
                fontfamily="monospace",
                fontweight="bold",
                color=COLORS["text_primary"],
            )
        
        # Progress bar
        ax.plot([0.1, 0.9], [0.08, 0.08], color=COLORS["grid_line"], linewidth=4)
        self.progress_line, = ax.plot(
            [0.1, 0.1], [0.08, 0.08],
            color=COLORS["nanotech_cyan"],
            linewidth=4,
        )
    
    def _update_hud(self, timestamp: float, latency_ms: float) -> None:
        """Update HUD metric displays."""
        # Calculate FPS
        current_time = time.perf_counter()
        frame_delta = current_time - self.last_frame_time
        self.last_frame_time = current_time
        self.frame_times.append(frame_delta)
        
        avg_frame_time = np.mean(self.frame_times) if self.frame_times else 0.05
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Calculate real-time factor
        avg_latency = np.mean(self.latencies) if self.latencies else latency_ms
        rt_factor = self.chunk_size_ms / avg_latency if avg_latency > 0 else 0
        
        # Update text
        self.metric_texts["latency"].set_text(f"{latency_ms:.1f} ms")
        self.metric_texts["latency"].set_color(
            COLORS["success_green"] if latency_ms < self.chunk_size_ms else COLORS["warning_red"]
        )
        
        self.metric_texts["fps"].set_text(f"{fps:.1f}")
        self.metric_texts["timestamp"].set_text(f"{timestamp:.2f} s")
        self.metric_texts["chunk"].set_text(f"{self.chunk_size_ms:.0f} ms")
        
        self.metric_texts["rt_factor"].set_text(f"{rt_factor:.1f}x")
        self.metric_texts["rt_factor"].set_color(
            COLORS["success_green"] if rt_factor > 1.0 else COLORS["warning_red"]
        )
        
        # Update progress bar
        progress = timestamp / self.streamer.duration_sec
        progress = min(1.0, max(0.0, progress))
        self.progress_line.set_xdata([0.1, 0.1 + 0.8 * progress])
    
    def animate(self, frame: int) -> list:
        """Animation update function."""
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
        self.latencies.append(decode_result.latency_ms)
        
        # Update buffers with new data
        chunk_samples = chunk.shape[1]
        for i in range(chunk_samples):
            # Sensor data (subset, normalized)
            sensor_vals = chunk[self.sensor_indices, i]
            sensor_vals = sensor_vals / (np.std(sensor_vals) + 1e-6) * 0.3
            self.sensor_buffer.append(sensor_vals)
            
            # Source data (normalized for display)
            source_vals = decode_result.sources[:, i]
            source_vals = source_vals / (np.std(decode_result.sources) + 1e-6)
            self.source_buffer.append(source_vals)
            
            # Time
            sample_time = timestamp * 1000 + i * 1000 / self.sampling_rate_hz
            self.time_buffer.append(sample_time)
        
        # Convert buffers to arrays
        sensor_data = np.array(self.sensor_buffer)  # (window_samples, display_sensors)
        source_data = np.array(self.source_buffer)  # (window_samples, 3)
        time_data = np.array(self.time_buffer)
        
        # Normalize time to window
        time_data = time_data - time_data[-1] + self.window_ms
        
        # Update sensor lines
        for i, (line, offset) in enumerate(self.sensor_lines):
            line.set_xdata(time_data)
            line.set_ydata(sensor_data[:, i] + offset)
        
        # Update source lines
        for i, (line, offset) in enumerate(self.source_lines):
            line.set_xdata(time_data)
            line.set_ydata(source_data[:, i] + offset)
        
        # Update HUD
        self._update_hud(timestamp, decode_result.latency_ms)
        
        # Return all artists that were modified
        artists = [line for line, _ in self.sensor_lines]
        artists += [line for line, _ in self.source_lines]
        artists += list(self.metric_texts.values())
        artists += [self.progress_line]
        
        return artists
    
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
        print(f"\n  Starting animation...")
        
        self.setup_figure()
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig,
            self.animate,
            interval=self.interval_ms,
            blit=False,  # Full redraw for HUD updates
            cache_frame_data=False,
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
    chunk_size_ms = config.get("realtime", {}).get("chunk_size_ms", 100.0)
    window_ms = config.get("realtime", {}).get("window_ms", 500.0)
    
    dashboard = RealtimeDashboard(
        chunk_size_ms=chunk_size_ms,
        window_ms=window_ms,
        display_sensors=50,
        interval_ms=50,
    )
    
    # Run interactively
    dashboard.run()
    
    # Or save a preview GIF (uncomment to use):
    # output_dir = get_project_root() / "data" / "processed"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # dashboard.run(save_path=output_dir / "phase4_realtime_preview.gif")


if __name__ == "__main__":
    main()

