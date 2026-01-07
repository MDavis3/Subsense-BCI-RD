"""
Data Streamer - Phase 4 Real-Time Simulation

Simulates a real-time data stream by loading pre-recorded sensor data
and yielding it in configurable chunks, mimicking the behavior of
a live neural recording system.

Usage:
    from subsense_bci.simulation.streamer import DataStreamer, StreamPacket

    streamer = DataStreamer()
    for packet in streamer.get_chunks(chunk_size_ms=100):
        # Process chunk in real-time
        decoded = decoder.decode(packet.chunk, packet.timestamp, packet.reference)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator
import time

import numpy as np

from subsense_bci.config import load_config


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent.parent


@dataclass
class StreamPacket:
    """
    Container for streamed neural data.

    Encapsulates a single chunk of data from the DataStreamer, including
    the neural recording, timestamp, and optional reference signal for
    artifact rejection.

    Using a dataclass instead of a tuple provides:
    1. Named attribute access (packet.chunk vs packet[0])
    2. Future extensibility without breaking changes
    3. Self-documenting code

    Attributes
    ----------
    chunk : np.ndarray
        Sensor data chunk with shape (n_sensors, chunk_samples).
    timestamp : float
        Timestamp of chunk start in seconds from recording start.
    reference : np.ndarray, optional
        Reference signal for artifact rejection (e.g., PPG/ECG).
        Shape (chunk_samples,) or None if not available.

    Examples
    --------
    >>> packet = StreamPacket(chunk=np.zeros((10, 100)), timestamp=0.5)
    >>> packet.chunk.shape
    (10, 100)
    >>> packet.reference is None
    True
    """

    chunk: np.ndarray
    timestamp: float
    reference: np.ndarray | None = None


class DataStreamer:
    """
    Streams pre-recorded sensor data in real-time chunks.
    
    Simulates a live neural recording by yielding data in fixed-size
    temporal windows, respecting the original sampling rate.
    
    Parameters
    ----------
    recording_path : Path, optional
        Path to recording .npy file. Defaults to data/raw/recording_simulation.npy.
    sampling_rate_hz : float, optional
        Sampling rate of the recording. Loaded from config if not specified.
    simulate_realtime : bool, optional
        If True, adds actual time delays between chunks to simulate real-time.
        Default is False (process as fast as possible).
        
    Attributes
    ----------
    recording : np.ndarray
        Full sensor recording, shape (n_sensors, n_samples).
    n_sensors : int
        Number of sensors in the recording.
    n_samples : int
        Total number of time samples.
    sampling_rate_hz : float
        Sampling rate in Hz.
    duration_sec : float
        Total recording duration in seconds.
    """
    
    def __init__(
        self,
        recording_path: Path | str | None = None,
        reference_path: Path | str | None = None,
        sampling_rate_hz: float | None = None,
        simulate_realtime: bool = False,
    ) -> None:
        # Load configuration
        config = load_config()

        # Set sampling rate
        if sampling_rate_hz is None:
            self.sampling_rate_hz = config["temporal"]["sampling_rate_hz"]
        else:
            self.sampling_rate_hz = sampling_rate_hz

        # Load recording data
        if recording_path is None:
            recording_path = get_project_root() / "data" / "raw" / "recording_simulation.npy"
        else:
            recording_path = Path(recording_path)

        self.recording = np.load(recording_path)
        self.n_sensors, self.n_samples = self.recording.shape
        self.duration_sec = self.n_samples / self.sampling_rate_hz

        # Load reference signal if provided (for artifact rejection)
        self.reference: np.ndarray | None = None
        if reference_path is not None:
            reference_path = Path(reference_path)
            self.reference = np.load(reference_path)
            # Validate reference length matches recording
            if len(self.reference) != self.n_samples:
                raise ValueError(
                    f"Reference length ({len(self.reference)}) must match "
                    f"recording samples ({self.n_samples})"
                )

        # Real-time simulation flag
        self.simulate_realtime = simulate_realtime

        # Internal state
        self._current_sample = 0
    
    def reset(self) -> None:
        """Reset streamer to beginning of recording."""
        self._current_sample = 0
    
    def ms_to_samples(self, ms: float) -> int:
        """Convert milliseconds to number of samples."""
        return int(ms * self.sampling_rate_hz / 1000.0)
    
    def samples_to_ms(self, samples: int) -> float:
        """Convert number of samples to milliseconds."""
        return samples * 1000.0 / self.sampling_rate_hz
    
    def get_next_chunk(
        self,
        chunk_size_ms: float = 100.0,
    ) -> StreamPacket | None:
        """
        Get the next chunk of data from the stream.

        Parameters
        ----------
        chunk_size_ms : float
            Size of each chunk in milliseconds. Default is 100ms.

        Returns
        -------
        StreamPacket | None
            StreamPacket containing chunk data, timestamp, and optional reference,
            or None if stream exhausted.
        """
        chunk_samples = self.ms_to_samples(chunk_size_ms)

        # Check if we've exhausted the recording
        if self._current_sample >= self.n_samples:
            return None

        # Calculate end index (may be truncated at end of recording)
        end_sample = min(self._current_sample + chunk_samples, self.n_samples)

        # Extract chunk
        chunk = self.recording[:, self._current_sample:end_sample]
        timestamp = self._current_sample / self.sampling_rate_hz

        # Extract reference chunk if available
        ref_chunk = None
        if self.reference is not None:
            ref_chunk = self.reference[self._current_sample:end_sample]

        # Simulate real-time delay if requested
        if self.simulate_realtime:
            actual_chunk_ms = self.samples_to_ms(end_sample - self._current_sample)
            time.sleep(actual_chunk_ms / 1000.0)

        # Advance position
        self._current_sample = end_sample

        return StreamPacket(chunk=chunk, timestamp=timestamp, reference=ref_chunk)
    
    def get_chunks(
        self,
        chunk_size_ms: float = 100.0,
    ) -> Generator[StreamPacket, None, None]:
        """
        Generator that yields all chunks from the recording.

        Parameters
        ----------
        chunk_size_ms : float
            Size of each chunk in milliseconds. Default is 100ms.

        Yields
        ------
        StreamPacket
            StreamPacket containing chunk data, timestamp, and optional reference.

        Example
        -------
        >>> streamer = DataStreamer()
        >>> for packet in streamer.get_chunks(chunk_size_ms=100):
        ...     print(f"t={packet.timestamp:.3f}s: chunk shape {packet.chunk.shape}")
        """
        self.reset()

        while True:
            packet = self.get_next_chunk(chunk_size_ms)
            if packet is None:
                break
            yield packet
    
    def get_chunk_count(self, chunk_size_ms: float = 100.0) -> int:
        """
        Calculate total number of chunks for given chunk size.
        
        Parameters
        ----------
        chunk_size_ms : float
            Size of each chunk in milliseconds.
            
        Returns
        -------
        int
            Total number of chunks (including potentially truncated final chunk).
        """
        chunk_samples = self.ms_to_samples(chunk_size_ms)
        return int(np.ceil(self.n_samples / chunk_samples))
    
    def __repr__(self) -> str:
        return (
            f"DataStreamer("
            f"n_sensors={self.n_sensors}, "
            f"duration={self.duration_sec:.2f}s, "
            f"fs={self.sampling_rate_hz:.0f}Hz)"
        )


def main() -> None:
    """Demo the DataStreamer."""
    print("=" * 60)
    print("  PHASE 4: DataStreamer Demo")
    print("=" * 60)

    streamer = DataStreamer()
    print(f"\nLoaded: {streamer}")

    chunk_size_ms = 100.0
    n_chunks = streamer.get_chunk_count(chunk_size_ms)
    print(f"\nChunk size: {chunk_size_ms}ms")
    print(f"Total chunks: {n_chunks}")

    print("\nStreaming first 5 chunks:")
    for i, packet in enumerate(streamer.get_chunks(chunk_size_ms)):
        ref_info = "with ref" if packet.reference is not None else "no ref"
        print(f"  [{i+1}/{n_chunks}] t={packet.timestamp:.3f}s | shape={packet.chunk.shape} | {ref_info}")
        if i >= 4:
            print("  ...")
            break

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

