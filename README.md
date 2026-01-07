# Subsense BCI R&D

Neural signal processing pipeline for magnetoelectric (ME) nanoparticle brain-computer interfaces.

## Executive Summary

**The primary purpose of this project is to provide a mathematically rigorous, real-time proof of concept for the Subsense BCI architecture by solving the "Inverse Problem"** — the extraction of clean neural intent from a massive, noisy sensor cloud.

By simulating a 3D stochastic distribution of 10,000 magnetoelectric nanoparticles with biological physics like 1/r signal decay, the project demonstrates that **high-bandwidth communication is achievable even with non-stationary sensors**. The software successfully bridges the gap between raw hardware output and usable neural signals, achieving:

| Metric | Result |
|--------|--------|
| **Signal Recovery** | r = 0.989 correlation with ground truth |
| **Latency** | 42.7ms per 100ms chunk (perceptually instantaneous) |
| **Real-Time Factor** | 2.3× (processes faster than data arrives) |
| **Sensor Scale** | 10,000 sensors processed successfully |

This pipeline establishes a validated R&D foundation for addressing critical real-world challenges such as hemodynamic artifact rejection and intravascular sensor drift.

## Overview

This repository implements a complete forward-inverse BCI simulation with real-time decoding:

1. **Phase 1: Sensor Cloud** — Generate 10,000 ME nanoparticle sensors in a 1mm³ volume
2. **Phase 2: Temporal Dynamics** — Simulate neural source waveforms (10Hz α, 20Hz β, pink noise) mixed through a physics-based lead field
3. **Phase 3: Source Unmixing** — Recover original sources via PCA dimensionality reduction + FastICA blind source separation
4. **Phase 4: Real-Time Decoding** — Online BCI pipeline with chunk-based streaming, achieving r=0.989 correlation at 42.7ms latency

### The Inverse Problem

Traditional BCIs use fixed electrode arrays. Subsense proposes **fluidic magnetoelectric nanoparticles** distributed throughout brain tissue. This creates a unique challenge:

- **10,000+ sensors** distributed stochastically in 3D space
- Each sensor receives a **mixture** of all neural sources (weighted by 1/r distance)
- Goal: recover the **original neural intent** from this messy mixture

This project proves the inverse problem is solvable with PCA+ICA blind source separation.

## Installation

```bash
# Clone the repository
git clone https://github.com/MDavis3/Subsense-BCI-RD.git
cd Subsense-BCI-RD

# Install in editable mode (recommended for development)
pip install -e .

# Or install with all optional dependencies
pip install -e ".[full]"

# Or install with dev tools (pytest, black, ruff)
pip install -e ".[dev]"
```

## Quick Start

```python
from subsense_bci.physics.transfer_function import compute_lead_field
from subsense_bci.filtering.unmixing import unmix_sources
from subsense_bci.visualization.theme import COLORS, apply_dark_theme

# Compute lead field for sensor-source geometry
lead_field, singularity_mask = compute_lead_field(sensors, sources)

# Recover sources from mixed recording
result = unmix_sources(recording, ground_truth)
print(f"Recovery correlations: {result.matched_correlations}")
```

### Real-Time Decoding Example

```python
from subsense_bci.simulation.streamer import DataStreamer
from subsense_bci.filtering.online_decoder import OnlineDecoder

# Train decoder on full recording
decoder = OnlineDecoder.from_phase3_data()

# Stream and decode in real-time
streamer = DataStreamer()
for chunk, timestamp in streamer.get_chunks(chunk_size_ms=100):
    result = decoder.decode(chunk, timestamp)
    print(f"t={timestamp:.2f}s | latency={result.latency_ms:.1f}ms")
```

## Project Structure

```
subsense-bci-rd/
├── configs/
│   └── default_sim.yaml      # Tunable simulation parameters
├── data/
│   ├── raw/                  # Generated simulation data (.npy)
│   └── processed/            # Dashboard outputs (.png)
├── notebooks/
│   ├── visualize_cloud.py    # Phase 1 dashboard
│   ├── visualize_signals.py  # Phase 2 dashboard
│   ├── validate_unmixing.py  # Phase 3 dashboard
│   └── realtime_dashboard.py # Phase 4 real-time HUD
├── src/subsense_bci/
│   ├── physics/              # Transfer functions, constants
│   ├── filtering/            # ICA, unmixing, online decoder
│   ├── simulation/           # Cloud generators, streamer
│   └── visualization/        # Dark lab theme
├── tests/
│   ├── test_physics.py       # Lead field validation (17 tests)
│   └── test_unmixing.py      # ICA pipeline validation (9 tests)
├── pyproject.toml            # Package configuration
└── RD_LOG.md                 # Research decisions audit trail
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=subsense_bci --cov-report=term-missing
```

**Current status: 26 tests passing**

## Configuration

Simulation parameters are centralized in `configs/default_sim.yaml`:

```yaml
temporal:
  sampling_rate_hz: 1000.0
  duration_sec: 2.0
  snr_level: 5.0

cloud:
  sensor_count: 10000
  random_seed: 42

realtime:
  chunk_size_ms: 100.0
  window_ms: 500.0
```

Load configuration in code:

```python
from subsense_bci.config import load_config
cfg = load_config()
print(cfg["temporal"]["snr_level"])  # 5.0
```

## Generating Dashboards

```bash
# Phase 1: Sensor cloud visualization
python notebooks/visualize_cloud.py

# Phase 2: Source waveform mixing
python notebooks/visualize_signals.py

# Phase 3: ICA recovery validation
python notebooks/validate_unmixing.py

# Phase 4: Real-time decoding HUD
python notebooks/realtime_dashboard.py
```

## Key Physics

### Forward Model (Volume Conductor Theory)

$$V = \frac{I}{4\pi\sigma r}$$

Where:
- σ = 0.33 S/m (brain tissue conductivity)
- r = sensor-source distance

The **lead field matrix** L captures how each source contributes to each sensor:
- Shape: (10,000 sensors × 3 sources)
- Each entry L[i,j] = signal strength from source j at sensor i

### Inverse Model (Blind Source Separation)

$$\hat{S} = W \cdot X_{PCA}$$

Pipeline:
1. **PCA**: 10,000 sensors → ~50 principal components (99.9% variance)
2. **FastICA**: Maximize non-Gaussianity to find independent sources
3. **Hungarian Matching**: Resolve permutation/sign ambiguity

## Performance Benchmarks

### Latency Breakdown (100ms chunk, 10,000 sensors)

| Operation | Time |
|-----------|------|
| Data transpose | ~0.5ms |
| Centering | ~0.1ms |
| PCA projection | ~35ms |
| ICA unmixing | ~5ms |
| Source reordering | ~0.1ms |
| Correlation calc | ~2ms |
| **Total** | **~42.7ms** |

### Source Recovery Quality

| Source | Correlation | Quality |
|--------|-------------|---------|
| Alpha (10 Hz) | r = 0.9948 | Excellent |
| Beta (20 Hz) | r = 0.9876 | Excellent |
| Pink Noise | r = 0.9999 | Excellent |
| **Average** | **r = 0.989** | **Excellent** |

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Stochastic nanoparticle cloud | ✅ Complete |
| 2 | Temporal dynamics & mixing | ✅ Complete |
| 3 | PCA/ICA source unmixing | ✅ Complete |
| 4 | Real-time decoding | ✅ Complete (r=0.989, 42.7ms) |
| 5 | Hemodynamic artifact rejection | 🔜 Next |

## Current Assumptions

1. **Volume Conductor**: Homogeneous isotropic medium (σ = 0.33 S/m)
2. **Sensor Behavior**: Point-source potential receiver (Voltage ∝ 1/r)
3. **Artifacts**: Currently modeling stationary noise only
4. **Coordinate System**: MNI coordinates, origin at anterior commissure

See `RD_LOG.md` for detailed research decisions and mathematical derivations.

## References

- Nunez & Srinivasan, "Electric Fields of the Brain" (2006)
- Hyvärinen & Oja, "Independent Component Analysis" (2000)
- Hyvärinen, "Fast and Robust Fixed-Point Algorithms for ICA" (1999)
- Gabriel et al., 1996 — Tissue conductivity values
- Kuhn, "The Hungarian Method for the Assignment Problem" (1955)

## License

MIT
