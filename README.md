# Subsense BCI R&D

Neural signal processing pipeline for magnetoelectric (ME) nanoparticle brain-computer interfaces.

## Overview

This repository implements a complete forward-inverse BCI simulation with real-time decoding:

1. **Phase 1: Sensor Cloud** — Generate 10,000 ME nanoparticle sensors in a 1mm³ volume
2. **Phase 2: Temporal Dynamics** — Simulate neural source waveforms (10Hz α, 20Hz β, pink noise) mixed through a physics-based lead field
3. **Phase 3: Source Unmixing** — Recover original sources via PCA dimensionality reduction + FastICA blind source separation
4. **Phase 4: Real-Time Decoding** — Online BCI pipeline with chunk-based streaming, achieving r=0.989 correlation at 42.7ms latency

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
│   └── validate_unmixing.py  # Phase 3 dashboard
├── src/subsense_bci/
│   ├── physics/              # Transfer functions, constants
│   ├── filtering/            # ICA, unmixing pipeline
│   ├── simulation/           # Cloud & waveform generators
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

The forward model follows volume conductor theory:

$$V = \frac{I}{4\pi\sigma r}$$

Where:
- σ = 0.33 S/m (brain tissue conductivity)
- r = sensor-source distance

The inverse problem uses ICA to exploit source independence:

$$\hat{S} = W \cdot X_{PCA}$$

Where W maximizes non-Gaussianity of recovered sources.

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Stochastic nanoparticle cloud | ✅ Complete |
| 2 | Temporal dynamics & mixing | ✅ Complete |
| 3 | PCA/ICA source unmixing | ✅ Complete |
| 4 | Real-time decoding | ✅ Complete (r=0.989, 42.7ms) |
| 5 | Hemodynamic artifact rejection | 🔜 Next |

## References

- Nunez & Srinivasan, "Electric Fields of the Brain" (2006)
- Hyvärinen & Oja, "Independent Component Analysis" (2000)
- Gabriel et al., 1996 — Tissue conductivity values

## License

MIT

