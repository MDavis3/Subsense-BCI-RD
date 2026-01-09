# Subsense BCI R&D

Neural signal processing pipeline for magnetoelectric (ME) nanoparticle brain-computer interfaces.

---

## 🚀 Quick Start (2 minutes)

### Prerequisites
- Python 3.9+ 
- pip

### Step 1: Clone & Install

```bash
git clone https://github.com/MDavis3/Subsense-BCI-RD.git
cd Subsense-BCI-RD
pip install -e .
```

### Step 2: Run the Full Demo

```bash
# Option A: Run all 4 phases sequentially (generates visualizations)
python notebooks/visualize_cloud.py      # Phase 1: Sensor cloud
python notebooks/visualize_signals.py    # Phase 2: Signal mixing
python notebooks/validate_unmixing.py    # Phase 3: Source recovery
python notebooks/realtime_dashboard.py   # Phase 4: Real-time decoding
```

```bash
# Option B: Launch interactive Signal Bench dashboard
python run_demo.py
```

### Step 3: Verify It Works

After running Phase 3 (`validate_unmixing.py`), you should see:

```
UNMIXING RESULTS
================
Source A (10Hz Alpha): r = 0.9948 [Excellent]
Source B (20Hz Beta):  r = 0.9876 [Excellent]  
Source C (Pink Noise): r = 0.9999 [Excellent]
```

**That's it!** You've just recovered 3 neural signals from 10,000 noisy sensor measurements.

### What Each Phase Does

| Phase | Command | Output | Time |
|-------|---------|--------|------|
| 1 | `visualize_cloud.py` | 3D sensor positions + lead field | ~5s |
| 2 | `visualize_signals.py` | Mixed sensor recordings | ~3s |
| 3 | `validate_unmixing.py` | Recovered sources (r=0.989) | ~10s |
| 4 | `realtime_dashboard.py` | Animated real-time HUD | ~30s |

---

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

## Installation Options

### Basic Install (Recommended)
```bash
git clone https://github.com/MDavis3/Subsense-BCI-RD.git
cd Subsense-BCI-RD
pip install -e .
```

### With All Dependencies
```bash
pip install -e ".[full]"      # Includes Streamlit dashboard
pip install -e ".[dev]"       # Includes pytest, black, ruff
```

### Verify Installation
```bash
python -c "from subsense_bci.physics.transfer_function import compute_lead_field; print('✓ Installation successful')"
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: subsense_bci` | Run `pip install -e .` from the project root |
| `FileNotFoundError: sensors_N10000...` | Run Phase 1 first: `python notebooks/visualize_cloud.py` |
| Dashboard won't start | Install Streamlit: `pip install streamlit` |
| Tests fail | Ensure you're in project root and ran `pip install -e ".[dev]"` |

---

## Code Examples

### Basic: Compute Lead Field
```python
from subsense_bci.physics.transfer_function import compute_lead_field
from subsense_bci.filtering.unmixing import unmix_sources

# Compute lead field for sensor-source geometry
lead_field, singularity_mask = compute_lead_field(sensors, sources)

# Recover sources from mixed recording
result = unmix_sources(recording, ground_truth)
print(f"Recovery correlations: {result.matched_correlations}")
```

### Advanced: Real-Time Decoding
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

## R&D Signal Bench

The **Signal Bench** is an interactive Streamlit dashboard for exploring cardiac artifact rejection in BCI systems. Designed for non-software researchers (biologists, PIs) to experiment with parameters without writing code.

### Launching the Dashboard

```bash
# One-click demo with standard cardiac preset
python run_demo.py

# Or launch Streamlit directly
streamlit run src/subsense_bci/dashboard/streamlit_app.py

# Or via installed command (after pip install -e .)
subsense-bench
```

### Dashboard Features

- **Interactive Parameter Control**: Adjust sensor count (up to 10k), filter type (LMS/RLS/PhaseAwareRLS), pulse wave velocity, and movement noise scale in real-time
- **Side-by-Side Visualization**: Compare raw signal, cleaned signal, and residual error
- **Automatic Validation**: Nyquist theorem checks and real-time budget warnings
- **Preset Configurations**: Load "Standard Cardiac", "Exercise Stress", or "High-Density" scenarios with one click
- **Nanoparticle Drift Toggle**: Experimental placeholder for SubSense 2026 roadmap

### Theoretical Basis: Lead Field Gradient Model

Cardiac artifacts arise from hemodynamic pulsation displacing sensors. The artifact model uses the gradient of the lead field:

$$A(t) = \nabla L \cdot \delta r(t)$$

where:
- $\nabla L = -\hat{r} / (4\pi\sigma r^2)$ is the lead field gradient (V/A/mm)
- $\delta r(t)$ is sensor displacement from cardiac pulsation (~50 μm)

This linearized model is accurate for small displacements typical of intravascular hemodynamic drift.

### Signal Bench Performance Benchmarks

| Configuration | Sensors | Filter | Taps | Latency | Budget (43ms) |
|---------------|---------|--------|------|---------|---------------|
| Demo Preset | 1,000 | PhaseAwareRLS | 8 | ~3ms | ✅ OK |
| Medium Scale | 5,000 | RLS | 16 | ~25ms | ✅ OK |
| Full Scale | 10,000 | RLS | 8 | ~15ms | ✅ OK |
| Full Scale | 10,000 | RLS | 32 | ~48ms | ⚠️ WARNING |

### Common Researcher Errors

1. **Nyquist Violation**: Ensure `sampling_rate > 2 × max(source_frequencies)`. The dashboard validates this automatically and provides corrective suggestions.

2. **Forgetting Factor Mismatch**: Use λ=0.95 for non-stationary artifacts (cardiac), λ=0.99 for stationary noise. The dashboard recommends optimal values.

3. **Over-tapping**: More taps ≠ better performance. For 10k sensors, `n_taps > 16` exceeds the 43ms real-time budget. Use 8 taps for PhaseAwareRLS.

4. **Phase Offset Neglect**: Always use `PhaseAwareRLS` (not standard `RLS`) for cardiac artifacts — it compensates for pulse wave propagation delays across the sensor cloud.

## Project Structure

```
subsense-bci-rd/
├── run_demo.py               # One-click Signal Bench demo
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
│   ├── visualization/        # Dark lab theme
│   ├── dashboard/            # Streamlit Signal Bench
│   ├── validation/           # Input validators, Nyquist checks
│   └── presets/              # Demo presets for Signal Bench
├── tests/
│   ├── test_physics.py         # Lead field validation (24 tests)
│   ├── test_unmixing.py        # ICA pipeline validation (9 tests)
│   ├── test_validation.py      # Input validation tests (27 tests)
│   └── test_heartbeat_stress.py # Cardiac artifact tests (17 tests)
├── pyproject.toml            # Package configuration
└── RD_LOG.md                 # Research decisions audit trail
```

## Running Tests

```bash
# Quick test (recommended first run)
pytest tests/test_physics.py -v

# Run all 77 tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=subsense_bci --cov-report=term-missing
```

**Current status: 77 tests passing** ✅

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
