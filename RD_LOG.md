# Subsense BCI R&D Log

## ⚠️ Active Physical Assumptions (Living Document)
*Last Updated: 2026-01-04*

1. **Volume Conductor**: Homogeneous isotropic medium ($\sigma = 0.33 \text{ S/m}$).
2. **Sensor Behavior**: Point-source potential receiver (Voltage $\propto 1/r$).
3. **Artifacts**: Currently modeling **Stationary** noise only (Hemodynamic pulsatility NOT yet implemented).
4. **Coordinate System**: MNI coordinates, origin at anterior commissure.

*(Update this section whenever a fundamental constraint changes)*

---

> **Purpose**: This document serves as the audit trail for all major R&D decisions, 
> mathematical derivations, and implementation changes. Every entry should explain 
> the "why" behind the math and design choices.

---

## Log Format

Each entry should follow this template:

```
### [YYYY-MM-DD] Title of Change

**Category**: Physics | Filtering | Simulation | Infrastructure
**Files Modified**: list of files

**Problem/Goal**:
Brief description of what needed to be solved or achieved.

**Approach**:
Mathematical or algorithmic approach taken, with equations if applicable.

**Why This Approach**:
Justification for the chosen method over alternatives.

**Validation**:
How the implementation was verified (tests, analytical solutions, literature).

**References**:
- Paper citations, textbook references, or documentation links
```

---

## Entries

### [2026-01-04] Project Initialization

**Category**: Infrastructure
**Files Modified**: All (initial structure)

**Problem/Goal**:
Establish the foundational R&D codebase structure for Subsense BCI signal processing
and magnetoelectric transducer simulation.

**Approach**:
Created modular directory structure separating:
- Physics (transfer functions, coordinate math)
- Filtering (DSP, ICA, unmixing)
- Simulation (forward models, source generators)

**Why This Approach**:
Separation of concerns allows independent testing of physics models vs signal processing
algorithms. The `.cursor/rules/` MDC files enforce domain-specific coding standards.

**Validation**:
- Directory structure follows Python packaging best practices
- Initial constants module includes values from published literature

**References**:
- Gabriel et al., 1996 - Tissue conductivity values
- ActiveEcho system specifications for ME transducer parameters

---

### [2026-01-04] Phase 1: Stochastic Nanoparticle Cloud

**Category**: Simulation | Physics
**Files Modified**: 
- `src/physics/constants.py` (added simulation parameters)
- `src/physics/transfer_function.py` (new)
- `src/simulation/cloud_generator.py` (new)
- `src/simulation/source_generator.py` (new)
- `notebooks/visualize_cloud.py` (new)
- `data/raw/sensors_N10000_seed42.npy` (generated)
- `data/raw/sources_3fixed.npy` (generated)

**Problem/Goal**:
Build the forward model foundation for Subsense volumetric BCI: generate a stochastic 
nanoparticle cloud representing 10,000 ME sensors, define neural point sources, and 
compute the lead field matrix to validate geometric source distinguishability.

**Approach**:

*1. Sensor Cloud Generation*
- Uniform random distribution in 1 mm³ cube centered at origin
- Coordinates: $[-0.5, +0.5]$ mm per axis
- Reproducible via `np.random.seed(42)`

*2. Lead Field Computation*
For a point current source in a homogeneous isotropic conductor:

$$L_{ij} = \frac{1}{4\pi\sigma r_{ij}}$$

where:
- $\sigma = 0.33$ S/m (brain tissue conductivity)
- $r_{ij}$ = Euclidean distance from sensor $i$ to source $j$

*3. Unit Conversion Strategy*
- API boundary: coordinates in mm (consistent with domain)
- Internal calculation: convert to meters ($r_m = r_{mm} \times 10^{-3}$)
- Conductivity: S/m (no conversion needed)
- Output: V/A (volts per ampere of source current)

*4. Singularity Handling*
**Decision: Distance clamping (not sensor pruning)**

$$r_{safe} = \max(r, 0.05 \text{ mm})$$

At threshold: $V_{max} = \frac{1}{4\pi \cdot 0.33 \cdot 5 \times 10^{-5}} \approx 4823$ V/A

**Why This Approach**:

1. **Clamping over pruning**: Pruning sensors changes array shapes, complicating 
   downstream indexing and requiring bookkeeping. Clamping preserves the 10,000-sensor 
   count while bounding maximum voltage to a finite (if unrealistic) value.

2. **No collision detection**: With 10,000 particles (r=100nm each) in 1 mm³, the 
   volume fraction is ~0.004%. At this density, particle collisions are statistically 
   negligible. Collision checking was omitted for computational efficiency.

3. **Vectorized implementation**: Used NumPy broadcasting for O(N×M) distance 
   computation without Python loops:
   ```python
   diff = sensors[:, np.newaxis, :] - sources[np.newaxis, :, :]
   distances = np.linalg.norm(diff, axis=2)
   ```

**Validation**:

| Check | Result |
|-------|--------|
| Sensor count | 10,000 ✓ |
| Sensor bounds | [-0.5, +0.5] mm ✓ |
| Lead field shape | (10000, 3) ✓ |
| No infinities | True ✓ |
| No NaNs | True ✓ |
| Sensors in exclusion zones | 10 total (2+4+4 per source) |
| Analytical spot check | 0.000000% relative error ✓ |

Analytical validation: For sensor 1304 at r=0.963 mm from Source A:
- Computed: 2.5048×10² V/A
- Expected: $\frac{1}{4\pi \cdot 0.33 \cdot 9.63 \times 10^{-4}}$ = 2.5048×10² V/A
- Match: Exact (within floating-point precision)

**References**:
- Nunez & Srinivasan, "Electric Fields of the Brain" (2006) - Volume conductor theory
- Hämäläinen et al., "Magnetoencephalography" (1993) - Lead field formulation

---

### [2026-01-04] Phase 1 Complete - Cloud & Physics Validation

**Category**: Simulation

**Validation Summary**:

Phase 1 implementation has been verified through both analytical and visual validation:

1. **$1/r$ Decay Verification**: The log-log plot of lead field values vs. distance 
   confirms exact agreement with the theoretical curve $V = \frac{1}{4\pi\sigma r}$. 
   No deviation observed across the full distance range.

2. **Singularity Clamping**: The exclusion zone analysis successfully identified 
   ~10 sensors within 0.05 mm of neural sources:
   - Source A: 2 sensors clamped
   - Source B: 4 sensors clamped  
   - Source C: 4 sensors clamped
   
   These sensors are highlighted in the 3D visualization and their lead field 
   values are bounded at $V_{max} \approx 4823$ V/A as expected.

3. **Cloud Distribution**: Visual inspection confirms uniform sensor distribution 
   within the 1 mm³ domain with no clustering artifacts.

**Status**: ✅ Phase 1 COMPLETE — Ready for Phase 2 (Source Localization / Inverse Problem)

---

<!-- Add new entries above this line -->

