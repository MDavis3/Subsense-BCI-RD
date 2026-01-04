"""Quick validation script for Phase 1 (no GUI)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from physics.transfer_function import compute_lead_field, validate_lead_field, compute_distance_matrix
from physics.constants import SINGULARITY_THRESHOLD_MM, BRAIN_CONDUCTIVITY_S_M

# Load data
data_dir = Path(__file__).parent.parent / 'data' / 'raw'
sensors = np.load(data_dir / 'sensors_N10000_seed42.npy')
sources = np.load(data_dir / 'sources_3fixed.npy')

# Compute lead field
lead_field, singularity_mask = compute_lead_field(sensors, sources)

# Validation
print('=' * 60)
print('PHASE 1 VALIDATION SUMMARY')
print('=' * 60)
print(f'\n[Sensor Cloud]')
print(f'  Count: {len(sensors)}')
print(f'  Bounds: [{sensors.min():.4f}, {sensors.max():.4f}] mm')

print(f'\n[Neural Sources]')
for i, (src, label) in enumerate(zip(sources, ['A', 'B', 'C'])):
    print(f'  Source {label}: [{src[0]:+.2f}, {src[1]:+.2f}, {src[2]:+.2f}] mm')

print(f'\n[Singularity Analysis]')
n_in_zone = np.sum(np.any(singularity_mask, axis=1))
print(f'  Threshold: {SINGULARITY_THRESHOLD_MM} mm')
print(f'  Sensors in ANY exclusion zone: {n_in_zone}')
for i, label in enumerate(['A', 'B', 'C']):
    print(f'  Sensors near Source {label}: {np.sum(singularity_mask[:, i])}')

print(f'\n[Lead Field]')
info = validate_lead_field(lead_field)
print(f'  Shape: {info["shape"]}')
print(f'  Valid: {info["is_valid"]}')
print(f'  Min value: {info["min_value"]:.2e} V/A')
print(f'  Max value: {info["max_value"]:.2e} V/A')

# Analytical spot check
distances = compute_distance_matrix(sensors, sources)
far_idx = np.argmax(distances[:, 0])
r_mm = distances[far_idx, 0]
r_m = r_mm * 1e-3
v_computed = lead_field[far_idx, 0]
v_expected = 1.0 / (4.0 * np.pi * BRAIN_CONDUCTIVITY_S_M * r_m)
rel_error = abs(v_computed - v_expected) / v_expected * 100
print(f'\n[Analytical Spot Check]')
print(f'  Sensor {far_idx} (r={r_mm:.3f} mm from Source A):')
print(f'    Computed: {v_computed:.4e} V/A')
print(f'    Expected: {v_expected:.4e} V/A')
print(f'    Relative error: {rel_error:.6f}%')
print('=' * 60)

