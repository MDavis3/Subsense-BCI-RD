"""
Physics Module

Contains transfer functions, 3D coordinate math, and physical models
for magnetoelectric transducers and wireless power transfer.
"""

from .constants import *
from .transfer_function import (
    compute_lead_field,
    compute_forward_solution,
    compute_distance_matrix,
    get_sensors_in_exclusion_zone,
    validate_lead_field,
)

