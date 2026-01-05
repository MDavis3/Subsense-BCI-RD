"""
Simulation Module

Contains cloud generators for source distributions and lead-field
matrix creators for forward modeling.
"""

from .cloud_generator import (
    generate_sensor_cloud,
    save_sensor_cloud,
    load_sensor_cloud,
)
from .source_generator import (
    get_fixed_sources,
    validate_sources,
    save_sources,
    load_sources,
    SOURCES_3FIXED,
)

