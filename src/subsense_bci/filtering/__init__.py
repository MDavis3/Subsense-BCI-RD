"""
Filtering Module

Contains DSP routines, ICA implementations, signal unmixing logic
for artifact rejection and source separation, and online decoding.
"""

from .unmixing import (
    UnmixingResult,
    unmix_sources,
    pca_denoise,
    run_fastica,
    match_sources,
    load_phase2_data,
)
from .online_decoder import (
    OnlineDecoder,
    DecodingResult,
)

__all__ = [
    "UnmixingResult",
    "unmix_sources",
    "pca_denoise",
    "run_fastica",
    "match_sources",
    "load_phase2_data",
    "OnlineDecoder",
    "DecodingResult",
]

