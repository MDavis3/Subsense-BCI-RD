"""
Streamlit Dashboard for Subsense R&D Signal Bench

Interactive interface for exploring cardiac artifact rejection
in BCI systems, designed for biologists and PIs.
"""

from __future__ import annotations

__all__ = ["main"]


def main():
    """Entry point for the Signal Bench dashboard."""
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
