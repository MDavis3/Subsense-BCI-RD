#!/usr/bin/env python
"""
SubSense R&D Signal Bench - One-Click Demo

Launches the interactive Streamlit dashboard with the "Standard Cardiac
Interference" preset pre-loaded. Demonstrates cardiac artifact rejection
in magnetoelectric nanoparticle BCI systems.

Usage:
    python run_demo.py

Requirements:
    - streamlit >= 1.28
    - plotly >= 5.18
    - numpy, scipy (standard scientific stack)

For custom configurations, launch directly:
    streamlit run src/subsense_bci/dashboard/streamlit_app.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def check_dependencies() -> list[str]:
    """Check for required dependencies and return list of missing ones."""
    missing = []

    try:
        import streamlit  # noqa: F401
    except ImportError:
        missing.append("streamlit")

    try:
        import plotly  # noqa: F401
    except ImportError:
        missing.append("plotly")

    return missing


def install_dependencies(packages: list[str]) -> bool:
    """Attempt to install missing packages."""
    print(f"Installing missing packages: {', '.join(packages)}")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", *packages, "--quiet"
        ])
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    """Launch the Signal Bench dashboard."""
    print()
    print("=" * 60)
    print("  SUBSENSE R&D SIGNAL BENCH")
    print("  Cardiac Artifact Rejection Demo")
    print("=" * 60)
    print()

    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        response = input("Install now? [Y/n]: ").strip().lower()
        if response in ("", "y", "yes"):
            if not install_dependencies(missing):
                print("Failed to install dependencies. Please install manually:")
                print(f"  pip install {' '.join(missing)}")
                sys.exit(1)
        else:
            print("Please install dependencies manually:")
            print(f"  pip install {' '.join(missing)}")
            sys.exit(1)

    # Get path to streamlit app
    project_root = Path(__file__).parent
    app_path = project_root / "src" / "subsense_bci" / "dashboard" / "streamlit_app.py"

    if not app_path.exists():
        print(f"Error: Dashboard not found at {app_path}")
        print("Please run from the project root directory.")
        sys.exit(1)

    print("Launching Signal Bench dashboard...")
    print()
    print("  Loading preset: Standard Cardiac Interference")
    print("  - 1,000 ME nanoparticle sensors")
    print("  - PhaseAwareRLS filter (8 taps, lambda=0.95)")
    print("  - Heart rate: 72 BPM")
    print("  - Pulse wave velocity: 7.5 m/s")
    print()
    print("Dashboard will open in your browser.")
    print("Press Ctrl+C to stop the server.")
    print()

    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--",
            "--preset", "standard_cardiac",
            "--demo-mode",
        ])
    except KeyboardInterrupt:
        print("\nShutting down Signal Bench...")


if __name__ == "__main__":
    main()
