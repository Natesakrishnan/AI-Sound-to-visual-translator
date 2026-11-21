"""
AI-Based Sound-to-Visual Translation for Intelligent Audio Pattern Recognition
Main Application Entry Point.
"""

import os
import sys
from pathlib import Path

# Enforce PyTorch backend for Keras 3 before any Keras imports
os.environ.setdefault("KERAS_BACKEND", "torch")

# Add project root and src folder to Python module search path
project_root = Path(__file__).resolve().parent
src_path = project_root / "src"

for p in [str(src_path), str(project_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from hardware_info import get_hardware_summary_text  # type: ignore
    from gui import launch_gui  # type: ignore
except (ImportError, ModuleNotFoundError):
    from src.hardware_info import get_hardware_summary_text  # type: ignore
    from src.gui import launch_gui  # type: ignore


def main():
    """Display startup diagnostics and launch the desktop GUI."""
    print("\n" + "=" * 60)
    print(" AI-BASED SOUND-TO-VISUAL TRANSLATION SYSTEM")
    print(" Intelligent Audio Pattern Recognition with ESC-50 CNN")
    print("=" * 60)
    print(get_hardware_summary_text())
    print("\n[System] Starting Desktop Application GUI...")
    launch_gui()


if __name__ == "__main__":
    main()
