"""
Drone Vision Phase 2 — launcher.
Run from the project root:
    python run.py
    python run.py --config-dir drone_vision/config
    python run.py --help
"""
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drone_vision.main import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone Vision Phase 2")
    parser.add_argument(
        "--config-dir",
        default=os.path.join(os.path.dirname(__file__), "drone_vision", "config"),
        help="Path to config directory (default: drone_vision/config)",
    )
    args = parser.parse_args()
    main(config_dir=args.config_dir)
