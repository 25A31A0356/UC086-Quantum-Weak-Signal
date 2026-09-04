"""
CLI Dataset Downloader for Quantum Radar and Sonar Signal Processing.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.kaggle_loader import KaggleDatasetManager


def main():
    parser = argparse.ArgumentParser(description="Download and verify Kaggle datasets for Quantum Radar/Sonar project.")
    parser.add_argument("--dataset", type=str, default="sonar", choices=["sonar", "statoil", "all"],
                        help="Which dataset to download (sonar, statoil, or all)")
    parser.add_argument("--kaggle-json", type=str, default=None,
                        help="Path to kaggle.json credential file if not in ~/.kaggle/")
    args = parser.parse_args()

    print("=" * 70)
    print("  QUANTUM RADAR & SONAR SIGNAL PROCESSING - DATASET DOWNLOADER")
    print("=" * 70)

    if args.kaggle_json:
        KaggleDatasetManager.setup_kaggle_credentials(kaggle_json_path=args.kaggle_json)

    manager = KaggleDatasetManager()

    if args.dataset in ["sonar", "all"]:
        print("\n[+] Processing Sonar Mines vs Rocks dataset...")
        try:
            sonar_file = manager.fetch_sonar_dataset()
            print(f"[OK] Ready at: {sonar_file}")
        except Exception as e:
            print(f"[!] Error downloading Sonar dataset: {e}")

    if args.dataset in ["statoil", "all"]:
        print("\n[+] Processing Statoil SAR Maritime Radar dataset...")
        try:
            radar_dir = manager.fetch_sar_radar_dataset()
            if radar_dir:
                print(f"[OK] Ready at: {radar_dir}")
        except Exception as e:
            print(f"[!] Error downloading Statoil dataset: {e}")

    print("\n[OK] Dataset initialization routine complete.")


if __name__ == "__main__":
    main()
