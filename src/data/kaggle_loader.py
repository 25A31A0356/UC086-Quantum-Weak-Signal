"""
Direct Kaggle Dataset Connector for Quantum Radar & Sonar Signal Processing.
Uses `kagglehub` and Kaggle API to connect directly to Kaggle datasets in the cloud
without manual download/upload workflows.
"""

import os
import sys
import json
import glob
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd


class KaggleDatasetManager:
    """
    Connects directly to Kaggle datasets using official Kaggle APIs (kagglehub / kaggle).
    Automatically authenticates using Kaggle credentials, environment variables, or Colab secrets.
    """

    DATASETS = {
        "sonar": {
            "handle": "mattcarter865/sonar-data",
            "alt_handle": "uciml/sonar-dataset",
            "csv_pattern": "*.csv",
            "description": "Sonar Mines vs. Rocks 60-band acoustic frequency modulation returns."
        },
        "sar_radar": {
            "handle": "c/statoil-iceberg-classifier-challenge",
            "description": "Statoil C-CORE SAR Radar Sentinel-1 dual-pol backscatter."
        }
    }

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._authenticate_kaggle()

    @staticmethod
    def _authenticate_kaggle():
        """
        Authenticate with Kaggle seamlessly.
        Checks:
        1. Google Colab Secrets (`userdata.get('KAGGLE_USERNAME')`, `userdata.get('KAGGLE_KEY')`)
        2. Environment variables (`KAGGLE_USERNAME`, `KAGGLE_KEY`)
        3. Local ~/.kaggle/kaggle.json
        """
        # Check Colab Secrets
        if "google.colab" in sys.modules:
            try:
                from google.colab import userdata
                username = userdata.get("KAGGLE_USERNAME")
                key = userdata.get("KAGGLE_KEY")
                if username and key:
                    os.environ["KAGGLE_USERNAME"] = username
                    os.environ["KAGGLE_KEY"] = key
                    print("[✓] Authenticated to Kaggle using Google Colab Secrets.")
                    return
            except Exception:
                pass

        # Check existing environment variables
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            print("[✓] Authenticated to Kaggle via environment variables.")
            return

        # Check ~/.kaggle/kaggle.json
        kaggle_file = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_file.exists():
            print(f"[✓] Authenticated to Kaggle using credentials at {kaggle_file}")
            return

    def fetch_sonar_dataset(self) -> str:
        """
        Connects directly to Kaggle and retrieves the Sonar Mines vs Rocks dataset.
        Uses `kagglehub` for zero-configuration cloud fetching.
        
        Returns:
            str: Path to the downloaded CSV file in the local Kaggle cache.
        """
        handle = self.DATASETS["sonar"]["handle"]
        print(f"[+] Connecting to Kaggle to fetch dataset: '{handle}'...")

        # 1. Try kagglehub (Recommended direct connection)
        try:
            import kagglehub
            dataset_path = kagglehub.dataset_download(handle)
            print(f"[✓] Connected to Kaggle! Dataset cached at: {dataset_path}")
            
            # Find CSV file
            csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
            if csv_files:
                return csv_files[0]
        except Exception as e:
            print(f"[i] kagglehub fetch note: {e}. Trying Kaggle API client...")

        # 2. Try Kaggle API client
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            target_dir = self.cache_dir or str(Path.home() / ".cache" / "kaggle_data" / "sonar")
            os.makedirs(target_dir, exist_ok=True)
            
            api.dataset_download_files(handle, path=target_dir, unzip=True)
            csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
            if csv_files:
                print(f"[✓] Retrieved dataset from Kaggle via API into: {csv_files[0]}")
                return csv_files[0]
        except Exception as e:
            print(f"[i] Kaggle API connection note: {e}. Connecting to high-speed public dataset mirror...")

        # 3. Fallback direct connection to UCI repository mirror
        target_dir = self.cache_dir or str(Path.home() / ".cache" / "kaggle_data" / "sonar")
        os.makedirs(target_dir, exist_ok=True)
        fallback_csv = os.path.join(target_dir, "sonar.all-data.csv")
        
        if not os.path.exists(fallback_csv):
            import urllib.request
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
            print(f"[+] Direct streaming from repository mirror: {url}")
            urllib.request.urlretrieve(url, fallback_csv)
            print(f"[✓] Loaded dataset to: {fallback_csv}")

        return fallback_csv

    def fetch_sar_radar_dataset(self) -> Optional[str]:
        """
        Connects directly to Kaggle competition dataset for Statoil SAR Radar backscatter.
        """
        handle = "statoil-iceberg-classifier-challenge"
        print(f"[+] Connecting to Kaggle Competition: '{handle}'...")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            target_dir = self.cache_dir or str(Path.home() / ".cache" / "kaggle_data" / "statoil_radar")
            os.makedirs(target_dir, exist_ok=True)
            api.competition_download_files(handle, path=target_dir)
            return target_dir
        except Exception as e:
            print(f"[!] Kaggle competition fetch requires competition terms acceptance: {e}")
            return None
