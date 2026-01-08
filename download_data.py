#!/usr/bin/env python3
"""
Python script to download Navier-Stokes dataset for flow matching PDE experiments.
This script downloads the necessary data files and places them in the appropriate directories.
"""

import os
import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    gdown = None

# Configuration
DATA_DIR = Path("data/_data/navier_stokes")
FILENAME = "fno_ns_Re20_N5000_T50.npy"
FILE_PATH = DATA_DIR / FILENAME

# Google Drive file ID (update this with the actual file ID if available)
FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"

# Alternative: Direct URL (if the file is hosted elsewhere)
DIRECT_URL = None  # e.g., "https://example.com/data/fno_ns_Re20_N5000_T50.npy"


def print_success(message):
    """Print success message in green."""
    print(f"\033[0;32m{message}\033[0m")


def print_warning(message):
    """Print warning message in yellow."""
    print(f"\033[1;33m{message}\033[0m")


def print_error(message):
    """Print error message in red."""
    print(f"\033[0;31m{message}\033[0m")


def download_from_google_drive(file_id, output_path):
    """Download file from Google Drive using gdown."""
    if gdown is None:
        return False
    
    if file_id == "YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
        return False
    
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading from Google Drive: {url}")
        gdown.download(url, str(output_path), quiet=False)
        return output_path.exists()
    except Exception as e:
        print_error(f"Failed to download from Google Drive: {e}")
        return False


def download_from_url(url, output_path):
    """Download file from direct URL."""
    try:
        import urllib.request
        print(f"Downloading from URL: {url}")
        urllib.request.urlretrieve(url, str(output_path))
        return output_path.exists()
    except Exception as e:
        print_error(f"Failed to download from URL: {e}")
        return False


def main():
    """Main download function."""
    print("Downloading Navier-Stokes dataset...")
    
    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if FILE_PATH.exists():
        print_warning(f"File {FILENAME} already exists at {FILE_PATH}")
        print_warning("Skipping download. Delete the file if you want to re-download.")
        return 0
    
    # Try Method 1: Google Drive
    if download_from_google_drive(FILE_ID, FILE_PATH):
        print_success(f"Successfully downloaded {FILENAME} to {DATA_DIR}")
        return 0
    
    # Try Method 2: Direct URL
    if DIRECT_URL and download_from_url(DIRECT_URL, FILE_PATH):
        print_success(f"Successfully downloaded {FILENAME} to {DATA_DIR}")
        return 0
    
    # Manual download instructions
    print_warning("Automatic download failed.")
    print_warning("Please download the file manually:")
    print(f"  1. Download {FILENAME}")
    print(f"  2. Place it in: {FILE_PATH}")
    print()
    print_warning("To use automatic download:")
    print("  - Update FILE_ID in this script with the Google Drive file ID")
    print("  - Or set DIRECT_URL with the direct download URL")
    print()
    print_warning("Data file requirements:")
    print(f"  - File: {FILENAME}")
    print(f"  - Location: {FILE_PATH}")
    print("  - Format: NumPy array with shape [T, H, W, N] where:")
    print("    * T = 50 (time steps)")
    print("    * H = W = 64 (spatial resolution)")
    print("    * N = 5000 (number of trajectories)")
    print("    * Re = 20 (Reynolds number)")
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
