#!/bin/bash
# Script to download Navier-Stokes dataset for flow matching PDE experiments
# This script downloads the necessary data files and places them in the appropriate directories

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="data/_data/navier_stokes"
FILENAME="fno_ns_Re20_N5000_T50.npy"

# Google Drive file ID (you may need to update this with the actual file ID)
# If the file is publicly available on Google Drive, use gdown with the file ID
# For now, we'll provide a template that can be customized
FILE_ID="YOUR_GOOGLE_DRIVE_FILE_ID_HERE"

# Alternative: Direct URL (if the file is hosted elsewhere)
# DIRECT_URL="https://example.com/data/fno_ns_Re20_N5000_T50.npy"

echo -e "${GREEN}Downloading Navier-Stokes dataset...${NC}"

# Create data directory if it doesn't exist
mkdir -p "${DATA_DIR}"

# Check if file already exists
if [ -f "${DATA_DIR}/${FILENAME}" ]; then
    echo -e "${YELLOW}File ${FILENAME} already exists at ${DATA_DIR}/${FILENAME}${NC}"
    echo -e "${YELLOW}Skipping download. Delete the file if you want to re-download.${NC}"
    exit 0
fi

# Method 1: Try downloading from Google Drive using gdown
if command -v gdown &> /dev/null; then
    if [ "$FILE_ID" != "YOUR_GOOGLE_DRIVE_FILE_ID_HERE" ]; then
        echo -e "${GREEN}Downloading from Google Drive using gdown...${NC}"
        gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${DATA_DIR}/${FILENAME}"
        
        if [ -f "${DATA_DIR}/${FILENAME}" ]; then
            echo -e "${GREEN}Successfully downloaded ${FILENAME} to ${DATA_DIR}/${NC}"
            exit 0
        fi
    fi
fi

# Method 2: Try downloading from direct URL
if [ ! -z "$DIRECT_URL" ]; then
    echo -e "${GREEN}Downloading from direct URL...${NC}"
    if command -v wget &> /dev/null; then
        wget -O "${DATA_DIR}/${FILENAME}" "${DIRECT_URL}"
    elif command -v curl &> /dev/null; then
        curl -L -o "${DATA_DIR}/${FILENAME}" "${DIRECT_URL}"
    else
        echo -e "${RED}Neither wget nor curl is available. Please install one of them.${NC}"
        exit 1
    fi
    
    if [ -f "${DATA_DIR}/${FILENAME}" ]; then
        echo -e "${GREEN}Successfully downloaded ${FILENAME} to ${DATA_DIR}/${NC}"
        exit 0
    fi
fi

# Method 3: Manual download instructions
echo -e "${YELLOW}Automatic download failed.${NC}"
echo -e "${YELLOW}Please download the file manually:${NC}"
echo -e "  1. Download ${FILENAME}"
echo -e "  2. Place it in: ${DATA_DIR}/${FILENAME}"
echo ""
echo -e "${YELLOW}To use automatic download:${NC}"
echo -e "  - Update FILE_ID in this script with the Google Drive file ID"
echo -e "  - Or set DIRECT_URL with the direct download URL"
echo ""
echo -e "${YELLOW}Data file requirements:${NC}"
echo -e "  - File: ${FILENAME}"
echo -e "  - Location: ${DATA_DIR}/${FILENAME}"
echo -e "  - Format: NumPy array with shape [T, H, W, N] where:"
echo -e "    * T = 50 (time steps)"
echo -e "    * H = W = 64 (spatial resolution)"
echo -e "    * N = 5000 (number of trajectories)"
echo -e "    * Re = 20 (Reynolds number)"

exit 1
