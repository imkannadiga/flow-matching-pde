#!/bin/bash

ENV_NAME="newenv"
ENV_YML="requirement.txt"  # Optional: if you want to create from this

# Function to check if environment exists
env_exists() {
    mamba env list | grep -qE "^$ENV_NAME\s"
}

# Create environment if it does not exist
if env_exists; then
    echo "Environment '$ENV_NAME' already exists."
else
    echo "Environment '$ENV_NAME' not found. Creating it..."
    mamba create -n "$ENV_NAME" python=3.10 -y
    mamba activate "$ENV_NAME"
    pip install --upgrade pip  # Upgrade pip in the new environment
    if [ -f "$ENV_YML" ]; then
        echo "Installing packages from '$ENV_YML'..."
        pip install -r "$ENV_YML"
    else
        echo "Warning: '$ENV_YML' not found. Skipping package installation."
    fi
fi

# Activate the environment
echo "Activating environment '$ENV_NAME'..."
# Ensure Mamba/Conda init is loaded
source "$(mamba info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Set environment variables
export PROJECT_ROOT=$(pwd)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"  # Customize as needed

echo "Environment variables set:"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Keep the environment active if run interactively
$SHELL
