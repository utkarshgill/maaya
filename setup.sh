#!/usr/bin/env bash
set -e

# Step 1: Install Miniforge if conda isn't available
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniforge3..."
    curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
    bash Miniforge3-MacOSX-arm64.sh -b -p "$HOME/miniforge3"
    eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
else
    # Initialize conda in the current shell
    eval "$(conda shell.bash hook)"
fi

# Step 2: Create environment if it doesn't exist
ENV_NAME="pybullet_env"
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "Conda environment ${ENV_NAME} already exists."
else
    echo "Creating conda environment ${ENV_NAME} with Python 3.10..."
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

# Step 3: Activate the environment
echo "Activating conda environment ${ENV_NAME}..."
conda activate "${ENV_NAME}"

# Step 4: Install pybullet via conda
echo "Installing pybullet..."
conda install -c conda-forge pybullet -y

# Step 5: Install Python dependencies via pip
echo "Installing other Python dependencies..."
pip install matplotlib hidapi gymnasium

echo "Setup complete. Conda environment ${ENV_NAME} is ready and activated." 