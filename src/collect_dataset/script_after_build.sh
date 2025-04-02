#!/bin/bash

# Ensure script exits on first error
set -e

# Environment name
CONDA_ENV_NAME="ERAWAN_env"

echo "Activating Anaconda environment: $CONDA_ENV_NAME"

# Initialize conda (ensures conda is available in the shell)
source /miniconda/etc/profile.d/conda.sh

# Activate the Conda environment (ensure it's done correctly)

conda activate "$CONDA_ENV_NAME"

# Ensure activation was successful
if [[ $? -ne 0 ]]; then
    echo "Failed to activate Conda environment: $CONDA_ENV_NAME"
    exit 1
fi

# Run each script separately to avoid argument issues
echo "Running scripts inside Conda environment..."
python src/collect_dataset/"$1"
python src/collect_dataset/"$2"
python src/collect_dataset/"$3"
python src/collect_dataset/"$4"
python src/collect_dataset/"$5"

echo "All scripts executed successfully!"
