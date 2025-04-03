#!/bin/bash

# Ensure script exits on first error
set -e

# Environment name
CONDA_ENV_NAME="ERAWAN_env"

# File path to check
FILE_PATH="/app/resources/tsdetect/test_smell_flink/"

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

# Check if the file exists
echo "check if exist"
if [[ -d "$FILE_PATH" ]]; then
    echo "File exists: $FILE_PATH"
else
    echo "Error: directory not found at $FILE_PATH"
    exit 1
fi

# Run each script separately to avoid argument issues
echo "Running scripts inside Conda environment..."
echo "Run at 1st"
python src/collect_dataset/"$1"
echo "Run at 2nd"
python src/collect_dataset/"$2"
echo "Run at 3rd"
python src/collect_dataset/"$3"
echo "Run at 4th"
python src/collect_dataset/"$4"
echo "Run at 5th"
python src/collect_dataset/"$5"

echo "All scripts executed successfully!"
