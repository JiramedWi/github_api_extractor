#!/bin/bash

# Exit on any error
set -e

# Environment and path
CONDA_ENV_NAME="ERAWAN_env"
FILE_PATH="/app/resources/tsdetect/test_smell_flink/"
SCRIPT_TO_RUN="src/collect_dataset/$1"

# Timestamp function
timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

echo "$(timestamp) ▶ Checking directory exists..."
if [[ -d "$FILE_PATH" ]]; then
    echo "$(timestamp) ✅ Directory found: $FILE_PATH"
else
    echo "$(timestamp) ❌ Error: directory not found at $FILE_PATH"
    exit 1
fi

# Run the Python script directly via conda (avoids activate overhead)
echo "$(timestamp) ▶ Running script with Conda: $SCRIPT_TO_RUN"
conda run -n "$CONDA_ENV_NAME" python "$SCRIPT_TO_RUN"
echo "$(timestamp) ✅ Script completed successfully!"
