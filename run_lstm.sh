#!/bin/bash

echo "Starting Battery xLSTM Training..."

# Ensure the config file exists
CONFIG_FILE="experiments/battery_xlstm.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at $CONFIG_FILE"
    exit 1
fi

# Optional: Activate your virtual environment if needed
# echo "Activating virtual environment..."
# source /path/to/your/venv/bin/activate
# if [ $? -ne 0 ]; then
#     echo "ERROR: Failed to activate virtual environment."
#     exit 1
# fi

# Optional: Set CUDA device if not using device from YAML
# export CUDA_VISIBLE_DEVICES=0

# Run the main training script
echo "Running main.py with config: $CONFIG_FILE"
python experiments/main.py --config "$CONFIG_FILE"

# Check exit status
status=$?
if [ $status -eq 0 ]; then
  echo "Training finished successfully."
else
  echo "Training failed with status $status."
fi

# Optional: Deactivate virtual environment
# deactivate

exit $status
