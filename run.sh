#!/bin/bash
# Exit on any error
set -e
# Check if requirements.txt exists
if [[ -f "requirements.txt" ]]; then
    echo "Installing Python dependencies from requirements.txt"
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
    exit 1
fi
# Run the Python script
echo "Running Experiment"
python entanglement_experiment_homogenous.py