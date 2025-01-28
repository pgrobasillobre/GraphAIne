#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Installing GraphAIne..."

# Step 1: Create a virtual environment (if it doesn't already exist)
if [ ! -d "venv" ]; then
    echo "Creating a virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Step 2: Activate the virtual environment
# Use the correct activation path for the platform
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate  # Linux/macOS
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate  # Windows
else
    echo "Error: Unable to find the virtual environment activation script."
    exit 1
fi

# Step 3: Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Step 4: Install dependencies from requirements.txt
echo "Installing dependencies..."
pip install -r requirements.txt

# Step 5: Install the package in editable mode
echo "Installing GraphAIne in editable mode..."
pip install -e .

echo "Installation complete!"

