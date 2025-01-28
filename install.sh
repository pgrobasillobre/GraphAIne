#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Installing GraphAIne..."

# Step 1: Create a virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    echo "Creating a virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
else
    echo "Virtual environment already exists. Activating..."
    source venv/bin/activate
fi

# Step 2: Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Step 3: Install dependencies from requirements.txt
echo "Installing dependencies..."
pip install -r requirements.txt

# Step 4: Install the package in editable mode
echo "Installing GraphAIne in editable mode..."
pip install -e .

echo "Installation complete!"

