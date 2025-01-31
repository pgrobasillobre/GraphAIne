#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Automatically detect script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add `src` directory to Python's path so `models.fnn` can be found
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "../../src")))

# Import the custom loss function after modifying sys.path
from models.fnn import masked_loss  # Custom loss function

# Define max number of atoms (padding requirement)
MAX_ATOMS = 7662

# Define the absolute model path
MODEL_PATH = os.path.join(SCRIPT_DIR, "../../checkpoints/ffn_model.keras")

def read_xyz(filename):
    """
    Read an XYZ file and extract atomic positions.

    Args:
        filename (str): Path to the XYZ file.

    Returns:
        np.ndarray: Atomic positions (N, 3) where N is the number of atoms.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    num_atoms = int(lines[0].strip())  # First line contains number of atoms
    geometries = []

    for line in lines[2:num_atoms+2]:  # Skip first two lines (header)
        parts = line.strip().split()
        x, y, z = map(float, parts[1:4])  # Extract XYZ coordinates
        geometries.append([x, y, z])

    geometries = np.array(geometries, dtype=np.float32)

    # Pad with np.nan if fewer than MAX_ATOMS
    if geometries.shape[0] < MAX_ATOMS:
        padding = np.full((MAX_ATOMS - geometries.shape[0], 3), np.nan, dtype=np.float32)
        geometries = np.vstack([geometries, padding])

    elif geometries.shape[0] > MAX_ATOMS:
        print(f"❌ Geometry file {filename} has more than {MAX_ATOMS} atoms.")
        sys.exit()

    return geometries

def predict_charges(model, geometries, frequency):
    """
    Predict atomic charges using the trained FFN model.

    Args:
        model (tf.keras.Model): The trained FFN model.
        geometries (numpy.ndarray): Atomic coordinates (N, 3).
        frequency (float): Frequency value.

    Returns:
        np.ndarray: Predicted atomic charges (N, 6).
    """
    # Replace NaN values in geometries with 1.0e6
    geometries = np.where(np.isnan(geometries), 1.0e6, geometries)

    # Expand dimensions to match the model's input format
    geometries = np.expand_dims(geometries, axis=0)  # Shape (1, N, 3)
    frequency = np.array([[frequency]], dtype=np.float32)  # Shape (1, 1)

    # Run model prediction
    predicted_charges = model.predict([geometries, frequency])

    return predicted_charges[0]  # Remove batch dimension

def save_results(filename, frequency, geometries, charges, output_dir):
    """
    Save the predicted charges to a file, excluding atoms with padding (np.nan).

    Args:
        filename (str): Base name of the input XYZ file.
        frequency (float): Frequency value.
        geometries (np.ndarray): Atomic coordinates (N, 3).
        charges (np.ndarray): Predicted atomic charges (N, 6).
        output_dir (str): Directory where results should be saved.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    output_filename = os.path.join(output_dir, f"{filename}-{frequency:.2f}.freq")

    # Filter out padded atoms (np.nan in geometries)
    valid_atoms = np.isfinite(geometries[:, 0])  # Check for valid X-coordinates
    valid_geometries = geometries[valid_atoms]   # Filtered atomic positions
    valid_charges = charges[valid_atoms]         # Corresponding charge values

    # Stack data into a single array for saving
    data_to_save = np.hstack([valid_geometries, valid_charges])  # [x, y, z, q_real_1, ..., q_imag_6]

    # Save to file
    np.savetxt(output_filename, data_to_save, fmt="%25.15f",
               delimiter="     ",  # 5 spaces between columns
               header="x y z q_real_1 q_real_2 q_real_3 q_imag_1 q_imag_2 q_imag_3",
               comments="")
    print(f"📄 Results saved to {output_filename}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict atomic charges using a trained FFN model.")
    parser.add_argument("-geom", type=str, required=True, help="Path to the XYZ file")
    parser.add_argument("-freq_ini", type=float, required=True, help="Starting frequency value")
    parser.add_argument("-freq_max", type=float, required=True, help="Maximum frequency value")
    parser.add_argument("-step", type=float, required=True, help="Step size for frequency range")

    args = parser.parse_args()

    # If the user requested help (-h), exit before loading the model
    if len(sys.argv) == 2 and sys.argv[1] in ["-h", "--help"]:
        parser.print_help()
        sys.exit(0)

    # Extract filename without extension for output naming
    base_filename = os.path.splitext(os.path.basename(args.geom))[0]

    # Read input geometry file
    print(f"📄 Reading XYZ file: {args.geom}")
    geometries = read_xyz(args.geom)

    # Generate frequency range
    frequencies = np.arange(args.freq_ini, args.freq_max, args.step)

    # Create results folder (in the directory where the script is run)
    results_folder = f"{base_filename}_results"
    os.makedirs(results_folder, exist_ok=True)

    # 🔄 **Load the model only if needed**
    print("🔄 Loading trained model...")
    model = load_model(MODEL_PATH, custom_objects={"masked_loss": masked_loss})
    print("✅ Model loaded successfully!")

    # Iterate over frequencies and compute charges
    print(f"🔄 Computing charges for {len(frequencies)} frequencies...")
    for freq in frequencies:
        print(f"⚡ Processing frequency: {freq:.2f}")
        charges = predict_charges(model, geometries, freq)  # Compute charges
        save_results(base_filename, freq, geometries, charges, results_folder)  # Pass geometries

    print("✅ Batch prediction complete!")

