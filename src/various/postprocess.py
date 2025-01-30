#!/usr/bin/env python3

import os
import sys
import numpy as np
import argparse
import pandas as pd
import glob

### ============================================================
### FUNCTION TO READ .FREQ FILES AND EXTRACT ATOMIC POSITIONS & CHARGES
### ============================================================
def read_freq_file(filename, charge_type):
    """
    Read a .freq file and extract atomic positions and chosen imaginary charges.

    Args:
        filename (str): Path to the .freq file.
        charge_type (str): "x" for q_imag_1, "y" for q_imag_2".

    Returns:
        np.ndarray: Atomic positions (N, 3).
        np.ndarray: Selected imaginary charges (N,).
    """
    try:
        data = np.loadtxt(filename, skiprows=1)  # Skip header
    except Exception as e:
        print(f"❌ Error reading file {filename}: {e}")
        return None, None

    if data.size == 0:
        print(f"⚠️ Warning: {filename} is empty.")
        return None, None

    # Extract x, y, z positions and imaginary charge based on selection
    positions = data[:, :3]  # Columns 0, 1, 2 -> x, y, z

    if charge_type == "x":
        charges = data[:, 6]  # Column 6 -> q_imag_1 (X-component)
    elif charge_type == "y":
        charges = data[:, 7]  # Column 7 -> q_imag_2 (Y-component)
    else:
        print("❌ Invalid charge type. Use 'x' for q_imag_1 or 'y' for q_imag_2.")
        return None, None

    return positions, charges

### ============================================================
### FUNCTION TO WRITE .PQR FILE
### ============================================================
def write_pqr(filename, positions, charges, charge_type, atom_type="C", eta=2.7):
    """
    Write atomic positions and selected charges to a properly formatted .pqr file.

    Args:
        filename (str): Name of the output .pqr file.
        positions (np.ndarray): Atomic positions (N, 3).
        charges (np.ndarray): Selected imaginary charges (N,).
        charge_type (str): Charge type ("x" or "y").
        atom_type (str): Atom label (default: "C" for Carbon).
        eta (float): Radius of the atom in Angstroms (default: 2.7).
    """
    output_pqr = filename.replace(".freq", f"_{charge_type}.pqr")

    with open(output_pqr, "w") as f:
        for i, (pos, charge) in enumerate(zip(positions, charges)):
            x, y, z = pos
            f.write(f"ATOM {i+1:6d}  {atom_type:<2s}   X     1     {x:7.3f} {y:7.3f} {z:7.3f} {charge:10.3E} {eta:5.3f}\n")

    print(f"✅ PQR file saved: {output_pqr}")

### ============================================================
### FUNCTION TO COMPUTE DIPOLE MOMENTS
### ============================================================
def compute_dipole(positions, charges):
    """
    Compute the dipole moment for a given set of atomic positions and charges.

    Args:
        positions (np.ndarray): Atomic positions (N, 3).
        charges (np.ndarray): Charge components (N, 3).

    Returns:
        tuple: Dipole moments (ζ_x, ζ_y, ζ_z)
    """
    dipole_x = np.sum(charges[:, 0] * positions[:, 0])  # Sum(q_i * x_i)
    dipole_y = np.sum(charges[:, 1] * positions[:, 1])  # Sum(q_i * y_i)
    dipole_z = np.sum(charges[:, 2] * positions[:, 2])  # Sum(q_i * z_i)

    return dipole_x, dipole_y, dipole_z

### ============================================================
### FUNCTION TO PROCESS MULTIPLE .FREQ FILES AND COMPUTE DIPOLES
### ============================================================
def process_freq_files(folder):
    """
    Iterate over all .freq files in a folder, compute real and imaginary dipoles,
    and save results in a CSV file.

    Args:
        folder (str): Path to the folder containing .freq files.
    """
    freq_files = sorted(glob.glob(os.path.join(folder, "*.freq")))

    if not freq_files:
        print(f"❌ No .freq files found in {folder}.")
        return

    print(f"🔍 Found {len(freq_files)} .freq files. Processing...")

    results = []

    for file in freq_files:
        try:
            frequency = float(file.split("-")[-1].replace(".freq", ""))
        except ValueError:
            print(f"⚠️ Skipping {file}: Unable to extract frequency from filename.")
            continue

        data = np.loadtxt(file, skiprows=1)  # Skip header

        if data.size == 0:
            print(f"⚠️ Warning: {file} is empty. Skipping.")
            continue

        positions = data[:, :3]  # x, y, z positions
        real_charges = data[:, 3:6]  # q_real_1, q_real_2, q_real_3
        imag_charges = data[:, 6:9]  # q_imag_1, q_imag_2, q_imag_3

        dipole_real = compute_dipole(positions, real_charges)
        dipole_imag = compute_dipole(positions, imag_charges)

        results.append([frequency] + list(dipole_real) + list(dipole_imag))

    df = pd.DataFrame(results, columns=[
        "#frequency", "x_polar_real", "y_polar_real", "z_polar_real",
        "x_polar_imag", "y_polar_imag", "z_polar_imag"
    ])
    df = df.sort_values(by="#frequency")  # Ensure frequencies are in ascending order

    output_csv = os.path.join(folder, "polarization_results.csv")
    df.to_csv(output_csv, index=False, sep=' ', float_format="%.8f")

    print(f"✅ Polarization data saved to {output_csv}")

### ============================================================
### MAIN SCRIPT: ARGUMENT PARSING
### ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process .freq files (convert to .pqr OR compute polarization).")
    parser.add_argument("-pqr", type=str, help="Path to a .freq file to convert to .pqr")
    parser.add_argument("-charge_type", type=str, choices=["x", "y"], help="Choose 'x' for q_imag_1 or 'y' for q_imag_2")
    parser.add_argument("-polar", type=str, help="Path to the folder containing .freq files for polarization calculation")
    parser.add_argument("-atom_type", type=str, default="C", help="Specify the atom type (default: C)")
    parser.add_argument("-eta", type=float, default=2.7, help="Specify the atomic radius in Å (default: 2.7)")

    args = parser.parse_args()

    if args.pqr and args.charge_type:
        if not os.path.exists(args.pqr):
            print(f"❌ Error: File {args.pqr} not found.")
            sys.exit(1)

        print(f"🔄 Processing {args.pqr} to .pqr using {args.charge_type}-component charges...")
        positions, charges = read_freq_file(args.pqr, args.charge_type)

        if positions is not None and charges is not None:
            write_pqr(args.pqr, positions, charges, args.charge_type, args.atom_type, args.eta)

    elif args.polar:
        process_freq_files(args.polar)

    else:
        print("❌ Error: You must specify either `-pqr` or `-polar` for processing.")
        sys.exit(1)

