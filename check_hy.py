#!/usr/bin/env python3
"""
Calculate RMS (Root Mean Square) difference between two simulation log files.
Extracts puncture positions for two black holes and computes relative differences.
"""

import re
import sys
import os
import math
import numpy as np
from typing import List, Tuple


def load_dat(filepath: str):
    """Load a .dat file, skipping comment/header lines starting with #."""
    return np.loadtxt(filepath, comments="#")


def parse_puncture_positions(filename: str) -> Tuple[List[Tuple[float, float, float]],
                                                       List[Tuple[float, float, float]]]:
    """
    Parse .dat file and extract puncture positions for both black holes.

    Expected columns:
      time, BH1_x, BH1_y, BH1_z, BH2_x, BH2_y, BH2_z
    Returns:
        (bh0_positions, bh1_positions): Lists of (x, y, z) tuples for each timestep
    """
    data = load_dat(filename)
    if data.size == 0:
        return [], []

    # Ensure 2D shape for single-row files
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Map BH0 -> BH1 columns, BH1 -> BH2 columns
    bh0_positions = [tuple(row[1:4]) for row in data]
    bh1_positions = [tuple(row[4:7]) for row in data]
    return bh0_positions, bh1_positions


def vector_magnitude(pos: Tuple[float, float, float]) -> float:
    """Calculate the magnitude of a 3D position vector."""
    x, y, z = pos
    return math.sqrt(x**2 + y**2 + z**2)


def euclidean_distance(pos1: Tuple[float, float, float],
                       pos2: Tuple[float, float, float]) -> float:
    """Calculate Euclidean distance between two 3D points."""
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)


def calculate_rms(positions1: List[Tuple[float, float, float]],
                  positions2: List[Tuple[float, float, float]]) -> float:
    """
    Calculate RMS difference between two sets of positions.

    RMS = sqrt(1/M * sum((Δd / Rmax)^2))
    where:
        Δd = |r1i - r2i| (Euclidean distance)
        Rmax = max(|r1i|, |r2i|) (normalization factor)
        M = number of timesteps
    """
    if len(positions1) != len(positions2):
        raise ValueError(f"Position lists have different lengths: {len(positions1)} vs {len(positions2)}")

    if len(positions1) == 0:
        raise ValueError("No positions found")

    M = len(positions1)
    sum_squared_diff = 0.0

    for i, (pos1, pos2) in enumerate(zip(positions1, positions2)):
        # Calculate Euclidean distance
        delta_d = euclidean_distance(pos1, pos2)

        # Calculate magnitudes
        r1_mag = vector_magnitude(pos1)
        r2_mag = vector_magnitude(pos2)

        # Normalization factor
        r_max = max(r1_mag, r2_mag)

        if r_max == 0:
            print(f"Warning: Both positions at timestep {i} are at origin, skipping")
            continue

        # Squared relative difference
        relative_diff = delta_d / r_max
        sum_squared_diff += relative_diff**2

    rms = math.sqrt(sum_squared_diff / M)
    return rms


def resolve_bh_dat_path(path: str) -> str:
    """If a directory is given, use <dir>/AMSS_NCKU_output/bssn_BH.dat."""
    if os.path.isdir(path):
        return os.path.join(path, "AMSS_NCKU_output", "bssn_BH.dat")
    return path


def main():
    if len(sys.argv) != 3:
        print("Usage: python cal_RMS.py <baseline_dir|file.dat> <optimized_dir|file.dat>")
        print("Example: python cal_RMS.py GW150914_baseline GW150914")
        sys.exit(1)

    file1 = resolve_bh_dat_path(sys.argv[1])
    file2 = resolve_bh_dat_path(sys.argv[2])

    print(f"Parsing file 1: {file1}")
    bh0_pos1, bh1_pos1 = parse_puncture_positions(file1)
    print(f"  Found {len(bh0_pos1)} positions for BH0, {len(bh1_pos1)} positions for BH1")

    print(f"\nParsing file 2: {file2}")
    bh0_pos2, bh1_pos2 = parse_puncture_positions(file2)
    print(f"  Found {len(bh0_pos2)} positions for BH0, {len(bh1_pos2)} positions for BH1")

    # Align lengths if needed
    if len(bh0_pos1) != len(bh0_pos2):
        M = min(len(bh0_pos1), len(bh0_pos2))
        print(f"WARNING: Different number of timesteps for BH0; using first {M}")
        bh0_pos1, bh0_pos2 = bh0_pos1[:M], bh0_pos2[:M]
    if len(bh1_pos1) != len(bh1_pos2):
        M = min(len(bh1_pos1), len(bh1_pos2))
        print(f"WARNING: Different number of timesteps for BH1; using first {M}")
        bh1_pos1, bh1_pos2 = bh1_pos1[:M], bh1_pos2[:M]

    # Initialize RMS values
    rms_bh0 = None
    rms_bh1 = None

    # Calculate RMS for BH0
    print("\n" + "="*60)
    print("Black Hole 0 (BH0) RMS Calculation:")
    print("="*60)
    try:
        rms_bh0 = calculate_rms(bh0_pos1, bh0_pos2)
        print(f"RMS difference: {rms_bh0:.6e}")
        print(f"RMS percentage: {rms_bh0 * 100:.4f}%")
        if rms_bh0 < 0.01:
            print("✓ PASS: RMS < 1%")
        else:
            print("✗ FAIL: RMS >= 1%")
    except Exception as e:
        print(f"Error calculating RMS for BH0: {e}")

    # Calculate RMS for BH1
    print("\n" + "="*60)
    print("Black Hole 1 (BH1) RMS Calculation:")
    print("="*60)
    try:
        rms_bh1 = calculate_rms(bh1_pos1, bh1_pos2)
        print(f"RMS difference: {rms_bh1:.6e}")
        print(f"RMS percentage: {rms_bh1 * 100:.4f}%")
        if rms_bh1 < 0.01:
            print("✓ PASS: RMS < 1%")
        else:
            print("✗ FAIL: RMS >= 1%")
    except Exception as e:
        print(f"Error calculating RMS for BH1: {e}")

    # Overall result
    print("\n" + "="*60)
    print("Overall Result:")
    print("="*60)
    if rms_bh0 is None or rms_bh1 is None:
        print("✗ FAIL: Could not complete RMS calculation")
        sys.exit(1)
    elif rms_bh0 < 0.01 and rms_bh1 < 0.01:
        print("✓ PASS: Both black holes have RMS < 1%")
        sys.exit(0)
    else:
        print("✗ FAIL: At least one black hole has RMS >= 1%")
        sys.exit(1)


if __name__ == "__main__":
    main()
