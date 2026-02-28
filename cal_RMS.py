#!/usr/bin/env python3
"""
Calculate RMS (Root Mean Square) difference between two bssn_BH.dat files.
File format: time  BH0_x  BH0_y  BH0_z  BH1_x  BH1_y  BH1_z
"""

import sys
import math
from typing import List, Tuple


def parse_bh_dat(filename: str) -> Tuple[List[Tuple[float, float, float]],
                                          List[Tuple[float, float, float]]]:
    """
    Parse bssn_BH.dat and extract puncture positions for both black holes.

    Returns:
        (bh0_positions, bh1_positions): Lists of (x, y, z) tuples for each timestep
    """
    bh0_positions = []
    bh1_positions = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            cols = line.split()
            if len(cols) < 7:
                continue
            # time  BH0_x  BH0_y  BH0_z  BH1_x  BH1_y  BH1_z
            bh0_positions.append((float(cols[1]), float(cols[2]), float(cols[3])))
            bh1_positions.append((float(cols[4]), float(cols[5]), float(cols[6])))

    return bh0_positions, bh1_positions


def vector_magnitude(pos: Tuple[float, float, float]) -> float:
    x, y, z = pos
    return math.sqrt(x**2 + y**2 + z**2)


def calculate_rms(positions1: List[Tuple[float, float, float]],
                  positions2: List[Tuple[float, float, float]]) -> float:
    """
    Calculate RMS difference between two sets of positions using Method 2.
    RMS is calculated separately for x and y coordinates, then averaged.
    Z coordinate is excluded per competition specification.

    RMS_x = sqrt(1/M * sum(((x1-x2) / R_max)^2))
    RMS_y = sqrt(1/M * sum(((y1-y2) / R_max)^2))
    RMS_avg = (RMS_x + RMS_y) / 2
    """
    if len(positions1) == 0 or len(positions2) == 0:
        raise ValueError("No positions found")

    M = min(len(positions1), len(positions2))
    if len(positions1) != len(positions2):
        print(f"  Warning: length mismatch ({len(positions1)} vs {len(positions2)}), "
              f"using first {M} timesteps")

    sum_sq_x = 0.0
    sum_sq_y = 0.0
    valid_M = 0
    for i in range(M):
        pos1, pos2 = positions1[i], positions2[i]
        r_max = max(vector_magnitude(pos1), vector_magnitude(pos2))
        if r_max == 0:
            print(f"  Warning: both positions at timestep {i} are at origin, skipping")
            continue
        x1, y1, _ = pos1
        x2, y2, _ = pos2
        sum_sq_x += ((x1 - x2) / r_max) ** 2
        sum_sq_y += ((y1 - y2) / r_max) ** 2
        valid_M += 1

    rms_x = math.sqrt(sum_sq_x / valid_M)
    rms_y = math.sqrt(sum_sq_y / valid_M)
    print(f"  RMS_x: {rms_x:.6e},  RMS_y: {rms_y:.6e}")
    return (rms_x + rms_y) / 2


def main():
    if len(sys.argv) != 3:
        print("Usage: python cal_RMS.py <file1> <file2>")
        print("Example: python cal_RMS.py run1/bssn_BH.dat run2/bssn_BH.dat")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]

    print(f"Parsing file 1: {file1}")
    bh0_pos1, bh1_pos1 = parse_bh_dat(file1)
    print(f"  Found {len(bh0_pos1)} timesteps")

    print(f"Parsing file 2: {file2}")
    bh0_pos2, bh1_pos2 = parse_bh_dat(file2)
    print(f"  Found {len(bh0_pos2)} timesteps")

    rms_bh0 = rms_bh1 = None

    print("\n" + "="*60)
    print("Black Hole 0 (BH0) RMS Calculation:")
    print("="*60)
    try:
        rms_bh0 = calculate_rms(bh0_pos1, bh0_pos2)
        print(f"RMS difference: {rms_bh0:.6e}")
        print(f"RMS percentage: {rms_bh0 * 100:.4f}%")
        print("PASS: RMS < 1%" if rms_bh0 < 0.01 else "FAIL: RMS >= 1%")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "="*60)
    print("Black Hole 1 (BH1) RMS Calculation:")
    print("="*60)
    try:
        rms_bh1 = calculate_rms(bh1_pos1, bh1_pos2)
        print(f"RMS difference: {rms_bh1:.6e}")
        print(f"RMS percentage: {rms_bh1 * 100:.4f}%")
        print("PASS: RMS < 1%" if rms_bh1 < 0.01 else "FAIL: RMS >= 1%")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "="*60)
    print("Overall Result:")
    print("="*60)
    if rms_bh0 is None or rms_bh1 is None:
        print("FAIL: Could not complete RMS calculation")
        sys.exit(1)
    elif rms_bh0 < 0.01 and rms_bh1 < 0.01:
        print("PASS: Both black holes have RMS < 1%")
        sys.exit(0)
    else:
        print("FAIL: At least one black hole has RMS >= 1%")
        sys.exit(1)


if __name__ == "__main__":
    main()