"""
Validation script for AMSS-NCKU optimization (ASC26).

Compares optimized output against baseline to verify correctness:
  1. BH Trajectory RMS error < 1%
  2. ADM Constraint violation (outermost grid level) abs < 2
  3. Required PDF files exist

Usage:
  python validate.py <baseline_dir> <optimized_dir>

Example:
  python validate.py GW150914_baseline GW150914
"""

import sys
import os
import numpy as np


def load_dat(filepath):
    """Load a .dat file, skipping comment/header lines starting with #."""
    return np.loadtxt(filepath, comments="#")


def check_bh_trajectory_rms(baseline_dir, optimized_dir):
    """
    Check BH trajectory RMS error < 1%.

    RMS = sqrt( (1/M) * sum( ((r1_i - r2_i) / max(r1_i, r2_i))^2 ) )

    where r1_i, r2_i are BH coordinate values from baseline and optimized runs.
    We compute this for each BH coordinate column (x1, y1, z1, x2, y2, z2).
    """
    baseline_file = os.path.join(baseline_dir, "bssn_BH.dat")
    optimized_file = os.path.join(optimized_dir, "bssn_BH.dat")

    if not os.path.exists(baseline_file):
        print(f"FAIL: Baseline file not found: {baseline_file}")
        return False
    if not os.path.exists(optimized_file):
        print(f"FAIL: Optimized file not found: {optimized_file}")
        return False

    data_base = load_dat(baseline_file)
    data_opt = load_dat(optimized_file)

    # Use the shorter of the two (they should match in length)
    M = min(len(data_base), len(data_opt))
    if M == 0:
        print("FAIL: Empty BH data files")
        return False

    if len(data_base) != len(data_opt):
        print(f"WARNING: Different number of timesteps: baseline={len(data_base)}, optimized={len(data_opt)}")
        print(f"         Using first {M} timesteps for comparison")

    data_base = data_base[:M]
    data_opt = data_opt[:M]

    # Columns: time, BH1_x, BH1_y, BH1_z, BH2_x, BH2_y, BH2_z
    # Compute RMS for each coordinate column (columns 1-6)
    ncols = data_base.shape[1]
    all_pass = True

    col_names = ["BH1_x", "BH1_y", "BH1_z", "BH2_x", "BH2_y", "BH2_z"]
    for col_idx in range(1, min(ncols, 7)):
        r1 = data_base[:, col_idx]
        r2 = data_opt[:, col_idx]

        denom = np.maximum(np.abs(r1), np.abs(r2))
        # Avoid division by zero for coordinates near zero
        mask = denom > 1e-12
        if np.sum(mask) == 0:
            continue

        rms = np.sqrt(np.mean(((r1[mask] - r2[mask]) / denom[mask]) ** 2))
        name = col_names[col_idx - 1] if col_idx - 1 < len(col_names) else f"col{col_idx}"
        status = "PASS" if rms < 0.01 else "FAIL"
        if rms >= 0.01:
            all_pass = False
        print(f"  {name}: RMS = {rms:.6f} ({rms*100:.4f}%)  [{status}]")

    # Also compute overall RMS across all BH coordinates
    r1_all = data_base[:, 1:7].flatten()
    r2_all = data_opt[:, 1:7].flatten()
    denom_all = np.maximum(np.abs(r1_all), np.abs(r2_all))
    mask_all = denom_all > 1e-12
    if np.sum(mask_all) > 0:
        rms_overall = np.sqrt(np.mean(((r1_all[mask_all] - r2_all[mask_all]) / denom_all[mask_all]) ** 2))
        status = "PASS" if rms_overall < 0.01 else "FAIL"
        if rms_overall >= 0.01:
            all_pass = False
        print(f"  Overall: RMS = {rms_overall:.6f} ({rms_overall*100:.4f}%)  [{status}]")

    return all_pass


def check_constraint_violation(optimized_dir, grid_levels=9):
    """
    Check ADM constraint violation on outermost grid level (level 0).
    Absolute value must not exceed 2.

    bssn_constraint.dat columns: time, Ham, Px, Py, Pz, Gx, Gy, Gz
    Data is interleaved: for each timestep, there are grid_levels rows.
    Row index for grid level L at timestep j: j * grid_levels + L
    """
    constraint_file = os.path.join(optimized_dir, "bssn_constraint.dat")

    if not os.path.exists(constraint_file):
        print(f"FAIL: Constraint file not found: {constraint_file}")
        return False

    data = load_dat(constraint_file)
    if len(data) == 0:
        print("FAIL: Empty constraint data")
        return False

    nrows = len(data)
    ntimesteps = nrows // grid_levels

    if nrows % grid_levels != 0:
        print(f"WARNING: Number of rows ({nrows}) not divisible by grid_levels ({grid_levels})")
        print(f"         Trying to extract outermost level anyway")

    # Extract outermost grid level (level 0) rows
    level0_indices = [j * grid_levels for j in range(ntimesteps)]
    level0_data = data[level0_indices]

    # Check Hamiltonian constraint (column 1) and momentum constraints (columns 2-4)
    constraint_names = ["Ham", "Px", "Py", "Pz"]
    all_pass = True

    for col_idx, name in enumerate(constraint_names, start=1):
        max_violation = np.max(np.abs(level0_data[:, col_idx]))
        status = "PASS" if max_violation < 2.0 else "FAIL"
        if max_violation >= 2.0:
            all_pass = False
        print(f"  {name}: max |violation| = {max_violation:.6f}  [{status}]")

    return all_pass


def check_required_files(output_base_dir):
    """
    Check that required PDF files exist.
    output_base_dir should be the GW150914 directory (parent of AMSS_NCKU_output).
    """
    required_files = [
        "figure/BH_Trajectory_XY.pdf",
        "figure/BH_Trajectory_21_XY.pdf",
        "figure/ADM_Constraint_Grid_Level_0.pdf",
    ]

    all_pass = True
    for f in required_files:
        filepath = os.path.join(output_base_dir, f)
        exists = os.path.exists(filepath)
        status = "PASS" if exists else "FAIL"
        if not exists:
            all_pass = False
        print(f"  {f}: [{status}]")

    return all_pass


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 validate.py <baseline_GW150914_dir> <optimized_GW150914_dir>")
        print("Example: python3 validate.py GW150914_baseline GW150914")
        sys.exit(1)

    baseline_dir = sys.argv[1]
    optimized_dir = sys.argv[2]

    baseline_output = os.path.join(baseline_dir, "AMSS_NCKU_output")
    optimized_output = os.path.join(optimized_dir, "AMSS_NCKU_output")

    print("=" * 60)
    print("AMSS-NCKU Optimization Validation (ASC26)")
    print("=" * 60)
    print(f"Baseline:  {baseline_dir}")
    print(f"Optimized: {optimized_dir}")
    print()

    # Check 1: BH Trajectory RMS
    print("[1/3] BH Trajectory RMS Error (must be < 1%)")
    print("-" * 40)
    bh_pass = check_bh_trajectory_rms(baseline_output, optimized_output)
    print()

    # Check 2: Constraint Violation
    print("[2/3] ADM Constraint Violation on Grid Level 0 (must be < 2)")
    print("-" * 40)
    constraint_pass = check_constraint_violation(optimized_output)
    print()

    # Check 3: Required Files
    print("[3/3] Required Output Files")
    print("-" * 40)
    files_pass = check_required_files(optimized_dir)
    print()

    # Summary
    print("=" * 60)
    results = [
        ("BH Trajectory RMS < 1%", bh_pass),
        ("Constraint Violation < 2", constraint_pass),
        ("Required Files Exist", files_pass),
    ]

    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name}: [{status}]")

    print()
    if all_pass:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print("RESULT: SOME CHECKS FAILED")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
