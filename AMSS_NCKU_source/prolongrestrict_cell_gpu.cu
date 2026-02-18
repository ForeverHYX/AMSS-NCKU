#include "prolongrestrict.h"

#include "fmisc.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

// ==========================================
// 1. Constants & Helper Functions
// ==========================================

// Prolongation Coefficients (5th order)
__constant__ double C_PROLONG[6] = {
    77.0 / 8192.0,    // C1
    -693.0 / 8192.0,  // C2
    3465.0 / 4096.0,  // C3
    1155.0 / 4096.0,  // C4
    -495.0 / 8192.0,  // C5
    63.0 / 8192.0     // C6
};

// Restriction Coefficients
__constant__ double C_RESTRICT[3] = {
    3.0 / 256.0,      // C1
    -25.0 / 256.0,    // C2
    75.0 / 128.0      // C3
};

// Fortran IDINT equivalent
__device__ int d_idint(double a) {
    if (fabs(a) < 1.0) return 0;
    return (int)(a);
}

// Memory Access Helper: Column-Major (Fortran Layout)
// i, j, k are 0-based indices
__device__ __forceinline__ int get_col_major_idx(int i, int j, int k, int nx, int ny, int nz) {
    return k * (nx * ny) + j * nx + i;
}

// ==========================================
// 2. Prolongation Device Function
// ==========================================

// Calculate prolonged value for a single point (i, j, k) on the FINE grid.
// i, j, k are 0-based indices [0, extf-1]
__device__ void d_prolong3_device(
    int i, int j, int k, 
    const double* llbc, const double* uubc, const int* extc, const double* func,
    const double* llbf, const double* uubf, const int* extf, double* funf, 
    const double* llbp, const double* uubp,
    const double* SoA, int Symmetry
) {
    // --- 1. Geometry & Alignment (Calculated exactly like Fortran to ensure bitwise match) ---
    double CD[3], FD[3];
    double base[3];
    int lbc[3], lbf[3], lbp[3], ubp[3], lbpc[3], ubpc[3];
    
    // Calculate cell dimensions
    for (int d = 0; d < 3; d++) {
        CD[d] = (uubc[d] - llbc[d]) / (double)extc[d];
        FD[d] = (uubf[d] - llbf[d]) / (double)extf[d];
    }

    // Alignment Logic
    for (int d = 0; d < 3; d++) {
        if (llbc[d] <= llbf[d]) {
            base[d] = llbc[d];
        } else {
            int j_val = d_idint((llbc[d] - llbf[d]) / FD[d] + 0.4f);
            if ((j_val / 2) * 2 == j_val) {
                base[d] = llbf[d];
            } else {
                base[d] = llbf[d] - CD[d] / 2.0;
            }
        }
    }

    // Calculate integer bounds (Resulting values are "1-based" relative to base)
    for (int d = 0; d < 3; d++) {
        lbf[d] = d_idint((llbf[d] - base[d]) / FD[d] + 0.4f) + 1;
        lbc[d] = d_idint((llbc[d] - base[d]) / CD[d] + 0.4f) + 1;
        
        lbp[d] = d_idint((llbp[d] - base[d]) / FD[d] + 0.4f) + 1; 
        ubp[d] = d_idint((uubp[d] - base[d]) / FD[d] + 0.4f);
        
        lbpc[d] = d_idint((llbp[d] - base[d]) / CD[d] + 0.4f) + 1;
        ubpc[d] = d_idint((uubp[d] - base[d]) / CD[d] + 0.4f); // Not strictly used for bounds check here
    }

    // Calculate valid range (1-based relative to loop start)
    // In Fortran: do i = imino, imaxo
    // imino = lbp - lbf + 1
    // In CUDA 0-based: valid i is [imino-1, imaxo-1]
    int imino = lbp[0] - lbf[0] + 1;
    int imaxo = ubp[0] - lbf[0] + 1;
    int jmino = lbp[1] - lbf[1] + 1;
    int jmaxo = ubp[1] - lbf[1] + 1;
    int kmino = lbp[2] - lbf[2] + 1;
    int kmaxo = ubp[2] - lbf[2] + 1;

    // Convert to 1-based for comparison
    int i_1b = i + 1;
    int j_1b = j + 1;
    int k_1b = k + 1;

    if (i_1b < imino || i_1b > imaxo || j_1b < jmino || j_1b > jmaxo || k_1b < kmino || k_1b > kmaxo) {
        return; 
    }

    // --- 2. Index Mapping ---
    
    // Global index (Still effectively 1-based logic for parity check)
    // Fortran: ii = i + lbf - 1. Since our i is 0-based, ii = (i+1) + lbf - 1 = i + lbf
    int ii = i + lbf[0];
    int jj = j + lbf[1];
    int kk = k + lbf[2];

    // Coarse Index Calculation
    // Fortran: cxI = i; cxI = (cxI + lbf - 1)/2; cxI = cxI - lbc + 1
    // CUDA: use i_1b for 'i'
    int cxI_i = (i_1b + lbf[0] - 1) / 2 - lbc[0] + 1;
    int cxI_j = (j_1b + lbf[1] - 1) / 2 - lbc[1] + 1;
    int cxI_k = (k_1b + lbf[2] - 1) / 2 - lbc[2] + 1;

    // Parity Checks (Even/Odd)
    bool k_even = ((kk / 2) * 2 == kk);
    bool j_even = ((jj / 2) * 2 == jj);
    bool i_even = ((ii / 2) * 2 == ii);

    // --- 3. Interpolation ---
    double tmp2[6][6];
    double tmp1[6];

    // Z-Direction Interpolation
    for (int m = 0; m < 6; m++) {
        for (int n = 0; n < 6; n++) {
            int cur_ic = cxI_i - 2 + n;
            int cur_jc = cxI_j - 2 + m;
            
            double val = 0.0;
            // 1-based indices passed to d_get_sym_val
            if (k_even) {
                val += C_PROLONG[0] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 2, SoA);
                val += C_PROLONG[1] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 1, SoA);
                val += C_PROLONG[2] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k    , SoA);
                val += C_PROLONG[3] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 1, SoA);
                val += C_PROLONG[4] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 2, SoA);
                val += C_PROLONG[5] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 3, SoA);
            } else {
                val += C_PROLONG[5] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 2, SoA);
                val += C_PROLONG[4] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 1, SoA);
                val += C_PROLONG[3] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k    , SoA);
                val += C_PROLONG[2] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 1, SoA);
                val += C_PROLONG[1] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 2, SoA);
                val += C_PROLONG[0] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 3, SoA);
            }
            tmp2[m][n] = val;
        }
    }

    // Y-Direction Interpolation
    for (int n = 0; n < 6; n++) {
        double val = 0.0;
        if (j_even) {
            val += C_PROLONG[0] * tmp2[0][n] + C_PROLONG[1] * tmp2[1][n] + C_PROLONG[2] * tmp2[2][n] +
                   C_PROLONG[3] * tmp2[3][n] + C_PROLONG[4] * tmp2[4][n] + C_PROLONG[5] * tmp2[5][n];
        } else {
            val += C_PROLONG[5] * tmp2[0][n] + C_PROLONG[4] * tmp2[1][n] + C_PROLONG[3] * tmp2[2][n] +
                   C_PROLONG[2] * tmp2[3][n] + C_PROLONG[1] * tmp2[4][n] + C_PROLONG[0] * tmp2[5][n];
        }
        tmp1[n] = val;
    }

    // X-Direction Interpolation
    double final_val = 0.0;
    if (i_even) {
        final_val += C_PROLONG[0] * tmp1[0] + C_PROLONG[1] * tmp1[1] + C_PROLONG[2] * tmp1[2] +
                     C_PROLONG[3] * tmp1[3] + C_PROLONG[4] * tmp1[4] + C_PROLONG[5] * tmp1[5];
    } else {
        final_val += C_PROLONG[5] * tmp1[0] + C_PROLONG[4] * tmp1[1] + C_PROLONG[3] * tmp1[2] +
                     C_PROLONG[2] * tmp1[3] + C_PROLONG[1] * tmp1[4] + C_PROLONG[0] * tmp1[5];
    }

    // Write Output (0-based index)
    int out_idx = get_col_major_idx(i, j, k, extf[0], extf[1], extf[2]);
    funf[out_idx] = final_val;
}

// ==========================================
// 3. Restriction Device Function
// ==========================================

// Calculate restricted value for a single point (i, j, k) on the COARSE grid.
// i, j, k are 0-based indices [0, extc-1] (conceptually within the valid restricted range)
__device__ void d_restrict3_device(
    int i, int j, int k, 
    const double* llbc, const double* uubc, const int* extc, double* func, // func is output
    const double* llbf, const double* uubf, const int* extf, const double* funf, // funf is input
    const double* llbr, const double* uubr,
    const double* SoA, int Symmetry
) {
    // --- 1. Geometry & Alignment ---
    double CD[3], FD[3];
    double base[3];
    int lbc[3], lbf[3], lbr[3], ubr[3];

    for (int d = 0; d < 3; d++) {
        CD[d] = (uubc[d] - llbc[d]) / (double)extc[d];
        FD[d] = (uubf[d] - llbf[d]) / (double)extf[d];
    }

    for (int d = 0; d < 3; d++) {
        if (llbc[d] <= llbf[d]) {
            base[d] = llbc[d];
        } else {
            int j_val = d_idint((llbc[d] - llbf[d]) / FD[d] + 0.4f);
            if ((j_val / 2) * 2 == j_val) {
                base[d] = llbf[d];
            } else {
                base[d] = llbf[d] - CD[d] / 2.0;
            }
        }
    }

    for (int d = 0; d < 3; d++) {
        lbf[d] = d_idint((llbf[d] - base[d]) / FD[d] + 0.4f) + 1;
        lbc[d] = d_idint((llbc[d] - base[d]) / CD[d] + 0.4f) + 1;
        lbr[d] = d_idint((llbr[d] - base[d]) / CD[d] + 0.4f) + 1;
        ubr[d] = d_idint((uubr[d] - base[d]) / CD[d] + 0.4f);
    }

    // Range Check
    int imino = lbr[0] - lbc[0] + 1;
    int imaxo = ubr[0] - lbc[0] + 1;
    int jmino = lbr[1] - lbc[1] + 1;
    int jmaxo = ubr[1] - lbc[1] + 1;
    int kmino = lbr[2] - lbc[2] + 1;
    int kmaxo = ubr[2] - lbc[2] + 1;

    int i_1b = i + 1;
    int j_1b = j + 1;
    int k_1b = k + 1;

    if (i_1b < imino || i_1b > imaxo || j_1b < jmino || j_1b > jmaxo || k_1b < kmino || k_1b > kmaxo) {
        return;
    }

    // --- 2. Index Mapping ---
    
    // Coarse to Fine mapping
    // Fortran: cxI = i; cxI = 2*(cxI+lbc-1) - 1; cxI = cxI - lbf + 1
    // CUDA: use i_1b for 'i'
    int if_fine = 2 * (i_1b + lbc[0] - 1) - 1 - lbf[0] + 1;
    int jf_fine = 2 * (j_1b + lbc[1] - 1) - 1 - lbf[1] + 1;
    int kf_fine = 2 * (k_1b + lbc[2] - 1) - 1 - lbf[2] + 1;

    // --- 3. Restriction ---
    double tmp2[6][6];
    double tmp1[6];

    // Z-Direction Restriction
    for (int m = 0; m < 6; m++) {
        for (int n = 0; n < 6; n++) {
            int cur_jf = jf_fine - 2 + m;
            int cur_if = if_fine - 2 + n;
            
            double val = 0.0;
            // Ord=2 passed to symmetry_bd as per Fortran restrict3
            // Indices: -2, -1, 0, 1, 2, 3 relative to fine center
            val += C_RESTRICT[0] * (
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine - 2, SoA) + 
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine + 3, SoA)
            );
            val += C_RESTRICT[1] * (
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine - 1, SoA) + 
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine + 2, SoA)
            );
            val += C_RESTRICT[2] * (
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine    , SoA) + 
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine + 1, SoA)
            );
            
            tmp2[m][n] = val;
        }
    }

    // Y-Direction Restriction
    for (int n = 0; n < 6; n++) {
        double val = 0.0;
        val += C_RESTRICT[0] * (tmp2[0][n] + tmp2[5][n]);
        val += C_RESTRICT[1] * (tmp2[1][n] + tmp2[4][n]);
        val += C_RESTRICT[2] * (tmp2[2][n] + tmp2[3][n]);
        tmp1[n] = val;
    }

    // X-Direction Restriction
    double final_val = 0.0;
    final_val += C_RESTRICT[0] * (tmp1[0] + tmp1[5]);
    final_val += C_RESTRICT[1] * (tmp1[1] + tmp1[4]);
    final_val += C_RESTRICT[2] * (tmp1[2] + tmp1[3]);

    // Write Output (0-based index)
    int out_idx = get_col_major_idx(i, j, k, extc[0], extc[1], extc[2]);
    func[out_idx] = final_val;
}