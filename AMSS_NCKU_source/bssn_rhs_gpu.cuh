#ifndef BSSN_RHS_GPU_CUH
#define BSSN_RHS_GPU_CUH

#include "fmisc_gpu.cuh"

__device__ __forceinline__ void load_field_to_smem(
    double smem[8][12][12], const double* f,
    const int ex[3], int block_i, int block_j, int block_k,
    double SYM1, double SYM2, double SYM3
) {
    int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    double SoA[3] = {SYM1, SYM2, SYM3};
    for(int idx = tid; idx < 1152; idx += 256) {
        int loc_k = idx / 144;
        int rem   = idx % 144;
        int loc_j = rem / 12;
        int loc_i = rem % 12;

        int glob_i = block_i + loc_i - 2;
        int glob_j = block_j + loc_j - 2;
        int glob_k = block_k + loc_k - 2;

        smem[loc_k][loc_j][loc_i] = d_symmetry_bd_0b(2, ex[0], ex[1], ex[2], f, glob_i, glob_j, glob_k, SoA[0], SoA[1], SoA[2]);
    }
}

__device__ __forceinline__ void compute_all_derivs_smem(
    const double smem[8][12][12],
    double* fx, double* fy, double* fz,
    double* fxx, double* fxy, double* fxz,
    double* fyy, double* fyz, double* fzz,
    double dX, double dY, double dZ,
    int i, int j, int k,
    int imin, int jmin, int kmin,
    int imax, int jmax, int kmax
) {
    *fx = 0.0; *fy = 0.0; *fz = 0.0;
    *fxx = 0.0; *fyy = 0.0; *fzz = 0.0;
    *fxy = 0.0; *fxz = 0.0; *fyz = 0.0;

    if (i >= imax || j >= jmax || k >= kmax) return;

    const double d12dx = 1.0 / 12.0 / dX;
    const double d12dy = 1.0 / 12.0 / dY;
    const double d12dz = 1.0 / 12.0 / dZ;
    const double d2dx = 1.0 / 2.0 / dX;
    const double d2dy = 1.0 / 2.0 / dY;
    const double d2dz = 1.0 / 2.0 / dZ;

    const double Sdxdx = 1.0 / (dX * dX);
    const double Sdydy = 1.0 / (dY * dY);
    const double Sdzdz = 1.0 / (dZ * dZ);
    const double Fdxdx = (1.0 / 12.0) / (dX * dX);
    const double Fdydy = (1.0 / 12.0) / (dY * dY);
    const double Fdzdz = (1.0 / 12.0) / (dZ * dZ);
    const double Sdxdy = 0.25 / (dX * dY);
    const double Sdxdz = 0.25 / (dX * dZ);
    const double Sdydz = 0.25 / (dY * dZ);
    const double Fdxdy = (1.0 / 144.0) / (dX * dY);
    const double Fdxdz = (1.0 / 144.0) / (dX * dZ);
    const double Fdydz = (1.0 / 144.0) / (dY * dZ);

    int c_i = threadIdx.x + 2;
    int c_j = threadIdx.y + 2;
    int c_k = threadIdx.z + 2;

    auto fh = [&](int di, int dj, int dk) -> double {
        return smem[c_k + dk][c_j + dj][c_i + di];
    };

    bool fit_4th = (i + 2 <= imax && i - 2 >= imin && j + 2 <= jmax && j - 2 >= jmin && k + 2 <= kmax && k - 2 >= kmin);
    bool fit_2nd = (i + 1 <= imax && i - 1 >= imin && j + 1 <= jmax && j - 1 >= jmin && k + 1 <= kmax && k - 1 >= kmin);

    if (fit_4th) {
        *fx = d12dx * (fh(-2,0,0) - 8.0*fh(-1,0,0) + 8.0*fh(1,0,0) - fh(2,0,0));
        *fy = d12dy * (fh(0,-2,0) - 8.0*fh(0,-1,0) + 8.0*fh(0,1,0) - fh(0,2,0));
        *fz = d12dz * (fh(0,0,-2) - 8.0*fh(0,0,-1) + 8.0*fh(0,0,1) - fh(0,0,2));

        *fxx = Fdxdx * (-fh(-2,0,0) + 16.0*fh(-1,0,0) - 30.0*fh(0,0,0) - fh(2,0,0) + 16.0*fh(1,0,0));
        *fyy = Fdydy * (-fh(0,-2,0) + 16.0*fh(0,-1,0) - 30.0*fh(0,0,0) - fh(0,2,0) + 16.0*fh(0,1,0));
        *fzz = Fdzdz * (-fh(0,0,-2) + 16.0*fh(0,0,-1) - 30.0*fh(0,0,0) - fh(0,0,2) + 16.0*fh(0,0,1));

        *fxy = Fdxdy * (    (fh(-2,-2,0) - 8.0*fh(-1,-2,0) + 8.0*fh(1,-2,0) - fh(2,-2,0))
                        -8.0*(fh(-2,-1,0) - 8.0*fh(-1,-1,0) + 8.0*fh(1,-1,0) - fh(2,-1,0))
                        +8.0*(fh(-2, 1,0) - 8.0*fh(-1, 1,0) + 8.0*fh(1, 1,0) - fh(2, 1,0))
                        -   (fh(-2, 2,0) - 8.0*fh(-1, 2,0) + 8.0*fh(1, 2,0) - fh(2, 2,0)) );
        *fxz = Fdxdz * (    (fh(-2,0,-2) - 8.0*fh(-1,0,-2) + 8.0*fh(1,0,-2) - fh(2,0,-2))
                        -8.0*(fh(-2,0,-1) - 8.0*fh(-1,0,-1) + 8.0*fh(1,0,-1) - fh(2,0,-1))
                        +8.0*(fh(-2,0, 1) - 8.0*fh(-1,0, 1) + 8.0*fh(1,0, 1) - fh(2,0, 1))
                        -   (fh(-2,0, 2) - 8.0*fh(-1,0, 2) + 8.0*fh(1,0, 2) - fh(2,0, 2)) );
        *fyz = Fdydz * (    (fh(0,-2,-2) - 8.0*fh(0,-1,-2) + 8.0*fh(0,1,-2) - fh(0,2,-2))
                        -8.0*(fh(0,-2,-1) - 8.0*fh(0,-1,-1) + 8.0*fh(0,1,-1) - fh(0,2,-1))
                        +8.0*(fh(0,-2, 1) - 8.0*fh(0,-1, 1) + 8.0*fh(0,1, 1) - fh(0,2, 1))
                        -   (fh(0,-2, 2) - 8.0*fh(0,-1, 2) + 8.0*fh(0,1, 2) - fh(0,2, 2)) );
    } 
    else if (fit_2nd) {
        *fx = d2dx * (-fh(-1,0,0) + fh(1,0,0));
        *fy = d2dy * (-fh(0,-1,0) + fh(0,1,0));
        *fz = d2dz * (-fh(0,0,-1) + fh(0,0,1));

        *fxx = Sdxdx * (fh(-1,0,0) - 2.0*fh(0,0,0) + fh(1,0,0));
        *fyy = Sdydy * (fh(0,-1,0) - 2.0*fh(0,0,0) + fh(0,1,0));
        *fzz = Sdzdz * (fh(0,0,-1) - 2.0*fh(0,0,0) + fh(0,0,1));

        *fxy = Sdxdy * (fh(-1,-1,0) - fh(1,-1,0) - fh(-1,1,0) + fh(1,1,0));
        *fxz = Sdxdz * (fh(-1,0,-1) - fh(1,0,-1) - fh(-1,0,1) + fh(1,0,1));
        *fyz = Sdydz * (fh(0,-1,-1) - fh(0,1,-1) - fh(0,-1,1) + fh(0,1,1));
    }
}

__device__ __forceinline__ void load_field_to_smem_rad3(
    double smem[10][14][14], const double* f,
    const int ex[3], int block_i, int block_j, int block_k,
    double SYM1, double SYM2, double SYM3
) {
    int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    double SoA[3] = {SYM1, SYM2, SYM3};
    // Tile 尺寸: 14 * 14 * 10 = 1960。总线程数 256
    for(int idx = tid; idx < 1960; idx += 256) {
        int loc_k = idx / 196;
        int rem   = idx % 196;
        int loc_j = rem / 14;
        int loc_i = rem % 14;

        int glob_i = block_i + loc_i - 3;
        int glob_j = block_j + loc_j - 3;
        int glob_k = block_k + loc_k - 3;

        // 调用原有的 symmetry boundary 函数来填充 Halo
        smem[loc_k][loc_j][loc_i] = d_symmetry_bd_0b(3, ex[0], ex[1], ex[2], f, glob_i, glob_j, glob_k, SoA[0], SoA[1], SoA[2]);
    }
}

__device__ __forceinline__ void compute_first_derivs_smem(
    const double smem[8][12][12],
    double* fx, double* fy, double* fz,
    double dX, double dY, double dZ,
    int i, int j, int k,
    int imin, int jmin, int kmin,
    int imax, int jmax, int kmax
) {
    *fx = 0.0; *fy = 0.0; *fz = 0.0;

    if (i >= imax || j >= jmax || k >= kmax) return;

    const double d12dx = 1.0 / 12.0 / dX;
    const double d12dy = 1.0 / 12.0 / dY;
    const double d12dz = 1.0 / 12.0 / dZ;
    const double d2dx  = 1.0 / 2.0  / dX;
    const double d2dy  = 1.0 / 2.0  / dY;
    const double d2dz  = 1.0 / 2.0  / dZ;

    int c_i = threadIdx.x + 2;
    int c_j = threadIdx.y + 2;
    int c_k = threadIdx.z + 2;

    auto fh = [&](int di, int dj, int dk) -> double {
        return smem[c_k + dk][c_j + dj][c_i + di];
    };

    bool fit_4th = (i + 2 <= imax && i - 2 >= imin && j + 2 <= jmax && j - 2 >= jmin && k + 2 <= kmax && k - 2 >= kmin);
    bool fit_2nd = (i + 1 <= imax && i - 1 >= imin && j + 1 <= jmax && j - 1 >= jmin && k + 1 <= kmax && k - 1 >= kmin);

    if (fit_4th) {
        *fx = d12dx * (fh(-2,0,0) - 8.0*fh(-1,0,0) + 8.0*fh(1,0,0) - fh(2,0,0));
        *fy = d12dy * (fh(0,-2,0) - 8.0*fh(0,-1,0) + 8.0*fh(0,1,0) - fh(0,2,0));
        *fz = d12dz * (fh(0,0,-2) - 8.0*fh(0,0,-1) + 8.0*fh(0,0,1) - fh(0,0,2));
    } 
    else if (fit_2nd) {
        *fx = d2dx * (-fh(-1,0,0) + fh(1,0,0));
        *fy = d2dy * (-fh(0,-1,0) + fh(0,1,0));
        *fz = d2dz * (-fh(0,0,-1) + fh(0,0,1));
    }
}

#endif // BSSN_RHS_GPU_CUH