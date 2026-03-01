#ifndef GPU_MISC_CUH
#define GPU_MISC_CUH

#include "fmisc_gpu.cuh"

__device__ __forceinline__ void load_field_to_smem( // 256 Threads per block
    double smem[8][12][12], const double* f,
    int ex0, int ex1, int ex2, 
    int block_i, int block_j, int block_k,
    double SYM1, double SYM2, double SYM3
) {
    int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    
    for(int idx = tid; idx < 1152; idx += 256) {
        int loc_k = idx / 144;
        int rem   = idx % 144;
        int loc_j = rem / 12;
        int loc_i = rem % 12;

        int glob_i = block_i + loc_i - 2;
        int glob_j = block_j + loc_j - 2;
        int glob_k = block_k + loc_k - 2;

        smem[loc_k][loc_j][loc_i] = d_symmetry_bd_0b(
            2, ex0, ex1, ex2, f, 
            glob_i, glob_j, glob_k, 
            SYM1, SYM2, SYM3
        );
    }
}

__device__ void compute_all_derivs_smem(
    const double f_smem[8][12][12],
    double* fx, double* fy, double* fz,
    double* fxx, double* fxy, double* fxz,
    double* fyy, double* fyz, double* fzz,
    double dX, double dY, double dZ,
    int i, int j, int k,
    int imin, int jmin, int kmin,
    int imax, int jmax, int kmax
) {
    constexpr double ONE = 1.0;
    constexpr double TWO = 2.0;
    constexpr double EIT = 8.0;
    constexpr double F12 = 12.0;
    constexpr double F1o4 = 0.25;
    constexpr double F1o12 = ONE / 12.0;
    constexpr double F1o144 = ONE / 144.0;
    constexpr double F8 = 8.0;
    constexpr double F16 = 16.0;
    constexpr double F30 = 30.0;
    constexpr double ZEO = 0.0;
    constexpr int NO_SYMM = 0, EQ_SYMM = 1;

    *fx = ZEO; *fy = ZEO; *fz = ZEO;
    *fxx = ZEO; *fyy = ZEO; *fzz = ZEO;
    *fxy = ZEO; *fxz = ZEO; *fyz = ZEO;
    if (i >= imax || j >= jmax || k >= kmax) return;

    int c_i = threadIdx.x + 2;
    int c_j = threadIdx.y + 2;
    int c_k = threadIdx.z + 2;

    auto fh = [&](int di, int dj, int dk) -> double {
        return f_smem[c_k + dk][c_j + dj][c_i + di];
    };

    bool ord_4th = (i + 2 <= imax && i - 2 >= imin &&
                    j + 2 <= jmax && j - 2 >= jmin &&
                    k + 2 <= kmax && k - 2 >= kmin);

    bool ord_2nd = (i + 1 <= imax && i - 1 >= imin &&
                    j + 1 <= jmax && j - 1 >= jmin &&
                    k + 1 <= kmax && k - 1 >= kmin);

    // fdrivs
    const double d12dx = ONE / F12 / dX;
    const double d12dy = ONE / F12 / dY;
    const double d12dz = ONE / F12 / dZ;

    const double d2dx = ONE / TWO / dX;
    const double d2dy = ONE / TWO / dY;
    const double d2dz = ONE / TWO / dZ;

    // fddrivs
    const double Sdxdx = ONE / (dX * dX);
    const double Sdydy = ONE / (dY * dY);
    const double Sdzdz = ONE / (dZ * dZ);

    const double Fdxdx = F1o12 / (dX * dX);
    const double Fdydy = F1o12 / (dY * dY);
    const double Fdzdz = F1o12 / (dZ * dZ);

    const double Sdxdy = F1o4 / (dX * dY);
    const double Sdxdz = F1o4 / (dX * dZ);
    const double Sdydz = F1o4 / (dY * dZ);

    const double Fdxdy = F1o144 / (dX * dY);
    const double Fdxdz = F1o144 / (dX * dZ);
    const double Fdydz = F1o144 / (dY * dZ);

    if (ord_4th) {
        *fx = d12dx * (fh(-2,0,0) - F8*fh(-1,0,0) + F8*fh(1,0,0) - fh(2,0,0));
        *fy = d12dy * (fh(0,-2,0) - F8*fh(0,-1,0) + F8*fh(0,1,0) - fh(0,2,0));
        *fz = d12dz * (fh(0,0,-2) - F8*fh(0,0,-1) + F8*fh(0,0,1) - fh(0,0,2));

        *fxx = Fdxdx * (-fh(-2,0,0) + F16*fh(-1,0,0) - F30*fh(0,0,0) - fh(2,0,0) + F16*fh(1,0,0));
        *fyy = Fdydy * (-fh(0,-2,0) + F16*fh(0,-1,0) - F30*fh(0,0,0) - fh(0,2,0) + F16*fh(0,1,0));
        *fzz = Fdzdz * (-fh(0,0,-2) + F16*fh(0,0,-1) - F30*fh(0,0,0) - fh(0,0,2) + F16*fh(0,0,1));

        *fxy = Fdxdy * (    (fh(-2,-2,0) - F8*fh(-1,-2,0) + F8*fh(1,-2,0) - fh(2,-2,0))
                        -F8*(fh(-2,-1,0) - F8*fh(-1,-1,0) + F8*fh(1,-1,0) - fh(2,-1,0))
                        +F8*(fh(-2, 1,0) - F8*fh(-1, 1,0) + F8*fh(1, 1,0) - fh(2, 1,0))
                        -   (fh(-2, 2,0) - F8*fh(-1, 2,0) + F8*fh(1, 2,0) - fh(2, 2,0)) );
                        
        *fxz = Fdxdz * (    (fh(-2,0,-2) - F8*fh(-1,0,-2) + F8*fh(1,0,-2) - fh(2,0,-2))
                        -F8*(fh(-2,0,-1) - F8*fh(-1,0,-1) + F8*fh(1,0,-1) - fh(2,0,-1))
                        +F8*(fh(-2,0, 1) - F8*fh(-1,0, 1) + F8*fh(1,0, 1) - fh(2,0, 1))
                        -   (fh(-2,0, 2) - F8*fh(-1,0, 2) + F8*fh(1,0, 2) - fh(2,0, 2)) );
                        
        *fyz = Fdydz * (    (fh(0,-2,-2) - F8*fh(0,-1,-2) + F8*fh(0,1,-2) - fh(0,2,-2))
                        -F8*(fh(0,-2,-1) - F8*fh(0,-1,-1) + F8*fh(0,1,-1) - fh(0,2,-1))
                        +F8*(fh(0,-2, 1) - F8*fh(0,-1, 1) + F8*fh(0,1, 1) - fh(0,2, 1))
                        -   (fh(0,-2, 2) - F8*fh(0,-1, 2) + F8*fh(0,1, 2) - fh(0,2, 2)) );
    } 
    else if (ord_2nd) {
        *fx = d2dx * (-fh(-1,0,0) + fh(1,0,0));
        *fy = d2dy * (-fh(0,-1,0) + fh(0,1,0));
        *fz = d2dz * (-fh(0,0,-1) + fh(0,0,1));

        *fxx = Sdxdx * (fh(-1,0,0) - TWO*fh(0,0,0) + fh(1,0,0));
        *fyy = Sdydy * (fh(0,-1,0) - TWO*fh(0,0,0) + fh(0,1,0));
        *fzz = Sdzdz * (fh(0,0,-1) - TWO*fh(0,0,0) + fh(0,0,1));

        *fxy = Sdxdy * (fh(-1,-1,0) - fh(1,-1,0) - fh(-1,1,0) + fh(1,1,0));
        *fxz = Sdxdz * (fh(-1,0,-1) - fh(1,0,-1) - fh(-1,0,1) + fh(1,0,1));
        *fyz = Sdydz * (fh(0,-1,-1) - fh(0,1,-1) - fh(0,-1,1) + fh(0,1,1));
    }
}

#endif // GPU_MISC_CUH