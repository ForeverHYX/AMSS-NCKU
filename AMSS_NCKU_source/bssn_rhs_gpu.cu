#include "bssn_rhs.h"

#include "fmisc_gpu.cuh"
#include "diff_new_gpu.cuh"
#include "kodiss_gpu.cuh"
#include "lopsidediff_gpu.cuh"
#include "gpu_manager.h"

#include <cuda_runtime.h>
#include <math.h>
#include <device_launch_parameters.h>
#include <iostream>

#define IDX3D(i, j, k, nx, ny, nz) ((i) + (nx) * ((j) + (ny) * (k)))

constexpr double SYM = 1.0;
constexpr double ANTI = -1.0;
constexpr double ZEO = 0.0;
constexpr double ONE = 1.0;
constexpr double TWO = 2.0;
constexpr double FOUR = 4.0;
constexpr double EIGHT = 8.0;
constexpr double PI = M_PI;
constexpr double F1o3 = 1.0 / 3.0;
constexpr double F2o3 = 2.0 / 3.0;
constexpr double F3o2 = 1.5;
constexpr double HALF = 0.5;
constexpr double FF = 0.75;
constexpr double eta = 2.0;
constexpr double F8 = 8.0;
constexpr double F16 = 16.0;

// ==============================================================================
// Helper: Shared Memory Load & Batch Derivatives
// ==============================================================================

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

__launch_bounds__(256, 2)
__global__ void bssn_ricci_kernel(
    const int ex0, const int ex1, const int ex2, 
    const int symmetry, const int lev,
    const double* __restrict__ X, const double* __restrict__ Y, const double* __restrict__ Z,
    const double* __restrict__ dxx, const double* __restrict__ gxy, const double* __restrict__ gxz,
    const double* __restrict__ dyy, const double* __restrict__ gyz, const double* __restrict__ dzz,
    const double* __restrict__ Gamx, const double* __restrict__ Gamy, const double* __restrict__ Gamz,
    // 【输出】严格为共形克氏符和共形里奇张量
    double* __restrict__ Gamxxx_out, double* __restrict__ Gamxxy_out, double* __restrict__ Gamxxz_out,
    double* __restrict__ Gamxyy_out, double* __restrict__ Gamxyz_out, double* __restrict__ Gamxzz_out,
    double* __restrict__ Gamyxx_out, double* __restrict__ Gamyxy_out, double* __restrict__ Gamyxz_out,
    double* __restrict__ Gamyyy_out, double* __restrict__ Gamyyz_out, double* __restrict__ Gamyzz_out,
    double* __restrict__ Gamzxx_out, double* __restrict__ Gamzxy_out, double* __restrict__ Gamzxz_out,
    double* __restrict__ Gamzyy_out, double* __restrict__ Gamzyz_out, double* __restrict__ Gamzzz_out,
    double* __restrict__ Rxx_out, double* __restrict__ Rxy_out, double* __restrict__ Rxz_out,
    double* __restrict__ Ryy_out, double* __restrict__ Ryz_out, double* __restrict__ Rzz_out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int dims[3] = {ex0, ex1, ex2};
    double dX = X[1] - X[0]; double dY = Y[1] - Y[0]; double dZ = Z[1] - Z[0];

    int imax = ex0 - 1; int jmax = ex1 - 1; int kmax = ex2 - 1;
    int imin = 0, jmin = 0, kmin = 0;
    if (symmetry > 0 && fabs(Z[0]) < dZ) kmin = -2;
    if (symmetry > 1 && fabs(X[0]) < dX) imin = -2;
    if (symmetry > 1 && fabs(Y[0]) < dY) jmin = -2;

    int block_i = blockIdx.x * blockDim.x;
    int block_j = blockIdx.y * blockDim.y;
    int block_k = blockIdx.z * blockDim.z;

    __shared__ double smem[8][12][12];

    // ================== BATCH DERIVATIVES: 混合导数（需一阶和二阶） ==================
    double gxxx, gxxy, gxxz, dxx_xx, dxx_xy, dxx_xz, dxx_yy, dxx_yz, dxx_zz;
    load_field_to_smem(smem, dxx, dims, block_i, block_j, block_k, SYM, SYM, SYM);
    __syncthreads(); compute_all_derivs_smem(smem, &gxxx, &gxxy, &gxxz, &dxx_xx, &dxx_xy, &dxx_xz, &dxx_yy, &dxx_yz, &dxx_zz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double gxyx, gxyy, gxyz, gxy_xx, gxy_xy, gxy_xz, gxy_yy, gxy_yz, gxy_zz;
    load_field_to_smem(smem, gxy, dims, block_i, block_j, block_k, ANTI, ANTI, SYM);
    __syncthreads(); compute_all_derivs_smem(smem, &gxyx, &gxyy, &gxyz, &gxy_xx, &gxy_xy, &gxy_xz, &gxy_yy, &gxy_yz, &gxy_zz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double gxzx, gxzy, gxzz, gxz_xx, gxz_xy, gxz_xz, gxz_yy, gxz_yz, gxz_zz;
    load_field_to_smem(smem, gxz, dims, block_i, block_j, block_k, ANTI, SYM, ANTI);
    __syncthreads(); compute_all_derivs_smem(smem, &gxzx, &gxzy, &gxzz, &gxz_xx, &gxz_xy, &gxz_xz, &gxz_yy, &gxz_yz, &gxz_zz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double gyyx, gyyy, gyyz, dyy_xx, dyy_xy, dyy_xz, dyy_yy, dyy_yz, dyy_zz;
    load_field_to_smem(smem, dyy, dims, block_i, block_j, block_k, SYM, SYM, SYM);
    __syncthreads(); compute_all_derivs_smem(smem, &gyyx, &gyyy, &gyyz, &dyy_xx, &dyy_xy, &dyy_xz, &dyy_yy, &dyy_yz, &dyy_zz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double gyzx, gyzy, gyzz, gyz_xx, gyz_xy, gyz_xz, gyz_yy, gyz_yz, gyz_zz;
    load_field_to_smem(smem, gyz, dims, block_i, block_j, block_k, SYM, ANTI, ANTI);
    __syncthreads(); compute_all_derivs_smem(smem, &gyzx, &gyzy, &gyzz, &gyz_xx, &gyz_xy, &gyz_xz, &gyz_yy, &gyz_yz, &gyz_zz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double gzzx, gzzy, gzzz, dzz_xx, dzz_xy, dzz_xz, dzz_yy, dzz_yz, dzz_zz;
    load_field_to_smem(smem, dzz, dims, block_i, block_j, block_k, SYM, SYM, SYM);
    __syncthreads(); compute_all_derivs_smem(smem, &gzzx, &gzzy, &gzzz, &dzz_xx, &dzz_xy, &dzz_xz, &dzz_yy, &dzz_yz, &dzz_zz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    // ================== BATCH DERIVATIVES: 特化导数（仅需一阶导数，消灭 dum 变量） ==================
    double dGamxx, dGamxy, dGamxz;
    load_field_to_smem(smem, Gamx, dims, block_i, block_j, block_k, ANTI, SYM, SYM);
    __syncthreads(); compute_first_derivs_smem(smem, &dGamxx, &dGamxy, &dGamxz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double dGamyx, dGamyy, dGamyz;
    load_field_to_smem(smem, Gamy, dims, block_i, block_j, block_k, SYM, ANTI, SYM);
    __syncthreads(); compute_first_derivs_smem(smem, &dGamyx, &dGamyy, &dGamyz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double dGamzx, dGamzy, dGamzz;
    load_field_to_smem(smem, Gamz, dims, block_i, block_j, block_k, SYM, SYM, ANTI);
    __syncthreads(); compute_first_derivs_smem(smem, &dGamzx, &dGamzy, &dGamzz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax);
    // 此后不再往 smem 载入新变量，故无需最后一次 __syncthreads()

    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    int idx = IDX3D(i, j, k, ex0, ex1, ex2);

    const double HALF = 0.5; const double TWO = 2.0; const double ONE = 1.0;

    double l_gxx = dxx[idx] + ONE; double l_gxy = gxy[idx]; double l_gxz = gxz[idx];
    double l_gyy = dyy[idx] + ONE; double l_gyz = gyz[idx]; double l_gzz = dzz[idx] + ONE;

    double gupzz = l_gxx * l_gyy * l_gzz + l_gxy * l_gyz * l_gxz + l_gxz * l_gxy * l_gyz - l_gxz * l_gyy * l_gxz - l_gxy * l_gxy * l_gzz - l_gxx * l_gyz * l_gyz;
    double gupxx = (l_gyy * l_gzz - l_gyz * l_gyz) / gupzz;
    double gupxy = -(l_gxy * l_gzz - l_gyz * l_gxz) / gupzz;
    double gupxz = (l_gxy * l_gyz - l_gyy * l_gxz) / gupzz;
    double gupyy = (l_gxx * l_gzz - l_gxz * l_gxz) / gupzz;
    double gupyz = -(l_gxx * l_gyz - l_gxy * l_gxz) / gupzz;
    gupzz = (l_gxx * l_gyy - l_gxy * l_gxy) / gupzz;

    double l_Gamxxx = HALF * (gupxx * gxxx + gupxy * (TWO * gxyx - gxxy) + gupxz * (TWO * gxzx - gxxz));
    double l_Gamyxx = HALF * (gupxy * gxxx + gupyy * (TWO * gxyx - gxxy) + gupyz * (TWO * gxzx - gxxz));
    double l_Gamzxx = HALF * (gupxz * gxxx + gupyz * (TWO * gxyx - gxxy) + gupzz * (TWO * gxzx - gxxz));
    double l_Gamxyy = HALF * (gupxx * (TWO * gxyy - gyyx) + gupxy * gyyy + gupxz * (TWO * gyzy - gyyz));
    double l_Gamyyy = HALF * (gupxy * (TWO * gxyy - gyyx) + gupyy * gyyy + gupyz * (TWO * gyzy - gyyz));
    double l_Gamzyy = HALF * (gupxz * (TWO * gxyy - gyyx) + gupyz * gyyy + gupzz * (TWO * gyzy - gyyz));
    double l_Gamxzz = HALF * (gupxx * (TWO * gxzz - gzzx) + gupxy * (TWO * gyzz - gzzy) + gupxz * gzzz);
    double l_Gamyzz = HALF * (gupxy * (TWO * gxzz - gzzx) + gupyy * (TWO * gyzz - gzzy) + gupyz * gzzz);
    double l_Gamzzz = HALF * (gupxz * (TWO * gxzz - gzzx) + gupyz * (TWO * gyzz - gzzy) + gupzz * gzzz);
    double l_Gamxxy = HALF * (gupxx * gxxy + gupxy * gyyx + gupxz * (gxzy + gyzx - gxyz));
    double l_Gamyxy = HALF * (gupxy * gxxy + gupyy * gyyx + gupyz * (gxzy + gyzx - gxyz));
    double l_Gamzxy = HALF * (gupxz * gxxy + gupyz * gyyx + gupzz * (gxzy + gyzx - gxyz));
    double l_Gamxxz = HALF * (gupxx * gxxz + gupxy * (gxyz + gyzx - gxzy) + gupxz * gzzx);
    double l_Gamyxz = HALF * (gupxy * gxxz + gupyy * (gxyz + gyzx - gxzy) + gupyz * gzzx);
    double l_Gamzxz = HALF * (gupxz * gxxz + gupyz * (gxyz + gyzx - gxzy) + gupzz * gzzx);
    double l_Gamxyz = HALF * (gupxx * (gxyz + gxzy - gyzx) + gupxy * gyyz + gupxz * gzzy);
    double l_Gamyyz = HALF * (gupxy * (gxyz + gxzy - gyzx) + gupyy * gyyz + gupyz * gzzy);
    double l_Gamzyz = HALF * (gupxz * (gxyz + gxzy - gyzx) + gupyz * gyyz + gupzz * gzzy);

    double Gamxa = gupxx * l_Gamxxx + gupyy * l_Gamxyy + gupzz * l_Gamxzz + TWO * (gupxy * l_Gamxxy + gupxz * l_Gamxxz + gupyz * l_Gamxyz);
    double Gamya = gupxx * l_Gamyxx + gupyy * l_Gamyyy + gupzz * l_Gamyzz + TWO * (gupxy * l_Gamyxy + gupxz * l_Gamyxz + gupyz * l_Gamyyz);
    double Gamza = gupxx * l_Gamzxx + gupyy * l_Gamzyy + gupzz * l_Gamzzz + TWO * (gupxy * l_Gamzxy + gupxz * l_Gamzxz + gupyz * l_Gamzyz);

    gxxx = l_gxx * l_Gamxxx + l_gxy * l_Gamyxx + l_gxz * l_Gamzxx;
    gxyx = l_gxx * l_Gamxxy + l_gxy * l_Gamyxy + l_gxz * l_Gamzxy;
    gxzx = l_gxx * l_Gamxxz + l_gxy * l_Gamyxz + l_gxz * l_Gamzxz;
    gyyx = l_gxx * l_Gamxyy + l_gxy * l_Gamyyy + l_gxz * l_Gamzyy;
    gyzx = l_gxx * l_Gamxyz + l_gxy * l_Gamyyz + l_gxz * l_Gamzyz;
    gzzx = l_gxx * l_Gamxzz + l_gxy * l_Gamyzz + l_gxz * l_Gamzzz;

    gxxy = l_gxy * l_Gamxxx + l_gyy * l_Gamyxx + l_gyz * l_Gamzxx;
    gxyy = l_gxy * l_Gamxxy + l_gyy * l_Gamyxy + l_gyz * l_Gamzxy;
    gxzy = l_gxy * l_Gamxxz + l_gyy * l_Gamyxz + l_gyz * l_Gamzxz;
    gyyy = l_gxy * l_Gamxyy + l_gyy * l_Gamyyy + l_gyz * l_Gamzyy;
    gyzy = l_gxy * l_Gamxyz + l_gyy * l_Gamyyz + l_gyz * l_Gamzyz;
    gzzy = l_gxy * l_Gamxzz + l_gyy * l_Gamyzz + l_gyz * l_Gamzzz;

    gxxz = l_gxz * l_Gamxxx + l_gyz * l_Gamyxx + l_gzz * l_Gamzxx;
    gxyz = l_gxz * l_Gamxxy + l_gyz * l_Gamyxy + l_gzz * l_Gamzxy;
    gxzz = l_gxz * l_Gamxxz + l_gyz * l_Gamyxz + l_gzz * l_Gamzxz;
    gyyz = l_gxz * l_Gamxyy + l_gyz * l_Gamyyy + l_gzz * l_Gamzyy;
    gyzz = l_gxz * l_Gamxyz + l_gyz * l_Gamyyz + l_gzz * l_Gamzyz;
    gzzz = l_gxz * l_Gamxzz + l_gyz * l_Gamyzz + l_gzz * l_Gamzzz;

    double l_Rxx = gupxx * dxx_xx + gupyy * dxx_yy + gupzz * dxx_zz + (gupxy * dxx_xy + gupxz * dxx_xz + gupyz * dxx_yz) * TWO;
    double l_Ryy = gupxx * dyy_xx + gupyy * dyy_yy + gupzz * dyy_zz + (gupxy * dyy_xy + gupxz * dyy_xz + gupyz * dyy_yz) * TWO;
    double l_Rzz = gupxx * dzz_xx + gupyy * dzz_yy + gupzz * dzz_zz + (gupxy * dzz_xy + gupxz * dzz_xz + gupyz * dzz_yz) * TWO;
    double l_Rxy = gupxx * gxy_xx + gupyy * gxy_yy + gupzz * gxy_zz + (gupxy * gxy_xy + gupxz * gxy_xz + gupyz * gxy_yz) * TWO;
    double l_Rxz = gupxx * gxz_xx + gupyy * gxz_yy + gupzz * gxz_zz + (gupxy * gxz_xy + gupxz * gxz_xz + gupyz * gxz_yz) * TWO;
    double l_Ryz = gupxx * gyz_xx + gupyy * gyz_yy + gupzz * gyz_zz + (gupxy * gyz_xy + gupxz * gyz_xz + gupyz * gyz_yz) * TWO;

    l_Rxx = -HALF * l_Rxx + l_gxx * dGamxx + l_gxy * dGamyx + l_gxz * dGamzx + Gamxa * gxxx + Gamya * gxyx + Gamza * gxzx + 
          gupxx * (TWO*(l_Gamxxx*gxxx + l_Gamyxx*gxyx + l_Gamzxx*gxzx) + l_Gamxxx*gxxx + l_Gamyxx*gxxy + l_Gamzxx*gxxz) +
          gupxy * (TWO*(l_Gamxxx*gxyx + l_Gamyxx*gyyx + l_Gamzxx*gyzx + l_Gamxxy*gxxx + l_Gamyxy*gxyx + l_Gamzxy*gxzx) + l_Gamxxy*gxxx + l_Gamyxy*gxxy + l_Gamzxy*gxxz + l_Gamxxx*gxyx + l_Gamyxx*gxyy + l_Gamzxx*gxyz) + 
          gupxz * (TWO*(l_Gamxxx*gxzx + l_Gamyxx*gyzx + l_Gamzxx*gzzx + l_Gamxxz*gxxx + l_Gamyxz*gxyx + l_Gamzxz*gxzx) + l_Gamxxz*gxxx + l_Gamyxz*gxxy + l_Gamzxz*gxxz + l_Gamxxx*gxzx + l_Gamyxx*gxzy + l_Gamzxx*gxzz) + 
          gupyy * (TWO*(l_Gamxxy*gxyx + l_Gamyxy*gyyx + l_Gamzxy*gyzx) + l_Gamxxy*gxyx + l_Gamyxy*gxyy + l_Gamzxy*gxyz) + 
          gupyz * (TWO*(l_Gamxxy*gxzx + l_Gamyxy*gyzx + l_Gamzxy*gzzx + l_Gamxxz*gxyx + l_Gamyxz*gyyx + l_Gamzxz*gyzx) + l_Gamxxz*gxyx + l_Gamyxz*gxyy + l_Gamzxz*gxyz + l_Gamxxy*gxzx + l_Gamyxy*gxzy + l_Gamzxy*gxzz) + 
          gupzz * (TWO*(l_Gamxxz*gxzx + l_Gamyxz*gyzx + l_Gamzxz*gzzx) + l_Gamxxz*gxzx + l_Gamyxz*gxzy + l_Gamzxz*gxzz);

    l_Ryy = -HALF * l_Ryy + l_gxy * dGamxy + l_gyy * dGamyy + l_gyz * dGamzy + Gamxa * gxyy + Gamya * gyyy + Gamza * gyzy + 
          gupxx * (TWO*(l_Gamxxy*gxxy + l_Gamyxy*gxyy + l_Gamzxy*gxzy) + l_Gamxxy*gxyx + l_Gamyxy*gxyy + l_Gamzxy*gxyz) + 
          gupxy * (TWO*(l_Gamxxy*gxyy + l_Gamyxy*gyyy + l_Gamzxy*gyzy + l_Gamxyy*gxxy + l_Gamyyy*gxyy + l_Gamzyy*gxzy) + l_Gamxyy*gxyx + l_Gamyyy*gxyy + l_Gamzyy*gxyz + l_Gamxxy*gyyx + l_Gamyxy*gyyy + l_Gamzxy*gyyz) + 
          gupxz * (TWO*(l_Gamxxy*gxzy + l_Gamyxy*gyzy + l_Gamzxy*gzzy + l_Gamxyz*gxxy + l_Gamyyz*gxyy + l_Gamzyz*gxzy) + l_Gamxyz*gxyx + l_Gamyyz*gxyy + l_Gamzyz*gxyz + l_Gamxxy*gyzx + l_Gamyxy*gyzy + l_Gamzxy*gyzz) + 
          gupyy * (TWO*(l_Gamxyy*gxyy + l_Gamyyy*gyyy + l_Gamzyy*gyzy) + l_Gamxyy*gyyx + l_Gamyyy*gyyy + l_Gamzyy*gyyz) + 
          gupyz * (TWO*(l_Gamxyy*gxzy + l_Gamyyy*gyzy + l_Gamzyy*gzzy + l_Gamxyz*gxyy + l_Gamyyz*gyyy + l_Gamzyz*gyzy) + l_Gamxyz*gyyx + l_Gamyyz*gyyy + l_Gamzyz*gyyz + l_Gamxyy*gyzx + l_Gamyyy*gyzy + l_Gamzyy*gyzz) + 
          gupzz * (TWO*(l_Gamxyz*gxzy + l_Gamyyz*gyzy + l_Gamzyz*gzzy) + l_Gamxyz*gyzx + l_Gamyyz*gyzy + l_Gamzyz*gyzz);

    l_Rzz = -HALF * l_Rzz + l_gxz * dGamxz + l_gyz * dGamyz + l_gzz * dGamzz + Gamxa * gxzz + Gamya * gyzz + Gamza * gzzz + 
          gupxx * (TWO*(l_Gamxxz*gxxz + l_Gamyxz*gxyz + l_Gamzxz*gxzz) + l_Gamxxz*gxzx + l_Gamyxz*gxzy + l_Gamzxz*gxzz) + 
          gupxy * (TWO*(l_Gamxxz*gxyz + l_Gamyxz*gyyz + l_Gamzxz*gyzz + l_Gamxyz*gxxz + l_Gamyyz*gxyz + l_Gamzyz*gxzz) + l_Gamxyz*gxzx + l_Gamyyz*gxzy + l_Gamzyz*gxzz + l_Gamxxz*gyzx + l_Gamyxz*gyzy + l_Gamzxz*gyzz) + 
          gupxz * (TWO*(l_Gamxxz*gxzz + l_Gamyxz*gyzz + l_Gamzxz*gzzz + l_Gamxzz*gxxz + l_Gamyzz*gxyz + l_Gamzzz*gxzz) + l_Gamxzz*gxzx + l_Gamyzz*gxzy + l_Gamzzz*gxzz + l_Gamxxz*gzzx + l_Gamyxz*gzzy + l_Gamzxz*gzzz) + 
          gupyy * (TWO*(l_Gamxyz*gxyz + l_Gamyyz*gyyz + l_Gamzyz*gyzz) + l_Gamxyz*gyzx + l_Gamyyz*gyzy + l_Gamzyz*gyzz) + 
          gupyz * (TWO*(l_Gamxyz*gxzz + l_Gamyyz*gyzz + l_Gamzyz*gzzz + l_Gamxzz*gxyz + l_Gamyzz*gyyz + l_Gamzzz*gyzz) + l_Gamxzz*gyzx + l_Gamyzz*gyzy + l_Gamzzz*gyzz + l_Gamxyz*gzzx + l_Gamyyz*gzzy + l_Gamzyz*gzzz) + 
          gupzz * (TWO*(l_Gamxzz*gxzz + l_Gamyzz*gyzz + l_Gamzzz*gzzz) + l_Gamxzz*gzzx + l_Gamyzz*gzzy + l_Gamzzz*gzzz);

    l_Rxy = HALF * ( - l_Rxy + l_gxx * dGamxy + l_gxy * dGamyy + l_gxz * dGamzy + l_gxy * dGamxx + l_gyy * dGamyx + l_gyz * dGamzx + Gamxa * gxyx + Gamya * gyyx + Gamza * gyzx + Gamxa * gxxy + Gamya * gxyy + Gamza * gxzy) + 
          gupxx * (l_Gamxxx*gxxy + l_Gamyxx*gxyy + l_Gamzxx*gxzy + l_Gamxxy*gxxx + l_Gamyxy*gxyx + l_Gamzxy*gxzx + l_Gamxxx*gxyx + l_Gamyxx*gxyy + l_Gamzxx*gxyz) + 
          gupxy * (l_Gamxxx*gxyy + l_Gamyxx*gyyy + l_Gamzxx*gyzy + l_Gamxxy*gxyx + l_Gamyxy*gyyx + l_Gamzxy*gyzx + l_Gamxxy*gxyx + l_Gamyxy*gxyy + l_Gamzxy*gxyz + l_Gamxxy*gxxy + l_Gamyxy*gxyy + l_Gamzxy*gxzy + l_Gamxyy*gxxx + l_Gamyyy*gxyx + l_Gamzyy*gxzx + l_Gamxxx*gyyx + l_Gamyxx*gyyy + l_Gamzxx*gyyz) + 
          gupxz * (l_Gamxxx*gxzy + l_Gamyxx*gyzy + l_Gamzxx*gzzy + l_Gamxxy*gxzx + l_Gamyxy*gyzx + l_Gamzxy*gzzx + l_Gamxxz*gxyx + l_Gamyxz*gxyy + l_Gamzxz*gxyz + l_Gamxxz*gxxy + l_Gamyxz*gxyy + l_Gamzxz*gxzy + l_Gamxyz*gxxx + l_Gamyyz*gxyx + l_Gamzyz*gxzx + l_Gamxxx*gyzx + l_Gamyxx*gyzy + l_Gamzxx*gyzz) + 
          gupyy * (l_Gamxxy*gxyy + l_Gamyxy*gyyy + l_Gamzxy*gyzy + l_Gamxyy*gxyx + l_Gamyyy*gyyx + l_Gamzyy*gyzx + l_Gamxxy*gyyx + l_Gamyxy*gyyy + l_Gamzxy*gyyz) + 
          gupyz * (l_Gamxxy*gxzy + l_Gamyxy*gyzy + l_Gamzxy*gzzy + l_Gamxyy*gxzx + l_Gamyyy*gyzx + l_Gamzyy*gzzx + l_Gamxxz*gyyx + l_Gamyxz*gyyy + l_Gamzxz*gyyz + l_Gamxxz*gxyy + l_Gamyxz*gyyy + l_Gamzxz*gyzy + l_Gamxyz*gxyx + l_Gamyyz*gyyx + l_Gamzyz*gyzx + l_Gamxxy*gyzx + l_Gamyxy*gyzy + l_Gamzxy*gyzz) + 
          gupzz * (l_Gamxxz*gxzy + l_Gamyxz*gyzy + l_Gamzxz*gzzy + l_Gamxyz*gxzx + l_Gamyyz*gyzx + l_Gamzyz*gzzx + l_Gamxxz*gyzx + l_Gamyxz*gyzy + l_Gamzxz*gyzz);

    l_Rxz = HALF * ( - l_Rxz + l_gxx * dGamxz + l_gxy * dGamyz + l_gxz * dGamzz + l_gxz * dGamxx + l_gyz * dGamyx + l_gzz * dGamzx + Gamxa * gxzx + Gamya * gyzx + Gamza * gzzx + Gamxa * gxxz + Gamya * gxyz + Gamza * gxzz) + 
          gupxx * (l_Gamxxx*gxxz + l_Gamyxx*gxyz + l_Gamzxx*gxzz + l_Gamxxz*gxxx + l_Gamyxz*gxyx + l_Gamzxz*gxzx + l_Gamxxx*gxzx + l_Gamyxx*gxzy + l_Gamzxx*gxzz) + 
          gupxy * (l_Gamxxx*gxyz + l_Gamyxx*gyyz + l_Gamzxx*gyzz + l_Gamxxz*gxyx + l_Gamyxz*gyyx + l_Gamzxz*gyzx + l_Gamxxy*gxzx + l_Gamyxy*gxzy + l_Gamzxy*gxzz + l_Gamxxy*gxxz + l_Gamyxy*gxyz + l_Gamzxy*gxzz + l_Gamxyz*gxxx + l_Gamyyz*gxyx + l_Gamzyz*gxzx + l_Gamxxx*gyzx + l_Gamyxx*gyzy + l_Gamzxx*gyzz) + 
          gupxz * (l_Gamxxx*gxzz + l_Gamyxx*gyzz + l_Gamzxx*gzzz + l_Gamxxz*gxzx + l_Gamyxz*gyzx + l_Gamzxz*gzzx + l_Gamxxz*gxzx + l_Gamyxz*gxzy + l_Gamzxz*gxzz + l_Gamxxz*gxxz + l_Gamyxz*gxyz + l_Gamzxz*gxzz + l_Gamxzz*gxxx + l_Gamyzz*gxyx + l_Gamzzz*gxzx + l_Gamxxx*gzzx + l_Gamyxx*gzzy + l_Gamzxx*gzzz) + 
          gupyy * (l_Gamxxy*gxyz + l_Gamyxy*gyyz + l_Gamzxy*gyzz + l_Gamxyz*gxyx + l_Gamyyz*gyyx + l_Gamzyz*gyzx + l_Gamxxy*gyzx + l_Gamyxy*gyzy + l_Gamzxy*gyzz) + 
          gupyz * (l_Gamxxy*gxzz + l_Gamyxy*gyzz + l_Gamzxy*gzzz + l_Gamxyz*gxzx + l_Gamyyz*gyzx + l_Gamzyz*gzzx + l_Gamxxz*gyzx + l_Gamyxz*gyzy + l_Gamzxz*gyzz + l_Gamxxz*gxyz + l_Gamyxz*gyyz + l_Gamzxz*gyzz + l_Gamxzz*gxyx + l_Gamyzz*gyyx + l_Gamzzz*gyzx + l_Gamxxy*gzzx + l_Gamyxy*gzzy + l_Gamzxy*gzzz) + 
          gupzz * (l_Gamxxz*gxzz + l_Gamyxz*gyzz + l_Gamzxz*gzzz + l_Gamxzz*gxzx + l_Gamyzz*gyzx + l_Gamzzz*gzzx + l_Gamxxz*gzzx + l_Gamyxz*gzzy + l_Gamzxz*gzzz);

    l_Ryz = HALF * ( - l_Ryz + l_gxy * dGamxz + l_gyy * dGamyz + l_gyz * dGamzz + l_gxz * dGamxy + l_gyz * dGamyy + l_gzz * dGamzy + Gamxa * gxzy + Gamya * gyzy + Gamza * gzzy + Gamxa * gxyz + Gamya * gyyz + Gamza * gyzz) + 
          gupxx * (l_Gamxxy*gxxz + l_Gamyxy*gxyz + l_Gamzxy*gxzz + l_Gamxxz*gxxy + l_Gamyxz*gxyy + l_Gamzxz*gxzy + l_Gamxxy*gxzx + l_Gamyxy*gxzy + l_Gamzxy*gxzz) + 
          gupxy * (l_Gamxxy*gxyz + l_Gamyxy*gyyz + l_Gamzxy*gyzz + l_Gamxxz*gxyy + l_Gamyxz*gyyy + l_Gamzxz*gyzy + l_Gamxyy*gxzx + l_Gamyyy*gxzy + l_Gamzyy*gxzz + l_Gamxyy*gxxz + l_Gamyyy*gxyz + l_Gamzyy*gxzz + l_Gamxyz*gxxy + l_Gamyyz*gxyy + l_Gamzyz*gxzy + l_Gamxxy*gyzx + l_Gamyxy*gyzy + l_Gamzxy*gyzz) + 
          gupxz * (l_Gamxxy*gxzz + l_Gamyxy*gyzz + l_Gamzxy*gzzz + l_Gamxxz*gxzy + l_Gamyxz*gyzy + l_Gamzxz*gzzy + l_Gamxyz*gxzx + l_Gamyyz*gxzy + l_Gamzyz*gxzz + l_Gamxyz*gxxz + l_Gamyyz*gxyz + l_Gamzyz*gxzz + l_Gamxzz*gxxy + l_Gamyzz*gxyy + l_Gamzzz*gxzy + l_Gamxxy*gzzx + l_Gamyxy*gzzy + l_Gamzxy*gzzz) + 
          gupyy * (l_Gamxyy*gxyz + l_Gamyyy*gyyz + l_Gamzyy*gyzz + l_Gamxyz*gxyy + l_Gamyyz*gyyy + l_Gamzyz*gyzy + l_Gamxyy*gyzx + l_Gamyyy*gyzy + l_Gamzyy*gyzz) + 
          gupyz * (l_Gamxyy*gxzz + l_Gamyyy*gyzz + l_Gamzyy*gzzz + l_Gamxyz*gxzy + l_Gamyyz*gyzy + l_Gamzyz*gzzy + l_Gamxyz*gyzx + l_Gamyyz*gyzy + l_Gamzyz*gyzz + l_Gamxyz*gxyz + l_Gamyyz*gyyz + l_Gamzyz*gyzz + l_Gamxzz*gxyy + l_Gamyzz*gyyy + l_Gamzzz*gyzy + l_Gamxyy*gzzx + l_Gamyyy*gzzy + l_Gamzyy*gzzz) + 
          gupzz * (l_Gamxyz*gxzz + l_Gamyyz*gyzz + l_Gamzyz*gzzz + l_Gamxzz*gxzy + l_Gamyzz*gyzy + l_Gamzzz*gzzy + l_Gamxyz*gzzx + l_Gamyyz*gzzy + l_Gamzyz*gzzz);

    Gamxxx_out[idx] = l_Gamxxx; Gamxxy_out[idx] = l_Gamxxy; Gamxxz_out[idx] = l_Gamxxz;
    Gamxyy_out[idx] = l_Gamxyy; Gamxyz_out[idx] = l_Gamxyz; Gamxzz_out[idx] = l_Gamxzz;
    Gamyxx_out[idx] = l_Gamyxx; Gamyxy_out[idx] = l_Gamyxy; Gamyxz_out[idx] = l_Gamyxz;
    Gamyyy_out[idx] = l_Gamyyy; Gamyyz_out[idx] = l_Gamyyz; Gamyzz_out[idx] = l_Gamyzz;
    Gamzxx_out[idx] = l_Gamzxx; Gamzxy_out[idx] = l_Gamzxy; Gamzxz_out[idx] = l_Gamzxz;
    Gamzyy_out[idx] = l_Gamzyy; Gamzyz_out[idx] = l_Gamzyz; Gamzzz_out[idx] = l_Gamzzz;

    Rxx_out[idx] = l_Rxx; Ryy_out[idx] = l_Ryy; Rzz_out[idx] = l_Rzz;
    Rxy_out[idx] = l_Rxy; Rxz_out[idx] = l_Rxz; Ryz_out[idx] = l_Ryz;
}

__launch_bounds__(256, 2)
__global__ void bssn_rhs_eval_kernel(
    const int ex0, const int ex1, const int ex2, 
    const int symmetry, const int lev,
    const double* __restrict__ X, const double* __restrict__ Y, const double* __restrict__ Z,
    const double* __restrict__ chi, const double* __restrict__ trK,
    const double* __restrict__ dxx, const double* __restrict__ gxy, const double* __restrict__ gxz,
    const double* __restrict__ dyy, const double* __restrict__ gyz, const double* __restrict__ dzz,
    const double* __restrict__ Axx, const double* __restrict__ Axy, const double* __restrict__ Axz,
    const double* __restrict__ Ayy, const double* __restrict__ Ayz, const double* __restrict__ Azz,
    const double* __restrict__ Gamx, const double* __restrict__ Gamy, const double* __restrict__ Gamz,
    const double* __restrict__ Lap, 
    const double* __restrict__ betax, const double* __restrict__ betay, const double* __restrict__ betaz,
    const double* __restrict__ dtSfx, const double* __restrict__ dtSfy, const double* __restrict__ dtSfz,
    const double* __restrict__ rho, const double* __restrict__ Sx, const double* __restrict__ Sy, const double* __restrict__ Sz,
    const double* __restrict__ Sxx, const double* __restrict__ Sxy, const double* __restrict__ Sxz, 
    const double* __restrict__ Syy, const double* __restrict__ Syz, const double* __restrict__ Szz,
    const double* __restrict__ Gamxxx_in, const double* __restrict__ Gamxxy_in, const double* __restrict__ Gamxxz_in,
    const double* __restrict__ Gamxyy_in, const double* __restrict__ Gamxyz_in, const double* __restrict__ Gamxzz_in,
    const double* __restrict__ Gamyxx_in, const double* __restrict__ Gamyxy_in, const double* __restrict__ Gamyxz_in,
    const double* __restrict__ Gamyyy_in, const double* __restrict__ Gamyyz_in, const double* __restrict__ Gamyzz_in,
    const double* __restrict__ Gamzxx_in, const double* __restrict__ Gamzxy_in, const double* __restrict__ Gamzxz_in,
    const double* __restrict__ Gamzyy_in, const double* __restrict__ Gamzyz_in, const double* __restrict__ Gamzzz_in,
    const double* __restrict__ Rxx_in, const double* __restrict__ Rxy_in, const double* __restrict__ Rxz_in,
    const double* __restrict__ Ryy_in, const double* __restrict__ Ryz_in, const double* __restrict__ Rzz_in,
    double* __restrict__ chi_rhs, double* __restrict__ trK_rhs,
    double* __restrict__ gxx_rhs, double* __restrict__ gxy_rhs, double* __restrict__ gxz_rhs,
    double* __restrict__ gyy_rhs, double* __restrict__ gyz_rhs, double* __restrict__ gzz_rhs,
    double* __restrict__ Axx_rhs, double* __restrict__ Axy_rhs, double* __restrict__ Axz_rhs,
    double* __restrict__ Ayy_rhs, double* __restrict__ Ayz_rhs, double* __restrict__ Azz_rhs,
    double* __restrict__ Gamx_rhs, double* __restrict__ Gamy_rhs, double* __restrict__ Gamz_rhs,
    double* __restrict__ Lap_rhs, 
    double* __restrict__ betax_rhs, double* __restrict__ betay_rhs, double* __restrict__ betaz_rhs,
    double* __restrict__ dtSfx_rhs, double* __restrict__ dtSfy_rhs, double* __restrict__ dtSfz_rhs
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int dims[3] = {ex0, ex1, ex2};
    double dX = X[1] - X[0]; double dY = Y[1] - Y[0]; double dZ = Z[1] - Z[0];

    int imax = ex0 - 1; int jmax = ex1 - 1; int kmax = ex2 - 1;
    int imin = 0, jmin = 0, kmin = 0;
    if (symmetry > 0 && fabs(Z[0]) < dZ) kmin = -2;
    if (symmetry > 1 && fabs(X[0]) < dX) imin = -2;
    if (symmetry > 1 && fabs(Y[0]) < dY) jmin = -2;

    int block_i = blockIdx.x * blockDim.x;
    int block_j = blockIdx.y * blockDim.y;
    int block_k = blockIdx.z * blockDim.z;

    __shared__ double smem[8][12][12];

    // ==============================================================================
    // BATCH DERIVATIVES: 统一进入 Smem 流水线计算
    // ==============================================================================
    
    // 1. Beta (Shift 向量) 及其一阶导数
    double betaxx, betaxy, betaxz, bx_gxxx, bx_gxyx, bx_gxzx, bx_gyyx, bx_gyzx, bx_gzzx;
    load_field_to_smem(smem, betax, dims, block_i, block_j, block_k, ANTI, SYM, SYM);
    __syncthreads(); compute_all_derivs_smem(smem, &betaxx, &betaxy, &betaxz, &bx_gxxx, &bx_gxyx, &bx_gxzx, &bx_gyyx, &bx_gyzx, &bx_gzzx, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double betayx, betayy, betayz, by_gxxy, by_gxyy, by_gxzy, by_gyyy, by_gyzy, by_gzzy;
    load_field_to_smem(smem, betay, dims, block_i, block_j, block_k, SYM, ANTI, SYM);
    __syncthreads(); compute_all_derivs_smem(smem, &betayx, &betayy, &betayz, &by_gxxy, &by_gxyy, &by_gxzy, &by_gyyy, &by_gyzy, &by_gzzy, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double betazx, betazy, betazz, bz_gxxz, bz_gxyz, bz_gxzz, bz_gyyz, bz_gyzz, bz_gzzz;
    load_field_to_smem(smem, betaz, dims, block_i, block_j, block_k, SYM, SYM, ANTI);
    __syncthreads(); compute_all_derivs_smem(smem, &betazx, &betazy, &betazz, &bz_gxxz, &bz_gxyz, &bz_gxzz, &bz_gyyz, &bz_gyzz, &bz_gzzz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    // 2. Lap (Lapse) - 需一阶和二阶导数 (消除极度昂贵的 d_fdderivs_point)
    double Lapx, Lapy, Lapz, fxx_Lap, fxy_Lap, fxz_Lap, fyy_Lap, fyz_Lap, fzz_Lap;
    load_field_to_smem(smem, Lap, dims, block_i, block_j, block_k, SYM, SYM, SYM);
    __syncthreads(); compute_all_derivs_smem(smem, &Lapx, &Lapy, &Lapz, &fxx_Lap, &fxy_Lap, &fxz_Lap, &fyy_Lap, &fyz_Lap, &fzz_Lap, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    // 3. chi (保角因子) - 需一阶和二阶导数 (消除极度昂贵的 d_fdderivs_point)
    double chix, chiy, chiz, chixx, chixy, chixz, chiyy, chiyz, chizz;
    load_field_to_smem(smem, chi, dims, block_i, block_j, block_k, SYM, SYM, SYM);
    __syncthreads(); compute_all_derivs_smem(smem, &chix, &chiy, &chiz, &chixx, &chixy, &chixz, &chiyy, &chiyz, &chizz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    // 4. trK - 仅需一阶导数，使用我们刚写的特化高性能版本 (消除 dum 占位符)
    double Kx, Ky, Kz;
    load_field_to_smem(smem, trK, dims, block_i, block_j, block_k, SYM, SYM, SYM);
    __syncthreads(); compute_first_derivs_smem(smem, &Kx, &Ky, &Kz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax);
    // 此后不再往 smem 载入新变量，故无需最后一次 __syncthreads()

    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    int idx = IDX3D(i, j, k, ex0, ex1, ex2);

    const double ONE = 1.0; const double TWO = 2.0; const double F1o3 = 1.0 / 3.0; const double F2o3 = 2.0 / 3.0;
    const double F3o2 = 1.5; const double HALF = 0.5; const double EIGHT = 8.0; const double FOUR = 4.0; const double F16 = 16.0; const double PI = M_PI;

    double val_Lap = Lap[idx]; double val_chi = chi[idx]; double val_trK = trK[idx];
    double alpn1 = val_Lap + ONE; double chin1 = val_chi + ONE;
    double inv_chin1 = 1.0 / chin1;

    double l_gxx = dxx[idx] + ONE; double l_gxy = gxy[idx]; double l_gxz = gxz[idx];
    double l_gyy = dyy[idx] + ONE; double l_gyz = gyz[idx]; double l_gzz = dzz[idx] + ONE;
    double l_Axx = Axx[idx]; double l_Axy = Axy[idx]; double l_Axz = Axz[idx];
    double l_Ayy = Ayy[idx]; double l_Ayz = Ayz[idx]; double l_Azz = Azz[idx];

    double div_beta = betaxx + betayy + betazz;

    chi_rhs[idx] = F2o3 * chin1 * (alpn1 * val_trK - div_beta);

    gxx_rhs[idx] = -TWO * alpn1 * l_Axx - F2o3 * l_gxx * div_beta + TWO * (l_gxx * betaxx + l_gxy * betayx + l_gxz * betazx);
    gyy_rhs[idx] = -TWO * alpn1 * l_Ayy - F2o3 * l_gyy * div_beta + TWO * (l_gxy * betaxy + l_gyy * betayy + l_gyz * betazy);
    gzz_rhs[idx] = -TWO * alpn1 * l_Azz - F2o3 * l_gzz * div_beta + TWO * (l_gxz * betaxz + l_gyz * betayz + l_gzz * betazz);
    gxy_rhs[idx] = -TWO * alpn1 * l_Axy + F1o3 * l_gxy * div_beta + l_gxx * betaxy + l_gxz * betazy + l_gyy * betayx + l_gyz * betazx - l_gxy * betazz;
    gyz_rhs[idx] = -TWO * alpn1 * l_Ayz + F1o3 * l_gyz * div_beta + l_gxy * betaxz + l_gyy * betayz + l_gxz * betaxy + l_gzz * betazy - l_gyz * betaxx;
    gxz_rhs[idx] = -TWO * alpn1 * l_Axz + F1o3 * l_gxz * div_beta + l_gxx * betaxz + l_gxy * betayz + l_gyz * betayx + l_gzz * betazx - l_gxz * betayy;

    double gupzz = l_gxx * l_gyy * l_gzz + l_gxy * l_gyz * l_gxz + l_gxz * l_gxy * l_gyz - l_gxz * l_gyy * l_gxz - l_gxy * l_gxy * l_gzz - l_gxx * l_gyz * l_gyz;
    double gupxx = (l_gyy * l_gzz - l_gyz * l_gyz) / gupzz;
    double gupxy = -(l_gxy * l_gzz - l_gyz * l_gxz) / gupzz;
    double gupxz = (l_gxy * l_gyz - l_gyy * l_gxz) / gupzz;
    double gupyy = (l_gxx * l_gzz - l_gxz * l_gxz) / gupzz;
    double gupyz = -(l_gxx * l_gyz - l_gxy * l_gxz) / gupzz;
    gupzz = (l_gxx * l_gyy - l_gxy * l_gxy) / gupzz;

    // 【剥离脏复用】：显式计算混合张量 A^i_j (消除大量重复冗余算式)
    double Au_d_xx = gupxx * l_Axx + gupxy * l_Axy + gupxz * l_Axz;
    double Au_d_xy = gupxx * l_Axy + gupxy * l_Ayy + gupxz * l_Ayz;
    double Au_d_xz = gupxx * l_Axz + gupxy * l_Ayz + gupxz * l_Azz;
    double Au_d_yx = gupxy * l_Axx + gupyy * l_Axy + gupyz * l_Axz;
    double Au_d_yy = gupxy * l_Axy + gupyy * l_Ayy + gupyz * l_Ayz;
    double Au_d_yz = gupxy * l_Axz + gupyy * l_Ayz + gupyz * l_Azz;
    double Au_d_zx = gupxz * l_Axx + gupyz * l_Axy + gupzz * l_Axz;
    double Au_d_zy = gupxz * l_Axy + gupyz * l_Ayy + gupzz * l_Ayz;
    double Au_d_zz = gupxz * l_Axz + gupyz * l_Ayz + gupzz * l_Azz;

    // 计算上指标外曲率 A^{ij} 
    double Aup_xx = Au_d_xx * gupxx + Au_d_xy * gupxy + Au_d_xz * gupxz;
    double Aup_xy = Au_d_xx * gupxy + Au_d_xy * gupyy + Au_d_xz * gupyz;
    double Aup_xz = Au_d_xx * gupxz + Au_d_xy * gupyz + Au_d_xz * gupzz;
    double Aup_yy = Au_d_yx * gupxy + Au_d_yy * gupyy + Au_d_yz * gupyz;
    double Aup_yz = Au_d_yx * gupxz + Au_d_yy * gupyz + Au_d_yz * gupzz;
    double Aup_zz = Au_d_zx * gupxz + Au_d_zy * gupyz + Au_d_zz * gupzz;

    // 引入真实的共形几何变量
    double l_Gamxxx = Gamxxx_in[idx]; double l_Gamxxy = Gamxxy_in[idx]; double l_Gamxxz = Gamxxz_in[idx];
    double l_Gamxyy = Gamxyy_in[idx]; double l_Gamxyz = Gamxyz_in[idx]; double l_Gamxzz = Gamxzz_in[idx];
    double l_Gamyxx = Gamyxx_in[idx]; double l_Gamyxy = Gamyxy_in[idx]; double l_Gamyxz = Gamyxz_in[idx];
    double l_Gamyyy = Gamyyy_in[idx]; double l_Gamyyz = Gamyyz_in[idx]; double l_Gamyzz = Gamyzz_in[idx];
    double l_Gamzxx = Gamzxx_in[idx]; double l_Gamzxy = Gamzxy_in[idx]; double l_Gamzxz = Gamzxz_in[idx];
    double l_Gamzyy = Gamzyy_in[idx]; double l_Gamzyz = Gamzyz_in[idx]; double l_Gamzzz = Gamzzz_in[idx];

    // l_Rxx 这里必须、永远、只能是里奇张量
    double l_Rxx = Rxx_in[idx]; double l_Rxy = Rxy_in[idx]; double l_Rxz = Rxz_in[idx];
    double l_Ryy = Ryy_in[idx]; double l_Ryz = Ryz_in[idx]; double l_Rzz = Rzz_in[idx];

    double val_Sx = Sx[idx]; double val_Sy = Sy[idx]; double val_Sz = Sz[idx];

    // 【修复 $\Gamma^i$ RHS 方程】：将原代码复用槽位中错误的 l_Rxx 替换回正确的 Aup_xx
    double val_Gamx_rhs = - TWO * (Lapx * Aup_xx + Lapy * Aup_xy + Lapz * Aup_xz) + 
        TWO * alpn1 * (-F3o2 * inv_chin1 * (chix * Aup_xx + chiy * Aup_xy + chiz * Aup_xz) - gupxx * (F2o3 * Kx + EIGHT * PI * val_Sx) - gupxy * (F2o3 * Ky + EIGHT * PI * val_Sy) - gupxz * (F2o3 * Kz + EIGHT * PI * val_Sz) + 
        l_Gamxxx * Aup_xx + l_Gamxyy * Aup_yy + l_Gamxzz * Aup_zz + TWO * (l_Gamxxy * Aup_xy + l_Gamxxz * Aup_xz + l_Gamxyz * Aup_yz));

    double val_Gamy_rhs = - TWO * (Lapx * Aup_xy + Lapy * Aup_yy + Lapz * Aup_yz) + 
        TWO * alpn1 * (-F3o2 * inv_chin1 * (chix * Aup_xy + chiy * Aup_yy + chiz * Aup_yz) - gupxy * (F2o3 * Kx + EIGHT * PI * val_Sx) - gupyy * (F2o3 * Ky + EIGHT * PI * val_Sy) - gupyz * (F2o3 * Kz + EIGHT * PI * val_Sz) + 
        l_Gamyxx * Aup_xx + l_Gamyyy * Aup_yy + l_Gamyzz * Aup_zz + TWO * (l_Gamyxy * Aup_xy + l_Gamyxz * Aup_xz + l_Gamyyz * Aup_yz));

    double val_Gamz_rhs = - TWO * (Lapx * Aup_xz + Lapy * Aup_yz + Lapz * Aup_zz) + 
        TWO * alpn1 * (-F3o2 * inv_chin1 * (chix * Aup_xz + chiy * Aup_yz + chiz * Aup_zz) - gupxz * (F2o3 * Kx + EIGHT * PI * val_Sx) - gupyz * (F2o3 * Ky + EIGHT * PI * val_Sy) - gupzz * (F2o3 * Kz + EIGHT * PI * val_Sz) + 
        l_Gamzxx * Aup_xx + l_Gamzyy * Aup_yy + l_Gamzzz * Aup_zz + TWO * (l_Gamzxy * Aup_xy + l_Gamzxz * Aup_xz + l_Gamzyz * Aup_yz));

    double fxx_beta = bx_gxxx + by_gxyy + bz_gxzz;
    double fxy_beta = bx_gxyx + by_gyyy + bz_gyzz;
    double fxz_beta = bx_gxzx + by_gyzy + bz_gzzz;

    double Gamxa = gupxx * l_Gamxxx + gupyy * l_Gamxyy + gupzz * l_Gamxzz + TWO * (gupxy * l_Gamxxy + gupxz * l_Gamxxz + gupyz * l_Gamxyz);
    double Gamya = gupxx * l_Gamyxx + gupyy * l_Gamyyy + gupzz * l_Gamyzz + TWO * (gupxy * l_Gamyxy + gupxz * l_Gamyxz + gupyz * l_Gamyyz);
    double Gamza = gupxx * l_Gamzxx + gupyy * l_Gamzyy + gupzz * l_Gamzzz + TWO * (gupxy * l_Gamzxy + gupxz * l_Gamzxz + gupyz * l_Gamzyz);
    
    val_Gamx_rhs += F2o3 * Gamxa * div_beta - (Gamxa * betaxx + Gamya * betaxy + Gamza * betaxz) + F1o3 * (gupxx * fxx_beta + gupxy * fxy_beta + gupxz * fxz_beta) + gupxx * bx_gxxx + gupyy * bx_gyyx + gupzz * bx_gzzx + TWO * (gupxy * bx_gxyx + gupxz * bx_gxzx + gupyz * bx_gyzx);
    val_Gamy_rhs += F2o3 * Gamya * div_beta - (Gamxa * betayx + Gamya * betayy + Gamza * betayz) + F1o3 * (gupxy * fxx_beta + gupyy * fxy_beta + gupyz * fxz_beta) + gupxx * by_gxxy + gupyy * by_gyyy + gupzz * by_gzzy + TWO * (gupxy * by_gxyy + gupxz * by_gxzy + gupyz * by_gyzy);
    val_Gamz_rhs += F2o3 * Gamza * div_beta - (Gamxa * betazx + Gamya * betazy + Gamza * betazz) + F1o3 * (gupxz * fxx_beta + gupyz * fxy_beta + gupzz * fxz_beta) + gupxx * bz_gxxz + gupyy * bz_gyyz + gupzz * bz_gzzz + TWO * (gupxy * bz_gxyz + gupxz * bz_gxzz + gupyz * bz_gyzz);
    
    // 【物理升级】：利用提取到 Smem 的 chixx 计算 fxx
    double fxx = chixx - l_Gamxxx * chix - l_Gamyxx * chiy - l_Gamzxx * chiz;
    double fxy = chixy - l_Gamxxy * chix - l_Gamyxy * chiy - l_Gamzxy * chiz;
    double fxz = chixz - l_Gamxxz * chix - l_Gamyxz * chiy - l_Gamzxz * chiz;
    double fyy = chiyy - l_Gamxyy * chix - l_Gamyyy * chiy - l_Gamzyy * chiz;
    double fyz = chiyz - l_Gamxyz * chix - l_Gamyyz * chiy - l_Gamzyz * chiz;
    double fzz = chizz - l_Gamxzz * chix - l_Gamyzz * chiy - l_Gamzzz * chiz;

    double f_scalar = gupxx * (fxx - F3o2*inv_chin1 * chix * chix) + gupyy * (fyy - F3o2*inv_chin1 * chiy * chiy) + gupzz * (fzz - F3o2*inv_chin1 * chiz * chiz) + 
                      TWO * (gupxy * (fxy - F3o2*inv_chin1 * chix * chiy) + gupxz * (fxz - F3o2*inv_chin1 * chix * chiz) + gupyz * (fyz - F3o2*inv_chin1 * chiy * chiz));
    
    // 【里奇张量升级】
    l_Rxx += (fxx - chix*chix*inv_chin1*HALF + l_gxx * f_scalar)*inv_chin1*HALF;
    l_Ryy += (fyy - chiy*chiy*inv_chin1*HALF + l_gyy * f_scalar)*inv_chin1*HALF;
    l_Rzz += (fzz - chiz*chiz*inv_chin1*HALF + l_gzz * f_scalar)*inv_chin1*HALF;
    l_Rxy += (fxy - chix*chiy*inv_chin1*HALF + l_gxy * f_scalar)*inv_chin1*HALF;
    l_Rxz += (fxz - chix*chiz*inv_chin1*HALF + l_gxz * f_scalar)*inv_chin1*HALF;
    l_Ryz += (fyz - chiy*chiz*inv_chin1*HALF + l_gyz * f_scalar)*inv_chin1*HALF;

    double gx_phy = (gupxx * chix + gupxy * chiy + gupxz * chiz) * inv_chin1;
    double gy_phy = (gupxy * chix + gupyy * chiy + gupyz * chiz) * inv_chin1;
    double gz_phy = (gupxz * chix + gupyz * chiy + gupzz * chiz) * inv_chin1;
    
    l_Gamxxx -= ((chix + chix)*inv_chin1 - l_gxx * gx_phy)*HALF; 
    l_Gamyxx -= (                        - l_gxx * gy_phy)*HALF; 
    l_Gamzxx -= (                        - l_gxx * gz_phy)*HALF; 
    l_Gamxyy -= (                        - l_gyy * gx_phy)*HALF; 
    l_Gamyyy -= ((chiy + chiy)*inv_chin1 - l_gyy * gy_phy)*HALF; 
    l_Gamzyy -= (                        - l_gyy * gz_phy)*HALF; 
    l_Gamxzz -= (                        - l_gzz * gx_phy)*HALF; 
    l_Gamyzz -= (                        - l_gzz * gy_phy)*HALF; 
    l_Gamzzz -= ((chiz + chiz)*inv_chin1 - l_gzz * gz_phy)*HALF; 

    l_Gamxxy -= (chiy*inv_chin1 - l_gxy * gx_phy)*HALF; 
    l_Gamyxy -= (chix*inv_chin1 - l_gxy * gy_phy)*HALF; 
    l_Gamzxy -= (             - l_gxy * gz_phy)*HALF; 
    l_Gamxxz -= (chiz*inv_chin1 - l_gxz * gx_phy)*HALF; 
    l_Gamyxz -= (             - l_gxz * gy_phy)*HALF; 
    l_Gamzxz -= (chix*inv_chin1 - l_gxz * gz_phy)*HALF; 
    l_Gamxyz -= (             - l_gyz * gx_phy)*HALF; 
    l_Gamyyz -= (chiz*inv_chin1 - l_gyz * gy_phy)*HALF; 
    l_Gamzyz -= (chiy*inv_chin1 - l_gyz * gz_phy)*HALF; 

    // 【Lap 物理升级】：利用提取到 Smem 的 fxx_Lap 计算
    fxx_Lap = fxx_Lap - l_Gamxxx*Lapx - l_Gamyxx*Lapy - l_Gamzxx*Lapz;
    fyy_Lap = fyy_Lap - l_Gamxyy*Lapx - l_Gamyyy*Lapy - l_Gamzyy*Lapz;
    fzz_Lap = fzz_Lap - l_Gamxzz*Lapx - l_Gamyzz*Lapy - l_Gamzzz*Lapz;
    fxy_Lap = fxy_Lap - l_Gamxxy*Lapx - l_Gamyxy*Lapy - l_Gamzxy*Lapz;
    fxz_Lap = fxz_Lap - l_Gamxxz*Lapx - l_Gamyxz*Lapy - l_Gamzxz*Lapz;
    fyz_Lap = fyz_Lap - l_Gamxyz*Lapx - l_Gamyyz*Lapy - l_Gamzyz*Lapz;

    double trK_rhs_val = gupxx * fxx_Lap + gupyy * fyy_Lap + gupzz * fzz_Lap + TWO* (gupxy * fxy_Lap + gupxz * fxz_Lap + gupyz * fyz_Lap);
    double S = chin1 * (gupxx * Sxx[idx] + gupyy * Syy[idx] + gupzz * Szz[idx] + TWO * (gupxy * Sxy[idx] + gupxz * Sxz[idx] + gupyz * Syz[idx]));

    // 【化简优化】：利用剥离的 Au_d 极简计算 term_xx
    double term_xx = l_Axx * Au_d_xx + l_Axy * Au_d_yx + l_Axz * Au_d_zx;
    double term_xy = l_Axx * Au_d_xy + l_Axy * Au_d_yy + l_Axz * Au_d_zy;
    double term_xz = l_Axx * Au_d_xz + l_Axy * Au_d_yz + l_Axz * Au_d_zz;
    double term_yy = l_Axy * Au_d_xy + l_Ayy * Au_d_yy + l_Ayz * Au_d_zy;
    double term_yz = l_Axy * Au_d_xz + l_Ayy * Au_d_yz + l_Ayz * Au_d_zz;
    double term_zz = l_Axz * Au_d_xz + l_Ayz * Au_d_yz + l_Azz * Au_d_zz;

    double trA2 = gupxx * term_xx + gupyy * term_yy + gupzz * term_zz + TWO * (gupxy * term_xy + gupxz * term_xz + gupyz * term_yz);

    double f = F2o3 * val_trK * val_trK - trA2 - F16*PI*rho[idx] + EIGHT*PI*S;
    double f_trace = -F1o3 * (trK_rhs_val + alpn1*inv_chin1 * f);

    double src_xx = alpn1 * (l_Rxx - EIGHT*PI*Sxx[idx]) - fxx_Lap;
    double src_yy = alpn1 * (l_Ryy - EIGHT*PI*Syy[idx]) - fyy_Lap;
    double src_zz = alpn1 * (l_Rzz - EIGHT*PI*Szz[idx]) - fzz_Lap;
    double src_xy = alpn1 * (l_Rxy - EIGHT*PI*Sxy[idx]) - fxy_Lap;
    double src_xz = alpn1 * (l_Rxz - EIGHT*PI*Sxz[idx]) - fxz_Lap;
    double src_yz = alpn1 * (l_Ryz - EIGHT*PI*Syz[idx]) - fyz_Lap;

    Axx_rhs[idx] = chin1 * (src_xx - l_gxx * f_trace) + alpn1 * (val_trK * l_Axx - TWO * term_xx) + TWO * (l_Axx * betaxx + l_Axy * betayx + l_Axz * betazx) - F2o3 * l_Axx * div_beta;
    Ayy_rhs[idx] = chin1 * (src_yy - l_gyy * f_trace) + alpn1 * (val_trK * l_Ayy - TWO * term_yy) + TWO * (l_Axy * betaxy + l_Ayy * betayy + l_Ayz * betazy) - F2o3 * l_Ayy * div_beta;
    Azz_rhs[idx] = chin1 * (src_zz - l_gzz * f_trace) + alpn1 * (val_trK * l_Azz - TWO * term_zz) + TWO * (l_Axz * betaxz + l_Ayz * betayz + l_Azz * betazz) - F2o3 * l_Azz * div_beta;
    
    Axy_rhs[idx] = chin1 * (src_xy - l_gxy * f_trace) + alpn1 * (val_trK * l_Axy - TWO * term_xy) + l_Axx * betaxy + l_Axz * betazy + l_Ayy * betayx + l_Ayz * betazx - l_Axy * betazz + F1o3 * l_Axy * div_beta;
    Ayz_rhs[idx] = chin1 * (src_yz - l_gyz * f_trace) + alpn1 * (val_trK * l_Ayz - TWO * term_yz) + l_Axy * betaxz + l_Ayy * betayz + l_Axz * betaxy + l_Azz * betazy - l_Ayz * betaxx + F1o3 * l_Ayz * div_beta;
    Axz_rhs[idx] = chin1 * (src_xz - l_gxz * f_trace) + alpn1 * (val_trK * l_Axz - TWO * term_xz) + l_Axx * betaxz + l_Axy * betayz + l_Ayz * betayx + l_Azz * betazx - l_Axz * betayy + F1o3 * l_Axz * div_beta;

    trK_rhs[idx] = -chin1 * trK_rhs_val + alpn1 * (F1o3 * val_trK * val_trK + trA2 + FOUR * PI * (rho[idx] + S));

    const double FF = 0.75; const double eta = 2.0;
    Lap_rhs[idx] = -TWO * alpn1 * val_trK;
    betax_rhs[idx] = FF * dtSfx[idx]; betay_rhs[idx] = FF * dtSfy[idx]; betaz_rhs[idx] = FF * dtSfz[idx];
    dtSfx_rhs[idx] = val_Gamx_rhs - eta * dtSfx[idx];
    dtSfy_rhs[idx] = val_Gamy_rhs - eta * dtSfy[idx];
    dtSfz_rhs[idx] = val_Gamz_rhs - eta * dtSfz[idx];

    Gamx_rhs[idx] = val_Gamx_rhs; Gamy_rhs[idx] = val_Gamy_rhs; Gamz_rhs[idx] = val_Gamz_rhs;
}

__launch_bounds__(256, 2)
__global__ void bssn_advection_dissipation_kernel(
    const int ex0, const int ex1, const int ex2,
    const int symmetry, const int lev, const double eps,
    const double* __restrict__ X, const double* __restrict__ Y, const double* __restrict__ Z,
    const double* __restrict__ betax, const double* __restrict__ betay, const double* __restrict__ betaz,
    const double* __restrict__ dxx, const double* __restrict__ gxy, const double* __restrict__ gxz,
    const double* __restrict__ dyy, const double* __restrict__ gyz, const double* __restrict__ dzz,
    const double* __restrict__ Axx, const double* __restrict__ Axy, const double* __restrict__ Axz,
    const double* __restrict__ Ayy, const double* __restrict__ Ayz, const double* __restrict__ Azz,
    const double* __restrict__ chi, const double* __restrict__ trK,
    const double* __restrict__ Gamx, const double* __restrict__ Gamy, const double* __restrict__ Gamz,
    const double* __restrict__ Lap,
    const double* __restrict__ dtSfx, const double* __restrict__ dtSfy, const double* __restrict__ dtSfz,
    double* __restrict__ gxx_rhs, double* __restrict__ gxy_rhs, double* __restrict__ gxz_rhs,
    double* __restrict__ gyy_rhs, double* __restrict__ gyz_rhs, double* __restrict__ gzz_rhs,
    double* __restrict__ Axx_rhs, double* __restrict__ Axy_rhs, double* __restrict__ Axz_rhs,
    double* __restrict__ Ayy_rhs, double* __restrict__ Ayz_rhs, double* __restrict__ Azz_rhs,
    double* __restrict__ chi_rhs, double* __restrict__ trK_rhs,
    double* __restrict__ Gamx_rhs, double* __restrict__ Gamy_rhs, double* __restrict__ Gamz_rhs,
    double* __restrict__ Lap_rhs,
    double* __restrict__ betax_rhs, double* __restrict__ betay_rhs, double* __restrict__ betaz_rhs,
    double* __restrict__ dtSfx_rhs, double* __restrict__ dtSfy_rhs, double* __restrict__ dtSfz_rhs
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int dims[3] = {ex0, ex1, ex2};
    int block_i = blockIdx.x * blockDim.x;
    int block_j = blockIdx.y * blockDim.y;
    int block_k = blockIdx.z * blockDim.z;

    double dX = X[1] - X[0]; double dY = Y[1] - Y[0]; double dZ = Z[1] - Z[0];
    double d12dx = 1.0 / 12.0 / dX;
    double d12dy = 1.0 / 12.0 / dY;
    double d12dz = 1.0 / 12.0 / dZ;

    int imax = ex0 - 1; int jmax = ex1 - 1; int kmax = ex2 - 1;

    int imin = 0, jmin = 0, kmin = 0;
    if (symmetry > 0 && fabs(Z[0]) < dZ) kmin = -3;
    if (symmetry > 1 && fabs(X[0]) < dX) imin = -3;
    if (symmetry > 1 && fabs(Y[0]) < dY) jmin = -3;

    double vx = 0.0, vy = 0.0, vz = 0.0;
    int idx = 0;
    if (i <= imax && j <= jmax && k <= kmax) {
        idx = i + ex0 * (j + ex1 * k);
        // 使用 __ldg 强制走 Read-Only Data Cache，缓解带宽高压
        vx = __ldg(&betax[idx]);
        vy = __ldg(&betay[idx]);
        vz = __ldg(&betaz[idx]);
    }

    __shared__ double smem[10][14][14];

    auto fh = [&](int di, int dj, int dk) -> double {
        return smem[threadIdx.z + 3 + dk][threadIdx.y + 3 + dj][threadIdx.x + 3 + di];
    };

    auto process_field = [&](const double* f, double* f_rhs, double SYM1, double SYM2, double SYM3) {
        load_field_to_smem_rad3(smem, f, dims, block_i, block_j, block_k, SYM1, SYM2, SYM3);
        __syncthreads(); 

        if (i <= imax && j <= jmax && k <= kmax) {
            double rhs_add = 0.0;

            if (vx > 0.0) {
                if (i + 3 <= imax) { rhs_add += vx * d12dx * (-3.0*fh(-1,0,0) - 10.0*fh(0,0,0) + 18.0*fh(1,0,0) - 6.0*fh(2,0,0) + fh(3,0,0)); }
                else if (i + 2 <= imax) { rhs_add += vx * d12dx * (fh(-2,0,0) - 8.0*fh(-1,0,0) + 8.0*fh(1,0,0) - fh(2,0,0)); }
                else if (i + 1 <= imax) { rhs_add -= vx * d12dx * (-3.0*fh(1,0,0) - 10.0*fh(0,0,0) + 18.0*fh(-1,0,0) - 6.0*fh(-2,0,0) + fh(-3,0,0)); }
            } else if (vx < 0.0) {
                if (i - 3 >= imin) { rhs_add -= vx * d12dx * (-3.0*fh(1,0,0) - 10.0*fh(0,0,0) + 18.0*fh(-1,0,0) - 6.0*fh(-2,0,0) + fh(-3,0,0)); }
                else if (i - 2 >= imin) { rhs_add += vx * d12dx * (fh(-2,0,0) - 8.0*fh(-1,0,0) + 8.0*fh(1,0,0) - fh(2,0,0)); }
                else if (i - 1 >= imin) { rhs_add += vx * d12dx * (-3.0*fh(-1,0,0) - 10.0*fh(0,0,0) + 18.0*fh(1,0,0) - 6.0*fh(2,0,0) + fh(3,0,0)); }
            }
            if (vy > 0.0) {
                if (j + 3 <= jmax) { rhs_add += vy * d12dy * (-3.0*fh(0,-1,0) - 10.0*fh(0,0,0) + 18.0*fh(0,1,0) - 6.0*fh(0,2,0) + fh(0,3,0)); }
                else if (j + 2 <= jmax) { rhs_add += vy * d12dy * (fh(0,-2,0) - 8.0*fh(0,-1,0) + 8.0*fh(0,1,0) - fh(0,2,0)); }
                else if (j + 1 <= jmax) { rhs_add -= vy * d12dy * (-3.0*fh(0,1,0) - 10.0*fh(0,0,0) + 18.0*fh(0,-1,0) - 6.0*fh(0,-2,0) + fh(0,-3,0)); }
            } else if (vy < 0.0) {
                if (j - 3 >= jmin) { rhs_add -= vy * d12dy * (-3.0*fh(0,1,0) - 10.0*fh(0,0,0) + 18.0*fh(0,-1,0) - 6.0*fh(0,-2,0) + fh(0,-3,0)); }
                else if (j - 2 >= jmin) { rhs_add += vy * d12dy * (fh(0,-2,0) - 8.0*fh(0,-1,0) + 8.0*fh(0,1,0) - fh(0,2,0)); }
                else if (j - 1 >= jmin) { rhs_add += vy * d12dy * (-3.0*fh(0,-1,0) - 10.0*fh(0,0,0) + 18.0*fh(0,1,0) - 6.0*fh(0,2,0) + fh(0,3,0)); }
            }
            if (vz > 0.0) {
                if (k + 3 <= kmax) { rhs_add += vz * d12dz * (-3.0*fh(0,0,-1) - 10.0*fh(0,0,0) + 18.0*fh(0,0,1) - 6.0*fh(0,0,2) + fh(0,0,3)); }
                else if (k + 2 <= kmax) { rhs_add += vz * d12dz * (fh(0,0,-2) - 8.0*fh(0,0,-1) + 8.0*fh(0,0,1) - fh(0,0,2)); }
                else if (k + 1 <= kmax) { rhs_add -= vz * d12dz * (-3.0*fh(0,0,1) - 10.0*fh(0,0,0) + 18.0*fh(0,0,-1) - 6.0*fh(0,0,-2) + fh(0,0,-3)); }
            } else if (vz < 0.0) {
                if (k - 3 >= kmin) { rhs_add -= vz * d12dz * (-3.0*fh(0,0,1) - 10.0*fh(0,0,0) + 18.0*fh(0,0,-1) - 6.0*fh(0,0,-2) + fh(0,0,-3)); }
                else if (k - 2 >= kmin) { rhs_add += vz * d12dz * (fh(0,0,-2) - 8.0*fh(0,0,-1) + 8.0*fh(0,0,1) - fh(0,0,2)); }
                else if (k - 1 >= kmin) { rhs_add += vz * d12dz * (-3.0*fh(0,0,-1) - 10.0*fh(0,0,0) + 18.0*fh(0,0,1) - 6.0*fh(0,0,2) + fh(0,0,3)); }
            }

            if (eps > 0.0 && i - 3 >= imin && i + 3 <= imax && j - 3 >= jmin && j + 3 <= jmax && k - 3 >= kmin && k + 3 <= kmax) {
                rhs_add += eps / 64.0 * (
                    ((fh(-3,0,0) + fh(3,0,0)) - 6.0*(fh(-2,0,0) + fh(2,0,0)) + 15.0*(fh(-1,0,0) + fh(1,0,0)) - 20.0*fh(0,0,0)) / dX +
                    ((fh(0,-3,0) + fh(0,3,0)) - 6.0*(fh(0,-2,0) + fh(0,2,0)) + 15.0*(fh(0,-1,0) + fh(0,1,0)) - 20.0*fh(0,0,0)) / dY +
                    ((fh(0,0,-3) + fh(0,0,3)) - 6.0*(fh(0,0,-2) + fh(0,0,2)) + 15.0*(fh(0,0,-1) + fh(0,0,1)) - 20.0*fh(0,0,0)) / dZ
                );
            }

            f_rhs[idx] += rhs_add;
        }
        __syncthreads(); 
    };

    process_field(dxx, gxx_rhs, SYM, SYM, SYM);
    process_field(gxy, gxy_rhs, ANTI, ANTI, SYM);
    process_field(gxz, gxz_rhs, ANTI, SYM, ANTI);
    process_field(dyy, gyy_rhs, SYM, SYM, SYM);
    process_field(gyz, gyz_rhs, SYM, ANTI, ANTI);
    process_field(dzz, gzz_rhs, SYM, SYM, SYM);
    
    process_field(Axx, Axx_rhs, SYM, SYM, SYM);
    process_field(Axy, Axy_rhs, ANTI, ANTI, SYM);
    process_field(Axz, Axz_rhs, ANTI, SYM, ANTI);
    process_field(Ayy, Ayy_rhs, SYM, SYM, SYM);
    process_field(Ayz, Ayz_rhs, SYM, ANTI, ANTI);
    process_field(Azz, Azz_rhs, SYM, SYM, SYM);

    process_field(chi, chi_rhs, SYM, SYM, SYM);
    process_field(trK, trK_rhs, SYM, SYM, SYM);
    
    process_field(Gamx, Gamx_rhs, ANTI, SYM, SYM);
    process_field(Gamy, Gamy_rhs, SYM, ANTI, SYM);
    process_field(Gamz, Gamz_rhs, SYM, SYM, ANTI);
    
    process_field(Lap, Lap_rhs, SYM, SYM, SYM);
    
    process_field(betax, betax_rhs, ANTI, SYM, SYM);
    process_field(betay, betay_rhs, SYM, ANTI, SYM);
    process_field(betaz, betaz_rhs, SYM, SYM, ANTI);
    
    process_field(dtSfx, dtSfx_rhs, ANTI, SYM, SYM);
    process_field(dtSfy, dtSfy_rhs, SYM, ANTI, SYM);
    process_field(dtSfz, dtSfz_rhs, SYM, SYM, ANTI);
}

__launch_bounds__(256, 2)
__global__ void bssn_constraints_kernel(
    const int ex0, const int ex1, const int ex2,
    const int symmetry, const int lev,
    const double* __restrict__ X, const double* __restrict__ Y, const double* __restrict__ Z,
    const double* __restrict__ chi, const double* __restrict__ trK,
    const double* __restrict__ dxx, const double* __restrict__ gxy, const double* __restrict__ gxz,
    const double* __restrict__ dyy, const double* __restrict__ gyz, const double* __restrict__ dzz,
    const double* __restrict__ Axx, const double* __restrict__ Axy, const double* __restrict__ Axz,
    const double* __restrict__ Ayy, const double* __restrict__ Ayz, const double* __restrict__ Azz,
    const double* __restrict__ Gamx, const double* __restrict__ Gamy, const double* __restrict__ Gamz,
    const double* __restrict__ rho, 
    const double* __restrict__ Sx, const double* __restrict__ Sy, const double* __restrict__ Sz,
    const double* __restrict__ Gamxxx_in, const double* __restrict__ Gamxxy_in, const double* __restrict__ Gamxxz_in,
    const double* __restrict__ Gamxyy_in, const double* __restrict__ Gamxyz_in, const double* __restrict__ Gamxzz_in,
    const double* __restrict__ Gamyxx_in, const double* __restrict__ Gamyxy_in, const double* __restrict__ Gamyxz_in,
    const double* __restrict__ Gamyyy_in, const double* __restrict__ Gamyyz_in, const double* __restrict__ Gamyzz_in,
    const double* __restrict__ Gamzxx_in, const double* __restrict__ Gamzxy_in, const double* __restrict__ Gamzxz_in,
    const double* __restrict__ Gamzyy_in, const double* __restrict__ Gamzyz_in, const double* __restrict__ Gamzzz_in,
    const double* __restrict__ Rxx_in, const double* __restrict__ Rxy_in, const double* __restrict__ Rxz_in,
    const double* __restrict__ Ryy_in, const double* __restrict__ Ryz_in, const double* __restrict__ Rzz_in,
    double* __restrict__ ham_Res, 
    double* __restrict__ movx_Res, double* __restrict__ movy_Res, double* __restrict__ movz_Res,
    double* __restrict__ Gmx_Res, double* __restrict__ Gmy_Res, double* __restrict__ Gmz_Res
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int dims[3] = {ex0, ex1, ex2};
    double dX = X[1] - X[0]; double dY = Y[1] - Y[0]; double dZ = Z[1] - Z[0];

    int imax = ex0 - 1; int jmax = ex1 - 1; int kmax = ex2 - 1;
    int imin = 0, jmin = 0, kmin = 0;
    if (symmetry > 0 && fabs(Z[0]) < dZ) kmin = -2;
    if (symmetry > 1 && fabs(X[0]) < dX) imin = -2;
    if (symmetry > 1 && fabs(Y[0]) < dY) jmin = -2;

    int block_i = blockIdx.x * blockDim.x;
    int block_j = blockIdx.y * blockDim.y;
    int block_k = blockIdx.z * blockDim.z;

    __shared__ double smem[8][12][12];
    double dum1, dum2, dum3, dum4, dum5, dum6;

    // --- 外曲率 A (需一阶导数) ---
    double d_Axx_x, d_Axx_y, d_Axx_z;
    load_field_to_smem(smem, Axx, dims, block_i, block_j, block_k, SYM, SYM, SYM);
    __syncthreads(); compute_first_derivs_smem(smem, &d_Axx_x, &d_Axx_y, &d_Axx_z, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double d_Axy_x, d_Axy_y, d_Axy_z;
    load_field_to_smem(smem, Axy, dims, block_i, block_j, block_k, ANTI, ANTI, SYM);
    __syncthreads(); compute_first_derivs_smem(smem, &d_Axy_x, &d_Axy_y, &d_Axy_z, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double d_Axz_x, d_Axz_y, d_Axz_z;
    load_field_to_smem(smem, Axz, dims, block_i, block_j, block_k, ANTI, SYM, ANTI);
    __syncthreads(); compute_first_derivs_smem(smem, &d_Axz_x, &d_Axz_y, &d_Axz_z, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double d_Ayy_x, d_Ayy_y, d_Ayy_z;
    load_field_to_smem(smem, Ayy, dims, block_i, block_j, block_k, SYM, SYM, SYM);
    __syncthreads(); compute_first_derivs_smem(smem, &d_Ayy_x, &d_Ayy_y, &d_Ayy_z, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double d_Ayz_x, d_Ayz_y, d_Ayz_z;
    load_field_to_smem(smem, Ayz, dims, block_i, block_j, block_k, SYM, ANTI, ANTI);
    __syncthreads(); compute_first_derivs_smem(smem, &d_Ayz_x, &d_Ayz_y, &d_Ayz_z, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double d_Azz_x, d_Azz_y, d_Azz_z;
    load_field_to_smem(smem, Azz, dims, block_i, block_j, block_k, SYM, SYM, SYM);
    __syncthreads(); compute_first_derivs_smem(smem, &d_Azz_x, &d_Azz_y, &d_Azz_z, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    // --- 标量场特化 ---
    double chix, chiy, chiz, chixx, chixy, chixz, chiyy, chiyz, chizz;
    load_field_to_smem(smem, chi, dims, block_i, block_j, block_k, SYM, SYM, SYM);
    __syncthreads(); compute_all_derivs_smem(smem, &chix, &chiy, &chiz, &chixx, &chixy, &chixz, &chiyy, &chiyz, &chizz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double Kx, Ky, Kz;
    load_field_to_smem(smem, trK, dims, block_i, block_j, block_k, SYM, SYM, SYM);
    __syncthreads(); compute_first_derivs_smem(smem, &Kx, &Ky, &Kz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); 

    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    int idx = IDX3D(i, j, k, ex0, ex1, ex2);

    const double HALF = 0.5; const double TWO = 2.0; const double F2o3 = 2.0 / 3.0; 
    const double F3o2 = 1.5; const double EIGHT = 8.0; const double F16 = 16.0; const double PI = M_PI;

    double val_chi = chi[idx]; double chin1 = val_chi + 1.0;
    double val_trK = trK[idx];

    double l_gxx = dxx[idx] + 1.0; double l_gxy = gxy[idx]; double l_gxz = gxz[idx];
    double l_gyy = dyy[idx] + 1.0; double l_gyz = gyz[idx]; double l_gzz = dzz[idx] + 1.0;
    double l_Axx = Axx[idx]; double l_Axy = Axy[idx]; double l_Axz = Axz[idx];
    double l_Ayy = Ayy[idx]; double l_Ayz = Ayz[idx]; double l_Azz = Azz[idx];

    double gupzz = l_gxx * l_gyy * l_gzz + l_gxy * l_gyz * l_gxz + l_gxz * l_gxy * l_gyz - l_gxz * l_gyy * l_gxz - l_gxy * l_gxy * l_gzz - l_gxx * l_gyz * l_gyz;
    double gupxx = (l_gyy * l_gzz - l_gyz * l_gyz) / gupzz;
    double gupxy = -(l_gxy * l_gzz - l_gyz * l_gxz) / gupzz;
    double gupxz = (l_gxy * l_gyz - l_gyy * l_gxz) / gupzz;
    double gupyy = (l_gxx * l_gzz - l_gxz * l_gxz) / gupzz;
    double gupyz = -(l_gxx * l_gyz - l_gxy * l_gxz) / gupzz;
    gupzz = (l_gxx * l_gyy - l_gxy * l_gxy) / gupzz;

    double l_Gamxxx = Gamxxx_in[idx]; double l_Gamxxy = Gamxxy_in[idx]; double l_Gamxxz = Gamxxz_in[idx];
    double l_Gamxyy = Gamxyy_in[idx]; double l_Gamxyz = Gamxyz_in[idx]; double l_Gamxzz = Gamxzz_in[idx];
    double l_Gamyxx = Gamyxx_in[idx]; double l_Gamyxy = Gamyxy_in[idx]; double l_Gamyxz = Gamyxz_in[idx];
    double l_Gamyyy = Gamyyy_in[idx]; double l_Gamyyz = Gamyyz_in[idx]; double l_Gamyzz = Gamyzz_in[idx];
    double l_Gamzxx = Gamzxx_in[idx]; double l_Gamzxy = Gamzxy_in[idx]; double l_Gamzxz = Gamzxz_in[idx];
    double l_Gamzyy = Gamzyy_in[idx]; double l_Gamzyz = Gamzyz_in[idx]; double l_Gamzzz = Gamzzz_in[idx];

    double Gamxa = gupxx * l_Gamxxx + gupyy * l_Gamxyy + gupzz * l_Gamxzz + TWO * (gupxy * l_Gamxxy + gupxz * l_Gamxxz + gupyz * l_Gamxyz);
    double Gamya = gupxx * l_Gamyxx + gupyy * l_Gamyyy + gupzz * l_Gamyzz + TWO * (gupxy * l_Gamyxy + gupxz * l_Gamyxz + gupyz * l_Gamyyz);
    double Gamza = gupxx * l_Gamzxx + gupyy * l_Gamzyy + gupzz * l_Gamzzz + TWO * (gupxy * l_Gamzxy + gupxz * l_Gamzxz + gupyz * l_Gamzyz);

    Gmx_Res[idx] = Gamx[idx] - Gamxa; Gmy_Res[idx] = Gamy[idx] - Gamya; Gmz_Res[idx] = Gamz[idx] - Gamza;

    double l_Rxx = Rxx_in[idx]; double l_Rxy = Rxy_in[idx]; double l_Rxz = Rxz_in[idx];
    double l_Ryy = Ryy_in[idx]; double l_Ryz = Ryz_in[idx]; double l_Rzz = Rzz_in[idx];

    double fxx = chixx - l_Gamxxx * chix - l_Gamyxx * chiy - l_Gamzxx * chiz;
    double fxy = chixy - l_Gamxxy * chix - l_Gamyxy * chiy - l_Gamzxy * chiz;
    double fxz = chixz - l_Gamxxz * chix - l_Gamyxz * chiy - l_Gamzxz * chiz;
    double fyy = chiyy - l_Gamxyy * chix - l_Gamyyy * chiy - l_Gamzyy * chiz;
    double fyz = chiyz - l_Gamxyz * chix - l_Gamyyz * chiy - l_Gamzyz * chiz;
    double fzz = chizz - l_Gamxzz * chix - l_Gamyzz * chiy - l_Gamzzz * chiz;

    double f_scalar = gupxx * (fxx - F3o2/chin1 * chix * chix) + gupyy * (fyy - F3o2/chin1 * chiy * chiy) + gupzz * (fzz - F3o2/chin1 * chiz * chiz) + TWO * (gupxy * (fxy - F3o2/chin1 * chix * chiy) + gupxz * (fxz - F3o2/chin1 * chix * chiz) + gupyz * (fyz - F3o2/chin1 * chiy * chiz));
    l_Rxx += (fxx - chix*chix/chin1/TWO + l_gxx * f_scalar)/chin1/TWO;
    l_Ryy += (fyy - chiy*chiy/chin1/TWO + l_gyy * f_scalar)/chin1/TWO;
    l_Rzz += (fzz - chiz*chiz/chin1/TWO + l_gzz * f_scalar)/chin1/TWO;
    l_Rxy += (fxy - chix*chiy/chin1/TWO + l_gxy * f_scalar)/chin1/TWO;
    l_Rxz += (fxz - chix*chiz/chin1/TWO + l_gxz * f_scalar)/chin1/TWO;
    l_Ryz += (fyz - chiy*chiz/chin1/TWO + l_gyz * f_scalar)/chin1/TWO;
    
    double ham_val = gupxx * l_Rxx + gupyy * l_Ryy + gupzz * l_Rzz + TWO * (gupxy * l_Rxy + gupxz * l_Rxz + gupyz * l_Ryz);

    double term_xx = gupxx * l_Axx * l_Axx + gupyy * l_Axy * l_Axy + gupzz * l_Axz * l_Axz + TWO * (gupxy * l_Axx * l_Axy + gupxz * l_Axx * l_Axz + gupyz * l_Axy * l_Axz);
    double term_yy = gupxx * l_Axy * l_Axy + gupyy * l_Ayy * l_Ayy + gupzz * l_Ayz * l_Ayz + TWO * (gupxy * l_Axy * l_Ayy + gupxz * l_Axy * l_Ayz + gupyz * l_Ayy * l_Ayz);
    double term_zz = gupxx * l_Axz * l_Axz + gupyy * l_Ayz * l_Ayz + gupzz * l_Azz * l_Azz + TWO * (gupxy * l_Axz * l_Ayz + gupxz * l_Axz * l_Azz + gupyz * l_Ayz * l_Azz);
    double term_xy = gupxx * l_Axx * l_Axy + gupyy * l_Axy * l_Ayy + gupzz * l_Axz * l_Ayz + gupxy * (l_Axx * l_Ayy + l_Axy * l_Axy) + gupxz * (l_Axx * l_Ayz + l_Axz * l_Axy) + gupyz * (l_Axy * l_Ayz + l_Axz * l_Ayy);
    double term_xz = gupxx * l_Axx * l_Axz + gupyy * l_Axy * l_Ayz + gupzz * l_Axz * l_Azz + gupxy * (l_Axx * l_Ayz + l_Axy * l_Axz) + gupxz * (l_Axx * l_Azz + l_Axz * l_Axz) + gupyz * (l_Axy * l_Azz + l_Axz * l_Ayz);
    double term_yz = gupxx * l_Axy * l_Axz + gupyy * l_Ayy * l_Ayz + gupzz * l_Ayz * l_Azz + gupxy * (l_Axy * l_Ayz + l_Ayy * l_Axz) + gupxz * (l_Axy * l_Azz + l_Ayz * l_Axz) + gupyz * (l_Ayy * l_Azz + l_Ayz * l_Ayz);

    double trA2 = gupxx * term_xx + gupyy * term_yy + gupzz * term_zz + TWO * (gupxy * term_xy + gupxz * term_xz + gupyz * term_yz);

    ham_Res[idx] = chin1 * ham_val + F2o3 * val_trK * val_trK - trA2 - F16 * PI * rho[idx];

    double gx_phy = (gupxx * chix + gupxy * chiy + gupxz * chiz)/chin1;
    double gy_phy = (gupxy * chix + gupyy * chiy + gupyz * chiz)/chin1;
    double gz_phy = (gupxz * chix + gupyz * chiy + gupzz * chiz)/chin1;
    
    l_Gamxxx -= ((chix + chix)/chin1 - l_gxx * gx_phy)*HALF; l_Gamyxx -= (- l_gxx * gy_phy)*HALF; l_Gamzxx -= (- l_gxx * gz_phy)*HALF; 
    l_Gamxyy -= (- l_gyy * gx_phy)*HALF; l_Gamyyy -= ((chiy + chiy)/chin1 - l_gyy * gy_phy)*HALF; l_Gamzyy -= (- l_gyy * gz_phy)*HALF; 
    l_Gamxzz -= (- l_gzz * gx_phy)*HALF; l_Gamyzz -= (- l_gzz * gy_phy)*HALF; l_Gamzzz -= ((chiz + chiz)/chin1 - l_gzz * gz_phy)*HALF; 
    l_Gamxxy -= (chiy/chin1 - l_gxy * gx_phy)*HALF; l_Gamyxy -= (chix/chin1 - l_gxy * gy_phy)*HALF; l_Gamzxy -= (- l_gxy * gz_phy)*HALF; 
    l_Gamxxz -= (chiz/chin1 - l_gxz * gx_phy)*HALF; l_Gamyxz -= (- l_gxz * gy_phy)*HALF; l_Gamzxz -= (chix/chin1 - l_gxz * gz_phy)*HALF; 
    l_Gamxyz -= (- l_gyz * gx_phy)*HALF; l_Gamyyz -= (chiz/chin1 - l_gyz * gy_phy)*HALF; l_Gamzyz -= (chiy/chin1 - l_gyz * gz_phy)*HALF;

    double DA_xxx = d_Axx_x - (l_Gamxxx * l_Axx + l_Gamyxx * l_Axy + l_Gamzxx * l_Axz + l_Gamxxx * l_Axx + l_Gamyxx * l_Axy + l_Gamzxx * l_Axz) - chix * l_Axx / chin1;
    double DA_xyx = d_Axy_x - (l_Gamxxy * l_Axx + l_Gamyxy * l_Axy + l_Gamzxy * l_Axz + l_Gamxxx * l_Axy + l_Gamyxx * l_Ayy + l_Gamzxx * l_Ayz) - chix * l_Axy / chin1;
    double DA_xzx = d_Axz_x - (l_Gamxxz * l_Axx + l_Gamyxz * l_Axy + l_Gamzxz * l_Axz + l_Gamxxx * l_Axz + l_Gamyxx * l_Ayz + l_Gamzxx * l_Azz) - chix * l_Axz / chin1;
    double DA_yyx = d_Ayy_x - (l_Gamxxy * l_Axy + l_Gamyxy * l_Ayy + l_Gamzxy * l_Ayz + l_Gamxxy * l_Axy + l_Gamyxy * l_Ayy + l_Gamzxy * l_Ayz) - chix * l_Ayy / chin1;
    double DA_yzx = d_Ayz_x - (l_Gamxxz * l_Axy + l_Gamyxz * l_Ayy + l_Gamzxz * l_Ayz + l_Gamxxy * l_Axz + l_Gamyxy * l_Ayz + l_Gamzxy * l_Azz) - chix * l_Ayz / chin1;
    double DA_zzx = d_Azz_x - (l_Gamxxz * l_Axz + l_Gamyxz * l_Ayz + l_Gamzxz * l_Azz + l_Gamxxz * l_Axz + l_Gamyxz * l_Ayz + l_Gamzxz * l_Azz) - chix * l_Azz / chin1;
    
    double DA_xxy = d_Axx_y - (l_Gamxxy * l_Axx + l_Gamyxy * l_Axy + l_Gamzxy * l_Axz + l_Gamxxy * l_Axx + l_Gamyxy * l_Axy + l_Gamzxy * l_Axz) - chiy * l_Axx / chin1;
    double DA_xyy = d_Axy_y - (l_Gamxyy * l_Axx + l_Gamyyy * l_Axy + l_Gamzyy * l_Axz + l_Gamxxy * l_Axy + l_Gamyxy * l_Ayy + l_Gamzxy * l_Ayz) - chiy * l_Axy / chin1;
    double DA_xzy = d_Axz_y - (l_Gamxyz * l_Axx + l_Gamyyz * l_Axy + l_Gamzyz * l_Axz + l_Gamxxy * l_Axz + l_Gamyxy * l_Ayz + l_Gamzxy * l_Azz) - chiy * l_Axz / chin1;
    double DA_yyy = d_Ayy_y - (l_Gamxyy * l_Axy + l_Gamyyy * l_Ayy + l_Gamzyy * l_Ayz + l_Gamxyy * l_Axy + l_Gamyyy * l_Ayy + l_Gamzyy * l_Ayz) - chiy * l_Ayy / chin1; 
    double DA_yzy = d_Ayz_y - (l_Gamxyz * l_Axy + l_Gamyyz * l_Ayy + l_Gamzyz * l_Ayz + l_Gamxyy * l_Axz + l_Gamyyy * l_Ayz + l_Gamzyy * l_Azz) - chiy * l_Ayz / chin1;
    double DA_zzy = d_Azz_y - (l_Gamxyz * l_Axz + l_Gamyyz * l_Ayz + l_Gamzyz * l_Azz + l_Gamxyz * l_Axz + l_Gamyyz * l_Ayz + l_Gamzyz * l_Azz) - chiy * l_Azz / chin1;
    
    double DA_xxz = d_Axx_z - (l_Gamxxz * l_Axx + l_Gamyxz * l_Axy + l_Gamzxz * l_Axz + l_Gamxxz * l_Axx + l_Gamyxz * l_Axy + l_Gamzxz * l_Axz) - chiz * l_Axx / chin1;
    double DA_xyz = d_Axy_z - (l_Gamxyz * l_Axx + l_Gamyyz * l_Axy + l_Gamzyz * l_Axz + l_Gamxxz * l_Axy + l_Gamyxz * l_Ayy + l_Gamzxz * l_Ayz) - chiz * l_Axy / chin1;
    double DA_xzz = d_Axz_z - (l_Gamxzz * l_Axx + l_Gamyzz * l_Axy + l_Gamzzz * l_Axz + l_Gamxxz * l_Axz + l_Gamyxz * l_Ayz + l_Gamzxz * l_Azz) - chiz * l_Axz / chin1;
    double DA_yyz = d_Ayy_z - (l_Gamxyz * l_Axy + l_Gamyyz * l_Ayy + l_Gamzyz * l_Ayz + l_Gamxyz * l_Axy + l_Gamyyz * l_Ayy + l_Gamzyz * l_Ayz) - chiz * l_Ayy / chin1;
    double DA_yzz = d_Ayz_z - (l_Gamxzz * l_Axy + l_Gamyzz * l_Ayy + l_Gamzzz * l_Ayz + l_Gamxyz * l_Axz + l_Gamyyz * l_Ayz + l_Gamzyz * l_Azz) - chiz * l_Ayz / chin1;
    double DA_zzz = d_Azz_z - (l_Gamxzz * l_Axz + l_Gamyzz * l_Ayz + l_Gamzzz * l_Azz + l_Gamxzz * l_Axz + l_Gamyzz * l_Ayz + l_Gamzzz * l_Azz) - chiz * l_Azz / chin1;

    movx_Res[idx] = gupxx * DA_xxx + gupyy * DA_xyy + gupzz * DA_xzz + gupxy * DA_xyx + gupxz * DA_xzx + gupyz * DA_xzy + gupxy * DA_xxy + gupxz * DA_xxz + gupyz * DA_xyz;
    movy_Res[idx] = gupxx * DA_xyx + gupyy * DA_yyy + gupzz * DA_yzz + gupxy * DA_yyx + gupxz * DA_yzx + gupyz * DA_yzy + gupxy * DA_xyy + gupxz * DA_xyz + gupyz * DA_yyz;
    movz_Res[idx] = gupxx * DA_xzx + gupyy * DA_yzy + gupzz * DA_zzz + gupxy * DA_yzx + gupxz * DA_zzx + gupyz * DA_zzy + gupxy * DA_xzy + gupxz * DA_xzz + gupyz * DA_yzz;

    movx_Res[idx] -= F2o3 * Kx + EIGHT * PI * Sx[idx];
    movy_Res[idx] -= F2o3 * Ky + EIGHT * PI * Sy[idx];
    movz_Res[idx] -= F2o3 * Kz + EIGHT * PI * Sz[idx];
}

void gpu_compute_rhs_bssn_launch(
    cudaStream_t &stream, int* ex, double T, double* X, double* Y, double* Z,
    double* chi, double* trK,
    double* dxx, double* gxy, double* gxz, double* dyy, double* gyz, double* dzz,
    double* Axx, double* Axy, double* Axz, double* Ayy, double* Ayz, double* Azz,
    double* Gamx, double* Gamy, double* Gamz, double* Lap,
    double* betax, double* betay, double* betaz,
    double* dtSfx, double* dtSfy, double* dtSfz,
    double* chi_rhs, double* trK_rhs,
    double* gxx_rhs, double* gxy_rhs, double* gxz_rhs, double* gyy_rhs, double* gyz_rhs, double* gzz_rhs,
    double* Axx_rhs, double* Axy_rhs, double* Axz_rhs, double* Ayy_rhs, double* Ayz_rhs, double* Azz_rhs,
    double* Gamx_rhs, double* Gamy_rhs, double* Gamz_rhs, double* Lap_rhs,
    double* betax_rhs, double* betay_rhs, double* betaz_rhs,
    double* dtSfx_rhs, double* dtSfy_rhs, double* dtSfz_rhs,
    double* rho, double* Sx, double* Sy, double* Sz,
    double* Sxx, double* Sxy, double* Sxz, double* Syy, double* Syz, double* Szz,
    double* Gamxxx, double* Gamxxy, double* Gamxxz, double* Gamxyy, double* Gamxyz, double* Gamxzz,
    double* Gamyxx, double* Gamyxy, double* Gamyxz, double* Gamyyy, double* Gamyyz, double* Gamyzz,
    double* Gamzxx, double* Gamzxy, double* Gamzxz, double* Gamzyy, double* Gamzyz, double* Gamzzz,
    double* Rxx, double* Rxy, double* Rxz, double* Ryy, double* Ryz, double* Rzz,
    double* ham_Res, double* movx_Res, double* movy_Res, double* movz_Res,
    double* Gmx_Res, double* Gmy_Res, double* Gmz_Res,
    int symmetry, int lev, double eps, int co
) {
    dim3 block(8, 8, 4);
    dim3 grid(
        (ex[0] + block.x - 1) / block.x,
        (ex[1] + block.y - 1) / block.y,
        (ex[2] + block.z - 1) / block.z
    );

    // 1. 生成共形几何：输出完全纯净的 Conformal R 和 Gamma
    bssn_ricci_kernel<<<grid, block, 0, stream>>>(
        ex[0], ex[1], ex[2], symmetry, lev, X, Y, Z,
        dxx, gxy, gxz, dyy, gyz, dzz, Gamx, Gamy, Gamz,
        Gamxxx, Gamxxy, Gamxxz, Gamxyy, Gamxyz, Gamxzz,
        Gamyxx, Gamyxy, Gamyxz, Gamyyy, Gamyyz, Gamyzz,
        Gamzxx, Gamzxy, Gamzxz, Gamzyy, Gamzyz, Gamzzz,
        Rxx, Rxy, Rxz, Ryy, Ryz, Rzz
    );

    // 2. 物理演化引擎：包含彻底剥离了错误复用的真实 Aup 混合张量计算
    bssn_rhs_eval_kernel<<<grid, block, 0, stream>>>(
        ex[0], ex[1], ex[2], symmetry, lev, X, Y, Z,
        chi, trK, dxx, gxy, gxz, dyy, gyz, dzz,
        Axx, Axy, Axz, Ayy, Ayz, Azz,
        Gamx, Gamy, Gamz, Lap, betax, betay, betaz, dtSfx, dtSfy, dtSfz,
        rho, Sx, Sy, Sz, Sxx, Sxy, Sxz, Syy, Syz, Szz,
        Gamxxx, Gamxxy, Gamxxz, Gamxyy, Gamxyz, Gamxzz,
        Gamyxx, Gamyxy, Gamyxz, Gamyyy, Gamyyz, Gamyzz,
        Gamzxx, Gamzxy, Gamzxz, Gamzyy, Gamzyz, Gamzzz,
        Rxx, Rxy, Rxz, Ryy, Ryz, Rzz,
        chi_rhs, trK_rhs, gxx_rhs, gxy_rhs, gxz_rhs, gyy_rhs, gyz_rhs, gzz_rhs,
        Axx_rhs, Axy_rhs, Axz_rhs, Ayy_rhs, Ayz_rhs, Azz_rhs,
        Gamx_rhs, Gamy_rhs, Gamz_rhs, Lap_rhs, betax_rhs, betay_rhs, betaz_rhs,
        dtSfx_rhs, dtSfy_rhs, dtSfz_rhs
    );

    // 3. 耗散与平流：已附加只读 __ldg 优化
    bssn_advection_dissipation_kernel<<<grid, block, 0, stream>>>(
        ex[0], ex[1], ex[2], symmetry, lev, eps, X, Y, Z,
        betax, betay, betaz, dxx, gxy, gxz, dyy, gyz, dzz,
        Axx, Axy, Axz, Ayy, Ayz, Azz, chi, trK, Gamx, Gamy, Gamz, Lap, dtSfx, dtSfy, dtSfz,
        gxx_rhs, gxy_rhs, gxz_rhs, gyy_rhs, gyz_rhs, gzz_rhs,
        Axx_rhs, Axy_rhs, Axz_rhs, Ayy_rhs, Ayz_rhs, Azz_rhs,
        chi_rhs, trK_rhs, Gamx_rhs, Gamy_rhs, Gamz_rhs, Lap_rhs,
        betax_rhs, betay_rhs, betaz_rhs, dtSfx_rhs, dtSfy_rhs, dtSfz_rhs
    );

    // 4. 诊断系统：全 Shared Memory 纯净化计算
    if (co == 0) {
        bssn_constraints_kernel<<<grid, block, 0, stream>>>(
            ex[0], ex[1], ex[2], symmetry, lev, X, Y, Z,
            chi, trK, dxx, gxy, gxz, dyy, gyz, dzz,
            Axx, Axy, Axz, Ayy, Ayz, Azz, Gamx, Gamy, Gamz, rho, Sx, Sy, Sz,
            Gamxxx, Gamxxy, Gamxxz, Gamxyy, Gamxyz, Gamxzz,
            Gamyxx, Gamyxy, Gamyxz, Gamyyy, Gamyyz, Gamyzz,
            Gamzxx, Gamzxy, Gamzxz, Gamzyy, Gamzyz, Gamzzz,
            Rxx, Rxy, Rxz, Ryy, Ryz, Rzz,
            ham_Res, movx_Res, movy_Res, movz_Res,
            Gmx_Res, Gmy_Res, Gmz_Res
        );
    }
}