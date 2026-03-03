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

    // ================== BATCH DERIVATIVES: 重命名导数变量，避免与克氏符混淆 ==================
    double dgxx_x, dgxx_y, dgxx_z, dxx_xx, dxx_xy, dxx_xz, dxx_yy, dxx_yz, dxx_zz;
    load_field_to_smem(smem, dxx, dims, block_i, block_j, block_k, SYM, SYM, SYM);
    __syncthreads(); compute_all_derivs_smem(smem, &dgxx_x, &dgxx_y, &dgxx_z, &dxx_xx, &dxx_xy, &dxx_xz, &dxx_yy, &dxx_yz, &dxx_zz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double dgxy_x, dgxy_y, dgxy_z, gxy_xx, gxy_xy, gxy_xz, gxy_yy, gxy_yz, gxy_zz;
    load_field_to_smem(smem, gxy, dims, block_i, block_j, block_k, ANTI, ANTI, SYM);
    __syncthreads(); compute_all_derivs_smem(smem, &dgxy_x, &dgxy_y, &dgxy_z, &gxy_xx, &gxy_xy, &gxy_xz, &gxy_yy, &gxy_yz, &gxy_zz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double dgxz_x, dgxz_y, dgxz_z, gxz_xx, gxz_xy, gxz_xz, gxz_yy, gxz_yz, gxz_zz;
    load_field_to_smem(smem, gxz, dims, block_i, block_j, block_k, ANTI, SYM, ANTI);
    __syncthreads(); compute_all_derivs_smem(smem, &dgxz_x, &dgxz_y, &dgxz_z, &gxz_xx, &gxz_xy, &gxz_xz, &gxz_yy, &gxz_yz, &gxz_zz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double dgyy_x, dgyy_y, dgyy_z, dyy_xx, dyy_xy, dyy_xz, dyy_yy, dyy_yz, dyy_zz;
    load_field_to_smem(smem, dyy, dims, block_i, block_j, block_k, SYM, SYM, SYM);
    __syncthreads(); compute_all_derivs_smem(smem, &dgyy_x, &dgyy_y, &dgyy_z, &dyy_xx, &dyy_xy, &dyy_xz, &dyy_yy, &dyy_yz, &dyy_zz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double dgyz_x, dgyz_y, dgyz_z, gyz_xx, gyz_xy, gyz_xz, gyz_yy, gyz_yz, gyz_zz;
    load_field_to_smem(smem, gyz, dims, block_i, block_j, block_k, SYM, ANTI, ANTI);
    __syncthreads(); compute_all_derivs_smem(smem, &dgyz_x, &dgyz_y, &dgyz_z, &gyz_xx, &gyz_xy, &gyz_xz, &gyz_yy, &gyz_yz, &gyz_zz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double dgzz_x, dgzz_y, dgzz_z, dzz_xx, dzz_xy, dzz_xz, dzz_yy, dzz_yz, dzz_zz;
    load_field_to_smem(smem, dzz, dims, block_i, block_j, block_k, SYM, SYM, SYM);
    __syncthreads(); compute_all_derivs_smem(smem, &dgzz_x, &dgzz_y, &dgzz_z, &dzz_xx, &dzz_xy, &dzz_xz, &dzz_yy, &dzz_yz, &dzz_zz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double dGamxx, dGamxy, dGamxz;
    load_field_to_smem(smem, Gamx, dims, block_i, block_j, block_k, ANTI, SYM, SYM);
    __syncthreads(); compute_first_derivs_smem(smem, &dGamxx, &dGamxy, &dGamxz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double dGamyx, dGamyy, dGamyz;
    load_field_to_smem(smem, Gamy, dims, block_i, block_j, block_k, SYM, ANTI, SYM);
    __syncthreads(); compute_first_derivs_smem(smem, &dGamyx, &dGamyy, &dGamyz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double dGamzx, dGamzy, dGamzz;
    load_field_to_smem(smem, Gamz, dims, block_i, block_j, block_k, SYM, SYM, ANTI);
    __syncthreads(); compute_first_derivs_smem(smem, &dGamzx, &dGamzy, &dGamzz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax);

    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    int idx = IDX3D(i, j, k, ex0, ex1, ex2);

    const double HALF = 0.5; const double TWO = 2.0; const double ONE = 1.0;

    double l_gxx = dxx[idx] + ONE; double l_gxy = gxy[idx]; double l_gxz = gxz[idx];
    double l_gyy = dyy[idx] + ONE; double l_gyz = gyz[idx]; double l_gzz = dzz[idx] + ONE;

    // 【优化 1：提炼逆度规的单次除法】
    double det = l_gxx * l_gyy * l_gzz + TWO * l_gxy * l_gyz * l_gxz - l_gxz * l_gyy * l_gxz - l_gxx * l_gyz * l_gyz - l_gzz * l_gxy * l_gxy;
    double inv_det = 1.0 / det;

    double gupxx = (l_gyy * l_gzz - l_gyz * l_gyz) * inv_det;
    double gupxy = (l_gyz * l_gxz - l_gxy * l_gzz) * inv_det;
    double gupxz = (l_gxy * l_gyz - l_gyy * l_gxz) * inv_det;
    double gupyy = (l_gxx * l_gzz - l_gxz * l_gxz) * inv_det;
    double gupyz = (l_gxy * l_gxz - l_gxx * l_gyz) * inv_det;
    double gupzz = (l_gxx * l_gyy - l_gxy * l_gxy) * inv_det;

    // 【优化 2：剥离第一类克氏符，消灭原代码中强行降指标的 160+ 次无谓乘法】
    double G1_xxx = HALF * dgxx_x;
    double G1_xxy = HALF * dgxx_y;
    double G1_xxz = HALF * dgxx_z;
    double G1_xyy = HALF * (TWO * dgxy_y - dgyy_x);
    double G1_xyz = HALF * (dgxy_z + dgxz_y - dgyz_x);
    double G1_xzz = HALF * (TWO * dgxz_z - dgzz_x);

    double G1_yxx = HALF * (TWO * dgxy_x - dgxx_y);
    double G1_yxy = HALF * dgyy_x;
    double G1_yxz = HALF * (dgxy_z + dgyz_x - dgxz_y);
    double G1_yyy = HALF * dgyy_y;
    double G1_yyz = HALF * dgyy_z;
    double G1_yzz = HALF * (TWO * dgyz_z - dgzz_y);

    double G1_zxx = HALF * (TWO * dgxz_x - dgxx_z);
    double G1_zxy = HALF * (dgxz_y + dgyz_x - dgxy_z);
    double G1_zxz = HALF * dgzz_x;
    double G1_zyy = HALF * (TWO * dgyz_y - dgyy_z);
    double G1_zyz = HALF * dgzz_y;
    double G1_zzz = HALF * dgzz_z;

    // 直接通过逆度规乘法计算第二类克氏符
    double G2_xxx = gupxx * G1_xxx + gupxy * G1_yxx + gupxz * G1_zxx;
    double G2_xxy = gupxx * G1_xxy + gupxy * G1_yxy + gupxz * G1_zxy;
    double G2_xxz = gupxx * G1_xxz + gupxy * G1_yxz + gupxz * G1_zxz;
    double G2_xyy = gupxx * G1_xyy + gupxy * G1_yyy + gupxz * G1_zyy;
    double G2_xyz = gupxx * G1_xyz + gupxy * G1_yyz + gupxz * G1_zyz;
    double G2_xzz = gupxx * G1_xzz + gupxy * G1_yzz + gupxz * G1_zzz;

    double G2_yxx = gupxy * G1_xxx + gupyy * G1_yxx + gupyz * G1_zxx;
    double G2_yxy = gupxy * G1_xxy + gupyy * G1_yxy + gupyz * G1_zxy;
    double G2_yxz = gupxy * G1_xxz + gupyy * G1_yxz + gupyz * G1_zxz;
    double G2_yyy = gupxy * G1_xyy + gupyy * G1_yyy + gupyz * G1_zyy;
    double G2_yyz = gupxy * G1_xyz + gupyy * G1_yyz + gupyz * G1_zyz;
    double G2_yzz = gupxy * G1_xzz + gupyy * G1_yzz + gupyz * G1_zzz;

    double G2_zxx = gupxz * G1_xxx + gupyz * G1_yxx + gupzz * G1_zxx;
    double G2_zxy = gupxz * G1_xxy + gupyz * G1_yxy + gupzz * G1_zxy;
    double G2_zxz = gupxz * G1_xxz + gupyz * G1_yxz + gupzz * G1_zxz;
    double G2_zyy = gupxz * G1_xyy + gupyz * G1_yyy + gupzz * G1_zyy;
    double G2_zyz = gupxz * G1_xyz + gupyz * G1_yyz + gupzz * G1_zyz;
    double G2_zzz = gupxz * G1_xzz + gupyz * G1_yzz + gupzz * G1_zzz;

    double Gamxa = gupxx * G2_xxx + gupyy * G2_xyy + gupzz * G2_xzz + TWO * (gupxy * G2_xxy + gupxz * G2_xxz + gupyz * G2_xyz);
    double Gamya = gupxx * G2_yxx + gupyy * G2_yyy + gupzz * G2_yzz + TWO * (gupxy * G2_yxy + gupxz * G2_yxz + gupyz * G2_yyz);
    double Gamza = gupxx * G2_zxx + gupyy * G2_zyy + gupzz * G2_zzz + TWO * (gupxy * G2_zxy + gupxz * G2_zxz + gupyz * G2_zyz);

    double l_Rxx = gupxx * dxx_xx + gupyy * dxx_yy + gupzz * dxx_zz + (gupxy * dxx_xy + gupxz * dxx_xz + gupyz * dxx_yz) * TWO;
    double l_Ryy = gupxx * dyy_xx + gupyy * dyy_yy + gupzz * dyy_zz + (gupxy * dyy_xy + gupxz * dyy_xz + gupyz * dyy_yz) * TWO;
    double l_Rzz = gupxx * dzz_xx + gupyy * dzz_yy + gupzz * dzz_zz + (gupxy * dzz_xy + gupxz * dzz_xz + gupyz * dzz_yz) * TWO;
    double l_Rxy = gupxx * gxy_xx + gupyy * gxy_yy + gupzz * gxy_zz + (gupxy * gxy_xy + gupxz * gxy_xz + gupyz * gxy_yz) * TWO;
    double l_Rxz = gupxx * gxz_xx + gupyy * gxz_yy + gupzz * gxz_zz + (gupxy * gxz_xy + gupxz * gxz_xz + gupyz * gxz_yz) * TWO;
    double l_Ryz = gupxx * gyz_xx + gupyy * gyz_yy + gupzz * gyz_zz + (gupxy * gyz_xy + gupxz * gyz_xz + gupyz * gyz_yz) * TWO;

    // 【优化 3：替换掉原代码非线性项中为了复用而杂糅的变量，保证了纯粹数学意义上的连续 MADD】
    l_Rxx = -HALF * l_Rxx + l_gxx * dGamxx + l_gxy * dGamyx + l_gxz * dGamzx + Gamxa * G1_xxx + Gamya * G1_xxy + Gamza * G1_xxz + 
          gupxx * (TWO*(G2_xxx*G1_xxx + G2_yxx*G1_xxy + G2_zxx*G1_xxz) + G2_xxx*G1_xxx + G2_yxx*G1_yxx + G2_zxx*G1_zxx) +
          gupxy * (TWO*(G2_xxx*G1_xxy + G2_yxx*G1_xyy + G2_zxx*G1_xyz + G2_xxy*G1_xxx + G2_yxy*G1_xxy + G2_zxy*G1_xxz) + G2_xxy*G1_xxx + G2_yxy*G1_yxx + G2_zxy*G1_zxx + G2_xxx*G1_xxy + G2_yxx*G1_yxy + G2_zxx*G1_zxy) + 
          gupxz * (TWO*(G2_xxx*G1_xxz + G2_yxx*G1_xyz + G2_zxx*G1_xzz + G2_xxz*G1_xxx + G2_yxz*G1_xxy + G2_zxz*G1_xxz) + G2_xxz*G1_xxx + G2_yxz*G1_yxx + G2_zxz*G1_zxx + G2_xxx*G1_xxz + G2_yxx*G1_yxz + G2_zxx*G1_zxz) + 
          gupyy * (TWO*(G2_xxy*G1_xxy + G2_yxy*G1_xyy + G2_zxy*G1_xyz) + G2_xxy*G1_xxy + G2_yxy*G1_yxy + G2_zxy*G1_zxy) + 
          gupyz * (TWO*(G2_xxy*G1_xxz + G2_yxy*G1_xyz + G2_zxy*G1_xzz + G2_xxz*G1_xxy + G2_yxz*G1_xyy + G2_zxz*G1_xyz) + G2_xxz*G1_xxy + G2_yxz*G1_yxy + G2_zxz*G1_zxy + G2_xxy*G1_xxz + G2_yxy*G1_yxz + G2_zxy*G1_zxz) + 
          gupzz * (TWO*(G2_xxz*G1_xxz + G2_yxz*G1_xyz + G2_zxz*G1_xzz) + G2_xxz*G1_xxz + G2_yxz*G1_yxz + G2_zxz*G1_zxz);

    l_Ryy = -HALF * l_Ryy + l_gxy * dGamxy + l_gyy * dGamyy + l_gyz * dGamzy + Gamxa * G1_yxy + Gamya * G1_yyy + Gamza * G1_yyz + 
          gupxx * (TWO*(G2_xxy*G1_yxx + G2_yxy*G1_yxy + G2_zxy*G1_yxz) + G2_xxy*G1_xxy + G2_yxy*G1_yxy + G2_zxy*G1_zxy) + 
          gupxy * (TWO*(G2_xxy*G1_yxy + G2_yxy*G1_yyy + G2_zxy*G1_yyz + G2_xyy*G1_yxx + G2_yyy*G1_yxy + G2_zyy*G1_yxz) + G2_xyy*G1_xxy + G2_yyy*G1_yxy + G2_zyy*G1_zxy + G2_xxy*G1_xyy + G2_yxy*G1_yyy + G2_zxy*G1_zyy) + 
          gupxz * (TWO*(G2_xxy*G1_yxz + G2_yxy*G1_yyz + G2_zxy*G1_yzz + G2_xyz*G1_yxx + G2_yyz*G1_yxy + G2_zyz*G1_yxz) + G2_xyz*G1_xxy + G2_yyz*G1_yxy + G2_zyz*G1_zxy + G2_xxy*G1_xyz + G2_yxy*G1_yyz + G2_zxy*G1_zyz) + 
          gupyy * (TWO*(G2_xyy*G1_yxy + G2_yyy*G1_yyy + G2_zyy*G1_yyz) + G2_xyy*G1_xyy + G2_yyy*G1_yyy + G2_zyy*G1_zyy) + 
          gupyz * (TWO*(G2_xyy*G1_yxz + G2_yyy*G1_yyz + G2_zyy*G1_yzz + G2_xyz*G1_yxy + G2_yyz*G1_yyy + G2_zyz*G1_yyz) + G2_xyz*G1_xyy + G2_yyz*G1_yyy + G2_zyz*G1_zyy + G2_xyy*G1_xyz + G2_yyy*G1_yyz + G2_zyy*G1_zyz) + 
          gupzz * (TWO*(G2_xyz*G1_yxz + G2_yyz*G1_yyz + G2_zyz*G1_yzz) + G2_xyz*G1_xyz + G2_yyz*G1_yyz + G2_zyz*G1_zyz);

    l_Rzz = -HALF * l_Rzz + l_gxz * dGamxz + l_gyz * dGamyz + l_gzz * dGamzz + Gamxa * G1_zxz + Gamya * G1_zyz + Gamza * G1_zzz + 
          gupxx * (TWO*(G2_xxz*G1_zxx + G2_yxz*G1_zxy + G2_zxz*G1_zxz) + G2_xxz*G1_xxz + G2_yxz*G1_yxz + G2_zxz*G1_zxz) + 
          gupxy * (TWO*(G2_xxz*G1_zxy + G2_yxz*G1_zyy + G2_zxz*G1_zyz + G2_xyz*G1_zxx + G2_yyz*G1_zxy + G2_zyz*G1_zxz) + G2_xyz*G1_xxz + G2_yyz*G1_yxz + G2_zyz*G1_zxz + G2_xxz*G1_xyz + G2_yxz*G1_yyz + G2_zxz*G1_zyz) + 
          gupxz * (TWO*(G2_xxz*G1_zxz + G2_yxz*G1_zyz + G2_zxz*G1_zzz + G2_xzz*G1_zxx + G2_yzz*G1_zxy + G2_zzz*G1_zxz) + G2_xzz*G1_xxz + G2_yzz*G1_yxz + G2_zzz*G1_zxz + G2_xxz*G1_xzz + G2_yxz*G1_yzz + G2_zxz*G1_zzz) + 
          gupyy * (TWO*(G2_xyz*G1_zxy + G2_yyz*G1_zyy + G2_zyz*G1_zyz) + G2_xyz*G1_xyz + G2_yyz*G1_yyz + G2_zyz*G1_zyz) + 
          gupyz * (TWO*(G2_xyz*G1_zxz + G2_yyz*G1_zyz + G2_zyz*G1_zzz + G2_xzz*G1_zxy + G2_yzz*G1_zyy + G2_zzz*G1_zyz) + G2_xzz*G1_xyz + G2_yzz*G1_yyz + G2_zzz*G1_zyz + G2_xyz*G1_xzz + G2_yyz*G1_yzz + G2_zyz*G1_zzz) + 
          gupzz * (TWO*(G2_xzz*G1_zxz + G2_yzz*G1_zyz + G2_zzz*G1_zzz) + G2_xzz*G1_xzz + G2_yzz*G1_yzz + G2_zzz*G1_zzz);

    l_Rxy = HALF * ( - l_Rxy + l_gxx * dGamxy + l_gxy * dGamyy + l_gxz * dGamzy + l_gxy * dGamxx + l_gyy * dGamyx + l_gyz * dGamzx + Gamxa * G1_xxy + Gamya * G1_xyy + Gamza * G1_xyz + Gamxa * G1_yxx + Gamya * G1_yxy + Gamza * G1_yxz) + 
          gupxx * (G2_xxx*G1_yxx + G2_yxx*G1_yxy + G2_zxx*G1_yxz + G2_xxy*G1_xxx + G2_yxy*G1_xxy + G2_zxy*G1_xxz + G2_xxx*G1_xxy + G2_yxx*G1_yxy + G2_zxx*G1_zxy) + 
          gupxy * (G2_xxx*G1_yxy + G2_yxx*G1_yyy + G2_zxx*G1_yyz + G2_xxy*G1_xxy + G2_yxy*G1_xyy + G2_zxy*G1_xyz + G2_xxy*G1_xxy + G2_yxy*G1_yxy + G2_zxy*G1_zxy + G2_xxy*G1_yxx + G2_yxy*G1_yxy + G2_zxy*G1_yxz + G2_xyy*G1_xxx + G2_yyy*G1_xxy + G2_zyy*G1_xxz + G2_xxx*G1_xyy + G2_yxx*G1_yyy + G2_zxx*G1_zyy) + 
          gupxz * (G2_xxx*G1_yxz + G2_yxx*G1_yyz + G2_zxx*G1_yzz + G2_xxy*G1_xxz + G2_yxy*G1_xyz + G2_zxy*G1_xzz + G2_xxz*G1_xxy + G2_yxz*G1_yxy + G2_zxz*G1_zxy + G2_xxz*G1_yxx + G2_yxz*G1_yxy + G2_zxz*G1_yxz + G2_xyz*G1_xxx + G2_yyz*G1_xxy + G2_zyz*G1_xxz + G2_xxx*G1_xyz + G2_yxx*G1_yyz + G2_zxx*G1_zyz) + 
          gupyy * (G2_xxy*G1_yxy + G2_yxy*G1_yyy + G2_zxy*G1_yyz + G2_xyy*G1_xxy + G2_yyy*G1_xyy + G2_zyy*G1_xyz + G2_xxy*G1_xyy + G2_yxy*G1_yyy + G2_zxy*G1_zyy) + 
          gupyz * (G2_xxy*G1_yxz + G2_yxy*G1_yyz + G2_zxy*G1_yzz + G2_xyy*G1_xxz + G2_yyy*G1_xyz + G2_zyy*G1_xzz + G2_xxz*G1_xyy + G2_yxz*G1_yyy + G2_zxz*G1_zyy + G2_xxz*G1_yxy + G2_yxz*G1_yyy + G2_zxz*G1_yyz + G2_xyz*G1_xxy + G2_yyz*G1_xyy + G2_zyz*G1_xyz + G2_xxy*G1_xyz + G2_yxy*G1_yyz + G2_zxy*G1_zyz) + 
          gupzz * (G2_xxz*G1_yxz + G2_yxz*G1_yyz + G2_zxz*G1_yzz + G2_xyz*G1_xxz + G2_yyz*G1_xyz + G2_zyz*G1_xzz + G2_xxz*G1_xyz + G2_yxz*G1_yyz + G2_zxz*G1_zyz);

    l_Rxz = HALF * ( - l_Rxz + l_gxx * dGamxz + l_gxy * dGamyz + l_gxz * dGamzz + l_gxz * dGamxx + l_gyz * dGamyx + l_gzz * dGamzx + Gamxa * G1_xxz + Gamya * G1_xyz + Gamza * G1_xzz + Gamxa * G1_zxx + Gamya * G1_zxy + Gamza * G1_zxz) + 
          gupxx * (G2_xxx*G1_zxx + G2_yxx*G1_zxy + G2_zxx*G1_zxz + G2_xxz*G1_xxx + G2_yxz*G1_xxy + G2_zxz*G1_xxz + G2_xxx*G1_xxz + G2_yxx*G1_yxz + G2_zxx*G1_zxz) + 
          gupxy * (G2_xxx*G1_zxy + G2_yxx*G1_zyy + G2_zxx*G1_zyz + G2_xxz*G1_xxy + G2_yxz*G1_xyy + G2_zxz*G1_xyz + G2_xxy*G1_xxz + G2_yxy*G1_yxz + G2_zxy*G1_zxz + G2_xxy*G1_zxx + G2_yxy*G1_zxy + G2_zxy*G1_zxz + G2_xyz*G1_xxx + G2_yyz*G1_xxy + G2_zyz*G1_xxz + G2_xxx*G1_xyz + G2_yxx*G1_yyz + G2_zxx*G1_zyz) + 
          gupxz * (G2_xxx*G1_zxz + G2_yxx*G1_zyz + G2_zxx*G1_zzz + G2_xxz*G1_xxz + G2_yxz*G1_xyz + G2_zxz*G1_xzz + G2_xxz*G1_xxz + G2_yxz*G1_yxz + G2_zxz*G1_zxz + G2_xxz*G1_zxx + G2_yxz*G1_zxy + G2_zxz*G1_zxz + G2_xzz*G1_xxx + G2_yzz*G1_xxy + G2_zzz*G1_xxz + G2_xxx*G1_xzz + G2_yxx*G1_yzz + G2_zxx*G1_zzz) + 
          gupyy * (G2_xxy*G1_zxy + G2_yxy*G1_zyy + G2_zxy*G1_zyz + G2_xyz*G1_xxy + G2_yyz*G1_xyy + G2_zyz*G1_xyz + G2_xxy*G1_xyz + G2_yxy*G1_yyz + G2_zxy*G1_zyz) + 
          gupyz * (G2_xxy*G1_zxz + G2_yxy*G1_zyz + G2_zxy*G1_zzz + G2_xyz*G1_xxz + G2_yyz*G1_xyz + G2_zyz*G1_xzz + G2_xxz*G1_xyz + G2_yxz*G1_yyz + G2_zxz*G1_zyz + G2_xxz*G1_zxy + G2_yxz*G1_zyy + G2_zxz*G1_zyz + G2_xzz*G1_xxy + G2_yzz*G1_xyy + G2_zzz*G1_xyz + G2_xxy*G1_xzz + G2_yxy*G1_yzz + G2_zxy*G1_zzz) + 
          gupzz * (G2_xxz*G1_zxz + G2_yxz*G1_zyz + G2_zxz*G1_zzz + G2_xzz*G1_xxz + G2_yzz*G1_xyz + G2_zzz*G1_xzz + G2_xxz*G1_xzz + G2_yxz*G1_yzz + G2_zxz*G1_zzz);

    l_Ryz = HALF * ( - l_Ryz + l_gxy * dGamxz + l_gyy * dGamyz + l_gyz * dGamzz + l_gxz * dGamxy + l_gyz * dGamyy + l_gzz * dGamzy + Gamxa * G1_yxz + Gamya * G1_yyz + Gamza * G1_yzz + Gamxa * G1_zxy + Gamya * G1_zyy + Gamza * G1_zyz) + 
          gupxx * (G2_xxy*G1_zxx + G2_yxy*G1_zxy + G2_zxy*G1_zxz + G2_xxz*G1_yxx + G2_yxz*G1_yxy + G2_zxz*G1_yxz + G2_xxy*G1_xxz + G2_yxy*G1_yxz + G2_zxy*G1_zxz) + 
          gupxy * (G2_xxy*G1_zxy + G2_yxy*G1_zyy + G2_zxy*G1_zyz + G2_xxz*G1_yxy + G2_yxz*G1_yyy + G2_zxz*G1_yyz + G2_xyy*G1_xxz + G2_yyy*G1_yxz + G2_zyy*G1_zxz + G2_xyy*G1_zxx + G2_yyy*G1_zxy + G2_zyy*G1_zxz + G2_xyz*G1_yxx + G2_yyz*G1_yxy + G2_zyz*G1_yxz + G2_xxy*G1_xyz + G2_yxy*G1_yyz + G2_zxy*G1_zyz) + 
          gupxz * (G2_xxy*G1_zxz + G2_yxy*G1_zyz + G2_zxy*G1_zzz + G2_xxz*G1_yxz + G2_yxz*G1_yyz + G2_zxz*G1_yzz + G2_xyz*G1_xxz + G2_yyz*G1_yxz + G2_zyz*G1_zxz + G2_xyz*G1_zxx + G2_yyz*G1_zxy + G2_zyz*G1_zxz + G2_xzz*G1_yxx + G2_yzz*G1_yxy + G2_zzz*G1_yxz + G2_xxy*G1_xzz + G2_yxy*G1_yzz + G2_zxy*G1_zzz) + 
          gupyy * (G2_xyy*G1_zxy + G2_yyy*G1_zyy + G2_zyy*G1_zyz + G2_xyz*G1_yxy + G2_yyz*G1_yyy + G2_zyz*G1_yyz + G2_xyy*G1_xyz + G2_yyy*G1_yyz + G2_zyy*G1_zyz) + 
          gupyz * (G2_xyy*G1_zxz + G2_yyy*G1_zyz + G2_zyy*G1_zzz + G2_xyz*G1_yxz + G2_yyz*G1_yyz + G2_zyz*G1_yzz + G2_xyz*G1_xyz + G2_yyz*G1_yyz + G2_zyz*G1_zyz + G2_xyz*G1_zxy + G2_yyz*G1_zyy + G2_zyz*G1_zyz + G2_xzz*G1_yxy + G2_yzz*G1_yyy + G2_zzz*G1_yyz + G2_xyy*G1_xzz + G2_yyy*G1_yzz + G2_zyy*G1_zzz) + 
          gupzz * (G2_xyz*G1_zxz + G2_yyz*G1_zyz + G2_zyz*G1_zzz + G2_xzz*G1_yxz + G2_yzz*G1_yyz + G2_zzz*G1_yzz + G2_xyz*G1_xzz + G2_yyz*G1_yzz + G2_zyz*G1_zzz);

    Gamxxx_out[idx] = G2_xxx; Gamxxy_out[idx] = G2_xxy; Gamxxz_out[idx] = G2_xxz;
    Gamxyy_out[idx] = G2_xyy; Gamxyz_out[idx] = G2_xyz; Gamxzz_out[idx] = G2_xzz;
    Gamyxx_out[idx] = G2_yxx; Gamyxy_out[idx] = G2_yxy; Gamyxz_out[idx] = G2_yxz;
    Gamyyy_out[idx] = G2_yyy; Gamyyz_out[idx] = G2_yyz; Gamyzz_out[idx] = G2_yzz;
    Gamzxx_out[idx] = G2_zxx; Gamzxy_out[idx] = G2_zxy; Gamzxz_out[idx] = G2_zxz;
    Gamzyy_out[idx] = G2_zyy; Gamzyz_out[idx] = G2_zyz; Gamzzz_out[idx] = G2_zzz;

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

    // (Smem Batch Derivatives 载入和求导部分保持不变，这段设计很高效)
    double betaxx, betaxy, betaxz, bx_gxxx, bx_gxyx, bx_gxzx, bx_gyyx, bx_gyzx, bx_gzzx;
    load_field_to_smem(smem, betax, dims, block_i, block_j, block_k, ANTI, SYM, SYM);
    __syncthreads(); compute_all_derivs_smem(smem, &betaxx, &betaxy, &betaxz, &bx_gxxx, &bx_gxyx, &bx_gxzx, &bx_gyyx, &bx_gyzx, &bx_gzzx, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double betayx, betayy, betayz, by_gxxy, by_gxyy, by_gxzy, by_gyyy, by_gyzy, by_gzzy;
    load_field_to_smem(smem, betay, dims, block_i, block_j, block_k, SYM, ANTI, SYM);
    __syncthreads(); compute_all_derivs_smem(smem, &betayx, &betayy, &betayz, &by_gxxy, &by_gxyy, &by_gxzy, &by_gyyy, &by_gyzy, &by_gzzy, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double betazx, betazy, betazz, bz_gxxz, bz_gxyz, bz_gxzz, bz_gyyz, bz_gyzz, bz_gzzz;
    load_field_to_smem(smem, betaz, dims, block_i, block_j, block_k, SYM, SYM, ANTI);
    __syncthreads(); compute_all_derivs_smem(smem, &betazx, &betazy, &betazz, &bz_gxxz, &bz_gxyz, &bz_gxzz, &bz_gyyz, &bz_gyzz, &bz_gzzz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double Lapx, Lapy, Lapz, fxx_Lap, fxy_Lap, fxz_Lap, fyy_Lap, fyz_Lap, fzz_Lap;
    load_field_to_smem(smem, Lap, dims, block_i, block_j, block_k, SYM, SYM, SYM);
    __syncthreads(); compute_all_derivs_smem(smem, &Lapx, &Lapy, &Lapz, &fxx_Lap, &fxy_Lap, &fxz_Lap, &fyy_Lap, &fyz_Lap, &fzz_Lap, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double chix, chiy, chiz, chixx, chixy, chixz, chiyy, chiyz, chizz;
    load_field_to_smem(smem, chi, dims, block_i, block_j, block_k, SYM, SYM, SYM);
    __syncthreads(); compute_all_derivs_smem(smem, &chix, &chiy, &chiz, &chixx, &chixy, &chixz, &chiyy, &chiyz, &chizz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax); __syncthreads();

    double Kx, Ky, Kz;
    load_field_to_smem(smem, trK, dims, block_i, block_j, block_k, SYM, SYM, SYM);
    __syncthreads(); compute_first_derivs_smem(smem, &Kx, &Ky, &Kz, dX, dY, dZ, i, j, k, imin, jmin, kmin, imax, jmax, kmax);

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

    double det = l_gxx * l_gyy * l_gzz + TWO * l_gxy * l_gyz * l_gxz - l_gxz * l_gyy * l_gxz - l_gxx * l_gyz * l_gyz - l_gzz * l_gxy * l_gxy;
    double inv_det = 1.0 / det;

    double gupxx = (l_gyy * l_gzz - l_gyz * l_gyz) * inv_det;
    double gupxy = (l_gyz * l_gxz - l_gxy * l_gzz) * inv_det;
    double gupxz = (l_gxy * l_gyz - l_gyy * l_gxz) * inv_det;
    double gupyy = (l_gxx * l_gzz - l_gxz * l_gxz) * inv_det;
    double gupyz = (l_gxy * l_gxz - l_gxx * l_gyz) * inv_det;
    double gupzz = (l_gxx * l_gyy - l_gxy * l_gxy) * inv_det;

    double Au_d_xx = gupxx * l_Axx + gupxy * l_Axy + gupxz * l_Axz;
    double Au_d_xy = gupxx * l_Axy + gupxy * l_Ayy + gupxz * l_Ayz;
    double Au_d_xz = gupxx * l_Axz + gupxy * l_Ayz + gupxz * l_Azz;
    double Au_d_yx = gupxy * l_Axx + gupyy * l_Axy + gupyz * l_Axz;
    double Au_d_yy = gupxy * l_Axy + gupyy * l_Ayy + gupyz * l_Ayz;
    double Au_d_yz = gupxy * l_Axz + gupyy * l_Ayz + gupyz * l_Azz;
    double Au_d_zx = gupxz * l_Axx + gupyz * l_Axy + gupzz * l_Axz;
    double Au_d_zy = gupxz * l_Axy + gupyz * l_Ayy + gupzz * l_Ayz;
    double Au_d_zz = gupxz * l_Axz + gupyz * l_Ayz + gupzz * l_Azz;

    double Aup_xx = Au_d_xx * gupxx + Au_d_xy * gupxy + Au_d_xz * gupxz;
    double Aup_xy = Au_d_xx * gupxy + Au_d_xy * gupyy + Au_d_xz * gupyz;
    double Aup_xz = Au_d_xx * gupxz + Au_d_xy * gupyz + Au_d_xz * gupzz;
    double Aup_yy = Au_d_yx * gupxy + Au_d_yy * gupyy + Au_d_yz * gupyz;
    double Aup_yz = Au_d_yx * gupxz + Au_d_yy * gupyz + Au_d_yz * gupzz;
    double Aup_zz = Au_d_zx * gupxz + Au_d_zy * gupyz + Au_d_zz * gupzz;

    // 【强宣誓常量化】：切断共形克氏符被篡改的通道
    const double l_Gamxxx = Gamxxx_in[idx]; const double l_Gamxxy = Gamxxy_in[idx]; const double l_Gamxxz = Gamxxz_in[idx];
    const double l_Gamxyy = Gamxyy_in[idx]; const double l_Gamxyz = Gamxyz_in[idx]; const double l_Gamxzz = Gamxzz_in[idx];
    const double l_Gamyxx = Gamyxx_in[idx]; const double l_Gamyxy = Gamyxy_in[idx]; const double l_Gamyxz = Gamyxz_in[idx];
    const double l_Gamyyy = Gamyyy_in[idx]; const double l_Gamyyz = Gamyyz_in[idx]; const double l_Gamyzz = Gamyzz_in[idx];
    const double l_Gamzxx = Gamzxx_in[idx]; const double l_Gamzxy = Gamzxy_in[idx]; const double l_Gamzxz = Gamzxz_in[idx];
    const double l_Gamzyy = Gamzyy_in[idx]; const double l_Gamzyz = Gamzyz_in[idx]; const double l_Gamzzz = Gamzzz_in[idx];

    double l_Rxx = Rxx_in[idx]; double l_Rxy = Rxy_in[idx]; double l_Rxz = Rxz_in[idx];
    double l_Ryy = Ryy_in[idx]; double l_Ryz = Ryz_in[idx]; double l_Rzz = Rzz_in[idx];

    double val_Sx = Sx[idx]; double val_Sy = Sy[idx]; double val_Sz = Sz[idx];

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
    
    double fxx = chixx - l_Gamxxx * chix - l_Gamyxx * chiy - l_Gamzxx * chiz;
    double fxy = chixy - l_Gamxxy * chix - l_Gamyxy * chiy - l_Gamzxy * chiz;
    double fxz = chixz - l_Gamxxz * chix - l_Gamyxz * chiy - l_Gamzxz * chiz;
    double fyy = chiyy - l_Gamxyy * chix - l_Gamyyy * chiy - l_Gamzyy * chiz;
    double fyz = chiyz - l_Gamxyz * chix - l_Gamyyz * chiy - l_Gamzyz * chiz;
    double fzz = chizz - l_Gamxzz * chix - l_Gamyzz * chiy - l_Gamzzz * chiz;

    double f_scalar = gupxx * (fxx - F3o2*inv_chin1 * chix * chix) + gupyy * (fyy - F3o2*inv_chin1 * chiy * chiy) + gupzz * (fzz - F3o2*inv_chin1 * chiz * chiz) + 
                      TWO * (gupxy * (fxy - F3o2*inv_chin1 * chix * chiy) + gupxz * (fxz - F3o2*inv_chin1 * chix * chiz) + gupyz * (fyz - F3o2*inv_chin1 * chiy * chiz));
    
    l_Rxx += (fxx - chix*chix*inv_chin1*HALF + l_gxx * f_scalar)*inv_chin1*HALF;
    l_Ryy += (fyy - chiy*chiy*inv_chin1*HALF + l_gyy * f_scalar)*inv_chin1*HALF;
    l_Rzz += (fzz - chiz*chiz*inv_chin1*HALF + l_gzz * f_scalar)*inv_chin1*HALF;
    l_Rxy += (fxy - chix*chiy*inv_chin1*HALF + l_gxy * f_scalar)*inv_chin1*HALF;
    l_Rxz += (fxz - chix*chiz*inv_chin1*HALF + l_gxz * f_scalar)*inv_chin1*HALF;
    l_Ryz += (fyz - chiy*chiz*inv_chin1*HALF + l_gyz * f_scalar)*inv_chin1*HALF;

    // 【全新架构】：直接提取共形导数，绕开对物理克氏符的 18 次昂贵更新
    double gx_phy = (gupxx * chix + gupxy * chiy + gupxz * chiz) * inv_chin1;
    double gy_phy = (gupxy * chix + gupyy * chiy + gupyz * chiz) * inv_chin1;
    double gz_phy = (gupxz * chix + gupyz * chiy + gupzz * chiz) * inv_chin1;
    double chi_dot_Lap = gx_phy * Lapx + gy_phy * Lapy + gz_phy * Lapz;

    // 1. 先用恒定的共形克氏符计算共形 Hessian
    double D2_Lap_xx = fxx_Lap - l_Gamxxx*Lapx - l_Gamyxx*Lapy - l_Gamzxx*Lapz;
    double D2_Lap_yy = fyy_Lap - l_Gamxyy*Lapx - l_Gamyyy*Lapy - l_Gamzyy*Lapz;
    double D2_Lap_zz = fzz_Lap - l_Gamxzz*Lapx - l_Gamyzz*Lapy - l_Gamzzz*Lapz;
    double D2_Lap_xy = fxy_Lap - l_Gamxxy*Lapx - l_Gamyxy*Lapy - l_Gamzxy*Lapz;
    double D2_Lap_xz = fxz_Lap - l_Gamxxz*Lapx - l_Gamyxz*Lapy - l_Gamzxz*Lapz;
    double D2_Lap_yz = fyz_Lap - l_Gamxyz*Lapx - l_Gamyyz*Lapy - l_Gamzyz*Lapz;

    // 2. 利用数学等价代换直接获得物理 Hessian，避免 18 个物理克氏符的计算
    double fxx_Lap_phy = D2_Lap_xx + (chix * Lapx) * inv_chin1 - HALF * l_gxx * chi_dot_Lap;
    double fyy_Lap_phy = D2_Lap_yy + (chiy * Lapy) * inv_chin1 - HALF * l_gyy * chi_dot_Lap;
    double fzz_Lap_phy = D2_Lap_zz + (chiz * Lapz) * inv_chin1 - HALF * l_gzz * chi_dot_Lap;
    double fxy_Lap_phy = D2_Lap_xy + HALF * (chix * Lapy + chiy * Lapx) * inv_chin1 - HALF * l_gxy * chi_dot_Lap;
    double fxz_Lap_phy = D2_Lap_xz + HALF * (chix * Lapz + chiz * Lapx) * inv_chin1 - HALF * l_gxz * chi_dot_Lap;
    double fyz_Lap_phy = D2_Lap_yz + HALF * (chiy * Lapz + chiz * Lapy) * inv_chin1 - HALF * l_gyz * chi_dot_Lap;

    // 【迹的数学捷径】：无需针对物理 Hessian 再做 6 次 MADD 缩并
    double conf_trK_rhs = gupxx * D2_Lap_xx + gupyy * D2_Lap_yy + gupzz * D2_Lap_zz + TWO * (gupxy * D2_Lap_xy + gupxz * D2_Lap_xz + gupyz * D2_Lap_yz);
    double trK_rhs_val = conf_trK_rhs - HALF * chi_dot_Lap; 

    double S = chin1 * (gupxx * Sxx[idx] + gupyy * Syy[idx] + gupzz * Szz[idx] + TWO * (gupxy * Sxy[idx] + gupxz * Sxz[idx] + gupyz * Syz[idx]));

    double term_xx = l_Axx * Au_d_xx + l_Axy * Au_d_yx + l_Axz * Au_d_zx;
    double term_xy = l_Axx * Au_d_xy + l_Axy * Au_d_yy + l_Axz * Au_d_zy;
    double term_xz = l_Axx * Au_d_xz + l_Axy * Au_d_yz + l_Axz * Au_d_zz;
    double term_yy = l_Axy * Au_d_xy + l_Ayy * Au_d_yy + l_Ayz * Au_d_zy;
    double term_yz = l_Axy * Au_d_xz + l_Ayy * Au_d_yz + l_Ayz * Au_d_zz;
    double term_zz = l_Axz * Au_d_xz + l_Ayz * Au_d_yz + l_Azz * Au_d_zz;

    // 【二次收缩捷径】：利用现成的 Aup 算迹，省掉对 term_ij 的矩阵收缩
    double trA2 = l_Axx * Aup_xx + l_Ayy * Aup_yy + l_Azz * Aup_zz + TWO * (l_Axy * Aup_xy + l_Axz * Aup_xz + l_Ayz * Aup_yz);

    double f = F2o3 * val_trK * val_trK - trA2 - F16*PI*rho[idx] + EIGHT*PI*S;
    double f_trace = -F1o3 * (trK_rhs_val + alpn1*inv_chin1 * f);

    // 换用无污染的物理 Hessian
    double src_xx = alpn1 * (l_Rxx - EIGHT*PI*Sxx[idx]) - fxx_Lap_phy;
    double src_yy = alpn1 * (l_Ryy - EIGHT*PI*Syy[idx]) - fyy_Lap_phy;
    double src_zz = alpn1 * (l_Rzz - EIGHT*PI*Szz[idx]) - fzz_Lap_phy;
    double src_xy = alpn1 * (l_Rxy - EIGHT*PI*Sxy[idx]) - fxy_Lap_phy;
    double src_xz = alpn1 * (l_Rxz - EIGHT*PI*Sxz[idx]) - fxz_Lap_phy;
    double src_yz = alpn1 * (l_Ryz - EIGHT*PI*Syz[idx]) - fyz_Lap_phy;

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
    bool is_valid_point = (i <= imax && j <= jmax && k <= kmax);

    if (is_valid_point) {
        idx = i + ex0 * (j + ex1 * k);
        vx = __ldg(&betax[idx]);
        vy = __ldg(&betay[idx]);
        vz = __ldg(&betaz[idx]);
    }

    // ==============================================================================
    // 线性算子融合 (Linear Operator Fusion)：
    // 提取所有方向的权重 W[-3...3] 并映射到 W[0...6]
    // 完美消灭重复判定，并将分支发散隔绝在 Lambda 之外。
    // ==============================================================================
    double Wx[7] = {0}, Wy[7] = {0}, Wz[7] = {0};

    if (is_valid_point) {
        // X 方向平流算子
        double v_dx = vx * d12dx;
        if (vx > 0.0) {
            if (i + 3 <= imax) { Wx[2]-=3*v_dx; Wx[3]-=10*v_dx; Wx[4]+=18*v_dx; Wx[5]-=6*v_dx; Wx[6]+=v_dx; }
            else if (i + 2 <= imax) { Wx[1]+=v_dx; Wx[2]-=8*v_dx; Wx[4]+=8*v_dx; Wx[5]-=v_dx; }
            else if (i + 1 <= imax) { Wx[4]+=3*v_dx; Wx[3]+=10*v_dx; Wx[2]-=18*v_dx; Wx[1]+=6*v_dx; Wx[0]-=v_dx; }
        } else if (vx < 0.0) {
            if (i - 3 >= imin) { Wx[4]+=3*v_dx; Wx[3]+=10*v_dx; Wx[2]-=18*v_dx; Wx[1]+=6*v_dx; Wx[0]-=v_dx; }
            else if (i - 2 >= imin) { Wx[1]+=v_dx; Wx[2]-=8*v_dx; Wx[4]+=8*v_dx; Wx[5]-=v_dx; }
            else if (i - 1 >= imin) { Wx[2]-=3*v_dx; Wx[3]-=10*v_dx; Wx[4]+=18*v_dx; Wx[5]-=6*v_dx; Wx[6]+=v_dx; }
        }

        // Y 方向平流算子
        double v_dy = vy * d12dy;
        if (vy > 0.0) {
            if (j + 3 <= jmax) { Wy[2]-=3*v_dy; Wy[3]-=10*v_dy; Wy[4]+=18*v_dy; Wy[5]-=6*v_dy; Wy[6]+=v_dy; }
            else if (j + 2 <= jmax) { Wy[1]+=v_dy; Wy[2]-=8*v_dy; Wy[4]+=8*v_dy; Wy[5]-=v_dy; }
            else if (j + 1 <= jmax) { Wy[4]+=3*v_dy; Wy[3]+=10*v_dy; Wy[2]-=18*v_dy; Wy[1]+=6*v_dy; Wy[0]-=v_dy; }
        } else if (vy < 0.0) {
            if (j - 3 >= jmin) { Wy[4]+=3*v_dy; Wy[3]+=10*v_dy; Wy[2]-=18*v_dy; Wy[1]+=6*v_dy; Wy[0]-=v_dy; }
            else if (j - 2 >= jmin) { Wy[1]+=v_dy; Wy[2]-=8*v_dy; Wy[4]+=8*v_dy; Wy[5]-=v_dy; }
            else if (j - 1 >= jmin) { Wy[2]-=3*v_dy; Wy[3]-=10*v_dy; Wy[4]+=18*v_dy; Wy[5]-=6*v_dy; Wy[6]+=v_dy; }
        }

        // Z 方向平流算子
        double v_dz = vz * d12dz;
        if (vz > 0.0) {
            if (k + 3 <= kmax) { Wz[2]-=3*v_dz; Wz[3]-=10*v_dz; Wz[4]+=18*v_dz; Wz[5]-=6*v_dz; Wz[6]+=v_dz; }
            else if (k + 2 <= kmax) { Wz[1]+=v_dz; Wz[2]-=8*v_dz; Wz[4]+=8*v_dz; Wz[5]-=v_dz; }
            else if (k + 1 <= kmax) { Wz[4]+=3*v_dz; Wz[3]+=10*v_dz; Wz[2]-=18*v_dz; Wz[1]+=6*v_dz; Wz[0]-=v_dz; }
        } else if (vz < 0.0) {
            if (k - 3 >= kmin) { Wz[4]+=3*v_dz; Wz[3]+=10*v_dz; Wz[2]-=18*v_dz; Wz[1]+=6*v_dz; Wz[0]-=v_dz; }
            else if (k - 2 >= kmin) { Wz[1]+=v_dz; Wz[2]-=8*v_dz; Wz[4]+=8*v_dz; Wz[5]-=v_dz; }
            else if (k - 1 >= kmin) { Wz[2]-=3*v_dz; Wz[3]-=10*v_dz; Wz[4]+=18*v_dz; Wz[5]-=6*v_dz; Wz[6]+=v_dz; }
        }

        // KO 耗散算子叠加 (Dissipation Operator Fusion)
        if (eps > 0.0 && i - 3 >= imin && i + 3 <= imax && j - 3 >= jmin && j + 3 <= jmax && k - 3 >= kmin && k + 3 <= kmax) {
            double edx = eps / 64.0 / dX; 
            double edy = eps / 64.0 / dY; 
            double edz = eps / 64.0 / dZ;
            Wx[0]+=edx; Wx[1]-=6*edx; Wx[2]+=15*edx; Wx[3]-=20*edx; Wx[4]+=15*edx; Wx[5]-=6*edx; Wx[6]+=edx;
            Wy[0]+=edy; Wy[1]-=6*edy; Wy[2]+=15*edy; Wy[3]-=20*edy; Wy[4]+=15*edy; Wy[5]-=6*edy; Wy[6]+=edy;
            Wz[0]+=edz; Wz[1]-=6*edz; Wz[2]+=15*edz; Wz[3]-=20*edz; Wz[4]+=15*edz; Wz[5]-=6*edz; Wz[6]+=edz;
        }
    }

    // 收缩三维中心点权重，彻底避免重复读取 fh(0,0,0)
    double W0_center = Wx[3] + Wy[3] + Wz[3];

    __shared__ double smem[10][14][14];

    // 新的 Lambda：纯粹的无分支内联点积
    auto process_field = [&](const double* f, double* f_rhs, double SYM1, double SYM2, double SYM3) {
        load_field_to_smem_rad3(smem, f, dims, block_i, block_j, block_k, SYM1, SYM2, SYM3);
        __syncthreads(); 

        if (is_valid_point) {
            // 直接计算局部偏移量，跳过函数调用的寻址开销
            int tx = threadIdx.x + 3;
            int ty = threadIdx.y + 3;
            int tz = threadIdx.z + 3;

            // 无条件、无分支的统一差分模板
            double rhs_add = W0_center * smem[tz][ty][tx] +
                Wx[0]*smem[tz][ty][tx-3] + Wx[1]*smem[tz][ty][tx-2] + Wx[2]*smem[tz][ty][tx-1] + Wx[4]*smem[tz][ty][tx+1] + Wx[5]*smem[tz][ty][tx+2] + Wx[6]*smem[tz][ty][tx+3] +
                Wy[0]*smem[tz][ty-3][tx] + Wy[1]*smem[tz][ty-2][tx] + Wy[2]*smem[tz][ty-1][tx] + Wy[4]*smem[tz][ty+1][tx] + Wy[5]*smem[tz][ty+2][tx] + Wy[6]*smem[tz][ty+3][tx] +
                Wz[0]*smem[tz-3][ty][tx] + Wz[1]*smem[tz-2][ty][tx] + Wz[2]*smem[tz-1][ty][tx] + Wz[4]*smem[tz+1][ty][tx] + Wz[5]*smem[tz+2][ty][tx] + Wz[6]*smem[tz+3][ty][tx];

            f_rhs[idx] += rhs_add;
        }
        __syncthreads(); 
    };

    // 此处保持 24 次处理不变，但单次执行成本已被压缩至极限
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
    const double EIGHT = 8.0; const double F16 = 16.0; const double PI = M_PI;

    double chin1 = chi[idx] + 1.0; double val_trK = trK[idx];

    double l_gxx = dxx[idx] + 1.0; double l_gxy = gxy[idx]; double l_gxz = gxz[idx];
    double l_gyy = dyy[idx] + 1.0; double l_gyz = gyz[idx]; double l_gzz = dzz[idx] + 1.0;
    double l_Axx = Axx[idx]; double l_Axy = Axy[idx]; double l_Axz = Axz[idx];
    double l_Ayy = Ayy[idx]; double l_Ayz = Ayz[idx]; double l_Azz = Azz[idx];

    double det = l_gxx * l_gyy * l_gzz + TWO * l_gxy * l_gyz * l_gxz - l_gxz * l_gyy * l_gxz - l_gxx * l_gyz * l_gyz - l_gzz * l_gxy * l_gxy;
    double inv_det = 1.0 / det;

    double gupxx = (l_gyy * l_gzz - l_gyz * l_gyz) * inv_det;
    double gupxy = (l_gyz * l_gxz - l_gxy * l_gzz) * inv_det;
    double gupxz = (l_gxy * l_gyz - l_gyy * l_gxz) * inv_det;
    double gupyy = (l_gxx * l_gzz - l_gxz * l_gxz) * inv_det;
    double gupyz = (l_gxy * l_gxz - l_gxx * l_gyz) * inv_det;
    double gupzz = (l_gxx * l_gyy - l_gxy * l_gxy) * inv_det;

    // 直接从内存读取共形克氏符，不可被污染
    const double l_Gamxxx = Gamxxx_in[idx]; const double l_Gamxxy = Gamxxy_in[idx]; const double l_Gamxxz = Gamxxz_in[idx];
    const double l_Gamxyy = Gamxyy_in[idx]; const double l_Gamxyz = Gamxyz_in[idx]; const double l_Gamxzz = Gamxzz_in[idx];
    const double l_Gamyxx = Gamyxx_in[idx]; const double l_Gamyxy = Gamyxy_in[idx]; const double l_Gamyxz = Gamyxz_in[idx];
    const double l_Gamyyy = Gamyyy_in[idx]; const double l_Gamyyz = Gamyyz_in[idx]; const double l_Gamyzz = Gamyzz_in[idx];
    const double l_Gamzxx = Gamzxx_in[idx]; const double l_Gamzxy = Gamzxy_in[idx]; const double l_Gamzxz = Gamzxz_in[idx];
    const double l_Gamzyy = Gamzyy_in[idx]; const double l_Gamzyz = Gamzyz_in[idx]; const double l_Gamzzz = Gamzzz_in[idx];

    double Gamxa = gupxx * l_Gamxxx + gupyy * l_Gamxyy + gupzz * l_Gamxzz + TWO * (gupxy * l_Gamxxy + gupxz * l_Gamxxz + gupyz * l_Gamxyz);
    double Gamya = gupxx * l_Gamyxx + gupyy * l_Gamyyy + gupzz * l_Gamyzz + TWO * (gupxy * l_Gamyxy + gupxz * l_Gamyxz + gupyz * l_Gamyyz);
    double Gamza = gupxx * l_Gamzxx + gupyy * l_Gamzyy + gupzz * l_Gamzzz + TWO * (gupxy * l_Gamzxy + gupxz * l_Gamzxz + gupyz * l_Gamzyz);

    Gmx_Res[idx] = Gamx[idx] - Gamxa; Gmy_Res[idx] = Gamy[idx] - Gamya; Gmz_Res[idx] = Gamz[idx] - Gamza;

    // 【数学降维 1：混合张量 $\tilde{A}^i_j$ 直接用于哈密顿迹计算】
    double Au_d_xx = gupxx * l_Axx + gupxy * l_Axy + gupxz * l_Axz;
    double Au_d_xy = gupxx * l_Axy + gupxy * l_Ayy + gupxz * l_Ayz;
    double Au_d_xz = gupxx * l_Axz + gupxy * l_Ayz + gupxz * l_Azz;
    double Au_d_yx = gupxy * l_Axx + gupyy * l_Axy + gupyz * l_Axz;
    double Au_d_yy = gupxy * l_Axy + gupyy * l_Ayy + gupyz * l_Ayz;
    double Au_d_yz = gupxy * l_Axz + gupyy * l_Ayz + gupyz * l_Azz;
    double Au_d_zx = gupxz * l_Axx + gupyz * l_Axy + gupzz * l_Axz;
    double Au_d_zy = gupxz * l_Axy + gupyz * l_Ayy + gupzz * l_Ayz;
    double Au_d_zz = gupxz * l_Axz + gupyz * l_Ayz + gupzz * l_Azz;

    // 原来那 6 个巨大的 `term_xx` 展开被这个优雅的点积完美替代
    double trA2 = Au_d_xx*Au_d_xx + Au_d_yy*Au_d_yy + Au_d_zz*Au_d_zz + TWO*(Au_d_xy*Au_d_yx + Au_d_xz*Au_d_zx + Au_d_yz*Au_d_zy);

    // 【数学降维 2：哈密顿标量捷径】
    double conf_R = gupxx * Rxx_in[idx] + gupyy * Ryy_in[idx] + gupzz * Rzz_in[idx] + TWO * (gupxy * Rxy_in[idx] + gupxz * Rxz_in[idx] + gupyz * Ryz_in[idx]);
    
    double fxx = chixx - l_Gamxxx * chix - l_Gamyxx * chiy - l_Gamzxx * chiz;
    double fyy = chiyy - l_Gamxyy * chix - l_Gamyyy * chiy - l_Gamzyy * chiz;
    double fzz = chizz - l_Gamxzz * chix - l_Gamyzz * chiy - l_Gamzzz * chiz;
    double fxy = chixy - l_Gamxxy * chix - l_Gamyxy * chiy - l_Gamzxy * chiz;
    double fxz = chixz - l_Gamxxz * chix - l_Gamyxz * chiy - l_Gamzxz * chiz;
    double fyz = chiyz - l_Gamxyz * chix - l_Gamyyz * chiy - l_Gamzyz * chiz;

    double D2_chi = gupxx * fxx + gupyy * fyy + gupzz * fzz + TWO * (gupxy * fxy + gupxz * fxz + gupyz * fyz);
    double grad_chi_sq = gupxx * chix*chix + gupyy * chiy*chiy + gupzz * chiz*chiz + TWO * (gupxy * chix*chiy + gupxz * chix*chiz + gupyz * chiy*chiz);
    
    // 直接组装物理标量曲率，绕过 6 个物理里奇张量的强行更新
    double phys_R = chin1 * conf_R + TWO * D2_chi - 2.5 / chin1 * grad_chi_sq;
    ham_Res[idx] = phys_R + F2o3 * val_trK * val_trK - trA2 - F16 * PI * rho[idx];

    // 【数学降维 3：动量约束的无物理克氏符推导】
    // \partial_j \tilde{A}^j_i 项：直接矩阵偏导
    double div_A_x = gupxx * d_Axx_x + gupyy * d_Axy_y + gupzz * d_Axz_z + gupxy * (d_Axx_y + d_Axy_x) + gupxz * (d_Axx_z + d_Axz_x) + gupyz * (d_Axy_z + d_Axz_y);
    double div_A_y = gupxx * d_Axy_x + gupyy * d_Ayy_y + gupzz * d_Ayz_z + gupxy * (d_Axy_y + d_Ayy_x) + gupxz * (d_Axy_z + d_Ayz_x) + gupyz * (d_Ayy_z + d_Ayz_y);
    double div_A_z = gupxx * d_Axz_x + gupyy * d_Ayz_y + gupzz * d_Azz_z + gupxy * (d_Axz_y + d_Ayz_x) + gupxz * (d_Axz_z + d_Azz_x) + gupyz * (d_Ayz_z + d_Azz_y);

    // 第一类共形导数补偿： - \tilde{\Gamma}^l \tilde{A}_{li}
    double Gam_A_x = Gamxa * l_Axx + Gamya * l_Axy + Gamza * l_Axz;
    double Gam_A_y = Gamxa * l_Axy + Gamya * l_Ayy + Gamza * l_Ayz;
    double Gam_A_z = Gamxa * l_Axz + Gamya * l_Ayz + Gamza * l_Azz;

    // 第二类共形导数补偿： - \tilde{A}^j_l \tilde{\Gamma}^l_{ji}
    double A_Gam_x = Au_d_xx * l_Gamxxx + Au_d_yx * l_Gamxxy + Au_d_zx * l_Gamxxz + Au_d_xy * l_Gamyxx + Au_d_yy * l_Gamyxy + Au_d_zy * l_Gamyxz + Au_d_xz * l_Gamzxx + Au_d_yz * l_Gamzxy + Au_d_zz * l_Gamzxz;
    double A_Gam_y = Au_d_xx * l_Gamxxy + Au_d_yx * l_Gamxyy + Au_d_zx * l_Gamxyz + Au_d_xy * l_Gamyxy + Au_d_yy * l_Gamyyy + Au_d_zy * l_Gamyyz + Au_d_xz * l_Gamzxy + Au_d_yz * l_Gamzyy + Au_d_zz * l_Gamzyz;
    double A_Gam_z = Au_d_xx * l_Gamxxz + Au_d_yx * l_Gamxyz + Au_d_zx * l_Gamxzz + Au_d_xy * l_Gamyxz + Au_d_yy * l_Gamyyz + Au_d_zy * l_Gamyzz + Au_d_xz * l_Gamzxz + Au_d_yz * l_Gamzyz + Au_d_zz * l_Gamzzz;

    // Conformal factor 修正项
    double chi_A_x = 1.5 / chin1 * (Au_d_xx * chix + Au_d_yx * chiy + Au_d_zx * chiz);
    double chi_A_y = 1.5 / chin1 * (Au_d_xy * chix + Au_d_yy * chiy + Au_d_zy * chiz);
    double chi_A_z = 1.5 / chin1 * (Au_d_xz * chix + Au_d_yz * chiy + Au_d_zz * chiz);

    // 最终组装物理动量约束 (彻底埋葬那 18 块臭长又慢的 `DA_ijk`)
    movx_Res[idx] = div_A_x - Gam_A_x - A_Gam_x - chi_A_x - F2o3 * Kx - EIGHT * PI * Sx[idx];
    movy_Res[idx] = div_A_y - Gam_A_y - A_Gam_y - chi_A_y - F2o3 * Ky - EIGHT * PI * Sy[idx];
    movz_Res[idx] = div_A_z - Gam_A_z - A_Gam_z - chi_A_z - F2o3 * Kz - EIGHT * PI * Sz[idx];
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