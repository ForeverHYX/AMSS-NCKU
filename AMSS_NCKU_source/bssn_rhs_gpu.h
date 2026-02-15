#ifndef BSSN_RHS_GPU_H
#define BSSN_RHS_GPU_H

#include <cuda_runtime.h>

// ==========================================
// Kernel 1: 导数、逆度规与连接系数
// ==========================================
__global__ void bssn_derivatives_kernel(
    // 维度信息
    const int ex0, const int ex1, const int ex2, 
    const int symmetry, const int lev, const int co,
    // 坐标
    const double* __restrict__ X, const double* __restrict__ Y, const double* __restrict__ Z,
    // 输入变量 (Metric, Gauge, Curvature)
    const double* __restrict__ chi, const double* __restrict__ trK,
    const double* __restrict__ dxx, const double* __restrict__ gxy, const double* __restrict__ gxz,
    const double* __restrict__ dyy, const double* __restrict__ gyz, const double* __restrict__ dzz,
    const double* __restrict__ Axx, const double* __restrict__ Axy, const double* __restrict__ Axz,
    const double* __restrict__ Ayy, const double* __restrict__ Ayz, const double* __restrict__ Azz,
    const double* __restrict__ Lap, 
    const double* __restrict__ betax, const double* __restrict__ betay, const double* __restrict__ betaz,
    const double* __restrict__ Gamx, const double* __restrict__ Gamy, const double* __restrict__ Gamz, // 用于 Res 计算
    // 输出: RHS 部分 (chi_rhs, gij_rhs)
    double* __restrict__ chi_rhs, 
    double* __restrict__ gxx_rhs, double* __restrict__ gxy_rhs, double* __restrict__ gxz_rhs,
    double* __restrict__ gyy_rhs, double* __restrict__ gyz_rhs, double* __restrict__ gzz_rhs,
    // 输出: 中间导数 (Global Memory, 供后续 Kernel 使用)
    double* __restrict__ chix_out, double* __restrict__ chiy_out, double* __restrict__ chiz_out,
    double* __restrict__ betaxx_out, double* __restrict__ betaxy_out, double* __restrict__ betaxz_out,
    double* __restrict__ betayx_out, double* __restrict__ betayy_out, double* __restrict__ betayz_out,
    double* __restrict__ betazx_out, double* __restrict__ betazy_out, double* __restrict__ betazz_out,
    // Metric 导数 (gij,k)
    double* __restrict__ gxxx_out, double* __restrict__ gxxy_out, double* __restrict__ gxxz_out,
    double* __restrict__ gxyx_out, double* __restrict__ gxyy_out, double* __restrict__ gxyz_out,
    double* __restrict__ gxzx_out, double* __restrict__ gxzy_out, double* __restrict__ gxzz_out,
    double* __restrict__ gyyx_out, double* __restrict__ gyyy_out, double* __restrict__ gyyz_out,
    double* __restrict__ gyzx_out, double* __restrict__ gyzy_out, double* __restrict__ gyzz_out,
    double* __restrict__ gzzx_out, double* __restrict__ gzzy_out, double* __restrict__ gzzz_out,
    // 输出: 连接系数 (Gam^k_ij)
    double* __restrict__ Gamxxx, double* __restrict__ Gamxxy, double* __restrict__ Gamxxz,
    double* __restrict__ Gamxyy, double* __restrict__ Gamxyz, double* __restrict__ Gamxzz,
    double* __restrict__ Gamyxx, double* __restrict__ Gamyxy, double* __restrict__ Gamyxz,
    double* __restrict__ Gamyyy, double* __restrict__ Gamyyz, double* __restrict__ Gamyzz,
    double* __restrict__ Gamzxx, double* __restrict__ Gamzxy, double* __restrict__ Gamzxz,
    double* __restrict__ Gamzyy, double* __restrict__ Gamzyz, double* __restrict__ Gamzzz,
    // 输出: 逆度规 (gup) 
    double* __restrict__ gupxx_out, double* __restrict__ gupxy_out, double* __restrict__ gupxz_out,
    double* __restrict__ gupyy_out, double* __restrict__ gupyz_out, double* __restrict__ gupzz_out,
    // 输出: 约束残差 (仅当 co==0)
    double* __restrict__ Gmx_Res, double* __restrict__ Gmy_Res, double* __restrict__ Gmz_Res
);

__global__ void bssn_rhs_core_kernel(
    const int ex0, const int ex1, const int ex2, 
    const int symmetry, const int lev,
    const double* __restrict__ X, const double* __restrict__ Y, const double* __restrict__ Z,
    // 原始变量
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
    // Kernel 1 中间输出 (Input)
    const double* __restrict__ chix_in, const double* __restrict__ chiy_in, const double* __restrict__ chiz_in,
    const double* __restrict__ gxxx_in, const double* __restrict__ gxxy_in, const double* __restrict__ gxxz_in,
    const double* __restrict__ gxyx_in, const double* __restrict__ gxyy_in, const double* __restrict__ gxyz_in,
    const double* __restrict__ gxzx_in, const double* __restrict__ gxzy_in, const double* __restrict__ gxzz_in,
    const double* __restrict__ gyyx_in, const double* __restrict__ gyyy_in, const double* __restrict__ gyyz_in,
    const double* __restrict__ gyzx_in, const double* __restrict__ gyzy_in, const double* __restrict__ gyzz_in,
    const double* __restrict__ gzzx_in, const double* __restrict__ gzzy_in, const double* __restrict__ gzzz_in,
    const double* __restrict__ gupxx_in, const double* __restrict__ gupxy_in, const double* __restrict__ gupxz_in,
    const double* __restrict__ gupyy_in, const double* __restrict__ gupyz_in, const double* __restrict__ gupzz_in,
    // Kernel 1 计算出的 Gam (Input & Output - 需原地更新为物理 Gam)
    double* __restrict__ Gamxxx, double* __restrict__ Gamxxy, double* __restrict__ Gamxxz,
    double* __restrict__ Gamxyy, double* __restrict__ Gamxyz, double* __restrict__ Gamxzz,
    double* __restrict__ Gamyxx, double* __restrict__ Gamyxy, double* __restrict__ Gamyxz,
    double* __restrict__ Gamyyy, double* __restrict__ Gamyyz, double* __restrict__ Gamyzz,
    double* __restrict__ Gamzxx, double* __restrict__ Gamzxy, double* __restrict__ Gamzxz,
    double* __restrict__ Gamzyy, double* __restrict__ Gamzyz, double* __restrict__ Gamzzz,
    // 输出 RHS (累计结果)
    double* __restrict__ trK_rhs,
    double* __restrict__ Axx_rhs, double* __restrict__ Axy_rhs, double* __restrict__ Axz_rhs,
    double* __restrict__ Ayy_rhs, double* __restrict__ Ayz_rhs, double* __restrict__ Azz_rhs,
    double* __restrict__ Gamx_rhs, double* __restrict__ Gamy_rhs, double* __restrict__ Gamz_rhs,
    double* __restrict__ Lap_rhs, 
    double* __restrict__ betax_rhs, double* __restrict__ betay_rhs, double* __restrict__ betaz_rhs,
    double* __restrict__ dtSfx_rhs, double* __restrict__ dtSfy_rhs, double* __restrict__ dtSfz_rhs,
    // Debug Output (可选)
    double* __restrict__ Rxx_out, double* __restrict__ Rxy_out, double* __restrict__ Rxz_out,
    double* __restrict__ Ryy_out, double* __restrict__ Ryz_out, double* __restrict__ Rzz_out
);

// ==========================================
// Kernel 3: 平流 (Advection) 与 耗散 (Dissipation)
// ==========================================
__global__ void bssn_advection_dissipation_kernel(
    // 维度与参数
    const int ex0, const int ex1, const int ex2,
    const int symmetry, const int lev,
    const double eps,
    // 坐标网格
    const double* __restrict__ X, const double* __restrict__ Y, const double* __restrict__ Z,
    // Shift Vector (用于平流速度场)
    const double* __restrict__ betax, const double* __restrict__ betay, const double* __restrict__ betaz,
    // 演化变量输入 (用于计算微分)
    const double* __restrict__ dxx, const double* __restrict__ gxy, const double* __restrict__ gxz,
    const double* __restrict__ dyy, const double* __restrict__ gyz, const double* __restrict__ dzz,
    const double* __restrict__ Axx, const double* __restrict__ Axy, const double* __restrict__ Axz,
    const double* __restrict__ Ayy, const double* __restrict__ Ayz, const double* __restrict__ Azz,
    const double* __restrict__ chi, const double* __restrict__ trK,
    const double* __restrict__ Gamx, const double* __restrict__ Gamy, const double* __restrict__ Gamz,
    const double* __restrict__ Lap,
    const double* __restrict__ dtSfx, const double* __restrict__ dtSfy, const double* __restrict__ dtSfz,
    // RHS 输出变量 (累加更新: In/Out)
    double* __restrict__ gxx_rhs, double* __restrict__ gxy_rhs, double* __restrict__ gxz_rhs,
    double* __restrict__ gyy_rhs, double* __restrict__ gyz_rhs, double* __restrict__ gzz_rhs,
    double* __restrict__ Axx_rhs, double* __restrict__ Axy_rhs, double* __restrict__ Axz_rhs,
    double* __restrict__ Ayy_rhs, double* __restrict__ Ayz_rhs, double* __restrict__ Azz_rhs,
    double* __restrict__ chi_rhs, double* __restrict__ trK_rhs,
    double* __restrict__ Gamx_rhs, double* __restrict__ Gamy_rhs, double* __restrict__ Gamz_rhs,
    double* __restrict__ Lap_rhs,
    double* __restrict__ betax_rhs, double* __restrict__ betay_rhs, double* __restrict__ betaz_rhs,
    double* __restrict__ dtSfx_rhs, double* __restrict__ dtSfy_rhs, double* __restrict__ dtSfz_rhs
);

// ==========================================
// Kernel 4: 约束求解 (Constraints)
// ==========================================
__global__ void bssn_constraints_kernel(
    // 维度与参数
    const int ex0, const int ex1, const int ex2,
    const int symmetry, const int lev,
    // 坐标
    const double* __restrict__ X, const double* __restrict__ Y, const double* __restrict__ Z,
    // 输入变量
    const double* __restrict__ chi, const double* __restrict__ trK,
    const double* __restrict__ Axx, const double* __restrict__ Axy, const double* __restrict__ Axz,
    const double* __restrict__ Ayy, const double* __restrict__ Ayz, const double* __restrict__ Azz,
    const double* __restrict__ rho, 
    const double* __restrict__ Sx, const double* __restrict__ Sy, const double* __restrict__ Sz,
    // 输入：逆度规 (需从 Kernel 1 输出中读取)
    const double* __restrict__ gupxx_in, const double* __restrict__ gupxy_in, const double* __restrict__ gupxz_in,
    const double* __restrict__ gupyy_in, const double* __restrict__ gupyz_in, const double* __restrict__ gupzz_in,
    // 输入：Ricci Tensor (需从 Kernel 2 输出中读取，或者如果未保存则需重算，这里假设已保存)
    // Fortran 代码中 Rxx 等是在 compute_rhs_bssn 过程中计算并被后续使用的。
    const double* __restrict__ Rxx_in, const double* __restrict__ Rxy_in, const double* __restrict__ Rxz_in,
    const double* __restrict__ Ryy_in, const double* __restrict__ Ryz_in, const double* __restrict__ Rzz_in,
    // 输入：连接系数 (物理 Gam, 需从 Kernel 2 输出或 Kernel 1 输出读取)
    const double* __restrict__ Gamxxx, const double* __restrict__ Gamxxy, const double* __restrict__ Gamxxz,
    const double* __restrict__ Gamxyy, const double* __restrict__ Gamxyz, const double* __restrict__ Gamxzz,
    const double* __restrict__ Gamyxx, const double* __restrict__ Gamyxy, const double* __restrict__ Gamyxz,
    const double* __restrict__ Gamyyy, const double* __restrict__ Gamyyz, const double* __restrict__ Gamyzz,
    const double* __restrict__ Gamzxx, const double* __restrict__ Gamzxy, const double* __restrict__ Gamzxz,
    const double* __restrict__ Gamzyy, const double* __restrict__ Gamzyz, const double* __restrict__ Gamzzz,
    // 输入：Chi 的导数 (从 Kernel 1 读取)
    const double* __restrict__ chix_in, const double* __restrict__ chiy_in, const double* __restrict__ chiz_in,
    // 输出：约束违反值
    double* __restrict__ ham_Res, 
    double* __restrict__ movx_Res, double* __restrict__ movy_Res, double* __restrict__ movz_Res
);

#endif /* BSSN_RHS_GPU_H */