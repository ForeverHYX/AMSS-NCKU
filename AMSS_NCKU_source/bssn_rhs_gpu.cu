#include "bssn_rhs_gpu.h"

#include "fmisc.h"
#include "derivatives.h"
#include "kodiss.h"
#include "lopsidediff.h"

#include <cuda_runtime.h>
#include <math.h>
#include <device_launch_parameters.h>
#include <iostream>

// ==========================================
// 宏定义 (对应 macrodef.fh 和 bssn_rhs.f90)
// ==========================================
// Fortran Column-Major Layout: x varies fastest
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
) {
    // 计算全局索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // 越界检查
    if (i >= ex0 || j >= ex1 || k >= ex2) return;

    int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    int dims[3] = {ex0, ex1, ex2}; // 用于传给 device 函数

    // ==========================================
    // 1. 读取基础变量并进行代数变换
    // ==========================================
    // Fortran: alpn1 = Lap + ONE, chin1 = chi + ONE
    double val_Lap = Lap[idx];
    double val_chi = chi[idx];
    double alpn1 = val_Lap + ONE;
    double chin1 = val_chi + ONE;

    // Metric (dxx 是偏差量，gxx 是物理量 gxx = dxx + 1)
    double val_gxx = dxx[idx] + ONE;
    double val_gxy = gxy[idx];
    double val_gxz = gxz[idx];
    double val_gyy = dyy[idx] + ONE;
    double val_gyz = gyz[idx];
    double val_gzz = dzz[idx] + ONE;

    double val_trK = trK[idx];

    // Extrinsic Curvature Aij
    double val_Axx = Axx[idx]; double val_Axy = Axy[idx]; double val_Axz = Axz[idx];
    double val_Ayy = Ayy[idx]; double val_Ayz = Ayz[idx]; double val_Azz = Azz[idx];

    // ==========================================
    // 2. 计算 Shift Vector (beta) 的导数
    // ==========================================
    double betaxx, betaxy, betaxz;
    double betayx, betayy, betayz;
    double betazx, betazy, betazz;

    // Fortran: call fderivs(ex,betax,betaxx,betaxy,betaxz,X,Y,Z,ANTI, SYM, SYM,Symmetry,Lev)
    d_fderivs_point(dims, betax, &betaxx, &betaxy, &betaxz, X, Y, Z, ANTI, SYM, SYM, symmetry, lev, i, j, k);
    // Fortran: call fderivs(ex,betay,betayx,betayy,betayz,X,Y,Z, SYM,ANTI, SYM,Symmetry,Lev)
    d_fderivs_point(dims, betay, &betayx, &betayy, &betayz, X, Y, Z, SYM, ANTI, SYM, symmetry, lev, i, j, k);
    // Fortran: call fderivs(ex,betaz,betazx,betazy,betazz,X,Y,Z, SYM, SYM,ANTI,Symmetry,Lev)
    d_fderivs_point(dims, betaz, &betazx, &betazy, &betazz, X, Y, Z, SYM, SYM, ANTI, symmetry, lev, i, j, k);

    // Fortran: div_beta = betaxx + betayy + betazz
    double div_beta = betaxx + betayy + betazz;

    // 写回 beta 导数到 Global Memory
    betaxx_out[idx] = betaxx; betaxy_out[idx] = betaxy; betaxz_out[idx] = betaxz;
    betayx_out[idx] = betayx; betayy_out[idx] = betayy; betayz_out[idx] = betayz;
    betazx_out[idx] = betazx; betazy_out[idx] = betazy; betazz_out[idx] = betazz;

    // ==========================================
    // 3. 计算 Chi 的导数与 RHS
    // ==========================================
    double chix, chiy, chiz;
    // Fortran: call fderivs(ex,chi,chix,chiy,chiz,X,Y,Z,SYM,SYM,SYM,symmetry,Lev)
    d_fderivs_point(dims, chi, &chix, &chiy, &chiz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    // Fortran: chi_rhs = F2o3 *chin1*( alpn1 * trK - div_beta )
    chi_rhs[idx] = F2o3 * chin1 * (alpn1 * val_trK - div_beta);
    
    // 写回 chi 导数
    chix_out[idx] = chix; chiy_out[idx] = chiy; chiz_out[idx] = chiz;

    // ==========================================
    // 4. 计算 Metric (gij) 导数
    // ==========================================
    double gxxx, gxxy, gxxz;
    double gxyx, gxyy, gxyz;
    double gxzx, gxzy, gxzz;
    double gyyx, gyyy, gyyz;
    double gyzx, gyzy, gyzz;
    double gzzx, gzzy, gzzz;

    // Fortran: call fderivs(ex,dxx,gxxx,gxxy,gxxz,...)
    d_fderivs_point(dims, dxx, &gxxx, &gxxy, &gxxz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    // Fortran: call fderivs(ex,gxy,gxyx,gxyy,gxyz,...)
    d_fderivs_point(dims, gxy, &gxyx, &gxyy, &gxyz, X, Y, Z, ANTI, ANTI, SYM, symmetry, lev, i, j, k);
    // Fortran: call fderivs(ex,gxz,gxzx,gxzy,gxzz,...)
    d_fderivs_point(dims, gxz, &gxzx, &gxzy, &gxzz, X, Y, Z, ANTI, SYM, ANTI, symmetry, lev, i, j, k);
    // Fortran: call fderivs(ex,dyy,gyyx,gyyy,gyyz,...)
    d_fderivs_point(dims, dyy, &gyyx, &gyyy, &gyyz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    // Fortran: call fderivs(ex,gyz,gyzx,gyzy,gyzz,...)
    d_fderivs_point(dims, gyz, &gyzx, &gyzy, &gyzz, X, Y, Z, SYM, ANTI, ANTI, symmetry, lev, i, j, k);
    // Fortran: call fderivs(ex,dzz,gzzx,gzzy,gzzz,...)
    d_fderivs_point(dims, dzz, &gzzx, &gzzy, &gzzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    // 写回 Metric 导数
    gxxx_out[idx] = gxxx; gxxy_out[idx] = gxxy; gxxz_out[idx] = gxxz;
    gxyx_out[idx] = gxyx; gxyy_out[idx] = gxyy; gxyz_out[idx] = gxyz;
    gxzx_out[idx] = gxzx; gxzy_out[idx] = gxzy; gxzz_out[idx] = gxzz;
    gyyx_out[idx] = gyyx; gyyy_out[idx] = gyyy; gyyz_out[idx] = gyyz;
    gyzx_out[idx] = gyzx; gyzy_out[idx] = gyzy; gyzz_out[idx] = gyzz;
    gzzx_out[idx] = gzzx; gzzy_out[idx] = gzzy; gzzz_out[idx] = gzzz;

    // ==========================================
    // 5. 计算 gij_rhs (部分: 源项 + Lie 导数项)
    // ==========================================
    // Fortran Line 21-28
    gxx_rhs[idx] = -TWO * alpn1 * val_Axx - F2o3 * val_gxx * div_beta + 
                    TWO * (val_gxx * betaxx + val_gxy * betayx + val_gxz * betazx);

    gyy_rhs[idx] = -TWO * alpn1 * val_Ayy - F2o3 * val_gyy * div_beta + 
                    TWO * (val_gxy * betaxy + val_gyy * betayy + val_gyz * betazy);

    gzz_rhs[idx] = -TWO * alpn1 * val_Azz - F2o3 * val_gzz * div_beta + 
                    TWO * (val_gxz * betaxz + val_gyz * betayz + val_gzz * betazz);

    gxy_rhs[idx] = -TWO * alpn1 * val_Axy + F1o3 * val_gxy * div_beta + 
                    val_gxx * betaxy + val_gxz * betazy + 
                    val_gyy * betayx + val_gyz * betazx - val_gxy * betazz;
    
    gyz_rhs[idx] = -TWO * alpn1 * val_Ayz + F1o3 * val_gyz * div_beta +
                    val_gxy * betaxz + val_gyy * betayz +
                    val_gxz * betaxy + val_gzz * betazy - val_gyz * betaxx;

    gxz_rhs[idx] = -TWO * alpn1 * val_Axz + F1o3 * val_gxz * div_beta + 
                    val_gxx * betaxz + val_gxy * betayz + 
                    val_gyz * betayx + val_gzz * betazx - val_gxz * betayy;

    // ==========================================
    // 6. 计算逆度规 (Invert Tilted Metric)
    // ==========================================
    // Fortran Line 29-30
    double gupzz = val_gxx * val_gyy * val_gzz + val_gxy * val_gyz * val_gxz + val_gxz * val_gxy * val_gyz -
                   val_gxz * val_gyy * val_gxz - val_gxy * val_gxy * val_gzz - val_gxx * val_gyz * val_gyz;
    
    double gupxx = (val_gyy * val_gzz - val_gyz * val_gyz) / gupzz;
    double gupxy = -(val_gxy * val_gzz - val_gyz * val_gxz) / gupzz;
    double gupxz = (val_gxy * val_gyz - val_gyy * val_gxz) / gupzz;
    double gupyy = (val_gxx * val_gzz - val_gxz * val_gxz) / gupzz;
    double gupyz = -(val_gxx * val_gyz - val_gxy * val_gxz) / gupzz;
    gupzz = (val_gxx * val_gyy - val_gxy * val_gxy) / gupzz; // 更新 gupzz 为逆分量

    // 写回 gup
    gupxx_out[idx] = gupxx; gupxy_out[idx] = gupxy; gupxz_out[idx] = gupxz;
    gupyy_out[idx] = gupyy; gupyz_out[idx] = gupyz; gupzz_out[idx] = gupzz;

    // ==========================================
    // 7. 计算连接系数残差 (仅 co == 0)
    // ==========================================
    // Fortran Line 31-35
    if (co == 0) {
        // Gmx_Res
        double term_x = gupxx*(gupxx*gxxx+gupxy*gxyx+gupxz*gxzx)
                      + gupxy*(gupxx*gxyx+gupxy*gyyx+gupxz*gyzx)
                      + gupxz*(gupxx*gxzx+gupxy*gyzx+gupxz*gzzx)
                      + gupxx*(gupxy*gxxy+gupyy*gxyy+gupyz*gxzy)
                      + gupxy*(gupxy*gxyy+gupyy*gyyy+gupyz*gyzy)
                      + gupxz*(gupxy*gxzy+gupyy*gyzy+gupyz*gzzy)
                      + gupxx*(gupxz*gxxz+gupyz*gxyz+gupzz*gxzz)
                      + gupxy*(gupxz*gxyz+gupyz*gyyz+gupzz*gyzz)
                      + gupxz*(gupxz*gxzz+gupyz*gyzz+gupzz*gzzz);
        Gmx_Res[idx] = Gamx[idx] - term_x;

        // Gmy_Res
        double term_y = gupxx*(gupxy*gxxx+gupyy*gxyx+gupyz*gxzx)
                      + gupxy*(gupxy*gxyx+gupyy*gyyx+gupyz*gyzx)
                      + gupxz*(gupxy*gxzx+gupyy*gyzx+gupyz*gzzx)
                      + gupxy*(gupxy*gxxy+gupyy*gxyy+gupyz*gxzy)
                      + gupyy*(gupxy*gxyy+gupyy*gyyy+gupyz*gyzy)
                      + gupyz*(gupxy*gxzy+gupyy*gyzy+gupyz*gzzy)
                      + gupxy*(gupxz*gxxz+gupyz*gxyz+gupzz*gxzz)
                      + gupyy*(gupxz*gxyz+gupyz*gyyz+gupzz*gyzz)
                      + gupyz*(gupxz*gxzz+gupyz*gyzz+gupzz*gzzz);
        Gmy_Res[idx] = Gamy[idx] - term_y;

        // Gmz_Res
        double term_z = gupxx*(gupxz*gxxx+gupyz*gxyx+gupzz*gxzx)
                      + gupxy*(gupxz*gxyx+gupyz*gyyx+gupzz*gyzx)
                      + gupxz*(gupxz*gxzx+gupyz*gyzx+gupzz*gzzx)
                      + gupxy*(gupxz*gxxy+gupyz*gxyy+gupzz*gxzy)
                      + gupyy*(gupxz*gxyy+gupyz*gyyy+gupzz*gyzy)
                      + gupyz*(gupxz*gxzy+gupyz*gyzy+gupzz*gzzy)
                      + gupxz*(gupxz*gxxz+gupyz*gxyz+gupzz*gxzz)
                      + gupyz*(gupxz*gxyz+gupyz*gyyz+gupzz*gyzz)
                      + gupzz*(gupxz*gxzz+gupyz*gyzz+gupzz*gzzz);
        Gmz_Res[idx] = Gamz[idx] - term_z;
    }

    // ==========================================
    // 8. 计算第二类 Christoffel 符号 (Gam^k_ij)
    // ==========================================
    // Fortran Line 36
    Gamxxx[idx] = HALF * (gupxx * gxxx + gupxy * (TWO * gxyx - gxxy) + gupxz * (TWO * gxzx - gxxz));
    Gamyxx[idx] = HALF * (gupxy * gxxx + gupyy * (TWO * gxyx - gxxy) + gupyz * (TWO * gxzx - gxxz));
    Gamzxx[idx] = HALF * (gupxz * gxxx + gupyz * (TWO * gxyx - gxxy) + gupzz * (TWO * gxzx - gxxz));

    Gamxyy[idx] = HALF * (gupxx * (TWO * gxyy - gyyx) + gupxy * gyyy + gupxz * (TWO * gyzy - gyyz));
    Gamyyy[idx] = HALF * (gupxy * (TWO * gxyy - gyyx) + gupyy * gyyy + gupyz * (TWO * gyzy - gyyz));
    Gamzyy[idx] = HALF * (gupxz * (TWO * gxyy - gyyx) + gupyz * gyyy + gupzz * (TWO * gyzy - gyyz));

    // Fortran Line 37
    Gamxzz[idx] = HALF * (gupxx * (TWO * gxzz - gzzx) + gupxy * (TWO * gyzz - gzzy) + gupxz * gzzz);
    Gamyzz[idx] = HALF * (gupxy * (TWO * gxzz - gzzx) + gupyy * (TWO * gyzz - gzzy) + gupyz * gzzz);
    Gamzzz[idx] = HALF * (gupxz * (TWO * gxzz - gzzx) + gupyz * (TWO * gyzz - gzzy) + gupzz * gzzz);

    Gamxxy[idx] = HALF * (gupxx * gxxy + gupxy * gyyx + gupxz * (gxzy + gyzx - gxyz));
    Gamyxy[idx] = HALF * (gupxy * gxxy + gupyy * gyyx + gupyz * (gxzy + gyzx - gxyz));
    Gamzxy[idx] = HALF * (gupxz * gxxy + gupyz * gyyx + gupzz * (gxzy + gyzx - gxyz));

    // Fortran Line 38
    Gamxxz[idx] = HALF * (gupxx * gxxz + gupxy * (gxyz + gyzx - gxzy) + gupxz * gzzx);
    Gamyxz[idx] = HALF * (gupxy * gxxz + gupyy * (gxyz + gyzx - gxzy) + gupyz * gzzx);
    Gamzxz[idx] = HALF * (gupxz * gxxz + gupyz * (gxyz + gyzx - gxzy) + gupzz * gzzx);

    Gamxyz[idx] = HALF * (gupxx * (gxyz + gxzy - gyzx) + gupxy * gyyz + gupxz * gzzy);
    Gamyyz[idx] = HALF * (gupxy * (gxyz + gxzy - gyzx) + gupyy * gyyz + gupyz * gzzy);
    Gamzyz[idx] = HALF * (gupxz * (gxyz + gxzy - gyzx) + gupyz * gyyz + gupzz * gzzy);
}



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
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= ex0 || j >= ex1 || k >= ex2) return;

    int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    int dims[3] = {ex0, ex1, ex2};

    // ==========================================
    // 0. 加载数据至寄存器 (Locals)
    // ==========================================
    double l_gxx = dxx[idx] + ONE; double l_gxy = gxy[idx]; double l_gxz = gxz[idx];
    double l_gyy = dyy[idx] + ONE; double l_gyz = gyz[idx]; double l_gzz = dzz[idx] + ONE;
    
    double gupxx = gupxx_in[idx]; double gupxy = gupxy_in[idx]; double gupxz = gupxz_in[idx];
    double gupyy = gupyy_in[idx]; double gupyz = gupyz_in[idx]; double gupzz = gupzz_in[idx];

    double l_Gamxxx = Gamxxx[idx]; double l_Gamxxy = Gamxxy[idx]; double l_Gamxxz = Gamxxz[idx];
    double l_Gamxyy = Gamxyy[idx]; double l_Gamxyz = Gamxyz[idx]; double l_Gamxzz = Gamxzz[idx];
    double l_Gamyxx = Gamyxx[idx]; double l_Gamyxy = Gamyxy[idx]; double l_Gamyxz = Gamyxz[idx];
    double l_Gamyyy = Gamyyy[idx]; double l_Gamyyz = Gamyyz[idx]; double l_Gamyzz = Gamyzz[idx];
    double l_Gamzxx = Gamzxx[idx]; double l_Gamzxy = Gamzxy[idx]; double l_Gamzxz = Gamzxz[idx];
    double l_Gamzyy = Gamzyy[idx]; double l_Gamzyz = Gamzyz[idx]; double l_Gamzzz = Gamzzz[idx];

    double l_gxxx = gxxx_in[idx]; double l_gxxy = gxxy_in[idx]; double l_gxxz = gxxz_in[idx];
    double l_gxyx = gxyx_in[idx]; double l_gxyy = gxyy_in[idx]; double l_gxzy = gxzy_in[idx]; // 注意: Fortran代码中命名不一致，这里对应 gxy_z
    double l_gxzx = gxzx_in[idx]; double l_gxzz = gxzz_in[idx]; 

    double l_gxyz = gxyz_in[idx]; double l_gyyx = gyyx_in[idx]; double l_gyyy = gyyy_in[idx];
    double l_gyyz = gyyz_in[idx]; double l_gyzx = gyzx_in[idx]; double l_gyzy = gyzy_in[idx];
    double l_gyzz = gyzz_in[idx]; double l_gzzx = gzzx_in[idx]; double l_gzzy = gzzy_in[idx];
    double l_gzzz = gzzz_in[idx];

    double l_Axx = Axx[idx]; double l_Axy = Axy[idx]; double l_Axz = Axz[idx];
    double l_Ayy = Ayy[idx]; double l_Ayz = Ayz[idx]; double l_Azz = Azz[idx];
    
    double val_Lap = Lap[idx];
    double val_chi = chi[idx];
    double val_trK = trK[idx];
    double alpn1 = val_Lap + ONE;
    double chin1 = val_chi + ONE;

    // ==========================================
    // Step 1: 初始化 Ricci Tensor (Aij 贡献)
    // ==========================================
    double Rxx, Rxy, Rxz, Ryy, Ryz, Rzz;

    Rxx = gupxx * gupxx * l_Axx + gupxy * gupxy * l_Ayy + gupxz * gupxz * l_Azz + 
          TWO*(gupxx * gupxy * l_Axy + gupxx * gupxz * l_Axz + gupxy * gupxz * l_Ayz);

    Ryy = gupxy * gupxy * l_Axx + gupyy * gupyy * l_Ayy + gupyz * gupyz * l_Azz + 
          TWO*(gupxy * gupyy * l_Axy + gupxy * gupyz * l_Axz + gupyy * gupyz * l_Ayz);

    Rzz = gupxz * gupxz * l_Axx + gupyz * gupyz * l_Ayy + gupzz * gupzz * l_Azz + 
          TWO*(gupxz * gupyz * l_Axy + gupxz * gupzz * l_Axz + gupyz * gupzz * l_Ayz);

    Rxy = gupxx * gupxy * l_Axx + gupxy * gupyy * l_Ayy + gupxz * gupyz * l_Azz + 
          (gupxx * gupyy + gupxy * gupxy)* l_Axy + 
          (gupxx * gupyz + gupxz * gupxy)* l_Axz + 
          (gupxy * gupyz + gupxz * gupyy)* l_Ayz;

    Rxz = gupxx * gupxz * l_Axx + gupxy * gupyz * l_Ayy + gupxz * gupzz * l_Azz + 
          (gupxx * gupyz + gupxy * gupxz)* l_Axy + 
          (gupxx * gupzz + gupxz * gupxz)* l_Axz + 
          (gupxy * gupzz + gupxz * gupyz)* l_Ayz;

    Ryz = gupxy * gupxz * l_Axx + gupyy * gupyz * l_Ayy + gupyz * gupzz * l_Azz + 
          (gupxy * gupyz + gupyy * gupxz)* l_Axy + 
          (gupxy * gupzz + gupyz * gupxz)* l_Axz + 
          (gupyy * gupzz + gupyz * gupyz)* l_Ayz;

    // ==========================================
    // Step 2: 计算 Gam^i_rhs (Part 1: No shift)
    // ==========================================
    double Lapx, Lapy, Lapz, Kx, Ky, Kz;
    d_fderivs_point(dims, Lap, &Lapx, &Lapy, &Lapz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, trK, &Kx, &Ky, &Kz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    double val_chix = chix_in[idx]; double val_chiy = chiy_in[idx]; double val_chiz = chiz_in[idx];
    double val_Sx = Sx[idx]; double val_Sy = Sy[idx]; double val_Sz = Sz[idx];

    // Gamx_rhs
    double val_Gamx_rhs = - TWO * (Lapx * Rxx + Lapy * Rxy + Lapz * Rxz) + 
        TWO * alpn1 * (
        -F3o2/chin1 * (val_chix * Rxx + val_chiy * Rxy + val_chiz * Rxz) - 
        gupxx * (F2o3 * Kx + EIGHT * PI * val_Sx) - 
        gupxy * (F2o3 * Ky + EIGHT * PI * val_Sy) - 
        gupxz * (F2o3 * Kz + EIGHT * PI * val_Sz) + 
        l_Gamxxx * Rxx + l_Gamxyy * Ryy + l_Gamxzz * Rzz + 
        TWO * (l_Gamxxy * Rxy + l_Gamxxz * Rxz + l_Gamxyz * Ryz));

    // Gamy_rhs
    double val_Gamy_rhs = - TWO * (Lapx * Rxy + Lapy * Ryy + Lapz * Ryz) + 
        TWO * alpn1 * (
        -F3o2/chin1 * (val_chix * Rxy + val_chiy * Ryy + val_chiz * Ryz) - 
        gupxy * (F2o3 * Kx + EIGHT * PI * val_Sx) - 
        gupyy * (F2o3 * Ky + EIGHT * PI * val_Sy) - 
        gupyz * (F2o3 * Kz + EIGHT * PI * val_Sz) + 
        l_Gamyxx * Rxx + l_Gamyyy * Ryy + l_Gamyzz * Rzz + 
        TWO * (l_Gamyxy * Rxy + l_Gamyxz * Rxz + l_Gamyyz * Ryz));

    // Gamz_rhs
    double val_Gamz_rhs = - TWO * (Lapx * Rxz + Lapy * Ryz + Lapz * Rzz) + 
        TWO * alpn1 * (
        -F3o2/chin1 * (val_chix * Rxz + val_chiy * Ryz + val_chiz * Rzz) - 
        gupxz * (F2o3 * Kx + EIGHT * PI * val_Sx) - 
        gupyz * (F2o3 * Ky + EIGHT * PI * val_Sy) - 
        gupzz * (F2o3 * Kz + EIGHT * PI * val_Sz) + 
        l_Gamzxx * Rxx + l_Gamzyy * Ryy + l_Gamzzz * Rzz + 
        TWO * (l_Gamzxy * Rxy + l_Gamzxz * Rxz + l_Gamzyz * Ryz));

    
    // betax 二阶导
    double bx_gxxx, bx_gxyx, bx_gxzx, bx_gyyx, bx_gyzx, bx_gzzx;
    d_fdderivs_point(dims, betax, &bx_gxxx, &bx_gxyx, &bx_gxzx, &bx_gyyx, &bx_gyzx, &bx_gzzx,
                     X, Y, Z, ANTI, SYM, SYM, symmetry, lev, i, j, k);

    // betay 二阶导
    double by_gxxy, by_gxyy, by_gxzy, by_gyyy, by_gyzy, by_gzzy;
    d_fdderivs_point(dims, betay, &by_gxxy, &by_gxyy, &by_gxzy, &by_gyyy, &by_gyzy, &by_gzzy,
                     X, Y, Z, SYM, ANTI, SYM, symmetry, lev, i, j, k);

    // betaz 二阶导
    double bz_gxxz, bz_gxyz, bz_gxzz, bz_gyyz, bz_gyzz, bz_gzzz;
    d_fdderivs_point(dims, betaz, &bz_gxxz, &bz_gxyz, &bz_gxzz, &bz_gyyz, &bz_gyzz, &bz_gzzz,
                     X, Y, Z, SYM, SYM, ANTI, symmetry, lev, i, j, k);

    double fxx, fxy, fxz, fyy, fyz, fzz;

    // fxx/fxy/fxz = Laplacian(beta^i) 的组合 (Fortran line 156)
    fxx = bx_gxxx + by_gxyy + bz_gxzz;
    fxy = bx_gxyx + by_gyyy + bz_gyzz;
    fxz = bx_gxzx + by_gyzy + bz_gzzz;

    double Gamxa = gupxx * l_Gamxxx + gupyy * l_Gamxyy + gupzz * l_Gamxzz +
                   TWO * (gupxy * l_Gamxxy + gupxz * l_Gamxxz + gupyz * l_Gamxyz);
    double Gamya = gupxx * l_Gamyxx + gupyy * l_Gamyyy + gupzz * l_Gamyzz +
                   TWO * (gupxy * l_Gamyxy + gupxz * l_Gamyxz + gupyz * l_Gamyyz);
    double Gamza = gupxx * l_Gamzxx + gupyy * l_Gamzyy + gupzz * l_Gamzzz +
                   TWO * (gupxy * l_Gamzxy + gupxz * l_Gamzxz + gupyz * l_Gamzyz);
    
    double dGamxx, dGamxy, dGamxz;
    double dGamyx, dGamyy, dGamyz;
    double dGamzx, dGamzy, dGamzz;
    d_fderivs_point(dims, Gamx, &dGamxx, &dGamxy, &dGamxz, X, Y, Z, ANTI, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Gamy, &dGamyx, &dGamyy, &dGamyz, X, Y, Z, SYM, ANTI, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Gamz, &dGamzx, &dGamzy, &dGamzz, X, Y, Z, SYM, SYM, ANTI, symmetry, lev, i, j, k);

    double betaxx, betaxy, betaxz, betayx, betayy, betayz, betazx, betazy, betazz;
    d_fderivs_point(dims, betax, &betaxx, &betaxy, &betaxz, X, Y, Z, ANTI, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, betay, &betayx, &betayy, &betayz, X, Y, Z, SYM, ANTI, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, betaz, &betazx, &betazy, &betazz, X, Y, Z, SYM, SYM, ANTI, symmetry, lev, i, j, k);
    double div_beta = betaxx + betayy + betazz;

    // Gamx_rhs (Fortran line 170)
    val_Gamx_rhs += F2o3 * Gamxa * div_beta
                 - (Gamxa * betaxx + Gamya * betaxy + Gamza * betaxz)
                 + F1o3 * (gupxx * fxx + gupxy * fxy + gupxz * fxz)
                 + gupxx * bx_gxxx + gupyy * bx_gyyx + gupzz * bx_gzzx
                 + TWO * (gupxy * bx_gxyx + gupxz * bx_gxzx + gupyz * bx_gyzx);

    // Gamy_rhs (Fortran line 174)
    val_Gamy_rhs += F2o3 * Gamya * div_beta
                 - (Gamxa * betayx + Gamya * betayy + Gamza * betayz)
                 + F1o3 * (gupxy * fxx + gupyy * fxy + gupyz * fxz)
                 + gupxx * by_gxxy + gupyy * by_gyyy + gupzz * by_gzzy
                 + TWO * (gupxy * by_gxyy + gupxz * by_gxzy + gupyz * by_gyzy);

    // Gamz_rhs (Fortran line 178)
    val_Gamz_rhs += F2o3 * Gamza * div_beta
                 - (Gamxa * betazx + Gamya * betazy + Gamza * betazz)
                 + F1o3 * (gupxz * fxx + gupyz * fxy + gupzz * fxz)
                 + gupxx * bz_gxxz + gupyy * bz_gyyz + gupzz * bz_gzzz
                 + TWO * (gupxy * bz_gxyz + gupxz * bz_gxzz + gupyz * bz_gyzz);
    
    // ==========================================
    // Step 3: Ricci (Metric 二阶导数部分)
    // ==========================================
    l_gxxx = l_gxx * l_Gamxxx + l_gxy * l_Gamyxx + l_gxz * l_Gamzxx;
    l_gxyx = l_gxx * l_Gamxxy + l_gxy * l_Gamyxy + l_gxz * l_Gamzxy;
    l_gxzx = l_gxx * l_Gamxxz + l_gxy * l_Gamyxz + l_gxz * l_Gamzxz;
    l_gyyx = l_gxx * l_Gamxyy + l_gxy * l_Gamyyy + l_gxz * l_Gamzyy;
    l_gyzx = l_gxx * l_Gamxyz + l_gxy * l_Gamyyz + l_gxz * l_Gamzyz;
    l_gzzx = l_gxx * l_Gamxzz + l_gxy * l_Gamyzz + l_gxz * l_Gamzzz;

    l_gxxy = l_gxy * l_Gamxxx + l_gyy * l_Gamyxx + l_gyz * l_Gamzxx;
    l_gxyy = l_gxy * l_Gamxxy + l_gyy * l_Gamyxy + l_gyz * l_Gamzxy;
    l_gxzy = l_gxy * l_Gamxxz + l_gyy * l_Gamyxz + l_gyz * l_Gamzxz;
    l_gyyy = l_gxy * l_Gamxyy + l_gyy * l_Gamyyy + l_gyz * l_Gamzyy;
    l_gyzy = l_gxy * l_Gamxyz + l_gyy * l_Gamyyz + l_gyz * l_Gamzyz;
    l_gzzy = l_gxy * l_Gamxzz + l_gyy * l_Gamyzz + l_gyz * l_Gamzzz;

    l_gxxz = l_gxz * l_Gamxxx + l_gyz * l_Gamyxx + l_gzz * l_Gamzxx;
    l_gxyz = l_gxz * l_Gamxxy + l_gyz * l_Gamyxy + l_gzz * l_Gamzxy;
    l_gxzz = l_gxz * l_Gamxxz + l_gyz * l_Gamyxz + l_gzz * l_Gamzxz;
    l_gyyz = l_gxz * l_Gamxyy + l_gyz * l_Gamyyy + l_gzz * l_Gamzyy;
    l_gyzz = l_gxz * l_Gamxyz + l_gyz * l_Gamyyz + l_gzz * l_Gamzyz;
    l_gzzz = l_gxz * l_Gamxzz + l_gyz * l_Gamyzz + l_gzz * l_Gamzzz;
    
    // Rxx
    d_fdderivs_point(dims, dxx, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    Rxx = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;
    // Ryy
    d_fdderivs_point(dims, dyy, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    Ryy = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;

    // Rzz
    d_fdderivs_point(dims, dzz, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    Rzz = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;

    // Rxy
    d_fdderivs_point(dims, gxy, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, ANTI, ANTI, SYM, symmetry, lev, i, j, k);
    Rxy = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;

    // Rxz
    d_fdderivs_point(dims, gxz, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, ANTI, SYM, ANTI, symmetry, lev, i, j, k);
    Rxz = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;

    // Ryz
    d_fdderivs_point(dims, gyz, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, ANTI, ANTI, symmetry, lev, i, j, k);
    Ryz = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;

    // ==========================================
    // Step 4: Ricci (连接系数项) - 完整展开
    // ==========================================

    // double Gam_dot_dg_xx = Gamxa * l_gxxx + Gamya * l_gxyx + Gamza * l_gxzx;
    // double Gam_dot_dg_yy = Gamxa * l_gxyy + Gamya * l_gyyy + Gamza * l_gyzy;
    // double Gam_dot_dg_zz = Gamxa * l_gxzz + Gamya * l_gyzz + Gamza * l_gzzz;

    // Rxx Correction
    Rxx = -HALF * Rxx + 
          l_gxx * dGamxx + l_gxy * dGamyx + l_gxz * dGamzx + 
          Gamxa * l_gxxx + Gamya * l_gxyx + Gamza * l_gxzx + 
          gupxx * (TWO*(l_Gamxxx*l_gxxx + l_Gamyxx*l_gxyx + l_Gamzxx*l_gxzx) + l_Gamxxx*l_gxxx + l_Gamyxx*l_gxxy + l_Gamzxx*l_gxxz) +
          gupxy * (TWO*(l_Gamxxx*l_gxyx + l_Gamyxx*l_gyyx + l_Gamzxx*l_gyzx + l_Gamxxy*l_gxxx + l_Gamyxy*l_gxyx + l_Gamzxy*l_gxzx) + l_Gamxxy*l_gxxx + l_Gamyxy*l_gxxy + l_Gamzxy*l_gxxz + l_Gamxxx*l_gxyx + l_Gamyxx*l_gxyy + l_Gamzxx*l_gxyz) + 
          gupxz * (TWO*(l_Gamxxx*l_gxzx + l_Gamyxx*l_gyzx + l_Gamzxx*l_gzzx + l_Gamxxz*l_gxxx + l_Gamyxz*l_gxyx + l_Gamzxz*l_gxzx) + l_Gamxxz*l_gxxx + l_Gamyxz*l_gxxy + l_Gamzxz*l_gxxz + l_Gamxxx*l_gxzx + l_Gamyxx*l_gxzy + l_Gamzxx*l_gxzz) + 
          gupyy * (TWO*(l_Gamxxy*l_gxyx + l_Gamyxy*l_gyyx + l_Gamzxy*l_gyzx) + l_Gamxxy*l_gxyx + l_Gamyxy*l_gxyy + l_Gamzxy*l_gxyz) + 
          gupyz * (TWO*(l_Gamxxy*l_gxzx + l_Gamyxy*l_gyzx + l_Gamzxy*l_gzzx + l_Gamxxz*l_gxyx + l_Gamyxz*l_gyyx + l_Gamzxz*l_gyzx) + l_Gamxxz*l_gxyx + l_Gamyxz*l_gxyy + l_Gamzxz*l_gxyz + l_Gamxxy*l_gxzx + l_Gamyxy*l_gxzy + l_Gamzxy*l_gxzz) + 
          gupzz * (TWO*(l_Gamxxz*l_gxzx + l_Gamyxz*l_gyzx + l_Gamzxz*l_gzzx) + l_Gamxxz*l_gxzx + l_Gamyxz*l_gxzy + l_Gamzxz*l_gxzz);

    // Ryy Correction
    Ryy = -HALF * Ryy + 
          l_gxy * dGamxy + l_gyy * dGamyy + l_gyz * dGamzy + 
          Gamxa * l_gxyy + Gamya * l_gyyy + Gamza * l_gyzy + 
          gupxx * (TWO*(l_Gamxxy*l_gxxy + l_Gamyxy*l_gxyy + l_Gamzxy*l_gxzy) + l_Gamxxy*l_gxyx + l_Gamyxy*l_gxyy + l_Gamzxy*l_gxyz) + 
          gupxy * (TWO*(l_Gamxxy*l_gxyy + l_Gamyxy*l_gyyy + l_Gamzxy*l_gyzy + l_Gamxyy*l_gxxy + l_Gamyyy*l_gxyy + l_Gamzyy*l_gxzy) + l_Gamxyy*l_gxyx + l_Gamyyy*l_gxyy + l_Gamzyy*l_gxyz + l_Gamxxy*l_gyyx + l_Gamyxy*l_gyyy + l_Gamzxy*l_gyyz) + 
          gupxz * (TWO*(l_Gamxxy*l_gxzy + l_Gamyxy*l_gyzy + l_Gamzxy*l_gzzy + l_Gamxyz*l_gxxy + l_Gamyyz*l_gxyy + l_Gamzyz*l_gxzy) + l_Gamxyz*l_gxyx + l_Gamyyz*l_gxyy + l_Gamzyz*l_gxyz + l_Gamxxy*l_gyzx + l_Gamyxy*l_gyzy + l_Gamzxy*l_gyzz) + 
          gupyy * (TWO*(l_Gamxyy*l_gxyy + l_Gamyyy*l_gyyy + l_Gamzyy*l_gyzy) + l_Gamxyy*l_gyyx + l_Gamyyy*l_gyyy + l_Gamzyy*l_gyyz) + 
          gupyz * (TWO*(l_Gamxyy*l_gxzy + l_Gamyyy*l_gyzy + l_Gamzyy*l_gzzy + l_Gamxyz*l_gxyy + l_Gamyyz*l_gyyy + l_Gamzyz*l_gyzy) + l_Gamxyz*l_gyyx + l_Gamyyz*l_gyyy + l_Gamzyz*l_gyyz + l_Gamxyy*l_gyzx + l_Gamyyy*l_gyzy + l_Gamzyy*l_gyzz) + 
          gupzz * (TWO*(l_Gamxyz*l_gxzy + l_Gamyyz*l_gyzy + l_Gamzyz*l_gzzy) + l_Gamxyz*l_gyzx + l_Gamyyz*l_gyzy + l_Gamzyz*l_gyzz);

    // Rzz Correction
    Rzz = -HALF * Rzz + 
          l_gxz * dGamxz + l_gyz * dGamyz + l_gzz * dGamzz + 
          Gamxa * l_gxzz + Gamya * l_gyzz + Gamza * l_gzzz + 
          gupxx * (TWO*(l_Gamxxz*l_gxxz + l_Gamyxz*l_gxyz + l_Gamzxz*l_gxzz) + l_Gamxxz*l_gxzx + l_Gamyxz*l_gxzy + l_Gamzxz*l_gxzz) + 
          gupxy * (TWO*(l_Gamxxz*l_gxyz + l_Gamyxz*l_gyyz + l_Gamzxz*l_gyzz + l_Gamxyz*l_gxxz + l_Gamyyz*l_gxyz + l_Gamzyz*l_gxzz) + l_Gamxyz*l_gxzx + l_Gamyyz*l_gxzy + l_Gamzyz*l_gxzz + l_Gamxxz*l_gyzx + l_Gamyxz*l_gyzy + l_Gamzxz*l_gyzz) + 
          gupxz * (TWO*(l_Gamxxz*l_gxzz + l_Gamyxz*l_gyzz + l_Gamzxz*l_gzzz + l_Gamxzz*l_gxxz + l_Gamyzz*l_gxyz + l_Gamzzz*l_gxzz) + l_Gamxzz*l_gxzx + l_Gamyzz*l_gxzy + l_Gamzzz*l_gxzz + l_Gamxxz*l_gzzx + l_Gamyxz*l_gzzy + l_Gamzxz*l_gzzz) + 
          gupyy * (TWO*(l_Gamxyz*l_gxyz + l_Gamyyz*l_gyyz + l_Gamzyz*l_gyzz) + l_Gamxyz*l_gyzx + l_Gamyyz*l_gyzy + l_Gamzyz*l_gyzz) + 
          gupyz * (TWO*(l_Gamxyz*l_gxzz + l_Gamyyz*l_gyzz + l_Gamzyz*l_gzzz + l_Gamxzz*l_gxyz + l_Gamyzz*l_gyyz + l_Gamzzz*l_gyzz) + l_Gamxzz*l_gyzx + l_Gamyzz*l_gyzy + l_Gamzzz*l_gyzz + l_Gamxyz*l_gzzx + l_Gamyyz*l_gzzy + l_Gamzyz*l_gzzz) + 
          gupzz * (TWO*(l_Gamxzz*l_gxzz + l_Gamyzz*l_gyzz + l_Gamzzz*l_gzzz) + l_Gamxzz*l_gzzx + l_Gamyzz*l_gzzy + l_Gamzzz*l_gzzz);

    // Rxy Correction
    Rxy = HALF * ( - Rxy + 
          l_gxx * dGamxy + l_gxy * dGamyy + l_gxz * dGamzy + 
          l_gxy * dGamxx + l_gyy * dGamyx + l_gyz * dGamzx + 
          Gamxa * l_gxyx + Gamya * l_gyyx + Gamza * l_gyzx + 
          Gamxa * l_gxxy + Gamya * l_gxyy + Gamza * l_gxzy) + 
          gupxx * (l_Gamxxx*l_gxxy + l_Gamyxx*l_gxyy + l_Gamzxx*l_gxzy + l_Gamxxy*l_gxxx + l_Gamyxy*l_gxyx + l_Gamzxy*l_gxzx + l_Gamxxx*l_gxyx + l_Gamyxx*l_gxyy + l_Gamzxx*l_gxyz) + 
          gupxy * (l_Gamxxx*l_gxyy + l_Gamyxx*l_gyyy + l_Gamzxx*l_gyzy + l_Gamxxy*l_gxyx + l_Gamyxy*l_gyyx + l_Gamzxy*l_gyzx + l_Gamxxy*l_gxyx + l_Gamyxy*l_gxyy + l_Gamzxy*l_gxyz + l_Gamxxy*l_gxxy + l_Gamyxy*l_gxyy + l_Gamzxy*l_gxzy + l_Gamxyy*l_gxxx + l_Gamyyy*l_gxyx + l_Gamzyy*l_gxzx + l_Gamxxx*l_gyyx + l_Gamyxx*l_gyyy + l_Gamzxx*l_gyyz) + 
          gupxz * (l_Gamxxx*l_gxzy + l_Gamyxx*l_gyzy + l_Gamzxx*l_gzzy + l_Gamxxy*l_gxzx + l_Gamyxy*l_gyzx + l_Gamzxy*l_gzzx + l_Gamxxz*l_gxyx + l_Gamyxz*l_gxyy + l_Gamzxz*l_gxyz + l_Gamxxz*l_gxxy + l_Gamyxz*l_gxyy + l_Gamzxz*l_gxzy + l_Gamxyz*l_gxxx + l_Gamyyz*l_gxyx + l_Gamzyz*l_gxzx + l_Gamxxx*l_gyzx + l_Gamyxx*l_gyzy + l_Gamzxx*l_gyzz) + 
          gupyy * (l_Gamxxy*l_gxyy + l_Gamyxy*l_gyyy + l_Gamzxy*l_gyzy + l_Gamxyy*l_gxyx + l_Gamyyy*l_gyyx + l_Gamzyy*l_gyzx + l_Gamxxy*l_gyyx + l_Gamyxy*l_gyyy + l_Gamzxy*l_gyyz) + 
          gupyz * (l_Gamxxy*l_gxzy + l_Gamyxy*l_gyzy + l_Gamzxy*l_gzzy + l_Gamxyy*l_gxzx + l_Gamyyy*l_gyzx + l_Gamzyy*l_gzzx + l_Gamxxz*l_gyyx + l_Gamyxz*l_gyyy + l_Gamzxz*l_gyyz + l_Gamxxz*l_gxyy + l_Gamyxz*l_gyyy + l_Gamzxz*l_gyzy + l_Gamxyz*l_gxyx + l_Gamyyz*l_gyyx + l_Gamzyz*l_gyzx + l_Gamxxy*l_gyzx + l_Gamyxy*l_gyzy + l_Gamzxy*l_gyzz) + 
          gupzz * (l_Gamxxz*l_gxzy + l_Gamyxz*l_gyzy + l_Gamzxz*l_gzzy + l_Gamxyz*l_gxzx + l_Gamyyz*l_gyzx + l_Gamzyz*l_gzzx + l_Gamxxz*l_gyzx + l_Gamyxz*l_gyzy + l_Gamzxz*l_gyzz);

    // Rxz Correction
    Rxz = HALF * ( - Rxz + 
          l_gxx * dGamxz + l_gxy * dGamyz + l_gxz * dGamzz + 
          l_gxz * dGamxx + l_gyz * dGamyx + l_gzz * dGamzx + 
          Gamxa * l_gxzx + Gamya * l_gyzx + Gamza * l_gzzx + 
          Gamxa * l_gxxz + Gamya * l_gxyz + Gamza * l_gxzz) + 
          gupxx * (l_Gamxxx*l_gxxz + l_Gamyxx*l_gxyz + l_Gamzxx*l_gxzz + l_Gamxxz*l_gxxx + l_Gamyxz*l_gxyx + l_Gamzxz*l_gxzx + l_Gamxxx*l_gxzx + l_Gamyxx*l_gxzy + l_Gamzxx*l_gxzz) + 
          gupxy * (l_Gamxxx*l_gxyz + l_Gamyxx*l_gyyz + l_Gamzxx*l_gyzz + l_Gamxxz*l_gxyx + l_Gamyxz*l_gyyx + l_Gamzxz*l_gyzx + l_Gamxxy*l_gxzx + l_Gamyxy*l_gxzy + l_Gamzxy*l_gxzz + l_Gamxxy*l_gxxz + l_Gamyxy*l_gxyz + l_Gamzxy*l_gxzz + l_Gamxyz*l_gxxx + l_Gamyyz*l_gxyx + l_Gamzyz*l_gxzx + l_Gamxxx*l_gyzx + l_Gamyxx*l_gyzy + l_Gamzxx*l_gyzz) + 
          gupxz * (l_Gamxxx*l_gxzz + l_Gamyxx*l_gyzz + l_Gamzxx*l_gzzz + l_Gamxxz*l_gxzx + l_Gamyxz*l_gyzx + l_Gamzxz*l_gzzx + l_Gamxxz*l_gxzx + l_Gamyxz*l_gxzy + l_Gamzxz*l_gxzz + l_Gamxxz*l_gxxz + l_Gamyxz*l_gxyz + l_Gamzxz*l_gxzz + l_Gamxzz*l_gxxx + l_Gamyzz*l_gxyx + l_Gamzzz*l_gxzx + l_Gamxxx*l_gzzx + l_Gamyxx*l_gzzy + l_Gamzxx*l_gzzz) + 
          gupyy * (l_Gamxxy*l_gxyz + l_Gamyxy*l_gyyz + l_Gamzxy*l_gyzz + l_Gamxyz*l_gxyx + l_Gamyyz*l_gyyx + l_Gamzyz*l_gyzx + l_Gamxxy*l_gyzx + l_Gamyxy*l_gyzy + l_Gamzxy*l_gyzz) + 
          gupyz * (l_Gamxxy*l_gxzz + l_Gamyxy*l_gyzz + l_Gamzxy*l_gzzz + l_Gamxyz*l_gxzx + l_Gamyyz*l_gyzx + l_Gamzyz*l_gzzx + l_Gamxxz*l_gyzx + l_Gamyxz*l_gyzy + l_Gamzxz*l_gyzz + l_Gamxxz*l_gxyz + l_Gamyxz*l_gyyz + l_Gamzxz*l_gyzz + l_Gamxzz*l_gxyx + l_Gamyzz*l_gyyx + l_Gamzzz*l_gyzx + l_Gamxxy*l_gzzx + l_Gamyxy*l_gzzy + l_Gamzxy*l_gzzz) + 
          gupzz * (l_Gamxxz*l_gxzz + l_Gamyxz*l_gyzz + l_Gamzxz*l_gzzz + l_Gamxzz*l_gxzx + l_Gamyzz*l_gyzx + l_Gamzzz*l_gzzx + l_Gamxxz*l_gzzx + l_Gamyxz*l_gzzy + l_Gamzxz*l_gzzz);

    // Ryz Correction
    Ryz = HALF * ( - Ryz + 
          l_gxy * dGamxz + l_gyy * dGamyz + l_gyz * dGamzz + 
          l_gxz * dGamxy + l_gyz * dGamyy + l_gzz * dGamzy + 
          Gamxa * l_gxzy + Gamya * l_gyzy + Gamza * l_gzzy + 
          Gamxa * l_gxyz + Gamya * l_gyyz + Gamza * l_gyzz) + 
          gupxx * (l_Gamxxy*l_gxxz + l_Gamyxy*l_gxyz + l_Gamzxy*l_gxzz + l_Gamxxz*l_gxxy + l_Gamyxz*l_gxyy + l_Gamzxz*l_gxzy + l_Gamxxy*l_gxzx + l_Gamyxy*l_gxzy + l_Gamzxy*l_gxzz) + 
          gupxy * (l_Gamxxy*l_gxyz + l_Gamyxy*l_gyyz + l_Gamzxy*l_gyzz + l_Gamxxz*l_gxyy + l_Gamyxz*l_gyyy + l_Gamzxz*l_gyzy + l_Gamxyy*l_gxzx + l_Gamyyy*l_gxzy + l_Gamzyy*l_gxzz + l_Gamxyy*l_gxxz + l_Gamyyy*l_gxyz + l_Gamzyy*l_gxzz + l_Gamxyz*l_gxxy + l_Gamyyz*l_gxyy + l_Gamzyz*l_gxzy + l_Gamxxy*l_gyzx + l_Gamyxy*l_gyzy + l_Gamzxy*l_gyzz) + 
          gupxz * (l_Gamxxy*l_gxzz + l_Gamyxy*l_gyzz + l_Gamzxy*l_gzzz + l_Gamxxz*l_gxzy + l_Gamyxz*l_gyzy + l_Gamzxz*l_gzzy + l_Gamxyz*l_gxzx + l_Gamyyz*l_gxzy + l_Gamzyz*l_gxzz + l_Gamxyz*l_gxxz + l_Gamyyz*l_gxyz + l_Gamzyz*l_gxzz + l_Gamxzz*l_gxxy + l_Gamyzz*l_gxyy + l_Gamzzz*l_gxzy + l_Gamxxy*l_gzzx + l_Gamyxy*l_gzzy + l_Gamzxy*l_gzzz) + 
          gupyy * (l_Gamxyy*l_gxyz + l_Gamyyy*l_gyyz + l_Gamzyy*l_gyzz + l_Gamxyz*l_gxyy + l_Gamyyz*l_gyyy + l_Gamzyz*l_gyzy + l_Gamxyy*l_gyzx + l_Gamyyy*l_gyzy + l_Gamzyy*l_gyzz) + 
          gupyz * (l_Gamxyy*l_gxzz + l_Gamyyy*l_gyzz + l_Gamzyy*l_gzzz + l_Gamxyz*l_gxzy + l_Gamyyz*l_gyzy + l_Gamzyz*l_gzzy + l_Gamxyz*l_gyzx + l_Gamyyz*l_gyzy + l_Gamzyz*l_gyzz + l_Gamxyz*l_gxyz + l_Gamyyz*l_gyyz + l_Gamzyz*l_gyzz + l_Gamxzz*l_gxyy + l_Gamyzz*l_gyyy + l_Gamzzz*l_gyzy + l_Gamxyy*l_gzzx + l_Gamyyy*l_gzzy + l_Gamzyy*l_gzzz) + 
          gupzz * (l_Gamxyz*l_gxzz + l_Gamyyz*l_gyzz + l_Gamzyz*l_gzzz + l_Gamxzz*l_gxzy + l_Gamyzz*l_gyzy + l_Gamzzz*l_gzzy + l_Gamxyz*l_gzzx + l_Gamyyz*l_gzzy + l_Gamzyz*l_gzzz);

    // ==========================================
    // Step 6: Chi 二阶导数与 Ricci 修正
    // ==========================================
    d_fdderivs_point(dims, chi, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    
    // 协变导数修正
    fxx -= l_Gamxxx * val_chix + l_Gamyxx * val_chiy + l_Gamzxx * val_chiz;
    fxy -= l_Gamxxy * val_chix + l_Gamyxy * val_chiy + l_Gamzxy * val_chiz;
    fxz -= l_Gamxxz * val_chix + l_Gamyxz * val_chiy + l_Gamzxz * val_chiz;
    fyy -= l_Gamxyy * val_chix + l_Gamyyy * val_chiy + l_Gamzyy * val_chiz;
    fyz -= l_Gamxyz * val_chix + l_Gamyyz * val_chiy + l_Gamzyz * val_chiz;
    fzz -= l_Gamxzz * val_chix + l_Gamyzz * val_chiy + l_Gamzzz * val_chiz;

    double f_scalar = gupxx * (fxx - F3o2/chin1 * val_chix * val_chix) + 
                      gupyy * (fyy - F3o2/chin1 * val_chiy * val_chiy) + 
                      gupzz * (fzz - F3o2/chin1 * val_chiz * val_chiz) + 
                      TWO * (gupxy * (fxy - F3o2/chin1 * val_chix * val_chiy) + 
                             gupxz * (fxz - F3o2/chin1 * val_chix * val_chiz) + 
                             gupyz * (fyz - F3o2/chin1 * val_chiy * val_chiz));
    
    // Add to Ricci
    Rxx += (fxx - val_chix*val_chix/chin1/TWO + l_gxx * f_scalar)/chin1/TWO;
    Ryy += (fyy - val_chiy*val_chiy/chin1/TWO + l_gyy * f_scalar)/chin1/TWO;
    Rzz += (fzz - val_chiz*val_chiz/chin1/TWO + l_gzz * f_scalar)/chin1/TWO;
    Rxy += (fxy - val_chix*val_chiy/chin1/TWO + l_gxy * f_scalar)/chin1/TWO;
    Rxz += (fxz - val_chix*val_chiz/chin1/TWO + l_gxz * f_scalar)/chin1/TWO;
    Ryz += (fyz - val_chiy*val_chiz/chin1/TWO + l_gyz * f_scalar)/chin1/TWO;

    // ==========================================
    // Step 7: Lapse 二阶导数 & trK_rhs
    // ==========================================
    d_fdderivs_point(dims, Lap, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    // 计算物理连接系数 (暂存到 Gam 数组中以节省寄存器，最后会写回 Global)
    double gx_phy = (gupxx * val_chix + gupxy * val_chiy + gupxz * val_chiz)/chin1;
    double gy_phy = (gupxy * val_chix + gupyy * val_chiy + gupyz * val_chiz)/chin1;
    double gz_phy = (gupxz * val_chix + gupyz * val_chiy + gupzz * val_chiz)/chin1;
    
    // 更新为物理连接系数 (对应 Fortran 241-258)
    l_Gamxxx -= ((val_chix + val_chix)/chin1 - l_gxx * gx_phy)*HALF; // Gamxxx[idx] = l_Gamxxx;
    l_Gamyxx -= (                            - l_gxx * gy_phy)*HALF; // Gamyxx[idx] = l_Gamyxx;
    l_Gamzxx -= (                            - l_gxx * gz_phy)*HALF; // Gamzxx[idx] = l_Gamzxx;
    l_Gamxyy -= (                            - l_gyy * gx_phy)*HALF; // Gamxyy[idx] = l_Gamxyy;
    l_Gamyyy -= ((val_chiy + val_chiy)/chin1 - l_gyy * gy_phy)*HALF; // Gamyyy[idx] = l_Gamyyy;
    l_Gamzyy -= (                            - l_gyy * gz_phy)*HALF; // Gamzyy[idx] = l_Gamzyy;
    l_Gamxzz -= (                            - l_gzz * gx_phy)*HALF; // Gamxzz[idx] = l_Gamxzz;
    l_Gamyzz -= (                            - l_gzz * gy_phy)*HALF; // Gamyzz[idx] = l_Gamyzz;
    l_Gamzzz -= ((val_chiz + val_chiz)/chin1 - l_gzz * gz_phy)*HALF; // Gamzzz[idx] = l_Gamzzz;

    l_Gamxxy -= (val_chiy/chin1 - l_gxy * gx_phy)*HALF; // Gamxxy[idx] = l_Gamxxy;
    l_Gamyxy -= (val_chix/chin1 - l_gxy * gy_phy)*HALF; // Gamyxy[idx] = l_Gamyxy;
    l_Gamzxy -= (                 - l_gxy * gz_phy)*HALF; // Gamzxy[idx] = l_Gamzxy;
    l_Gamxxz -= (val_chiz/chin1 - l_gxz * gx_phy)*HALF; // Gamxxz[idx] = l_Gamxxz;
    l_Gamyxz -= (                 - l_gxz * gy_phy)*HALF; // Gamyxz[idx] = l_Gamyxz;
    l_Gamzxz -= (val_chix/chin1 - l_gxz * gz_phy)*HALF; // Gamzxz[idx] = l_Gamzxz;
    l_Gamxyz -= (                 - l_gyz * gx_phy)*HALF; // Gamxyz[idx] = l_Gamxyz;
    l_Gamyyz -= (val_chiz/chin1 - l_gyz * gy_phy)*HALF; // Gamyyz[idx] = l_Gamyyz;
    l_Gamzyz -= (val_chiy/chin1 - l_gyz * gz_phy)*HALF; // Gamzyz[idx] = l_Gamzyz;

    // Lapse 的协变导数 D_i D_j alpha
    fxx = fxx - l_Gamxxx*Lapx - l_Gamyxx*Lapy - l_Gamzxx*Lapz;
    fyy = fyy - l_Gamxyy*Lapx - l_Gamyyy*Lapy - l_Gamzyy*Lapz;
    fzz = fzz - l_Gamxzz*Lapx - l_Gamyzz*Lapy - l_Gamzzz*Lapz;
    fxy = fxy - l_Gamxxy*Lapx - l_Gamyxy*Lapy - l_Gamzxy*Lapz;
    fxz = fxz - l_Gamxxz*Lapx - l_Gamyxz*Lapy - l_Gamzxz*Lapz;
    fyz = fyz - l_Gamxyz*Lapx - l_Gamyyz*Lapy - l_Gamzyz*Lapz;

    double trK_rhs_val = gupxx * fxx + gupyy * fyy + gupzz * fzz + TWO* (gupxy * fxy + gupxz * fxz + gupyz * fyz);

    // ==========================================
    // Step 8: 组装 Aij_rhs & trK_rhs
    // ==========================================
    double S = chin1 * (gupxx * Sxx[idx] + gupyy * Syy[idx] + gupzz * Szz[idx] + 
               TWO * (gupxy * Sxy[idx] + gupxz * Sxz[idx] + gupyz * Syz[idx]));

    double term_xx = gupxx * l_Axx * l_Axx + gupyy * l_Axy * l_Axy + gupzz * l_Axz * l_Axz + TWO * (gupxy * l_Axx * l_Axy + gupxz * l_Axx * l_Axz + gupyz * l_Axy * l_Axz);
    double term_yy = gupxx * l_Axy * l_Axy + gupyy * l_Ayy * l_Ayy + gupzz * l_Ayz * l_Ayz + TWO * (gupxy * l_Axy * l_Ayy + gupxz * l_Axy * l_Ayz + gupyz * l_Ayy * l_Ayz);
    double term_zz = gupxx * l_Axz * l_Axz + gupyy * l_Ayz * l_Ayz + gupzz * l_Azz * l_Azz + TWO * (gupxy * l_Axz * l_Ayz + gupxz * l_Axz * l_Azz + gupyz * l_Ayz * l_Azz);
    double term_xy = gupxx * l_Axx * l_Axy + gupyy * l_Axy * l_Ayy + gupzz * l_Axz * l_Ayz + gupxy * (l_Axx * l_Ayy + l_Axy * l_Axy) + gupxz * (l_Axx * l_Ayz + l_Axz * l_Axy) + gupyz * (l_Axy * l_Ayz + l_Axz * l_Ayy);
    double term_xz = gupxx * l_Axx * l_Axz + gupyy * l_Axy * l_Ayz + gupzz * l_Axz * l_Azz + gupxy * (l_Axx * l_Ayz + l_Axy * l_Axz) + gupxz * (l_Axx * l_Azz + l_Axz * l_Axz) + gupyz * (l_Axy * l_Azz + l_Axz * l_Ayz);
    double term_yz = gupxx * l_Axy * l_Axz + gupyy * l_Ayy * l_Ayz + gupzz * l_Ayz * l_Azz + gupxy * (l_Axy * l_Ayz + l_Ayy * l_Axz) + gupxz * (l_Axy * l_Azz + l_Ayz * l_Axz) + gupyz * (l_Ayy * l_Azz + l_Ayz * l_Ayz);

    double trA2 = gupxx * term_xx + gupyy * term_yy + gupzz * term_zz + TWO * (gupxy * term_xy + gupxz * term_xz + gupyz * term_yz);

    double f = F2o3 * val_trK * val_trK - trA2 - F16*PI*rho[idx] + EIGHT*PI*S;
    double f_trace = -F1o3 * (trK_rhs_val + alpn1/chin1 * f);

    // 计算 Aij 源项
    double src_xx = alpn1 * (Rxx - EIGHT*PI*Sxx[idx]) - fxx; // fxx is D_i D_j Lap
    double src_yy = alpn1 * (Ryy - EIGHT*PI*Syy[idx]) - fyy;
    double src_zz = alpn1 * (Rzz - EIGHT*PI*Szz[idx]) - fzz;
    double src_xy = alpn1 * (Rxy - EIGHT*PI*Sxy[idx]) - fxy;
    double src_xz = alpn1 * (Rxz - EIGHT*PI*Sxz[idx]) - fxz;
    double src_yz = alpn1 * (Ryz - EIGHT*PI*Syz[idx]) - fyz;

    double Axx_rhs_val = src_xx - l_gxx * f_trace;
    double Ayy_rhs_val = src_yy - l_gyy * f_trace;
    double Azz_rhs_val = src_zz - l_gzz * f_trace;
    double Axy_rhs_val = src_xy - l_gxy * f_trace;
    double Axz_rhs_val = src_xz - l_gxz * f_trace;
    double Ayz_rhs_val = src_yz - l_gyz * f_trace;

    // 添加平流项 (Lie derivative of Aij)
    Axx_rhs[idx] = chin1 * Axx_rhs_val + alpn1 * (val_trK * l_Axx - TWO * term_xx) + TWO * (l_Axx * betaxx + l_Axy * betayx + l_Axz * betazx) - F2o3 * l_Axx * div_beta;
    Ayy_rhs[idx] = chin1 * Ayy_rhs_val + alpn1 * (val_trK * l_Ayy - TWO * term_yy) + TWO * (l_Axy * betaxy + l_Ayy * betayy + l_Ayz * betazy) - F2o3 * l_Ayy * div_beta;
    Azz_rhs[idx] = chin1 * Azz_rhs_val + alpn1 * (val_trK * l_Azz - TWO * term_zz) + TWO * (l_Axz * betaxz + l_Ayz * betayz + l_Azz * betazz) - F2o3 * l_Azz * div_beta;
    
    Axy_rhs[idx] = chin1 * Axy_rhs_val + alpn1 * (val_trK * l_Axy - TWO * term_xy) + l_Axx * betaxy + l_Axz * betazy + l_Ayy * betayx + l_Ayz * betazx - l_Axy * betazz + F1o3 * l_Axy * div_beta;
    Ayz_rhs[idx] = chin1 * Ayz_rhs_val + alpn1 * (val_trK * l_Ayz - TWO * term_yz) + l_Axy * betaxz + l_Ayy * betayz + l_Axz * betaxy + l_Azz * betazy - l_Ayz * betaxx + F1o3 * l_Ayz * div_beta;
    Axz_rhs[idx] = chin1 * Axz_rhs_val + alpn1 * (val_trK * l_Axz - TWO * term_xz) + l_Axx * betaxz + l_Axy * betayz + l_Ayz * betayx + l_Azz * betazx - l_Axz * betayy + F1o3 * l_Axz * div_beta;

    S = chin1 * (gupxx * Sxx[idx] + gupyy * Syy[idx] + gupzz * Szz[idx] + TWO * (gupxy * Sxy[idx] + gupxz * Sxz[idx] + gupyz * Syz[idx]));

    trK_rhs[idx] = -chin1 * trK_rhs_val + alpn1 * (F1o3 * val_trK * val_trK + trA2 + FOUR * PI * (rho[idx] + S));

    // Gauge vars RHS
    Lap_rhs[idx] = -TWO * alpn1 * val_trK;
    betax_rhs[idx] = FF * dtSfx[idx];
    betay_rhs[idx] = FF * dtSfy[idx];
    betaz_rhs[idx] = FF * dtSfz[idx];
    dtSfx_rhs[idx] = val_Gamx_rhs - eta * dtSfx[idx];
    dtSfy_rhs[idx] = val_Gamy_rhs - eta * dtSfy[idx];
    dtSfz_rhs[idx] = val_Gamz_rhs - eta * dtSfz[idx];

    // 写回 Gam_rhs
    Gamx_rhs[idx] = val_Gamx_rhs;
    Gamy_rhs[idx] = val_Gamy_rhs;
    Gamz_rhs[idx] = val_Gamz_rhs;

    Gamxxx[idx] = l_Gamxxx; Gamxxy[idx] = l_Gamxxy; Gamxxz[idx] = l_Gamxxz;
    Gamxyy[idx] = l_Gamxyy; Gamxyz[idx] = l_Gamxyz; Gamxzz[idx] = l_Gamxzz;

    Gamyxx[idx] = l_Gamyxx; Gamyxy[idx] = l_Gamyxy; Gamyxz[idx] = l_Gamyxz;
    Gamyyy[idx] = l_Gamyyy; Gamyyz[idx] = l_Gamyyz; Gamyzz[idx] = l_Gamyzz;

    Gamzxx[idx] = l_Gamzxx; Gamzxy[idx] = l_Gamzxy; Gamzxz[idx] = l_Gamzxz;
    Gamzyy[idx] = l_Gamzyy; Gamzyz[idx] = l_Gamzyz; Gamzzz[idx] = l_Gamzzz;

    Rxx_out[idx] = Rxx; Ryy_out[idx] = Ryy; Rzz_out[idx] = Rzz;
    Rxy_out[idx] = Rxy; Rxz_out[idx] = Rxz; Ryz_out[idx] = Ryz;
}



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
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= ex0 || j >= ex1 || k >= ex2) return;

    int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    int dims[3] = {ex0, ex1, ex2};

    // 准备平流所需的速度场 (Shift)
    // lopsided 需要传入 shift 的指针来判断上风方向
    // device 函数内部会根据 i,j,k 读取 betax[idx] 等

    // 定义对称性常量 (对应 Fortran 的 array 定义)
    // SSS: (1, 1, 1)
    // AAS: (-1, -1, 1)
    // ASA: (-1, 1, -1)
    // SAA: (1, -1, -1)
    // ASS: (-1, 1, 1)
    // SAS: (1, -1, 1)
    // SSA: (1, 1, -1)

    // =========================================================
    // Block 1: Metric Variables (gxx, gxy, gxz, gyy, gyz, gzz)
    // =========================================================
    
    // gxx (SSS)
    // Fortran: call lopsided(..., gxx, gxx_rhs, ..., SSS)
    // Note: Passing dxx for derivative calculation is equivalent to gxx
    gxx_rhs[idx] += d_lopsided_point(dims, dxx, gxx_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
    if (eps > 0.0) gxx_rhs[idx] += d_kodis_point(dims, dxx, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    // gxy (AAS)
    gxy_rhs[idx] += d_lopsided_point(dims, gxy, gxy_rhs, betax, betay, betaz, X, Y, Z, symmetry, ANTI, ANTI, SYM, i, j, k);
    if (eps > 0.0) gxy_rhs[idx] += d_kodis_point(dims, gxy, X, Y, Z, ANTI, ANTI, SYM, symmetry, eps, i, j, k);

    // gxz (ASA)
    gxz_rhs[idx] += d_lopsided_point(dims, gxz, gxz_rhs, betax, betay, betaz, X, Y, Z, symmetry, ANTI, SYM, ANTI, i, j, k);
    if (eps > 0.0) gxz_rhs[idx] += d_kodis_point(dims, gxz, X, Y, Z, ANTI, SYM, ANTI, symmetry, eps, i, j, k);

    // gyy (SSS)
    gyy_rhs[idx] += d_lopsided_point(dims, dyy, gyy_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
    if (eps > 0.0) gyy_rhs[idx] += d_kodis_point(dims, dyy, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    // gyz (SAA)
    gyz_rhs[idx] += d_lopsided_point(dims, gyz, gyz_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, ANTI, ANTI, i, j, k);
    if (eps > 0.0) gyz_rhs[idx] += d_kodis_point(dims, gyz, X, Y, Z, SYM, ANTI, ANTI, symmetry, eps, i, j, k);

    // gzz (SSS)
    gzz_rhs[idx] += d_lopsided_point(dims, dzz, gzz_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
    if (eps > 0.0) gzz_rhs[idx] += d_kodis_point(dims, dzz, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    // =========================================================
    // Block 2: Extrinsic Curvature (Axx ... Azz)
    // =========================================================

    // Axx (SSS)
    Axx_rhs[idx] += d_lopsided_point(dims, Axx, Axx_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
    if (eps > 0.0) Axx_rhs[idx] += d_kodis_point(dims, Axx, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    // Axy (AAS)
    Axy_rhs[idx] += d_lopsided_point(dims, Axy, Axy_rhs, betax, betay, betaz, X, Y, Z, symmetry, ANTI, ANTI, SYM, i, j, k);
    if (eps > 0.0) Axy_rhs[idx] += d_kodis_point(dims, Axy, X, Y, Z, ANTI, ANTI, SYM, symmetry, eps, i, j, k);

    // Axz (ASA)
    Axz_rhs[idx] += d_lopsided_point(dims, Axz, Axz_rhs, betax, betay, betaz, X, Y, Z, symmetry, ANTI, SYM, ANTI, i, j, k);
    if (eps > 0.0) Axz_rhs[idx] += d_kodis_point(dims, Axz, X, Y, Z, ANTI, SYM, ANTI, symmetry, eps, i, j, k);

    // Ayy (SSS)
    Ayy_rhs[idx] += d_lopsided_point(dims, Ayy, Ayy_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
    if (eps > 0.0) Ayy_rhs[idx] += d_kodis_point(dims, Ayy, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    // Ayz (SAA)
    Ayz_rhs[idx] += d_lopsided_point(dims, Ayz, Ayz_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, ANTI, ANTI, i, j, k);
    if (eps > 0.0) Ayz_rhs[idx] += d_kodis_point(dims, Ayz, X, Y, Z, SYM, ANTI, ANTI, symmetry, eps, i, j, k);

    // Azz (SSS)
    Azz_rhs[idx] += d_lopsided_point(dims, Azz, Azz_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
    if (eps > 0.0) Azz_rhs[idx] += d_kodis_point(dims, Azz, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    // =========================================================
    // Block 3: Scalar Variables (chi, trK)
    // =========================================================

    // chi (SSS)
    chi_rhs[idx] += d_lopsided_point(dims, chi, chi_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
    if (eps > 0.0) chi_rhs[idx] += d_kodis_point(dims, chi, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    // trK (SSS)
    trK_rhs[idx] += d_lopsided_point(dims, trK, trK_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
    if (eps > 0.0) trK_rhs[idx] += d_kodis_point(dims, trK, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    // =========================================================
    // Block 4: Gauge Variables - Conformal Connection (Gam)
    // =========================================================

    // Gamx (ASS)
    Gamx_rhs[idx] += d_lopsided_point(dims, Gamx, Gamx_rhs, betax, betay, betaz, X, Y, Z, symmetry, ANTI, SYM, SYM, i, j, k);
    if (eps > 0.0) Gamx_rhs[idx] += d_kodis_point(dims, Gamx, X, Y, Z, ANTI, SYM, SYM, symmetry, eps, i, j, k);

    // Gamy (SAS)
    Gamy_rhs[idx] += d_lopsided_point(dims, Gamy, Gamy_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, ANTI, SYM, i, j, k);
    if (eps > 0.0) Gamy_rhs[idx] += d_kodis_point(dims, Gamy, X, Y, Z, SYM, ANTI, SYM, symmetry, eps, i, j, k);

    // Gamz (SSA)
    Gamz_rhs[idx] += d_lopsided_point(dims, Gamz, Gamz_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, ANTI, i, j, k);
    if (eps > 0.0) Gamz_rhs[idx] += d_kodis_point(dims, Gamz, X, Y, Z, SYM, SYM, ANTI, symmetry, eps, i, j, k);

    // =========================================================
    // Block 5: Gauge Variables - Lapse & Shift
    // =========================================================

    // Lap (SSS) - Note: bam code does not apply dissipation on gauge vars usually, but Fortran logic here DOES for Lap
    Lap_rhs[idx] += d_lopsided_point(dims, Lap, Lap_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
    if (eps > 0.0) Lap_rhs[idx] += d_kodis_point(dims, Lap, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    // betax (ASS)
    betax_rhs[idx] += d_lopsided_point(dims, betax, betax_rhs, betax, betay, betaz, X, Y, Z, symmetry, ANTI, SYM, SYM, i, j, k);
    if (eps > 0.0) betax_rhs[idx] += d_kodis_point(dims, betax, X, Y, Z, ANTI, SYM, SYM, symmetry, eps, i, j, k);

    // betay (SAS)
    betay_rhs[idx] += d_lopsided_point(dims, betay, betay_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, ANTI, SYM, i, j, k);
    if (eps > 0.0) betay_rhs[idx] += d_kodis_point(dims, betay, X, Y, Z, SYM, ANTI, SYM, symmetry, eps, i, j, k);

    // betaz (SSA)
    betaz_rhs[idx] += d_lopsided_point(dims, betaz, betaz_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, ANTI, i, j, k);
    if (eps > 0.0) betaz_rhs[idx] += d_kodis_point(dims, betaz, X, Y, Z, SYM, SYM, ANTI, symmetry, eps, i, j, k);

    // =========================================================
    // Block 6: Gauge Variables - Time derivative of Shift (dtSf)
    // =========================================================

    // dtSfx (ASS)
    dtSfx_rhs[idx] += d_lopsided_point(dims, dtSfx, dtSfx_rhs, betax, betay, betaz, X, Y, Z, symmetry, ANTI, SYM, SYM, i, j, k);
    if (eps > 0.0) dtSfx_rhs[idx] += d_kodis_point(dims, dtSfx, X, Y, Z, ANTI, SYM, SYM, symmetry, eps, i, j, k);

    // dtSfy (SAS)
    dtSfy_rhs[idx] += d_lopsided_point(dims, dtSfy, dtSfy_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, ANTI, SYM, i, j, k);
    if (eps > 0.0) dtSfy_rhs[idx] += d_kodis_point(dims, dtSfy, X, Y, Z, SYM, ANTI, SYM, symmetry, eps, i, j, k);

    // dtSfz (SSA)
    dtSfz_rhs[idx] += d_lopsided_point(dims, dtSfz, dtSfz_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, ANTI, i, j, k);
    if (eps > 0.0) dtSfz_rhs[idx] += d_kodis_point(dims, dtSfz, X, Y, Z, SYM, SYM, ANTI, symmetry, eps, i, j, k);
}

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
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= ex0 || j >= ex1 || k >= ex2) return;

    int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    int dims[3] = {ex0, ex1, ex2};

    // ==========================================
    // 0. 加载数据
    // ==========================================
    double val_chi = chi[idx];
    double chin1 = val_chi + ONE;
    double val_trK = trK[idx];

    double gupxx = gupxx_in[idx]; double gupxy = gupxy_in[idx]; double gupxz = gupxz_in[idx];
    double gupyy = gupyy_in[idx]; double gupyz = gupyz_in[idx]; double gupzz = gupzz_in[idx];

    double l_Axx = Axx[idx]; double l_Axy = Axy[idx]; double l_Axz = Axz[idx];
    double l_Ayy = Ayy[idx]; double l_Ayz = Ayz[idx]; double l_Azz = Azz[idx];

    double l_Rxx = Rxx_in[idx]; double l_Rxy = Rxy_in[idx]; double l_Rxz = Rxz_in[idx];
    double l_Ryy = Ryy_in[idx]; double l_Ryz = Ryz_in[idx]; double l_Rzz = Rzz_in[idx];

    // ==========================================
    // 1. Hamiltonian Constraint
    // ==========================================
    // ham_Res = trR + 2/3 K^2 - A_ij A^ij - 16 PI rho

    // 计算 trR (Respect to physical metric)
    // Fortran Line 372
    double ham_val = gupxx * l_Rxx + gupyy * l_Ryy + gupzz * l_Rzz + 
               TWO * (gupxy * l_Rxy + gupxz * l_Rxz + gupyz * l_Ryz);

    double term_xx = gupxx * l_Axx * l_Axx + gupyy * l_Axy * l_Axy + gupzz * l_Axz * l_Axz + TWO * (gupxy * l_Axx * l_Axy + gupxz * l_Axx * l_Axz + gupyz * l_Axy * l_Axz);
    double term_yy = gupxx * l_Axy * l_Axy + gupyy * l_Ayy * l_Ayy + gupzz * l_Ayz * l_Ayz + TWO * (gupxy * l_Axy * l_Ayy + gupxz * l_Axy * l_Ayz + gupyz * l_Ayy * l_Ayz);
    double term_zz = gupxx * l_Axz * l_Axz + gupyy * l_Ayz * l_Ayz + gupzz * l_Azz * l_Azz + TWO * (gupxy * l_Axz * l_Ayz + gupxz * l_Axz * l_Azz + gupyz * l_Ayz * l_Azz);
    
    double term_xy = gupxx * l_Axx * l_Axy + gupyy * l_Axy * l_Ayy + gupzz * l_Axz * l_Ayz + gupxy * (l_Axx * l_Ayy + l_Axy * l_Axy) + gupxz * (l_Axx * l_Ayz + l_Axz * l_Axy) + gupyz * (l_Axy * l_Ayz + l_Axz * l_Ayy);
    double term_xz = gupxx * l_Axx * l_Axz + gupyy * l_Axy * l_Ayz + gupzz * l_Axz * l_Azz + gupxy * (l_Axx * l_Ayz + l_Axy * l_Axz) + gupxz * (l_Axx * l_Azz + l_Axz * l_Axz) + gupyz * (l_Axy * l_Azz + l_Axz * l_Ayz);
    double term_yz = gupxx * l_Axy * l_Axz + gupyy * l_Ayy * l_Ayz + gupzz * l_Ayz * l_Azz + gupxy * (l_Axy * l_Ayz + l_Ayy * l_Axz) + gupxz * (l_Axy * l_Azz + l_Ayz * l_Axz) + gupyz * (l_Ayy * l_Azz + l_Ayz * l_Ayz);

    double trA2 = gupxx * term_xx + gupyy * term_yy + gupzz * term_zz + TWO * (gupxy * term_xy + gupxz * term_xz + gupyz * term_yz);

    // Final Hamiltonian Calculation
    // Fortran Line 375
    ham_Res[idx] = chin1 * ham_val + F2o3 * val_trK * val_trK - trA2 - F16 * PI * rho[idx];
    // ==========================================
    // 2. Momentum Constraint
    // ==========================================
    // mov_Res_j = D_k A^k_j - 2/3 d_j trK - 8 PI S_j

    // 需要 trK 的导数
    double Kx, Ky, Kz;
    d_fderivs_point(dims, trK, &Kx, &Ky, &Kz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    // 需要 Aij 的导数 (Fortran calls fderivs 6 times)
    // 为了节省寄存器和避免创建大数组，我们分量计算并直接应用 Covariant 修正
    
    double val_chix = chix_in[idx];
    double val_chiy = chiy_in[idx];
    double val_chiz = chiz_in[idx];

    // --- Compute D_i A_jk stored in variables named like `dA_xxx` (meaning D_x A_xx) ---
    
    // 1. Axx (SYM, SYM, SYM)
    double d_Axx_x, d_Axx_y, d_Axx_z;
    double d_Axy_x, d_Axy_y, d_Axy_z;
    double d_Axz_x, d_Axz_y, d_Axz_z;
    double d_Ayy_x, d_Ayy_y, d_Ayy_z;
    double d_Ayz_x, d_Ayz_y, d_Ayz_z;
    double d_Azz_x, d_Azz_y, d_Azz_z;
    d_fderivs_point(dims, Axx, &d_Axx_x, &d_Axx_y, &d_Axx_z, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Axy, &d_Axy_x, &d_Axy_y, &d_Axy_z, X, Y, Z, ANTI, ANTI, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Axz, &d_Axz_x, &d_Axz_y, &d_Axz_z, X, Y, Z, ANTI, SYM, ANTI, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Ayy, &d_Ayy_x, &d_Ayy_y, &d_Ayy_z, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Ayz, &d_Ayz_x, &d_Ayz_y, &d_Ayz_z, X, Y, Z, SYM, ANTI, ANTI, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Azz, &d_Azz_x, &d_Azz_y, &d_Azz_z, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);


    double DA_xxx = d_Axx_x - (Gamxxx[idx] * l_Axx + Gamyxx[idx] * l_Axy + Gamzxx[idx] * l_Axz 
                             + Gamxxx[idx] * l_Axx + Gamyxx[idx] * l_Axy + Gamzxx[idx] * l_Axz) - val_chix * l_Axx / chin1;
    double DA_xyx = d_Axy_x - (Gamxxy[idx] * l_Axx + Gamyxy[idx] * l_Axy + Gamzxy[idx] * l_Axz
                             + Gamxxx[idx] * l_Axy + Gamyxx[idx] * l_Ayy + Gamzxx[idx] * l_Ayz) - val_chix * l_Axy / chin1;
    double DA_xzx = d_Axz_x - (Gamxxz[idx] * l_Axx + Gamyxz[idx] * l_Axy + Gamzxz[idx] * l_Axz
                             + Gamxxx[idx] * l_Axz + Gamyxx[idx] * l_Ayz + Gamzxx[idx] * l_Azz) - val_chix * l_Axz / chin1;
    double DA_yyx = d_Ayy_x - (Gamxxy[idx] * l_Axy + Gamyxy[idx] * l_Ayy + Gamzxy[idx] * l_Ayz
                             + Gamxxy[idx] * l_Axy + Gamyxy[idx] * l_Ayy + Gamzxy[idx] * l_Ayz) - val_chix * l_Ayy / chin1;
    double DA_yzx = d_Ayz_x - (Gamxxz[idx] * l_Axy + Gamyxz[idx] * l_Ayy + Gamzxz[idx] * l_Ayz
                             + Gamxxy[idx] * l_Axz + Gamyxy[idx] * l_Ayz + Gamzxy[idx] * l_Azz) - val_chix * l_Ayz / chin1;
    double DA_zzx = d_Azz_x - (Gamxxz[idx] * l_Axz + Gamyxz[idx] * l_Ayz + Gamzxz[idx] * l_Azz
                             + Gamxxz[idx] * l_Axz + Gamyxz[idx] * l_Ayz + Gamzxz[idx] * l_Azz) - val_chix * l_Azz / chin1;
    double DA_xxy = d_Axx_y - (Gamxxy[idx] * l_Axx + Gamyxy[idx] * l_Axy + Gamzxy[idx] * l_Axz
                             + Gamxxy[idx] * l_Axx + Gamyxy[idx] * l_Axy + Gamzxy[idx] * l_Axz) - val_chiy * l_Axx / chin1;
    double DA_xyy = d_Axy_y - (Gamxyy[idx] * l_Axx + Gamyyy[idx] * l_Axy + Gamzyy[idx] * l_Axz
                             + Gamxxy[idx] * l_Axy + Gamyxy[idx] * l_Ayy + Gamzxy[idx] * l_Ayz) - val_chiy * l_Axy / chin1;
    double DA_xzy = d_Axz_y - (Gamxyz[idx] * l_Axx + Gamyyz[idx] * l_Axy + Gamzyz[idx] * l_Axz
                             + Gamxxy[idx] * l_Axz + Gamyxy[idx] * l_Ayz + Gamzxy[idx] * l_Azz) - val_chiy * l_Axz / chin1;
    double DA_yyy = d_Ayy_y - (Gamxyy[idx] * l_Axy + Gamyyy[idx] * l_Ayy + Gamzyy[idx] * l_Ayz
                             + Gamxyy[idx] * l_Axy + Gamyyy[idx] * l_Ayy + Gamzyy[idx] * l_Ayz) - val_chiy * l_Ayy / chin1; 
    double DA_yzy = d_Ayz_y - (Gamxyz[idx] * l_Axy + Gamyyz[idx] * l_Ayy + Gamzyz[idx] * l_Ayz
                             + Gamxyy[idx] * l_Axz + Gamyyy[idx] * l_Ayz + Gamzyy[idx] * l_Azz) - val_chiy * l_Ayz / chin1;
    double DA_zzy = d_Azz_y - (Gamxyz[idx] * l_Axz + Gamyyz[idx] * l_Ayz + Gamzyz[idx] * l_Azz
                             + Gamxyz[idx] * l_Axz + Gamyyz[idx] * l_Ayz + Gamzyz[idx] * l_Azz) - val_chiy * l_Azz / chin1;
    double DA_xxz = d_Axx_z - (Gamxxz[idx] * l_Axx + Gamyxz[idx] * l_Axy + Gamzxz[idx] * l_Axz
                             + Gamxxz[idx] * l_Axx + Gamyxz[idx] * l_Axy + Gamzxz[idx] * l_Axz) - val_chiz * l_Axx / chin1;
    double DA_xyz = d_Axy_z - (Gamxyz[idx] * l_Axx + Gamyyz[idx] * l_Axy + Gamzyz[idx] * l_Axz
                             + Gamxxz[idx] * l_Axy + Gamyxz[idx] * l_Ayy + Gamzxz[idx] * l_Ayz) - val_chiz * l_Axy / chin1;
    double DA_xzz = d_Axz_z - (Gamxzz[idx] * l_Axx + Gamyzz[idx] * l_Axy + Gamzzz[idx] * l_Axz
                             + Gamxxz[idx] * l_Axz + Gamyxz[idx] * l_Ayz + Gamzxz[idx] * l_Azz) - val_chiz * l_Axz / chin1;
    double DA_yyz = d_Ayy_z - (Gamxyz[idx] * l_Axy + Gamyyz[idx] * l_Ayy + Gamzyz[idx] * l_Ayz
                             + Gamxyz[idx] * l_Axy + Gamyyz[idx] * l_Ayy + Gamzyz[idx] * l_Ayz) - val_chiz * l_Ayy / chin1;
    double DA_yzz = d_Ayz_z - (Gamxzz[idx] * l_Axy + Gamyzz[idx] * l_Ayy + Gamzzz[idx] * l_Ayz
                             + Gamxyz[idx] * l_Axz + Gamyyz[idx] * l_Ayz + Gamzyz[idx] * l_Azz) - val_chiz * l_Ayz / chin1;
    double DA_zzz = d_Azz_z - (Gamxzz[idx] * l_Axz + Gamyzz[idx] * l_Ayz + Gamzzz[idx] * l_Azz
                             + Gamxzz[idx] * l_Axz + Gamyzz[idx] * l_Ayz + Gamzzz[idx] * l_Azz) - val_chiz * l_Azz / chin1;


    // ==========================================
    // 3. Contraction (Compute mov_Res)
    // ==========================================
    
    // movx_Res (Fortran Lines 424-426)
    // Note: Use matching DA components. 
    // gupxx*gxxx -> gupxx * DA_xxx
    // gupyy*gxyy -> gupyy * DA_xyy
    // gupzz*gxzz -> gupzz * DA_xzz
    // gupxy*gxyx -> gupxy * DA_xyx
    // gupxz*gxzx -> gupxz * DA_xzx
    // gupyz*gxzy -> gupyz * DA_xzy
    // gupxy*gxxy -> gupxy * DA_xxy
    // gupxz*gxxz -> gupxz * DA_xxz
    // gupyz*gxyz -> gupyz * DA_xyz
    movx_Res[idx] = gupxx * DA_xxx + gupyy * DA_xyy + gupzz * DA_xzz
                  + gupxy * DA_xyx + gupxz * DA_xzx + gupyz * DA_xzy
                  + gupxy * DA_xxy + gupxz * DA_xxz + gupyz * DA_xyz;

    // movy_Res (Fortran Lines 427-429)
    movy_Res[idx] = gupxx * DA_xyx + gupyy * DA_yyy + gupzz * DA_yzz
                  + gupxy * DA_yyx + gupxz * DA_yzx + gupyz * DA_yzy
                  + gupxy * DA_xyy + gupxz * DA_xyz + gupyz * DA_yyz;

    // movz_Res (Fortran Lines 430-432)
    movz_Res[idx] = gupxx * DA_xzx + gupyy * DA_yzy + gupzz * DA_zzz
                  + gupxy * DA_yzx + gupxz * DA_zzx + gupyz * DA_zzy
                  + gupxy * DA_xzy + gupxz * DA_xzz + gupyz * DA_yzz; // Note: last term gupyz*gyzz -> DA_yzz

    // Subtract K derivatives and Matter terms
    // Fortran Lines 434-436
    movx_Res[idx] = movx_Res[idx] - F2o3 * Kx - F8 * PI * Sx[idx];
    movy_Res[idx] = movy_Res[idx] - F2o3 * Ky - F8 * PI * Sy[idx];
    movz_Res[idx] = movz_Res[idx] - F2o3 * Kz - F8 * PI * Sz[idx];
}