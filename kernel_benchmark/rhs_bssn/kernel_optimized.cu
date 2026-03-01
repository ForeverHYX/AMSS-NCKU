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

__global__ void bssn_core_rhs_kernel(
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
    double* __restrict__ chi_rhs, double* __restrict__ trK_rhs,
    double* __restrict__ gxx_rhs, double* __restrict__ gxy_rhs, double* __restrict__ gxz_rhs,
    double* __restrict__ gyy_rhs, double* __restrict__ gyz_rhs, double* __restrict__ gzz_rhs,
    double* __restrict__ Axx_rhs, double* __restrict__ Axy_rhs, double* __restrict__ Axz_rhs,
    double* __restrict__ Ayy_rhs, double* __restrict__ Ayz_rhs, double* __restrict__ Azz_rhs,
    double* __restrict__ Gamx_rhs, double* __restrict__ Gamy_rhs, double* __restrict__ Gamz_rhs,
    double* __restrict__ Lap_rhs, 
    double* __restrict__ betax_rhs, double* __restrict__ betay_rhs, double* __restrict__ betaz_rhs,
    double* __restrict__ dtSfx_rhs, double* __restrict__ dtSfy_rhs, double* __restrict__ dtSfz_rhs,
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

    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    int dims[3] = {ex0, ex1, ex2};

    double val_Lap = Lap[idx];
    double val_chi = chi[idx];
    double alpn1 = val_Lap + ONE;
    double chin1 = val_chi + ONE;

    double l_gxx = dxx[idx] + ONE; double l_gxy = gxy[idx]; double l_gxz = gxz[idx];
    double l_gyy = dyy[idx] + ONE; double l_gyz = gyz[idx]; double l_gzz = dzz[idx] + ONE;

    double val_trK = trK[idx];

    double l_Axx = Axx[idx]; double l_Axy = Axy[idx]; double l_Axz = Axz[idx];
    double l_Ayy = Ayy[idx]; double l_Ayz = Ayz[idx]; double l_Azz = Azz[idx];

    double betaxx, betaxy, betaxz, betayx, betayy, betayz, betazx, betazy, betazz;
    d_fderivs_point(dims, betax, &betaxx, &betaxy, &betaxz, X, Y, Z, ANTI, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, betay, &betayx, &betayy, &betayz, X, Y, Z, SYM, ANTI, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, betaz, &betazx, &betazy, &betazz, X, Y, Z, SYM, SYM, ANTI, symmetry, lev, i, j, k);

    double div_beta = betaxx + betayy + betazz;

    double chix, chiy, chiz;
    d_fderivs_point(dims, chi, &chix, &chiy, &chiz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    chi_rhs[idx] = F2o3 * chin1 * (alpn1 * val_trK - div_beta);

    double gxxx, gxxy, gxxz, gxyx, gxyy, gxyz, gxzx, gxzy, gxzz;
    double gyyx, gyyy, gyyz, gyzx, gyzy, gyzz, gzzx, gzzy, gzzz;

    d_fderivs_point(dims, dxx, &gxxx, &gxxy, &gxxz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, gxy, &gxyx, &gxyy, &gxyz, X, Y, Z, ANTI, ANTI, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, gxz, &gxzx, &gxzy, &gxzz, X, Y, Z, ANTI, SYM, ANTI, symmetry, lev, i, j, k);
    d_fderivs_point(dims, dyy, &gyyx, &gyyy, &gyyz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, gyz, &gyzx, &gyzy, &gyzz, X, Y, Z, SYM, ANTI, ANTI, symmetry, lev, i, j, k);
    d_fderivs_point(dims, dzz, &gzzx, &gzzy, &gzzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

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

    double l_Rxx = gupxx * gupxx * l_Axx + gupxy * gupxy * l_Ayy + gupxz * gupxz * l_Azz + TWO*(gupxx * gupxy * l_Axy + gupxx * gupxz * l_Axz + gupxy * gupxz * l_Ayz);
    double l_Ryy = gupxy * gupxy * l_Axx + gupyy * gupyy * l_Ayy + gupyz * gupyz * l_Azz + TWO*(gupxy * gupyy * l_Axy + gupxy * gupyz * l_Axz + gupyy * gupyz * l_Ayz);
    double l_Rzz = gupxz * gupxz * l_Axx + gupyz * gupyz * l_Ayy + gupzz * gupzz * l_Azz + TWO*(gupxz * gupyz * l_Axy + gupxz * gupzz * l_Axz + gupyz * gupzz * l_Ayz);
    double l_Rxy = gupxx * gupxy * l_Axx + gupxy * gupyy * l_Ayy + gupxz * gupyz * l_Azz + (gupxx * gupyy + gupxy * gupxy)* l_Axy + (gupxx * gupyz + gupxz * gupxy)* l_Axz + (gupxy * gupyz + gupxz * gupyy)* l_Ayz;
    double l_Rxz = gupxx * gupxz * l_Axx + gupxy * gupyz * l_Ayy + gupxz * gupzz * l_Azz + (gupxx * gupyz + gupxy * gupxz)* l_Axy + (gupxx * gupzz + gupxz * gupxz)* l_Axz + (gupxy * gupzz + gupxz * gupyz)* l_Ayz;
    double l_Ryz = gupxy * gupxz * l_Axx + gupyy * gupyz * l_Ayy + gupyz * gupzz * l_Azz + (gupxy * gupyz + gupyy * gupxz)* l_Axy + (gupxy * gupzz + gupyz * gupxz)* l_Axz + (gupyy * gupzz + gupyz * gupyz)* l_Ayz;

    double Lapx, Lapy, Lapz, Kx, Ky, Kz;
    d_fderivs_point(dims, Lap, &Lapx, &Lapy, &Lapz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, trK, &Kx, &Ky, &Kz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    double val_Sx = Sx[idx]; double val_Sy = Sy[idx]; double val_Sz = Sz[idx];

    double val_Gamx_rhs = - TWO * (Lapx * l_Rxx + Lapy * l_Rxy + Lapz * l_Rxz) + 
        TWO * alpn1 * (-F3o2/chin1 * (chix * l_Rxx + chiy * l_Rxy + chiz * l_Rxz) - gupxx * (F2o3 * Kx + EIGHT * PI * val_Sx) - gupxy * (F2o3 * Ky + EIGHT * PI * val_Sy) - gupxz * (F2o3 * Kz + EIGHT * PI * val_Sz) + 
        l_Gamxxx * l_Rxx + l_Gamxyy * l_Ryy + l_Gamxzz * l_Rzz + TWO * (l_Gamxxy * l_Rxy + l_Gamxxz * l_Rxz + l_Gamxyz * l_Ryz));

    double val_Gamy_rhs = - TWO * (Lapx * l_Rxy + Lapy * l_Ryy + Lapz * l_Ryz) + 
        TWO * alpn1 * (-F3o2/chin1 * (chix * l_Rxy + chiy * l_Ryy + chiz * l_Ryz) - gupxy * (F2o3 * Kx + EIGHT * PI * val_Sx) - gupyy * (F2o3 * Ky + EIGHT * PI * val_Sy) - gupyz * (F2o3 * Kz + EIGHT * PI * val_Sz) + 
        l_Gamyxx * l_Rxx + l_Gamyyy * l_Ryy + l_Gamyzz * l_Rzz + TWO * (l_Gamyxy * l_Rxy + l_Gamyxz * l_Rxz + l_Gamyyz * l_Ryz));

    double val_Gamz_rhs = - TWO * (Lapx * l_Rxz + Lapy * l_Ryz + Lapz * l_Rzz) + 
        TWO * alpn1 * (-F3o2/chin1 * (chix * l_Rxz + chiy * l_Ryz + chiz * l_Rzz) - gupxz * (F2o3 * Kx + EIGHT * PI * val_Sx) - gupyz * (F2o3 * Ky + EIGHT * PI * val_Sy) - gupzz * (F2o3 * Kz + EIGHT * PI * val_Sz) + 
        l_Gamzxx * l_Rxx + l_Gamzyy * l_Ryy + l_Gamzzz * l_Rzz + TWO * (l_Gamzxy * l_Rxy + l_Gamzxz * l_Rxz + l_Gamzyz * l_Ryz));

    double bx_gxxx, bx_gxyx, bx_gxzx, bx_gyyx, bx_gyzx, bx_gzzx;
    d_fdderivs_point(dims, betax, &bx_gxxx, &bx_gxyx, &bx_gxzx, &bx_gyyx, &bx_gyzx, &bx_gzzx, X, Y, Z, ANTI, SYM, SYM, symmetry, lev, i, j, k);

    double by_gxxy, by_gxyy, by_gxzy, by_gyyy, by_gyzy, by_gzzy;
    d_fdderivs_point(dims, betay, &by_gxxy, &by_gxyy, &by_gxzy, &by_gyyy, &by_gyzy, &by_gzzy, X, Y, Z, SYM, ANTI, SYM, symmetry, lev, i, j, k);

    double bz_gxxz, bz_gxyz, bz_gxzz, bz_gyyz, bz_gyzz, bz_gzzz;
    d_fdderivs_point(dims, betaz, &bz_gxxz, &bz_gxyz, &bz_gxzz, &bz_gyyz, &bz_gyzz, &bz_gzzz, X, Y, Z, SYM, SYM, ANTI, symmetry, lev, i, j, k);

    double fxx, fxy, fxz, fyy, fyz, fzz;

    fxx = bx_gxxx + by_gxyy + bz_gxzz;
    fxy = bx_gxyx + by_gyyy + bz_gyzz;
    fxz = bx_gxzx + by_gyzy + bz_gzzz;

    double Gamxa = gupxx * l_Gamxxx + gupyy * l_Gamxyy + gupzz * l_Gamxzz + TWO * (gupxy * l_Gamxxy + gupxz * l_Gamxxz + gupyz * l_Gamxyz);
    double Gamya = gupxx * l_Gamyxx + gupyy * l_Gamyyy + gupzz * l_Gamyzz + TWO * (gupxy * l_Gamyxy + gupxz * l_Gamyxz + gupyz * l_Gamyyz);
    double Gamza = gupxx * l_Gamzxx + gupyy * l_Gamzyy + gupzz * l_Gamzzz + TWO * (gupxy * l_Gamzxy + gupxz * l_Gamzxz + gupyz * l_Gamzyz);
    
    double dGamxx, dGamxy, dGamxz, dGamyx, dGamyy, dGamyz, dGamzx, dGamzy, dGamzz;
    d_fderivs_point(dims, Gamx, &dGamxx, &dGamxy, &dGamxz, X, Y, Z, ANTI, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Gamy, &dGamyx, &dGamyy, &dGamyz, X, Y, Z, SYM, ANTI, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Gamz, &dGamzx, &dGamzy, &dGamzz, X, Y, Z, SYM, SYM, ANTI, symmetry, lev, i, j, k);

    val_Gamx_rhs += F2o3 * Gamxa * div_beta - (Gamxa * betaxx + Gamya * betaxy + Gamza * betaxz) + F1o3 * (gupxx * fxx + gupxy * fxy + gupxz * fxz) + gupxx * bx_gxxx + gupyy * bx_gyyx + gupzz * bx_gzzx + TWO * (gupxy * bx_gxyx + gupxz * bx_gxzx + gupyz * bx_gyzx);
    val_Gamy_rhs += F2o3 * Gamya * div_beta - (Gamxa * betayx + Gamya * betayy + Gamza * betayz) + F1o3 * (gupxy * fxx + gupyy * fxy + gupyz * fxz) + gupxx * by_gxxy + gupyy * by_gyyy + gupzz * by_gzzy + TWO * (gupxy * by_gxyy + gupxz * by_gxzy + gupyz * by_gyzy);
    val_Gamz_rhs += F2o3 * Gamza * div_beta - (Gamxa * betazx + Gamya * betazy + Gamza * betazz) + F1o3 * (gupxz * fxx + gupyz * fxy + gupzz * fxz) + gupxx * bz_gxxz + gupyy * bz_gyyz + gupzz * bz_gzzz + TWO * (gupxy * bz_gxyz + gupxz * bz_gxzz + gupyz * bz_gyzz);
    
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
    
    d_fdderivs_point(dims, dxx, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    l_Rxx = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;
    
    d_fdderivs_point(dims, dyy, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    l_Ryy = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;

    d_fdderivs_point(dims, dzz, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    l_Rzz = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;

    d_fdderivs_point(dims, gxy, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, ANTI, ANTI, SYM, symmetry, lev, i, j, k);
    l_Rxy = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;

    d_fdderivs_point(dims, gxz, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, ANTI, SYM, ANTI, symmetry, lev, i, j, k);
    l_Rxz = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;

    d_fdderivs_point(dims, gyz, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, ANTI, ANTI, symmetry, lev, i, j, k);
    l_Ryz = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;

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

    d_fdderivs_point(dims, chi, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    
    fxx -= l_Gamxxx * chix + l_Gamyxx * chiy + l_Gamzxx * chiz;
    fxy -= l_Gamxxy * chix + l_Gamyxy * chiy + l_Gamzxy * chiz;
    fxz -= l_Gamxxz * chix + l_Gamyxz * chiy + l_Gamzxz * chiz;
    fyy -= l_Gamxyy * chix + l_Gamyyy * chiy + l_Gamzyy * chiz;
    fyz -= l_Gamxyz * chix + l_Gamyyz * chiy + l_Gamzyz * chiz;
    fzz -= l_Gamxzz * chix + l_Gamyzz * chiy + l_Gamzzz * chiz;

    double f_scalar = gupxx * (fxx - F3o2/chin1 * chix * chix) + gupyy * (fyy - F3o2/chin1 * chiy * chiy) + gupzz * (fzz - F3o2/chin1 * chiz * chiz) + 
                      TWO * (gupxy * (fxy - F3o2/chin1 * chix * chiy) + gupxz * (fxz - F3o2/chin1 * chix * chiz) + gupyz * (fyz - F3o2/chin1 * chiy * chiz));
    
    l_Rxx += (fxx - chix*chix/chin1/TWO + l_gxx * f_scalar)/chin1/TWO;
    l_Ryy += (fyy - chiy*chiy/chin1/TWO + l_gyy * f_scalar)/chin1/TWO;
    l_Rzz += (fzz - chiz*chiz/chin1/TWO + l_gzz * f_scalar)/chin1/TWO;
    l_Rxy += (fxy - chix*chiy/chin1/TWO + l_gxy * f_scalar)/chin1/TWO;
    l_Rxz += (fxz - chix*chiz/chin1/TWO + l_gxz * f_scalar)/chin1/TWO;
    l_Ryz += (fyz - chiy*chiz/chin1/TWO + l_gyz * f_scalar)/chin1/TWO;

    d_fdderivs_point(dims, Lap, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    double gx_phy = (gupxx * chix + gupxy * chiy + gupxz * chiz)/chin1;
    double gy_phy = (gupxy * chix + gupyy * chiy + gupyz * chiz)/chin1;
    double gz_phy = (gupxz * chix + gupyz * chiy + gupzz * chiz)/chin1;
    
    l_Gamxxx -= ((chix + chix)/chin1 - l_gxx * gx_phy)*HALF; 
    l_Gamyxx -= (                            - l_gxx * gy_phy)*HALF; 
    l_Gamzxx -= (                            - l_gxx * gz_phy)*HALF; 
    l_Gamxyy -= (                            - l_gyy * gx_phy)*HALF; 
    l_Gamyyy -= ((chiy + chiy)/chin1 - l_gyy * gy_phy)*HALF; 
    l_Gamzyy -= (                            - l_gyy * gz_phy)*HALF; 
    l_Gamxzz -= (                            - l_gzz * gx_phy)*HALF; 
    l_Gamyzz -= (                            - l_gzz * gy_phy)*HALF; 
    l_Gamzzz -= ((chiz + chiz)/chin1 - l_gzz * gz_phy)*HALF; 

    l_Gamxxy -= (chiy/chin1 - l_gxy * gx_phy)*HALF; 
    l_Gamyxy -= (chix/chin1 - l_gxy * gy_phy)*HALF; 
    l_Gamzxy -= (                 - l_gxy * gz_phy)*HALF; 
    l_Gamxxz -= (chiz/chin1 - l_gxz * gx_phy)*HALF; 
    l_Gamyxz -= (                 - l_gxz * gy_phy)*HALF; 
    l_Gamzxz -= (chix/chin1 - l_gxz * gz_phy)*HALF; 
    l_Gamxyz -= (                 - l_gyz * gx_phy)*HALF; 
    l_Gamyyz -= (chiz/chin1 - l_gyz * gy_phy)*HALF; 
    l_Gamzyz -= (chiy/chin1 - l_gyz * gz_phy)*HALF; 

    fxx = fxx - l_Gamxxx*Lapx - l_Gamyxx*Lapy - l_Gamzxx*Lapz;
    fyy = fyy - l_Gamxyy*Lapx - l_Gamyyy*Lapy - l_Gamzyy*Lapz;
    fzz = fzz - l_Gamxzz*Lapx - l_Gamyzz*Lapy - l_Gamzzz*Lapz;
    fxy = fxy - l_Gamxxy*Lapx - l_Gamyxy*Lapy - l_Gamzxy*Lapz;
    fxz = fxz - l_Gamxxz*Lapx - l_Gamyxz*Lapy - l_Gamzxz*Lapz;
    fyz = fyz - l_Gamxyz*Lapx - l_Gamyyz*Lapy - l_Gamzyz*Lapz;

    double trK_rhs_val = gupxx * fxx + gupyy * fyy + gupzz * fzz + TWO* (gupxy * fxy + gupxz * fxz + gupyz * fyz);

    double S = chin1 * (gupxx * Sxx[idx] + gupyy * Syy[idx] + gupzz * Szz[idx] + TWO * (gupxy * Sxy[idx] + gupxz * Sxz[idx] + gupyz * Syz[idx]));

    double term_xx = gupxx * l_Axx * l_Axx + gupyy * l_Axy * l_Axy + gupzz * l_Axz * l_Axz + TWO * (gupxy * l_Axx * l_Axy + gupxz * l_Axx * l_Axz + gupyz * l_Axy * l_Axz);
    double term_yy = gupxx * l_Axy * l_Axy + gupyy * l_Ayy * l_Ayy + gupzz * l_Ayz * l_Ayz + TWO * (gupxy * l_Axy * l_Ayy + gupxz * l_Axy * l_Ayz + gupyz * l_Ayy * l_Ayz);
    double term_zz = gupxx * l_Axz * l_Axz + gupyy * l_Ayz * l_Ayz + gupzz * l_Azz * l_Azz + TWO * (gupxy * l_Axz * l_Ayz + gupxz * l_Axz * l_Azz + gupyz * l_Ayz * l_Azz);
    double term_xy = gupxx * l_Axx * l_Axy + gupyy * l_Axy * l_Ayy + gupzz * l_Axz * l_Ayz + gupxy * (l_Axx * l_Ayy + l_Axy * l_Axy) + gupxz * (l_Axx * l_Ayz + l_Axz * l_Axy) + gupyz * (l_Axy * l_Ayz + l_Axz * l_Ayy);
    double term_xz = gupxx * l_Axx * l_Axz + gupyy * l_Axy * l_Ayz + gupzz * l_Axz * l_Azz + gupxy * (l_Axx * l_Ayz + l_Axy * l_Axz) + gupxz * (l_Axx * l_Azz + l_Axz * l_Axz) + gupyz * (l_Axy * l_Azz + l_Axz * l_Ayz);
    double term_yz = gupxx * l_Axy * l_Axz + gupyy * l_Ayy * l_Ayz + gupzz * l_Ayz * l_Azz + gupxy * (l_Axy * l_Ayz + l_Ayy * l_Axz) + gupxz * (l_Axy * l_Azz + l_Ayz * l_Axz) + gupyz * (l_Ayy * l_Azz + l_Ayz * l_Ayz);

    double trA2 = gupxx * term_xx + gupyy * term_yy + gupzz * term_zz + TWO * (gupxy * term_xy + gupxz * term_xz + gupyz * term_yz);

    double f = F2o3 * val_trK * val_trK - trA2 - F16*PI*rho[idx] + EIGHT*PI*S;
    double f_trace = -F1o3 * (trK_rhs_val + alpn1/chin1 * f);

    double src_xx = alpn1 * (l_Rxx - EIGHT*PI*Sxx[idx]) - fxx;
    double src_yy = alpn1 * (l_Ryy - EIGHT*PI*Syy[idx]) - fyy;
    double src_zz = alpn1 * (l_Rzz - EIGHT*PI*Szz[idx]) - fzz;
    double src_xy = alpn1 * (l_Rxy - EIGHT*PI*Sxy[idx]) - fxy;
    double src_xz = alpn1 * (l_Rxz - EIGHT*PI*Sxz[idx]) - fxz;
    double src_yz = alpn1 * (l_Ryz - EIGHT*PI*Syz[idx]) - fyz;

    double Axx_rhs_val = src_xx - l_gxx * f_trace;
    double Ayy_rhs_val = src_yy - l_gyy * f_trace;
    double Azz_rhs_val = src_zz - l_gzz * f_trace;
    double Axy_rhs_val = src_xy - l_gxy * f_trace;
    double Axz_rhs_val = src_xz - l_gxz * f_trace;
    double Ayz_rhs_val = src_yz - l_gyz * f_trace;

    Axx_rhs[idx] = chin1 * Axx_rhs_val + alpn1 * (val_trK * l_Axx - TWO * term_xx) + TWO * (l_Axx * betaxx + l_Axy * betayx + l_Axz * betazx) - F2o3 * l_Axx * div_beta;
    Ayy_rhs[idx] = chin1 * Ayy_rhs_val + alpn1 * (val_trK * l_Ayy - TWO * term_yy) + TWO * (l_Axy * betaxy + l_Ayy * betayy + l_Ayz * betazy) - F2o3 * l_Ayy * div_beta;
    Azz_rhs[idx] = chin1 * Azz_rhs_val + alpn1 * (val_trK * l_Azz - TWO * term_zz) + TWO * (l_Axz * betaxz + l_Ayz * betayz + l_Azz * betazz) - F2o3 * l_Azz * div_beta;
    
    Axy_rhs[idx] = chin1 * Axy_rhs_val + alpn1 * (val_trK * l_Axy - TWO * term_xy) + l_Axx * betaxy + l_Axz * betazy + l_Ayy * betayx + l_Ayz * betazx - l_Axy * betazz + F1o3 * l_Axy * div_beta;
    Ayz_rhs[idx] = chin1 * Ayz_rhs_val + alpn1 * (val_trK * l_Ayz - TWO * term_yz) + l_Axy * betaxz + l_Ayy * betayz + l_Axz * betaxy + l_Azz * betazy - l_Ayz * betaxx + F1o3 * l_Ayz * div_beta;
    Axz_rhs[idx] = chin1 * Axz_rhs_val + alpn1 * (val_trK * l_Axz - TWO * term_xz) + l_Axx * betaxz + l_Axy * betayz + l_Ayz * betayx + l_Azz * betazx - l_Axz * betayy + F1o3 * l_Axz * div_beta;

    trK_rhs[idx] = -chin1 * trK_rhs_val + alpn1 * (F1o3 * val_trK * val_trK + trA2 + FOUR * PI * (rho[idx] + S));

    Lap_rhs[idx] = -TWO * alpn1 * val_trK;
    betax_rhs[idx] = FF * dtSfx[idx];
    betay_rhs[idx] = FF * dtSfy[idx];
    betaz_rhs[idx] = FF * dtSfz[idx];
    dtSfx_rhs[idx] = val_Gamx_rhs - eta * dtSfx[idx];
    dtSfy_rhs[idx] = val_Gamy_rhs - eta * dtSfy[idx];
    dtSfz_rhs[idx] = val_Gamz_rhs - eta * dtSfz[idx];

    Gamx_rhs[idx] = val_Gamx_rhs;
    Gamy_rhs[idx] = val_Gamy_rhs;
    Gamz_rhs[idx] = val_Gamz_rhs;

    Gamxxx_out[idx] = l_Gamxxx; Gamxxy_out[idx] = l_Gamxxy; Gamxxz_out[idx] = l_Gamxxz;
    Gamxyy_out[idx] = l_Gamxyy; Gamxyz_out[idx] = l_Gamxyz; Gamxzz_out[idx] = l_Gamxzz;
    Gamyxx_out[idx] = l_Gamyxx; Gamyxy_out[idx] = l_Gamyxy; Gamyxz_out[idx] = l_Gamyxz;
    Gamyyy_out[idx] = l_Gamyyy; Gamyyz_out[idx] = l_Gamyyz; Gamyzz_out[idx] = l_Gamyzz;
    Gamzxx_out[idx] = l_Gamzxx; Gamzxy_out[idx] = l_Gamzxy; Gamzxz_out[idx] = l_Gamzxz;
    Gamzyy_out[idx] = l_Gamzyy; Gamzyz_out[idx] = l_Gamzyz; Gamzzz_out[idx] = l_Gamzzz;

    Rxx_out[idx] = l_Rxx; Ryy_out[idx] = l_Ryy; Rzz_out[idx] = l_Rzz;
    Rxy_out[idx] = l_Rxy; Rxz_out[idx] = l_Rxz; Ryz_out[idx] = l_Ryz;
}

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

    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    int dims[3] = {ex0, ex1, ex2};
    int idx = IDX3D(i, j, k, ex0, ex1, ex2);

    gxx_rhs[idx] += d_lopsided_point(dims, dxx, gxx_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
    if (eps > 0.0) gxx_rhs[idx] += d_kodis_point(dims, dxx, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    gxy_rhs[idx] += d_lopsided_point(dims, gxy, gxy_rhs, betax, betay, betaz, X, Y, Z, symmetry, ANTI, ANTI, SYM, i, j, k);
    if (eps > 0.0) gxy_rhs[idx] += d_kodis_point(dims, gxy, X, Y, Z, ANTI, ANTI, SYM, symmetry, eps, i, j, k);

    gxz_rhs[idx] += d_lopsided_point(dims, gxz, gxz_rhs, betax, betay, betaz, X, Y, Z, symmetry, ANTI, SYM, ANTI, i, j, k);
    if (eps > 0.0) gxz_rhs[idx] += d_kodis_point(dims, gxz, X, Y, Z, ANTI, SYM, ANTI, symmetry, eps, i, j, k);

    gyy_rhs[idx] += d_lopsided_point(dims, dyy, gyy_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
    if (eps > 0.0) gyy_rhs[idx] += d_kodis_point(dims, dyy, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    gyz_rhs[idx] += d_lopsided_point(dims, gyz, gyz_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, ANTI, ANTI, i, j, k);
    if (eps > 0.0) gyz_rhs[idx] += d_kodis_point(dims, gyz, X, Y, Z, SYM, ANTI, ANTI, symmetry, eps, i, j, k);

    gzz_rhs[idx] += d_lopsided_point(dims, dzz, gzz_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
    if (eps > 0.0) gzz_rhs[idx] += d_kodis_point(dims, dzz, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    Axx_rhs[idx] += d_lopsided_point(dims, Axx, Axx_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
    if (eps > 0.0) Axx_rhs[idx] += d_kodis_point(dims, Axx, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    Axy_rhs[idx] += d_lopsided_point(dims, Axy, Axy_rhs, betax, betay, betaz, X, Y, Z, symmetry, ANTI, ANTI, SYM, i, j, k);
    if (eps > 0.0) Axy_rhs[idx] += d_kodis_point(dims, Axy, X, Y, Z, ANTI, ANTI, SYM, symmetry, eps, i, j, k);

    Axz_rhs[idx] += d_lopsided_point(dims, Axz, Axz_rhs, betax, betay, betaz, X, Y, Z, symmetry, ANTI, SYM, ANTI, i, j, k);
    if (eps > 0.0) Axz_rhs[idx] += d_kodis_point(dims, Axz, X, Y, Z, ANTI, SYM, ANTI, symmetry, eps, i, j, k);

    Ayy_rhs[idx] += d_lopsided_point(dims, Ayy, Ayy_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
    if (eps > 0.0) Ayy_rhs[idx] += d_kodis_point(dims, Ayy, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    Ayz_rhs[idx] += d_lopsided_point(dims, Ayz, Ayz_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, ANTI, ANTI, i, j, k);
    if (eps > 0.0) Ayz_rhs[idx] += d_kodis_point(dims, Ayz, X, Y, Z, SYM, ANTI, ANTI, symmetry, eps, i, j, k);

    Azz_rhs[idx] += d_lopsided_point(dims, Azz, Azz_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
    if (eps > 0.0) Azz_rhs[idx] += d_kodis_point(dims, Azz, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    chi_rhs[idx] += d_lopsided_point(dims, chi, chi_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
    if (eps > 0.0) chi_rhs[idx] += d_kodis_point(dims, chi, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    trK_rhs[idx] += d_lopsided_point(dims, trK, trK_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
    if (eps > 0.0) trK_rhs[idx] += d_kodis_point(dims, trK, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    Gamx_rhs[idx] += d_lopsided_point(dims, Gamx, Gamx_rhs, betax, betay, betaz, X, Y, Z, symmetry, ANTI, SYM, SYM, i, j, k);
    if (eps > 0.0) Gamx_rhs[idx] += d_kodis_point(dims, Gamx, X, Y, Z, ANTI, SYM, SYM, symmetry, eps, i, j, k);

    Gamy_rhs[idx] += d_lopsided_point(dims, Gamy, Gamy_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, ANTI, SYM, i, j, k);
    if (eps > 0.0) Gamy_rhs[idx] += d_kodis_point(dims, Gamy, X, Y, Z, SYM, ANTI, SYM, symmetry, eps, i, j, k);

    Gamz_rhs[idx] += d_lopsided_point(dims, Gamz, Gamz_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, ANTI, i, j, k);
    if (eps > 0.0) Gamz_rhs[idx] += d_kodis_point(dims, Gamz, X, Y, Z, SYM, SYM, ANTI, symmetry, eps, i, j, k);

    Lap_rhs[idx] += d_lopsided_point(dims, Lap, Lap_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
    if (eps > 0.0) Lap_rhs[idx] += d_kodis_point(dims, Lap, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    betax_rhs[idx] += d_lopsided_point(dims, betax, betax_rhs, betax, betay, betaz, X, Y, Z, symmetry, ANTI, SYM, SYM, i, j, k);
    if (eps > 0.0) betax_rhs[idx] += d_kodis_point(dims, betax, X, Y, Z, ANTI, SYM, SYM, symmetry, eps, i, j, k);

    betay_rhs[idx] += d_lopsided_point(dims, betay, betay_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, ANTI, SYM, i, j, k);
    if (eps > 0.0) betay_rhs[idx] += d_kodis_point(dims, betay, X, Y, Z, SYM, ANTI, SYM, symmetry, eps, i, j, k);

    betaz_rhs[idx] += d_lopsided_point(dims, betaz, betaz_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, ANTI, i, j, k);
    if (eps > 0.0) betaz_rhs[idx] += d_kodis_point(dims, betaz, X, Y, Z, SYM, SYM, ANTI, symmetry, eps, i, j, k);

    dtSfx_rhs[idx] += d_lopsided_point(dims, dtSfx, dtSfx_rhs, betax, betay, betaz, X, Y, Z, symmetry, ANTI, SYM, SYM, i, j, k);
    if (eps > 0.0) dtSfx_rhs[idx] += d_kodis_point(dims, dtSfx, X, Y, Z, ANTI, SYM, SYM, symmetry, eps, i, j, k);

    dtSfy_rhs[idx] += d_lopsided_point(dims, dtSfy, dtSfy_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, ANTI, SYM, i, j, k);
    if (eps > 0.0) dtSfy_rhs[idx] += d_kodis_point(dims, dtSfy, X, Y, Z, SYM, ANTI, SYM, symmetry, eps, i, j, k);

    dtSfz_rhs[idx] += d_lopsided_point(dims, dtSfz, dtSfz_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, ANTI, i, j, k);
    if (eps > 0.0) dtSfz_rhs[idx] += d_kodis_point(dims, dtSfz, X, Y, Z, SYM, SYM, ANTI, symmetry, eps, i, j, k);
}

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
    double* __restrict__ ham_Res, 
    double* __restrict__ movx_Res, double* __restrict__ movy_Res, double* __restrict__ movz_Res,
    double* __restrict__ Gmx_Res, double* __restrict__ Gmy_Res, double* __restrict__ Gmz_Res
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    int dims[3] = {ex0, ex1, ex2};

    double val_chi = chi[idx]; double chin1 = val_chi + ONE;
    double val_trK = trK[idx];

    double l_gxx = dxx[idx] + ONE; double l_gxy = gxy[idx]; double l_gxz = gxz[idx];
    double l_gyy = dyy[idx] + ONE; double l_gyz = gyz[idx]; double l_gzz = dzz[idx] + ONE;
    double l_Axx = Axx[idx]; double l_Axy = Axy[idx]; double l_Axz = Axz[idx];
    double l_Ayy = Ayy[idx]; double l_Ayz = Ayz[idx]; double l_Azz = Azz[idx];

    double gupzz = l_gxx * l_gyy * l_gzz + l_gxy * l_gyz * l_gxz + l_gxz * l_gxy * l_gyz - l_gxz * l_gyy * l_gxz - l_gxy * l_gxy * l_gzz - l_gxx * l_gyz * l_gyz;
    double gupxx = (l_gyy * l_gzz - l_gyz * l_gyz) / gupzz;
    double gupxy = -(l_gxy * l_gzz - l_gyz * l_gxz) / gupzz;
    double gupxz = (l_gxy * l_gyz - l_gyy * l_gxz) / gupzz;
    double gupyy = (l_gxx * l_gzz - l_gxz * l_gxz) / gupzz;
    double gupyz = -(l_gxx * l_gyz - l_gxy * l_gxz) / gupzz;
    gupzz = (l_gxx * l_gyy - l_gxy * l_gxy) / gupzz;

    double gxxx, gxxy, gxxz, gxyx, gxyy, gxyz, gxzx, gxzy, gxzz;
    double gyyx, gyyy, gyyz, gyzx, gyzy, gyzz, gzzx, gzzy, gzzz;
    d_fderivs_point(dims, dxx, &gxxx, &gxxy, &gxxz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, gxy, &gxyx, &gxyy, &gxyz, X, Y, Z, ANTI, ANTI, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, gxz, &gxzx, &gxzy, &gxzz, X, Y, Z, ANTI, SYM, ANTI, symmetry, lev, i, j, k);
    d_fderivs_point(dims, dyy, &gyyx, &gyyy, &gyyz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, gyz, &gyzx, &gyzy, &gyzz, X, Y, Z, SYM, ANTI, ANTI, symmetry, lev, i, j, k);
    d_fderivs_point(dims, dzz, &gzzx, &gzzy, &gzzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    double term_x = gupxx*(gupxx*gxxx+gupxy*gxyx+gupxz*gxzx) + gupxy*(gupxx*gxyx+gupxy*gyyx+gupxz*gyzx) + gupxz*(gupxx*gxzx+gupxy*gyzx+gupxz*gzzx) + gupxx*(gupxy*gxxy+gupyy*gxyy+gupyz*gxzy) + gupxy*(gupxy*gxyy+gupyy*gyyy+gupyz*gyzy) + gupxz*(gupxy*gxzy+gupyy*gyzy+gupyz*gzzy) + gupxx*(gupxz*gxxz+gupyz*gxyz+gupzz*gxzz) + gupxy*(gupxz*gxyz+gupyz*gyyz+gupzz*gyzz) + gupxz*(gupxz*gxzz+gupyz*gyzz+gupzz*gzzz);
    double term_y = gupxx*(gupxy*gxxx+gupyy*gxyx+gupyz*gxzx) + gupxy*(gupxy*gxyx+gupyy*gyyx+gupyz*gyzx) + gupxz*(gupxy*gxzx+gupyy*gyzx+gupyz*gzzx) + gupxy*(gupxy*gxxy+gupyy*gxyy+gupyz*gxzy) + gupyy*(gupxy*gxyy+gupyy*gyyy+gupyz*gyzy) + gupyz*(gupxy*gxzy+gupyy*gyzy+gupyz*gzzy) + gupxy*(gupxz*gxxz+gupyz*gxyz+gupzz*gxzz) + gupyy*(gupxz*gxyz+gupyz*gyyz+gupzz*gyzz) + gupyz*(gupxz*gxzz+gupyz*gyzz+gupzz*gzzz);
    double term_z = gupxx*(gupxz*gxxx+gupyz*gxyx+gupzz*gxzx) + gupxy*(gupxz*gxyx+gupyz*gyyx+gupzz*gyzx) + gupxz*(gupxz*gxzx+gupyz*gyzx+gupzz*gzzx) + gupxy*(gupxz*gxxy+gupyz*gxyy+gupzz*gxzy) + gupyy*(gupxz*gxyy+gupyz*gyyy+gupzz*gyzy) + gupyz*(gupxz*gxzy+gupyz*gyzy+gupzz*gzzy) + gupxz*(gupxz*gxxz+gupyz*gxyz+gupzz*gxzz) + gupyz*(gupxz*gxyz+gupyz*gyyz+gupzz*gyzz) + gupzz*(gupxz*gxzz+gupyz*gyzz+gupzz*gzzz);
    Gmx_Res[idx] = Gamx[idx] - term_x;
    Gmy_Res[idx] = Gamy[idx] - term_y;
    Gmz_Res[idx] = Gamz[idx] - term_z;

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

    double dGamxx, dGamxy, dGamxz, dGamyx, dGamyy, dGamyz, dGamzx, dGamzy, dGamzz;
    d_fderivs_point(dims, Gamx, &dGamxx, &dGamxy, &dGamxz, X, Y, Z, ANTI, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Gamy, &dGamyx, &dGamyy, &dGamyz, X, Y, Z, SYM, ANTI, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Gamz, &dGamzx, &dGamzy, &dGamzz, X, Y, Z, SYM, SYM, ANTI, symmetry, lev, i, j, k);

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

    double fxx, fxy, fxz, fyy, fyz, fzz;
    double l_Rxx, l_Rxy, l_Rxz, l_Ryy, l_Ryz, l_Rzz;

    d_fdderivs_point(dims, dxx, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    l_Rxx = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;
    d_fdderivs_point(dims, dyy, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    l_Ryy = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;
    d_fdderivs_point(dims, dzz, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    l_Rzz = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;
    d_fdderivs_point(dims, gxy, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, ANTI, ANTI, SYM, symmetry, lev, i, j, k);
    l_Rxy = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;
    d_fdderivs_point(dims, gxz, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, ANTI, SYM, ANTI, symmetry, lev, i, j, k);
    l_Rxz = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;
    d_fdderivs_point(dims, gyz, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, ANTI, ANTI, symmetry, lev, i, j, k);
    l_Ryz = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;

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

    double chix, chiy, chiz;
    d_fderivs_point(dims, chi, &chix, &chiy, &chiz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fdderivs_point(dims, chi, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    
    fxx -= l_Gamxxx * chix + l_Gamyxx * chiy + l_Gamzxx * chiz;
    fxy -= l_Gamxxy * chix + l_Gamyxy * chiy + l_Gamzxy * chiz;
    fxz -= l_Gamxxz * chix + l_Gamyxz * chiy + l_Gamzxz * chiz;
    fyy -= l_Gamxyy * chix + l_Gamyyy * chiy + l_Gamzyy * chiz;
    fyz -= l_Gamxyz * chix + l_Gamyyz * chiy + l_Gamzyz * chiz;
    fzz -= l_Gamxzz * chix + l_Gamyzz * chiy + l_Gamzzz * chiz;

    double f_scalar = gupxx * (fxx - F3o2/chin1 * chix * chix) + gupyy * (fyy - F3o2/chin1 * chiy * chiy) + gupzz * (fzz - F3o2/chin1 * chiz * chiz) + 
                      TWO * (gupxy * (fxy - F3o2/chin1 * chix * chiy) + gupxz * (fxz - F3o2/chin1 * chix * chiz) + gupyz * (fyz - F3o2/chin1 * chiy * chiz));
    
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
    
    l_Gamxxx -= ((chix + chix)/chin1 - l_gxx * gx_phy)*HALF; 
    l_Gamyxx -= (                            - l_gxx * gy_phy)*HALF; 
    l_Gamzxx -= (                            - l_gxx * gz_phy)*HALF; 
    l_Gamxyy -= (                            - l_gyy * gx_phy)*HALF; 
    l_Gamyyy -= ((chiy + chiy)/chin1 - l_gyy * gy_phy)*HALF; 
    l_Gamzyy -= (                            - l_gyy * gz_phy)*HALF; 
    l_Gamxzz -= (                            - l_gzz * gx_phy)*HALF; 
    l_Gamyzz -= (                            - l_gzz * gy_phy)*HALF; 
    l_Gamzzz -= ((chiz + chiz)/chin1 - l_gzz * gz_phy)*HALF; 

    l_Gamxxy -= (chiy/chin1 - l_gxy * gx_phy)*HALF; 
    l_Gamyxy -= (chix/chin1 - l_gxy * gy_phy)*HALF; 
    l_Gamzxy -= (                 - l_gxy * gz_phy)*HALF; 
    l_Gamxxz -= (chiz/chin1 - l_gxz * gx_phy)*HALF; 
    l_Gamyxz -= (                 - l_gxz * gy_phy)*HALF; 
    l_Gamzxz -= (chix/chin1 - l_gxz * gz_phy)*HALF; 
    l_Gamxyz -= (                 - l_gyz * gx_phy)*HALF; 
    l_Gamyyz -= (chiz/chin1 - l_gyz * gy_phy)*HALF; 
    l_Gamzyz -= (chiy/chin1 - l_gyz * gz_phy)*HALF;

    double Kx, Ky, Kz;
    d_fderivs_point(dims, trK, &Kx, &Ky, &Kz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

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

    movx_Res[idx] = movx_Res[idx] - F2o3 * Kx - F8 * PI * Sx[idx];
    movy_Res[idx] = movy_Res[idx] - F2o3 * Ky - F8 * PI * Sy[idx];
    movz_Res[idx] = movz_Res[idx] - F2o3 * Kz - F8 * PI * Sz[idx];
}

void gpu_compute_rhs_bssn_launch_opt(
    cudaStream_t &stream, int* ex, double T, double* X, double* Y, double* Z,
    double* chi, double* trK,
    double* dxx, double* gxy, double* gxz, double* dyy, double* gyz, double* dzz,
    double* Axx, double* Axy, double* Axz, double* Ayy, double* Ayz, double* Azz,
    double* Gamx, double* Gamy, double* Gamz, double* Lap,
    double* betax, double* betay, double* betaz,
    double* dtSfx, double* dtSfy, double* dtSfz,
    // RHS 输出
    double* chi_rhs, double* trK_rhs,
    double* gxx_rhs, double* gxy_rhs, double* gxz_rhs, double* gyy_rhs, double* gyz_rhs, double* gzz_rhs,
    double* Axx_rhs, double* Axy_rhs, double* Axz_rhs, double* Ayy_rhs, double* Ayz_rhs, double* Azz_rhs,
    double* Gamx_rhs, double* Gamy_rhs, double* Gamz_rhs, double* Lap_rhs,
    double* betax_rhs, double* betay_rhs, double* betaz_rhs,
    double* dtSfx_rhs, double* dtSfy_rhs, double* dtSfz_rhs,
    // 物理源项
    double* rho, double* Sx, double* Sy, double* Sz,
    double* Sxx, double* Sxy, double* Sxz, double* Syy, double* Syz, double* Szz,
    // 连接系数输出 (Gam)
    double* Gamxxx, double* Gamxxy, double* Gamxxz, double* Gamxyy, double* Gamxyz, double* Gamxzz,
    double* Gamyxx, double* Gamyxy, double* Gamyxz, double* Gamyyy, double* Gamyyz, double* Gamyzz,
    double* Gamzxx, double* Gamzxy, double* Gamzxz, double* Gamzyy, double* Gamzyz, double* Gamzzz,
    // Ricci Tensor 输出
    double* Rxx, double* Rxy, double* Rxz, double* Ryy, double* Ryz, double* Rzz,
    // 约束输出
    double* ham_Res, double* movx_Res, double* movy_Res, double* movz_Res,
    double* Gmx_Res, double* Gmy_Res, double* Gmz_Res,
    // 配置参数
    int symmetry, int lev, double eps, int co
) {
    dim3 block(8, 8, 4);
    dim3 grid(
        (ex[0] + block.x - 1) / block.x,
        (ex[1] + block.y - 1) / block.y,
        (ex[2] + block.z - 1) / block.z
    );

    // ==========================================
    // 串联 Launch 1: 核心代数与李导数 (覆盖写入 = )
    // ==========================================
    bssn_core_rhs_kernel<<<grid, block, 0, stream>>>(
        ex[0], ex[1], ex[2], symmetry, lev, X, Y, Z,
        chi, trK, dxx, gxy, gxz, dyy, gyz, dzz,
        Axx, Axy, Axz, Ayy, Ayz, Azz,
        Gamx, Gamy, Gamz, Lap, betax, betay, betaz, dtSfx, dtSfy, dtSfz,
        rho, Sx, Sy, Sz, Sxx, Sxy, Sxz, Syy, Syz, Szz,
        chi_rhs, trK_rhs, gxx_rhs, gxy_rhs, gxz_rhs, gyy_rhs, gyz_rhs, gzz_rhs,
        Axx_rhs, Axy_rhs, Axz_rhs, Ayy_rhs, Ayz_rhs, Azz_rhs,
        Gamx_rhs, Gamy_rhs, Gamz_rhs, Lap_rhs, betax_rhs, betay_rhs, betaz_rhs,
        dtSfx_rhs, dtSfy_rhs, dtSfz_rhs,
        Gamxxx, Gamxxy, Gamxxz, Gamxyy, Gamxyz, Gamxzz,
        Gamyxx, Gamyxy, Gamyxz, Gamyyy, Gamyyz, Gamyzz,
        Gamzxx, Gamzxy, Gamzxz, Gamzyy, Gamzyz, Gamzzz,
        Rxx, Rxy, Rxz, Ryy, Ryz, Rzz
    );

    // ==========================================
    // 串联 Launch 2: 迎风平流与耗散 (累加更新 += )
    // ==========================================
    bssn_advection_dissipation_kernel<<<grid, block, 0, stream>>>(
        ex[0], ex[1], ex[2], symmetry, lev, eps, X, Y, Z,
        betax, betay, betaz, dxx, gxy, gxz, dyy, gyz, dzz,
        Axx, Axy, Axz, Ayy, Ayz, Azz, chi, trK, Gamx, Gamy, Gamz, Lap, dtSfx, dtSfy, dtSfz,
        gxx_rhs, gxy_rhs, gxz_rhs, gyy_rhs, gyz_rhs, gzz_rhs,
        Axx_rhs, Axy_rhs, Axz_rhs, Ayy_rhs, Ayz_rhs, Azz_rhs,
        chi_rhs, trK_rhs, Gamx_rhs, Gamy_rhs, Gamz_rhs, Lap_rhs,
        betax_rhs, betay_rhs, betaz_rhs, dtSfx_rhs, dtSfy_rhs, dtSfz_rhs
    );

    // ==========================================
    // 串联 Launch 3: 约束诊断
    // ==========================================
    if (co == 0) {
        bssn_constraints_kernel<<<grid, block, 0, stream>>>(
            ex[0], ex[1], ex[2], symmetry, lev, X, Y, Z,
            chi, trK, dxx, gxy, gxz, dyy, gyz, dzz,
            Axx, Axy, Axz, Ayy, Ayz, Azz, Gamx, Gamy, Gamz, rho, Sx, Sy, Sz,
            ham_Res, movx_Res, movy_Res, movz_Res,
            Gmx_Res, Gmy_Res, Gmz_Res // <-- 传递新增指针
        );
    }
}