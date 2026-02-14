
#ifndef BSSN_GPU_H_
#define BSSN_GPU_H_
#include "bssn_macro.h"
#include "macrodef.fh"

#define DEVICE_ID 0
// #define DEVICE_ID_BY_MPI_RANK
#define RHS_GRID_DIM 256
#define RHS_BLOCK_DIM 128

#define _FH2_(i, j, k) fh[(i) + (j) * _1D_SIZE[2] + (k) * _2D_SIZE[2]]
#define _FH3_(i, j, k) fh[(i) + (j) * _1D_SIZE[3] + (k) * _2D_SIZE[3]]
#define pow2(x) ((x) * (x))
#define TimeBetween(a, b) ((b.tv_sec - a.tv_sec) + (b.tv_usec - a.tv_usec) / 1000000.0f)
#define M_ metac.
#define Mh_ meta->
#define Ms_ metassc.
#define Msh_ metass->

// #define TIMING

#define RHS_SS_PARA int calledby, int mpi_rank, int *ex, double &T, double *crho, double *sigma, double *R, double *X, double *Y, double *Z, double *drhodx, double *drhody, double *drhodz, double *dsigmadx, double *dsigmady, double *dsigmadz, double *dRdx, double *dRdy, double *dRdz, double *drhodxx, double *drhodxy, double *drhodxz, double *drhodyy, double *drhodyz, double *drhodzz, double *dsigmadxx, double *dsigmadxy, double *dsigmadxz, double *dsigmadyy, double *dsigmadyz, double *dsigmadzz, double *dRdxx, double *dRdxy, double *dRdxz, double *dRdyy, double *dRdyz, double *dRdzz, double *chi, double *trK, double *dxx, double *gxy, double *gxz, double *dyy, double *gyz, double *dzz, double *Axx, double *Axy, double *Axz, double *Ayy, double *Ayz, double *Azz, double *Gamx, double *Gamy, double *Gamz, double *Lap, double *betax, double *betay, double *betaz, double *dtSfx, double *dtSfy, double *dtSfz, double *chi_rhs, double *trK_rhs, double *gxx_rhs, double *gxy_rhs, double *gxz_rhs, double *gyy_rhs, double *gyz_rhs, double *gzz_rhs, double *Axx_rhs, double *Axy_rhs, double *Axz_rhs, double *Ayy_rhs, double *Ayz_rhs, double *Azz_rhs, double *Gamx_rhs, double *Gamy_rhs, double *Gamz_rhs, double *Lap_rhs, double *betax_rhs, double *betay_rhs, double *betaz_rhs, double *dtSfx_rhs, double *dtSfy_rhs, double *dtSfz_rhs, double *rho, double *Sx, double *Sy, double *Sz, double *Sxx, double *Sxy, double *Sxz, double *Syy, double *Syz, double *Szz, double *Gamxxx, double *Gamxxy, double *Gamxxz, double *Gamxyy, double *Gamxyz, double *Gamxzz, double *Gamyxx, double *Gamyxy, double *Gamyxz, double *Gamyyy, double *Gamyyz, double *Gamyzz, double *Gamzxx, double *Gamzxy, double *Gamzxz, double *Gamzyy, double *Gamzyz, double *Gamzzz, double *Rxx, double *Rxy, double *Rxz, double *Ryy, double *Ryz, double *Rzz, double *ham_Res, double *movx_Res, double *movy_Res, double *movz_Res, double *Gmx_Res, double *Gmy_Res, double *Gmz_Res, int &Symmetry, int &Lev, double &eps, int &sst, int &co

struct GPU_RHS_CONTEXT {
    int calledby, mpi_rank, *ex;
    double &T, *X, *Y, *Z;
    double *chi, *trK;
    double *dxx , *gxy , *gxz , *dyy, *gyz, *dzz;
    double *Axx , *Axy , *Axz , *Ayy , *Ayz , *Azz;
    double *Gamx , *Gamy , *Gamz;
    double *Lap , *betax , *betay , *betaz;
    double *dtSfx, *dtSfy , *dtSfz;
    double *chi_rhs, *trK_rhs;
    double *gxx_rhs, *gxy_rhs, *gxz_rhs, *gyy_rhs, *gyz_rhs, *gzz_rhs;
    double *Axx_rhs, *Axy_rhs, *Axz_rhs, *Ayy_rhs, *Ayz_rhs, *Azz_rhs;
    double *Gamx_rhs, *Gamy_rhs, *Gamz_rhs;
    double *Lap_rhs, *betax_rhs, *betay_rhs, *betaz_rhs;
    double *dtSfx_rhs, *dtSfy_rhs, *dtSfz_rhs;
    double *rho, *Sx, *Sy, *Sz, *Sxx;
    double *Sxy, *Sxz, *Syy, *Syz, *Szz;
    double *Gamxxx, *Gamxxy, *Gamxxz, *Gamxyy, *Gamxyz, *Gamxzz;
    double *Gamyxx, *Gamyxy, *Gamyxz, *Gamyyy, *Gamyyz, *Gamyzz;
    double *Gamzxx, *Gamzxy, *Gamzxz, *Gamzyy, *Gamzyz, *Gamzzz;
    double *Rxx, *Rxy, *Rxz, *Ryy, *Ryz, *Rzz;
    double *ham_Res, *movx_Res, *movy_Res, *movz_Res;
    double *Gmx_Res, *Gmy_Res, *Gmz_Res;
    int &Symmetry, &Lev;
    double &eps;
    int &co;

    GPU_RHS_CONTEXT (
        int calledby_, int mpi_rank_, int *ex_,
        double &T_, double *X_, double *Y_, double *Z_,
        double *chi_, double *trK_,
        double *dxx_, double *gxy_, double *gxz_, double *dyy_, double *gyz_, double *dzz_,
        double *Axx_, double *Axy_, double *Axz_, double *Ayy_, double *Ayz_, double *Azz_,
        double *Gamx_, double *Gamy_, double *Gamz_,
        double *Lap_, double *betax_, double *betay_, double *betaz_,
        double *dtSfx_, double *dtSfy_, double *dtSfz_,
        double *chi_rhs_, double *trK_rhs_,
        double *gxx_rhs_, double *gxy_rhs_, double *gxz_rhs_, double *gyy_rhs_, double *gyz_rhs_, double *gzz_rhs_,
        double *Axx_rhs_, double *Axy_rhs_, double *Axz_rhs_, double *Ayy_rhs_, double *Ayz_rhs_, double *Azz_rhs_,
        double *Gamx_rhs_, double *Gamy_rhs_, double *Gamz_rhs_,
        double *Lap_rhs_, double *betax_rhs_, double *betay_rhs_, double *betaz_rhs_,
        double *dtSfx_rhs_, double *dtSfy_rhs_, double *dtSfz_rhs_,
        double *rho_, double *Sx_, double *Sy_, double *Sz_, double *Sxx_,
        double *Sxy_, double *Sxz_, double *Syy_, double *Syz_, double *Szz_,
        double *Gamxxx_, double *Gamxxy_, double *Gamxxz_, double *Gamxyy_, double *Gamxyz_, double *Gamxzz_,
        double *Gamyxx_, double *Gamyxy_, double *Gamyxz_, double *Gamyyy_, double *Gamyyz_, double *Gamyzz_,
        double *Gamzxx_, double *Gamzxy_, double *Gamzxz_, double *Gamzyy_, double *Gamzyz_, double *Gamzzz_,
        double *Rxx_, double *Rxy_, double *Rxz_, double *Ryy_, double *Ryz_, double *Rzz_,
        double *ham_Res_, double *movx_Res_, double *movy_Res_, double *movz_Res_,
        double *Gmx_Res_, double *Gmy_Res_, double *Gmz_Res_,
        int &Symmetry_, int &Lev_, double &eps_, int &co_)
        : calledby(calledby_), mpi_rank(mpi_rank_), ex(ex_), T(T_), X(X_), Y(Y_), Z(Z_),
          chi(chi_), trK(trK_),
          dxx(dxx_), gxy(gxy_), gxz(gxz_), dyy(dyy_), gyz(gyz_), dzz(dzz_),
          Axx(Axx_), Axy(Axy_), Axz(Axz_), Ayy(Ayy_), Ayz(Ayz_), Azz(Azz_),
          Gamx(Gamx_), Gamy(Gamy_), Gamz(Gamz_),
          Lap(Lap_), betax(betax_), betay(betay_), betaz(betaz_),
          dtSfx(dtSfx_), dtSfy(dtSfy_), dtSfz(dtSfz_),
          chi_rhs(chi_rhs_), trK_rhs(trK_rhs_),
          gxx_rhs(gxx_rhs_), gxy_rhs(gxy_rhs_), gxz_rhs(gxz_rhs_), gyy_rhs(gyy_rhs_), gyz_rhs(gyz_rhs_), gzz_rhs(gzz_rhs_),
          Axx_rhs(Axx_rhs_), Axy_rhs(Axy_rhs_), Axz_rhs(Axz_rhs_), Ayy_rhs(Ayy_rhs_), Ayz_rhs(Ayz_rhs_), Azz_rhs(Azz_rhs_),
          Gamx_rhs(Gamx_rhs_), Gamy_rhs(Gamy_rhs_), Gamz_rhs(Gamz_rhs_),
          Lap_rhs(Lap_rhs_), betax_rhs(betax_rhs_), betay_rhs(betay_rhs_), betaz_rhs(betaz_rhs_),
          dtSfx_rhs(dtSfx_rhs_), dtSfy_rhs(dtSfy_rhs_), dtSfz_rhs(dtSfz_rhs_),
          rho(rho_), Sx(Sx_), Sy(Sy_), Sz(Sz_), Sxx(Sxx_),
          Sxy(Sxy_), Sxz(Sxz_), Syy(Syy_), Syz(Syz_), Szz(Szz_),
          Gamxxx(Gamxxx_), Gamxxy(Gamxxy_), Gamxxz(Gamxxz_), Gamxyy(Gamxyy_), Gamxyz(Gamxyz_), Gamxzz(Gamxzz_),
          Gamyxx(Gamyxx_), Gamyxy(Gamyxy_), Gamyxz(Gamyxz_), Gamyyy(Gamyyy_), Gamyyz(Gamyyz_), Gamyzz(Gamyzz_),
          Gamzxx(Gamzxx_), Gamzxy(Gamzxy_), Gamzxz(Gamzxz_), Gamzyy(Gamzyy_), Gamzyz(Gamzyz_), Gamzzz(Gamzzz_),
          Rxx(Rxx_), Rxy(Rxy_), Rxz(Rxz_), Ryy(Ryy_), Ryz(Ryz_), Rzz(Rzz_),
          ham_Res(ham_Res_), movx_Res(movx_Res_), movy_Res(movy_Res_), movz_Res(movz_Res_),
          Gmx_Res(Gmx_Res_), Gmy_Res(Gmy_Res_), Gmz_Res(Gmz_Res_),
          Symmetry(Symmetry_), Lev(Lev_), eps(eps_), co(co_) {}
};

#include "gpu_rhs_mem.h"

Meta * gpu_get_meta();
void gpu_destroy_meta();
void gpu_init_meta(GPU_RHS_CONTEXT &ctx);
void gpu_to_device(GPU_RHS_CONTEXT &ctx);
void gpu_init_constant(GPU_RHS_CONTEXT &ctx);
void gpu_back_to_host(GPU_RHS_CONTEXT &ctx);

/**  main function */
int gpu_rhs(GPU_RHS_CONTEXT &ctx);

int gpu_rhs_ss(RHS_SS_PARA);

/** Init GPU side data in GPUMeta. */
// void init_fluid_meta_gpu(GPUMeta *gpu_meta);

#endif
