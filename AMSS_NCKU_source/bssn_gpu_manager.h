#ifndef BSSN_GPU_MANAGER_H
#define BSSN_GPU_MANAGER_H

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// 简单的 CUDA 错误检查宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

class BssnCudaManager {
public:
    BssnCudaManager();
    ~BssnCudaManager();

    // 初始化/调整显存大小 (如果 ex 改变，重新分配)
    void resize(const int ex[3]);

    // 主调用接口 (对应 Fortran 的 compute_rhs_bssn)
    void compute_rhs_bssn(
        const int ex[3], double T, 
        const double* X, const double* Y, const double* Z,
        // 输入/输出变量
        const double* chi, const double* trK,
        const double* dxx, const double* gxy, const double* gxz, 
        const double* dyy, const double* gyz, const double* dzz,
        const double* Axx, const double* Axy, const double* Axz, 
        const double* Ayy, const double* Ayz, const double* Azz,
        const double* Gamx, const double* Gamy, const double* Gamz,
        const double* Lap, 
        const double* betax, const double* betay, const double* betaz,
        const double* dtSfx, const double* dtSfy, const double* dtSfz,
        // RHS 输出
        double* chi_rhs, double* trK_rhs,
        double* gxx_rhs, double* gxy_rhs, double* gxz_rhs,
        double* gyy_rhs, double* gyz_rhs, double* gzz_rhs,
        double* Axx_rhs, double* Axy_rhs, double* Axz_rhs,
        double* Ayy_rhs, double* Ayz_rhs, double* Azz_rhs,
        double* Gamx_rhs, double* Gamy_rhs, double* Gamz_rhs,
        double* Lap_rhs, 
        double* betax_rhs, double* betay_rhs, double* betaz_rhs,
        double* dtSfx_rhs, double* dtSfy_rhs, double* dtSfz_rhs,
        // 源项
        const double* rho, const double* Sx, const double* Sy, const double* Sz,
        const double* Sxx, const double* Sxy, const double* Sxz,
        const double* Syy, const double* Syz, const double* Szz,
        // 输出诊断变量 (物理 Gam, Ricci, Constraints)
        double* Gamxxx, double* Gamxxy, double* Gamxxz,
        double* Gamxyy, double* Gamxyz, double* Gamxzz,
        double* Gamyxx, double* Gamyxy, double* Gamyxz,
        double* Gamyyy, double* Gamyyz, double* Gamyzz,
        double* Gamzxx, double* Gamzxy, double* Gamzxz,
        double* Gamzyy, double* Gamzyz, double* Gamzzz,
        double* Rxx, double* Rxy, double* Rxz, 
        double* Ryy, double* Ryz, double* Rzz,
        double* ham_Res, double* movx_Res, double* movy_Res, double* movz_Res,
        double* Gmx_Res, double* Gmy_Res, double* Gmz_Res,
        // 标量参数
        int symmetry, int lev, double eps, int co
    );

private:
    // 内部 helper
    void allocate_intermediates(size_t size);
    
    void copy_inputs_to_device(
        const double* chi, const double* trK,
        const double* dxx, const double* gxy, const double* gxz, 
        const double* dyy, const double* gyz, const double* dzz,
        const double* Axx, const double* Axy, const double* Axz, 
        const double* Ayy, const double* Ayz, const double* Azz,
        const double* Gamx, const double* Gamy, const double* Gamz,
        const double* Lap, 
        const double* betax, const double* betay, const double* betaz,
        const double* dtSfx, const double* dtSfy, const double* dtSfz,
        const double* rho, const double* Sx, const double* Sy, const double* Sz,
        const double* Sxx, const double* Sxy, const double* Sxz, 
        const double* Syy, const double* Syz, const double* Szz,
        const double* X, const double* Y, const double* Z
    );

    void run_kernels(int ex[3], int symmetry, int lev, double eps, int co);

    void copy_outputs_to_host(
        double* chi_rhs, double* trK_rhs,
        double* gxx_rhs, double* gxy_rhs, double* gxz_rhs,
        double* gyy_rhs, double* gyz_rhs, double* gzz_rhs,
        double* Axx_rhs, double* Axy_rhs, double* Axz_rhs,
        double* Ayy_rhs, double* Ayz_rhs, double* Azz_rhs,
        double* Gamx_rhs, double* Gamy_rhs, double* Gamz_rhs,
        double* Lap_rhs, 
        double* betax_rhs, double* betay_rhs, double* betaz_rhs,
        double* dtSfx_rhs, double* dtSfy_rhs, double* dtSfz_rhs,
        double* Gamxxx, double* Gamxxy, double* Gamxxz,
        double* Gamxyy, double* Gamxyz, double* Gamxzz,
        double* Gamyxx, double* Gamyxy, double* Gamyxz,
        double* Gamyyy, double* Gamyyz, double* Gamyzz,
        double* Gamzxx, double* Gamzxy, double* Gamzxz,
        double* Gamzyy, double* Gamzyz, double* Gamzzz,
        double* Rxx, double* Rxy, double* Rxz, 
        double* Ryy, double* Ryz, double* Rzz,
        double* ham_Res, double* movx_Res, double* movy_Res, double* movz_Res,
        double* Gmx_Res, double* Gmy_Res, double* Gmz_Res,
        int co
    );

    // 状态变量
    int current_ex[3] = {0, 0, 0};
    size_t current_grid_size = 0;
    cudaStream_t stream;

    // ==========================================
    // Device 指针 (GPU 显存)
    // ==========================================
    
    // 1. 输入变量 (Device Copy)
    double *d_X, *d_Y, *d_Z;
    double *d_chi, *d_trK;
    double *d_dxx, *d_gxy, *d_gxz, *d_dyy, *d_gyz, *d_dzz;
    double *d_Axx, *d_Axy, *d_Axz, *d_Ayy, *d_Ayz, *d_Azz;
    double *d_Gamx, *d_Gamy, *d_Gamz;
    double *d_Lap;
    double *d_betax, *d_betay, *d_betaz;
    double *d_dtSfx, *d_dtSfy, *d_dtSfz;
    double *d_rho, *d_Sx, *d_Sy, *d_Sz;
    double *d_Sxx, *d_Sxy, *d_Sxz, *d_Syy, *d_Syz, *d_Szz;

    // 2. 输出 RHS 变量 (Device Copy)
    double *d_chi_rhs, *d_trK_rhs;
    double *d_gxx_rhs, *d_gxy_rhs, *d_gxz_rhs, *d_gyy_rhs, *d_gyz_rhs, *d_gzz_rhs;
    double *d_Axx_rhs, *d_Axy_rhs, *d_Axz_rhs, *d_Ayy_rhs, *d_Ayz_rhs, *d_Azz_rhs;
    double *d_Gamx_rhs, *d_Gamy_rhs, *d_Gamz_rhs;
    double *d_Lap_rhs;
    double *d_betax_rhs, *d_betay_rhs, *d_betaz_rhs;
    double *d_dtSfx_rhs, *d_dtSfy_rhs, *d_dtSfz_rhs;

    // 3. 诊断/中间输出变量 (Device Copy)
    double *d_Gamxxx, *d_Gamxxy, *d_Gamxxz, *d_Gamxyy, *d_Gamxyz, *d_Gamxzz;
    double *d_Gamyxx, *d_Gamyxy, *d_Gamyxz, *d_Gamyyy, *d_Gamyyz, *d_Gamyzz;
    double *d_Gamzxx, *d_Gamzxy, *d_Gamzxz, *d_Gamzyy, *d_Gamzyz, *d_Gamzzz;
    double *d_Rxx, *d_Rxy, *d_Rxz, *d_Ryy, *d_Ryz, *d_Rzz;
    double *d_ham_Res, *d_movx_Res, *d_movy_Res, *d_movz_Res;
    double *d_Gmx_Res, *d_Gmy_Res, *d_Gmz_Res;

    // 4. *** 纯中间变量 (Intermediate Buffers) *** // Kernel 1 -> Kernel 2/4 传递
    double *d_chix, *d_chiy, *d_chiz;
    double *d_betaxx, *d_betaxy, *d_betaxz, *d_betayx, *d_betayy, *d_betayz, *d_betazx, *d_betazy, *d_betazz;
    double *d_gxxx, *d_gxxy, *d_gxxz, *d_gxyx, *d_gxyy, *d_gxyz;
    double *d_gxzx, *d_gxzy, *d_gxzz, *d_gyyx, *d_gyyy, *d_gyyz;
    double *d_gyzx, *d_gyzy, *d_gyzz, *d_gzzx, *d_gzzy, *d_gzzz;
    double *d_gupxx, *d_gupxy, *d_gupxz, *d_gupyy, *d_gupyz, *d_gupzz;
};

bool gpu_compute_rhs_bssn(
    int* ex, double T, double* X, double* Y, double* Z,
    double* chi, double* trK,
    double* dxx, double* gxy, double* gxz,
    double* dyy, double* gyz, double* dzz,
    double* Axx, double* Axy, double* Axz,
    double* Ayy, double* Ayz, double* Azz,
    double* Gamx, double* Gamy, double* Gamz,
    double* Lap,
    double* betax, double* betay, double* betaz,
    double* dtSfx, double* dtSfy, double* dtSfz,
    double* chi_rhs, double* trK_rhs,
    double* gxx_rhs, double* gxy_rhs, double* gxz_rhs,
    double* gyy_rhs, double* gyz_rhs, double* gzz_rhs,
    double* Axx_rhs, double* Axy_rhs, double* Axz_rhs,
    double* Ayy_rhs, double* Ayz_rhs, double* Azz_rhs,
    double* Gamx_rhs, double* Gamy_rhs, double* Gamz_rhs,
    double* Lap_rhs,
    double* betax_rhs, double* betay_rhs, double* betaz_rhs,
    double* dtSfx_rhs, double* dtSfy_rhs, double* dtSfz_rhs,
    double* rho, double* Sx, double* Sy, double* Sz,
    double* Sxx, double* Sxy, double* Sxz,
    double* Syy, double* Syz, double* Szz,
    double* Gamxxx, double* Gamxxy, double* Gamxxz,
    double* Gamxyy, double* Gamxyz, double* Gamxzz,
    double* Gamyxx, double* Gamyxy, double* Gamyxz,
    double* Gamyyy, double* Gamyyz, double* Gamyzz,
    double* Gamzxx, double* Gamzxy, double* Gamzxz,
    double* Gamzyy, double* Gamzyz, double* Gamzzz,
    double* Rxx, double* Rxy, double* Rxz,
    double* Ryy, double* Ryz, double* Rzz,
    double* ham_Res, double* movx_Res, double* movy_Res, double* movz_Res,
    double* Gmx_Res, double* Gmy_Res, double* Gmz_Res,
    int symmetry, int lev, double eps, int co
);

#endif /* BSSN_GPU_MANAGER_H */