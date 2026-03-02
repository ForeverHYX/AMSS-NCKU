#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <string>
#include <iomanip>

// ==========================================
// ANSI 颜色代码
// ==========================================
#define ANSI_RESET   "\x1b[0m"
#define ANSI_GREEN   "\x1b[32m"
#define ANSI_RED     "\x1b[31m"
#define ANSI_BLUE    "\x1b[34m"
#define ANSI_YELLOW  "\x1b[33m"

// ==========================================
// 外部 Kernel Launch 声明
// ==========================================
extern void gpu_compute_rhs_bssn_launch_std(
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
);

extern void gpu_compute_rhs_bssn_launch_opt(
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
);

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

struct GridShape {
    int level;
    int shape[3];
    double resolution[3];
    double range[6];
};

// ==========================================
// 辅助内存分配与初始化宏
// ==========================================

// 常规输入变量 (通常给坐标等)
#define ALLOC_INPUT(name, size) \
    double *name; \
    CHECK_CUDA(cudaMallocManaged(&name, size * sizeof(double))); \
    for(size_t i = 0; i < size; ++i) name[i] = ((double)rand() / RAND_MAX) - 0.5;

// 安全正数变量 (给 chi, Lap 等作为分母或根号内项)
#define ALLOC_INPUT_POS(name, size) \
    double *name; \
    CHECK_CUDA(cudaMallocManaged(&name, size * sizeof(double))); \
    for(size_t i = 0; i < size; ++i) name[i] = ((double)rand() / RAND_MAX) * 0.5 + 0.1;

// 物理微扰变量 (防 NaN 核心：确保度规矩阵正定)
#define ALLOC_INPUT_SMALL(name, size) \
    double *name; \
    CHECK_CUDA(cudaMallocManaged(&name, size * sizeof(double))); \
    for(size_t i = 0; i < size; ++i) name[i] = (((double)rand() / RAND_MAX) - 0.5) * 0.01;

// 输出变量：给相同的初始随机脏数据，以验证 Kernel 是否能正确覆写而不是乱加
#define ALLOC_OUTPUT(name, size) \
    double *std_##name, *opt_##name; \
    CHECK_CUDA(cudaMallocManaged(&std_##name, size * sizeof(double))); \
    CHECK_CUDA(cudaMallocManaged(&opt_##name, size * sizeof(double))); \
    for(size_t i = 0; i < size; ++i) { \
        double val = ((double)rand() / RAND_MAX) - 0.5; \
        std_##name[i] = val; \
        opt_##name[i] = val; \
    }

// 释放宏
#define FREE_INPUT(name) CHECK_CUDA(cudaFree(name));
#define FREE_OUTPUT(name) CHECK_CUDA(cudaFree(std_##name)); CHECK_CUDA(cudaFree(opt_##name));

// 校验函数
bool check_error(const double* std_arr, const double* opt_arr, size_t size, const std::string& name) {
    double max_err = 0.0;
    for(size_t i = 0; i < size; ++i) {
        double err = std::abs(std_arr[i] - opt_arr[i]);
        if (std::isnan(err)) {
            std::cout << "    " << ANSI_RED << "[FATAL]" << ANSI_RESET << " NaN detected in " << name << "!\n";
            return false;
        }
        if (err > max_err) max_err = err;
    }
    // 数值相对论的大型偏微分方程经过复杂重算后，1e-8 的舍入误差是可以接受的
    if (max_err > 1e-8) { 
        std::cout << "    [FAIL] Mismatch in " << std::left << std::setw(10) << name 
                  << " | Max Error = " << std::scientific << max_err << "\n";
        return false;
    }
    return true;
}

#define CHECK_RESULT(name, size) \
    all_passed &= check_error(std_##name, opt_##name, size, #name);

// ==========================================
// 核心测试逻辑
// ==========================================
void run_benchmark(const GridShape& grid) {
    size_t num_elements = grid.shape[0] * grid.shape[1] * grid.shape[2];
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Testing Level " << grid.level << " | Shape: [" 
              << grid.shape[0] << "," << grid.shape[1] << "," << grid.shape[2] << "]\n";

    // 1. 分配输入数组（采用修正后的安全初始化）
    ALLOC_INPUT(X, num_elements); ALLOC_INPUT(Y, num_elements); ALLOC_INPUT(Z, num_elements);
    
    ALLOC_INPUT_POS(chi, num_elements); ALLOC_INPUT_POS(Lap, num_elements); ALLOC_INPUT_POS(rho, num_elements);
    
    ALLOC_INPUT_SMALL(trK, num_elements);
    ALLOC_INPUT_SMALL(dxx, num_elements); ALLOC_INPUT_SMALL(gxy, num_elements); ALLOC_INPUT_SMALL(gxz, num_elements);
    ALLOC_INPUT_SMALL(dyy, num_elements); ALLOC_INPUT_SMALL(gyz, num_elements); ALLOC_INPUT_SMALL(dzz, num_elements);
    ALLOC_INPUT_SMALL(Axx, num_elements); ALLOC_INPUT_SMALL(Axy, num_elements); ALLOC_INPUT_SMALL(Axz, num_elements);
    ALLOC_INPUT_SMALL(Ayy, num_elements); ALLOC_INPUT_SMALL(Ayz, num_elements); ALLOC_INPUT_SMALL(Azz, num_elements);
    ALLOC_INPUT_SMALL(Gamx, num_elements); ALLOC_INPUT_SMALL(Gamy, num_elements); ALLOC_INPUT_SMALL(Gamz, num_elements);
    ALLOC_INPUT_SMALL(betax, num_elements); ALLOC_INPUT_SMALL(betay, num_elements); ALLOC_INPUT_SMALL(betaz, num_elements);
    ALLOC_INPUT_SMALL(dtSfx, num_elements); ALLOC_INPUT_SMALL(dtSfy, num_elements); ALLOC_INPUT_SMALL(dtSfz, num_elements);
    ALLOC_INPUT_SMALL(Sx, num_elements); ALLOC_INPUT_SMALL(Sy, num_elements); ALLOC_INPUT_SMALL(Sz, num_elements);
    ALLOC_INPUT_SMALL(Sxx, num_elements); ALLOC_INPUT_SMALL(Sxy, num_elements); ALLOC_INPUT_SMALL(Sxz, num_elements);
    ALLOC_INPUT_SMALL(Syy, num_elements); ALLOC_INPUT_SMALL(Syz, num_elements); ALLOC_INPUT_SMALL(Szz, num_elements);

    // 原 std Kernel 中间变量，随便初始化即可，会被完全覆盖
    ALLOC_INPUT(Gamxxx, num_elements); ALLOC_INPUT(Gamxxy, num_elements); ALLOC_INPUT(Gamxxz, num_elements);
    ALLOC_INPUT(Gamxyy, num_elements); ALLOC_INPUT(Gamxyz, num_elements); ALLOC_INPUT(Gamxzz, num_elements);
    ALLOC_INPUT(Gamyxx, num_elements); ALLOC_INPUT(Gamyxy, num_elements); ALLOC_INPUT(Gamyxz, num_elements);
    ALLOC_INPUT(Gamyyy, num_elements); ALLOC_INPUT(Gamyyz, num_elements); ALLOC_INPUT(Gamyzz, num_elements);
    ALLOC_INPUT(Gamzxx, num_elements); ALLOC_INPUT(Gamzxy, num_elements); ALLOC_INPUT(Gamzxz, num_elements);
    ALLOC_INPUT(Gamzyy, num_elements); ALLOC_INPUT(Gamzyz, num_elements); ALLOC_INPUT(Gamzzz, num_elements);
    ALLOC_INPUT(Rxx, num_elements); ALLOC_INPUT(Rxy, num_elements); ALLOC_INPUT(Rxz, num_elements);
    ALLOC_INPUT(Ryy, num_elements); ALLOC_INPUT(Ryz, num_elements); ALLOC_INPUT(Rzz, num_elements);

    // 2. 分配带有两套拷贝的 RHS / Res 输出数组
    ALLOC_OUTPUT(chi_rhs, num_elements); ALLOC_OUTPUT(trK_rhs, num_elements);
    ALLOC_OUTPUT(gxx_rhs, num_elements); ALLOC_OUTPUT(gxy_rhs, num_elements); ALLOC_OUTPUT(gxz_rhs, num_elements);
    ALLOC_OUTPUT(gyy_rhs, num_elements); ALLOC_OUTPUT(gyz_rhs, num_elements); ALLOC_OUTPUT(gzz_rhs, num_elements);
    ALLOC_OUTPUT(Axx_rhs, num_elements); ALLOC_OUTPUT(Axy_rhs, num_elements); ALLOC_OUTPUT(Axz_rhs, num_elements);
    ALLOC_OUTPUT(Ayy_rhs, num_elements); ALLOC_OUTPUT(Ayz_rhs, num_elements); ALLOC_OUTPUT(Azz_rhs, num_elements);
    ALLOC_OUTPUT(Gamx_rhs, num_elements); ALLOC_OUTPUT(Gamy_rhs, num_elements); ALLOC_OUTPUT(Gamz_rhs, num_elements);
    ALLOC_OUTPUT(Lap_rhs, num_elements);
    ALLOC_OUTPUT(betax_rhs, num_elements); ALLOC_OUTPUT(betay_rhs, num_elements); ALLOC_OUTPUT(betaz_rhs, num_elements);
    ALLOC_OUTPUT(dtSfx_rhs, num_elements); ALLOC_OUTPUT(dtSfy_rhs, num_elements); ALLOC_OUTPUT(dtSfz_rhs, num_elements);
    
    ALLOC_OUTPUT(ham_Res, num_elements); 
    ALLOC_OUTPUT(movx_Res, num_elements); ALLOC_OUTPUT(movy_Res, num_elements); ALLOC_OUTPUT(movz_Res, num_elements);
    ALLOC_OUTPUT(Gmx_Res, num_elements); ALLOC_OUTPUT(Gmy_Res, num_elements); ALLOC_OUTPUT(Gmz_Res, num_elements);

    // 设置执行参数
    int ex[3] = {grid.shape[0], grid.shape[1], grid.shape[2]};
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    int symmetry = 0; double eps = 0.1; int co = 0; double T = 0.0;
    int warmup = 3;
    int iterations = 20;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // ==========================================
    // 3. 性能测试: Standard (原版)
    // ==========================================
    for(int i = 0; i < warmup; ++i) {
        gpu_compute_rhs_bssn_launch_std(stream, ex, T, X, Y, Z, chi, trK, dxx, gxy, gxz, dyy, gyz, dzz,
            Axx, Axy, Axz, Ayy, Ayz, Azz, Gamx, Gamy, Gamz, Lap, betax, betay, betaz, dtSfx, dtSfy, dtSfz,
            std_chi_rhs, std_trK_rhs, std_gxx_rhs, std_gxy_rhs, std_gxz_rhs, std_gyy_rhs, std_gyz_rhs, std_gzz_rhs,
            std_Axx_rhs, std_Axy_rhs, std_Axz_rhs, std_Ayy_rhs, std_Ayz_rhs, std_Azz_rhs,
            std_Gamx_rhs, std_Gamy_rhs, std_Gamz_rhs, std_Lap_rhs, std_betax_rhs, std_betay_rhs, std_betaz_rhs,
            std_dtSfx_rhs, std_dtSfy_rhs, std_dtSfz_rhs, rho, Sx, Sy, Sz, Sxx, Sxy, Sxz, Syy, Syz, Szz,
            Gamxxx, Gamxxy, Gamxxz, Gamxyy, Gamxyz, Gamxzz, Gamyxx, Gamyxy, Gamyxz, Gamyyy, Gamyyz, Gamyzz,
            Gamzxx, Gamzxy, Gamzxz, Gamzyy, Gamzyz, Gamzzz, Rxx, Rxy, Rxz, Ryy, Ryz, Rzz,
            std_ham_Res, std_movx_Res, std_movy_Res, std_movz_Res, std_Gmx_Res, std_Gmy_Res, std_Gmz_Res,
            symmetry, grid.level, eps, co);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEventRecord(start, stream);
    for(int i = 0; i < iterations; ++i) {
        gpu_compute_rhs_bssn_launch_std(stream, ex, T, X, Y, Z, chi, trK, dxx, gxy, gxz, dyy, gyz, dzz,
            Axx, Axy, Axz, Ayy, Ayz, Azz, Gamx, Gamy, Gamz, Lap, betax, betay, betaz, dtSfx, dtSfy, dtSfz,
            std_chi_rhs, std_trK_rhs, std_gxx_rhs, std_gxy_rhs, std_gxz_rhs, std_gyy_rhs, std_gyz_rhs, std_gzz_rhs,
            std_Axx_rhs, std_Axy_rhs, std_Axz_rhs, std_Ayy_rhs, std_Ayz_rhs, std_Azz_rhs,
            std_Gamx_rhs, std_Gamy_rhs, std_Gamz_rhs, std_Lap_rhs, std_betax_rhs, std_betay_rhs, std_betaz_rhs,
            std_dtSfx_rhs, std_dtSfy_rhs, std_dtSfz_rhs, rho, Sx, Sy, Sz, Sxx, Sxy, Sxz, Syy, Syz, Szz,
            Gamxxx, Gamxxy, Gamxxz, Gamxyy, Gamxyz, Gamxzz, Gamyxx, Gamyxy, Gamyxz, Gamyyy, Gamyyz, Gamyzz,
            Gamzxx, Gamzxy, Gamzxz, Gamzyy, Gamzyz, Gamzzz, Rxx, Rxy, Rxz, Ryy, Ryz, Rzz,
            std_ham_Res, std_movx_Res, std_movy_Res, std_movz_Res, std_Gmx_Res, std_Gmy_Res, std_Gmz_Res,
            symmetry, grid.level, eps, co);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float std_ms = 0;
    cudaEventElapsedTime(&std_ms, start, stop);
    std_ms /= iterations;

    // ==========================================
    // 4. 性能测试: Optimized (优化解耦版)
    // ==========================================
    for(int i = 0; i < warmup; ++i) {
        gpu_compute_rhs_bssn_launch_opt(stream, ex, T, X, Y, Z, chi, trK, dxx, gxy, gxz, dyy, gyz, dzz,
            Axx, Axy, Axz, Ayy, Ayz, Azz, Gamx, Gamy, Gamz, Lap, betax, betay, betaz, dtSfx, dtSfy, dtSfz,
            opt_chi_rhs, opt_trK_rhs, opt_gxx_rhs, opt_gxy_rhs, opt_gxz_rhs, opt_gyy_rhs, opt_gyz_rhs, opt_gzz_rhs,
            opt_Axx_rhs, opt_Axy_rhs, opt_Axz_rhs, opt_Ayy_rhs, opt_Ayz_rhs, opt_Azz_rhs,
            opt_Gamx_rhs, opt_Gamy_rhs, opt_Gamz_rhs, opt_Lap_rhs, opt_betax_rhs, opt_betay_rhs, opt_betaz_rhs,
            opt_dtSfx_rhs, opt_dtSfy_rhs, opt_dtSfz_rhs, rho, Sx, Sy, Sz, Sxx, Sxy, Sxz, Syy, Syz, Szz,
            Gamxxx, Gamxxy, Gamxxz, Gamxyy, Gamxyz, Gamxzz, Gamyxx, Gamyxy, Gamyxz, Gamyyy, Gamyyz, Gamyzz,
            Gamzxx, Gamzxy, Gamzxz, Gamzyy, Gamzyz, Gamzzz, Rxx, Rxy, Rxz, Ryy, Ryz, Rzz,
            opt_ham_Res, opt_movx_Res, opt_movy_Res, opt_movz_Res, opt_Gmx_Res, opt_Gmy_Res, opt_Gmz_Res,
            symmetry, grid.level, eps, co);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEventRecord(start, stream);
    for(int i = 0; i < iterations; ++i) {
        gpu_compute_rhs_bssn_launch_opt(stream, ex, T, X, Y, Z, chi, trK, dxx, gxy, gxz, dyy, gyz, dzz,
            Axx, Axy, Axz, Ayy, Ayz, Azz, Gamx, Gamy, Gamz, Lap, betax, betay, betaz, dtSfx, dtSfy, dtSfz,
            opt_chi_rhs, opt_trK_rhs, opt_gxx_rhs, opt_gxy_rhs, opt_gxz_rhs, opt_gyy_rhs, opt_gyz_rhs, opt_gzz_rhs,
            opt_Axx_rhs, opt_Axy_rhs, opt_Axz_rhs, opt_Ayy_rhs, opt_Ayz_rhs, opt_Azz_rhs,
            opt_Gamx_rhs, opt_Gamy_rhs, opt_Gamz_rhs, opt_Lap_rhs, opt_betax_rhs, opt_betay_rhs, opt_betaz_rhs,
            opt_dtSfx_rhs, opt_dtSfy_rhs, opt_dtSfz_rhs, rho, Sx, Sy, Sz, Sxx, Sxy, Sxz, Syy, Syz, Szz,
            Gamxxx, Gamxxy, Gamxxz, Gamxyy, Gamxyz, Gamxzz, Gamyxx, Gamyxy, Gamyxz, Gamyyy, Gamyyz, Gamyzz,
            Gamzxx, Gamzxy, Gamzxz, Gamzyy, Gamzyz, Gamzzz, Rxx, Rxy, Rxz, Ryy, Ryz, Rzz,
            opt_ham_Res, opt_movx_Res, opt_movy_Res, opt_movz_Res, opt_Gmx_Res, opt_Gmy_Res, opt_Gmz_Res,
            symmetry, grid.level, eps, co);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float opt_ms = 0;
    cudaEventElapsedTime(&opt_ms, start, stop);
    opt_ms /= iterations;

    // 打印性能信息
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Standard Time : " << std_ms << " ms\n";
    std::cout << "  Optimized Time: " << opt_ms << " ms\n";
    std::cout << "  Speedup       : " << ANSI_BLUE << std_ms / opt_ms << ANSI_RESET << " x\n";

    // ==========================================
    // 5. 结果正确性强校验
    // ==========================================
    CHECK_CUDA(cudaDeviceSynchronize());
    bool all_passed = true;
    
    CHECK_RESULT(chi_rhs, num_elements); CHECK_RESULT(trK_rhs, num_elements);
    CHECK_RESULT(gxx_rhs, num_elements); CHECK_RESULT(gxy_rhs, num_elements); CHECK_RESULT(gxz_rhs, num_elements);
    CHECK_RESULT(gyy_rhs, num_elements); CHECK_RESULT(gyz_rhs, num_elements); CHECK_RESULT(gzz_rhs, num_elements);
    CHECK_RESULT(Axx_rhs, num_elements); CHECK_RESULT(Axy_rhs, num_elements); CHECK_RESULT(Axz_rhs, num_elements);
    CHECK_RESULT(Ayy_rhs, num_elements); CHECK_RESULT(Ayz_rhs, num_elements); CHECK_RESULT(Azz_rhs, num_elements);
    CHECK_RESULT(Gamx_rhs, num_elements); CHECK_RESULT(Gamy_rhs, num_elements); CHECK_RESULT(Gamz_rhs, num_elements);
    CHECK_RESULT(Lap_rhs, num_elements);
    CHECK_RESULT(betax_rhs, num_elements); CHECK_RESULT(betay_rhs, num_elements); CHECK_RESULT(betaz_rhs, num_elements);
    CHECK_RESULT(dtSfx_rhs, num_elements); CHECK_RESULT(dtSfy_rhs, num_elements); CHECK_RESULT(dtSfz_rhs, num_elements);
    
    CHECK_RESULT(ham_Res, num_elements); 
    CHECK_RESULT(movx_Res, num_elements); CHECK_RESULT(movy_Res, num_elements); CHECK_RESULT(movz_Res, num_elements);
    CHECK_RESULT(Gmx_Res, num_elements); CHECK_RESULT(Gmy_Res, num_elements); CHECK_RESULT(Gmz_Res, num_elements);

    if (all_passed) {
        std::cout << "  " << ANSI_GREEN << "[SUCCESS]" << ANSI_RESET << " All outputs match exactly!\n";
    } else {
        std::cout << "  " << ANSI_RED << "[FAILED]" << ANSI_RESET << " Some outputs diverged. Check your optimizations!\n";
    }

    // ==========================================
    // 6. 内存释放
    // ==========================================
    FREE_INPUT(X); FREE_INPUT(Y); FREE_INPUT(Z);
    FREE_INPUT(chi); FREE_INPUT(trK);
    FREE_INPUT(dxx); FREE_INPUT(gxy); FREE_INPUT(gxz); FREE_INPUT(dyy); FREE_INPUT(gyz); FREE_INPUT(dzz);
    FREE_INPUT(Axx); FREE_INPUT(Axy); FREE_INPUT(Axz); FREE_INPUT(Ayy); FREE_INPUT(Ayz); FREE_INPUT(Azz);
    FREE_INPUT(Gamx); FREE_INPUT(Gamy); FREE_INPUT(Gamz); FREE_INPUT(Lap);
    FREE_INPUT(betax); FREE_INPUT(betay); FREE_INPUT(betaz);
    FREE_INPUT(dtSfx); FREE_INPUT(dtSfy); FREE_INPUT(dtSfz);
    FREE_INPUT(rho); FREE_INPUT(Sx); FREE_INPUT(Sy); FREE_INPUT(Sz);
    FREE_INPUT(Sxx); FREE_INPUT(Sxy); FREE_INPUT(Sxz); FREE_INPUT(Syy); FREE_INPUT(Syz); FREE_INPUT(Szz);
    
    FREE_INPUT(Gamxxx); FREE_INPUT(Gamxxy); FREE_INPUT(Gamxxz); FREE_INPUT(Gamxyy); FREE_INPUT(Gamxyz); FREE_INPUT(Gamxzz);
    FREE_INPUT(Gamyxx); FREE_INPUT(Gamyxy); FREE_INPUT(Gamyxz); FREE_INPUT(Gamyyy); FREE_INPUT(Gamyyz); FREE_INPUT(Gamyzz);
    FREE_INPUT(Gamzxx); FREE_INPUT(Gamzxy); FREE_INPUT(Gamzxz); FREE_INPUT(Gamzyy); FREE_INPUT(Gamzyz); FREE_INPUT(Gamzzz);
    FREE_INPUT(Rxx); FREE_INPUT(Rxy); FREE_INPUT(Rxz); FREE_INPUT(Ryy); FREE_INPUT(Ryz); FREE_INPUT(Rzz);

    FREE_OUTPUT(chi_rhs); FREE_OUTPUT(trK_rhs);
    FREE_OUTPUT(gxx_rhs); FREE_OUTPUT(gxy_rhs); FREE_OUTPUT(gxz_rhs);
    FREE_OUTPUT(gyy_rhs); FREE_OUTPUT(gyz_rhs); FREE_OUTPUT(gzz_rhs);
    FREE_OUTPUT(Axx_rhs); FREE_OUTPUT(Axy_rhs); FREE_OUTPUT(Axz_rhs);
    FREE_OUTPUT(Ayy_rhs); FREE_OUTPUT(Ayz_rhs); FREE_OUTPUT(Azz_rhs);
    FREE_OUTPUT(Gamx_rhs); FREE_OUTPUT(Gamy_rhs); FREE_OUTPUT(Gamz_rhs);
    FREE_OUTPUT(Lap_rhs);
    FREE_OUTPUT(betax_rhs); FREE_OUTPUT(betay_rhs); FREE_OUTPUT(betaz_rhs);
    FREE_OUTPUT(dtSfx_rhs); FREE_OUTPUT(dtSfy_rhs); FREE_OUTPUT(dtSfz_rhs);
    FREE_OUTPUT(ham_Res); FREE_OUTPUT(movx_Res); FREE_OUTPUT(movy_Res); FREE_OUTPUT(movz_Res);
    FREE_OUTPUT(Gmx_Res); FREE_OUTPUT(Gmy_Res); FREE_OUTPUT(Gmz_Res);

    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // 消除随机性带来的执行分支差异
    srand(42); 

    std::vector<GridShape> shapes = {
        {0, {96,96,48}, {6.66667,6.66667,6.66667}, {-320,320, -320,320, 0,320}},
        {1, {96,96,48}, {3.33333,3.33333,3.33333}, {-160,160, -160,160, 0,160}},
        {2, {96,96,48}, {1.66667,1.66667,1.66667}, {-80,80, -80,80, 0,80}},
        {3, {96,96,48}, {0.833333,0.833333,0.833333}, {-40,40, -40,40, 0,40}},
        {4, {96,96,48}, {0.416667,0.416667,0.416667}, {-20,20, -20,20, 0,20}},
        {5, {48,96,24}, {0.208333,0.208333,0.208333}, {-5,5, -10.625,9.375, 0,5}},
        {6, {48,48,24}, {0.104167,0.104167,0.104167}, {-2.5,2.5, 1.97917,6.97917, 0,2.5}},
        {6, {48,48,24}, {0.104167,0.104167,0.104167}, {-2.5,2.5, -8.02083,-3.02083, 0,2.5}},
        {7, {48,48,24}, {0.0520833,0.0520833,0.0520833}, {-1.25,1.25, 3.22917,5.72917, 0,1.25}},
        {7, {48,48,24}, {0.0520833,0.0520833,0.0520833}, {-1.25,1.25, -6.77083,-4.27083, 0,1.25}},
        {8, {48,48,24}, {0.0260417,0.0260417,0.0260417}, {-0.625,0.625, 3.82812,5.07812, 0,0.625}},
        {8, {48,48,24}, {0.0260417,0.0260417,0.0260417}, {-0.625,0.625, -6.17188,-4.92188, 0,0.625}}
    };

    for (const auto& grid : shapes) {
        run_benchmark(grid);
    }

    return 0;
}