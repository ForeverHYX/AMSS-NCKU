#include "bssn_gpu_manager.h"

#include "bssn_rhs_gpu.h"

BssnCudaManager::BssnCudaManager() {
    CHECK_CUDA(cudaStreamCreate(&stream));
    // 初始化所有指针为 nullptr
    d_X = d_Y = d_Z = nullptr;
    d_chi = d_trK = nullptr;
    d_dxx = d_gxy = d_gxz = d_dyy = d_gyz = d_dzz = nullptr;
    d_Axx = d_Axy = d_Axz = d_Ayy = d_Ayz = d_Azz = nullptr;
    d_Gamx = d_Gamy = d_Gamz = nullptr;
    d_Lap = nullptr;
    d_betax = d_betay = d_betaz = nullptr;
    d_dtSfx = d_dtSfy = d_dtSfz = nullptr;
    d_rho = d_Sx = d_Sy = d_Sz = nullptr;
    d_Sxx = d_Sxy = d_Sxz = d_Syy = d_Syz = d_Szz = nullptr;

    d_chi_rhs = d_trK_rhs = nullptr;
    d_gxx_rhs = d_gxy_rhs = d_gxz_rhs = d_gyy_rhs = d_gyz_rhs = d_gzz_rhs = nullptr;
    d_Axx_rhs = d_Axy_rhs = d_Axz_rhs = d_Ayy_rhs = d_Ayz_rhs = d_Azz_rhs = nullptr;
    d_Gamx_rhs = d_Gamy_rhs = d_Gamz_rhs = nullptr;
    d_Lap_rhs = nullptr;
    d_betax_rhs = d_betay_rhs = d_betaz_rhs = nullptr;
    d_dtSfx_rhs = d_dtSfy_rhs = d_dtSfz_rhs = nullptr;

    d_Gamxxx = d_Gamxxy = d_Gamxxz = d_Gamxyy = d_Gamxyz = d_Gamxzz = nullptr;
    d_Gamyxx = d_Gamyxy = d_Gamyxz = d_Gamyyy = d_Gamyyz = d_Gamyzz = nullptr;
    d_Gamzxx = d_Gamzxy = d_Gamzxz = d_Gamzyy = d_Gamzyz = d_Gamzzz = nullptr;
    d_Rxx = d_Rxy = d_Rxz = d_Ryy = d_Ryz = d_Rzz = nullptr;
    d_ham_Res = d_movx_Res = d_movy_Res = d_movz_Res = nullptr;
    d_Gmx_Res = d_Gmy_Res = d_Gmz_Res = nullptr;

    d_chix = d_chiy = d_chiz = nullptr;
    d_betaxx = d_betaxy = d_betaxz = d_betayx = d_betayy = d_betayz = d_betazx = d_betazy = d_betazz = nullptr;
    d_gxxx = d_gxxy = d_gxxz = d_gxyx = d_gxyy = d_gxyz = nullptr;
    d_gxzx = d_gxzy = d_gxzz = d_gyyx = d_gyyy = d_gyyz = nullptr;
    d_gyzx = d_gyzy = d_gyzz = d_gzzx = d_gzzy = d_gzzz = nullptr;
    d_gupxx = d_gupxy = d_gupxz = d_gupyy = d_gupyz = d_gupzz = nullptr;
}

BssnCudaManager::~BssnCudaManager() {
    // 释放所有显存
    if (d_X) cudaFree(d_X);
    if (d_Y) cudaFree(d_Y);
    if (d_Z) cudaFree(d_Z);
    if (d_chi) cudaFree(d_chi);
    if (d_trK) cudaFree(d_trK);
    if (d_dxx) cudaFree(d_dxx);
    if (d_gxy) cudaFree(d_gxy);
    if (d_gxz) cudaFree(d_gxz);
    if (d_dyy) cudaFree(d_dyy);
    if (d_gyz) cudaFree(d_gyz);
    if (d_dzz) cudaFree(d_dzz);
    if (d_Axx) cudaFree(d_Axx);
    if (d_Axy) cudaFree(d_Axy);
    if (d_Axz) cudaFree(d_Axz);
    if (d_Ayy) cudaFree(d_Ayy);
    if (d_Ayz) cudaFree(d_Ayz);
    if (d_Azz) cudaFree(d_Azz);
    if (d_Gamx) cudaFree(d_Gamx);
    if (d_Gamy) cudaFree(d_Gamy);
    if (d_Gamz) cudaFree(d_Gamz);
    if (d_Lap) cudaFree(d_Lap);
    if (d_betax) cudaFree(d_betax);
    if (d_betay) cudaFree(d_betay);
    if (d_betaz) cudaFree(d_betaz);
    if (d_dtSfx) cudaFree(d_dtSfx);
    if (d_dtSfy) cudaFree(d_dtSfy);
    if (d_dtSfz) cudaFree(d_dtSfz);
    if (d_rho) cudaFree(d_rho);
    if (d_Sx) cudaFree(d_Sx);
    if (d_Sy) cudaFree(d_Sy);
    if (d_Sz) cudaFree(d_Sz);
    if (d_Sxx) cudaFree(d_Sxx);
    if (d_Sxy) cudaFree(d_Sxy);
    if (d_Sxz) cudaFree(d_Sxz);
    if (d_Syy) cudaFree(d_Syy);
    if (d_Syz) cudaFree(d_Syz);
    if (d_Szz) cudaFree(d_Szz);

    if (d_chi_rhs) cudaFree(d_chi_rhs);
    if (d_trK_rhs) cudaFree(d_trK_rhs);
    if (d_gxx_rhs) cudaFree(d_gxx_rhs);
    if (d_gxy_rhs) cudaFree(d_gxy_rhs);
    if (d_gxz_rhs) cudaFree(d_gxz_rhs);
    if (d_gyy_rhs) cudaFree(d_gyy_rhs);
    if (d_gyz_rhs) cudaFree(d_gyz_rhs);
    if (d_gzz_rhs) cudaFree(d_gzz_rhs);
    if (d_Axx_rhs) cudaFree(d_Axx_rhs);
    if (d_Axy_rhs) cudaFree(d_Axy_rhs);
    if (d_Axz_rhs) cudaFree(d_Axz_rhs);
    if (d_Ayy_rhs) cudaFree(d_Ayy_rhs);
    if (d_Ayz_rhs) cudaFree(d_Ayz_rhs);
    if (d_Azz_rhs) cudaFree(d_Azz_rhs);
    if (d_Gamx_rhs) cudaFree(d_Gamx_rhs);
    if (d_Gamy_rhs) cudaFree(d_Gamy_rhs);
    if (d_Gamz_rhs) cudaFree(d_Gamz_rhs);
    if (d_Lap_rhs) cudaFree(d_Lap_rhs);
    if (d_betax_rhs) cudaFree(d_betax_rhs);
    if (d_betay_rhs) cudaFree(d_betay_rhs);
    if (d_betaz_rhs) cudaFree(d_betaz_rhs);
    if (d_dtSfx_rhs) cudaFree(d_dtSfx_rhs);
    if (d_dtSfy_rhs) cudaFree(d_dtSfy_rhs);
    if (d_dtSfz_rhs) cudaFree(d_dtSfz_rhs);

    if (d_Gamxxx) cudaFree(d_Gamxxx);
    if (d_Gamxxy) cudaFree(d_Gamxxy);
    if (d_Gamxxz) cudaFree(d_Gamxxz);
    if (d_Gamxyy) cudaFree(d_Gamxyy);
    if (d_Gamxyz) cudaFree(d_Gamxyz);
    if (d_Gamxzz) cudaFree(d_Gamxzz);
    if (d_Gamyxx) cudaFree(d_Gamyxx);
    if (d_Gamyxy) cudaFree(d_Gamyxy);
    if (d_Gamyxz) cudaFree(d_Gamyxz);
    if (d_Gamyyy) cudaFree(d_Gamyyy);
    if (d_Gamyyz) cudaFree(d_Gamyyz);
    if (d_Gamyzz) cudaFree(d_Gamyzz);
    if (d_Gamzxx) cudaFree(d_Gamzxx);
    if (d_Gamzxy) cudaFree(d_Gamzxy);
    if (d_Gamzxz) cudaFree(d_Gamzxz);
    if (d_Gamzyy) cudaFree(d_Gamzyy);
    if (d_Gamzyz) cudaFree(d_Gamzyz);
    if (d_Gamzzz) cudaFree(d_Gamzzz);
    if (d_Rxx) cudaFree(d_Rxx);
    if (d_Rxy) cudaFree(d_Rxy);
    if (d_Rxz) cudaFree(d_Rxz);
    if (d_Ryy) cudaFree(d_Ryy);
    if (d_Ryz) cudaFree(d_Ryz);
    if (d_Rzz) cudaFree(d_Rzz);
    if (d_ham_Res) cudaFree(d_ham_Res);
    if (d_movx_Res) cudaFree(d_movx_Res);
    if (d_movy_Res) cudaFree(d_movy_Res);
    if (d_movz_Res) cudaFree(d_movz_Res);
    if (d_Gmx_Res) cudaFree(d_Gmx_Res);
    if (d_Gmy_Res) cudaFree(d_Gmy_Res);
    if (d_Gmz_Res) cudaFree(d_Gmz_Res);

    if (d_chix) cudaFree(d_chix);
    if (d_chiy) cudaFree(d_chiy);
    if (d_chiz) cudaFree(d_chiz);
    if (d_betaxx) cudaFree(d_betaxx);
    if (d_betaxy) cudaFree(d_betaxy);
    if (d_betaxz) cudaFree(d_betaxz);
    if (d_betayx) cudaFree(d_betayx);
    if (d_betayy) cudaFree(d_betayy);
    if (d_betayz) cudaFree(d_betayz);
    if (d_betazx) cudaFree(d_betazx);
    if (d_betazy) cudaFree(d_betazy);
    if (d_betazz) cudaFree(d_betazz);
    if (d_gxxx) cudaFree(d_gxxx);
    if (d_gxxy) cudaFree(d_gxxy);
    if (d_gxxz) cudaFree(d_gxxz);
    if (d_gxyx) cudaFree(d_gxyx);
    if (d_gxyy) cudaFree(d_gxyy);
    if (d_gxyz) cudaFree(d_gxyz);
    if (d_gxzx) cudaFree(d_gxzx);
    if (d_gxzy) cudaFree(d_gxzy);
    if (d_gxzz) cudaFree(d_gxzz);
    if (d_gyyx) cudaFree(d_gyyx);
    if (d_gyyy) cudaFree(d_gyyy);
    if (d_gyyz) cudaFree(d_gyyz);
    if (d_gyzx) cudaFree(d_gyzx);
    if (d_gyzy) cudaFree(d_gyzy);
    if (d_gyzz) cudaFree(d_gyzz);
    if (d_gzzx) cudaFree(d_gzzx);
    if (d_gzzy) cudaFree(d_gzzy);
    if (d_gzzz) cudaFree(d_gzzz);
    if (d_gupxx) cudaFree(d_gupxx);
    if (d_gupxy) cudaFree(d_gupxy);
    if (d_gupxz) cudaFree(d_gupxz);
    if (d_gupyy) cudaFree(d_gupyy);
    if (d_gupyz) cudaFree(d_gupyz);
    if (d_gupzz) cudaFree(d_gupzz);

    CHECK_CUDA(cudaStreamDestroy(stream));
}

void BssnCudaManager::allocate_intermediates(size_t total_elements) {
    // 辅助 lambda，简化 malloc
    auto malloc_dev = [&](double*& ptr) {
        if(ptr) cudaFree(ptr);
        CHECK_CUDA(cudaMalloc(&ptr, total_elements * sizeof(double)));
        CHECK_CUDA(cudaMemset(ptr, 0, total_elements * sizeof(double)));
    };

    // 1. Inputs
    malloc_dev(d_X); malloc_dev(d_Y); malloc_dev(d_Z);
    malloc_dev(d_chi); malloc_dev(d_trK);
    malloc_dev(d_dxx); malloc_dev(d_gxy); malloc_dev(d_gxz); malloc_dev(d_dyy); malloc_dev(d_gyz); malloc_dev(d_dzz);
    malloc_dev(d_Axx); malloc_dev(d_Axy); malloc_dev(d_Axz); malloc_dev(d_Ayy); malloc_dev(d_Ayz); malloc_dev(d_Azz);
    malloc_dev(d_Gamx); malloc_dev(d_Gamy); malloc_dev(d_Gamz);
    malloc_dev(d_Lap);
    malloc_dev(d_betax); malloc_dev(d_betay); malloc_dev(d_betaz);
    malloc_dev(d_dtSfx); malloc_dev(d_dtSfy); malloc_dev(d_dtSfz);
    malloc_dev(d_rho); malloc_dev(d_Sx); malloc_dev(d_Sy); malloc_dev(d_Sz);
    malloc_dev(d_Sxx); malloc_dev(d_Sxy); malloc_dev(d_Sxz); malloc_dev(d_Syy); malloc_dev(d_Syz); malloc_dev(d_Szz);

    // 2. Outputs (RHS)
    malloc_dev(d_chi_rhs); malloc_dev(d_trK_rhs);
    malloc_dev(d_gxx_rhs); malloc_dev(d_gxy_rhs); malloc_dev(d_gxz_rhs); 
    malloc_dev(d_gyy_rhs); malloc_dev(d_gyz_rhs); malloc_dev(d_gzz_rhs);
    malloc_dev(d_Axx_rhs); malloc_dev(d_Axy_rhs); malloc_dev(d_Axz_rhs); 
    malloc_dev(d_Ayy_rhs); malloc_dev(d_Ayz_rhs); malloc_dev(d_Azz_rhs);
    malloc_dev(d_Gamx_rhs); malloc_dev(d_Gamy_rhs); malloc_dev(d_Gamz_rhs);
    malloc_dev(d_Lap_rhs);
    malloc_dev(d_betax_rhs); malloc_dev(d_betay_rhs); malloc_dev(d_betaz_rhs);
    malloc_dev(d_dtSfx_rhs); malloc_dev(d_dtSfy_rhs); malloc_dev(d_dtSfz_rhs);

    // 3. Diagnostics
    malloc_dev(d_Gamxxx); malloc_dev(d_Gamxxy); malloc_dev(d_Gamxxz);
    malloc_dev(d_Gamxyy); malloc_dev(d_Gamxyz); malloc_dev(d_Gamxzz);
    malloc_dev(d_Gamyxx); malloc_dev(d_Gamyxy); malloc_dev(d_Gamyxz);
    malloc_dev(d_Gamyyy); malloc_dev(d_Gamyyz); malloc_dev(d_Gamyzz);
    malloc_dev(d_Gamzxx); malloc_dev(d_Gamzxy); malloc_dev(d_Gamzxz);
    malloc_dev(d_Gamzyy); malloc_dev(d_Gamzyz); malloc_dev(d_Gamzzz);
    malloc_dev(d_Rxx); malloc_dev(d_Rxy); malloc_dev(d_Rxz); 
    malloc_dev(d_Ryy); malloc_dev(d_Ryz); malloc_dev(d_Rzz);
    malloc_dev(d_ham_Res); malloc_dev(d_movx_Res); malloc_dev(d_movy_Res); malloc_dev(d_movz_Res);
    malloc_dev(d_Gmx_Res); malloc_dev(d_Gmy_Res); malloc_dev(d_Gmz_Res);

    // 4. Intermediates (Kernel 1 -> 2)
    malloc_dev(d_chix); malloc_dev(d_chiy); malloc_dev(d_chiz);
    malloc_dev(d_betaxx); malloc_dev(d_betaxy); malloc_dev(d_betaxz);
    malloc_dev(d_betayx); malloc_dev(d_betayy); malloc_dev(d_betayz);
    malloc_dev(d_betazx); malloc_dev(d_betazy); malloc_dev(d_betazz);
    malloc_dev(d_gxxx); malloc_dev(d_gxxy); malloc_dev(d_gxxz);
    malloc_dev(d_gxyx); malloc_dev(d_gxyy); malloc_dev(d_gxyz);
    malloc_dev(d_gxzx); malloc_dev(d_gxzy); malloc_dev(d_gxzz);
    malloc_dev(d_gyyx); malloc_dev(d_gyyy); malloc_dev(d_gyyz);
    malloc_dev(d_gyzx); malloc_dev(d_gyzy); malloc_dev(d_gyzz);
    malloc_dev(d_gzzx); malloc_dev(d_gzzy); malloc_dev(d_gzzz);
    malloc_dev(d_gupxx); malloc_dev(d_gupxy); malloc_dev(d_gupxz);
    malloc_dev(d_gupyy); malloc_dev(d_gupyz); malloc_dev(d_gupzz);
}

void BssnCudaManager::resize(const int ex[3]) {
    size_t new_size = ex[0] * ex[1] * ex[2];
    if (new_size != current_grid_size) {
        current_grid_size = new_size;
        current_ex[0] = ex[0]; current_ex[1] = ex[1]; current_ex[2] = ex[2];
        allocate_intermediates(current_grid_size);
    }
}

void BssnCudaManager::copy_inputs_to_device(
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
) {
    size_t bytes = current_grid_size * sizeof(double);
    size_t bytes_1d_x = current_ex[0] * sizeof(double);
    size_t bytes_1d_y = current_ex[1] * sizeof(double);
    size_t bytes_1d_z = current_ex[2] * sizeof(double);

    // 坐标 (1D Arrays)
    CHECK_CUDA(cudaMemcpyAsync(d_X, X, bytes_1d_x, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_Y, Y, bytes_1d_y, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_Z, Z, bytes_1d_z, cudaMemcpyHostToDevice, stream));

    // 3D Fields
    auto cpy = [&](double* dst, const double* src) {
        CHECK_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
    };

    cpy(d_chi, chi); cpy(d_trK, trK);
    cpy(d_dxx, dxx); cpy(d_gxy, gxy); cpy(d_gxz, gxz); cpy(d_dyy, dyy); cpy(d_gyz, gyz); cpy(d_dzz, dzz);
    cpy(d_Axx, Axx); cpy(d_Axy, Axy); cpy(d_Axz, Axz); cpy(d_Ayy, Ayy); cpy(d_Ayz, Ayz); cpy(d_Azz, Azz);
    cpy(d_Gamx, Gamx); cpy(d_Gamy, Gamy); cpy(d_Gamz, Gamz);
    cpy(d_Lap, Lap);
    cpy(d_betax, betax); cpy(d_betay, betay); cpy(d_betaz, betaz);
    cpy(d_dtSfx, dtSfx); cpy(d_dtSfy, dtSfy); cpy(d_dtSfz, dtSfz);
    cpy(d_rho, rho); cpy(d_Sx, Sx); cpy(d_Sy, Sy); cpy(d_Sz, Sz);
    cpy(d_Sxx, Sxx); cpy(d_Sxy, Sxy); cpy(d_Sxz, Sxz); 
    cpy(d_Syy, Syy); cpy(d_Syz, Syz); cpy(d_Szz, Szz);
}

void BssnCudaManager::run_kernels(int ex[3], int symmetry, int lev, double eps, int co) {
    dim3 block(8, 8, 4); // 调整 block size 以适应架构
    dim3 grid(
        (ex[0] + block.x - 1) / block.x,
        (ex[1] + block.y - 1) / block.y,
        (ex[2] + block.z - 1) / block.z
    );

    // 1. Kernel 1: Derivatives & Connection Coefficients
    bssn_derivatives_kernel<<<grid, block, 0, stream>>>(
        ex[0], ex[1], ex[2], symmetry, lev, co,
        d_X, d_Y, d_Z,
        d_chi, d_trK,
        d_dxx, d_gxy, d_gxz, d_dyy, d_gyz, d_dzz,
        d_Axx, d_Axy, d_Axz, d_Ayy, d_Ayz, d_Azz,
        d_Lap, d_betax, d_betay, d_betaz,
        d_Gamx, d_Gamy, d_Gamz,
        // RHS Outputs (Partial)
        d_chi_rhs, 
        d_gxx_rhs, d_gxy_rhs, d_gxz_rhs, d_gyy_rhs, d_gyz_rhs, d_gzz_rhs,
        // Intermediate Outputs
        d_chix, d_chiy, d_chiz,
        d_betaxx, d_betaxy, d_betaxz, d_betayx, d_betayy, d_betayz, d_betazx, d_betazy, d_betazz,
        d_gxxx, d_gxxy, d_gxxz, d_gxyx, d_gxyy, d_gxyz, d_gxzx, d_gxzy, d_gxzz,
        d_gyyx, d_gyyy, d_gyyz, d_gyzx, d_gyzy, d_gyzz, d_gzzx, d_gzzy, d_gzzz,
        d_Gamxxx, d_Gamxxy, d_Gamxxz, d_Gamxyy, d_Gamxyz, d_Gamxzz,
        d_Gamyxx, d_Gamyxy, d_Gamyxz, d_Gamyyy, d_Gamyyz, d_Gamyzz,
        d_Gamzxx, d_Gamzxy, d_Gamzxz, d_Gamzyy, d_Gamzyz, d_Gamzzz,
        d_gupxx, d_gupxy, d_gupxz, d_gupyy, d_gupyz, d_gupzz,
        d_Gmx_Res, d_Gmy_Res, d_Gmz_Res
    );
    CHECK_CUDA(cudaGetLastError());

    // 2. Kernel 2: Curvature & RHS Core
    bssn_rhs_core_kernel<<<grid, block, 0, stream>>>(
        ex[0], ex[1], ex[2], symmetry, lev,
        d_X, d_Y, d_Z,
        d_chi, d_trK,
        d_dxx, d_gxy, d_gxz, d_dyy, d_gyz, d_dzz,
        d_Axx, d_Axy, d_Axz, d_Ayy, d_Ayz, d_Azz,
        d_Gamx, d_Gamy, d_Gamz,
        d_Lap, d_betax, d_betay, d_betaz,
        d_dtSfx, d_dtSfy, d_dtSfz,
        d_rho, d_Sx, d_Sy, d_Sz,
        d_Sxx, d_Sxy, d_Sxz, d_Syy, d_Syz, d_Szz,
        // Intermediates Inputs
        d_chix, d_chiy, d_chiz,
        d_gxxx, d_gxxy, d_gxxz, d_gxyx, d_gxyy, d_gxyz, d_gxzx, d_gxzy, d_gxzz,
        d_gyyx, d_gyyy, d_gyyz, d_gyzx, d_gyzy, d_gyzz, d_gzzx, d_gzzy, d_gzzz,
        d_gupxx, d_gupxy, d_gupxz, d_gupyy, d_gupyz, d_gupzz,
        // Gam (In/Out)
        d_Gamxxx, d_Gamxxy, d_Gamxxz, d_Gamxyy, d_Gamxyz, d_Gamxzz,
        d_Gamyxx, d_Gamyxy, d_Gamyxz, d_Gamyyy, d_Gamyyz, d_Gamyzz,
        d_Gamzxx, d_Gamzxy, d_Gamzxz, d_Gamzyy, d_Gamzyz, d_Gamzzz,
        // Outputs
        d_trK_rhs,
        d_Axx_rhs, d_Axy_rhs, d_Axz_rhs, d_Ayy_rhs, d_Ayz_rhs, d_Azz_rhs,
        d_Gamx_rhs, d_Gamy_rhs, d_Gamz_rhs,
        d_Lap_rhs,
        d_betax_rhs, d_betay_rhs, d_betaz_rhs,
        d_dtSfx_rhs, d_dtSfy_rhs, d_dtSfz_rhs,
        d_Rxx, d_Rxy, d_Rxz, d_Ryy, d_Ryz, d_Rzz
    );
    CHECK_CUDA(cudaGetLastError());

    // 3. Kernel 3: Advection & Dissipation
    bssn_advection_dissipation_kernel<<<grid, block, 0, stream>>>(
        ex[0], ex[1], ex[2], symmetry, lev, eps,
        d_X, d_Y, d_Z,
        d_betax, d_betay, d_betaz,
        d_dxx, d_gxy, d_gxz, d_dyy, d_gyz, d_dzz,
        d_Axx, d_Axy, d_Axz, d_Ayy, d_Ayz, d_Azz,
        d_chi, d_trK,
        d_Gamx, d_Gamy, d_Gamz,
        d_Lap,
        d_dtSfx, d_dtSfy, d_dtSfz,
        // Update RHS
        d_gxx_rhs, d_gxy_rhs, d_gxz_rhs, d_gyy_rhs, d_gyz_rhs, d_gzz_rhs,
        d_Axx_rhs, d_Axy_rhs, d_Axz_rhs, d_Ayy_rhs, d_Ayz_rhs, d_Azz_rhs,
        d_chi_rhs, d_trK_rhs,
        d_Gamx_rhs, d_Gamy_rhs, d_Gamz_rhs,
        d_Lap_rhs,
        d_betax_rhs, d_betay_rhs, d_betaz_rhs,
        d_dtSfx_rhs, d_dtSfy_rhs, d_dtSfz_rhs
    );
    CHECK_CUDA(cudaGetLastError());

    // 4. Kernel 4: Constraints (Optional)
    if (co == 0) {
        bssn_constraints_kernel<<<grid, block, 0, stream>>>(
            ex[0], ex[1], ex[2], symmetry, lev,
            d_X, d_Y, d_Z,
            d_chi, d_trK,
            d_Axx, d_Axy, d_Axz, d_Ayy, d_Ayz, d_Azz,
            d_rho, d_Sx, d_Sy, d_Sz,
            d_gupxx, d_gupxy, d_gupxz, d_gupyy, d_gupyz, d_gupzz,
            d_Rxx, d_Rxy, d_Rxz, d_Ryy, d_Ryz, d_Rzz,
            d_Gamxxx, d_Gamxxy, d_Gamxxz, d_Gamxyy, d_Gamxyz, d_Gamxzz,
            d_Gamyxx, d_Gamyxy, d_Gamyxz, d_Gamyyy, d_Gamyyz, d_Gamyzz,
            d_Gamzxx, d_Gamzxy, d_Gamzxz, d_Gamzyy, d_Gamzyz, d_Gamzzz,
            d_chix, d_chiy, d_chiz,
            d_ham_Res, d_movx_Res, d_movy_Res, d_movz_Res
        );
        CHECK_CUDA(cudaGetLastError());
    }
}

void BssnCudaManager::copy_outputs_to_host(
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
) {
    size_t bytes = current_grid_size * sizeof(double);
    auto cpy_d2h = [&](double* dst, const double* src) {
        if(dst) CHECK_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream));
    };

    cpy_d2h(chi_rhs, d_chi_rhs); cpy_d2h(trK_rhs, d_trK_rhs);
    cpy_d2h(gxx_rhs, d_gxx_rhs); cpy_d2h(gxy_rhs, d_gxy_rhs); cpy_d2h(gxz_rhs, d_gxz_rhs);
    cpy_d2h(gyy_rhs, d_gyy_rhs); cpy_d2h(gyz_rhs, d_gyz_rhs); cpy_d2h(gzz_rhs, d_gzz_rhs);
    cpy_d2h(Axx_rhs, d_Axx_rhs); cpy_d2h(Axy_rhs, d_Axy_rhs); cpy_d2h(Axz_rhs, d_Axz_rhs);
    cpy_d2h(Ayy_rhs, d_Ayy_rhs); cpy_d2h(Ayz_rhs, d_Ayz_rhs); cpy_d2h(Azz_rhs, d_Azz_rhs);
    cpy_d2h(Gamx_rhs, d_Gamx_rhs); cpy_d2h(Gamy_rhs, d_Gamy_rhs); cpy_d2h(Gamz_rhs, d_Gamz_rhs);
    cpy_d2h(Lap_rhs, d_Lap_rhs);
    cpy_d2h(betax_rhs, d_betax_rhs); cpy_d2h(betay_rhs, d_betay_rhs); cpy_d2h(betaz_rhs, d_betaz_rhs);
    cpy_d2h(dtSfx_rhs, d_dtSfx_rhs); cpy_d2h(dtSfy_rhs, d_dtSfy_rhs); cpy_d2h(dtSfz_rhs, d_dtSfz_rhs);

    // Diagnostics Copy
    cpy_d2h(Gamxxx, d_Gamxxx); cpy_d2h(Gamxxy, d_Gamxxy); cpy_d2h(Gamxxz, d_Gamxxz);
    cpy_d2h(Gamxyy, d_Gamxyy); cpy_d2h(Gamxyz, d_Gamxyz); cpy_d2h(Gamxzz, d_Gamxzz);
    cpy_d2h(Gamyxx, d_Gamyxx); cpy_d2h(Gamyxy, d_Gamyxy); cpy_d2h(Gamyxz, d_Gamyxz);
    cpy_d2h(Gamyyy, d_Gamyyy); cpy_d2h(Gamyyz, d_Gamyyz); cpy_d2h(Gamyzz, d_Gamyzz);
    cpy_d2h(Gamzxx, d_Gamzxx); cpy_d2h(Gamzxy, d_Gamzxy); cpy_d2h(Gamzxz, d_Gamzxz);
    cpy_d2h(Gamzyy, d_Gamzyy); cpy_d2h(Gamzyz, d_Gamzyz); cpy_d2h(Gamzzz, d_Gamzzz);
    cpy_d2h(Rxx, d_Rxx); cpy_d2h(Rxy, d_Rxy); cpy_d2h(Rxz, d_Rxz);
    cpy_d2h(Ryy, d_Ryy); cpy_d2h(Ryz, d_Ryz); cpy_d2h(Rzz, d_Rzz);

    if (co == 0) {
        cpy_d2h(ham_Res, d_ham_Res);
        cpy_d2h(movx_Res, d_movx_Res); cpy_d2h(movy_Res, d_movy_Res); cpy_d2h(movz_Res, d_movz_Res);
        cpy_d2h(Gmx_Res, d_Gmx_Res); cpy_d2h(Gmy_Res, d_Gmy_Res); cpy_d2h(Gmz_Res, d_Gmz_Res);
    }

    // 阻塞 Stream，直到拷贝完成（或者可以在外部 wait）
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

// 主入口实现
void BssnCudaManager::compute_rhs_bssn(
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
) {
    // 1. 检查大小并分配显存 (Pool)
    resize(ex);

    // 2. 拷贝输入 (H2D)
    copy_inputs_to_device(
        chi, trK, dxx, gxy, gxz, dyy, gyz, dzz,
        Axx, Axy, Axz, Ayy, Ayz, Azz,
        Gamx, Gamy, Gamz, Lap,
        betax, betay, betaz, dtSfx, dtSfy, dtSfz,
        rho, Sx, Sy, Sz, Sxx, Sxy, Sxz, Syy, Syz, Szz,
        X, Y, Z
    );

    // 3. 执行 Kernels
    run_kernels(const_cast<int*>(ex), symmetry, lev, eps, co);

    // 4. 拷贝输出 (D2H)
    copy_outputs_to_host(
        chi_rhs, trK_rhs,
        gxx_rhs, gxy_rhs, gxz_rhs, gyy_rhs, gyz_rhs, gzz_rhs,
        Axx_rhs, Axy_rhs, Axz_rhs, Ayy_rhs, Ayz_rhs, Azz_rhs,
        Gamx_rhs, Gamy_rhs, Gamz_rhs,
        Lap_rhs, 
        betax_rhs, betay_rhs, betaz_rhs,
        dtSfx_rhs, dtSfy_rhs, dtSfz_rhs,
        Gamxxx, Gamxxy, Gamxxz, Gamxyy, Gamxyz, Gamxzz,
        Gamyxx, Gamyxy, Gamyxz, Gamyyy, Gamyyz, Gamyzz,
        Gamzxx, Gamzxy, Gamzxz, Gamzyy, Gamzyz, Gamzzz,
        Rxx, Rxy, Rxz, Ryy, Ryz, Rzz,
        ham_Res, movx_Res, movy_Res, movz_Res,
        Gmx_Res, Gmy_Res, Gmz_Res,
        co
    );
}

static BssnCudaManager* g_bssn_manager = nullptr;

void cleanup_bssn_gpu() {
    if (g_bssn_manager) {
        delete g_bssn_manager;
        g_bssn_manager = nullptr;
    }
}

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
) {
    if (!g_bssn_manager) {
        g_bssn_manager = new BssnCudaManager();
    }

    g_bssn_manager->compute_rhs_bssn(
        ex, T, X, Y, Z,
        chi, trK,
        dxx, gxy, gxz,
        dyy, gyz, dzz,
        Axx, Axy, Axz,
        Ayy, Ayz, Azz,
        Gamx, Gamy, Gamz,
        Lap,
        betax, betay, betaz,
        dtSfx, dtSfy, dtSfz,
        chi_rhs, trK_rhs,
        gxx_rhs, gxy_rhs, gxz_rhs,
        gyy_rhs, gyz_rhs, gzz_rhs,
        Axx_rhs, Axy_rhs, Axz_rhs,
        Ayy_rhs, Ayz_rhs, Azz_rhs,
        Gamx_rhs, Gamy_rhs, Gamz_rhs,
        Lap_rhs,
        betax_rhs, betay_rhs, betaz_rhs,
        dtSfx_rhs, dtSfy_rhs, dtSfz_rhs,
        rho, Sx, Sy, Sz,
        Sxx, Sxy, Sxz,
        Syy, Syz, Szz,
        Gamxxx, Gamxxy, Gamxxz,
        Gamxyy, Gamxyz, Gamxzz,
        Gamyxx, Gamyxy, Gamyxz,
        Gamyyy, Gamyyz, Gamyzz,
        Gamzxx, Gamzxy, Gamzxz,
        Gamzyy, Gamzyz, Gamzzz,
        Rxx, Rxy, Rxz,
        Ryy, Ryz, Rzz,
        ham_Res, movx_Res, movy_Res, movz_Res,
        Gmx_Res, Gmy_Res, Gmz_Res,
        symmetry, lev, eps, co
    );

    return 0;
}