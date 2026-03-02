#include <cuda_runtime.h>
#include <math.h>

#include <iostream>
#include "gpu_manager.h"
#include "fmisc_gpu.cuh"

__global__ void lowerboundset_kernel(int n, double* chi0, double TINNY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        if (chi0[idx] < TINNY) {
            chi0[idx] = TINNY;
        }
    }
}

void gpu_lowerboundset_launch(
    cudaStream_t &stream,
    int ex[3],
    double* d_chi0, double TINNY
) {
    int n = ex[0] * ex[1] * ex[2];
    int block = 256;
    int grid = (n + block - 1) / block;

    lowerboundset_kernel<<<grid, block, 0, stream>>>(n, d_chi0, TINNY);
}

__global__ void gpu_pack_kernel(
    const double* __restrict__ src_3d, double* __restrict__ dst_1d,
    int src_nx, int src_ny, 
    int dst_nx, int dst_ny, int dst_nz,
    int off_x, int off_y, int off_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dst_nx * dst_ny * dst_nz;
    if (idx >= total) return;
    
    int i = idx % dst_nx;
    int j = (idx / dst_nx) % dst_ny;
    int k = idx / (dst_nx * dst_ny);
    
    int src_idx = (k + off_z) * (src_nx * src_ny) + (j + off_y) * src_nx + (i + off_x);
    dst_1d[idx] = src_3d[src_idx];
}

__global__ void gpu_unpack_kernel(
    const double* __restrict__ src_1d, double* __restrict__ dst_3d,
    int dst_nx, int dst_ny, 
    int src_nx, int src_ny, int src_nz,
    int off_x, int off_y, int off_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = src_nx * src_ny * src_nz;
    if (idx >= total) return;
    
    int i = idx % src_nx;
    int j = (idx / src_nx) % src_ny;
    int k = idx / (src_nx * src_ny);
    
    int dst_idx = (k + off_z) * (dst_nx * dst_ny) + (j + off_y) * dst_nx + (i + off_x);
    dst_3d[dst_idx] = src_1d[idx];
}

void gpu_pack_launch(
	cudaStream_t stream, const double* d_src_3d, double* d_dst_1d,
	int src_nx, int src_ny, int dst_nx, int dst_ny, int dst_nz,
	int off_x, int off_y, int off_z
) {
	int n = dst_nx * dst_ny * dst_nz;
	int block = 256;
	int grid = (n + block - 1) / block;
	gpu_pack_kernel<<<grid, block, 0, stream>>>(d_src_3d, d_dst_1d, src_nx, src_ny, dst_nx, dst_ny, dst_nz, off_x, off_y, off_z);
}

void gpu_unpack_launch(
	cudaStream_t stream, const double* d_src_1d, double* d_dst_3d,
	int dst_nx, int dst_ny, int src_nx, int src_ny, int src_nz,
	int off_x, int off_y, int off_z
) {
	int n = src_nx * src_ny * src_nz;
	int block = 256;
	int grid = (n + block - 1) / block;
	gpu_unpack_kernel<<<grid, block, 0, stream>>>(d_src_1d, d_dst_3d, dst_nx, dst_ny, src_nx, src_ny, src_nz, off_x, off_y, off_z);
}

// =====================================================================
// Time Level Interpolation Kernels
// =====================================================================

__global__ void average_kernel(int n, const double* f1, const double* f2, double* fout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fout[idx] = 0.5 * (f1[idx] + f2[idx]);
    }
}

__global__ void average3_kernel(int n, const double* f1, const double* f2, double* fout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fout[idx] = 0.75 * f1[idx] + 0.25 * f2[idx];
    }
}

__global__ void average2_kernel(int n, const double* f1, const double* f2, const double* f3, double* fout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fout[idx] = (3.0 / 8.0) * f1[idx] + (3.0 / 4.0) * f2[idx] - (1.0 / 8.0) * f3[idx];
    }
}

__global__ void average2p_kernel(int n, const double* f1, const double* f2, const double* f3, double* fout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fout[idx] = (21.0 / 32.0) * f1[idx] + (7.0 / 16.0) * f2[idx] - (3.0 / 32.0) * f3[idx];
    }
}

__global__ void average2m_kernel(int n, const double* f1, const double* f2, const double* f3, double* fout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fout[idx] = (5.0 / 32.0) * f1[idx] + (15.0 / 16.0) * f2[idx] - (3.0 / 32.0) * f3[idx];
    }
}

void gpu_average_launch(cudaStream_t stream, const int ext[3], const double* d_f1, const double* d_f2, double* d_fout) {
    int n = ext[0] * ext[1] * ext[2];
    int block = 256;
    int grid = (n + block - 1) / block;
    average_kernel<<<grid, block, 0, stream>>>(n, d_f1, d_f2, d_fout);
}

void gpu_average3_launch(cudaStream_t stream, const int ext[3], const double* d_f1, const double* d_f2, double* d_fout) {
    int n = ext[0] * ext[1] * ext[2];
    int block = 256;
    int grid = (n + block - 1) / block;
    average3_kernel<<<grid, block, 0, stream>>>(n, d_f1, d_f2, d_fout);
}

void gpu_average2_launch(cudaStream_t stream, const int ext[3], const double* d_f1, const double* d_f2, const double* d_f3, double* d_fout) {
    int n = ext[0] * ext[1] * ext[2];
    int block = 256;
    int grid = (n + block - 1) / block;
    average2_kernel<<<grid, block, 0, stream>>>(n, d_f1, d_f2, d_f3, d_fout);
}

void gpu_average2p_launch(cudaStream_t stream, const int ext[3], const double* d_f1, const double* d_f2, const double* d_f3, double* d_fout) {
    int n = ext[0] * ext[1] * ext[2];
    int block = 256;
    int grid = (n + block - 1) / block;
    average2p_kernel<<<grid, block, 0, stream>>>(n, d_f1, d_f2, d_f3, d_fout);
}

void gpu_average2m_launch(cudaStream_t stream, const int ext[3], const double* d_f1, const double* d_f2, const double* d_f3, double* d_fout) {
    int n = ext[0] * ext[1] * ext[2];
    int block = 256;
    int grid = (n + block - 1) / block;
    average2m_kernel<<<grid, block, 0, stream>>>(n, d_f1, d_f2, d_f3, d_fout);
}

__global__ void global_interp_kernel(
    int NN, int DIM,
    double* d_XX_0, double* d_XX_1, double* d_XX_2,
    int ex0, int ex1, int ex2, 
    double* d_X_0, double* d_X_1, double* d_X_2,
    double* d_field,
    double llb_0, double llb_1, double llb_2,
    double uub_0, double uub_1, double uub_2,
    int ordn, double SoA_0, double SoA_1, double SoA_2, int Symmetry,
    int var_idx, int num_var,
    double* d_shellf, int* d_weight
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= NN) return;

    // 获取当前点的坐标
    double px = d_XX_0[j];
    double py = d_XX_1[j];
    double pz = d_XX_2[j];

    // 边界检查（Bounding Box 判断）
    if (px < llb_0 || px > uub_0) return;
    if (py < llb_1 || py > uub_1) return;
    if (pz < llb_2 || pz > uub_2) return;

    // 组装传给已有插值库的指针
    double* d_X_arr[3] = {d_X_0, d_X_1, d_X_2};
	double SoA_arr[3] = {SoA_0, SoA_1, SoA_2};
	const int ex[3] = {ex0, ex1, ex2};

    // 调用你们原有的设备端插值函数
    double val = 0.0;
    global_interp_device(
        ex, d_X_arr[0], d_X_arr[1], d_X_arr[2],
        d_field, &val,
        px, py, pz,
        ordn, SoA_arr, Symmetry
    );

    // 将结果原子累加到对应位置（处理 Ghost Zone 多个 Block 重叠的情况）
    atomicAdd(&d_shellf[j * num_var + var_idx], val);

    if (var_idx == 0) {
        atomicAdd(&d_weight[j], 1);
    }
}

void gpu_global_interp_launch(
	cudaStream_t stream,
    int NN, int DIM,
    double* d_XX_0, double* d_XX_1, double* d_XX_2,
    int shape_0, int shape_1, int shape_2,
    double* d_X_0, double* d_X_1, double* d_X_2,
    double* d_field,
    double llb_0, double llb_1, double llb_2,
    double uub_0, double uub_1, double uub_2,
    int ordn, double SoA_0, double SoA_1, double SoA_2, 
    int Symmetry, int var_idx, int num_var,
    double* d_shellf, int* d_weight
) {
	int blockSize = 256;
    int gridSize = (NN + blockSize - 1) / blockSize;

	global_interp_kernel<<<gridSize, blockSize, 0, stream>>>(
		NN, DIM, 
        d_XX_0, d_XX_1, d_XX_2, 
        shape_0, shape_1, shape_2, d_X_0, d_X_1, d_X_2, d_field,
        llb_0, llb_1, llb_2, uub_0, uub_1, uub_2,
        ordn, SoA_0, SoA_1, SoA_2, Symmetry, 
        var_idx, num_var, d_shellf, d_weight
	);
}

__global__ void l2normhelper_kernel(
    const double* __restrict__ f,
    int imin, int imax,
    int jmin, int jmax,
    int kmin, int kmax,
    int nx, int ny, int nz,
    double* __restrict__ d_out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + imin;
    int j = blockIdx.y * blockDim.y + threadIdx.y + jmin;
    int k = blockIdx.z * blockDim.z + threadIdx.z + kmin;

    double my_val = 0.0;

    if (i <= imax && j <= jmax && k <= kmax) {
        long long idx = (long long)i + (long long)j * nx + (long long)k * nx * ny;
        double val = f[idx];
        my_val = val * val;
    }

    // Warp 内规约求和 (假设你的 warpReduceSum 内部使用了正确的 __shfl_down_sync)
    my_val = warpReduceSum(my_val);

    // 修正: 计算当前线程在 Block 内的 1D 线性 ID
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    // 只有每个 Warp 的第 0 个线程 (线性 ID 是 warpSize 的整数倍) 负责写入
    if ((tid % warpSize) == 0) {
        atomicAdd(d_out, my_val);
    }
}

void gpu_l2normhelper_launch(
	cudaStream_t stream, 
	const int* ex, 
	const double* X, const double* Y, const double* Z,
	double xmin, double ymin, double zmin,
	double xmax, double ymax, double zmax,
	const double* d_f, double& f_out, int gw
) {
    double dX = X[1] - X[0];
    double dY = Y[1] - Y[0];
    double dZ = Z[1] - Z[0];

    // 将 Fortran 的 1-indexed 逻辑转换为 C/C++ 的 0-indexed 逻辑
    int imin = gw;
    int jmin = gw;
    int kmin = gw;

    int imax = ex[0] - gw - 1;
    int jmax = ex[1] - gw - 1;
    int kmax = ex[2] - gw - 1;

    // 边界判断 (与 Fortran 逻辑完全一致)
    if (fabs(X[ex[0] - 1] - xmax) < dX) imax = ex[0] - 1;
    if (fabs(Y[ex[1] - 1] - ymax) < dY) jmax = ex[1] - 1;
    if (fabs(Z[ex[2] - 1] - zmax) < dZ) kmax = ex[2] - 1;
    
    if (fabs(X[0] - xmin) < dX) imin = 0;
    if (fabs(Y[0] - ymin) < dY) jmin = 0;
    if (fabs(Z[0] - zmin) < dZ) kmin = 0;

    int nx_proc = imax - imin + 1;
    int ny_proc = jmax - jmin + 1;
    int nz_proc = kmax - kmin + 1;

    if (nx_proc <= 0 || ny_proc <= 0 || nz_proc <= 0) {
        f_out = 0.0;
        return;
    }

    double* d_sum = GPUManager::getInstance().allocate_device_memory<double>(1);

    // 设置线程块大小，通常 8x8x8 = 512 线程效率较好
    dim3 blockDim(8, 8, 8); 
    dim3 gridDim((nx_proc + blockDim.x - 1) / blockDim.x,
                 (ny_proc + blockDim.y - 1) / blockDim.y,
                 (nz_proc + blockDim.z - 1) / blockDim.z);

    // 启动 Kernel
    l2normhelper_kernel<<<gridDim, blockDim, 0, stream>>>(
        d_f, 
        imin, imax, jmin, jmax, kmin, kmax, 
        ex[0], ex[1], ex[2], 
        d_sum
    );

    // 强制同步：由于后续的 MPI_Allreduce 立刻需要用到 f_out 的 CPU 数据，这里必须等待 GPU 计算并拷贝完成
    cudaStreamSynchronize(stream);

    double h_sum = 0.0;
    // 将结果拷回 CPU
    GPUManager::getInstance().sync_to_cpu(&h_sum, d_sum, 1);
    GPUManager::getInstance().free_device_memory(d_sum, 1);

    f_out = h_sum * dX * dY * dZ;
}

__global__ void global_interp_amr_kernel(
    int active_count, int DIM,
    int* d_active_indices,
    double* d_XX_0, double* d_XX_1, double* d_XX_2,
    int ex0, int ex1, int ex2, 
    double* d_X_0, double* d_X_1, double* d_X_2,
    double* d_field,
    int ordn, double SoA_0, double SoA_1, double SoA_2, int Symmetry,
    int var_idx, int num_var,
    double* d_shellf
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_count) return;

    // 获取该点在全局 n=1000 数组中的真实索引
    int j = d_active_indices[idx]; 

    double px = d_XX_0[j];
    double py = d_XX_1[j];
    double pz = d_XX_2[j];

    double* d_X_arr[3] = {d_X_0, d_X_1, d_X_2};
    double SoA_arr[3] = {SoA_0, SoA_1, SoA_2};
    const int ex[3] = {ex0, ex1, ex2};

    double val = 0.0;
    // 调用现有的设备端插值核心
    global_interp_device(
        ex, d_X_arr[0], d_X_arr[1], d_X_arr[2],
        d_field, &val,
        px, py, pz,
        ordn, SoA_arr, Symmetry
    );

    // 直接赋值，不再需要 atomicAdd，因为每个点已被 CPU 保证全局唯一认领
    d_shellf[j * num_var + var_idx] = val;
}

void gpu_global_interp_amr_launch(
    cudaStream_t stream,
    int active_count, int DIM,
    int* d_active_indices,
    double* d_XX_0, double* d_XX_1, double* d_XX_2,
    int shape_0, int shape_1, int shape_2,
    double* d_X_0, double* d_X_1, double* d_X_2,
    double* d_field,
    int ordn, double SoA_0, double SoA_1, double SoA_2, 
    int Symmetry, int var_idx, int num_var,
    double* d_shellf
) {
    if (active_count == 0) return;
    int blockSize = 256;
    int gridSize = (active_count + blockSize - 1) / blockSize;

    global_interp_amr_kernel<<<gridSize, blockSize, 0, stream>>>(
        active_count, DIM, d_active_indices,
        d_XX_0, d_XX_1, d_XX_2, 
        shape_0, shape_1, shape_2, d_X_0, d_X_1, d_X_2, d_field,
        ordn, SoA_0, SoA_1, SoA_2, Symmetry, 
        var_idx, num_var, d_shellf
    );
}