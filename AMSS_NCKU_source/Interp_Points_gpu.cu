#include "Interp_Points_gpu.h"

#include "fmisc.h"

__global__ void Batch_Interp_Kernel(InterpTask* tasks, double* results, int num_tasks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tasks) return;

    InterpTask t = tasks[idx];
    double val = 0.0;

    global_interp_device(
        t.shape, t.d_X, t.d_Y, t.d_Z,
        t.d_field, 
        &val, 
        t.px, t.py, t.pz, 
        t.ordn, t.d_SoA, t.symmetry
    );

    // 将结果写入临时结果数组
    results[idx] = val;
}

GpuBatchInterp::GpuBatchInterp() {}

GpuBatchInterp::~GpuBatchInterp() {
    clear_gpu_memory();
}

template <typename T>
T* GpuBatchInterp::upload_ptr(T* host_ptr, size_t size_bytes) {
    if (ptr_cache.find((void*)host_ptr) != ptr_cache.end()) {
        return (T*)ptr_cache[(void*)host_ptr];
    }

    T* device_ptr = nullptr;
    cudaMalloc(&device_ptr, size_bytes);
    cudaMemcpy(device_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice);

    ptr_cache[(void*)host_ptr] = (double*)device_ptr;
    trash_bin.push_back(device_ptr);

    return device_ptr;
}

void GpuBatchInterp::clear_gpu_memory() {
    for (void* ptr : trash_bin) {
        cudaFree(ptr);
    }
    trash_bin.clear();
    ptr_cache.clear();
    tasks.clear();
}

// ----------------------

void GpuBatchInterp::add_task(
    int* shape,
    double** X,
    double* SoA,
    double* field_data,
    double* pox,
    int ordn,
    int symmetry,
    int result_idx)
{
    InterpTask t;

    t.shape[0] = shape[0];
    t.shape[1] = shape[1];
    t.shape[2] = shape[2];

    size_t sx = shape[0] * sizeof(double);
    size_t sy = shape[1] * sizeof(double);
    size_t sz = shape[2] * sizeof(double);

    t.d_X   = upload_ptr(X[0], sx);
    t.d_Y   = upload_ptr(X[1], sy);
    t.d_Z   = upload_ptr(X[2], sz);
    t.d_SoA = upload_ptr(SoA, 3 * sizeof(double));

    size_t sf = (size_t)shape[0] * shape[1] * shape[2] * sizeof(double);
    t.d_field = upload_ptr(field_data, sf);

    t.px = pox[0];
    t.py = pox[1];
    t.pz = pox[2];

    t.ordn = ordn;
    t.symmetry = symmetry;
    t.result_idx = result_idx;

    tasks.push_back(t);
}

// ----------------------

void GpuBatchInterp::execute(double* Shellf_host) {
    if (tasks.empty()) return;
    size_t n = tasks.size();

    InterpTask* d_tasks = nullptr;
    double* d_results = nullptr;

    cudaMalloc(&d_tasks, n * sizeof(InterpTask));
    cudaMalloc(&d_results, n * sizeof(double));

    cudaMemcpy(d_tasks,
               tasks.data(),
               n * sizeof(InterpTask),
               cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    Batch_Interp_Kernel<<<gridSize, blockSize>>>(d_tasks, d_results, n);
    cudaDeviceSynchronize();

    std::vector<double> host_results(n);

    cudaMemcpy(host_results.data(),
               d_results,
               n * sizeof(double),
               cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; i++) {
        Shellf_host[tasks[i].result_idx] = host_results[i];
    }

    cudaFree(d_tasks);
    cudaFree(d_results);

    tasks.clear();
}