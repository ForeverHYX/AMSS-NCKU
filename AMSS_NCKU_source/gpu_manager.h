#ifndef GPU_MANAGER_H
#define GPU_MANAGER_H

#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

class GPUManager {
private:
    // 声明一个不透明的结构体指针，具体实现在 .cu 中
    struct Impl;
    Impl* pimpl;

    GPUManager();
    ~GPUManager();

public:
    static GPUManager& getInstance();

    template<typename T> T* allocate_device_memory(size_t num_elements);
    template<typename T> void free_device_memory(T* d_ptr, size_t num_elements);
    void clear_pool();

    template<typename T> static void sync_to_gpu(const T* h_ptr, T* d_ptr, size_t num_elements);
    template<typename T> static void sync_to_cpu(T* h_ptr, const T* d_ptr, size_t num_elements);

    cudaStream_t get_stream();
    void synchronize_all();
    void synchronize_memory();
    static void synchronize_stream(cudaStream_t stream);
};

extern template double* GPUManager::allocate_device_memory<double>(size_t num_elements);
extern template int* GPUManager::allocate_device_memory<int>(size_t num_elements);
extern template void GPUManager::free_device_memory<double>(double* d_ptr, size_t num_elements);
extern template void GPUManager::free_device_memory<int>(int* d_ptr, size_t num_elements);
extern template void GPUManager::sync_to_gpu<double>(const double* h_ptr, double* d_ptr, size_t num_elements);
extern template void GPUManager::sync_to_gpu<int>(const int* h_ptr, int* d_ptr, size_t num_elements);
extern template void GPUManager::sync_to_cpu<double>(double* h_ptr, const double* d_ptr, size_t num_elements);
extern template void GPUManager::sync_to_cpu<int>(int* h_ptr, const int* d_ptr, size_t num_elements);

#endif /* GPU_MANAGER_H */