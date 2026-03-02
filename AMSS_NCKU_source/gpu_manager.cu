#include "gpu_manager.h"
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>

#include <cassert>

struct GPUManager::Impl {
    static constexpr int NUM_STREAMS = 16;
    cudaStream_t stream_pool[NUM_STREAMS];
    std::atomic<unsigned int> stream_idx{0};
    cudaStream_t memory_stream;
};

// 单例获取
GPUManager& GPUManager::getInstance() {
    static GPUManager instance;
    return instance;
}

// 构造与析构
GPUManager::GPUManager() : pimpl(new Impl()) {
    for (int i = 0; i < Impl::NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&pimpl->stream_pool[i]));
    }
    CUDA_CHECK(cudaStreamCreate(&pimpl->memory_stream));
}

GPUManager::~GPUManager() {
    for (int i = 0; i < Impl::NUM_STREAMS; ++i) {
        cudaStreamDestroy(pimpl->stream_pool[i]);
    }
    cudaStreamDestroy(pimpl->memory_stream);
    delete pimpl;
}

template<typename T>
T* GPUManager::allocate_device_memory(size_t num_elements) {
    T* d_ptr = nullptr;
    CUDA_CHECK(cudaMallocAsync((void**)&d_ptr, num_elements * sizeof(T), pimpl->memory_stream));
    CUDA_CHECK(cudaMemsetAsync(d_ptr, 0, num_elements * sizeof(T), pimpl->memory_stream));
    CUDA_CHECK(cudaStreamSynchronize(pimpl->memory_stream));
    return d_ptr;
}

template<typename T>
void GPUManager::free_device_memory(T* d_ptr, size_t num_elements = 0) {
    if (d_ptr == nullptr) return;
    CUDA_CHECK(cudaFreeAsync(d_ptr, pimpl->memory_stream));
    CUDA_CHECK(cudaStreamSynchronize(pimpl->memory_stream));
}

cudaStream_t GPUManager::get_stream() {
    unsigned int cur = pimpl->stream_idx.fetch_add(1);
    return pimpl->stream_pool[cur % Impl::NUM_STREAMS];
}

template<typename T>
void GPUManager::sync_to_gpu(const T* h_ptr, T* d_ptr, size_t num_elements) {
    CUDA_CHECK(cudaMemcpyAsync(d_ptr, h_ptr, num_elements * sizeof(T), cudaMemcpyHostToDevice, getInstance().pimpl->memory_stream));
    CUDA_CHECK(cudaStreamSynchronize(getInstance().pimpl->memory_stream));
}

template<typename T>
void GPUManager::sync_to_cpu(T* h_ptr, const T* d_ptr, size_t num_elements) {
    CUDA_CHECK(cudaMemcpyAsync(h_ptr, d_ptr, num_elements * sizeof(T), cudaMemcpyDeviceToHost, getInstance().pimpl->memory_stream));
    CUDA_CHECK(cudaStreamSynchronize(getInstance().pimpl->memory_stream));
}

void GPUManager::synchronize_memory() {
    CUDA_CHECK(cudaStreamSynchronize(pimpl->memory_stream));
}

void GPUManager::synchronize_all() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

void GPUManager::synchronize_stream(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

template double* GPUManager::allocate_device_memory<double>(size_t num_elements);
template int* GPUManager::allocate_device_memory<int>(size_t num_elements);
template void GPUManager::free_device_memory<double>(double* d_ptr, size_t num_elements);
template void GPUManager::free_device_memory<int>(int* d_ptr, size_t num_elements);
template void GPUManager::sync_to_gpu<double>(const double* h_ptr, double* d_ptr, size_t num_elements);
template void GPUManager::sync_to_gpu<int>(const int* h_ptr, int* d_ptr, size_t num_elements);
template void GPUManager::sync_to_cpu<double>(double* h_ptr, const double* d_ptr, size_t num_elements);
template void GPUManager::sync_to_cpu<int>(int* h_ptr, const int* d_ptr, size_t num_elements);