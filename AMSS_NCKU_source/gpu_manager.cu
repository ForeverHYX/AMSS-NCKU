#include "gpu_manager.h"
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>

#include <cassert>

// 真正的管理器数据结构，只有 nvcc 编译器知道它的存在
struct GPUManager::Impl {
    std::unordered_map<size_t, std::vector<double*>> memory_pool;
    std::mutex pool_mutex;

    static constexpr int NUM_STREAMS = 16; // 强烈建议开多个流以支持并发
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
    clear_pool();
    for (int i = 0; i < Impl::NUM_STREAMS; ++i) {
        cudaStreamDestroy(pimpl->stream_pool[i]);
    }
    cudaStreamDestroy(pimpl->memory_stream);
    delete pimpl;
}

// 显存池操作路由到 pimpl
double* GPUManager::allocate_device_memory(size_t num_elements) {
    double* d_ptr = nullptr;
    CUDA_CHECK(cudaMallocAsync((void**)&d_ptr, num_elements * sizeof(double), pimpl->memory_stream));
    CUDA_CHECK(cudaMemsetAsync(d_ptr, 0, num_elements * sizeof(double), pimpl->memory_stream));
    CUDA_CHECK(cudaStreamSynchronize(pimpl->memory_stream));
    return d_ptr;
}

void GPUManager::free_device_memory(double* d_ptr, size_t num_elements) {
    if (d_ptr == nullptr) return;
    CUDA_CHECK(cudaFreeAsync(d_ptr, pimpl->memory_stream));
    CUDA_CHECK(cudaStreamSynchronize(pimpl->memory_stream));
}

void GPUManager::clear_pool() {
    pimpl->memory_pool.clear();
}

cudaStream_t GPUManager::get_stream() {
    unsigned int cur = pimpl->stream_idx.fetch_add(1);
    return pimpl->stream_pool[cur % Impl::NUM_STREAMS];
}

void GPUManager::synchronize_all() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

void GPUManager::sync_to_gpu(const double* h_ptr, double* d_ptr, size_t num_elements) {
    CUDA_CHECK(cudaMemcpyAsync(d_ptr, h_ptr, num_elements * sizeof(double), cudaMemcpyHostToDevice, getInstance().pimpl->memory_stream));
    CUDA_CHECK(cudaStreamSynchronize(getInstance().pimpl->memory_stream));
}

void GPUManager::sync_to_cpu(double* h_ptr, const double* d_ptr, size_t num_elements) {
    CUDA_CHECK(cudaMemcpyAsync(h_ptr, d_ptr, num_elements * sizeof(double), cudaMemcpyDeviceToHost, getInstance().pimpl->memory_stream));
    CUDA_CHECK(cudaStreamSynchronize(getInstance().pimpl->memory_stream));
}