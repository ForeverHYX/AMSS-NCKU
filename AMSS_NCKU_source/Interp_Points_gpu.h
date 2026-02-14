#ifndef INTERP_POINTS_GPU_H
#define INTERP_POINTS_GPU_H

#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <iostream>

// GPU 任务描述符 (POD 类型，可直接传给 Kernel)
struct InterpTask {
    // 几何信息 (Block)
    int shape[3];
    double *d_X, *d_Y, *d_Z, *d_SoA;
    
    // 场信息
    double *d_field; 
    
    // 点信息
    double px, py, pz;
    
    // 参数
    int ordn;
    int symmetry;
    
    // 结果回填索引 (对应 shellf 的下标)
    int result_idx; 
};

class GpuBatchInterp {
private:
    std::map<void*, double*> ptr_cache;
    std::vector<void*> trash_bin;
    std::vector<InterpTask> tasks;

    template <typename T>
    T* upload_ptr(T* host_ptr, size_t size_bytes);

public:
    GpuBatchInterp();
    ~GpuBatchInterp();

    void add_task(
        int* shape,
        double** X,
        double* SoA,
        double* field_data,
        double* pox,
        int ordn,
        int symmetry,
        int result_idx
    );

    void execute(double* Shellf_host);
    void clear_gpu_memory();
};

#endif