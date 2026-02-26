#ifndef PROLONGRESTRICT_GPU_MANAGER_H
#define PROLONGRESTRICT_GPU_MANAGER_H

#include <vector>
#include <cuda_runtime.h>

// 任务类型
enum TaskType {
    TASK_PROLONG = 3,
    TASK_RESTRICT = 2
};

// 发给 GPU 的任务结构体 (POD)
// 里面的指针将指向 Manager 内部分配的 GPU 显存，而不是用户传入的 Host 指针
struct ProlongRestrictTask {
    TaskType type;
    double* src_ptr;   // 指向 d_input_buffer_ 中的偏移位置
    double* dst_ptr;   // 指向 d_output_buffer_ 中的偏移位置
    
    // 几何参数
    double llbc[3], uubc[3]; int extc[3];
    double llbf[3], uubf[3]; int extf[3];
    double llbt[3], uubt[3]; // Target bounds
    double SoA[3]; int Symmetry;
};

// 用于记录“计算完后数据该拷回哪里”的元数据
struct OutputCopyBackInfo {
    double* host_dst_ptr; // 用户原本提供的 CPU 目标地址
    size_t offset;        // 在 Manager 输出 buffer 中的偏移量
    size_t size;          // 数据大小 (元素个数)
};

class ProlongRestrictManager {
public:
    ProlongRestrictManager();
    ~ProlongRestrictManager();

    // 修改：这里接收的是 Host 指针
    void AddTask(
        TaskType type,
        double* h_src, double* h_dst, // <--- CPU 指针
        const double* llbc, const double* uubc, const int* extc,
        const double* llbf, const double* uubf, const int* extf,
        const double* llbt, const double* uubt,
        const double* SoA, int Symmetry
    );

    // 执行：H2D -> Kernel -> D2H -> Scatter
    void ExecuteBatch();

    void Clear();

private:
    // Host 端暂存区 (Staging Buffers)
    // 使用 vector 方便动态扩容，直接 memcpy 到这里
    std::vector<double> h_input_staging_; 
    std::vector<double> h_output_staging_;

    // 记录结果回传任务
    std::vector<OutputCopyBackInfo> output_infos_;

    // Host 端任务列表 (准备发给 GPU 的)
    std::vector<ProlongRestrictTask> host_tasks_;

    // Device 端内存池
    double* d_input_buffer_;
    double* d_output_buffer_;
    ProlongRestrictTask* d_tasks_; // GPU 上的任务列表

    // 容量记录 (用于避免频繁 malloc)
    size_t d_in_cap_, d_out_cap_, d_task_cap_;

    // 辅助：确保 Device 内存够大
    void ReserveDeviceMemory(size_t in_size, size_t out_size, size_t num_tasks);
};

#endif