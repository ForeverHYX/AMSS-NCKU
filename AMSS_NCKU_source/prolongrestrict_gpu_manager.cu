#include "prolongrestrict_gpu_manager.h"

#include "prolongrestrict.h"

#include "gpu_manager.h"

#include <cstdio>
#include <iostream>

// =========================================================
// 1. Batched Kernel 实现
// =========================================================

// 这个 Kernel 采用 "One Block per Task" 的策略
// blockIdx.x 对应任务 ID
// threadIdx 负责并行处理该任务内的网格点
__global__ void batched_prolong_restrict_kernel(ProlongRestrictTask* tasks, int num_tasks) {
    int task_idx = blockIdx.x;
    if (task_idx >= num_tasks) return;

    // 读取当前任务的元数据到寄存器
    ProlongRestrictTask t = tasks[task_idx];

    // 计算该任务需要遍历的网格范围
    // 我们需要重新计算一遍 imino, imaxo 等范围，以确定循环边界
    // 为了节省寄存器和代码复用，我们可以直接计算出总的迭代点数，然后用 stride loop
    
    // --- Step 1: 确定遍历空间 ---
    // Prolong: 遍历 Fine Grid 空间 (extf) 的一个子集 (由 llbt/uubt 决定)
    // Restrict: 遍历 Coarse Grid 空间 (extc) 的一个子集
    
    // 注意：这里的逻辑必须与 device 函数内部的范围检查一致。
    // 为了简化，我们让线程遍历由 llbt/uubt 确定的 "Target Box" 范围。
    // 我们需要把 llbt/uubt 转换成整数索引范围。
    
    double CD[3], FD[3], base[3];
    int lb_target[3], ub_target[3], lb_ref[3]; 
    // lb_ref 是参考坐标系的基准 offset (Prolong时是lbf, Restrict时是lbc)
    
    // 计算几何参数 (逻辑同 device 函数)
    for(int d=0; d<3; d++) {
        CD[d] = (t.uubc[d] - t.llbc[d]) / (double)t.extc[d];
        FD[d] = (t.uubf[d] - t.llbf[d]) / (double)t.extf[d];
        
        // Base alignment logic
        if (t.llbc[d] <= t.llbf[d]) {
            base[d] = t.llbc[d];
        } else {
            int j_val = d_idint((t.llbc[d] - t.llbf[d]) / FD[d] + 0.4);
            if ((j_val / 2) * 2 == j_val) base[d] = t.llbf[d];
            else base[d] = t.llbf[d] - CD[d] / 2.0;
        }
    }

    // 计算迭代范围 (0-based)
    // 逻辑：imino = lb_target - lb_ref + 1 (Fortran 1-based)
    // 换算成 0-based loop: range [lb_target - lb_ref, ub_target - lb_ref]
    // 这里的 lb_target 对应 Fortran 中的 lbp (prolong) 或 lbr (restrict)
    
    int i_start, i_end, j_start, j_end, k_start, k_end;

    if (t.type == TASK_PROLONG) {
        // Prolong: Loop over Fine Grid indices
        for(int d=0; d<3; d++) {
            // lbp/ubp calculation
            int lbp = d_idint((t.llbt[d] - base[d]) / FD[d] + 0.4) + 1;
            int ubp = d_idint((t.uubt[d] - base[d]) / FD[d] + 0.4);
            int lbf = d_idint((t.llbf[d] - base[d]) / FD[d] + 0.4) + 1;
            
            // Fortran Loop: do i = lbp - lbf + 1, ubp - lbf + 1
            // 0-based: start = lbp - lbf; end = ubp - lbf
            if (d==0) { i_start = lbp - lbf; i_end = ubp - lbf; }
            if (d==1) { j_start = lbp - lbf; j_end = ubp - lbf; }
            if (d==2) { k_start = lbp - lbf; k_end = ubp - lbf; }
        }
    } else { // TASK_RESTRICT
        // Restrict: Loop over Coarse Grid indices
        for(int d=0; d<3; d++) {
            // lbr/ubr calculation
            int lbr = d_idint((t.llbt[d] - base[d]) / CD[d] + 0.4) + 1;
            int ubr = d_idint((t.uubt[d] - base[d]) / CD[d] + 0.4);
            int lbc = d_idint((t.llbc[d] - base[d]) / CD[d] + 0.4) + 1;
            
            // Fortran Loop: do i = lbr - lbc + 1, ubr - lbc + 1
            if (d==0) { i_start = lbr - lbc; i_end = ubr - lbc; }
            if (d==1) { j_start = lbr - lbc; j_end = ubr - lbc; }
            if (d==2) { k_start = lbr - lbc; k_end = ubr - lbc; }
        }
    }

    // --- Step 2: Grid-Stride Loop 并行执行 ---
    // 将 3D 循环展平为 1D，利用 Block 内的所有线程并行
    int ni = i_end - i_start + 1;
    int nj = j_end - j_start + 1;
    int nk = k_end - k_start + 1;
    
    if (ni <= 0 || nj <= 0 || nk <= 0) return; // 空任务保护

    int total_points = ni * nj * nk;
    
    for (int idx = threadIdx.x; idx < total_points; idx += blockDim.x) {
        // 解码 1D idx -> 3D (local loop variables)
        int k_local = idx / (ni * nj);
        int rem   = idx % (ni * nj);
        int j_local = rem / ni;
        int i_local = rem % ni;

        // 还原为全局 0-based 坐标
        int i = i_start + i_local;
        int j = j_start + j_local;
        int k = k_start + k_local;

        if (t.type == TASK_PROLONG) {
            d_prolong3_device(
                i, j, k,
                t.llbc, t.uubc, t.extc, t.src_ptr,
                t.llbf, t.uubf, t.extf, t.dst_ptr,
                t.llbt, t.uubt,
                t.SoA, t.Symmetry
            );
        } else {
            d_restrict3_device(
                i, j, k,
                t.llbc, t.uubc, t.extc, t.dst_ptr, // 注意：Restrict中dst是coarse(输出), src是fine(输入)
                t.llbf, t.uubf, t.extf, t.src_ptr, // 所以这里 dst_ptr 传给 func(out), src_ptr 传给 funf(in)
                t.llbt, t.uubt,
                t.SoA, t.Symmetry
            );
        }
    }
}

// --------------------------------------------------------------------------
// 2. Manager 实现
// --------------------------------------------------------------------------

ProlongRestrictManager::ProlongRestrictManager() 
    : d_input_buffer_(nullptr), d_output_buffer_(nullptr), d_tasks_(nullptr),
      d_in_cap_(0), d_out_cap_(0), d_task_cap_(0) 
{}

ProlongRestrictManager::~ProlongRestrictManager() {
    if (d_input_buffer_) cudaFree(d_input_buffer_);
    if (d_output_buffer_) cudaFree(d_output_buffer_);
    if (d_tasks_) cudaFree(d_tasks_);
}

void ProlongRestrictManager::ReserveDeviceMemory(size_t required_in, size_t required_out, size_t required_tasks) {
    // Input Buffer
    if (required_in > d_in_cap_) {
        if (d_input_buffer_) cudaFree(d_input_buffer_);
        d_in_cap_ = (size_t)(required_in * 1.5);
        if (d_in_cap_ < 1024) d_in_cap_ = 1024;
        cudaMalloc((void**)&d_input_buffer_, d_in_cap_ * sizeof(double));
    }
    // Output Buffer
    if (required_out > d_out_cap_) {
        if (d_output_buffer_) cudaFree(d_output_buffer_);
        d_out_cap_ = (size_t)(required_out * 1.5);
        if (d_out_cap_ < 1024) d_out_cap_ = 1024;
        cudaMalloc((void**)&d_output_buffer_, d_out_cap_ * sizeof(double));
    }
    // Task List
    if (required_tasks > d_task_cap_) {
        if (d_tasks_) cudaFree(d_tasks_);
        d_task_cap_ = (size_t)(required_tasks * 1.5);
        if (d_task_cap_ < 16) d_task_cap_ = 16;
        cudaMalloc((void**)&d_tasks_, d_task_cap_ * sizeof(ProlongRestrictTask));
    }
}

void ProlongRestrictManager::AddTask(
    TaskType type,
    double* h_src, double* h_dst,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    const double* SoA, int Symmetry
) {
    // 1. 计算当前任务的数据大小
    size_t src_count = 0;
    size_t dst_count = 0;

    // Prolong: Input=Coarse(src), Output=Fine(dst)
    // Restrict: Input=Fine(src), Output=Coarse(dst)
    // 注意：这里的 extc/extf 是整个 grid patch 的尺寸
    size_t size_c = (size_t)extc[0] * extc[1] * extc[2];
    size_t size_f = (size_t)extf[0] * extf[1] * extf[2];

    if (type == TASK_PROLONG) {
        src_count = size_c; // 输入是粗网格
        dst_count = size_f; // 输出是细网格
    } else { // RESTRICT
        src_count = size_f; // 输入是细网格
        dst_count = size_c; // 输出是粗网格
    }

    // 2. 将 Input 数据拷贝到 Host Staging Buffer
    size_t current_in_offset = h_input_staging_.size();
    h_input_staging_.resize(current_in_offset + src_count);
    // 这里执行了一次 Host-to-Host copy，把分散的数据聚拢
    memcpy(h_input_staging_.data() + current_in_offset, h_src, src_count * sizeof(double));

    // 3. 预留 Output Buffer 空间 (Host Staging)
    // 我们暂时不需要 resize h_output_staging_，因为它是用来接收 GPU 结果的
    // 我们只需要记录 offsets
    size_t current_out_offset = 0; 
    // 为了简化逻辑，我们在 ExecuteBatch 时再 resize output staging
    // 这里我们只是累加一个 output counter 吗？不，我们需要知道每个任务的绝对偏移。
    // 简单做法：我们维护一个 running counter，或者在 output_infos 里存相对 offset
    // 让我们用一个成员变量记录当前总输出大小
    static size_t total_output_size = 0; // 注意：每次 Clear() 需要重置
    if (host_tasks_.empty()) total_output_size = 0; // Hack: 第一次添加时重置
    
    current_out_offset = total_output_size;
    total_output_size += dst_count;

    // 4. 构建 Task (存的是 GPU 上的“未来”地址)
    ProlongRestrictTask t;
    t.type = type;
    // 注意：这里赋值的是 偏移量！Kernel 无法直接用。
    // 我们需要在 Kernel 启动前，或者在 Kernel 内部把 base_ptr + offset 加上。
    // 为了让 Kernel 代码不改动，我们这里存 nullptr，等 ExecuteBatch 申请好 GPU 内存后，
    // 再统一把这些指针修补成 (d_base + offset)。
    // 这里我们暂时把 offset 强转成指针存进去 (Dirty Hack)，或者在 ExecuteBatch 里再遍历一遍。
    // 方案：在 ExecuteBatch 拷贝 Task 列表前，遍历 host_tasks_ 修正指针。
    t.src_ptr = (double*)current_in_offset; 
    t.dst_ptr = (double*)current_out_offset;

    // 复制几何参数
    for(int i=0; i<3; i++) {
        t.llbc[i]=llbc[i]; t.uubc[i]=uubc[i]; t.extc[i]=extc[i];
        t.llbf[i]=llbf[i]; t.uubf[i]=uubf[i]; t.extf[i]=extf[i];
        t.llbt[i]=llbt[i]; t.uubt[i]=uubt[i]; t.SoA[i]=SoA[i];
    }
    t.Symmetry = Symmetry;

    host_tasks_.push_back(t);

    // 5. 记录回传信息
    OutputCopyBackInfo info;
    info.host_dst_ptr = h_dst;
    info.offset = current_out_offset;
    info.size = dst_count;
    output_infos_.push_back(info);
}

void ProlongRestrictManager::ExecuteBatch() {
    if (host_tasks_.empty()) return;

    size_t num_tasks = host_tasks_.size();
    size_t total_in_doubles = h_input_staging_.size();
    
    // 计算总输出大小
    size_t total_out_doubles = 0;
    if (!output_infos_.empty()) {
        OutputCopyBackInfo& last = output_infos_.back();
        total_out_doubles = last.offset + last.size;
    }

    // 1. 准备 GPU 内存
    ReserveDeviceMemory(total_in_doubles, total_out_doubles, num_tasks);

    // 2. H2D: 拷贝输入数据 (Staging -> Device)
    // 这是一个大的连续拷贝，带宽利用率高
    cudaMemcpy(d_input_buffer_, h_input_staging_.data(), 
               total_in_doubles * sizeof(double), cudaMemcpyHostToDevice);

    // 3. 修正 Task 中的指针
    // 之前 AddTask 里存的是 offset，现在我们要加上 d_input_buffer_ 的基地址
    for (int i = 0; i < num_tasks; i++) {
        size_t in_off = (size_t)host_tasks_[i].src_ptr;
        size_t out_off = (size_t)host_tasks_[i].dst_ptr;
        
        host_tasks_[i].src_ptr = d_input_buffer_ + in_off;
        host_tasks_[i].dst_ptr = d_output_buffer_ + out_off;
    }

    // 4. H2D: 拷贝任务列表
    cudaMemcpy(d_tasks_, host_tasks_.data(), 
               num_tasks * sizeof(ProlongRestrictTask), cudaMemcpyHostToDevice);

    // 5. 启动 Kernel
    batched_prolong_restrict_kernel<<<num_tasks, 256>>>(d_tasks_, num_tasks);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. D2H: 拷贝输出结果 (Device -> Staging)
    h_output_staging_.resize(total_out_doubles); // 确保 CPU buffer 够大
    cudaMemcpy(h_output_staging_.data(), d_output_buffer_, 
               total_out_doubles * sizeof(double), cudaMemcpyDeviceToHost);

    // 7. Scatter: 将数据分发回用户指针
    // 这也是一次 Host-to-Host copy
    for (const auto& info : output_infos_) {
        memcpy(info.host_dst_ptr, 
               h_output_staging_.data() + info.offset, 
               info.size * sizeof(double));
    }

    // 8. 清理
    Clear();
}

void ProlongRestrictManager::Clear() {
    host_tasks_.clear();
    output_infos_.clear();
    h_input_staging_.clear();
    h_output_staging_.clear();
    // d_input_buffer_ 等显存不释放，留作池化复用
}