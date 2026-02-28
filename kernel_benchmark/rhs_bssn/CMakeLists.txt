cmake_minimum_required(VERSION 3.18)

# 在 project() 之前设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 80)

project(BSSN_Kernel_Benchmark LANGUAGES CXX CUDA)

# 要求 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用 CUDA 的可重定位设备代码 (RDC)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# 配置 CUDA 编译参数
# -O3: 最高优化等级
# -use_fast_math: 牺牲极小的数学精度换取极大的吞吐量提升 (对广义相对论计算这通常是必须的)
# -arch=sm_80: 假设使用 A100 等 Ampere 架构，如果是在普通 PC/Mac，可能需要改为 sm_75 或 sm_89 (Ada)
# -rdc=true: 启用可重定位设备代码，允许设备代码之间的相互调用
# -gencode: 为指定架构生成代码和PTX
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math -rdc=true")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=compute_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xptxas=-v") # 打印寄存器使用和 Spill 信息

# 包含头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 指定所有源文件
set(SOURCES
    benchmark.cu
    kernel_standard.cu
    kernel_optimized.cu
    diff_new_gpu.cu
    fmisc_gpu.cu 
    kodiss_gpu.cu 
    lopsidediff_gpu.cu
)

# 指定所有头文件（用于依赖跟踪）
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")

# 编译生成可执行文件
add_executable(bssn_benchmark ${SOURCES})

# 确保头文件变更时重新编译
target_include_directories(bssn_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 链接 CUDA 运行时库
target_link_libraries(bssn_benchmark PRIVATE cudart)