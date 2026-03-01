#!/bin/bash
set -e # 发生错误即退出

spack load cuda

echo ">>> Running CMake..."
cmake -B build

echo ">>> Compiling Benchmark..."
cmake --build build -j

echo ">>> Running Benchmark..."
cd build
./bssn_benchmark

# ncu -f --kernel-name "regex:(bssn_core_rhs_kernel|bssn_advection_dissipation_kernel|bssn_constraints_kernel)" --set full --launch-skip 10 --launch-count 6 -o report ./build/bssn_benchmark