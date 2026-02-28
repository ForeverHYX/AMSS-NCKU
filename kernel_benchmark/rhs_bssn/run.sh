#!/bin/bash
set -e # 发生错误即退出

spack load cuda

echo ">>> Running CMake..."
cmake -B build

echo ">>> Compiling Benchmark..."
cmake --build build

echo ">>> Running Benchmark..."
cd build
./bssn_benchmark