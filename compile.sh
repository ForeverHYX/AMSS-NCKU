#!/bin/bash

# ================= 1. 环境加载 =================
. /home/jjsnam/spack/share/spack/setup-env.sh
spack load intel-oneapi-vtune
spack load intel-oneapi-mpi
# spack load openmpi
spack load intel-oneapi-compilers

spack load cuda

source /home/jjsnam/anaconda3/etc/profile.d/conda.sh
conda activate AMSS

# MPI 优化参数
export I_MPI_FABRICS=shm
export I_MPI_PIN_DOMAIN=core
export I_MPI_DEBUG=5 

ulimit -s unlimited
export OMP_NUM_THREADS=1

# ================= 2. 变量定义 [关键修复] =================
# 获取当前脚本所在的绝对路径，防止 cd 后相对路径失效
PROJECT_ROOT="$(pwd)"

# 使用绝对路径定义工作目录
WORK_DIR="${PROJECT_ROOT}/GW"
ORIGINAL_SRC="${PROJECT_ROOT}/AMSS_NCKU_source"

# 编译源码的位置
TARGET_SRC="${WORK_DIR}/AMSS_NCKU_source_copy"
# 最终运行目录 (存放二进制文件)
OUTPUT_RUN_DIR="${WORK_DIR}/AMSS_NCKU_output"
# 最终数据目录
OUTPUT_BIN_DIR="${OUTPUT_RUN_DIR}/binary_output"
FIGURE_DIR="${WORK_DIR}/figure"
INPUT_FILE="${PROJECT_ROOT}/AMSS_NCKU_Input.py"

# 检查原始源码
if [ ! -d "$ORIGINAL_SRC" ]; then
    echo "Error: Source directory '$ORIGINAL_SRC' not found!"
    exit 1
fi

# ================= 3. 预处理：构建目录结构 =================
echo -e "\n>>> [Step 1] Building directory structure in ${WORK_DIR}..."

mkdir -p "${WORK_DIR}"
mkdir -p "${OUTPUT_RUN_DIR}"
mkdir -p "${OUTPUT_BIN_DIR}"
mkdir -p "${FIGURE_DIR}"

cp -u "${INPUT_FILE}" "${WORK_DIR}/"

# ================= 4. 生成配置 =================
echo -e "\n>>> [Step 2] Generating configuration files from ${INPUT_FILE}..."

# 1. 生成 macrodef.h
python3 -c "import generate_macrodef; generate_macrodef.generate_macrodef_h()"
if [ $? -ne 0 ]; then 
    echo "Error: Failed to generate macrodef.h"
    exit 1
fi

# 2. 生成 macrodef.fh
python3 -c "import generate_macrodef; generate_macrodef.generate_macrodef_fh()"
if [ $? -ne 0 ]; then 
    echo "Warning: Failed to generate macrodef.fh (ignore if not needed)"
fi

# 3. 生成 TwoPuncture 输入文件
python3 -c "import generate_TwoPuncture_input; generate_TwoPuncture_input.generate_AMSSNCKU_TwoPuncture_input()"
if [ $? -ne 0 ]; then 
    echo "Error: Failed to generate TwoPuncture input"
    exit 1
fi

# 拷贝生成的 input 到 output 目录并改名为 TwoPunctureinput.par
TP_INPUT_NAME="AMSS-NCKU-TwoPuncture.input"
if [ -f "${WORK_DIR}/${TP_INPUT_NAME}" ]; then
    TP_SRC="${WORK_DIR}/${TP_INPUT_NAME}"
elif [ -f "${PROJECT_ROOT}/${TP_INPUT_NAME}" ]; then
    TP_SRC="${PROJECT_ROOT}/${TP_INPUT_NAME}"
else
    echo "Warning: Generated ${TP_INPUT_NAME} not found, skipping copy."
fi

if [ -n "$TP_SRC" ]; then
    cp "$TP_SRC" "${OUTPUT_RUN_DIR}/TwoPunctureinput.par"
    echo "   Copied ${TP_INPUT_NAME} -> ${OUTPUT_RUN_DIR}/TwoPunctureinput.par"
fi

# ================= 5. 源码同步 =================
MODE=$1 

if [ "$MODE" == "full" ]; then
    echo -e "\n>>> [Step 3] Mode: \033[31mFULL REBUILD\033[0m"
    if [ -d "$TARGET_SRC" ]; then
        rm -rf "${TARGET_SRC}"
    fi
    cp -r "${ORIGINAL_SRC}" "${TARGET_SRC}"
else
    echo -e "\n>>> [Step 3] Mode: \033[32mINCREMENTAL BUILD\033[0m"
    if [ ! -d "$TARGET_SRC" ]; then
        cp -r "${ORIGINAL_SRC}" "${TARGET_SRC}"
    else
        # 增量更新源码
        cp -rup "${ORIGINAL_SRC}/"* "${TARGET_SRC}/"
    fi
fi

# ================= 6. [智能] 注入配置参数 =================
echo -e "\n>>> [Step 4] Injecting generated configs into build tree..."

# 定义一个智能拷贝函数：只在内容改变时才拷贝
smart_copy() {
    local src="$1"
    local dest_dir="$2"
    local filename=$(basename "$src")
    local dest_file="${dest_dir}/${filename}"

    if [ ! -f "$src" ]; then
        return
    fi

    # 如果目标文件不存在，直接拷贝
    if [ ! -f "$dest_file" ]; then
        cp -p "$src" "$dest_file"
        echo "   [New] Copied $filename"
    else
        # 核心逻辑：比较内容 (cmp -s 静默比较)
        if cmp -s "$src" "$dest_file"; then
            echo "   [Skip] $filename unchanged (Preserving timestamp)"
        else
            cp -p "$src" "$dest_file"
            echo "   [Update] $filename changed (Will trigger recompile)"
        fi
    fi
}

# 应用智能拷贝到所有生成的配置文件
smart_copy "${WORK_DIR}/macrodef.h"       "${TARGET_SRC}"
smart_copy "${WORK_DIR}/macrodef.fh"      "${TARGET_SRC}"
smart_copy "${WORK_DIR}/TwoPunctures.par" "${TARGET_SRC}"

# ================= 7. 执行编译 =================
cd "${TARGET_SRC}" || exit
echo -e "\n>>> [Step 5] Starting Compilation in $(pwd)..."

if [ "$MODE" == "full" ]; then
    make clean > /dev/null 2>&1
fi

J_NUM=16

echo -e "   [Make] TwoPunctureABE..."
make -j${J_NUM} TwoPunctureABE
if [ $? -ne 0 ]; then echo "Error: TwoPunctureABE build failed!"; exit 1; fi

echo -e "   [Make] ABEGPU..."
make -j${J_NUM} ABEGPU
if [ $? -ne 0 ]; then echo "Error: ABEGPU build failed!"; exit 1; fi

# ================= 8. 安装二进制文件 =================
echo -e "\n>>> [Step 6] Installing executables to ${OUTPUT_RUN_DIR}..."

# 因为前面使用了绝对路径 OUTPUT_RUN_DIR，这里就算在 source_copy 目录下也没问题
if [ -f "ABE" ]; then
    cp -u "ABE" "${OUTPUT_RUN_DIR}/"
    echo "   Installed: ABE"
elif [ -f "ABEGPU" ]; then
    cp -u "ABEGPU" "${OUTPUT_RUN_DIR}/"
    echo "   Installed: ABEGPU"
else
    echo "Error: ABE/ABEGPU executable not found after make!"
    exit 1
fi

if [ -f "TwoPunctureABE" ]; then
    cp -u "TwoPunctureABE" "${OUTPUT_RUN_DIR}/"
    echo "   Installed: TwoPunctureABE"
fi

if [ -f "Ansorg.psid" ]; then
    cp -u "Ansorg.psid" "${OUTPUT_RUN_DIR}/"
    echo "   Copied: Ansorg.psid"
fi

echo -e "\n\033[32m=== SUCCESS: Build & Setup Complete ===\033[0m"
echo "Work Directory: ${WORK_DIR}"
echo "Executables are ready in: ${OUTPUT_RUN_DIR}"