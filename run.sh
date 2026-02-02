#!/bin/bash
#SBATCH -J AMSS_CPU_Run           # 任务名
#SBATCH -p gpu_partition                # 分区名 (请根据你们集群实际情况修改，如 w7900d, cpu, batch 等)
#SBATCH -N 1                      # 申请 1 个节点
#SBATCH --ntasks=16              # 申请 8 个核心 (对应 MPI_processes)
#SBATCH --cpus-per-task=1         # 每个任务 1 个核
#SBATCH --output=slurm_%j.out     # 标准输出日志
#SBATCH --error=slurm_%j.err      # 错误日志
#SBATCH --time=01:00:00           # 限制运行时间 1 小时 (测试用)

echo "Job start at $(date)"
echo "Node: $(hostname)"

. ~/spack/share/spack/setup-env.sh
spack load openmpi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate AMSS

ulimit -s unlimited
export OMP_NUM_THREADS=1

stdbuf -oL python3 AMSS_NCKU_Program.py

echo "Job finished at $(date)"