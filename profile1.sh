#!/bin/bash
nvidia-cuda-mps-control -d
. /home/jjsnam/spack/share/spack/setup-env.sh
spack load intel-oneapi-vtune
spack load intel-oneapi-mpi
# spack load openmpi
spack load intel-oneapi-compilers

spack load cuda

source /home/jjsnam/anaconda3/etc/profile.d/conda.sh
conda activate AMSS

export I_MPI_FABRICS=shm
export I_MPI_PIN_DOMAIN=core

ulimit -s unlimited
export OMP_NUM_THREADS=1

cd GW/AMSS_NCKU_output/

TS=$(date +"%Y%m%d_%H%M%S")

export I_MPI_DEBUG=0

export LD_LIBRARY_PATH=/home/jjsnam/spack/opt/spack/linux-icelake/cuda-13.0.2-nxiq75wz7g54wiu5ublimzvgspxrweit/lib64:${LD_LIBRARY_PATH}

# vtune -collect hotspots -result-dir ../../profile/TwoPunctureABE/${TS}/ -- ./TwoPunctureABE
# ./TwoPunctureABE
mpirun -bootstrap fork -np 1 ./ABEGPU
# mpirun -bootstrap fork -np 1 vtune -collect hotspots -trace-mpi -result-dir ../../profile/ABEGPU/${TS}/ -- ./ABEGPU
echo quit | nvidia-cuda-mps-control
# mpirun -bootstrap fork -np 2 ncu --target-processes all ./ABEGPU
# uarch-exploration
# mpirun -bootstrap fork -np 1 vtune -collect uarch-exploration -trace-mpi -result-dir ../../profile/ABEGPU/ue${TS}/ -- ./ABEGPU