#!/bin/bash

. /home/jjsnam/spack/share/spack/setup-env.sh
spack load intel-oneapi-vtune
spack load intel-oneapi-mpi
# spack load openmpi
spack load intel-oneapi-compilers

source /home/jjsnam/anaconda3/etc/profile.d/conda.sh
conda activate AMSS

export I_MPI_FABRICS=shm
export I_MPI_PIN_DOMAIN=core

ulimit -s unlimited
export OMP_NUM_THREADS=1

cd GW150914/AMSS_NCKU_output/

TS=$(date +"%Y%m%d_%H%M%S")

export I_MPI_DEBUG=0

# vtune -collect hotspots -result-dir ../../profile/TwoPunctureABE/${TS}/ -- ./TwoPunctureABE
# ./TwoPunctureABE
# mpirun -bootstrap fork -np 16 vtune -collect hotspots -trace-mpi -result-dir ../../profile/ABE/${TS}/ -- ./ABE
# mpirun -bootstrap fork -np 16 ./ABE
mpirun -bootstrap fork -np 2 ./ABEGPU
# mpirun -bootstrap fork -np 2 ncu --target-processes all ./ABEGPU
# mpirun -bootstrap fork -np 2 vtune -collect hotspots -trace-mpi -result-dir ../../profile/ABEGPU/${TS}/ -- ./ABEGPU