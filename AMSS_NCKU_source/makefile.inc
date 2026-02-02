## filein  = -I/usr/include -I/usr/lib/x86_64-linux-gnu/mpich/include -I/usr/lib/x86_64-linux-gnu/openmpi/lib/ -I/usr/lib/gcc/x86_64-linux-gnu/11/ -I/usr/include/c++/11/

MPI_DIR = /home/ASC1514/spack/opt/spack/linux-zen4/openmpi-5.0.9-rsvr2wy2glnvnqpnopaqto267dqdm37g
ROCM_DIR = /opt/rocm-7.1.1

filein  = -I$(MPI_DIR)/include \
          -I$(MPI_DIR)/lib \
          -I$(CONDA_PREFIX)/include \
          -I$(CONDA_PREFIX)/include/opencv4 \
          -I$(ROCM_DIR)/include \
          -I/usr/include \
          -I/usr/include/c++/11

## LDLIBS  = -L/usr/lib/x86_64-linux-gnu -lmpich -L/usr/lib64 -L/usr/lib/gcc/x86_64-linux-gnu/11 -lgfortran
LDLIBS  = -L$(MPI_DIR)/lib \
          -L$(ROCM_DIR)/lib/llvm/lib \
          -L$(ROCM_DIR)/lib \
          -L$(CONDA_DIR)/lib \
          -L/usr/lib/x86_64-linux-gnu \
          -lmpi -lmpi_mpifh -lmpi_usempi_ignore_tkr -lmpi_usempif08 \
          -lFortranRuntime -lFortranDecimal -lFortranFloat128Math \
          -lstdc++ -lm

CXXAPPFLAGS  = -O3 -Wno-deprecated -Dfortran3 -Dnewc
#f90appflags = -O3 -fpp
f90appflags  = -O3 -x f95-cpp-input
f90          = mpif90 
f77          = mpif77 
CXX          = mpicxx
CC           = mpicc
CLINKER      = mpicxx