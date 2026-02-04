MPI_DIR = /home/jjsnam/spack/opt/spack/linux-haswell/intel-oneapi-mpi-2021.17.2-tzh7zdi7akujs3w3z6xura62oofpnlw7/mpi/2021.17/

CUDA_DIR = /usr/local/cuda
CUDA_COMPUTE_CAPABILITY = 80

# Simplified includes - Intel MPI wrappers handle most paths automatically
filein  = 

# Simplified libraries - Intel compilers/MPI handle standard libraries automatically
# Only keep what's explicitly needed for your application
LDLIBS  = -lifcoremt -L$(CUDA_DIR)/lib64 -lcudart

# Compiler flags
CXXAPPFLAGS  = -O3 -Wno-deprecated -Dfortran3 -Dnewc
f90appflags  = -O3 -fpp
#f90appflags  = -O3 -x f95-cpp-input

# Intel MPI compiler wrappers (these automatically link MPI and Fortran runtime libraries)
f90          = mpiifx
f77          = mpiifx
CXX          = mpiicpx
CC           = mpiicc
CLINKER      = mpiicpx

Cu = $(CUDA_DIR)/bin/nvcc
CUDA_LIB_PATH = -L/usr/lib/cuda/lib64 -I/usr/include -I/usr/lib/cuda/include
#CUDA_APP_FLAGS = -c -g -O3 --ptxas-options=-v -arch compute_13 -code compute_13,sm_13 -Dfortran3 -Dnewc
CUDA_APP_FLAGS = -c -g -O3 --ptxas-options=-v -Dfortran3 -Dnewc  -arch=sm_$(CUDA_COMPUTE_CAPABILITY)