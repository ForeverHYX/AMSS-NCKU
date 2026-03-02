MPI_DIR = /home/foreverhyx/spack/opt/spack/linux-x86_64_v3/intel-oneapi-mpi-2021.17.2-geoxesl7gjei2ybcrfbkeglnhcpiukzn/mpi/2021.17

CUDA_DIR = /home/foreverhyx/spack/opt/spack/linux-icelake/cuda-13.0.2-r2bcws4tslasb4sd35jta22m6zxcnzf4
CUDA_COMPUTE_CAPABILITY = 80

# Simplified includes - Intel MPI wrappers handle most paths automatically
filein  = 

# Simplified libraries - Intel compilers/MPI handle standard libraries automatically
# Only keep what's explicitly needed for your application
LDLIBS  = -lifcoremt -L$(CUDA_DIR)/lib64 -lcudart

# Compiler flags
CXXAPPFLAGS  = -O3 -Wno-deprecated -Dfortran3 -Dnewc -I$(CUDA_DIR)/include
f90appflags  = -O3 -fpp
#f90appflags  = -O3 -x f95-cpp-input

# Intel MPI compiler wrappers (these automatically link MPI and Fortran runtime libraries)
f90          = mpiifx
f77          = mpiifx
CXX          = mpiicpx
CC           = mpiicc
CLINKER      = mpiicpx

Cu = $(CUDA_DIR)/bin/nvcc
CUDA_LIB_PATH = -L$(CUDA_DIR)/lib64 -I$(CUDA_DIR)/include
CUDA_APP_FLAGS = -g -O3 --ptxas-options=-v -Dfortran3 -Dnewc  -arch=sm_$(CUDA_COMPUTE_CAPABILITY)