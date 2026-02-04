MPI_DIR = /home/jjsnam/spack/opt/spack/linux-haswell/intel-oneapi-mpi-2021.17.2-tzh7zdi7akujs3w3z6xura62oofpnlw7/mpi/2021.17/

# Simplified includes - Intel MPI wrappers handle most paths automatically
filein  = 

# Simplified libraries - Intel compilers/MPI handle standard libraries automatically
# Only keep what's explicitly needed for your application
LDLIBS  = -lifcoremt

# Compiler flags
CXXAPPFLAGS  = -O3 -Wno-deprecated -Dfortran3 -Dnewc
f90appflags  = -O3 -fpp
#f90appflags  = -O3 -x f95-cpp-input

# Intel MPI compiler wrappers (these automatically link MPI and Fortran runtime libraries)
#f90          = mpif90
#f77          = mpif77
#CXX          = mpicxx
#CC           = mpicc
#CLINKER      = mpicxx
f90          = mpiifx
f77          = mpiifx
CXX          = mpiicpx
CC           = mpiicc
CLINKER      = mpiicpx