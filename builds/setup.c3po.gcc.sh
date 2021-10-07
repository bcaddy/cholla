#!/bin/bash

#-- This script needs to be sourced in the terminal, e.g.
#   source ./setup.c3po.gcc.sh


# export GCC_ROOT=$GCC_UMS_DIR

# echo "Using GCC in $GCC_ROOT"

# export PATH=$GCC_ROOT/bin:${PATH}

# export OMPI_CC=${GCC_ROOT}/bin/gcc
# export OMPI_CXX=${GCC_ROOT}/bin/g++
# export OMPI_FC=${GCC_ROOT}/bin/gfortran
# export LD_LIBRARY_PATH=${GCC_ROOT}/lib64:${LD_LIBRARY_PATH}

# echo "mpicxx --version is: "
# mpicxx --version

export MPI_GPU="-DMPI_GPU"
# export F_OFFLOAD="-fopenmp -foffload=nvptx-none='-lm -Ofast'"
export MPI_GPU="-DMPI_GPU"
export F_OFFLOAD="-fopenmp -foffload=disable"
export CHOLLA_ENVSET=1
export CHOLLA_MACHINE=c3po
