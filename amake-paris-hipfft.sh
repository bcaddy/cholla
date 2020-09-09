#!/bin/bash

module load rocm
module load PrgEnv-cray
module load hdf5
module load gcc
module list

export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"

export CXX=CC
export DFLAGS='-DPARIS_NO_GPU_MPI'
export HIP_PLATFORM=hcc
export MPI_HOME=$(dirname $(dirname $(which mpicc)))
export OMP_NUM_THREADS=16
export POISSON_SOLVER='-DCUFFT -DPARIS'
export SUFFIX='.paris.hipfft'
export TYPE=gravity

make clean
make -j
