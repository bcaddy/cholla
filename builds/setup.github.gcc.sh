#!/bin/bash

#-- This script needs to be sourced in the terminal, e.g.
#   source ./setup.c3po.gcc.sh

# export MPI_GPU="-DMPI_GPU"
export F_OFFLOAD="-fopenmp -foffload=disable"
export CHOLLA_ENVSET=1
