#!/bin/sh

#PJM --rsc-list "node=2"
#PJM --rsc-list "elapse=00:02:00"
#PJM --mpi use-rankdir

. /work/system/Env_base

mpiexec ./module_test

