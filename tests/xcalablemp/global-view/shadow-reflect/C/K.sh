#!/bin/sh

#PJM --rsc-list "node=4"
#PJM --rsc-list "elapse=00:02:00"
#PJM --mpi use-rankdir

. /work/system/Env_base

mpiexec ./shadow_reflect.x

