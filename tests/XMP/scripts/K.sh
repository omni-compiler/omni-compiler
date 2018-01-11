#!/bin/sh

#PJM --rsc-list "node=4"
#PJM --rsc-list "elapse=00:05:00"
#PJM --mpi use-rankdir

. /work/system/Env_base

mpiexec -n 4 ../global-view/shadow-reflect/C/shadow_reflect.x
mpiexec -n 2 ../local-view/coarray/C/coarray_scalar.x
mpiexec -n 2 ../local-view/coarray/C/coarray_vector.x
mpiexec -n 2 ../local-view/coarray/C/coarray_stride.x
mpiexec -n 4 ../local-view/post-wait/C/post_wait.x
mpiexec -n 2 ../others/Fortran/module_test.x
mpiexec -n 2 ../others/Fortran/exit.x
