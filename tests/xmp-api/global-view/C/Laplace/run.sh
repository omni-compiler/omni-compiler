#! /bin/bash 
#PJM --rsc-list "node=8"
#PJM --rsc-list "elapse=0:10:00"
# run
mpiexec ./xmp_api_laplace || exit
