#!/bin/bash
#SBATCH --get-user-env
#SBATCH -p batch
#SBATCH -J @JOBNAME@
#SBATCH -N 1         #num of node
#SBATCH -n 2         #num of total mpi processes
#SBATCH -c 1         #num of threads per mpi processes
#SBATCH -o @DIRNAME@/log
#SBATCH -e @DIRNAME@/err
#SBATCH -t 00:20:00

cd @DIRNAME@

make -j2
make run

echo $? > slurm_error_code
