#!/bin/bash
#SBATCH --get-user-env
#SBATCH -p batch
#SBATCH -J @JOBNAME@
#SBATCH -N 1         #num of node
#SBATCH -n 2         #num of total mpi processes
#SBATCH -c 1         #num of threads per mpi processes
#SBATCH -o @DIRNAME@/log
#SBATCH -e @DIRNAME@/err
#SBATCH -t 00:30:00

cd @DIRNAME@
export PATH=@XMP_PATH@/bin:$PATH
export PATH=@MPI_PATH@/bin:$PATH

## Date for creating graph of time line
OUTPUT_FILENAME=timeline.dat
echo @JOBNAME@ > $OUTPUT_FILENAME
hostname >> $OUTPUT_FILENAME
date +"%Y,%m,%d,%H,%M,%S" >> $OUTPUT_FILENAME

## Execute Job
make -j2
make run

echo $? > slurm_ret_code

## Date for creating graph of time line
date +"%Y,%m,%d,%H,%M,%S" >> $OUTPUT_FILENAME

