#PBS -S /bin/bash
#PBS -N xacc_test
#PBS -A TCAGEN
#PBS -q tcaq-q1
#PBS -l select=8:ncpus=4:mpiprocs=1
###PBS -l select=1:ncpus=1:host=tcag-0001+1:ncpus=1:host=tcag-0002+1:ncpus=1:host=tcag-0003+1:ncpus=1:host=tcag-0004+1:ncpus=1:host=tcag-0005+1:ncpus=1:host=tcag-0006+1:ncpus=1:host=tcag-0007+1:ncpus=1:host=tcag-0008
#PBS -l walltime=00:10:00
##PBS -o o_xacc_test
##PBS -e e_xacc_test
#---------------
# select=NODES:ncpus=CORES:mpiprocs=PROCS:ompthreads=THREADS:mem=MEMORY
# NODES   : num of nodes
# CORES   : num of cores per node
# PROCS   : num of procs per node
# THREADS : num of threads per process
#----------------
. /opt/Modules/default/init/bash
module purge
module load cuda/8.0.44 mvapich2/2.2_gnu_medium_cuda-8.0.44

cd $PBS_O_WORKDIR

make run JOBSCHED=PBS
