#PBS -S /bin/bash
#PBS -N xacc_shadow_reflect_test
#PBS -A XMPTCA
#PBS -q tcaq
#PBS -l select=1:ncpus=1:host=tcag-0001+1:ncpus=1:host=tcag-0002+1:ncpus=1:host=tcag-0003+1:ncpus=1:host=tcag-0004+1:ncpus=1:host=tcag-0005+1:ncpus=1:host=tcag-0006+1:ncpus=1:host=tcag-0007+1:ncpus=1:host=tcag-0008
#PBS -l walltime=00:10:00
#PBS -o o_shadow_reflect
#PBS -e e_shadow_reflect
#---------------
# select=NODES:ncpus=CORES:mpiprocs=PROCS:ompthreads=THREADS:mem=MEMORY
# NODES   : num of nodes
# CORES   : num of cores per node
# PROCS   : num of procs per node
# THREADS : num of threads per process
#----------------
. /opt/Modules/default/init/bash
module purge
module load cuda/7.5.18 mvapich2-gdr/2.1_gnu_cuda-7.5
cd $PBS_O_WORKDIR

make run RUNTYPE=PBS
