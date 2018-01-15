#PBS -S /bin/bash
#PBS -N mnakao_job
#PBS -A XMPTCA
#PBS -q tcaq
#PBS -l select=1:ncpus=20:mpiprocs=2:ompthreads=5
#PBS -l walltime=00:02:00
#PBS -j oe
. /opt/Modules/default/init/bash
#---------------
# select=NODES:ncpus=CORES:mpiprocs=PROCS:ompthreads=THREADS:mem=MEMORY
# NODES   : num of nodes
# CORES   : num of cores per node
# PROCS   : num of procs per node
# THREADS : num of threads per process
#----------------
module purge
module load intelmpi mkl intel cuda/6.5.14
cd $PBS_O_WORKDIR
export KMP_AFFINITY=granularity=fine,compact
mpirun -np 2 ./STREAM 357913942 0.68
