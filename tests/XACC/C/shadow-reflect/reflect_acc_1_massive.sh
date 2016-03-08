#PBS -S /bin/bash
#PBS -N mnakao_job
#PBS -A XMPTCA
#PBS -q tcaq
#PBS -l select=1:ncpus=1:host=tcag-0001+1:ncpus=1:host=tcag-0002+1:ncpus=1:host=tcag-0003+1:ncpus=1:host=tcag-0004+1:ncpus=1:host=tcag-0005+1:ncpus=1:host=tcag-0006+1:ncpus=1:host=tcag-0007+1:ncpus=1:host=tcag-0008
NP=8
#PBS -l walltime=00:01:00
#PBS -o o_reflect_acc_1_massive
#PBS -e e_reflect_acc_1_massive
. /opt/Modules/default/init/bash
#---------------
# select=NODES:ncpus=CORES:mpiprocs=PROCS:ompthreads=THREADS:mem=MEMORY
# NODES   : num of nodes
# CORES   : num of cores per node
# PROCS   : num of procs per node
# THREADS : num of threads per process
#----------------
module purge
module load cuda/7.5.18 mvapich2-gdr/2.1_gnu_cuda-7.5
cd $PBS_O_WORKDIR
mpirun_rsh -np $NP -hostfile $PBS_NODEFILE MV2_ENABLE_AFFINITY=0 MV2_SHOW_CPU_BINDING=1 numactl --cpunodebind=0 --localalloc ./reflect_acc_1_massive.x
