#PBS -S /bin/bash
#PBS -N mnakao_job
#PBS -A XMPTCA
#PBS -q tcaq
#PBS -l select=1:ncpus=1:host=tcag-0001+1:ncpus=1:host=tcag-0002+1:ncpus=1:host=tcag-0003+1:ncpus=1:host=tcag-0004
#PBS -l walltime=00:01:00
#PBS -o o_reflect_acc_2_dup_1
#PBS -e e_reflect_acc_2_dup_1
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

NP=4

mpirun_rsh -np $NP -hostfile $PBS_NODEFILE MV2_ENABLE_AFFINITY=0 CUDA_VISIBLE_DEVICES=0 MV2_USE_GPUDIRECT_GDRCOPY=0 numactl --cpunodebind=0 --localalloc ./reflect_acc_2_dup_1.x
