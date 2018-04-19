#PBS -S /bin/bash
#PBS -N mnakao_job
#PBS -A XMPTCA
#PBS -q tcaq-q1
#PBS -l select=1:ncpus=1:host=tcag-0001-eth0+1:ncpus=1:host=tcag-0002-eth0+1:ncpus=1:host=tcag-0003-eth0+1:ncpus=1:host=tcag-0004-eth0+1:ncpus=1:host=tcag-0005-eth0+1:ncpus=1:host=tcag-0006-eth0+1:ncpus=1:host=tcag-0007-eth0+1:ncpus=1:host=tcag-0008-eth0
#PBS -l walltime=00:01:00
. /opt/Modules/default/init/bash
#---------------
# select=NODES:ncpus=CORES:mpiprocs=PROCS:ompthreads=THREADS:mem=MEMORY
# NODES   : num of nodes
# CORES   : num of cores per node
# PROCS   : num of procs per node
# THREADS : num of threads per process
#----------------
NP=8
module purge
module load cuda/6.5.14 mvapich2-gdr/2.0_gnu_cuda-6.5
export LD_LIBRARY_PATH=/work/XMPTCA/mnakao/omni-compiler/PEACH2:$LD_LIBRARY_PATH
cd $PBS_O_WORKDIR
OPT="MV2_GPUDIRECT_LIMIT=4194304 MV2_ENABLE_AFFINITY=0 MV2_SHOW_CPU_BINDING=1 numactl --cpunodebind=0 --localalloc"
for i in $(seq 1 10)
do
mpirun_rsh -np $NP -hostfile $PBS_NODEFILE $OPT ./L
done
