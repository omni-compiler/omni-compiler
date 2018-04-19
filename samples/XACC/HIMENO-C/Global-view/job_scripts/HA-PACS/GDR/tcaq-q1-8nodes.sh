#PBS -S /bin/bash
#PBS -N mnakao_job
#PBS -A XMPTCA
#PBS -q tcaq-q1
#PBS -l select=8:ncpus=1:mpiprocs=1
#PBS -l walltime=00:03:00
. /opt/Modules/default/init/bash
#---------------
# select=NODES:ncpus=CORES:mpiprocs=PROCS:ompthreads=THREADS:mem=MEMORY
# NODES   : num of nodes
# CORES   : num of cores per node
# PROCS   : num of procs per node
# THREADS : num of threads per process
#----------------
module purge
module load cuda/6.5.14 mvapich2-gdr/2.0_gnu_cuda-6.5
NP=8
OPT="MV2_ENABLE_AFFINITY=0 CUDA_VISIBLE_DEVICES=2 MV2_USE_CUDA=1 MV2_NUM_PORTS=2 \
MV2_GPUDIRECT_LIMIT=524288 MV2_USE_GPUDIRECT_RECEIVE_LIMIT=524288 \
MV2_USE_GPUDIRECT_GDRCOPY=1 MV2_GPUDIRECT_GDRCOPY_LIB=/work/TCAPROF/hanawa/GDRcopy/lib/libgdrapi.so \
MV2_USE_GPUDIRECT_GDRCOPY_LIMIT=16384 MV2_USE_GPUDIRECT_GDRCOPY_NAIVE_LIMIT=16384 \
numactl --cpunodebind=1 --localalloc"
cd $PBS_O_WORKDIR

for i in $(seq 1 10)
do
mpirun_rsh -np $NP -hostfile $PBS_NODEFILE XACC_COMM_MODE=1 $OPT ./M
done
