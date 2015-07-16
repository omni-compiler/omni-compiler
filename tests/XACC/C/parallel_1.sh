#PBS -S /bin/bash
#PBS -N mnakao_job
#PBS -A XMPTCA
#PBS -q tcaq
#PBS -l select=1:ncpus=1:host=tcag-0001+1:ncpus=1:host=tcag-0002
#PBS -l walltime=00:01:00
#PBS -o o_parallel_1
#PBS -e e_parallel_1
# . /opt/Modules/default/init/bash
#---------------
# select=NODES:ncpus=CORES:mpiprocs=PROCS:ompthreads=THREADS:mem=MEMORY
# NODES   : num of nodes
# CORES   : num of cores per node
# PROCS   : num of procs per node
# THREADS : num of threads per process
#----------------
NP=2
# module purge
# export MODULEPATH=/work/TCAPROF/hanawa/Modules:$MODULEPATH
# module load cuda/6.0 mvapich2/gdr-2.0b-cuda6
cd $PBS_O_WORKDIR
mpirun_rsh -np $NP -hostfile $PBS_NODEFILE MV2_ENABLE_AFFINITY=0 MV2_SHOW_CPU_BINDING=1 numactl --cpunodebind=0 --localalloc ./parallel_1.x
