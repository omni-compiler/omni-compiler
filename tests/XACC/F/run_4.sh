#PBS -S /bin/bash
#PBS -N tabuchi_job
#PBS -A TCAGEN
#PBS -q tcaq-q1
#PBS -l select=1:ncpus=4:mpiprocs=4:ompthreads=1
#PBS -l place=scatter
#PBS -l walltime=00:01:00
. /opt/Modules/default/init/bash
cd $PBS_O_WORKDIR
uniq $PBS_NODEFILE

module purge
module load cuda/8.0.44 pgi/16.10 mvapich2/2.2_pgi_cuda-8.0.44

for i in {1..1}
do
    MPIOPT="MV2_ENABLE_AFFINITY=0 MV2_SHOW_CPU_BINDING=1 MV2_USE_CUDA=1 XACC_COMM_MODE=1 MV2_CUDA_IPC=0"
    mpirun_rsh -np 4 -hostfile $PBS_NODEFILE $MPIOPT ./numa.sh ./xacc_distarray.x
    mpirun_rsh -np 4 -hostfile $PBS_NODEFILE $MPIOPT ./numa.sh ./xacc_reflect.x
    mpirun_rsh -np 4 -hostfile $PBS_NODEFILE $MPIOPT ./numa.sh ./xacc_reduction.x
    mpirun_rsh -np 4 -hostfile $PBS_NODEFILE $MPIOPT ./numa.sh ./xacc_reflect_oneside.x
done
