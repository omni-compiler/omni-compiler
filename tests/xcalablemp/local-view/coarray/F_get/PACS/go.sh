#PBS -S /bin/bash
#PBS -A XMPTCA
#PBS -q tcaq 

#PBS -l select=3:mpiprocs=1
#PBS -l walltime=00:05:00

. /opt/Modules/default/init/bash
#module load intel/15.0.2 intelmpi/5.0.0
module load gnu/4.4.7 mvapich2/1.8.1_gnu_cuda-6.0.37 cuda/6.0.37

cd $PBS_O_WORKDIR

echo Environment -------
module list 2>&1
echo "pwd = "; pwd
echo -------------------


OUT_basic="a1d1-4.x a1d1-1.x a6d3-1.x cn1-1NGB.x"
OUT_io="cn1d1-writeNGB.x"
OUT_arg="actualarg.x triplet.x actualarg-2.x actualarg-3.x"
OUT_nest="nest.x putget-2NGB.x putget-3NGB.x putget-4.x putget-5.x putget-6NGB.x"
OUT_lib="a1d1-5.x"
OUT_entire="axd0-1.x axd1-1.x axd2-1.x"
OUT_bound="cn1-2okB.x l0-1okB.x"

OUT="${OUT_basic} ${OUT_io} ${OUT_arg} ${OUT_nest} ${OUT_lib} ${OUT_mod} ${OUT_entire} ${OUT_bound}"

EXE=$OUT

for f in ${EXE}; do
    echo $f
    $HOME/local/GASNet-1.24.2-Jun9/bin/gasnetrun_ibv -n 3 -spawner=mpi ../$f
done

