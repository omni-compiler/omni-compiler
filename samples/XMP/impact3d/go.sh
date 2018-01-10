#!/bin/sh
trap exit 1 2 3
#
export OMP_NUM_THREADS=2
if [ "$EXEPR" = "" -o "$EXEPR" = "1" ]
then
   echo "impact3d-serialh started at `date`"
   echo "impact3d-serialh started at `date`" > ./results/serialh.list
   ./lm/serialh < /dev/null >> ./results/serialh.list 2> ./results/serialh.err
   echo "impact3d-serialh  ended  at `date`" >> ./results/serialh.list
   echo "impact3d-serialh  ended  at `date`"
   sleep 5
fi
if [ "$EXEPR" = "" -o "$EXEPR" = "1" ]
then
   echo "impact3d-mpi1h-1 started at `date`"
   echo "impact3d-mpi1h-1 started at `date`" > ./results/mpi1h-1.list
   mpirun -np 1 ./lm/mpi1h-1 < /dev/null >> ./results/mpi1h-1.list 2> ./results/mpi1h-1.err
   echo "impact3d-mpi1h-1  ended  at `date`" >> ./results/mpi1h-1.list
   echo "impact3d-mpi1h-1  ended  at `date`"
   sleep 5
fi
if [ "$EXEPR" = "" -o "$EXEPR" = "2" ]
then
   echo "impact3d-mpi1h-2 started at `date`"
   echo "impact3d-mpi1h-2 started at `date`" > ./results/mpi1h-2.list
   mpirun -np 2 ./lm/mpi1h-2 < /dev/null >> ./results/mpi1h-2.list 2> ./results/mpi1h-2.err
   echo "impact3d-mpi1h-2  ended  at `date`" >> ./results/mpi1h-2.list
   echo "impact3d-mpi1h-2  ended  at `date`"
   sleep 5
fi
