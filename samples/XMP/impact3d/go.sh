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
if [ "$EXEPR" = "" -o "$EXEPR" = "1" ]
then
   echo "impact3d-mpi2h-1 started at `date`"
   echo "impact3d-mpi2h-1 started at `date`" > ./results/mpi2h-1.list
   mpirun -np 1 ./lm/mpi2h-1 < /dev/null >> ./results/mpi2h-1.list 2> ./results/mpi2h-1.err
   echo "impact3d-mpi2h-1  ended  at `date`" >> ./results/mpi2h-1.list
   echo "impact3d-mpi2h-1  ended  at `date`"
   sleep 5
fi
if [ "$EXEPR" = "" -o "$EXEPR" = "4" ]
then
   echo "impact3d-mpi2h-4 started at `date`"
   echo "impact3d-mpi2h-4 started at `date`" > ./results/mpi2h-4.list
   mpirun -np 4 ./lm/mpi2h-4 < /dev/null >> ./results/mpi2h-4.list 2> ./results/mpi2h-4.err
   echo "impact3d-mpi2h-4  ended  at `date`" >> ./results/mpi2h-4.list
   echo "impact3d-mpi2h-4  ended  at `date`"
   sleep 5
fi
if [ "$EXEPR" = "" -o "$EXEPR" = "1" ]
then
   echo "impact3d-mpi3h-1 started at `date`"
   echo "impact3d-mpi3h-1 started at `date`" > ./results/mpi3h-1.list
   mpirun -np 1 ./lm/mpi3h-1 < /dev/null >> ./results/mpi3h-1.list 2> ./results/mpi3h-1.err
   echo "impact3d-mpi3h-1  ended  at `date`" >> ./results/mpi3h-1.list
   echo "impact3d-mpi3h-1  ended  at `date`"
   sleep 5
fi
if [ "$EXEPR" = "" -o "$EXEPR" = "8" ]
then
   echo "impact3d-mpi3h-8 started at `date`"
   echo "impact3d-mpi3h-8 started at `date`" > ./results/mpi3h-8.list
   mpirun -np 8 ./lm/mpi3h-8 < /dev/null >> ./results/mpi3h-8.list 2> ./results/mpi3h-8.err
   echo "impact3d-mpi3h-8  ended  at `date`" >> ./results/mpi3h-8.list
   echo "impact3d-mpi3h-8  ended  at `date`"
   sleep 5
fi
if [ "$EXEPR" = "" -o "$EXEPR" = "1" ]
then
   echo "impact3d-xmp1h-1 started at `date`"
   echo "impact3d-xmp1h-1 started at `date`" > ./results/xmp1h-1.list
   mpirun -np 1 ./lm/xmp1h-1 < /dev/null >> ./results/xmp1h-1.list 2> ./results/xmp1h-1.err
   echo "impact3d-xmp1h-1  ended  at `date`" >> ./results/xmp1h-1.list
   echo "impact3d-xmp1h-1  ended  at `date`"
   sleep 5
fi
if [ "$EXEPR" = "" -o "$EXEPR" = "2" ]
then
   echo "impact3d-xmp1h-2 started at `date`"
   echo "impact3d-xmp1h-2 started at `date`" > ./results/xmp1h-2.list
   mpirun -np 2 ./lm/xmp1h-2 < /dev/null >> ./results/xmp1h-2.list 2> ./results/xmp1h-2.err
   echo "impact3d-xmp1h-2  ended  at `date`" >> ./results/xmp1h-2.list
   echo "impact3d-xmp1h-2  ended  at `date`"
   sleep 5
fi
if [ "$EXEPR" = "" -o "$EXEPR" = "1" ]
then
   echo "impact3d-xmp2rh-1 started at `date`"
   echo "impact3d-xmp2rh-1 started at `date`" > ./results/xmp2rh-1.list
   mpirun -np 1 ./lm/xmp2rh-1 < /dev/null >> ./results/xmp2rh-1.list 2> ./results/xmp2rh-1.err
   echo "impact3d-xmp2rh-1  ended  at `date`" >> ./results/xmp2rh-1.list
   echo "impact3d-xmp2rh-1  ended  at `date`"
   sleep 5
fi
if [ "$EXEPR" = "" -o "$EXEPR" = "4" ]
then
   echo "impact3d-xmp2rh-4 started at `date`"
   echo "impact3d-xmp2rh-4 started at `date`" > ./results/xmp2rh-4.list
   mpirun -np 4 ./lm/xmp2rh-4 < /dev/null >> ./results/xmp2rh-4.list 2> ./results/xmp2rh-4.err
   echo "impact3d-xmp2rh-4  ended  at `date`" >> ./results/xmp2rh-4.list
   echo "impact3d-xmp2rh-4  ended  at `date`"
   sleep 5
fi
if [ "$EXEPR" = "" -o "$EXEPR" = "1" ]
then
   echo "impact3d-xmp3rh-1 started at `date`"
   echo "impact3d-xmp3rh-1 started at `date`" > ./results/xmp3rh-1.list
   mpirun -np 1 ./lm/xmp3rh-1 < /dev/null >> ./results/xmp3rh-1.list 2> ./results/xmp3rh-1.err
   echo "impact3d-xmp3rh-1  ended  at `date`" >> ./results/xmp3rh-1.list
   echo "impact3d-xmp3rh-1  ended  at `date`"
   sleep 5
fi
if [ "$EXEPR" = "" -o "$EXEPR" = "8" ]
then
   echo "impact3d-xmp3rh-8 started at `date`"
   echo "impact3d-xmp3rh-8 started at `date`" > ./results/xmp3rh-8.list
   mpirun -np 8 ./lm/xmp3rh-8 < /dev/null >> ./results/xmp3rh-8.list 2> ./results/xmp3rh-8.err
   echo "impact3d-xmp3rh-8  ended  at `date`" >> ./results/xmp3rh-8.list
   echo "impact3d-xmp3rh-8  ended  at `date`"
   sleep 5
fi
exit
