#!/bin/bash
#SBATCH -J mnakao-mpi-job
#SBATCH -p mixed
#SBATCH -N 2
#SBATCH --distribution=cyclic
#SBATCH -t 0:10:00

module purge
module load intelmpi/4.1.3
cd $SLURM_SUBMIT_DIR
mpirun -np 2 $TESTDIR/others/Fortran/exit.x
mpirun -np 2 $TESTDIR/others/Fortran/module_test.x
mpirun -np 4 $TESTDIR/global-view/shadow-reflect/C/shadow_reflect.x
mpirun -np 2 $TESTDIR/local-view/coarray/C/coarray_scalar.x
mpirun -np 2 $TESTDIR/local-view/coarray/C/coarray_vector.x
mpirun -np 2 $TESTDIR/local-view/coarray/C/coarray_stride.x
mpirun -np 2 $TESTDIR/local-view/coarray/C/coarray_put_1dim.x
mpirun -np 2 $TESTDIR/local-view/coarray/C/coarray_get_1dim.x
mpirun -np 2 $TESTDIR/local-view/coarray/C/coarray_put_2dims.x
mpirun -np 2 $TESTDIR/local-view/coarray/C/coarray_get_2dims.x
mpirun -np 2 $TESTDIR/local-view/coarray/C/coarray_put_3dims.x
mpirun -np 2 $TESTDIR/local-view/coarray/C/coarray_get_3dims.x
mpirun -np 2 $TESTDIR/local-view/coarray/C/coarray_put_4dims.x
mpirun -np 2 $TESTDIR/local-view/coarray/C/coarray_get_4dims.x




