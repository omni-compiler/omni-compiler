#-----------------
# XMP/gfortran/GASNet
#-----------------
#MPI=mpif90
#CAF=xmpf90 --debug
##MPILIB=$$(dirname $$(dirname $$(which mpif90)))/include
##CAF=xmpf90 --debug -I$(MPILIB) -g

#-----------------
# XMP/Fujitsu/RDMA
#-----------------
#MPI=mpifrtpx
#CAF=xmpf90
##MPILIB=/opt/FJSVtclang/GM-1.2.0-17/include/mpi/fujitsu/
##CAF=xmpf90 -I"$(MPILIB)" --debug

#-----------------
# UHCAF/GNU/mvapich2-GDR
#-----------------
#MPI=mpif90
#CAF=uhcaf


#-----------------
# ifort/Intel-MPI
#-----------------
MPI=mpiifort -O
#CAF=ifort -O -coarray=distributed -I/opt/MPI/IntelMPI/5.0.0.028/include64
CAF=mpiifort -O -coarray=distributed
#--------
# type as follows before make
# % module purge
# % module load intel/15.0.2 intelmpi/5.0.0
#-------

EXE=mpi.x caf-direct.x caf-wide.x caf-narrow.x caf-fit.x mpi-nocom.x caf-nocom.x

all: $(EXE)

### ex.
### $ (echo m; echo 1 2 2) | mpirun -n 4 mpi.x > mpi122m.result
### $ (echo m; echo 1 2 2) | mpirun -n 4 caf-wide.x > caf-wide122m.result

%.x: %.f90
	$(CAF) $< -o $@

caf-%.x: himenoBMTcaf-%.f90
	$(CAF) $< -o $@

mpi.x: himenoBMTxpr.f90
	$(MPI) $< -o $@

mpi-nocom.x: himenoBMTxpr-nocom.f90
	$(MPI) $< -o $@

run:	$(EXE:%.x=%.result)

%.result: %.x
	echo m >stdin.m122
	echo "1 2 2" >>stdin.m122
	mpirun -n 4 $< <stdin.m122

clean:
	rm -f $(EXE)
	rm -rf a.out *.mod *.xmod *.o *~ __omni_tmp__
