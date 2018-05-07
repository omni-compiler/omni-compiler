#-----------------
# XMP/gfortran/GASNet
#-----------------
MPI=mpif90
CAF=xmpf90 --debug -g
#MPILIB=$$(dirname $$(dirname $$(which mpif90)))/include
#CAF=xmpf90 --debug -I$(MPILIB) -g

#-----------------
# XMP/Fujitsu/RDMA
#-----------------
#MPI=mpifrtpx
#CAF=xmpf90
##MPILIB=/opt/FJSVtclang/GM-1.2.0-17/include/mpi/fujitsu/
##CAF=xmpf90 -I"$(MPILIB)" --debug

#-----------------
# UHCAF/Intel/GASNet
#-----------------
#MPI=mpif90
#MPILIB=$$(dirname $$(dirname $$(which mpif90)))/include64
#MPILIB=/opt/MPI/IntelMPI/impi/4.1.1.036/include64
#CAF=uhcaf --mpi -I$(MPILIB)

#-----------------
# UHCAF/GNU/mvapich2-GDR
#-----------------
#MPI=mpif90
#MPILIB=$$(dirname $$(dirname $$(which mpif90)))/include64
#MPILIB=/opt/MPI/MVAPICH2-GDR/2.0/GNU/CUDA/6.5/include64
#MPILIB=/opt/MPI/IntelMPI/impi/4.1.1.036/include64
#CAF=uhcaf --mpi -I$(MPILIB)



EXE=mpi.x caf-wide.x caf-narrow.x caf-fit.x

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

clean:
	rm -f $(EXE)
	rm -rf a.out *.mod *.xmod *.o *~ __omni_tmp__
