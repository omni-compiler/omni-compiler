MPILIB=/opt/mpich/include
#MPILIB=/opt/FJSVtclang/GM-1.2.0-17/include/mpi/fujitsu/

#XMP=xmpf90 --debug -I$(MPILIB) -g
XMP=xmpf90 -I$(MPILIB)

MPI=mpif90
#MPI=mpifrtpx

OBJ=mpi caf-wide caf-narrow caf-fit

all: $(OBJ)

### ex.
### $ (echo m; echo 1 2 2) | mpirun -n 4 mpi > mpi122m.result
### $ (echo m; echo 1 2 2) | mpirun -n 4 caf-wide > caf-wide122m.result

caf-wide: himenoBMTcaf-wide.f90
	$(XMP) $< -o $@

caf-narrow: himenoBMTcaf-narrow.f90
	$(XMP) $< -o $@

caf-fit: himenoBMTcaf-fit.f90
	$(XMP) $< -o $@

mpi: himenoBMTxpr.f90
	$(MPI) $< -o $@

clean:
	rm -f $(OBJ)
	rm -rf a.out *.mod *.xmod *.o *~ __omni_tmp__
