#-----------------
# ifort/Intel-MPI
#-----------------
MPI=mpiifort -O
#CAF=ifort -O -coarray=distributed -I/opt/MPI/IntelMPI/5.0.0.028/include64
CAF=mpiifort -O -coarray=distributed -mt_mpi
SUBOBJ=co_reduction.o

#EXE=mpi.x caf-direct.x caf-wide.x caf-narrow.x caf-fit.x mpi-nocom.x caf-nocom.x caf-direct-nocom.x
EXE=mpi.x caf-direct.x caf-wide.x mpi-nocom.x caf-nocom.x caf-direct-nocom.x

all:  $(SUBOBJ) $(EXE)

### ex.
### $ (echo m; echo 1 2 2) | mpirun -n 4 mpi.x > mpi122m.result
### $ (echo m; echo 1 2 2) | mpirun -n 4 caf-wide.x > caf-wide122m.result

%.x: %.f90
	$(CAF) $< -o $@

caf-%.x: himenoBMTcaf-%.f90
	$(CAF) $< $(SUBOBJ) -o $@

mpi.x: himenoBMTxpr.f90
	$(MPI) $< -o $@

mpi-nocom.x: himenoBMTxpr-nocom.f90
	$(MPI) $< -o $@

co_reduction.o: co_reduction.f90
	$(CAF) -c $< -o $@

run:	$(EXE:%.x=%.result)

%.result: %.x
	echo m >stdin.m122
	echo "1 2 2" >>stdin.m122
	mpirun -n 4 $< <stdin.m122

clean:
	rm -f $(EXE)
	rm -rf a.out *.mod *.xmod *.o *~ __omni_tmp__
