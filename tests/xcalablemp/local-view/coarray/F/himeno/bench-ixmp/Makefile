#-----------------
# XMP/gfortran/GASNet
#-----------------
#MPI=mpif90 -O3
#CAF=xmpf90 -O3 --debug --verbose
#MPILIB=$$(dirname $$(dirname $$(which mpif90)))/include
#CAF=xmpf90 --debug -I$(MPILIB) -g

#-----------------
# IG Configuration XMP/ifort/IntelMPI&(GASNet/GNU)
#-----------------
#OPT_O=-O2
#MPI=mpiifort $(OPT_O)
#CAF=xmpf90 $(OPT_O) --debug --verbose -I/opt/MPI/IntelMPI/5.0.0.028/include64

#-----------------
# XMP/ifort/GASNet
#-----------------
OPT_O=-O2
MPI=mpiifort $(OPT_O)
CAF=xmpf90 $(OPT_O) --debug --verbose

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



#EXE=mpi.x mpi-nocom.x caf-wide.x caf-narrow.x caf-fit.x caf-nocom.x caf-direct.x caf-direct-nocom.x \
#    caf-fitw.x caf-narroww.x caf-fits.x caf-fitc.x \
#    caf-wideg.x caf-narrowg.x caf-fitg.x caf-directg.x
#    caf-fitp.x

EXE=mpi.x caf-wide.x caf-direct.x caf-nobuf.x caf-nocom.x

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
