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
OPT_O=-O2
MPI=mpiifort $(OPT_O)
CAF=xmpf90 $(OPT_O) --debug --verbose -I/opt/MPI/IntelMPI/5.0.0.028/include64

#-----------------
# XMP/ifort/GASNet
#-----------------
#OPT_O=-O2
#MPI=mpiifort $(OPT_O)
#CAF=xmpf90 $(OPT_O) --debug --verbose

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



#EXE=mpi.x mpi-nocom.x caf-wide.x caf-narrow.x caf-fit.x caf-nocom.x caf-direct.x caf-direct-nocom.x
EXE=caf-fitw.x caf-narroww.x caf-fits.x caf-fitc.x

#EXE=mpi-nocom.x caf-tt1.x caf-tt2.x caf-tt3.x caf-tt4.x caf-tt5.x caf-tt6.x caf-tt7.x caf-tt9.x caf-ttm.x
#EXE=mpi_try2.x fit_try2.x fit_try2w.x fit_try2g.x
#   fit_try2s.x: add one syncall at the head of sendp -- no effect 
#   fit_try2b.x: add arguments buf[123][ul] in xmpf_coarray_put_array -- no effect

all: $(EXE)

### ex.
### $ (echo m; echo 1 2 2) | mpirun -n 4 mpi.x > mpi122m.result
### $ (echo m; echo 1 2 2) | mpirun -n 4 caf-wide.x > caf-wide122m.result

## for bench-IG
fit_%.x: fit_%.f90
	mpiifort $(OPT_O) -o $@ $< /home/XMPTCA/hideto/Project/OMNI-IG/lib/xmpf_main.o  -I/opt/MPI/IntelMPI/5.0.0.028/include64 -L/home/XMPTCA/hideto/Project/OMNI-IG/lib -lxmpf -lxmp -openmp -L/home/XMPTCA/hideto/work/local/GASNet-1.24.2-mvapich2-gdr/lib -L/usr/lib64 -L/opt/mellanox/fca/lib -lgasnet-ibv-seq -libverbs -lfca -lpthread -lrt -L/usr/lib/gcc/x86_64-redhat-linux/4.4.7 -lgcc -lm


%.x: %.f90
	$(CAF) $< -o $@

caf-%.x: himenoBMTcaf-%.f90
	$(CAF) $< -o $@

mpi.x: himenoBMTxpr.f90
	$(MPI) $< -o $@

mpi-nocom.x: himenoBMTxpr-nocom.f90
	$(MPI) $< -o $@



mpi_try2.x: mpi_try2.f90
	mpiifort $(OPT_O) -o $@ $<  -I/opt/MPI/IntelMPI/5.0.0.028/include64 



run:	$(EXE:%.x=%.result)

%.result: %.x
	echo m >stdin.m122
	echo "1 2 2" >>stdin.m122
	mpirun -n 4 $< <stdin.m122

clean:
	rm -f $(EXE)
	rm -rf a.out *.mod *.xmod *.o *~ __omni_tmp__
