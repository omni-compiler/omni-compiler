INCDIR = ../../../../../Driver/etc
include $(INCDIR)/native.conf.xmpf
XFC     = xmpf90 -O2
XRUN    = mpiexec
TESTS = $(wildcard *.f90)
EXES  = $(TESTS:.f90=.x)
OBJS  = $(TESTS:.f90=.o)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .f90

ixmp_sub.o: ./sub/ixmp_sub.f90
	$(XFC) -c $^ -o $@

sub_mpi.o: ./sub/sub_mpi.f90
	$(COMPILER) -c $^ -o $@

%.x: %.f90
	$(XFC) -o $@ $<

mpi_xmp.x: mpi_xmp.f90 ixmp_sub.o
	$(COMPILER) -fopenmp -o $@ mpi_xmp.f90 ixmp_sub.o -L../../../../../libxmpf/src -lxmpf -L../../../../../libxmp/src -lxmp

xmp_mpi.x: xmp_mpi.f90 sub_mpi.o
	$(XFC) -o $@ xmp_mpi.f90 sub_mpi.o

run: $(EXES)
	$(XRUN) -n 12 ./inquiry.x
#	$(XRUN) -n  4 ./xmp_mpi.x
	$(XRUN) -n  4 ./mpi_xmp.x

clean:
	rm -f $(EXES) $(OBJS) ixmp_sub.o sub_mpi.o
