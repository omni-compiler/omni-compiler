INCDIR = ../../../../../Driver/etc
XMP_LINK_FLAGS=$(shell grep "LinkerOption=" $(INCDIR)/omf.conf | sed "s/LinkerOption=//g" | sed "s/\"//g")
MPI_FC=$(shell grep "COMPILER=" $(INCDIR)/native.conf.xmpf | sed "s/COMPILER=//g"         | sed "s/\"//g")
XMPC_LINK_FLAGS=$(shell xmpcc --show-env | sed -n '/OMNI_LINKER_OPT=/p' | sed 's/OMNI_LINKER_OPT=//' | sed "s/\"//g" | sed "s/-std=gnu99//") # Fix soon by mnakao

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
	$(MPI_FC) $(XMP_LINK_FLAGS) -c $^ -o $@

%.x: %.f90
	$(XFC) -o $@ $<

mpi_xmp.x: mpi_xmp.f90 ixmp_sub.o
	$(MPI_FC) $(XMP_LINK_FLAGS) -o $@ mpi_xmp.f90 ixmp_sub.o -L../../../../../libxmpf/src -lxmpf -L../../../../../libxmp/src -lxmp $(XMPC_LINK_FLAGS)

xmp_mpi.x: xmp_mpi.f90 sub_mpi.o
	$(XFC) -o $@ xmp_mpi.f90 sub_mpi.o

run: $(EXES)
	$(XRUN) -n  4 ./xmp_mpi.x
	$(XRUN) -n  4 ./mpi_xmp.x

clean:
	rm -f $(EXES) $(OBJS) ixmp_sub.o sub_mpi.o
