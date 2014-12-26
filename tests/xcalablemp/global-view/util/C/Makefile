INCDIR = ../../../../../Driver/etc
XMP_LINK_FLAGS=$(shell awk -FLinkerOption= '/LinkerOption=/ {print $$2}' $(INCDIR)/omc.conf | sed "s/\"//g")
MPI_CC:=$(shell awk -FCOMPILER= '/COMPILER=/ {print $$2}' $(INCDIR)/native.conf.xmp         | sed "s/\"//g")
GASNET_LIB=$(shell awk -FGASNET_LIB= '/GASNET_LIB=/ {print $$2}' $(INCDIR)/linker.conf.xmp  | sed "s/\"//g")
XCC     = xmpcc -O2
XRUN    = mpiexec
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .c

ixmp_sub.o: ./sub/ixmp_sub.c
	$(XCC) -c $^ -o $@

sub_mpi.o: ./sub/sub_mpi.c
	$(MPI_CC) $(XMP_LINK_FLAGS) -c $^ -o $@

%.x: %.c 
	$(XCC) -o $@ $<

mpi_xmp.x: mpi_xmp.c ixmp_sub.o
	$(MPI_CC) $(XMP_LINK_FLAGS) -o $@ mpi_xmp.c ixmp_sub.o -L../../../../../libxmp/src -lxmp $(GASNET_LIB)

xmp_mpi.x: xmp_mpi.c sub_mpi.o
	$(XCC) -o $@ xmp_mpi.c sub_mpi.o

run: $(EXES)
	$(XRUN) -n 4 ./xmp_mpi.x
	$(XRUN) -n 4 ./mpi_xmp.x

clean:
	rm -f $(EXES) $(OBJS) ixmp_sub.o sub_mpi.o
