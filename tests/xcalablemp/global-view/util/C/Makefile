INCDIR = ../../../../../Driver/etc
MPICOMPILER=$(shell cat $(INCDIR)/native.conf.xmp | grep "COMPILER=" | sed "s/COMPILER=//g" | sed "s/\"//g")
GASNETLIB=$(shell cat $(INCDIR)/linker.conf.xmp | grep "GASNET_LIB=" | sed "s/GASNET_LIB=//g" | sed "s/\"//g")
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
	$(MPICOMPILER) -c $^ -o $@

%.x: %.c 
	$(XCC) -o $@ $<

mpi_xmp.x: mpi_xmp.c ixmp_sub.o
	$(MPICOMPILER) -o $@ -fopenmp mpi_xmp.c ixmp_sub.o -L../../../../../libxmp/src -lxmp $(GASNETLIB)

xmp_mpi.x: xmp_mpi.c sub_mpi.o
	$(XCC) -o $@ xmp_mpi.c sub_mpi.o

run: $(EXES)
	$(XRUN) -n 4 ./xmp_mpi.x
	$(XRUN) -n 4 ./mpi_xmp.x

clean:
	rm -f $(EXES) $(OBJS) ixmp_sub.o sub_mpi.o
