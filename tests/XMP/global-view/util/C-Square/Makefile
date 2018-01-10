INCDIR = ../../../../../Driver/etc
XMP_LINK_FLAGS=$(shell xmpcc --show-env | sed -n '/OMNI_LINKER_OPT=/p' | sed 's/OMNI_LINKER_OPT=//' | sed "s/\"//g")
MPI_CC=$(shell xmpcc --show-env         | sed -n '/OMNI_MPICC_CMD=/p'  | sed 's/OMNI_MPICC_CMD=//'  | sed "s/\"//g")
OMNI_HOME=$(shell xmpcc --show-env      | sed -n '/OMNI_HOME=/p'       | sed 's/OMNI_HOME=//'  | sed "s/\"//g")

XCC     = xmpcc
XRUN    = mpiexec
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all default run submit showlog cleanlog

all default: $(EXES)

.SUFFIXES: .x .c

ixmp_sub.o: ./sub/ixmp_sub.c
	$(XCC) -c $^ -o $@

sub_mpi.o: ./sub/sub_mpi.c
	$(MPI_CC) $(XMP_LINK_FLAGS) -c $^ -o $@

%.x: %.c 
	$(XCC) -o $@ $<

mpi_xmp.o: mpi_xmp.c
	$(MPI_CC) -o $@ -c mpi_xmp.c -I$(OMNI_HOME)/include

mpi_xmp.x: mpi_xmp.o ixmp_sub.o
	$(XCC) -o $@ mpi_xmp.o ixmp_sub.o

xmp_mpi.x: xmp_mpi.c sub_mpi.o
	$(XCC) -o $@ xmp_mpi.c sub_mpi.o

run: $(EXES)
	$(XRUN) -n 4 ./xmp_mpi.x
	$(XRUN) -n 4 ./mpi_xmp.x

RUN:
	mkdir RUN

RUN/%.x:: %.x
	cp $< $@

RUN/go.sh: go.template Makefile
	cp $< $@ && grep XRUN Makefile | sed -e 's/(XRUN)/{XRUN}/' -e 's/ *= */=/' | grep -v Makefile >>$@

submit: all RUN RUN/go.sh $(EXES:%=RUN/%)
	cd RUN; pjsub go.sh

showlog:
	cat RUN/go.sh.e* RUN/go.sh.o*

cleanlog:
	rm -rf RUN

clean: cleanlog
	rm -f $(EXES) $(OBJS) ixmp_sub.o sub_mpi.o

