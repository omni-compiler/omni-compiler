MPI_FC=$(shell xmpf90 --show-env          | sed -n '/OMNI_MPIFC_CMD=/p'  | sed 's/OMNI_MPIFC_CMD=//'  | sed "s/\"//g")
XMPF_LINK_FLAGS=$(shell xmpf90 --show-env | sed -n '/OMNI_LINKER_OPT=/p' | sed 's/OMNI_LINKER_OPT=//' | sed "s/\"//g") 

XFC     = xmpf90
XRUN    = mpiexec
TESTS = $(wildcard *.f90)
EXES  = $(TESTS:.f90=.x)
OBJS  = $(TESTS:.f90=.o)

.PHONY: clean all default run submit showlog cleanlog

all default: $(EXES)

.SUFFIXES: .x .f90

ixmp_sub.o: ./sub/ixmp_sub.f90
	$(XFC) -c $^ -o $@

sub_mpi.o: ./sub/sub_mpi.f90
	$(MPI_FC) -c $^ -o $@

%.x: %.f90
	$(XFC) -o $@ $<

mpi_xmp.o : mpi_xmp.f90
	$(MPI_FC) -c mpi_xmp.f90

mpi_xmp.x: mpi_xmp.o ixmp_sub.o
	$(XFC) -o $@ mpi_xmp.o ixmp_sub.o

xmp_mpi.x: xmp_mpi.f90 sub_mpi.o
	$(XFC) -o $@ xmp_mpi.f90 sub_mpi.o

run: $(EXES)
	$(XRUN) -n  4 ./xmp_mpi.x
	$(XRUN) -n  4 ./mpi_xmp.x

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

