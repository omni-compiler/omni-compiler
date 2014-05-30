XCC     = xmpf90 -O2
XRUN    = mpiexec
TESTS = $(wildcard *.f90)
EXES  = $(TESTS:.f90=.x)
OBJS  = $(TESTS:.f90=.o)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .f90

.f90.x:
	$(XCC) -o $@ $<

run: $(EXES)
	$(XRUN) -n 4 ./barrier.x

clean:
	rm -f $(EXES) $(OBJS)
