XFC     = xmpf90 -O2
XRUN    = mpiexec
TESTS = $(wildcard *.f90)
EXES  = $(TESTS:.f90=.x)
OBJS  = $(TESTS:.f90=.o)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .f90

%.x: %.f90
	$(XFC) -o $@ $<

run: $(EXES)
	$(XRUN) -n 12 ./inquiry.x

clean:
	rm -f $(EXES) $(OBJS)
