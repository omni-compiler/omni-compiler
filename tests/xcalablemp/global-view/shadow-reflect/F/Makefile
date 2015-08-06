XCC     = xmpf90
XRUN    = mpiexec
TESTS = $(wildcard *.f90)
EXES  = $(TESTS:.f90=.x)
OBJS  = $(TESTS:.f90=.o)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .f90

.f90.x:
	$(XCC) -o $@ $<

run: full.x
	$(XRUN) -n 2 ./full.x
	$(XRUN) -n 8 ./ordinal_reflect.x

clean:
	rm -f $(EXES) $(OBJS)
