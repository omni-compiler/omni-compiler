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

run: $(EXES)
	$(XRUN) -n 8 ./bcast_async.x
	$(XRUN) -n 8 ./plus2_async.x
	$(XRUN) -n 4 ./gmove_async.x

clean:
	rm -f $(EXES) $(OBJS)
