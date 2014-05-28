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
	$(XRUN) -n 4 ./and.x
	$(XRUN) -n 4 ./mult.x
	$(XRUN) -n 4 ./or.x
	$(XRUN) -n 4 ./max.x
	$(XRUN) -n 4 ./min.x
	$(XRUN) -n 4 ./mult2.x
	$(XRUN) -n 4 ./iand.x
	$(XRUN) -n 4 ./or.x
	$(XRUN) -n 8 ./max2.x

clean:
	rm -f $(EXES) $(OBJS)
