XFC     = xmpf90
XRUN    = mpiexec
TESTS = $(wildcard *.f90)
EXES  = $(TESTS:.f90=.x)
OBJS  = $(TESTS:.f90=.o)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .f90

.f90.x:
	$(XFC) -o $@ $<

run: $(EXES)
#	$(XRUN) -n 8 ./361_exprInOnClause.x
#	$(XRUN) -n 4 ./array.x
	$(XRUN) -n 4 ./offset.x
	$(XRUN) -n 4 ./361_array.x

clean:
	rm -f $(EXES) $(OBJS)
