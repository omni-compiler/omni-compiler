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
	$(XRUN) -n 4 ./block.x
	$(XRUN) -n 4 ./gblock.x
	$(XRUN) -n 4 ./reflect.x
	$(XRUN) -n 16 ./gblock2.x
	$(XRUN) -n 16 ./cyclic3.x
	$(XRUN) -n 16 ./cyclic_block.x
	$(XRUN) -n 16 ./cyclic_block3.x

clean:
	rm -f $(EXES) $(OBJS)
