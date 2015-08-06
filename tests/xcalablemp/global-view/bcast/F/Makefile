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
	$(XRUN) -n 4 ./bcast.x
	$(XRUN) -n 4 ./bcast2.x
	$(XRUN) -n 4 ./bcast3.x
	$(XRUN) -n 4 ./bcast4.x
	$(XRUN) -n 8 ./bcast5.x
	$(XRUN) -n 4 ./bcast6.x
	$(XRUN) -n 4 ./bcast7.x
	$(XRUN) -n 4 ./bcast8.x
	$(XRUN) -n 8 ./bcast9.x
	$(XRUN) -n 8 ./bcast10.x
	$(XRUN) -n 8 ./bcast11.x
	$(XRUN) -n 8 ./bcast12.x

clean:
	rm -f $(EXES) $(OBJS)
