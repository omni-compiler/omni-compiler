XFC     = xmpf90
FCFLAGS = -xacc --Wn"-acc -Minfo=accel" --Wl"-acc" -lstdc++
XRUN    = mpiexec
TESTS = $(wildcard *.f90)
EXES  = $(TESTS:.f90=.x)
OBJS  = $(TESTS:.f90=.o)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .f90

.f90.x:
	$(XFC) -o $@ $< $(FCFLAGS)

run: $(EXES)
	$(XRUN) -n 4 ./xacc_distarray.x
	$(XRUN) -n 4 ./xacc_reflect.x
	$(XRUN) -n 4 ./xacc_reflect_oneside.x
	$(XRUN) -n 4 ./xacc_reduction.x
	$(XRUN) -n 4 ./xacc_bcast.x

clean:
	rm -f $(EXES) $(OBJS)
