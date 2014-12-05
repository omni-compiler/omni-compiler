XFC     = xmpf90 -O2
XRUN    = mpiexec
TESTS = $(wildcard *.f90)
EXES  = $(TESTS:.f90=.x)
OBJS  = $(TESTS:.f90=.o)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .f90

chk.o: ./chk/chk.f90
	$(XFC) -c $^ -o $@

%.x: %.f90 chk.o
	$(XFC) -o $@ $< ./chk.o

run: $(EXES)
	$(XRUN) -n 4 ./intrinsic_transpose.x
	$(XRUN) -n 16 ./intrinsic_transpose_FJ.x
	$(XRUN) -n 4 ./intrinsic_matmul.x
	$(XRUN) -n 4 ./intrinsic_matmul_replica.x
	$(XRUN) -n 16 ./intrinsic_matmul_FJ.x

clean:
	rm -f $(EXES) $(OBJS) _xmpf_module_INIT.o chk.o
