XFC     = xmpf90 -O2
XRUN    = mpiexec
F_TESTS = $(wildcard *.f90)
F_EXES  = $(F_TESTS:.f90=.x)
F_OBJS  = $(F_TESTS:.f90=.o)
EXES    = $(F_EXES)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .f90

.f90.x:
	$(XFC) -o $@ $<

run: $(EXES)
	$(XRUN) -n 2 ./gmove_ata_2a1t_c.x
	$(XRUN) -n 2 ./gmove_ata_2a1t_b_h.x

clean:
	rm -f $(EXES) $(F_OBJS) _xmpf_module_INIT.o
