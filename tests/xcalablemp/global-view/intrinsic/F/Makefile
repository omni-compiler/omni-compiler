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
	$(XRUN) -n 4 ./intrinsic_transpose_4_FJ.x
	$(XRUN) -n 8 ./intrinsic_transpose_8_FJ.x
	$(XRUN) -n 16 ./intrinsic_transpose_FJ.x
	$(XRUN) -n 4 ./intrinsic_matmul.x
	$(XRUN) -n 4 ./intrinsic_matmul_replica.x
	$(XRUN) -n 2 ./intrinsic_matmul_2_FJ.x
	$(XRUN) -n 4 ./intrinsic_matmul_4_FJ.x
	$(XRUN) -n 8 ./intrinsic_matmul_8_FJ.x
	$(XRUN) -n 16 ./intrinsic_matmul_FJ.x
	$(XRUN) -n 8 ./intrinsic_sort.x
	$(XRUN) -n 1 ./363_loc.x
	$(XRUN) -n 4 ./intrinsic_scatter.x
	$(XRUN) -n 4 ./intrinsic_scatter2x2.x
	$(XRUN) -n 4 ./intrinsic_gather.x
	$(XRUN) -n 4 ./intrinsic_gather2x2.x
	$(XRUN) -n 4 ./intrinsic_pack_unpack.x

clean:
	rm -f $(EXES) $(OBJS) _xmpf_module_INIT.o chk.o *.mod *.xmod
