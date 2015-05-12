XCC     = xmpcc -O2
XRUN    = mpiexec
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .c

chk.o: ./chk/chk.c
	$(XCC) -c $^ -o $@

%.x: %.c chk.o
	$(XCC) -o $@ $< ./chk.o

run: $(EXES)
	$(XRUN) -n 4 ./intrinsic_transpose.x
	$(XRUN) -n 8 ./intrinsic_transpose_FJ.x
	$(XRUN) -n 16 ./intrinsic_transpose_101_FJ.x
	$(XRUN) -n 4 ./intrinsic_matmul.x
	$(XRUN) -n 4 ./intrinsic_matmul_replica.x
	$(XRUN) -n 4 ./intrinsic_matmul_4_FJ.x
	$(XRUN) -n 8 ./intrinsic_matmul_FJ.x
	$(XRUN) -n 8 ./intrinsic_sort.x
	$(XRUN) -n 4 ./intrinsic_scatter.x
	$(XRUN) -n 4 ./intrinsic_scatter2x2.x
	$(XRUN) -n 4 ./intrinsic_gather.x
	$(XRUN) -n 4 ./intrinsic_gather2x2.x
	$(XRUN) -n 4 ./intrinsic_pack_unpack.x

clean:
	rm -f $(EXES) $(OBJS) chk.o
