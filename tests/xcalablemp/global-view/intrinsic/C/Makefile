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
	$(XRUN) -n 4 ./intrinsic_matmul.x
	$(XRUN) -n 4 ./intrinsic_matmul_replica.x

clean:
	rm -f $(EXES) $(OBJS) chk.o
