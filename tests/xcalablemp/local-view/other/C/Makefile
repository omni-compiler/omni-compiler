XCC     = xmpcc
XRUN    = mpiexec

.PHONY: clean all default run

all: extern.x

extern.x: extern_a.c extern_b.c
	$(XCC) extern_a.c extern_b.c -o extern.x

run:
	$(XRUN) -n 2 ./extern.x

clean:
	rm -f *.o *.x
