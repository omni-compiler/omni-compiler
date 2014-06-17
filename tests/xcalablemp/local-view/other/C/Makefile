XCC     = xmpcc -O2
XRUN    = mpirun

.PHONY: clean all default run

all: extern

extern: extern_a.c extern_b.c
	$(XCC) extern_a.c extern_b.c -o extern.x

run:
	$(XRUN) -n 2 extern.x

clean:
	rm -f *.o *.x
