XCC     = xmpcc -O3
XRUN    = mpirun
C_TESTS = $(wildcard *.c)
C_EXES  = $(C_TESTS:.c=.x)
C_OBJS  = $(C_TESTS:.c=.o)
EXES    = $(C_EXES)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x

.c.x:
	$(XCC) -o $@ $<

run: $(EXES)
	$(XRUN) -np 2 coarray_scalar.x
	$(XRUN) -np 2 coarray_vector.x
	$(XRUN) -np 2 coarray_stride.x
	$(XRUN) -np 4 post_wait.x
	$(XRUN) -np 4 shadow.x

clean:
	rm -f $(EXES) $(C_OBJS)
