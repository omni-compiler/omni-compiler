XCC     = xmpcc -O2
XRUN    = mpirun
C_TESTS = $(wildcard *.c)
C_EXES  = $(C_TESTS:.c=.x)
C_OBJS  = $(C_TESTS:.c=.o)

.PHONY: clean all default run

all default: $(C_EXES)

.SUFFIXES: .x

.c.x:
	$(XCC) -o $@ $<

run: 
	$(XRUN) -np 1 coarray_translation.x
	$(XRUN) -np 2 coarray_scalar.x
	$(XRUN) -np 2 coarray_vector.x
	$(XRUN) -np 2 coarray_stride.x
	$(XRUN) -np 2 coarray_put_1dim.x
	$(XRUN) -np 2 coarray_put_2dims.x
	$(XRUN) -np 2 coarray_put_3dims.x
	$(XRUN) -np 2 coarray_put_4dims.x
	$(XRUN) -np 2 coarray_get_1dim.x
	$(XRUN) -np 2 coarray_get_2dims.x
	$(XRUN) -np 2 coarray_get_3dims.x
	$(XRUN) -np 2 coarray_get_4dims.x

clean:
	rm -f $(C_EXES) $(C_OBJS)

