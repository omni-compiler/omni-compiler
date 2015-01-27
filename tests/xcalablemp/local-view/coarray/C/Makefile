XCC     = xmpcc -O2
XRUN    = mpirun
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all default run

all default: $(EXES)

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
	$(XRUN) -np 3 coarray_local_put.x
	$(XRUN) -np 3 coarray_local_get.x

clean:
	rm -f $(EXES) $(OBJS)

