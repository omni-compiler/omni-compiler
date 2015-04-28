XCC     = xmpcc -O2
XRUN    = mpiexec
C_TESTS = $(wildcard *.c)
C_EXES  = $(C_TESTS:.c=.x)
C_OBJS  = $(C_TESTS:.c=.o)
EXES    = $(C_EXES)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x

.c.x:
	$(XCC) -o $@ $<

run: 
	$(XRUN) -n 2 ./coarray_put_5dims.x
	$(XRUN) -n 2 ./coarray_put_6dims.x
	$(XRUN) -n 2 ./coarray_put_7dims.x
	$(XRUN) -n 2 ./coarray_get_5dims.x
	$(XRUN) -n 2 ./coarray_get_6dims.x
	$(XRUN) -n 2 ./coarray_get_7dims.x

clean:
	rm -f $(EXES) $(C_OBJS)

