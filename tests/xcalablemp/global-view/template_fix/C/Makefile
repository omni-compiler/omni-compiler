XCC     = xmpcc
XRUN    = mpiexec
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .c

.c.x:
	$(XCC) -o $@ $<

run: $(EXES)
	$(XRUN) -n 4 ./reflect.x
	$(XRUN) -n 8 ./reflect2.x
	$(XRUN) -n 4 ./block.x
	$(XRUN) -n 4 ./block_cyclic.x
	$(XRUN) -n 4 ./gblock.x

clean:
	rm -f $(EXES) $(OBJS)
