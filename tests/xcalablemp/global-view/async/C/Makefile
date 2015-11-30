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
	$(XRUN) -n 16 ./bcast_async.x
	$(XRUN) -n 8 ./plus2_async.x
	$(XRUN) -n 4 ./gmove_async.x

clean:
	rm -f $(EXES) $(OBJS)
