XCC     = xmpcc -O2
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
	$(XRUN) -n 4 ./and.x
	$(XRUN) -n 4 ./mult.x
	$(XRUN) -n 4 ./plus.x
	$(XRUN) -n 4 ./or.x

clean:
	rm -f $(EXES) $(OBJS)
