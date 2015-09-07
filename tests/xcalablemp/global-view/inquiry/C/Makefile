XCC     = xmpcc
XRUN    = mpiexec
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .c

%.x: %.c 
	$(XCC) -o $@ $<

run: $(EXES)
	$(XRUN) -n 12 ./inquiry.x

clean:
	rm -f $(EXES) $(OBJS)
