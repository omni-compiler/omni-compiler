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
	$(XRUN) -n 4 ./barrier.x
	$(XRUN) -n 4 ./reduction.x
	$(XRUN) -n 8 ./tasks.x
	$(XRUN) -n 16 ./nocomm.x

clean:
	rm -f $(EXES) $(OBJS)
