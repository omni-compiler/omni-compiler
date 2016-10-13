XCC     = xmpcc
XRUN    = mpiexec
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x

.c.x:
	$(XCC) -o $@ $<

run: 
	$(XRUN) -n 2 ./atomic_define.x
	$(XRUN) -n 2 ./atomic_ref.x
	$(XRUN) -n 2 ./atomic_define_ref.x

clean:
	rm -f $(EXES) $(OBJS)

