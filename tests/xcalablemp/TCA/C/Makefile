XCC     = xmpcc --Wf-facc --Wn-acc --Wl-acc
XRUN    = mpiexe
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .c

.c.x:
	$(XCC) -o $@ $<

clean:
	rm -f $(EXES) $(OBJS)
