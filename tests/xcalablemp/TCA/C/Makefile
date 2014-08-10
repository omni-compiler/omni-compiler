XCC     = xmpcc --Wf-facc --Wn-acc --Wl-acc -O2
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
	$(XRUN) -n 2 ./reflect_acc_1.x
	$(XRUN) -n 8 ./reflect_acc_2.x

clean:
	rm -f $(EXES) $(OBJS)
