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
	$(XRUN) -n 4 ./and.x
	$(XRUN) -n 4 ./or.x
	$(XRUN) -n 4 ./logical_and.x
	$(XRUN) -n 4 ./logical_or.x
	$(XRUN) -n 4 ./mult.x
	$(XRUN) -n 4 ./plus.x
	$(XRUN) -n 4 ./max.x
	$(XRUN) -n 4 ./min.x
	$(XRUN) -n 4 ./raise.x
	$(XRUN) -n 8 ./plus2.x
	$(XRUN) -n 8 ./max2.x
	$(XRUN) -n 8 ./min2.x
	$(XRUN) -n 8 ./mult2.x
	$(XRUN) -n 3 ./array.x

clean:
	rm -f $(EXES) $(OBJS)
