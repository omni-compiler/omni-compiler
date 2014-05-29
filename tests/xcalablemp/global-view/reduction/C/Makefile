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
	$(XRUN) -n 4 ./max.x
	$(XRUN) -n 4 ./min.x
	$(XRUN) -n 4 ./logical_add.x
	$(XRUN) -n 4 ./logical_or.x
	$(XRUN) -n 8 ./sum.x	
	$(XRUN) -n 8 ./sum2.x
	$(XRUN) -n 8 ./mult2.x
	$(XRUN) -n 8 ./or2.x
	$(XRUN) -n 8 ./max2.x
	$(XRUN) -n 8 ./min2.x
	$(XRUN) -n 8 ./logical_add2.x
	$(XRUN) -n 8 ./logical_or2.x

clean:
	rm -f $(EXES) $(OBJS)
