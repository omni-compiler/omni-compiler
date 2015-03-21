XCC     = xmpcc -O2
XRUN    = mpirun
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x

.c.x:
	$(XCC) -o $@ $<

run: shadow_reflect.x
	$(XRUN) -np 4 ./shadow_reflect.x

clean:
	rm -f $(EXES) $(OBJS)
