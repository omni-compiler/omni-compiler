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

run: shadow_reflect.x
	$(XRUN) -n 4 ./shadow_reflect.x
	$(XRUN) -n 8 ./ordinal_reflect.x
	$(XRUN) -n 4 ./full2.x
	$(XRUN) -n 4 ./loop.x

clean:
	rm -f $(EXES) $(OBJS)
