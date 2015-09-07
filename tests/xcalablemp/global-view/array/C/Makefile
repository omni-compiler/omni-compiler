XCC     = xmpcc
XRUN    = mpiexec
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .c

array.x: array.c
	$(XCC) -o $@ $< -lm

.c.x:
	$(XCC) -o $@ $<

run: $(EXES)
	$(XRUN) -n 4 ./array.x
	$(XRUN) -n 4 ./array2.x
	$(XRUN) -n 2 ./316_array.x
	$(XRUN) -n 4 ./361_array.x
	$(XRUN) -n 4 ./417_array1.x
	$(XRUN) -n 4 ./417_array2.x

clean:
	rm -f $(EXES) $(OBJS)
