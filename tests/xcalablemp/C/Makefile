XCC     = xmpcc -O3
XRUN    = mpirun
C_TESTS = $(wildcard *.c)
C_EXES  = $(C_TESTS:.c=.x)
C_OBJS  = $(C_TESTS:.c=.o)
EXES    = $(C_EXES)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x

%.x: %.c
			$(XCC) -o $@ $^

run: $(EXES)
			$(XRUN) -np 2 $(EXES)

clean:
			rm -f $(EXES) $(C_OBJS)
