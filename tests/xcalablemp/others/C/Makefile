XCC     = xmpcc -O2
XRUN    = mpirun
C_TESTS = $(wildcard *.c)
C_EXES  = $(C_TESTS:.c=.x)
C_OBJS  = $(C_TESTS:.c=.o)
EXES    = $(C_EXES)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .c

#-- default rules
.c.x:
	$(XCC) -o $@ $<

run:
	$(XRUN) -n 4 ./macro_in_pragma.x
	export XMP_NODE_SIZE0=2; \
	export XMP_NODE_SIZE1=2; \
	$(XRUN) -n 8 ./dnodes.x

#-- cleaning
clean:
	rm -f $(EXES) $(C_OBJS)
