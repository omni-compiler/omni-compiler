COMPILER=$(shell xmpcc --show-env | grep OMNI_MPICC_CMD | sed 's/OMNI_MPICC_CMD=//' | sed "s/\"//g")
MPIPATH=$(shell which $(COMPILER) | sed "s/bin\/$(COMPILER)/include/g")
XCC     = xmpcc -O2
XRUN    = mpirun
C_TESTS = $(wildcard *.c)
C_EXES  = $(C_TESTS:.c=.x)
C_OBJS  = $(C_TESTS:.c=.o)
EXES    = $(C_EXES)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .c

344_include.x: ./344_include.c
	$(XCC) -I$(MPIPATH) $^ -o $@

#-- default rules
.c.x:
	$(XCC) -o $@ $<

run:
	$(XRUN) -n 4 ./macro_in_pragma.x
	export XMP_NODE_SIZE0=2; \
	export XMP_NODE_SIZE1=2; \
	$(XRUN) -n 8 ./dnodes.x
	$(XRUN) -n 1 ./openmp.x
	$(XRUN) -n 1 ./119.x
	XMP_NODE_SIZE0=1 XMP_NODE_SIZE1=1 $(XRUN) -n 1 ./343_dynamicnode.x
	XMP_NODE_SIZE0=1 XMP_NODE_SIZE1=2 $(XRUN) -n 2 ./343_dynamicnode.x
	XMP_NODE_SIZE0=2 XMP_NODE_SIZE1=1 $(XRUN) -n 2 ./343_dynamicnode.x
	XMP_NODE_SIZE0=1 XMP_NODE_SIZE1=4 $(XRUN) -n 4 ./343_dynamicnode.x
	XMP_NODE_SIZE0=2 XMP_NODE_SIZE1=2 $(XRUN) -n 4 ./343_dynamicnode.x
	XMP_NODE_SIZE0=4 XMP_NODE_SIZE1=1 $(XRUN) -n 4 ./343_dynamicnode.x
	XMP_NODE_SIZE0=1 $(XRUN) -n 4 ./343_dynamicnode.x
	XMP_NODE_SIZE0=2 $(XRUN) -n 4 ./343_dynamicnode.x
	XMP_NODE_SIZE0=4 $(XRUN) -n 4 ./343_dynamicnode.x
	$(XRUN) -n 1 ./344_include.x
	$(XRUN) -n 1 ./293-hyphen.x

#-- cleaning
clean:
	rm -f $(EXES) $(C_OBJS)
