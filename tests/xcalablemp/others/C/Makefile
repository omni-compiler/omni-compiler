COMPILER=$(shell xmpcc --show-env | grep OMNI_MPICC_CMD | sed 's/OMNI_MPICC_CMD=//' | sed "s/\"//g" | xargs basename)
MPIPATH=$(shell which $(COMPILER) | xargs dirname | awk '{print $$1"/../include"}')
XCC     = xmpcc -O2
XRUN    = mpiexec
EXES    = 117.x 119.x 278\ spacefile.x 293-hyphen.x 343_dynamicnode.x 344_include.x 355.x 377.x dnodes.x macro_in_pragma.x openmp.x

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .c

344_include.x: 344_include.c
	$(XCC) -I$(MPIPATH) $^ -o $@

278\ spacefile.x: 278\ spacefile.c
	$(XCC) 278\ spacefile.c -DPASS='"P A S S"' -o 278\ spacefile.x

openmp.x: openmp.c
	$(XCC) -omp openmp.c -o openmp.x

377.x: 377.c
	$(XCC) -c -o 377.x $<; \
	[ ! -f 377.x ] && exit 1; \
	rm -rf __omni_tmp__; \
	$(XCC) -c $< --debug 1> /dev/null; \
	[ ! -d __omni_tmp__ ] && exit 1; \
	echo "PASS"; 

#-- default rules
.c.x:
	$(XCC) -o $@ $<

run: macro_in_pragma.x dnodes.x openmp.x 119.x 343_dynamicnode.x 344_include.x 293-hyphen.x
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
	$(XRUN) -n 1 ./278\ spacefile.x

#-- cleaning
clean:
	rm -f $(EXES) *.o
	rm -rf __omni_tmp__
