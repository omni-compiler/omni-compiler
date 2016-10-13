COMPILER=$(shell xmpcc --show-env | grep OMNI_MPICC_CMD | sed 's/OMNI_MPICC_CMD=//' | sed "s/\"//g" | xargs basename)
MPIPATH=$(shell which $(COMPILER) | xargs dirname | awk '{print $$1"/../include"}')
XCC     = xmpcc
XRUN    = mpiexec
###EXES    = 117.x 119.x 278\ spacefile.x 293-hyphen.x 343_dynamicnode.x 344_include.x 355.x 377.x dnodes.x macro_in_pragma.x openmp.x static_desc.x 419.x ident.x tlog.x
### mpiexec -n 4 ./static_desc.x
### Fatal error in PMPI_Comm_split: Other MPI error, error stack:
### PMPI_Comm_split(495).................: MPI_Comm_split(comm=0xc4000005, color=1, key=0, new_comm=0x1af9200) failed
### PMPI_Comm_split(477).................: 
### MPIR_Comm_split_impl(249)............: 
### MPIR_Get_contextid_sparse_group(1193): Too many communicators (0/2048 free on this process; ignore_id=0)
EXES    = 117.x 119.x 278\ spacefile.x 293-hyphen.x 343_dynamicnode.x 344_include.x 355.x 377.x dnodes.x macro_in_pragma.x openmp.x  419.x ident.x tlog.x

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .c

tlog.x: tlog.c
	$(XCC) $^ -o $@ --profile tlog

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

419.x: 419.c
	$(XCC) -omp 419.c -o 419.x

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
	$(XRUN) -n 1 ./278\ spacefile.x
	$(XRUN) -n 4 ./419.x
	$(XRUN) -n 4 ./tlog.x; if [ ! -f trace.log ]; then exit 1; fi
#	$(XRUN) -n 4 ./static_desc.x

#-- cleaning
clean:
	rm -f $(EXES) *.o trace.log
	rm -rf __omni_tmp__
