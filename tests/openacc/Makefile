XCC = ompcc -acc
XRUN = mpirun -n 1
C_TESTS = $(wildcard *.c)
C_EXES = $(C_TESTS:.c=.x)
C_OBJS = $(C_TESTS:.c=.o)
EXES = $(C_EXES)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x

.c.x:
	$(XCC) -o $@ $<

run: $(EXES)
	$(XRUN) parallel_private.x
	$(XRUN) parallel-loop_private.x
	$(XRUN) loop_private_gang.x
	$(XRUN) loop_private_vector.x
	$(XRUN) parallel_if.x
	$(XRUN) loop_reduction_int.x
	$(XRUN) loop_reduction_float.x
	$(XRUN) loop_reduction_double.x
#	$(XRUN) loop_reduction.x
	$(XRUN) loop_private.x
	$(XRUN) kernel_config.x
	$(XRUN) update_array.x
	$(XRUN) update_subarray.x
	$(XRUN) update_async.x

clean:
	rm -f $(EXES) $(C_OBJS) *.cu
