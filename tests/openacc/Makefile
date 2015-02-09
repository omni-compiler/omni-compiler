XCC = ompcc -acc
XRUN = mpirun -n 1
C_TESTS = $(wildcard *.c)
C_EXES = $(C_TESTS:.c=.x)
C_OBJS = $(C_TESTS:.c=.o)
EXES = $(C_EXES)

.PHONY: clean all default run

all default: $(EXES) update_host_subarray_type update_device_subarray_type data_present_type

.SUFFIXES: .x

.c.x:
	$(XCC) -o $@ $<

run: $(EXES)
	$(XRUN) parallel_private.x
	$(XRUN) parallel-loop_private.x
	$(XRUN) parallel-loop_reduction.x
	$(XRUN) loop_private_gang.x
	$(XRUN) loop_private_vector.x
	$(XRUN) parallel_if.x
	$(XRUN) loop_reduction_int.x
	$(XRUN) loop_reduction_float.x
	$(XRUN) loop_reduction_double.x
	$(XRUN) loop_reduction.x
	$(XRUN) loop_private.x
	$(XRUN) kernel_config.x
	$(XRUN) update_array.x
	$(XRUN) update_subarray.x
	$(XRUN) update_async.x
	$(XRUN) update_host_subarray.x
	$(XRUN) update_device_subarray.x

	$(XRUN) update_host_subarray_char.x
	$(XRUN) update_host_subarray_short.x
	$(XRUN) update_host_subarray_int.x
	$(XRUN) update_host_subarray_float.x
	$(XRUN) update_host_subarray_double.x

	$(XRUN) update_device_subarray_char.x
	$(XRUN) update_device_subarray_short.x
	$(XRUN) update_device_subarray_int.x
	$(XRUN) update_device_subarray_float.x
	$(XRUN) update_device_subarray_double.x

	$(XRUN) data_present_char.x
	$(XRUN) data_present_short.x
	$(XRUN) data_present_int.x
	$(XRUN) data_present_float.x
	$(XRUN) data_present_double.x

	$(XRUN) host_data.x
	$(XRUN) multi_device.x
	$(XRUN) longlong_loopvar.x
	$(XRUN) data_subarray.x
	$(XRUN) kernel_sync.x
	$(XRUN) enter_exit_data.x
	$(XRUN) bug_exit_data.x
	$(XRUN) bug_clause_comma.x

update_host_subarray_type:
	$(XCC) -o update_host_subarray_char.x update_host_subarray.c -DCHAR
	$(XCC) -o update_host_subarray_short.x update_host_subarray.c -DSHORT
	$(XCC) -o update_host_subarray_int.x update_host_subarray.c -DINT
	$(XCC) -o update_host_subarray_float.x update_host_subarray.c -DFLOAT
	$(XCC) -o update_host_subarray_double.x update_host_subarray.c -DDOUBLE

update_device_subarray_type:
	$(XCC) -o update_device_subarray_char.x update_device_subarray.c -DCHAR
	$(XCC) -o update_device_subarray_short.x update_device_subarray.c -DSHORT
	$(XCC) -o update_device_subarray_int.x update_device_subarray.c -DINT
	$(XCC) -o update_device_subarray_float.x update_device_subarray.c -DFLOAT
	$(XCC) -o update_device_subarray_double.x update_device_subarray.c -DDOUBLE

data_present_type:
	$(XCC) -o data_present_char.x data_present.c -DCHAR
	$(XCC) -o data_present_short.x data_present.c -DSHORT
	$(XCC) -o data_present_int.x data_present.c -DINT
	$(XCC) -o data_present_float.x data_present.c -DFLOAT
	$(XCC) -o data_present_double.x data_present.c -DDOUBLE

clean:
	rm -f $(EXES) $(C_OBJS) *.cu *.x
