ACC_INC_DIR=../include
CC=@BECC@
CUDA_DIR	  = @GPGPU_PREFIX@
GPUCC=@GPU_CMD@
GPUCFLAGS	= @GPU_CFLAGS@

lib_LIBRARIES		= libacc.a
AM_CFLAGS     = -I$(ACC_INC_DIR) -I$(CUDA_DIR)/include -I$(TLOG_INC_DIR) @BECC_OPTION@ 
COMPILE = $(CC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) \
	$(CPPFLAGS) $(AM_CFLAGS)
LINK = $(CCLD) $(AM_CFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@

libacc_a_SOURCES	= $(ACC_INC_DIR)/acc_internal.h $(ACC_INC_DIR)/acc_gpu_internal.h \
			  $(ACC_INC_DIR)/acc_gpu_constant.h $(ACC_INC_DIR)/acc_gpu_data_struct.h \
			  acc_gpu_data_table.c acc_runtime.c acc_util.c acc_memory_map.c acc_gpu_runtime.c acc_gpu_data.c acc_gpu_stream.c acc_gpu_util.c acc_gpu_mpool.c acc_gpu_pack.cu 

#libacc_a_LIBADD		= acc_runtime.o acc_gpu_data_table.o acc_util.o \
			  acc_gpu_runtime.o acc_gpu_data.o acc_gpu_stream.o acc_gpu_util.o acc_gpu_mpool.o acc_gpu_pack.o acc_memory_map.o

.cu.o:
	$(GPUCC) -I$(ACC_INC_DIR) -O3 -c -o $@ $<

clean-local:
	rm -f *~ *.core core core.*
