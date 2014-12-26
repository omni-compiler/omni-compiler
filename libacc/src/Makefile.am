ACC_INC_DIR=../include
CC=@MPI_CC@
GPUCC=@GPU_CMD@
GPUCFLAGS=@GPU_CFLAGS@
AM_CFLAGS     = -I$(ACC_INC_DIR) @MPI_CFLAGS@
COMPILE = $(CC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) \
	$(CPPFLAGS) $(AM_CFLAGS)
LINK = $(CCLD) $(AM_CFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@

lib_LIBRARIES		= libacc.a
libacc_a_SOURCES	= $(ACC_INC_DIR)/acc_internal.h $(ACC_INC_DIR)/acc_gpu_internal.h \
			  $(ACC_INC_DIR)/acc_gpu_constant.h $(ACC_INC_DIR)/acc_gpu_data_struct.h \
			  acc_gpu_data_table.c acc_runtime.c acc_util.c

libacc_a_LIBADD		= acc_runtime.o acc_gpu_data_table.o acc_util.o \
			  acc_gpu_runtime.o acc_gpu_data.o acc_gpu_stream.o acc_gpu_util.o acc_gpu_mpool.o acc_gpu_pack.o

.cu.o:
	$(GPUCC) $(GPUCFLAGS) -I$(ACC_INC_DIR) -c -o $@ $<

