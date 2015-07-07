XMP_INC_DIR=../include
CC=@MPI_CC@
GPUCC=@GPU_CMD@
GPUCFLAGS=@GPU_CFLAGS@
AM_CFLAGS = -I$(XMP_INC_DIR) @MPI_CFLAGS@ #-I$(TLOG_INC_DIR)
COMPILE=$(CC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) \
	$(CPPFLAGS) $(AM_CFLAGS)

lib_LIBRARIES		= libxmp_gpu.a
libxmp_gpu_a_SOURCES	= $(XMP_INC_DIR)/xmp_internal.h $(XMP_INC_DIR)/xmp_constant.h \
                          $(XMP_INC_DIR)/xmp_data_struct.h xmp_gpu_internal.h \
                          xmp_gpu_lib.c xmp_gpu_shadow.c

libxmp_gpu_a_LIBADD	= xmp_gpu_runtime.o xmp_gpu_util.o xmp_gpu_data.o xmp_gpu_sync.o xmp_gpu_array_section.o

.cu.o:
	$(GPUCC) $(GPUCFLAGS) -I$(XMP_INC_DIR) -c $@ $<


