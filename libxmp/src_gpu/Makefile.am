XMP_INC_DIR=../include
CC=@CC@
GPUCC=@GPU_CMD@

lib_LIBRARIES			= libxmp_gpu.a
libxmp_gpu_a_SOURCES		= $(XMP_INC_DIR)/xmp_internal.h $(XMP_INC_DIR)/xmp_constant.h xmp_gpu_internal.h \
                                  xmp_gpu_lib.c

libxmp_gpu_a_LIBADD		= xmp_gpu_runtime.o xmp_gpu_util.o xmp_gpu_data.o xmp_gpu_sync.o xmp_gpu_schedule.o

.cu.o:
	$(GPUCC) -I$(XMP_INC_DIR) -O2 -c $@ $<

AM_CFLAGS			= -Wall -std=c99 -I$(XMP_INC_DIR)
