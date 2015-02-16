CC            = @MPI_CC@
XMP_INC_DIR   = ../../libxmp/include
XACC_INC_DIR  = ../include
CUDA_DIR	  = @GPGPU_PREFIX@
GPUCC		= @GPU_CMD@
GPUCFLAGS	= @GPU_CFLAGS@
AM_CFLAGS     = -I$(XMP_INC_DIR) -I$(XACC_INC_DIR) -I$(CUDA_DIR) @MPI_CFLAGS@
COMPILE = $(CC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) \
	$(CPPFLAGS) $(AM_CFLAGS)
LINK = $(CCLD) $(AM_CFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@

lib_LIBRARIES = libxacc.a
libxacc_a_SOURCES = $(XACC_INC_DIR)/xacc_internal.h \
	xacc.c xacc_pack.cu xacc_reflect.c
#                   xacc_device_stack.c
#					$(XMP_INC_DIR)/xmp_internal.h

.cu.o:
	$(GPUCC) $(GPUCFLAGS) -I$(XMP_INC_DIR) -c $@ $<

clean-local:
	rm -f *~ *.core core core.*
