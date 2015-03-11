CC            = @MPI_CC@
TLOG_INC_DIR  = $(top_builddir)/@LIBTLOG_MPI_DIR@
lib_LIBRARIES = libxmp.a
CFLAGS        = @MPI_CFLAGS@ -I../include -I$(TLOG_INC_DIR)
LDFLAGS       = @MPI_CLIBS@

libxmp_a_SOURCES = ../include/xmp_internal.h ../include/xmp_constant.h ../include/xmp_data_struct.h \
                   ../include/xmp_math_function.h xmp_lib.c xmp_util.c xmp_world.c xmp_nodes_stack.c \
                   xmp_nodes.c xmp_template.c xmp_barrier.c xmp_reduce.c xmp_bcast.c \
                   xmp_align.c xmp_shadow.c xmp_array_section.c xmp_loop.c xmp_gmove.c \
                   xmp_math_function.c xmp_runtime.c xmp_tlog_runtime.c xmp_io.c xmp_reflect.c xmp_pack_vector.c \
		   xmp_reflect_acc.c xmp_reduce_acc.c xmp_intrinsic.c xmp_sort.c xmp_async.c

if GASNET
    libxmp_a_SOURCES += xmp_coarray_set.c xmp_coarray_do.c xmp_gasnet_coarray.c xmp_lock.c xmp_gasnet_lock.c xmp_post_wait.c xmp_gasnet_post_wait.c
    CFLAGS           += -D_XMP_COARRAY_GASNET -D_XMP_COARRAY
endif

if FJRDMA
    libxmp_a_SOURCES += xmp_coarray_set.c xmp_coarray_do.c xmp_fjrdma_coarray_set.c xmp_fjrdma_coarray_do.c xmp_post_wait.c xmp_fjrdma_post_wait.c
    CFLAGS           += -D_XMP_COARRAY_FJRDMA -D_XMP_COARRAY
endif

if USER_LIBBLAS
    CFLAGS += -D_XMP_USER_LIBBLAS
endif

if TCA
    libxmp_a_SOURCES += xmp_reflect_tca.c
    CFLAGS        += -D_XMP_TCA
endif

if ENABLE_XACC
    libxmp_a_SOURCES += xmp_reflect_gpu.c xmp_gpu_pack.cu xmp_reduce_gpu.c
    CFLAGS           += -D_XMP_XACC -I@GPGPU_PREFIX@
    GPUCC             = @GPU_CMD@
    GPUCFLAGS         = @GPU_CFLAGS@

.cu.o:
	$(GPUCC) $(GPUCFLAGS) -I../include -c $@ $<
endif

clean-local:
	rm -f *~ *.core core core.*
