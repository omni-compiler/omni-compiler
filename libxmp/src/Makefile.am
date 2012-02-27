XMP_INC_DIR=../include
TLOG_INC_DIR=$(top_builddir)/@LIBTLOG_MPI_DIR@
CC=@MPI_CMD@

lib_LIBRARIES			= libxmp.a
libxmp_a_SOURCES		= $(XMP_INC_DIR)/xmp_internal.h $(XMP_INC_DIR)/xmp_constant.h \
                                  $(XMP_INC_DIR)/xmp_data_struct.h \
                                  xmp_math_function.h \
				  xmp_lib.c xmp_util.c xmp_world.c xmp_nodes_stack.c \
                                  xmp_nodes.c xmp_template.c xmp_barrier.c xmp_reduce.c xmp_bcast.c \
                                  xmp_align.c xmp_shadow.c xmp_array_section.c xmp_loop.c xmp_gmove.c \
                                  xmp_math_function.c xmp_runtime.c xmp_coarray.c xmp_coarray_RMA.c xmp_tlog_runtime.c xmp_io.c

AM_CFLAGS   			= -Wall -Wpointer-arith -std=c99 -I$(XMP_INC_DIR) -I$(TLOG_INC_DIR)

clean-local:
	rm -f *~ *.core core core.*
