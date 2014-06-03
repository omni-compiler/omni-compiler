CC            = @BECC@
XMP_INC_DIR   = ../include
TLOG_INC_DIR  = $(top_builddir)/@LIBTLOG_MPI_DIR@
lib_LIBRARIES = libxmp.a
AM_CFLAGS     = -I$(XMP_INC_DIR) -I$(TLOG_INC_DIR) @BECC_OPTION@ 
COMPILE = $(CC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) \
	$(CPPFLAGS) $(AM_CFLAGS)
LINK = $(CCLD) $(AM_CFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@

libxmp_a_SOURCES = $(XMP_INC_DIR)/xmp_internal.h $(XMP_INC_DIR)/xmp_constant.h $(XMP_INC_DIR)/xmp_data_struct.h \
                   xmp_math_function.h xmp_lib.c xmp_util.c xmp_world.c xmp_nodes_stack.c \
                   xmp_nodes.c xmp_template.c xmp_barrier.c xmp_reduce.c xmp_bcast.c \
                   xmp_align.c xmp_shadow.c xmp_array_section.c xmp_loop.c xmp_gmove.c \
                   xmp_math_function.c xmp_runtime.c xmp_tlog_runtime.c xmp_io.c xmp_reflect.c xmp_pack_vector.c

if GASNET
    include @GASNET_PREFIX@/include/@GASNET_CONDUIT@-conduit/@GASNET_CONDUIT@-seq.mak
    libxmp_a_SOURCES   += xmp_coarray_set.c xmp_coarray_do.c xmp_gasnet_coarray.c xmp_lock.c xmp_gasnet_lock.c xmp_post.c xmp_gasnet_post.c
    AM_LDFLAGS          = $(GASNET_LDFLAGS) $(GASNET_LIBS)
    AM_CFLAGS          += -D_XMP_COARRAY_GASNET $(GASNET_CPPFLAGS) $(GASNET_CFLAGS)
endif

if FJRDMA
    libxmp_a_SOURCES += xmp_coarray_set.c xmp_coarray_do.c xmp_fjrdma_coarray_set.c xmp_fjrdma_coarray_do.c
    AM_CFLAGS        += -Wall -Wpointer-arith -std=gnu99 -D_XMP_COARRAY_FJRDMA
endif

xmp_pack_vector.o: xmp_pack_vector.c
	@BECC@ $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) \
	$(CPPFLAGS) $(AM_CFLAGS) @BECC_OMP_OPTION@ -c $<

clean-local:
	rm -f *~ *.core core core.*
