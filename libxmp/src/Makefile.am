CC=@MPI_CMD@

include_HEADERS			= xmp.h xmp_index_macro.h xmp_comm_macro.h
lib_LIBRARIES			= libxmp.a
libxmp_a_SOURCES		= xmp_internal.h xmp_math_function.h xmp_constant.h \
				  xmp_lib.c xmp_util.c xmp_world.c xmp_nodes_stack.c \
                                  xmp_nodes.c xmp_template.c xmp_barrier.c xmp_reduce.c xmp_bcast.c \
                                  xmp_align.c xmp_shadow.c xmp_array_section.c xmp_loop.c xmp_gmove.c \
                                  xmp_math_function.c xmp_runtime.c

EXTRA_libxmp_a_SOURCES		= xmp_crt0.c xmp_threads_crt0.c
XMPCRTOBJ			= xmp_crt0.o
XMPTHREADSCRTOBJ		= xmp_threads_crt0.o

AM_CFLAGS   			= -std=c99

all-local: $(XMPCRTOBJ) $(XMPTHREADSCRTOBJ)

clean-local:
	rm -f *~ *.core core core.*

install-exec-local: $(XMPCRTOBJ) $(XMPTHREADSCRTOBJ)
	test -z "$(libdir)" || $(MKDIR_P) $(DESTDIR)$(libdir)
	$(INSTALL_DATA) $(XMPCRTOBJ) $(DESTDIR)$(libdir)
	$(INSTALL_DATA) $(XMPTHREADSCRTOBJ) $(DESTDIR)$(libdir)
