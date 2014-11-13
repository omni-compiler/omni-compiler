CC=@BECC@
XMP_INC_DIR  =../../libxmp/include
XMP_SRC_DIR  =../../libxmp/src
XMPF_INC_DIR =../include
#TLOG_INC_DIR=$(top_builddir)/@LIBTLOG_MPI_DIR@
#GASNET_INC_DIR=@GASNET_PREFIX@/include
#GASNET_CONDUIT_DIR=$(GASNET_INC_DIR)/@GASNET_CONDUIT@-conduit
AM_CFLAGS = -I$(XMPF_INC_DIR) -I$(XMP_INC_DIR) -I$(XMP_SRC_DIR)  @BECC_OPTION@

lib_LIBRARIES     = libxmpf.a
libxmpf_a_SOURCES = $(XMPF_INC_DIR)/xmpf_internal.h xmpf_array.c xmpf_gcomm.c \
                    xmpf_gmove.c xmpf_index.c xmpf_lib.c xmpf_loop.c xmpf_misc.c \
                    xmpf_nodes.c xmpf_pack.c xmpf_reflect.c \
                    xmpf_task.c xmpf_template.c xmpf_coarray.c \
                    xmpf_coarray_put77.c

COMPILE = $(CC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) \
	$(CPPFLAGS) $(AM_CFLAGS)
LINK = $(CCLD) $(AM_CFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@

xmpf_main.o: xmpf_main.f
	@BEFC@ @BEFC_OPTION@ -c xmpf_main.f

install-exec-local: xmpf_main.o
	test -z $(DESTDIR)$(libdir) || /bin/mkdir -p $(DESTDIR)$(libdir)
	$(INSTALL_DATA) xmpf_main.o $(DESTDIR)$(libdir)

clean-local:
	rm -f *~ *.core core core.*
