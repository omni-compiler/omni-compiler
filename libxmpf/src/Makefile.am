XMP_INC_DIR=../../libxmp/include
XMP_SRC_DIR=../../libxmp/src
XMPF_INC_DIR=../include
#TLOG_INC_DIR=$(top_builddir)/@LIBTLOG_MPI_DIR@
#GASNET_INC_DIR=@GASNET_PREFIX@/include
#GASNET_CONDUIT_DIR=$(GASNET_INC_DIR)/@GASNET_CONDUIT@-conduit
CC=@MPI_CMD@

lib_LIBRARIES		= libxmpf.a
libxmpf_a_SOURCES	= $(XMPF_INC_DIR)/xmpf_internal.h \
		xmpf_array.c  xmpf_gcomm.c  xmpf_gmove.c  xmpf_index.c  \
		xmpf_lib.c  xmpf_loop.c  xmpf_misc.c  xmpf_nodes.c  \
		xmpf_pack.c xmpf_template.c
AM_CFLAGS   		= -Wall -Wpointer-arith -std=c99 -I$(XMPF_INC_DIR) -I$(XMP_INC_DIR) -I$(XMP_SRC_DIR) -I$(TLOG_INC_DIR)

xmpf_main.o: xmpf_main.f
	@MPI_F90@ -c xmpf_main.f

install-exec-local: xmpf_main.o
	test -z $(libdir) || /bin/mkdir -p $(libdir)
	$(INSTALL_DATA) xmpf_main.o $(libdir)

clean-local:
	rm -f *~ *.core core core.*
