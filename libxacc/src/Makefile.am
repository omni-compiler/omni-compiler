CC            = @BECC@
XMP_INC_DIR   = ../../libxmp/include
XACC_INC_DIR  = ../include
TLOG_INC_DIR  = $(top_builddir)/@LIBTLOG_MPI_DIR@
lib_LIBRARIES = libxacc.a
AM_CFLAGS     = -I$(XMP_INC_DIR) -I$(XACC_INC_DIR) -I$(TLOG_INC_DIR) @BECC_OPTION@ 
COMPILE = $(CC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) \
	$(CPPFLAGS) $(AM_CFLAGS)
LINK = $(CCLD) $(AM_CFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@

libxacc_a_SOURCES = $(XACC_INC_DIR)/xacc_internal.h \
	xacc.c
#                   xacc_device_stack.c
#					$(XMP_INC_DIR)/xmp_internal.h

clean-local:
	rm -f *~ *.core core core.*
