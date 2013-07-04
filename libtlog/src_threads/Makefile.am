# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $

CC                = @BECC@
include_HEADERS	  = tlog.h
lib_LIBRARIES	  = libtlog.a
libtlog_a_SOURCES = tlog-ompc.c tlog.c tlog-time.c

AM_CFLAGS     = -I$(XMP_INC_DIR) -I$(TLOG_INC_DIR) @BECC_OPTION@ 
COMPILE = $(CC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) \
	$(CPPFLAGS) $(AM_CFLAGS)
LINK = $(CCLD) $(AM_CFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@

all-local:	$(lib_LIBRARIES)

clean-local:
	rm -f $(TARGET) $(RUNTIME) *.o *~ core

distclean:	clean-local
	rm -f config.* Makefile .depend*

install-exec-local:	$(lib_LIBRARIES)
	test -z "$(libdir)" || $(MKDIR_P) $(DESTDIR)$(libdir)
	$(INSTALL_DATA) $(lib_LIBRARIES) $(DESTDIR)$(libdir)

html-local:
	if test ! -e "../doc"; then mkdir ../doc; fi
	doxygen >/dev/null

