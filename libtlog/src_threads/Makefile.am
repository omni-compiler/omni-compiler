CC                = @MPI_CC@
include_HEADERS	  = tlog.h
lib_LIBRARIES	  = libtlog.a
libtlog_a_SOURCES = tlog-ompc.c tlog.c tlog-time.c

AM_CFLAGS     = @MPI_CFLAGS@ 
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
	if test ! -e "../doc"; then $(MKDIR_P) ../doc; fi
	doxygen >/dev/null

