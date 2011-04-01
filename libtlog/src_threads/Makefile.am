# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
include_HEADERS	=	tlog.h

lib_LIBRARIES	=	libtlog.a

libtlog_a_SOURCES		= tlog-ompc.c tlog.c tlog-time.c

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

