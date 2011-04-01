# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
include_HEADERS	=	tlog-mpi.h tlog_event.h

lib_LIBRARIES	=	libtlog-mpi.a
sri_TLOGVIEWS	=	tlogview
jar_TLOGVIEWS	=	tlogview.jar

libtlog_a_SOURCES = tlog.c tlog_conv.c tlog_event.c tlog_misc.c tlog-time.c tlog_dump.c tlog_event.h

all-local:	$(lib_LIBRARIES)

clean-local:
	rm -f $(TARGET) $(RUNTIME) *.o *~ core

distclean:	clean-local
	rm -f config.* Makefile .depend*

install-exec-local:	$(lib_LIBRARIES)
	test -z "$(libdir)" || $(MKDIR_P) $(DESTDIR)$(libdir)
	$(INSTALL_PROGRAM) ${tmp_file} $(DESTDIR)$(bindir)
	$(INSTALL_PROGRAM) $(jar_TLOGVIEWS) $(DESTDIR)$(bindir)

html-local:
	if test ! -e "../doc"; then mkdir ../doc; fi
	doxygen >/dev/null

