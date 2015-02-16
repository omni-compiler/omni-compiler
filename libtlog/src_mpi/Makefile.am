CC              = @MPI_CC@
include_HEADERS	= tlog_mpi.h tlog_event.h
lib_LIBRARIES	= libtlog_mpi.a
sri_TLOGVIEWS	= ./bin/tlogview
jar_TLOGVIEWS	= ./bin/tlogview.jar
libtlog_mpi_a_SOURCES = tlog.c tlog_event.c tlog_misc.c tlog-time.c tlog_event.h
CFLAGS = @MPI_CFLAGS@

all-local: $(lib_LIBRARIES)

clean-local:
	rm -f $(TARGET) $(RUNTIME) *.o *~ core

distclean: clean-local
	rm -f config.* Makefile .depend*

install-exec-local:	$(lib_LIBRARIES)
	test -z "$(libdir)" || $(MKDIR_P) $(DESTDIR)$(libdir)
	$(INSTALL_PROGRAM) $(sri_TLOGVIEWS) $(DESTDIR)$(bindir)
	$(INSTALL_PROGRAM) $(jar_TLOGVIEWS) $(DESTDIR)$(libdir)

