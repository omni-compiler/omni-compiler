TOPDIR		= ../..
prefix		= @prefix@
CFLAGS		= @MPI_CFLAGS@ -I. -I$(TOPDIR)/include @DEFS@ @CPPFLAGS@
CC		= @MPI_CC@
AR		= @AR@ cr
RANLIB		= @RANLIB@

SRCS		= ompc_thread.c ompc_lock.c ompc_lib.c ompc_runtime.c timer.c ompf_runtime.c ompf_thread.c
OMPCRTOBJ	= ompc_main.o
OMPCRTOBJ_SRC	= ompc_main.c
OMPFRTOBJ	= ompf_main.o
OMPFRTOBJ_SRC	= ompf_main.F90

DEFAULT_LIB_NAME = libompc.a

include_HEADERS	= ../../include/omp.h

lib_LIBRARIES	= $(LIB_TARGET)

all-local:	$(OMPCRTOBJ) $(OMPFRTOBJ) doAllLib

clean:
	rm -f *.o *~ core $(LIB_TARGET) *._*_o

clean-local:	clean
	rm -f config.* Makefile .depend* ./target.tmpl

@INCLUDECOM@ ./target.tmpl

$(OMPCRTOBJ): $(OMPCRTOBJ_SRC)
	$(MPI_CC) $(MPI_CFLAGS) -c $<

$(OMPFRTOBJ): $(OMPFRTOBJ_SRC)
	$(MPI_FC) $(MPI_FCFLAGS) -c $<

install-exec-local:	$(doAllLib)
	test -z "$(libdir)" || $(MKDIR_P) $(DESTDIR)$(libdir)
	$(INSTALL_DATA) $(OMPCRTOBJ) $(DESTDIR)$(libdir)
	$(INSTALL_DATA) $(OMPFRTOBJ) $(DESTDIR)$(libdir)
	(cd $(DESTDIR)$(libdir); rm -f $(DEFAULT_LIB_NAME); ln -s $(DEFAULT_LIB) $(DEFAULT_LIB_NAME))

html-local:
	if test ! -e "../doc"; then $(MKDIR_P) ../doc; fi
	doxygen >/dev/null

