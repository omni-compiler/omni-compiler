include_HEADERS = xmpf_coarray_decl.mod

CC           = @MPI_CC@
FC           = @MPI_FC@
FCFLAGS      = @MPI_FCFLAGS@ @OMNI_NTV_FCFLAGS@
XMP_INC_DIR  =../../libxmp/include
XMPF_INC_DIR =../include
CFLAGS       = -I$(XMPF_INC_DIR) -I$(XMP_INC_DIR) @MPI_CFLAGS@
PPFCCOMPILE = $(FC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) \
              $(AM_CPPFLAGS) $(AM_FCFLAGS) $(FCFLAGS)
LDFLAGS      = @MPI_CLIBS@

lib_LIBRARIES     = libxmpf.a
libxmpf_a_SOURCES = $(XMPF_INC_DIR)/xmpf_internal.h $(XMPF_INC_DIR)/xmpf_internal_coarray.h \
                    xmpf_main.F90 xmpf_array.c xmpf_async.c xmpf_gcomm.c \
                    xmpf_gmove.c xmpf_index.c xmpf_lib.c xmpf_loop.c xmpf_misc.c \
                    xmpf_nodes.c xmpf_pack.c xmpf_reflect.c xmpf_sort.c xmpf_task.c \
		    xmpf_template.c
libxmpf_a_SOURCES += xmpf_coarray.c xmpf_coarray_wrap.f90 \
	            xmpf_coarray_lib.c \
                    xmpf_coarray_put.c xmpf_coarray_put_wrap.f90 \
                    xmpf_coarray_get.c xmpf_coarray_get_wrap.f90 \
                    xmpf_coarray_alloc.c xmpf_coarray_alloc_wrap.f90 \
                    xmpf_coarray_reduction.F90 xmpf_coarray_decl.f90

if GASNET
    CFLAGS            += -D_XMP_GASNET
if POWERPC
    FCFLAGS           += -WF,-D_XMP_GASNET
else
    FCFLAGS           += -D_XMP_GASNET
endif
endif

if MPI3_ONESIDED
    CFLAGS            += -D_XMP_MPI3_ONESIDED
    FCFLAGS           += -D_XMP_MPI3_ONESIDED
endif

if MPI3
    CFLAGS += -D_XMP_MPI3
endif

if FJRDMA
    CFLAGS            += -D_XMP_FJRDMA
    FCFLAGS           += -D_XMP_FJRDMA
endif

if SUPERUX
  COARRAY_SYNC_H = xmp_coarray_sync_sxace.h
else
  COARRAY_SYNC_H = xmp_coarray_sync.h
endif

WORKDIR = ./include_work

%.f90: %.f90.sh
	. $< $(TARGET) > $@

%.F90: %.F90.sh
	. $< $(TARGET) > $@

xmpf_coarray_decl.mod xmpf_coarray_decl.o:  xmpf_coarray_decl.f90
	rm -rf $(WORKDIR)
	mkdir $(WORKDIR)
	cp -p ../include/*.h $(WORKDIR)
	cp -p ../include/$(COARRAY_SYNC_H) $(WORKDIR)/xmp_coarray_sync.h
	$(FC) $(FCFLAGS) -c -I $(WORKDIR) $<

xmpf_main.o: xmpf_main.F90
	$(FC) $(FCFLAGS) $< -c

install-exec-local: xmpf_main.o
	test -z $(DESTDIR)$(libdir) || $(MKDIR_P) $(DESTDIR)$(libdir)
	$(INSTALL_DATA) xmpf_main.o $(DESTDIR)$(libdir)

clean-local:
	rm -f *~ *.core core core.*
	rm -f xmpf_coarray_put_wrap.f90
	rm -f xmpf_coarray_get_wrap.f90
	rm -f xmpf_coarray_alloc_wrap.f90
	rm -f xmpf_coarray_reduction.F90
	rm -rf $(WORKDIR)
	rm -f xmpf_coarray_decl.o  xmpf_coarray_decl.mod
