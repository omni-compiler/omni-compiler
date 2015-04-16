CC           = @MPI_CC@
FC           = @MPI_FC@
FCFLAGS      = @MPI_FCFLAGS@ @OMNI_NTV_FCFLAGS@
XMP_INC_DIR  =../../libxmp/include
XMPF_INC_DIR =../include
CFLAGS       = -I$(XMPF_INC_DIR) -I$(XMP_INC_DIR) @MPI_CFLAGS@
LDFLAGS      = @MPI_CLIBS@

lib_LIBRARIES     = libxmpf.a
libxmpf_a_SOURCES = $(XMPF_INC_DIR)/xmpf_internal.h xmpf_main.F90 xmpf_array.c xmpf_async.c xmpf_gcomm.c \
                    xmpf_gmove.c xmpf_index.c xmpf_lib.c xmpf_loop.c xmpf_misc.c \
                    xmpf_nodes.c xmpf_pack.c xmpf_reflect.c xmpf_sort.c xmpf_task.c \
		    xmpf_template.c

if GASNET
    libxmpf_a_SOURCES += xmp_coarray_alloc.sh xmp_coarray_get.sh \
                         xmp_coarray_sync.f90 xmp_coarray.f90 \
                         xmpf_coarray.c xmpf_coarray_lib.c xmpf_coarray_alloc.c \
                         xmpf_coarray_put.c xmpf_coarray_get.c
    CFLAGS            += -D_XMP_COARRAY_GASNET
    include_HEADERS    = xmp_coarray.mod xmp_coarray_sync.mod xmp_coarray_alloc.mod xmp_coarray_get.mod

if POWERPC
    FCFLAGS           += -WF,-D_XMP_GASNET
else
    FCFLAGS           += -D_XMP_GASNET
endif
endif

if MPI3
    CFLAGS += -D_XMP_MPI3
endif

if FJRDMA
    libxmpf_a_SOURCES += xmp_coarray_alloc.sh xmp_coarray_get.sh \
                         xmp_coarray_sync.f90 xmp_coarray.f90 \
                         xmpf_coarray.c xmpf_coarray_lib.c xmpf_coarray_alloc.c \
                         xmpf_coarray_put.c xmpf_coarray_get.c
    CFLAGS            += -D_XMP_COARRAY_FJRDMA
    FCFLAGS           += -D_XMP_COARRAY_FJRDMA
    include_HEADERS    = xmp_coarray.mod xmp_coarray_sync.mod xmp_coarray_alloc.mod xmp_coarray_get.mod
endif

%.o: %.f90
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.sh
	. $< > $*.f90
	$(FC) $(FCFLAGS) -c $*.f90 -o $@

xmp_coarray.o: xmp_coarray_alloc.o xmp_coarray_get.o xmp_coarray_sync.o


install-exec-local: xmpf_main.o
	test -z $(DESTDIR)$(libdir) || /bin/mkdir -p $(DESTDIR)$(libdir)
	$(INSTALL_DATA) xmpf_main.o $(DESTDIR)$(libdir)

clean-local:
	rm -f *~ *.core core core.*
	rm -f xmp_coarray_alloc.f90 xmp_coarray_get.f90

