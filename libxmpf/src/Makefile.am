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

MOD1 = xmp_coarray_sync.mod xmp_coarray_alloc.mod xmp_coarray_get.mod
XMOD1 = xmp_coarray_sync.xmod xmp_coarray_alloc.xmod xmp_coarray_get.xmod

if GASNET
    libxmpf_a_SOURCES += xmp_coarray_alloc.sh xmp_coarray_get.sh \
                         xmp_coarray_sync.f90 xmp_coarray.f90 \
                         xmpf_coarray.c xmpf_coarray_lib.c xmpf_coarray_alloc.c \
                         xmpf_coarray_put.c xmpf_coarray_get.c
    CFLAGS            += -D_XMP_COARRAY_GASNET
    include_HEADERS    = xmp_coarray.mod $(MOD1) xmp_coarray.xmod $(XMOD1)
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
    include_HEADERS    = xmp_coarray.mod $(MOD1) xmp_coarray.xmod $(XMOD1)
endif

FFRONT=../../F-FrontEnd/src/F_Front -fxmp

%.o: %.f90
	$(FC) $(FCFLAGS) -c $<

%.mod: %.f90
	$(FC) $(FCFLAGS) -c $<

%.o: %.sh
	. $< > $*.f90
	$(FC) $(FCFLAGS) -c $*.f90

%.mod: %.sh
	. $< > $*.f90
	$(FC) $(FCFLAGS) -c $*.f90

%.xmod: %.f90
	$(FFRONT) $< >/dev/null

%.xmod: %.sh
	. $< -xmp > $*4xmod.f90
	$(FFRONT) $*4xmod.f90 >/dev/null

xmp_coarray.o: $(MOD1)

xmp_coarray.xmod: $(XMOD1)

install-exec-local: xmpf_main.o
	test -z $(DESTDIR)$(libdir) || /bin/mkdir -p $(DESTDIR)$(libdir)
	$(INSTALL_DATA) xmpf_main.o $(DESTDIR)$(libdir)

clean-local:
	rm -f *~ *.core core core.*
	rm -f *.mod *.xmod
	rm -f xmp_coarray_alloc.f90 xmp_coarray_get.f90
	rm -f xmp_coarray_alloc4xmod.f90 xmp_coarray_get4xmod.f90

