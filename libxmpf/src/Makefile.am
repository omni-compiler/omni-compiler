CC           = @MPI_CC@
FC           = @MPI_FC@
FCFLAGS      = @MPI_FCFLAGS@
XMP_INC_DIR  =../../libxmp/include
XMPF_INC_DIR =../include
CFLAGS       = -I$(XMPF_INC_DIR) -I$(XMP_INC_DIR) @MPI_CFLAGS@
LDFLAGS      = @MPI_CLIBS@

lib_LIBRARIES     = libxmpf.a
libxmpf_a_SOURCES = $(XMPF_INC_DIR)/xmpf_internal.h xmpf_main.F90 xmpf_array.c xmpf_gcomm.c \
                    xmpf_gmove.c xmpf_index.c xmpf_lib.c xmpf_loop.c xmpf_misc.c \
                    xmpf_nodes.c xmpf_pack.c xmpf_reflect.c xmpf_sort.c xmpf_task.c \
		    xmpf_template.c

if GASNET
    libxmpf_a_SOURCES += xmpf_coarray.c xmpf_coarray_wrap.f90 xmpf_coarray_put.c \
                         xmpf_coarray_get.c xmpf_coarray_get_wrap.f90
    CFLAGS            += -D_XMP_COARRAY_GASNET
    FCFLAGS           += -D_XMP_COARRAY_GASNET
endif

if FJRDMA
    libxmpf_a_SOURCES += xmpf_coarray.c xmpf_coarray_wrap.f90 xmpf_coarray_put.c \
                         xmpf_coarray_get.c xmpf_coarray_get_wrap.f90
    CFLAGS            += -D_XMP_COARRAY_FJRDMA
    FCFLAGS           += -D_XMP_COARRAY_FJRDMA
endif

xmpf_coarray_get_wrap.f90: xmpf_coarray_get_wrap.f90.sh
	. $< > $@

install-exec-local: xmpf_main.o
	test -z $(DESTDIR)$(libdir) || /bin/mkdir -p $(DESTDIR)$(libdir)
	$(INSTALL_DATA) xmpf_main.o $(DESTDIR)$(libdir)

clean-local:
	rm -f *~ *.core core core.* xmpf_coarray_get_wrap.f90

