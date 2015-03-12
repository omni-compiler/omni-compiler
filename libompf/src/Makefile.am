FC		= $(top_srcdir)/F-FrontEnd/src/F_Front
FCFLAGS		=

SRCS		= omp_lib_kind.f90 omp_lib.f90

include_HEADERS	= ../../include/omp_lib.h \
                  ../../include/omp_lib_kinds.h \
		  omp_lib.xmod \
                  omp_lib_kinds.xmod

omp_lib.f90: omp_lib_kinds.xmod

clean-local:
	rm -f *.xmod *.xmodf

SUFFIXES = .f90 .xmod

.f90.xmod:
	$(FC) $(FCFLAGS) $< > /dev/null
