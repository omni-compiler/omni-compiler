OBJS = himeno.o
FC = xmpf90 
PROG = himeno$(NODES)

ifeq ($(arch),K)
FFLAGS = -Kfast -Qt -omp
else
FFLAGS = -omp -O2
endif

LDFLAGS = ${FFLAGS}

all : ${PROG}

${PROG} : ${OBJS}
	${FC} -o $@ ${OBJS} ${LDFLAGS}

${OBJS} : himeno.f90
	${FC} -c ${FFLAGS} $<

clean :
	rm -f ${PROG} ${OBJS} core *~
