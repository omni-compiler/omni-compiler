OMNI_HOME = $(shell xmpcc --show-env | grep OMNI_HOME | sed -e 's/OMNI_HOME=//' -e 's/"//g')
FFLAGS=-I$(OMNI_HOME)/include -J$(OMNI_HOME)/include
LIBS = $(shell xmpf90 --show-env | grep OMNI_LINKER_OPT | sed -e 's/OMNI_LINKER_OPT=//' -e 's/"//g')

OBJS = xmp_api_himeno.o
FC = mpif90 
PROG = xmp_api_himeno$(NODES)

ifeq ($(arch),Fugaku)
FFLAGS += -Kfast -Qt -omp
else
FFLAGS += -fopenmp -g -O2
endif

LDFLAGS = ${FFLAGS}

all : ${PROG}

${PROG} : ${OBJS}
	${FC} -o $@ ${OBJS} ${LDFLAGS} ${LIBS} -lxmpf

${OBJS} : xmp_api_himeno.f90
	${FC} -c ${FFLAGS} $<

clean :
	rm -f ${PROG} ${OBJS} core *~
