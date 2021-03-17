OMNI_HOME = $(shell xmpcc --show-env | grep OMNI_HOME | sed -e 's/OMNI_HOME=//' -e 's/"//g')
OMNI_FLAGS = $(shell xmpcc --show-env | grep OMNI_CPP_MUST_OPT | sed -e 's/OMNI_CPP_MUST_OPT=//' -e 's/"//g')
OMNI_LIBS = $(shell xmpcc --show-env | grep OMNI_LINKER_OPT | sed -e 's/OMNI_LINKER_OPT=//' -e 's/"//g')

CC = mpicc
CFLAGS = -g $(OMNI_FLAGS)

ifndef NDX
NDX = 1
endif

ifndef NDY
NDY = 1
endif

all: himeno.c
#	${CC} ${CFLAGS} -o XS xmp_api_himeno.c -DXSMALL -DNDX=$(NDX) -DNDY=$(NDY) $(OMNI_LIBS)
	${CC} ${CFLAGS} -o S  xmp_api_himeno.c -DSMALL -DNDX=$(NDX) -DNDY=$(NDY) $(OMNI_LIBS)
	${CC} ${CFLAGS} -o M  xmp_api_himeno.c -DMIDDLE -DNDX=$(NDX) -DNDY=$(NDY) $(OMNI_LIBS)
#	${CC} ${CFLAGS} -o L  xmp_api_himeno.c -DLARGE -DNDX=$(NDX) -DNDY=$(NDY) $(OMNI_LIBS)

clean:
	rm -f XS S M L *.o
