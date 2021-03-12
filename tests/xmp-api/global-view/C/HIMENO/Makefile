CC = mpicc
CFLAGS = -O2

ifndef NDX
NDX = 1
endif

ifndef NDY
NDY = 1
endif

all: himeno.c
#	${CC} ${CFLAGS} -o XS xmp_api_himeno.c -DXSMALL -DNDX=$(NDX) -DNDY=$(NDY)
	${CC} ${CFLAGS} -o S  xmp_api_himeno.c -DSMALL -DNDX=$(NDX) -DNDY=$(NDY)
	${CC} ${CFLAGS} -o M  xmp_api_himeno.c -DMIDDLE -DNDX=$(NDX) -DNDY=$(NDY)
#	${CC} ${CFLAGS} -o L  xmp_api_himeno.c -DLARGE -DNDX=$(NDX) -DNDY=$(NDY)

clean:
	rm -f XS S M L *.o
