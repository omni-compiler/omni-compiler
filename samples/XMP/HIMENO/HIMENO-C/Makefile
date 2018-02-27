CC = xmpcc
CFLAGS = -O2

ifndef NDX
NDX = 1
endif

ifndef NDY
NDY = 1
endif

all: himeno.c
#	${CC} ${CFLAGS} -o XS himeno.c -DXSMALL -DNDX=$(NDX) -DNDY=$(NDY)
	${CC} ${CFLAGS} -o S  himeno.c -DSMALL -DNDX=$(NDX) -DNDY=$(NDY)
	${CC} ${CFLAGS} -o M  himeno.c -DMIDDLE -DNDX=$(NDX) -DNDY=$(NDY)
#	${CC} ${CFLAGS} -o L  himeno.c -DLARGE -DNDX=$(NDX) -DNDY=$(NDY)

clean:
	rm -f a.out *.o
