CC = xmpcc
CFLAGS = -xacc -O2

ifndef NDX
NDX = 1
endif

ifndef NDY
NDY = 1
endif

all: himeno_xacc.c
#	${CC} ${CFLAGS} -o XS himeno_xacc.c -DXSMALL -DNDX=$(NDX) -DNDY=$(NDY)
	${CC} ${CFLAGS} -o S  himeno_xacc.c -DSMALL -DNDX=$(NDX) -DNDY=$(NDY)
	${CC} ${CFLAGS} -o M  himeno_xacc.c -DMIDDLE -DNDX=$(NDX) -DNDY=$(NDY)
#	${CC} ${CFLAGS} -o L  himeno_xacc.c -DLARGE -DNDX=$(NDX) -DNDY=$(NDY)

clean:
	rm -f a.out *.o
