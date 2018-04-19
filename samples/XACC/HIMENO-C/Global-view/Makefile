CC     = xmpcc
CFLAGS = -xacc -O2

SIZE ?= S
NDX  ?= 1
NDY  ?= 1

PROG = himeno_$(SIZE)_$(NDX)x$(NDY)

all: $(PROG)

$(PROG): himeno.c param.h
	${CC} ${CFLAGS} -o $@ himeno.c -DSIZE_$(SIZE) -DNDX=$(NDX) -DNDY=$(NDY) -DNDZ=1
#	${CC} ${CFLAGS} -o $(SIZE)  himeno.c -DSIZE_$(SIZE) -DNDX=$(NDX) -DNDY=$(NDY) -DNDZ=1

clean:
	rm -f a.out *.o
