CC     = xmpcc
CFLAGS = -xacc -O2

SIZE ?= S
NDX  ?= 1
NDY  ?= 1

PROG_PUT = himeno_put_$(SIZE)_$(NDX)x$(NDY)
PROG_GET = himeno_get_$(SIZE)_$(NDX)x$(NDY)

PROG = $(PROG_PUT) $(PROG_GET)

all: $(PROG)

$(PROG_PUT): himeno.c
	./paramset.sh $(SIZE) $(NDX) $(NDY) 1
	${CC} ${CFLAGS} -o $@ $^ -DUSE_PUT

$(PROG_GET): himeno.c
	./paramset.sh $(SIZE) $(NDX) $(NDY) 1
	${CC} ${CFLAGS} -o $@ $^ -DUSE_GET

clean:
	rm -f a.out *.o param.h
