CC     = ompcc
CFLAGS = -acc -O2

SIZE ?= S
NDX  ?= 1
NDY  ?= 1

PROG = himeno_$(SIZE)_$(NDX)x$(NDY)

all: $(PROG)

$(PROG): himeno.c
	./paramset.sh $(SIZE) $(NDX) $(NDY) 1
	${CC} ${CFLAGS} -o $@ himeno.c
#	${CC} ${CFLAGS} -o $(SIZE) himeno.c

clean:
	rm -f a.out *.o
