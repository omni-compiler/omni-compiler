ifeq ($(arch),K)
CFLAGS=-Kfast
else
CFLAGS=-O2
endif

all: RandomAccess.o
	xmpcc ${CFLAGS} $^ -o RandomAccess -lm

RandomAccess.o: RandomAccess.c
	xmpcc ${CFLAGS} $^ -lm -c

clean:
	rm -f RandomAccess.o
