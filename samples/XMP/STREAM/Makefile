ifeq ($(compiler),K)
CFLAGS=-Kfast,ocl,openmp -D_FUJITSU
else ifeq ($(compiler),intel)
CFLAGS=--openmp -O2 -mcmodel=medium
else
CFLAGS=--openmp -O2 -mcmodel=medium
endif

all: STREAM-static STREAM-dynamic

STREAM-static: STREAM-static.o
	xmpcc $(CFLAGS) STREAM-static.o -o STREAM-static

STREAM-dynamic: STREAM-dynamic.o
	xmpcc $(CFLAGS) STREAM-dynamic.o -o STREAM-dynamic

STREAM-static.o: STREAM-static.c
	xmpcc $(CFLAGS) -c STREAM-static.c

STREAM-dynamic.o: STREAM-dynamic.c
	xmpcc $(CFLAGS) -c STREAM-dynamic.c

clean:
	rm -rf *.o
