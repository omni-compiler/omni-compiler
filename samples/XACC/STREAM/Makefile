all: STREAM.c
	xmpcc -xacc --openmp STREAM.c -o STREAM -O3 -xAVX -opt-streaming-stores always

clean:
	rm -f *.o *~
