all: lap-C lap-F

lap-C: xmp_laplace.c
	xmpcc xmp_laplace.c -o lap-C

lap-F: xmp_laplace.f90
	xmpf90 xmp_laplace.f90 -o lap-F

clean:
	rm -f lap-C lap-F *.o

run: lap-C lap-F
	mpirun -np 4 ./lap-C
	mpirun -np 4 ./lap-F
