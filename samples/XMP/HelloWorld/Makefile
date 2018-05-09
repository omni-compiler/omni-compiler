all: C_hello F_hello

C_hello: main.c
	xmpcc main.c -o C_hello

F_hello: main.f90
	xmpf90 main.f90 -o F_hello

clean:
	rm -f C_hello F_hello *.o

run: C_hello F_hello
	mpirun -np 2 ./C_hello
	mpirun -np 2 ./F_hello

