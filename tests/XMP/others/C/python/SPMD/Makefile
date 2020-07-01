all: test.so
test.so : test.c
	xmpcc -fPIC -shared test.c -o test.so
run:
	mpirun -np 4 python3 ./test.py
clean:
	rm -f *.o *.so *.pyc

