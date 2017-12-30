all: test.so
test.so : test.c
	xmpcc -fPIC -shared test.c -o test.so

run: test.so
	mpirun -np 1 python3 ./test.py
	mpirun -np 1 python3 ./test_async.py
	mpirun -np 1 python3 ./test_numpy.py

clean:
	rm -f *.o *.so *.pyc

