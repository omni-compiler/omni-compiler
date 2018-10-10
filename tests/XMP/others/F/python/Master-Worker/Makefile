all: test.so
test.so : test.f90
	xmpf90 -fPIC -shared test.f90 -o test.so

run: test.so
	mpirun -np 1 python3 ./test.py
	mpirun -np 1 python3 ./test_async.py
	mpirun -np 1 python3 ./test_numpy.py

clean:
	rm -f *.o *.so *.pyc

