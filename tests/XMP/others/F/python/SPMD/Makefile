all: test.so
test.so : test.f90
	xmpf90 -fPIC -shared test.f90 -o test.so
run:
	mpirun -np 4 python3 ./test.py
clean:
	rm -f *.o *.so *.pyc

