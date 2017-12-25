all: test.so
test.so: test.c
	xmpcc -fPIC -shared test.c -o test.so

run: test.so
	mpirun -np 1 python ./test.py
	mpirun -np 1 python ./test_async.py
clean:
	rm -f *.so *.o
