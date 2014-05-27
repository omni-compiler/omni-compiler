XFC     = xmpf90 -O2
XRUN    = mpirun
EXES    = exit.x module_test.x module_test0.x

all: module_test2.o module_test1.o module0.o main.o
	$(XFC) -o module_test.x module_test2.o module_test1.o
	$(XFC) -o exit.x exit.f90
	$(XFC) -o module_test0.x module0.o main.o

module_test2.o: module_test2.f90
	$(XFC) -c module_test2.f90

module_test1.o: module_test1.f90
	$(XFC) -c module_test1.f90

module0.o: module0.f90
	mkdir -p mod
	$(XFC) -c module0.f90 -Jmod

main.o: main.f90
	mkdir -p mod
	$(XFC) -c main.f90 -Jmod

run: $(EXES)
	$(XRUN) -np 2 module_test.x
	$(XRUN) -np 2 exit.x

clean:
	rm -f $(EXES) *.o *.*mod
	rm -rf mod
