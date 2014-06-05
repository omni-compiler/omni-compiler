XFC     = xmpf90 -O2
XRUN    = mpirun
EXES    = exit.x module_test.x module_test0.x

all: module_test.x exit.x module_test0.x

module_test0.x: module0.f90 main.f90
	rm -rf mod
	mkdir -p mod
	$(XFC) -c module0.f90 -Jmod
	$(XFC) -c main.f90 -Jmod
	$(XFC) -o module_test0.x module0.o main.o

exit.x: exit.o
	$(XFC) -o exit.x exit.o

exit.o: exit.f90
	$(XFC) -c exit.f90

module_test1.o: module_test2.o module_test1.f90
	$(XFC) -c module_test1.f90

module_test.x: module_test2.o module_test1.o
	$(XFC) -o module_test.x module_test2.o module_test1.o

module_test2.o: module_test2.f90
	$(XFC) -c module_test2.f90

run: $(EXES)
	$(XRUN) -np 2 module_test.x
	$(XRUN) -np 2 exit.x
	$(XRUN) -np 2 ./module_test0.x

clean:
	rm -f $(EXES) *.o *.*mod
	rm -rf mod
