XFC     = xmpf90 -O2
XRUN    = mpirun
EXES    = exit.x module_test.x module_test0.x arg.x

all: module_test.x exit.x module_test0.x arg.x

module_test0.x: module0.o main.o
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

module0.o: module0.f90
	rm -rf mod
	mkdir -p mod
	$(XFC) -c module0.f90 -Jmod

main.o: main.f90 module0.o
	$(XFC) -c main.f90 -Jmod

arg.x: arg.o
	$(XFC) -o arg.x arg.o

arg.o: arg.f90
	$(XFC) -c arg.f90

run: $(EXES)
	$(XRUN) -np 2 ./module_test.x
	$(XRUN) -np 2 ./exit.x
	$(XRUN) -np 2 ./module_test0.x
	$(XRUN) -np 4 ./arg.x

clean:
	rm -f $(EXES) *.o *.*mod
	rm -rf mod
