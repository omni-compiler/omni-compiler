XFC     = xmpf90 -O2
XRUN    = mpirun
EXES    = exit module_test module_test0

all: module_test2.o module_test1.o module0.o main.o
	$(XFC) -o module_test module_test2.o module_test1.o
	$(XFC) -o exit exit.f90
	$(XFC) -o module_test0 module0.o main.o

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
	$(XRUN) -np 2 module_test
	$(XRUN) -np 2 exit

clean:
	rm -f $(EXES) *.o *.*mod
	rm -rf mod
