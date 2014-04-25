XFC     = xmpf90 -O2
XRUN    = mpirun
EXES    = exit module_test

all: module_test2.o module_test1.o
	$(XFC) -o module_test module_test2.o module_test1.o
	$(XFC) -o exit exit.f90

module_test2.o: module_test2.f90
	$(XFC) -c module_test2.f90

module_test1.o: module_test1.f90
	$(XFC) -c module_test1.f90

run: $(EXES)
	$(XRUN) -np 2 module_test
	$(XRUN) -np 2 exit

clean:
	rm -f $(EXES) *.o *mod
