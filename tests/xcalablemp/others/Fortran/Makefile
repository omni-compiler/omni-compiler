XFC     = xmpf90 -O2
XRUN    = mpirun

all: 
	$(XFC) -o module_test module_test2.f90 module_test1.f90

run: $(EXES)
	$(XRUN) -np 2 module_test

clean:
	rm -f $(EXES) *.o *mod
