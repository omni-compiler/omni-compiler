XFC     = xmpf90
XRUN    = mpiexec

.PHONY: clean all default run

all: 476.x

476-1.o: 476-1.f90
	xmpf90 476-1.f90 -c

476-2.o: 476-2.f90
	xmpf90 476-2.f90 -c

libb.a: 476-2.o
	ar r libb.a 476-2.o; ranlib libb.a

476.x: 476-1.o libb.a
	$(XFC) 476-1.o libb.a -o 476.x

run: $(EXES)
	$(XRUN) -n 1 ./476.x

clean:
	rm -f *.o 476.x libb.a

