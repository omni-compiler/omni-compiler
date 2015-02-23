XFC     = xmpf90 -O2
XRUN    = mpirun
EXES    = exit.x module_test.x module_test0.x arg.x dnodes.x refobj.x 228.x 358_result.x

all: module_test.x exit.x module_test0.x arg.x dnodes.x external.o interface.o xmp_lib.o threadprivate.o refobj.x 228.x 45_save.o 358_result.x 377.x 261.o 162.o 383.x

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

external.o: external.f90
	$(XFC) -c external.f90

main.o: main.f90 module0.o
	$(XFC) -c main.f90 -Jmod

arg.x: arg.o
	$(XFC) -o arg.x arg.o

arg.o: arg.f90
	$(XFC) -c arg.f90

dnodes.x: dnodes.o
	$(XFC) -o dnodes.x dnodes.o

dnodes.o: dnodes.f90
	$(XFC) -c dnodes.f90

interface.o: interface.f90
	$(XFC) -c interface.f90

xmp_lib.o: xmp_lib.f90
	$(XFC) -c xmp_lib.f90

threadprivate.o: threadprivate.f90
	$(XFC) -c -openmp threadprivate.f90

refobj.x: refobj.f90
	$(XFC) -o refobj.x refobj.f90

228.x: 228.f
	$(XFC) -o 228.x 228.f

45_save.o: 45_save.f90
	$(XFC) -c 45_save.f90

358_result.x: 358_result.f90
	$(XFC) -o $@ $<

377.x: 377.f90
	$(XFC) -c -o 377.x $<; \
	[ ! -f 377.x ] && exit 1; \
	rm -rf __omni_tmp__; \
	$(XFC) -c $< --debug 1> /dev/null;\
	[ ! -d __omni_tmp__ ] && exit 1; \
        echo "PASS"

261.o: 261.f90
	$(XFC) -c $<

162.o: 162.f90
	$(XFC) -c $<

383.x: 383.f90
	$(XFC) -o $@ $<

run: $(EXES)
	$(XRUN) -np 2 ./module_test.x
	$(XRUN) -np 2 ./exit.x
	$(XRUN) -np 2 ./module_test0.x
	$(XRUN) -np 4 ./arg.x
	export XMP_NODE_SIZE0=2; \
	export XMP_NODE_SIZE1=2; \
	$(XRUN) -np 8 ./dnodes.x
	$(XRUN) -np 4 ./refobj.x
	$(XRUN) -np 1 ./228.x
	./358_result.x
	$(XRUN) -np 2 ./383.x

clean:
	rm -f $(EXES) *.o *.*mod
	rm -rf mod __omni_tmp__
