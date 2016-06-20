XFC     = xmpf90
XRUN    = mpiexec
OBJS    = *.o libb.a
EXES    = 476.x

.PHONY: clean all default run submit showlog cleanlog

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

RUN:
	mkdir RUN

RUN/%.x:: %.x
	cp $< $@

RUN/go.sh: go.template Makefile
	cp $< $@ && grep XRUN Makefile | sed -e 's/(XRUN)/{XRUN}/' -e 's/ *= */=/' | grep -v Makefile >>$@

submit: all RUN RUN/go.sh $(EXES:%=RUN/%)
	cd RUN; pjsub go.sh

showlog:
	cat RUN/go.sh.e* RUN/go.sh.o*

cleanlog:
	rm -rf RUN

clean: cleanlog
	rm -f $(EXES) $(OBJS)

