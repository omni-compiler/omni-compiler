XFC     = xmpf90
XRUN    = mpiexec
TESTS = $(wildcard *.f90)
EXES  = offset.x offset2.x test.x
OBJS  = $(TESTS:.f90=.o)

.PHONY: clean all default run submit showlog cleanlog

all default: $(EXES)

.SUFFIXES: .x .f90

.f90.x:
	$(XFC) -o $@ $<

test.x: main.f90 init.o
	$(XFC) -fbounds-check main.f90 init.o -o $@

init.o: init.f90
	$(XFC) -fbounds-check $< -c

run: $(EXES)
	$(XRUN) -n 2 ./offset.x
	$(XRUN) -n 4 ./offset2.x
	$(XRUN) -n 2 ./test.x

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
	rm -f $(EXES) $(OBJS) *.mod *.xmod

