XFC     = xmpf90
XRUN    = mpiexec
TESTS = $(wildcard *.f90)
EXES  = $(TESTS:.f90=.x)
OBJS  = $(TESTS:.f90=.o)

.PHONY: clean all default run submit showlog cleanlog

all default: $(EXES)

.SUFFIXES: .x .f90

.f90.x:
	$(XFC) -o $@ $<

run: $(EXES)
	$(XRUN) -n 4 ./plus.x
	$(XRUN) -n 4 ./mult.x
	$(XRUN) -n 4 ./max.x
	$(XRUN) -n 4 ./min.x
	$(XRUN) -n 4 ./raise.x
	$(XRUN) -n 4 ./func.x
	$(XRUN) -n 4 ./reduction_iand.x
	$(XRUN) -n 4 ./reduction_ior.x
	$(XRUN) -n 4 ./reduction_ieor.x
	$(XRUN) -n 8 ./plus2.x
	$(XRUN) -n 8 ./mult2.x
	$(XRUN) -n 8 ./max2.x
	$(XRUN) -n 8 ./min2.x
	$(XRUN) -n 3 ./array.x
	$(XRUN) -n 3 ./allocatable.x

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

