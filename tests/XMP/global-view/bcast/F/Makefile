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
	$(XRUN) -n 4 ./bcast.x
	$(XRUN) -n 4 ./bcast2.x
	$(XRUN) -n 4 ./bcast3.x
	$(XRUN) -n 4 ./bcast4.x
	$(XRUN) -n 8 ./bcast5.x
	$(XRUN) -n 4 ./bcast6.x
	$(XRUN) -n 4 ./bcast7.x
	$(XRUN) -n 4 ./bcast8.x
	$(XRUN) -n 8 ./bcast9.x
	$(XRUN) -n 8 ./bcast10.x
	$(XRUN) -n 8 ./bcast11.x
	$(XRUN) -n 8 ./bcast12.x

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

