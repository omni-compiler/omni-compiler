ONESIDED_LIB=$(shell xmpf90 --show-env | sed -n '/ONESIDED_LIB=/p' | sed 's/ONESIDED_LIB=//' | sed "s/\"//g")
XCC     = xmpcc
XRUN    = mpiexec
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all default run submit showlog cleanlog

all default: $(EXES)

.SUFFIXES: .x .c

CFLAG =
ifeq ($(ONESIDED_LIB),MPI3)
 CFLAG = -D_MPI3
endif

.c.x:
	$(XCC) $(CFLAG) -o $@ $<

run: $(EXES)
	$(XRUN) -n 4 ./barrier.x
	$(XRUN) -n 4 ./reduction.x
	$(XRUN) -n 8 ./tasks.x
	$(XRUN) -n 16 ./nocomm.x
	$(XRUN) -n 3 ./gmove_in.x

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

