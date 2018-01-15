XCC     = xmpcc
XRUN    = mpiexec
C_TESTS = $(wildcard *.c)
C_EXES  = $(C_TESTS:.c=.x)
C_OBJS  = $(C_TESTS:.c=.o)
EXES    = $(C_EXES)

.PHONY: clean all default run submit showlog cleanlog

all default: $(EXES)

.SUFFIXES: .x

.c.x:
	$(XCC) -o $@ $<

run: 
	$(XRUN) -n 2 ./coarray_put_5dims.x
	$(XRUN) -n 2 ./coarray_put_6dims.x
	$(XRUN) -n 2 ./coarray_put_7dims.x
	$(XRUN) -n 2 ./coarray_get_5dims.x
	$(XRUN) -n 2 ./coarray_get_6dims.x
	$(XRUN) -n 2 ./coarray_get_7dims.x
	$(XRUN) -n 2 ./coarray_scalar_mput.x

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
	rm -f $(EXES) $(C_OBJS)

