XCC     = xmpcc
XRUN    = mpiexec
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all default run submit showlog cleanlog

all default: $(EXES)

.SUFFIXES: .x

.c.x:
	$(XCC) -o $@ $<

run: 
	$(XRUN) -n 1 ./coarray_translation.x
	$(XRUN) -n 2 ./coarray_scalar.x
	$(XRUN) -n 2 ./coarray_vector.x
	$(XRUN) -n 2 ./coarray_stride.x
	$(XRUN) -n 2 ./coarray_scalar_mput.x
	$(XRUN) -n 2 ./coarray_scalar_mget.x
	$(XRUN) -n 2 ./coarray_scalar_mget_2.x
	$(XRUN) -n 2 ./coarray_put_1dim.x
	$(XRUN) -n 2 ./coarray_put_2dims.x
	$(XRUN) -n 2 ./coarray_put_3dims.x
	$(XRUN) -n 2 ./coarray_put_4dims.x
	$(XRUN) -n 2 ./coarray_get_1dim.x
	$(XRUN) -n 2 ./coarray_get_2dims.x
	$(XRUN) -n 2 ./coarray_get_3dims.x
	$(XRUN) -n 2 ./coarray_get_4dims.x
	$(XRUN) -n 1 ./coarray_local_put.x
	$(XRUN) -n 1 ./coarray_local_get.x

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

