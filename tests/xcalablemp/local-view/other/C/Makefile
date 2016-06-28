XCC     = xmpcc
XRUN    = mpiexec
OBJS    = extern.o
EXES    = extern.x

.PHONY: clean all default run submit showlog cleanlog

all: extern.x

extern.x: extern_a.c extern_b.c
	$(XCC) extern_a.c extern_b.c -o extern.x

run:
	$(XRUN) -n 2 ./extern.x

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

