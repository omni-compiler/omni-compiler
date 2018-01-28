XCC     = xmpcc
XRUN    = mpiexec
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all run submit showlog cleanlog

all: $(EXES)

.SUFFIXES: .x .c

.c.x:
	$(XCC) -o $@ $<

run:
	$(XRUN) -n 4 ./shadow_reflect.x
	$(XRUN) -n 8 ./ordinal_reflect.x
	$(XRUN) -n 4 ./full2.x
	$(XRUN) -n 4 ./full3.x
	$(XRUN) -n 4 ./loop.x
	$(XRUN) -n 2 ./basic.x
	$(XRUN) -n 4 ./shadow_check.x
	$(XRUN) -n 16 ./reduce_shadow.x
	$(XRUN) -n 4 ./separated_nested_loop.x

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

