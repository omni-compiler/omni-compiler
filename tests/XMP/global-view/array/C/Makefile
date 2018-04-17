XCC     = xmpcc
XRUN    = mpiexec
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all default run submit showlog cleanlog

all default: $(EXES)

.SUFFIXES: .x .c

array.x: array.c
	$(XCC) -o $@ $< -lm

.c.x:
	$(XCC) -o $@ $<

run: $(EXES)
	$(XRUN) -n 4 ./array.x
	$(XRUN) -n 4 ./array2.x
	$(XRUN) -n 2 ./316_array.x
	$(XRUN) -n 4 ./361_array.x
	$(XRUN) -n 4 ./417_array1.x
	$(XRUN) -n 4 ./417_array2.x
	$(XRUN) -n 1 ./516.x
	$(XRUN) -n 4 ./array3.x

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

