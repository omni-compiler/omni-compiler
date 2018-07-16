XCC     = xmpcc
XRUN    = mpiexec
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all default run submit showlog cleanlog

all default: $(EXES)

.SUFFIXES: .x .c

.c.x:
	$(XCC) -o $@ $<

run: $(EXES)
	$(XRUN) -n 2 ./314_loop.x
	$(XRUN) -n 4 ./gblock.x
	$(XRUN) -n 4 ./reduction_and.x
	$(XRUN) -n 4 ./reduction_firstmax.x
	$(XRUN) -n 4 ./reduction_firstmin.x
	$(XRUN) -n 4 ./reduction_lastmax.x
	$(XRUN) -n 4 ./reduction_lastmin.x
	$(XRUN) -n 4 ./reduction_max.x
	$(XRUN) -n 4 ./reduction_min.x
	$(XRUN) -n 4 ./reduction_mult.x
	$(XRUN) -n 4 ./reduction_or.x
	$(XRUN) -n 4 ./reduction_plus.x
	$(XRUN) -n 4 ./reduction_logical_and.x
	$(XRUN) -n 4 ./reduction_logical_or.x
	$(XRUN) -n 4 ./reduction_raise.x
	$(XRUN) -n 8 ./reduction_and2.x
	$(XRUN) -n 16 ./reduction_firstmax2.x
	$(XRUN) -n 16 ./reduction_firstmin2.x
	$(XRUN) -n 8 ./reduction_lastmax2.x
	$(XRUN) -n 16 ./reduction_lastmin2.x
	$(XRUN) -n 8 ./reduction_raise2.x
	$(XRUN) -n 8 ./reduction_max2.x
	$(XRUN) -n 8 ./reduction_min2.x
	$(XRUN) -n 8 ./reduction_or2.x
	$(XRUN) -n 8 ./reduction_plus.x
	$(XRUN) -n 16 ./reduction_plus3.x
	$(XRUN) -n 2 ./different_template.x
	$(XRUN) -n 2 ./158_idx.x
	$(XRUN) -n 2 ./105_loop.x
	$(XRUN) -n 4 ./expand.x
	$(XRUN) -n 4 ./margin.x
	$(XRUN) -n 4 ./expand_margin.x
	$(XRUN) -n 4 ./peel_and_wait.x

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

