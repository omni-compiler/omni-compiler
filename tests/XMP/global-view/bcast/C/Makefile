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
	$(XRUN) -n 4 ./bcast.x
	$(XRUN) -n 4 ./bcast_from_p.x
	$(XRUN) -n 4 ./bcast_from_p_on_p.x
	$(XRUN) -n 4 ./bcast_on_p.x
	$(XRUN) -n 4 ./bcast_on_split_p.x
#	$(XRUN) -n 4 ./bcast_from_p_on_t2.x
	$(XRUN) -n 4 ./bcast_on_split_p_from_p.x
	$(XRUN) -n 4 ./bcast_on_t2.x
	$(XRUN) -n 4 ./bcast_array_section.x
	$(XRUN) -n 8 ./bcast2.x
	$(XRUN) -n 8 ./bcast_from_p2.x
	$(XRUN) -n 8 ./bcast_on_p2.x
	$(XRUN) -n 8 ./bcast_on_split_p2.x
	$(XRUN) -n 8 ./bcast_from_p2_on_p2.x
	$(XRUN) -n 16 ./bcast_on_split_p2_from_p2.x
	$(XRUN) -n 2 ./bcast3.x

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

