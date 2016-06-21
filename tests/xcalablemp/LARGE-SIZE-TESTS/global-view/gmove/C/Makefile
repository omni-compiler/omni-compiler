XCC     = xmpcc
XRUN    = mpiexec
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all default run submit showlog cleanlog

all default: $(EXES)

.SUFFIXES: .x .c

.c.x:
	$(XCC) -o $@ $< ./chk/chk.c

run: $(EXES)
	$(XRUN) -n 32 ./gmove_bca_5a5t_b.x
	$(XRUN) -n 32 ./gmove_bca_5a5t_b2.x
	$(XRUN) -n 64 ./gmove_bca_6a6t_b.x
	$(XRUN) -n 64 ./gmove_bca_6a6t_b2.x
	$(XRUN) -n 128 ./gmove_bca_7a7t_b.x
	$(XRUN) -n 128 ./gmove_bca_7a7t_b2.x
	$(XRUN) -n 32 ./gmove_cp_5a5t_b_bc.x
	$(XRUN) -n 64 ./gmove_cp_6a6t_b_bc.x
	$(XRUN) -n 128 ./gmove_cp_7a7t_b_bc.x
	$(XRUN) -n 32 ./gmove_G2L_5a5t_bc.x
	$(XRUN) -n 64 ./gmove_G2L_6a6t_bc.x
	$(XRUN) -n 128 ./gmove_G2L_7a7t_bc.x
	$(XRUN) -n 32 ./gmove_G2L_1a5t_bc.x
	$(XRUN) -n 64 ./gmove_G2L_1a6t_bc.x
	$(XRUN) -n 128 ./gmove_G2L_1a7t_bc.x

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
	rm -f $(EXES) $(OBJS) chk.o
