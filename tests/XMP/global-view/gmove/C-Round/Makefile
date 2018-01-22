ONESIDED_LIB=$(shell xmpf90 --show-env | sed -n '/ONESIDED_LIB=/p' | sed 's/ONESIDED_LIB=//' | sed "s/\"//g")

XCC     = xmpcc
XRUN    = mpiexec
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all default run submit showlog cleanlog

all default: $(EXES)

.SUFFIXES: .x .c

chk.o: ./chk/chk.c
	$(XCC) -c $^ -o $@

CFLAG =
ifeq ($(ONESIDED_LIB),MPI3)
	CFLAG = -D_MPI3
else
	CFLAG = 
endif

%.x: %.c chk.o
	  $(XCC) ${CFLAG} -o $@ $< ./chk.o

run: $(EXES)
	$(XRUN) -n 2 ./gmove_lc_2.x
	$(XRUN) -n 4 ./gmove_lc_4.x
	$(XRUN) -n 8 ./gmove_lc_8.x
	$(XRUN) -n 2 ./gmove_ata_2.x
	$(XRUN) -n 4 ./gmove_ata_4.x
	$(XRUN) -n 2 ./gmove_bca_2.x
	$(XRUN) -n 4 ./gmove_bca_4.x
	$(XRUN) -n 8 ./gmove_bca_8.x
	$(XRUN) -n 12 ./gmove_bca_12.x
	$(XRUN) -n 16 ./gmove_bca_16.x
	$(XRUN) -n 2 ./gmove_cp_2.x
	$(XRUN) -n 4 ./gmove_cp_4.x
	$(XRUN) -n 8 ./gmove_cp_8.x
	$(XRUN) -n 16 ./gmove_cp_16.x
	$(XRUN) -n 2 ./gmove_G2L_2.x
#	$(XRUN) -n 4 ./gmove_G2L_4.x
#	$(XRUN) -n 8 ./gmove_G2L_8.x
#	$(XRUN) -n 16 ./gmove_G2L_16.x
	$(XRUN) -n 4 ./337_gmove.x
	$(XRUN) -n 2 ./417_gmove.x
	$(XRUN) -n 2 ./gmove_inout_1.x
	$(XRUN) -n 8 ./gmove_inout_3.x
#	$(XRUN) -n 4 ./gmove_test_normal.x
#	$(XRUN) -n 8 ./gmove_test_in.x
#	$(XRUN) -n 8 ./gmove_test_out.x
	$(XRUN) -n 8 ./gmove_opt_53.x
	$(XRUN) -n 4 ./gmove_opt_54.x
	$(XRUN) -n 4 ./141_gmove.x
	$(XRUN) -n 2 ./272_gmove.x

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

