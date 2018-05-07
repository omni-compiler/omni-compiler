XCC     = xmpcc -xacc -O2
TESTS   = $(wildcard *.c)
EXES    = $(TESTS:.c=.x)
OBJS    = $(TESTS:.c=.o)

ifeq ($(JOBSCHED), PBS)
XRUN    = mpirun_rsh -hostfile $$PBS_NODEFILE
XSCRIPT = ../run_pbs.sh
XSUBMIT = qsub
endif
XRUN    ?= mpiexec

.PHONY: clean all default run submit

all default: $(EXES)

.SUFFIXES: .x .c

chk.o: ./chk/chk.c
	$(XCC) -c $^ -o $@
%.x: %.c chk.o
	$(XCC) -o $@ $< ./chk.o

run: $(EXES)
	$(XRUN) -n 2 ./gmove_bca_2.x

submit: $(EXES) $(XSCRIPT)
	$(XSUBMIT) $(XSCRIPT)

clean:
	rm -f $(EXES) $(OBJS) chk.o
