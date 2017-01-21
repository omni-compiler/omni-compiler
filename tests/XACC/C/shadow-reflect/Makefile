XCC     = xmpcc -xacc -O2
TESTS   = $(wildcard *.c)
EXES    = $(TESTS:.c=.x)
OBJS    = $(TESTS:.c=.o)

ifeq ($(JOBSCHED), PBS)
XRUN    = mpirun_rsh -hostfile $$PBS_NODEFILE
XSCRIPT = run_pbs.sh
XSUBMIT = qsub
endif
XRUN    ?= mpiexec

.PHONY: clean all default run submit

all default: $(EXES)

.SUFFIXES: .x .c

.c.x:
	$(XCC) -o $@ $<

run: $(EXES)
	$(XRUN) -n 2 ./reflect_acc_1.x
	$(XRUN) -n 8 ./reflect_acc_2.x
	$(XRUN) -n 2 ./reflect_acc_3.x
	$(XRUN) -n 8 ./reflect_acc_1_massive.x
	$(XRUN) -n 8 ./reflect_acc_2_massive.x
	$(XRUN) -n 4 ./reflect_acc_2_dup_1.x
	$(XRUN) -n 8 ./reflect_acc_2_dup_2.x
	$(XRUN) -n 4 ./shadow_reflect.x
	$(XRUN) -n 2 ./reflect_acc_2_dup_1-2.x
	$(XRUN) -n 4 ./shadow_reflect_mem.x
	$(XRUN) -n 4 ./reflect_acc_oneside.x

submit: $(EXES) $(XSCRIPT)
	$(XSUBMIT) $(XSCRIPT)

clean:
	rm -f $(EXES) $(OBJS)
