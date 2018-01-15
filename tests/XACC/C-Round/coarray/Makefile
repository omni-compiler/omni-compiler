XCC     = xmpcc -xacc -O2
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

ifeq ($(JOBSCHED), PBS)
XRUN    = mpirun_rsh -hostfile $$PBS_NODEFILE
XRUN_ENV = MV2_ENABLE_AFFINITY=0 MV2_SHOW_CPU_BINDING=1 MV2_USE_CUDA=1 XACC_COMM_MODE=1 MV2_CUDA_IPC=0
XSCRIPT = ../run_pbs.sh
XSUBMIT = qsub
endif
XRUN    ?= mpiexec

.PHONY: clean all default run submit

all default: $(EXES)

.SUFFIXES: .x .c

.c.x:
	$(XCC) -o $@ $<

run: $(EXES)
	$(XRUN) -n 2 $(XRUN_ENV) ./coarray_put_1dim.x
	$(XRUN) -n 2 $(XRUN_ENV) ./coarray_get_1dim.x

submit: $(EXES) $(XSCRIPT)
	$(XSUBMIT) $(XSCRIPT)

clean:
	rm -f $(EXES) $(OBJS)
