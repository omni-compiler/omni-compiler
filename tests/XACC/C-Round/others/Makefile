XCC     = xmpcc -xacc -O2 -lstdc++
TESTS   = $(wildcard *.c)
EXES    = $(TESTS:.c=.x)
OBJS    = $(TESTS:.c=.o)

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
	$(XRUN) -n 1 $(XRUN_ENV) ./voidmain.x

submit: $(EXES) $(XSCRIPT)
	$(XSUBMIT) $(XSCRIPT)

clean:
	rm -f $(EXES) $(OBJS)
