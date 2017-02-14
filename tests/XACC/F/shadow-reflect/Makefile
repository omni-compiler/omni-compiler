XFC     = xmpf90
FCFLAGS = -xacc --Wn"-acc -Minfo=accel" --Wl-acc -lstdc++
TESTS   = $(wildcard *.f90)
EXES    = $(TESTS:.f90=.x)
OBJS    = $(TESTS:.f90=.o)

ifeq ($(JOBSCHED), PBS)
XRUN    = mpirun_rsh -hostfile $$PBS_NODEFILE
XRUN_ENV = MV2_ENABLE_AFFINITY=0 MV2_SHOW_CPU_BINDING=1 MV2_NUM_PORTS=2 MV2_USE_CUDA=1 XACC_COMM_MODE=1 MV2_CUDA_IPC=0
XSCRIPT = run_pbs.sh
XSUBMIT = qsub
endif
XRUN    ?= mpiexec

.PHONY: clean all default run submit

all default: $(EXES)

.SUFFIXES: .x .f90

.f90.x:
	$(XFC) $(FCFLAGS) -o $@ $<

run: $(EXES)
	$(XRUN) -n 2 $(XRUN_ENV) ./reflect_array4d_dist3d.x
	$(XRUN) -n 2 $(XRUN_ENV) ./reflect_array4d_dist4d.x
	$(XRUN) -n 2 $(XRUN_ENV) ./reflect_array4d_dist4d_width2.x
	$(XRUN) -n 2 $(XRUN_ENV) ./reflect_array4d_dist4d_width2_2.x

submit: $(EXES) $(XSCRIPT)
	$(XSUBMIT) $(XSCRIPT)

clean:
	rm -f $(EXES) $(OBJS) o_shadow_reflect e_shadow_reflect
