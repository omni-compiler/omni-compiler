XFC     = xmpf90
FCFLAGS = -xacc
TESTS   = $(wildcard *.f90)
EXES    = $(TESTS:.f90=.x)
OBJS    = $(TESTS:.f90=.o)

ifeq ($(JOBSCHED), PBS)
XRUN    = mpirun_rsh -hostfile $$PBS_NODEFILE
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
	$(XRUN) -n 2 ./reflect_array4d_dist3d.x
	$(XRUN) -n 2 ./reflect_array4d_dist4d.x
	$(XRUN) -n 2 ./reflect_array4d_dist4d_width2.x
	$(XRUN) -n 2 ./reflect_array4d_dist4d_width2_2.x

submit: $(EXES) $(XSCRIPT)
	$(XSUBMIT) $(XSCRIPT)

clean:
	rm -f $(EXES) $(OBJS)
