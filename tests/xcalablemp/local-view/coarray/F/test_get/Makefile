#OPT= --debug
#OPT= -g --debug --verbose

OUT_basic  = a1d1-4 a1d1-1 a6d3-1 cn1-1NGB
OUT_io     = cn1d1-writeNGB
#OUT_io     = cn1d1-writeNGB cn1d1-write2
OUT_arg    = actualarg triplet actualarg-2 actualarg-3
OUT_nest   = nest nest-2NGB putgetNGB putget-2NGB putget-3NGB putget-4 putget-5 putget-6NGB
OUT_lib    = a1d1-5
OUT_darg   = dummyarg-1
###OUT_mod    = module-1
OUT_entire = axd0-1 axd1-1 axd2-1
OUT_bound  = cn1-2okB l0-1okB

#not supported
#OUT_struct = struct0d
#OUT=bug3

#bug #354        cnd0-1 cn0d0-write (see bug2.f90)

OUT = $(OUT_basic) $(OUT_io) $(OUT_arg) $(OUT_nest) $(OUT_lib) $(OUT_darg) $(OUT_mod) $(OUT_entire) $(OUT_bound)

SRC= $(OUT:%=%.f90)
RESULT= $(OUT:%=%.result)

%:  %.f90
	xmpf90 $(OPT) -o $@ $<

%:  %.c
	xmpcc $(OPT) -o $@ $<

%.result: %
	mpirun -n 3 $< 2>&1 | tee $@

%.result: %.f90
	xmpf90 $(OPT) -o $* $<
	mpirun -n 3 $* 2>&1 | tee $@

%.result: %.c
	xmpcc $(OPT) -o $* $<
	mpirun -n 3 $* 2>&1 | tee $@

RUN/%:: %
	cp $< $@

RUN/%:: 
	touch $@

### compile test programs
all:     $(OUT)

### execute on local environment
run:	$(RESULT)

### prepare to execute on K
submit: $(OUT) RUN/go.sh $(OUT:%=RUN/%)
	cd RUN; pjsub go.sh

RUN/go.sh: go.template Makefile
	sed "s/@OUT@/$(OUT)/" $< >$@

### clean
clean:
	rm -f *~ *.o $(OUT) $(RESULT)
	rm -f __omni_tmp__/*.F90 __omni_tmp__/*.xml
	rm -f __omni_tmp__/*~ __omni_tmp__/*.in
	rm -f omni_traverse_*.o
	rm -f $(OUT:%=RUN/%)
	rm -f *.mod *.xmod
