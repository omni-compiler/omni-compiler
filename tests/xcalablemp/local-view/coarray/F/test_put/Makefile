#OPT= --debug
#OPT= -g --debug --verbose

OUT_OK    = a1d1-3 a1d1-2 a1d1-1 a6d3 a6d4-2  a6d3-1 a6d3-2 a3d3 a2d2  a2d2-1-1 a2d2-1-5  s-2 s-1 a1d1-4 a6d4-1  a0d0-2 a0d0-3
OUT_NG    = a2d0 a2d1  a2d1-2 a2d1-3
OUT_darg  = dummyarg-2
OUT_bound = l0-3okB
OUT_NGB   = a2d2-2NGB
#OUT_type  = a1d1-typeNG
#OUT_err  = l0-2NG

#OUT=$(OUT_OK) $(OUT_NG) $(OUT_darg) $(OUT_bound) $(OUT_type) $(OUT_NGB)
OUT=$(OUT_OK) $(OUT_NG) $(OUT_darg) $(OUT_bound) $(OUT_NGB)

SRC= $(OUT:%=%.f90)
RESULT= $(OUT:%=%.result)

%:  %.f90
	xmpf90 $(OPT) -o $@ $<

%:  %.c
	xmpcc $(OPT) -o $@ $<

%.result: %.f90
	xmpf90 $(OPT) -o $* $<
	mpirun -n 3 $* 2>&1 | tee $@

%.result: %.c
	xmpcc $(OPT) -o $* $<
	mpirun -n 3 $* 2>&1 | tee $@

%.result: %
	mpirun -n 3 $< 2>&1 | tee $@

RUN/%:: %
	cp $< $@

RUN/%:: 
	touch $@

### compile test programs
all:     $(OUT)

### execute on local environment
run:	$(RESULT)

### prepare to execute on K
submit: all RUN/go.sh $(OUT:%=RUN/%)
	cd RUN; pjsub go.sh

RUN/go.sh: go.template Makefile
	sed "s/@OUT@/$(OUT)/" $< >$@

### clean
clean:
	rm -f *~ *.o $(OUT) *.result a.out
	rm -f __omni_tmp__/*.F90 __omni_tmp__/*.xml
	rm -f __omni_tmp__/*~ __omni_tmp__/*.in
	rm -f omni_traverse*.o
	rm -f $(OUT:%=RUN/%)
	rm -f *.mod *.xmod

