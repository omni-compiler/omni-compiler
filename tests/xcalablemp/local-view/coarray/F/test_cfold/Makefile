#OPT= --debug
#OPT= -g --debug --verbose

#OUT=param1 param3 param2 param4
OUT=param1 param3

SRC= $(OUT:%=%.f90)
RESULT= $(OUT:%=%.result)

%:  %.f90
	xmpf90 $(OPT) -o $@ $<

%:  %.c
	xmpcc $(OPT) -o $@ $<

%.result: %.f90
	xmpf90 $(OPT) -o $* $<
	mpiexec -n 3 $* 2>&1 | tee $@

%.result: %.c
	xmpcc $(OPT) -o $* $<
	mpiexec -n 3 $* 2>&1 | tee $@

%.result: %
	mpiexec -n 3 $< 2>&1 | tee $@

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

