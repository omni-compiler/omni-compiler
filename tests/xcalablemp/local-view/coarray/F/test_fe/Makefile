#OPT=--debug

#OUT=call_sync_all sync_all syncall lextest goto 
OUT=call_sync_all sync_all syncall goto 

SRC=$(OUT:%=%.f90)
RESULT=$(OUT:%=%.result)


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

RUN/%: %
	cp $< $@

### compile test programs
all:     $(OUT)

### execute on local environment
run:	$(RESULT)

### submit on K/FX10
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
