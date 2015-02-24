#OPT= --debug
#OPT= -g --debug --verbose
NPE= 2

OUT=costmtInSpdo costmtInSpdo-2

SRC= $(OUT:%=%.f90)
RESULT= $(OUT:%=%.result)

%:  %.f90
	xmpf90 $(OPT) -o $@ $<

%:  %.c
	xmpcc $(OPT) -o $@ $<

%.result: %
	mpirun -n $(NPE) $< 2>&1 | tee $@

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
submit: RUN/go.sh $(OUT:%=RUN/%)
	cd RUN; pjsub go.sh

RUN/go.sh: go.template Makefile
	sed "s/@OUT@/$(OUT)/" $< >$@

### clean
clean:
	rm -f *~ *.o $(OUT) *.result a.out
	rm -rf __omni_tmp__/
	rm -f $(OUT:%=RUN/%)
	rm -f *.mod *.xmod

