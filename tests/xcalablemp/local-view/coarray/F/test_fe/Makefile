OPT=--debug

#OUT=call_sync_all sync_all syncall lextest goto 
OUT=call_sync_all sync_all syncall goto 

SRC=$(OUT:%=%.f90)
RESULT=$(OUT:%=%.result)


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

RUN/%: %
	cp $< $@

### compile test programs
all:     $(OUT)

### execute on local environment
run:	$(RESULT)

### submit on K/FX10
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

