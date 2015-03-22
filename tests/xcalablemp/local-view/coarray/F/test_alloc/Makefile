#OPT= --debug
#OPT= -g --debug --verbose

OUT_decl= decl1 decl2
OUT_size= # size2 sample #size1-err
#OUT_allo= allo1 allo3 allo4 #allo2

OUT = $(OUT_decl) $(OUT_size) $(OUT_allo)


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

xml:
	for f in __omni_tmp__/*_f90_out.xml; do \
	  xmllint --format $$f > $$f.xml; \
	done


### clean
clean:
	rm -f *~ *.o $(OUT) *.result a.out
	rm -rf __omni_tmp__/
	rm -f $(OUT:%=RUN/%)
	rm -f *.mod *.xmod

