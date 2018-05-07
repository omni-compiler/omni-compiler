XFC     = xmpf90
XRUN    = mpiexec
EXES    = allocatable_character.x pointer_assignment_with_lower_bound.x 

all: allocatable_character.x iso_c_binding.o pointer_assignment_with_lower_bound.x

.PHONY: clean submit showlog cleanlog

.SUFFIXES: .x .o .f90 .F90

.f90.x:
	$(XFC) -o $@ $<

.F90.x:
	$(XFC) -o $@ $<

.f90.o:
	$(XFC) -c $<

.F90.o:
	$(XFC) -c $<

.f.x:
	$(XFC) -o $@ $<

.f.o:
	$(XFC) -c $<

.F.o:
	$(XFC) -c $<

run:

RUN:
	mkdir RUN

RUN/%.x:: %.x
	cp $< $@

RUN/go.sh: go.template Makefile
	cp $< $@ && grep XRUN Makefile | sed -e 's/(XRUN)/{XRUN}/' -e 's/ *= */=/' | grep -v Makefile >>$@

submit: all RUN RUN/go.sh $(EXES:%=RUN/%)
	cp 278\ spacefile.x RUN
	cd RUN; pjsub go.sh

showlog:
	cat RUN/go.sh.e* RUN/go.sh.o*

cleanlog:
	rm -rf RUN

clean: cleanlog
	rm -f $(EXES) *.o *.*mod *.s *.x
