SHELL=/bin/sh

TESTDIR = $(shell pwd)
PROJDIR = $(shell pwd)/../..
INCDIR = $(PROJDIR)/include
TMPDIR = $(TESTDIR)

prefix = /home/mnakao/work/xmp-trunk
FFLAGS	= -O -I$(TESTDIR) -I$(INCDIR) \
	-Topenmp --tempdir $(TMPDIR) --debug
LDFLAGS	= -Topenmp
FC = /home/mnakao/work/xmp-trunk/bin/omf2f

SRCS=atmc001.f brir001.f cndcmp001.f crit001.f \
	do001.f do002.f do003.f \
	gtnthd001.f lock001.f ordr001.f \
	lstprv001.f \
	par001.f par002.f \
	pardo001.f pardo002.f \
	parsec001.f sngl001.f stnthd001.f \
	rdctn001.f rdctn002.f \
	thdprv001.f thdprv002.f thdprv003.f \
	dflt001.f indvar001.f \

all:	test

fullbuild:
	@for i in $(SRCS); do \
		(EXE=./`basename $$i .f`; \
			rm -f $$EXE; \
			echo build $$i; \
			$(FC) $(FFLAGS) $$i -o $$EXE); \
	done

build:
	@for i in $(SRCS); do \
		(EXE=./`basename $$i .f`; \
			if test ! -x $$EXE; then \
				echo build $$i; \
				$(FC) $(FFLAGS) $$i -o $$EXE; \
			fi); \
	done
	
test:	build
	@for i in $(SRCS); do \
		(OMP_NUM_THREADS=16; export OMP_NUM_THREADS; \
			EXE=./`basename $$i .f`; NM=`basename $$i .f`; \
			if test -x $$EXE; then \
				okStr=`$$EXE 2>&1 | grep "$$NM PASSED" || true`; \
				if test ! -z "$$okStr"; then \
					echo "$$NM ...	PASSED"; \
				else \
					echo "$$NM ...	*FAILED*"; \
				fi; \
			else \
				exit 1; \
			fi); \
		(if test $$? -ne 0; then \
			echo "can't test `basename $$i .f`."; \
			fi); \
	done

install:
	true

clean:
	@for i in $(SRCS); do \
		(EXE=./`basename $$i .f`; \
			rm -f $$EXE); \
	done
	rm -f *.o *.mod *~ core

distclean:	clean
	rm -f Makefile

Makefile::
	(cd ../../..; sh ./config.status)

