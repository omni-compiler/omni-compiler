TESTDIR = $(shell pwd)
PROJDIR = $(shell pwd)/../..
INCDIR = $(PROJDIR)/include
TMPDIR = $(TESTDIR)

CC		= /home/mnakao/work/xmp-trunk/bin/omc2c
CFLAGS	= -O -Topenmp -I$(TESTDIR) -I$(INCDIR) \
	--tempdir $(TMPDIR) --debug --Wx-dxcode \
	--Wf--no-warn-W002 --Wf--no-warn-W003
LDFLAGS	= -Topenmp

SUBDIRS	= directive/parallel \
	  directive/wkshr/for directive/wkshr/sngl directive/wkshr/sctn \
	  directive/sync/master directive/sync/ordrd directive/sync/barrier \
	  directive/sync/atomic directive/sync/flush directive/sync/crtcl \
	  directive/data/prvt directive/data/fstprvt directive/data/lstprvt \
	  directive/data/shrd directive/data/dflt directive/data/rdctn \
	  directive/data/cpyin directive/data/attr directive/data/thdprvt \
	  directive/combine/for directive/combine/sctn \
	  directive/cndcmp \
	  directive/nest \
	  rtlib/get_nstd rtlib/get_thrdn rtlib/get_mthrd rtlib/get_nthrd \
	  rtlib/set_nthrd rtlib/get_nproc rtlib/in_prll rtlib/set_dynmc \
	  rtlib/get_dynmc rtlib/set_nstd rtlib/init_lck rtlib/init_nlck \
	  rtlib/dstry_lck rtlib/dstry_nlck rtlib/set_nlck rtlib/set_lck \
	  rtlib/uset_lck rtlib/uset_nlck rtlib/tst_nlck rtlib/tst_lck \
	  env \
	  other

all:
	for i in ${SUBDIRS} ;  do \
	  (cd $$i;  $(MAKE) -i CC="${CC}" CFLAGS="${CFLAGS}" LDFLAGS="${LDFLAGS}"); \
	done

clean:
	for i in ${SUBDIRS} ;  do \
	  (cd $$i;  $(MAKE) clean); \
	done
	rm -rf $(TESTDIR)/__omni_tmp__

test:
	for i in ${SUBDIRS} ;  do \
	  (cd $$i;  $(MAKE) -i CC="${CC}" CFLAGS="${CFLAGS}" LDFLAGS="${LDFLAGS}" test); \
	done

distclean:	clean
	true

