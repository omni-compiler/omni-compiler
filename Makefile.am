SUBDIRS=XcodeML-Common C-FrontEnd/src C-BackEnd F-FrontEnd/src F-BackEnd \
	XcodeML-Exc-Tools Driver/src libtlog/src_threads libtlog/src_mpi \
        libxmp/include libxmp/src libxmp/src_threads libxmpf/include libxmpf/src

if !SUPERUX
    SUBDIRS+=libompc/src libompf/src
endif

if ACC
    SUBDIRS+=libacc/src libacc/include
endif

if ATOOL
    SUBDIRS+=Driver/atool
endif

XMPPKG_NAME=@XMPPKG_NAME@
XMPPKG_TMPDIR_TOP=/tmp/@XMPPKG_NAME@
XMPPKG_TMPDIR:=$(XMPPKG_TMPDIR_TOP)/tmp
DEB_VER_FILE=/etc/debian_version
RH_REL_FILE=/etc/redhat-release

release-rpm:
	if test -f $(RH_REL_FILE) ; then \
	  mkdir -p $(XMPPKG_TMPDIR_TOP); \
	  bash release/make_rpm.sh $(CURDIR) $(XMPPKG_TMPDIR_TOP); \
	  rm -rf $(XMPPKG_TMPDIR_TOP); \
	fi

release-deb:
	if test -f $(DEB_VER_FILE) ; then \
	  mkdir -p $(XMPPKG_TMPDIR); \
	  bash release/make_deb.sh $(CURDIR) $(XMPPKG_TMPDIR); \
	  rm -rf $(XMPPKG_TMPDIR_TOP); \
	fi

release: release-rpm release-deb

BASE_TESTDIR = "tests/xcalablemp/"
TESTDIRS = "global-view/array/C/" "global-view/array/F/" "global-view/bcast/C/" "global-view/bcast/F/" \
	   "global-view/loop/C" "global-view/loop/F" "global-view/reduction/C" "global-view/reduction/F" \
	   "global-view/shadow-reflect/C/" "global-view/shadow-reflect/F/" "global-view/task/C"  "global-view/task/F" \
	   "global-view/template_fix/C/" "global-view/template_fix/F/" "global-view/gmove/C/" "global-view/gmove/F/" \
	   "global-view/inquiry/C/" "global-view/inquiry/F/" "global-view/gblock/C/" "global-view/gblock/F/" \
	   "global-view/align/C/" "global-view/align/F/" "global-view/intrinsic/C/" "global-view/intrinsic/F/" \
	   "global-view/util/C/" "global-view/util/F/" "others/F" "others/C"
BASE_LARGE_TESTDIR = "tests/xcalablemp/LARGE-SIZE-TESTS/"
LARGE_TESTDIRS = "others/C"

if GASNET
   TESTDIRS += "local-view/post-wait/C" "local-view/coarray/C" "local-view/other/C"  "local-view/coarray/F"
   LARGE_TESTDIRS += "local-view/coarray/C/"
endif
if FJRDMA
   TESTDIRS += "local-view/post-wait/C" "local-view/coarray/C" "local-view/other/C" "local-view/coarray/F"
   LARGE_TESTDIRS += "local-view/coarray/C/"
endif
if MPI3
   TESTDIRS += "global-view/async/C/" "global-view/async/F/"
endif

.PHONY: tests clean-tests run-tests large-tests clean-large-tests run-large-tests

tests: 
	@for subdir in $(TESTDIRS) ; do \
	  $(MAKE) -C $(BASE_TESTDIR)/$$subdir ; \
	  if test $$? -ne 0; then exit 1; fi ;\
	done

clean-tests:
	@for subdir in $(TESTDIRS) ; do \
	  (cd $(BASE_TESTDIR)/$$subdir && $(MAKE) clean ) ;\
	done

run-tests:
	@for subdir in $(TESTDIRS); do \
	  $(MAKE) run -C $(BASE_TESTDIR)$$subdir; \
	  if test $$? -ne 0; then exit 1; fi ;\
	done

large-tests:
	@for subdir in $(LARGE_TESTDIRS) ; do \
	  $(MAKE) -C $(BASE_LARGE_TESTDIR)/$$subdir ; \
	  if test $$? -ne 0; then exit 1; fi ;\
	done

clean-large-tests:
	@for subdir in $(LARGE_TESTDIRS) ; do \
	  (cd $(BASE_LARGE_TESTDIR)/$$subdir && $(MAKE) clean ) ;\
	done

run-large-tests:
	@for subdir in $(LARGE_TESTDIRS); do \
	  $(MAKE) run -C $(BASE_LARGE_TESTDIR)$$subdir; \
	  if test $$? -ne 0; then exit 1; fi ;\
	done

slurm:
	@SORTED_LIST=/tmp/xmp_sorted.$$RANDOM; \
	bash ./tests/xcalablemp/scripts/create_sorted_list.sh $(BASE_TESTDIR) $(TESTDIRS) > $$SORTED_LIST; \
	bash ./tests/xcalablemp/scripts/make_slurm.sh $(XMP_PATH) $$SORTED_LIST $(BASE_TESTDIR); \
	rm -f $$SORTED_LIST

slurm-check:
	@bash ./tests/xcalablemp/scripts/check_error_code.sh $(BASE_TESTDIR) $(TESTDIRS)

slurm-timeline:
	@bash ./tests/xcalablemp/scripts/create_timeline.sh /home/mnakao/public_html/timeline $(JOB_NUM) $(ELAPSE_TIME_SEC) $(BASE_TESTDIR) $(TESTDIRS)

clean-local:
	rm -rf autom4te.cache
	find . -type f \( \
		-name '*~' -o \
		-name '*.core' \
	\) -exec rm -f {} \;

DELETABLES = Driver/bin/ompcc \
	Driver/bin/ompf90 \
	Driver/bin/xmpf90 \
	Driver/bin/xmpcc \
	Driver/etc/xmpcc.conf \
	Driver/etc/xmpf90.conf \
	Driver/etc/ompcc.conf \
	Driver/etc/ompf90.conf \
	Driver/libexec/omni_traverse \
        C-FrontEnd/src/c-exprcode.c \
        C-FrontEnd/src/c-exprcode.h \
        C-FrontEnd/src/c-parser.c \
        C-FrontEnd/src/c-parser.h \
	C-FrontEnd/src/c-token.c \
	F-FrontEnd/src/C-exprcode.c \
	F-FrontEnd/src/C-exprcode.h \
	F-FrontEnd/src/F95-parser.c \
	F-FrontEnd/src/F95-parser.output \
	XcodeML-Exc-Tools/src/exc/util/MachineDepConst.java \
	autom4te.cache/ \
	libtlog/src_mpi/bin/tlogview \
	libtlog/src_mpi/libtlog_mpi.a \
	libtlog/src_threads/libtlog.a \
	tests/C-test/Makefile \
	tests/F-test/OMP-test/Makefile \
	tests/clinpack/Makefile \
	tests/tiny/Makefile \
	$(XMPPKG_TMPDIR_TOP) \
	${XMPPKG_NAME}*.tar.gz \
	${XMPPKG_NAME}*.deb \
	${XMPPKG_NAME}*.rpm

distclean-local:
	if test ! -z "$(DELETABLES)"; then \
		rm -rf $(DELETABLES) ; \
	fi	
