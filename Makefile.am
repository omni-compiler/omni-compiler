SUBDIRS=XcodeML-Common C-FrontEnd/src C-BackEnd F-FrontEnd/src F-BackEnd \
	XcodeML-Exc-Tools Driver/src libtlog/src_threads libtlog/src_mpi \
        libxmp/include libxmp/src libxmp/src_threads libxmpf/include libxmpf/src

if MOD2XMOD
    SUBDIRS+=F-FrontEnd/gnu_module
endif

if !SUPERUX
    SUBDIRS+=libompc/src libompf/src
endif

if ACC
    SUBDIRS+=libacc/src libacc/include
endif

if ATOOL
    SUBDIRS+=Driver/atool
endif

BASE_TESTDIR = "tests/xcalablemp/"
TESTDIRS = "global-view/array/C/" "global-view/array/F/" "global-view/bcast/C/" "global-view/bcast/F/" \
	   "global-view/loop/C" "global-view/loop/F" "global-view/reduction/C" "global-view/reduction/F" \
	   "global-view/shadow-reflect/C/" "global-view/shadow-reflect/F/" "global-view/task/C"  "global-view/task/F" \
	   "global-view/template_fix/C/" "global-view/template_fix/F/" "global-view/gmove/C/" "global-view/gmove/F/" \
	   "global-view/inquiry/C/" "global-view/inquiry/F/" "global-view/gblock/C/" "global-view/gblock/F/" \
	   "global-view/align/C/" "global-view/align/F/" "global-view/intrinsic/F/" \
	   "global-view/util/C/" "global-view/util/F/" "others/F" "others/C"
BASE_LARGE_TESTDIR = "tests/xcalablemp/LARGE-SIZE-TESTS/"
LARGE_TESTDIRS = "others/C"

# "global-view/intrinsic/C/" includes imaginary numbers.
# But SUPERUX and POWERPC cannot deal with imaginary numbers.
if !SUPERUX_OR_POWERPC
  TESTDIRS += "global-view/intrinsic/C/"
endif

if GASNET
   TESTDIRS += "local-view/post-wait/C" "local-view/coarray/C" "local-view/other/C" "local-view/other/F" \
	       "local-view/coarray/F/MIX" "local-view/coarray/F/ALLOC" "local-view/coarray/F/LIB" \
	       "local-view/coarray/F/GET" "local-view/coarray/F/PUT" "local-view/coarray/F/SCOPE" \
	       "local-view/coarray/F/CFOLD" "local-view/lock-unlock/C" "local-view/intrinsic/C"
   LARGE_TESTDIRS += "local-view/coarray/C/"
endif
if FJRDMA
   TESTDIRS += "local-view/post-wait/C" "local-view/coarray/C" "local-view/other/C" "local-view/other/F" \
               "local-view/coarray/F/MIX" "local-view/coarray/F/ALLOC" "local-view/coarray/F/LIB" \
	       "local-view/coarray/F/GET" "local-view/coarray/F/PUT" "local-view/coarray/F/SCOPE" \
	       "local-view/coarray/F/CFOLD" 
   LARGE_TESTDIRS += "local-view/coarray/C/"
endif
if MPI3
   TESTDIRS += "global-view/async/C/" "global-view/async/F/"
endif

.PHONY: tests clean-tests run-tests large-tests clean-large-tests run-large-tests release slurm slurm-check remote-tests clean-local distclean-local testdirs

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

testdirs:
	echo $(TESTDIRS)

PACKAGE_NAME = omnicompiler-$(VERSION)
WORKDIR = /tmp/$(PACKAGE_NAME)
release:
	@if [ -e $(WORKDIR) ]; then \
	  echo "Exist ! $(WORKDIR). Please delete it."; exit 1; \
	fi; \
	cp -a . $(WORKDIR); cd $(WORKDIR); \
	rm -rf .git .gitignore; \
	sh autogen.sh; ./configure; \
	find . -name Makefile.in -exec touch \{\} \;
	cd /tmp; tar cjf @TOPDIR@/$(PACKAGE_NAME).tar.bz2 $(PACKAGE_NAME); \
	echo "Created $(PACKAGE_NAME).tar.bz2"; \
	rm -rf $(WORKDIR)

slurm:
	bash ./tests/xcalablemp/scripts/create_sorted_list.sh $(BASE_TESTDIR) $(TESTDIRS) > $(SORTED_LIST); \
	bash ./tests/xcalablemp/scripts/make_slurm.sh $(XMP_PATH) $(MPI_PATH) $(SORTED_LIST) $(BASE_TESTDIR)

slurm-check:
	@bash ./tests/xcalablemp/scripts/check_error_code.sh $(BASE_TESTDIR) $(TESTDIRS)

remote-tests:
	@bash ./tests/xcalablemp/scripts/remote-tests.sh $(USER)

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
	tests/tiny/Makefile

distclean-local:
	if test ! -z "$(DELETABLES)"; then \
		rm -rf $(DELETABLES) ; \
	fi	
