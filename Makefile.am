SUBDIRS=@XCODEML_COMMON_DIR@  @C_FRONTEND_DIR@ @C_BACKEND_DIR@ \
        @F_FRONTEND_DIR@ @F_BACKEND_DIR@ @XCODEML_EXC_TOOLS_DIR@ \
        @OMDRIVER_DIR@ @LIBTLOG_DIR@ @LIBTLOG_MPI_DIR@ \
        @LIBXMP_HEADER_DIR@ @LIBXMP_DIR@ @LIBXMP_THREADS_DIR@ \
        @LIBXMPF_HEADER_DIR@ @LIBXMPF_DIR@

if !IS_SUPERUX
    SUBDIRS+=@LIBOMPC_DIR@ @LIBOMPF_DIR@
endif

if ENABLE_ACC
    SUBDIRS+=@LIBACC_DIR@ @LIBACC_HEADER_DIR@
endif

if ENABLE_ATOOL
    SUBDIRS+=@ATOOL_DIR@
endif

XMPPKG_NAME:="@XMPPKG_NAME@"
XMPPKG_TMPDIR_TOP:="/tmp/@XMPPKG_NAME@"
XMPPKG_TMPDIR:=$(XMPPKG_TMPDIR_TOP)/tmp.$$
DEB_VER_FILE="/etc/debian_version"
RH_REL_FILE="/etc/redhat-release"

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
   TESTDIRS += "local-view/post-wait/C" "local-view/coarray/C" "local-view/coarray/C" "local-view/other/C"
   LARGE_TESTDIRS += "local-view/coarray/C/"
endif
if FJRDMA
   TESTDIRS += "local-view/post-wait/C" "local-view/coarray/C" "local-view/coarray/C" "local-view/other/C"
   LARGE_TESTDIRS += "local-view/coarray/C/"
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

clean-local:
	rm -rf autom4te.cache
	find . -type f \( \
		-name '*~' -o \
		-name '*.core' \
	\) -exec rm -f {} \;

DELETABLES = C-BackEnd/bin/C_Back \
	C-FrontEnd/src/c-exprcode.c \
	C-FrontEnd/src/c-exprcode.h \
	C-FrontEnd/src/c-parser.c \
	C-FrontEnd/src/c-parser.h \
	Driver/bin/omc2c \
	Driver/bin/omc2x \
	Driver/bin/omclinker \
	Driver/bin/omcnative \
	Driver/bin/omcpp \
	Driver/bin/omcx2x \
	Driver/bin/omf2f \
	Driver/bin/omf2x \
	Driver/bin/omflinker \
	Driver/bin/omfnative \
	Driver/bin/omfpp \
	Driver/bin/omfx2x \
	Driver/bin/oml2x \
	Driver/bin/omlinker \
	Driver/bin/omnative \
	Driver/bin/ompcc \
	Driver/bin/ompf90 \
	Driver/bin/ompp \
	Driver/bin/omx2c \
	Driver/bin/omx2f \
	Driver/bin/omx2l \
	Driver/bin/omx2x \
	Driver/bin/xmpcc \
	Driver/bin/xmpf90 \
	Driver/bin/omtranslate \
	Driver/bin/omni_collect_init \
	Driver/etc/java.conf \
	Driver/etc/l2x.conf.openmp \
	Driver/etc/l2x.conf.xmp \
	Driver/etc/l2x.conf.xmpf \
	Driver/etc/l2x.conf.acc \
	Driver/etc/linker.conf.openmp \
	Driver/etc/linker.conf.xmp \
	Driver/etc/linker.conf.xmpf \
	Driver/etc/linker.conf.acc \
	Driver/etc/native.conf.openmp \
	Driver/etc/native.conf.xmp \
	Driver/etc/native.conf.xmpf \
	Driver/etc/native.conf.acc \
	Driver/etc/omc.conf \
	Driver/etc/omf.conf \
	Driver/etc/pp.conf.openmp \
	Driver/etc/pp.conf.xmp \
	Driver/etc/pp.conf.xmpf \
	Driver/etc/pp.conf.acc \
	Driver/etc/x2x.conf.openmp \
	Driver/etc/x2x.conf.xmp \
	Driver/etc/x2x.conf.xmpf \
	Driver/etc/x2x.conf.acc \
	Driver/etc/setup.conf \
	C-FrontEnd/src/c-token.c \
	F-BackEnd/bin/F_Back \
	F-BackEnd/bin/F_Trans \
	F-BackEnd/bin/F_Decomp \
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
