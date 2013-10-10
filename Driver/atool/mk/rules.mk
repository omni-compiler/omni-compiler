# $Id: rules.mk 86 2012-07-30 05:33:07Z m-hirano $

.SUFFIXES:	.a .la .ln .o .lo .s .S .c .cc .cpp .i

.c.o:
	LC_ALL=C $(CC) $(CFLAGS) $(CPPFLAGS) -c $(srcdir)/$< -o $@

.c.lo:
	$(LTCOMPILE_CC) -c $(srcdir)/$*.c

.c.i:
	LC_ALL=C $(CC) -E $(CPPFLAGS) $(srcdir)/$*.c | uniq > $(srcdir)/$*.i

.cc.o:
	LC_ALL=C $(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(srcdir)/$< -o $@

.cc.lo:
	$(LTCOMPILE_CXX) -c $(srcdir)/$< -o $@

.cc.i:
	LC_ALL=C $(CXX) -E $(CPPFLAGS) $(srcdir)/$*.cc | uniq > $(srcdir)/$*.i

.cpp.o:
	LC_ALL=C $(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(srcdir)/$< -o $@

.cpp.lo:
	$(LTCOMPILE_CXX) -c $(srcdir)/$< -o $@

.cpp.i:
	LC_ALL=C $(CXX) -E $(CPPFLAGS) $(srcdir)/$*.cpp | uniq > $(srcdir)/$*.i

.s.lo:
	$(LTCOMPILE_CC) -c $(srcdir)/$*.s

.s.o:
	LC_ALL=C $(CC) $(CFLAGS) $(CPPFLAGS) -c $(srcdir)/$*.s

.S.lo:
	$(LTCOMPILE_CC) -c $(srcdir)/$*.S

.S.o:
	LC_ALL=C $(CC) $(CFLAGS) $(CPPFLAGS) -c $(srcdir)/$*.S

.c.s:
	LC_ALL=C $(CC) -S $(CFLAGS) $(CPPFLAGS) -c $(srcdir)/$*.c \
		-o $(srcdir)/$*.s

.cc.s:
	LC_ALL=C $(CXX) -S $(CXXFLAGS) $(CPPFLAGS) -c $(srcdir)/$*.cc \
		-o $(srcdir)/$*.s

.cpp.s:
	LC_ALL=C $(CXX) -S $(CXXFLAGS) $(CPPFLAGS) -c $(srcdir)/$*.cpp \
		-o $(srcdir)/$*.s

all::		ALL
ALL::		$(TARGETS)

ifdef INSTALL_EXE_TARGETS
install::	install-exe
install-exe::	$(INSTALL_EXE_TARGETS)
	@if test ! -z "$(INSTALL_EXE_TARGETS)" -a \
		 ! -z "$(INSTALL_EXE_DIR)" ; then \
		$(MKDIR) $(INSTALL_EXE_DIR) > /dev/null 2>&1 ; \
		for i in $(INSTALL_EXE_TARGETS) ; do \
			$(LTINSTALL_EXE) $$i $(INSTALL_EXE_DIR) ; \
		done ; \
	fi
else
install::	install-exe
install-exe::
	@true
endif

ifdef INSTALL_LIB_TARGETS
install::	install-lib
install-lib::	$(INSTALL_LIB_TARGETS)
	@if test ! -z "$(INSTALL_LIB_TARGETS)" -a \
		 ! -z "$(INSTALL_LIB_DIR)" ; then \
		$(MKDIR) $(INSTALL_LIB_DIR) > /dev/null 2>&1 ; \
		for i in $(INSTALL_LIB_TARGETS) ; do \
			$(LTINSTALL_LIB) $$i $(INSTALL_LIB_DIR) ; \
		done ; \
	fi
else
install::	install-lib
install-lib::
	@true
endif

ifdef INSTALL_HEADER_TARGETS
install::	install-header
install-header::	$(INSTALL_HEADER_TARGETS)
	@if test ! -z "$(INSTALL_HEADER_TARGETS)" -a \
		 ! -z "$(INSTALL_HEADER_DIR)" ; then \
		$(MKDIR) $(INSTALL_HEADER_DIR) > /dev/null 2>&1 ; \
		for i in $(INSTALL_HEADER_TARGETS) ; do \
			$(LTINSTALL_HEADER) $$i $(INSTALL_HEADER_DIR) ; \
		done ; \
	fi
else
install::	install-header
install-header::
	@true
endif

ifdef SRCS
depend::
	@if test ! -z "$(SRCS)" ; then \
		> .depend ; \
		$(CXX) -M $(CPPFLAGS) $(SRCS) | sed 's:\.o\::\.lo\::' \
		> .depend ; \
		if test $$? -ne 0 ; then \
			echo depend in `pwd` failed. ; \
		else \
			echo depend in `pwd` succeeded. ; \
		fi ; \
	fi
else
depend::
	@true
endif

ifdef TARGET_LIB
$(TARGET_LIB):	$(OBJS)
	$(LTCLEAN) $@
	$(LTLINK_CXX) -o $@ $(OBJS) $(LDFLAGS) $(DEP_LIBS)
endif

ifdef TARGET_EXE
$(TARGET_EXE):	$(OBJS)
	$(LTCLEAN) $@
	$(LTLINK_CXX) -o $@ $(OBJS) $(LDFLAGS) $(DEP_LIBS)
endif

clean::
	$(LTCLEAN) $(OBJS) *.i *~ *.~*~ core core.* *.core $(TARGETS)

distclean::	clean
	$(RM) Makefile .depend
	$(RM) -rf html

Makefiles::
	@if test -x $(TOPDIR)/config.status; then \
		(cd $(TOPDIR); LC_ALL=C sh ./config.status; \
			$(RM) config.log) ; \
	fi

ifdef LIBTOOL_DEPS
libtool::	$(LIBTOOL_DEPS)
	@if test -x $(TOPDIR)/config.status; then \
		(cd $(TOPDIR); LC_ALL=C sh ./config.status libtool; \
			$(RM) config.log) ; \
	fi
endif

doxygen::
	sh $(MKRULESDIR)/mkfiles.sh
	$(RM) -rf ./html ./latex
	doxygen $(MKRULESDIR)/doxygen.conf
	$(RM) -rf ./latex ./.files

wc::
	@find . -type f -name '*.c' -o -name '*.cpp' -o -name '*.h' | \
	xargs wc

dostext::
	sh $(MKRULESDIR)/doDosText.sh

ifdef DIRS
all depend clean distclean install install-exe install-lib install-header::
	@for i in / $(DIRS) ; do \
		case $$i in \
			/) continue ;; \
			*) (cd $$i; $(MAKE) $@) ;; \
		esac ; \
	done
endif
