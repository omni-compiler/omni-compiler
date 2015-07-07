bin_PROGRAMS=C_Front
MKDIR_P = @MKDIR_P@

C_Front_SOURCES=\
	c-parser.y c-lexer.l c-expr.c c-token.c c-exprcode.c \
	c-dump.c c-comp.c c-reduce.c c-xcodeml.c c-const.c \
	c-gcctype.c c-option.c c-type.c c-convert.c \
	ccol-list.c ccol-hash.c c-main.c \
	c-pragma.c c-omp-pragma.c c-xmp-pragma.c c-acc-pragma.c \
	c-omp-xcodeml.c c-xmp-xcodeml.c c-acc-xcodeml.c c-ptree.c

ccol.h: ccol-list.h ccol-hash.h
ccol-list.c: ccol-list.h
ccol-hash.c: ccol-hash.h
c-expr.h: ccol.h c-exprcode.h c-error.h
c-expr.c: c-cmn.h c-expr.h
c-lexyacc.h: ccol.h
c-parser.y: c-cmn.h c-lexyacc.h c-expr.h
c-parser.h: c-parser.y
c-lexer.l: c-cmn.h c-lexyacc.h c-expr.h c-token.h c-parser.h
c-option.c: c-option.h
c-gcctype.c: c-gcctype.h
c-convert.c: c-expr.h
c-const.c: c-expr.h c-const.h
c-comp.c: c-expr.h c-comp.h
c-option.c: c-expr.h c-option.h
c-pragma.c: c-pragma.h c-omp.h c-xmp.h c-acc.h
c-omp-pragma.c: c-pragma.h c-omp.h
c-xmp-pragma.c: c-pragma.h c-xmp.h
c-acc-pragma.c: c-pragma.h c-acc.h
c-omp-pragma.c: c-pragma.h c-omp.h
c-xmp-pragma.c: c-pragma.h c-xmp.h
c-acc-pragma.c: c-pragma.h c-acc.h
c-xcodeml.c: c-expr.h

EXTRA_C_Front_SOURCES=c-dbgmain.c

AM_YFLAGS=-v -d
#AM_CFLAGS=-std=c99 -Wall $(DEBUG_FLAGS)

$(C_Front_SOURCES): c-exprcode.h

c-exprcode.c c-exprcode.h: c-exprcode.pl c-exprcode.def
	$(PERL5) c-exprcode.pl

c-token.c c-token.h: c-token.pl c-parser.h
	$(PERL5) c-token.pl

c-lexer.c : c-lexer.l
	$(LEX) -oc-lexer.c c-lexer.l

debug: C_debugfront

C_debugfront: $(C_Front_OBJECTS) c-dbgmain.o
	$(LINK) $(C_Front_LDADD) $(LIBS) c-parser.o c-lexer.o c-expr.o \
	c-token.o c-exprcode.o c-dump.o c-comp.o c-reduce.o c-const.o \
	c-gcctype.o c-option.o c-type.o c-convert.o ccol-list.o ccol-hash.o \
	c-dbgmain.o c-pragma.o c-omp-pragma.o c-xmp-pragma.o c-acc-pragma.o \
	c-xcodeml.o c-omp-xcodeml.o c-xmp-xcodeml.o c-acc-xcodeml.o

clean-local:
	rm -f C_debugfront tags *.o *.log *.output c-lexer.c
	rm -rf ../doc

html-local:
	if test ! -e "../doc"; then $(MKDIR_P) ../doc; fi
	doxygen >/dev/null

.NOTPARALLEL:
