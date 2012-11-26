# $TSUKUBA_Release: Omni OpenMP Compiler 3 $ # $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
bin_PROGRAMS=F_Front

F_Front_SOURCES= F95-main.c C-expr-mem.c C-exprcode.c F-datatype.c F-ident.c \
	F95-parser.y F-mem.c F-compile.c F-compile-decl.c F-compile-expr.c \
	F-opt-expv.c F-output-xcodeml.c F-io-xcodeml.c \
	F-data.c F-datasub.c F-equiv.c F-varutil.c \
	F-intrinsic.c F-intrinsics-table.c F-OMP.c F-XMP.c \
	xcodeml-node.c xcodeml-parse.c \
	xcodeml-util.c xcodeml-type.c xcodeml-traverse.c xcodeml-output-F.c \
	F-dump.c F-type-attr-tbl.c module-manager.c hash.c F-input-xmod.c \
	F-module-procedure.c

AM_YFLAGS=-v

AM_CFLAGS=-DBUGFIX -D__XCODEML__ -DYYDEBUG -I$(LIBXML_PREFIX)/include/libxml2 -Wall

F_Front_LDADD=-L$(LIBXML_PREFIX)/lib -lxml2 -lm

C-exprcode.h: C-exprcode.def C-exprcode.h.awk
	$(AWK) -f C-exprcode.h.awk < C-exprcode.def > C-exprcode.h

C-exprcode.c: C-exprcode.def C-exprcode.c.awk
	$(AWK) -f C-exprcode.c.awk < C-exprcode.def > C-exprcode.c

$(F_Front_SOURCES): C-exprcode.h

F-intrinsic.c F-intrinsics-table.c: F-intrinsics-types.h

clean-local:
	rm -rf ../doc
	rm -f *~ core core.* *.core

html-local:
	if test ! -e "../doc"; then mkdir ../doc; fi
	doxygen >/dev/null

