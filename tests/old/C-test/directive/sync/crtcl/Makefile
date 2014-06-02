# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= /bin/sh -c
CC		= omcc
CFLAGS		= -O

PROGRAMS	= crtcl001 crtcl002 crtcl003 crtcl004 \
		  crtcl005 crtcl006 crtcl007
OBJS		= crtcl001.o crtcl002.o crtcl003.o crtcl004.o \
		  crtcl005.o crtcl006.o crtcl007.o
SRCS		= crtcl001.c crtcl002.c crtcl003.c crtcl004.c \
		  crtcl005.c crtcl006.c crtcl007.c


all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
