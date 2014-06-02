# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= /bin/sh -c
CC		= omcc
CFLAGS		= -O

PROGRAMS	= atomic001 atomic002 atomic003 atomic004 \
		  atomic005 atomic006 atomic007 atomic008 \
		  atomic009 atomic010 atomic011 atomic012
OBJS		= atomic001.o atomic002.o atomic003.o atomic004.o \
		  atomic005.o atomic006.o atomic007.o atomic008.o \
		  atomic009.o atomic010.o atomic011.o atomic012.o
SRCS		= atomic001.c atomic002.c atomic003.c atomic004.c \
		  atomic005.c atomic006.c atomic007.c atomic008.c \
		  atomic009.c atomic010.c atomic011.c atomic012.c


all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
