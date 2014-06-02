# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= /bin/sh -c
CC		= omcc
CFLAGS		= -O

PROGRAMS	= in_prll001 in_prll002 in_prll003 in_prll004 \
		  in_prll005 in_prll006 in_prll007
OBJS		= in_prll001.o in_prll002.o in_prll003.o in_prll004.o \
		  in_prll005.o in_prll006.o in_prll007.o
SRCS		= in_prll001.c in_prll002.c in_prll003.c in_prll004.c \
		  in_prll005.c in_prll006.c in_prll007.c


all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
