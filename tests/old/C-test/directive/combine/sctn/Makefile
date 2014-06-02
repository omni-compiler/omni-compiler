# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= /bin/sh -c
CC		= omcc
CFLAGS		= -O

PROGRAMS	= prlsctn001 prlsctn002 prlsctn003 prlsctn004 \
	          prlsctn005 prlsctn006 prlsctn007 prlsctn008
OBJS		= prlsctn001.o prlsctn002.o prlsctn003.o prlsctn004.o \
	          prlsctn005.o prlsctn006.o prlsctn007.o prlsctn008.o
SRCS		= prlsctn001.c prlsctn002.c prlsctn003.c prlsctn004.c \
	          prlsctn005.c prlsctn006.c prlsctn007.c prlsctn008.c

all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
