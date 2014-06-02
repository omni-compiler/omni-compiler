# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= /bin/sh -c
CC		= omcc
CFLAGS		= -O

PROGRAMS	= set_nstd001 set_nstd002
OBJS		= set_nstd001.o set_nstd002.o
SRCS		= set_nstd001.c set_nstd002.c


all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
