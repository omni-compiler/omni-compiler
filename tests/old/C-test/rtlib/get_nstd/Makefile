# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= /bin/sh -c
CC		= omcc
CFLAGS		= -O

PROGRAMS	= get_nstd001 get_nstd002 get_nstd003
OBJS		= get_nstd001.o get_nstd002.o get_nstd003.o
SRCS		= get_nstd001.c get_nstd002.c get_nstd003.c


all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
