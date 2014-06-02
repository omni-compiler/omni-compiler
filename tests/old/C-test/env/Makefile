# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= 
CC		= omcc
CFLAGS		= -O

PROGRAMS	= envparam schedule
OBJS		= envparam.o schedule.o
SRCS		= envparam.c schedule.c

TESTPROG	= schdl001 schdl002 schdl003 \
		  nthrd001 dynmc001 nstd001

all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${TESTPROG}; do	\
	  echo "run : $$prog";		\
	  EXEC="${EXEC}" ./$$prog;	\
	done


