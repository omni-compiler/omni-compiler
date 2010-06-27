# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= 
CC		= omcc
CFLAGS		= -O

PROGRAMS	= exit stdarg001 vla001 vla002 vla003 vla004
OBJS		= exit.o stdarg001.o vla001.o vla002.o vla003.o vla004.o
SRCS		= exit.c stdarg001.c vla001.c vla002.c vla003.c vla004.c
TESTSCRIPTS	= exit001 exit002
TESTPROGRAMS	= stdarg001 vla001 vla002 vla003 vla004

all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${TESTSCRIPTS}; do		\
	  echo "run : $$prog";			\
	  EXEC="$(EXEC)" ./$$prog;		\
	done
	@for prog in ${TESTPROGRAMS}; do	\
	  echo "run : $$prog";			\
	  $(EXEC) ./$$prog;			\
	done


