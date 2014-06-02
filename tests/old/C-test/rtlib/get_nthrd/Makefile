# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= /bin/sh -c
CC		= omcc
CFLAGS		= -O

PROGRAMS	= get_nthrd001 get_nthrd002 get_nthrd003 get_nthrd004 \
		  get_nthrd005 get_nthrd006
OBJS		= get_nthrd001.o get_nthrd002.o get_nthrd003.o get_nthrd004.o \
		  get_nthrd005.o get_nthrd006.o
SRCS		= get_nthrd001.c get_nthrd002.c get_nthrd003.c get_nthrd004.c \
		  get_nthrd005.c get_nthrd006.c


all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
