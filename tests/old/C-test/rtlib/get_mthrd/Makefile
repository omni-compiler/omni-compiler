# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= /bin/sh -c
CC		= omcc
CFLAGS		= -O

PROGRAMS	= get_mthrd001 get_mthrd002 get_mthrd003 get_mthrd004 \
		  get_mthrd005 get_mthrd006
OBJS		= get_mthrd001.o get_mthrd002.o get_mthrd003.o get_mthrd004.o \
		  get_mthrd005.o get_mthrd006.o
SRCS		= get_mthrd001.c get_mthrd002.c get_mthrd003.c get_mthrd004.c \
		  get_mthrd005.c get_mthrd006.c


all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
